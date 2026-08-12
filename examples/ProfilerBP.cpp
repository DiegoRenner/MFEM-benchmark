// CEED bake-off problem (BP) profiler, aligned with the Nektar++ ProfilerBP.
//
// BP definitions (ceed.exascaleproject.org/bps, pulled 2026-08-12):
//   BP1 = scalar CG solve with the mass matrix, homogeneous Neumann.
//   BP3 = scalar CG solve with the Poisson (stiffness) operator,
//         homogeneous Dirichlet.
//   BP13 is NOT a bake-off problem: Helmholtz (mass + stiffness), kept as a
//   parity reference for the Nektar++ BP13.
// Basis: nodal Lagrange on GLL points. Quadrature: q = p+2 Gauss-Legendre
// per dimension, pinned through an explicit IntegrationRule of order 2p+3.
// Solver: CG with diagonal Jacobi preconditioning (spec-legal, matches the
// Nektar++ DiagPreconOp), relative tolerance 1e-6 (spec stopping criterion),
// iteration cap 5000 (matches NekLinSysMaxIterations).
// Metric: time per CG iteration, computed from N_test repeated solves after
// N_warmup warm-ups, device synchronised, setup excluded.
//
// Usage: ProfilerBP <mesh_dir> <out_dir> [bp] [order_min] [order_max]
//   bp in {1, 3, 13}; default runs BP1 and BP3.
#include "mfem.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace mfem;

struct BPDef
{
    int bp;
    std::string name;
    bool dirichlet; // homogeneous Dirichlet on the whole boundary
};

int main(int argc, char *argv[])
{
    // 1) Initialize device (e.g., CUDA)
    mfem::Device device("cuda");

    // 2) Benchmark parameters
    const int N_warmup    = 2;
    const int N_test      = 100;
    const int max_cg_iter = 5000; // matches Nektar++ NekLinSysMaxIterations
    const double rel_tol  = 1e-6; // CEED stopping criterion

    std::string mesh_dir = (argc > 1) ? argv[1] : ".";
    std::string out_dir  = (argc > 2) ? argv[2] : ".";
    const int bp_select  = (argc > 3) ? std::atoi(argv[3]) : 0;
    const int order_min  = (argc > 4) ? std::atoi(argv[4]) : 1;
    const int order_max  = (argc > 5) ? std::atoi(argv[5]) : 7;

    std::vector<BPDef> bps;
    if (bp_select == 0 || bp_select == 1)
    {
        bps.push_back({1, "Mass", false});
    }
    if (bp_select == 0 || bp_select == 3)
    {
        bps.push_back({3, "Stiffness", true});
    }
    if (bp_select == 13)
    {
        bps.push_back({13, "Helmholtz", true});
    }

    const std::string mesh_sizes[] = {"8", "16", "24", "32", "48", "64"};

    for (const auto &def : bps)
    {
        for (const std::string &mesh_size : mesh_sizes)
        {
            const std::string mesh_path =
                mesh_dir + "/cubeHex" + mesh_size + "_mesh.msh";
            for (int order = order_min; order <= order_max; ++order)
            {
                // 3) Mesh and FE space: nodal Lagrange on GLL points.
                mfem::Mesh mesh(mesh_path);
                mfem::H1_FECollection fec(order, mesh.Dimension(),
                                          BasisType::GaussLobatto);
                mfem::FiniteElementSpace fes(&mesh, &fec);

                const auto global_dofs =
                    static_cast<long long>(fes.GetTrueVSize());
                const auto num_elements =
                    static_cast<long long>(mesh.GetNE());

                // Essential (Dirichlet) dofs on the whole boundary for the
                // second-order operators, per the BP3 definition.
                mfem::Array<int> ess_tdof_list;
                if (def.dirichlet && mesh.bdr_attributes.Size() > 0)
                {
                    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
                    ess_bdr = 1;
                    fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
                }

                // 4) Operator with the CEED quadrature: q = p+2 GL points
                // per dimension, i.e. rule order 2p+3.
                const mfem::IntegrationRule &ir = mfem::IntRules.Get(
                    mfem::Geometry::CUBE, 2 * order + 3);

                BilinearForm oper(&fes);
                if (def.bp == 1)
                {
                    auto *mi = new MassIntegrator();
                    mi->SetIntegrationRule(ir);
                    oper.AddDomainIntegrator(mi);
                }
                else if (def.bp == 3)
                {
                    auto *di = new DiffusionIntegrator();
                    di->SetIntegrationRule(ir);
                    oper.AddDomainIntegrator(di);
                }
                else // BP13, Nektar parity reference
                {
                    auto *mi = new MassIntegrator();
                    mi->SetIntegrationRule(ir);
                    auto *di = new DiffusionIntegrator();
                    di->SetIntegrationRule(ir);
                    oper.AddDomainIntegrator(mi);
                    oper.AddDomainIntegrator(di);
                }
                oper.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
                oper.Assemble();

                mfem::OperatorPtr A;
                oper.FormSystemMatrix(ess_tdof_list, A);

                // 5) Manufactured consistent RHS: b = A x_exact.
                mfem::Vector x_exact(fes.GetTrueVSize());
                mfem::Vector b(fes.GetTrueVSize());
                x_exact.UseDevice(true);
                b.UseDevice(true);

                mfem::GridFunction x_gf(&fes);
                x_gf.Randomize(1); // seed 1 for reproducibility
                x_gf.GetTrueDofs(x_exact);
                if (ess_tdof_list.Size() > 0)
                {
                    // Homogeneous Dirichlet: zero the essential dofs of the
                    // exact solution so the manufactured RHS is consistent.
                    x_exact.SetSubVector(ess_tdof_list, 0.0);
                }

                A->Mult(x_exact, b);

                // 6) CG with diagonal Jacobi preconditioning.
                mfem::OperatorJacobiSmoother M(oper, ess_tdof_list);
                mfem::CGSolver pcg;
                pcg.SetOperator(*A);
                pcg.SetPreconditioner(M);
                pcg.SetMaxIter(max_cg_iter);
                pcg.SetRelTol(rel_tol);
                pcg.SetAbsTol(0.0);
                pcg.SetPrintLevel(-1);

                mfem::Vector x(fes.GetTrueVSize());
                x.UseDevice(true);

                // 7) Warm-up solves.
                for (int i = 0; i < N_warmup; ++i)
                {
                    x = 0.0;
                    pcg.Mult(b, x);
                }

                // 8) Timed solves.
                cudaDeviceSynchronize();
                const auto begin = std::chrono::steady_clock::now();

                long long cg_iter = 0;
                for (int i = 0; i < N_test; ++i)
                {
                    x = 0.0; // reset initial guess
                    pcg.Mult(b, x);
                    cg_iter += pcg.GetNumIterations();
                }

                cudaDeviceSynchronize();
                const auto end = std::chrono::steady_clock::now();

                // 9) Report: nm (= order+1), elements, T-vector dofs, time
                // per CG iteration in microseconds, iterations per solve.
                const double total_time_us =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        end - begin)
                        .count();
                const double time_per_cg_iteration =
                    total_time_us / static_cast<double>(cg_iter);
                const double iters_per_solve =
                    static_cast<double>(cg_iter) / N_test;

                std::ofstream out_file(out_dir + "/log_mfem_profilerBP_" +
                                           def.name + ".log",
                                       std::ios::app);
                out_file << order + 1 << " " << num_elements << " "
                         << global_dofs << " " << time_per_cg_iteration << " "
                         << iters_per_solve << std::endl;
                std::cout << "BP" << def.bp << " " << def.name << " mesh "
                          << mesh_size << " p" << order << ": "
                          << time_per_cg_iteration << " us/iter, "
                          << iters_per_solve << " iters/solve" << std::endl;
            }
        }
    }
    return 0;
}

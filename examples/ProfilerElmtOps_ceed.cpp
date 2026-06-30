// CEED-faithful MFEM bake-off KERNEL (BK) profiler, deformed meshes.
// Differences from the original ProfilerElmtOps.cpp:
//   * L2 basis on GAUSS-LOBATTO (GLL) nodes  -> matches CEED nodal-GLL basis
//     (was default L2 = Gauss-Legendre nodes).
//   * explicit Gauss-Legendre quadrature with q = p+2 points per dim
//     (CEED BK1/BK3); a 1D GL rule of order 2p+3 has p+2 points.
// Still L2 (E-vector == L-vector) + PARTIAL assembly so oper.Mult is the pure
// element kernel (E-vec -> E-vec), no gather/scatter -> a true BK.
//
// Build like the original ProfilerElmtOps (same CMake target / mfem link).
#include "mfem.hpp"
#include <chrono>
#include <random>
#include <fstream>
#include <cmath>
#include <iostream>

using namespace mfem;

int main(int argc, char *argv[]) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0] << " <device> <operator> <shape> <size> <order>\n";
    return 1;
  }
  std::string device_str = argv[1];
  std::string operator_name = argv[2];
  std::string shape = argv[3];
  std::string mesh_size = argv[4];
  int order = std::stoi(argv[5]);

  Device device(device_str.c_str());

  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::default_random_engine re;

  std::string mesh_path_prefix = "/home/dbr25/MFEM-benchmark_bckp/nektar-benchmark_mesh/cube";
  std::string mesh_path_suffix = "_mesh.msh";
  std::string mesh_path = mesh_path_prefix + shape + mesh_size + mesh_path_suffix;

  Mesh mesh(mesh_path);
  mesh.SetCurvature(order, true, mesh.Dimension(), mfem::Ordering::byVDIM); // deformed

  int N_warmup = 100;
  int N_test = 100;

  // --- CEED-faithful discretisation ---
  L2_FECollection fec(order, mesh.Dimension(), BasisType::GaussLobatto); // nodal GLL
  FiniteElementSpace fes(&mesh, &fec);

  // q = p+2 Gauss-Legendre points per dimension (rule order 2p+3).
  Geometry::Type geom = fes.GetFE(0)->GetGeomType();
  const IntegrationRule &ir = IntRules.Get(geom, 2 * order + 3);

  ConstantCoefficient one(1.0);
  Vector xv(fes.GetNDofs()); xv.UseDevice(true);
  for (int i = 0; i < fes.GetNDofs(); ++i) { xv(i) = unif(re); }
  Vector yv(fes.GetNDofs()); yv.UseDevice(true);

  BilinearForm oper(&fes);
  if (operator_name == "Mass") {
    oper.AddDomainIntegrator(new MassIntegrator(&ir));
  } else if (operator_name == "Stiffness") {
    oper.AddDomainIntegrator(new DiffusionIntegrator(&ir));
  } else if (operator_name == "Helmholtz") {
    oper.AddDomainIntegrator(new MassIntegrator(&ir));
    oper.AddDomainIntegrator(new DiffusionIntegrator(&ir));
  }
  oper.SetAssemblyLevel(AssemblyLevel::PARTIAL);

  oper.Assemble();
  for (int i = 0; i < N_warmup; ++i) { oper.Mult(xv, yv); }
  cudaDeviceSynchronize();
  auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < N_test; ++i) { oper.Mult(xv, yv); }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();

  double time_diff_avg =
      std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 100.0;
  auto num_elements = static_cast<long long>(mesh.GetNE());

  long long local_dofs = 0;
  int p = order;
  if (shape == "Hex")        local_dofs = std::pow(p + 1, 3);
  else if (shape == "Tet")   local_dofs = (p + 1) * (p + 2) * (p + 3) / 6;
  else if (shape == "Prism") local_dofs = (p + 1) * (p + 1) * (p + 2) / 2;
  else if (shape == "Pyr")   local_dofs = (p + 1) * (p + 2) * (2 * p + 3) / 6;

  long long total_dofs = num_elements * local_dofs;
  double total_dofs_per_second = total_dofs / time_diff_avg * 1e6;

  std::string clean_device = device_str;
  for (char &c : clean_device) { if (c == ':' || c == '/') c = '_'; }

  std::ofstream out_file("mfem_ceed_" + clean_device + "_" + operator_name + "_" + shape + ".log",
                         std::ios::app);
  out_file << mesh_size << " " << total_dofs << " " << order << " " << total_dofs_per_second << std::endl;
  return 0;
}

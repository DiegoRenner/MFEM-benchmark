#include "mfem.hpp"
#include <chrono>
#include <random>
#include <fstream>

using namespace mfem;
std::string AssemblyLevelToString(mfem::AssemblyLevel level) {
  switch (level) {
  case mfem::AssemblyLevel::LEGACY:
    return "LEGACY";
  case mfem::AssemblyLevel::FULL:
    return "FULL";
  case mfem::AssemblyLevel::ELEMENT:
    return "ELEMENT";
  case mfem::AssemblyLevel::PARTIAL:
    return "PARTIAL";
  case mfem::AssemblyLevel::NONE:
    return "NONE";
  default:
    return "UNKNOWN";
  }
}

int main(int argc, char *argv[]) {
  Device device("cuda");

  double lower_bound = 0;
  double upper_bound = 1;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::default_random_engine re;
  std::string shapes[] = {"Hex"};
  std::string mesh_sizes[] = {"8", "16", "24", "32", "48", "64"};
  std::string operators[] = {"Mass", "Stiffness", "Helmholtz"};

  for (std::string operator_name: operators) {
    for (std::string shape : shapes) {
      for (std::string mesh_size : mesh_sizes) {
        //std::string mesh_path_prefix = "/home/diego/studies/uni/phd/MFEM-benchmark/nektar-benchmark_mesh/cube";
        std::string mesh_path_prefix = "/home/dbr25/MFEM-benchmark_bckp/nektar-benchmark_mesh/cube";
        std::string mesh_path_suffix = "_mesh.msh";

        std::string mesh_path =
            mesh_path_prefix + shape + mesh_size + mesh_path_suffix;

        Mesh mesh(mesh_path);

        int N_warmup = 100;
        int N_test = 100;
        for (int order = 1; order <= 7; ++order) {
          H1_FECollection fec(order, mesh.Dimension());
          FiniteElementSpace fes(&mesh, &fec);
          ConstantCoefficient one(1.0);
          Vector xv(fes.GetNDofs());
          xv.UseDevice(true);
          for (int i = 0; i < fes.GetNDofs(); ++i) {
            xv(i) = unif(re);
          }
          Vector yv(fes.GetNDofs());
          yv.UseDevice(true);

          BilinearForm oper(&fes);
          if (operator_name=="Mass") {
            oper.AddDomainIntegrator(new MassIntegrator());
          } else if (operator_name=="Stiffness") {
            oper.AddDomainIntegrator(new DiffusionIntegrator());
          } else if (operator_name=="Helmholtz") {
            oper.AddDomainIntegrator(new MassIntegrator());
            oper.AddDomainIntegrator(new DiffusionIntegrator());
          }
          oper.SetAssemblyLevel(AssemblyLevel::PARTIAL);
          oper.Assemble();
          for (int i = 0; i < N_warmup; ++i) {
            oper.Mult(xv, yv);
          }
          std::chrono::steady_clock::time_point begin =
              std::chrono::steady_clock::now();
          cudaDeviceSynchronize();
          for (int i = 0; i < N_test; ++i) {
            oper.Mult(xv, yv);
          }
          cudaDeviceSynchronize();
          std::chrono::steady_clock::time_point end =
              std::chrono::steady_clock::now();

          double time_diff_avg =
              std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
                  .count() /
              100.0;
          auto ndofs = static_cast<std::int64_t>(fes.GetNDofs());
          auto ntest = static_cast<std::int64_t>(N_test);
          double total_dofs_per_second = ndofs / time_diff_avg * 1e6;
          std::ofstream out_file("log_mfem_profilerElmtOps_" + operator_name, std::ios::app);
          out_file << mesh_size << " " << ndofs << " " << order << " "
                    << total_dofs_per_second << std::endl;



        }
      }
    }
  }
    return 0;
}
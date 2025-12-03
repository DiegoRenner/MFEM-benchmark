#include "mfem.hpp"
#include <chrono>

using namespace mfem;
int main(int argc, char *argv[]) {
	// 1) Initialize device (e.g., CUDA)
	mfem::Device device("cuda");
	//device.Print();

	// 2) Set up benchmark parameters
	const int N_warmup = 2;
	const int N_test = 100;
	const int max_cg_iter = 5000; // Match Nektar++ benchmark's hardcoded value

	//std::cout << "Order | NumElements | GlobalDoFs | TimePerCGIter (us)\n";
	//std::cout << "--------------------------------------------------------\n";

	std::string mesh_sizes[] = {"8", "16", "24", "32", "48"};
  std::string operators[] = {"Mass", "Helmholtz"};

  for (std::string operator_name: operators) {
   for (std::string mesh_size : mesh_sizes) {
     //std::string mesh_path_prefix = "/home/diego/studies/uni/phd/MFEM-benchmark/nektar-benchmark_mesh/cube";
     std::string mesh_path_prefix = "/home/dbr25/MFEM-benchmark_bckp/nektar-benchmark_mesh/cube";
     std::string mesh_path_suffix = "_mesh.msh";
   	for (int order = 1; order <= 7; ++order) {
        std::string mesh_path =
            mesh_path_prefix + "Hex" + mesh_size + mesh_path_suffix;
   		// 3) Load mesh and create the Finite Element space
   		mfem::Mesh mesh(mesh_path);
   		mfem::H1_FECollection fec(order, mesh.Dimension());
   		mfem::FiniteElementSpace fes(&mesh, &fec);

   		auto global_dofs = static_cast<long long>(fes.GetTrueVSize());
   		auto num_elements = static_cast<long long>(mesh.GetNE());

       BilinearForm oper(&fes);
       if (operator_name=="Mass") {
         oper.AddDomainIntegrator(new MassIntegrator());
       } else if (operator_name=="Helmholtz") {
         oper.AddDomainIntegrator(new MassIntegrator());
         oper.AddDomainIntegrator(new DiffusionIntegrator());
       }
   		oper.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
   		oper.Assemble();

   		mfem::Vector x_exact(fes.GetTrueVSize());
   		mfem::Vector b(fes.GetTrueVSize());
   		x_exact.UseDevice(true);
   		b.UseDevice(true);

   		// Use a GridFunction to easily generate a random field, then get the true DOFs
   		mfem::GridFunction x_gf(&fes);
   		x_gf.Randomize(1); // Seed with 1 for reproducibility
   		x_gf.GetTrueDofs(x_exact);

   		oper.Mult(x_exact, b);

   		// 6) Set up the Preconditioned Conjugate Gradient (PCG) solver
   		//mfem::OperatorJacobiSmoother M(a, fes.GetEssentialTrueDofs()); // Diagonal Preconditioner
   		// With these three lines:
   		//mfem::Array<int> ess_tdof_list;
   		//fes.GetEssentialTrueDofs(ess_tdof_list);
   		mfem::OperatorJacobiSmoother M;
   		M.SetOperator(oper);
   		mfem::CGSolver pcg;
   		pcg.SetOperator(oper);
   		pcg.SetPreconditioner(M);
   		pcg.SetMaxIter(max_cg_iter);
   		pcg.SetRelTol(1e-16);
   		pcg.SetAbsTol(1e-16);
   		pcg.SetPrintLevel(-1); // Suppress solver output

   		// Vector for the solver to write its result into
   		mfem::Vector x(fes.GetTrueVSize());
   		x.UseDevice(true);

   		// 7) Run warm-up iterations
   		for (int i = 0; i < N_warmup; ++i) {
   			x = 0.0;
   			pcg.Mult(b, x);
   		}

   		// 8) Run the timed benchmark
   		cudaDeviceSynchronize();
   		auto begin = std::chrono::steady_clock::now();

   		int cg_iter = 0;
   		for (int i = 0; i < N_test; ++i) {
   			x = 0.0; // Reset initial guess
   			pcg.Mult(b, x);
   			cg_iter += pcg.GetNumIterations();
   		}

   		cudaDeviceSynchronize();
   		auto end = std::chrono::steady_clock::now();

   		// 9) Calculate and print results
   		double total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

   		// The Nektar++ benchmark normalizes time by a hardcoded max iteration count (5000).
   		// We do the same here for a direct comparison.
   		double time_per_cg_iteration = total_time_us / (cg_iter) ;

   		//std::cout << std::setw(5) << order << " | "
   		//          << std::setw(11) << num_elements << " | "
   		//          << std::setw(10) << global_dofs << " | "
   		//          << std::fixed << std::setprecision(4) << time_per_cg_iteration
   		//          << std::endl;
      std::ofstream out_file("log_mfem_profilerBP_" + operator_name + ".log", std::ios::app);
   		out_file << order+1 << " "
   			<< num_elements << " "
   			<< global_dofs << " "
   			<< time_per_cg_iteration
   			<< std::endl;
   	}
   }
  }
	return 0;
}

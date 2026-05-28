#include "jacobi_solver.hpp"
#include "vtk_exporter.hpp"
#include "utils.hpp"
#include "GetPot"


int main (int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    // 1. GetPot initialization
    GetPot command_line(argc, argv);
    std::string config_filename = command_line.follow("parameters.pot", "-f");
    GetPot datafile(config_filename.c_str());

    // 2. Read parameters (the second value is the default in case of error)
    int n = datafile("Jacobi_solver_parameters/num_elements", 100);
    int max_iters = datafile("Jacobi_solver_parameters/max_iters", 10000);
    double tolerance = datafile("Jacobi_solver_parameters/tolerance", 1e-6);

    // 3. Read the forcing term and boundary condition expression as a string
    std::string force_expr = datafile("simulazione/forcing_term", "8*pi*pi*sin(2*pi*x)*sin(2*pi*y)");
    std::string bc_expr = datafile("simulazione/boundary_condition", "0.0");

    // Control print only for Rank 0
    if (rank == 0) {
        std::cout << "=== Parameters read from " << config_filename << " ===" << std::endl;
        std::cout << "Grid size (n): " << n << std::endl;
        std::cout << "Maximum iterations: " << max_iters << std::endl;
        std::cout << "Tolerance: " << tolerance << std::endl;
        std::cout << "Forcing term: " << force_expr << std::endl;
        std::cout << "Boundary condition: " << bc_expr << std::endl;
        std::cout << "Total MPI processes: " << size << std::endl;
        std::cout << "========================================" << std::endl;
    }

    utils::Function force(force_expr);
    utils::Function bc(bc_expr);

    MPI_Comm comm = MPI_COMM_WORLD;

    double x_i = 0.0, y_i = 0.0;
    double x_f = 1.0, y_f = 1.0;

    // Pass the variable 'n' read by GetPot
    parallel_jacobi::JacobiSolver solver(n, x_i, y_i, x_f, y_f, force, bc, comm);
    
    // Execute the solver with dynamic parameters
    solver.solve(max_iters, tolerance);

    // Only Rank 0 gathers and exports!
    if (rank == 0) {
        auto result = solver.get_global_matrix();
        std::cout << "Computation finished! Resulting matrix: " << result.rows() << "x" << result.cols() << std::endl;
        
        // Compute spacing 'h'
        double h = (x_f - x_i) / (n - 1);
        
        // Call the VTK exporter
        utils::export_vtk("solution.vtk", result, n, x_i, y_i, h);
    }

    MPI_Finalize();
    return 0;
}
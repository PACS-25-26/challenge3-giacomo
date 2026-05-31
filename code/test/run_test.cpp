#include "jacobi_solver.hpp"
#include "vtk_exporter.hpp"
#include "GetPot"
#include "utils.hpp"
#include <filesystem>
#include <sstream>


/**
 * @file run_test.cpp
 * @brief Spatial convergence and accuracy test for the parallel Jacobi Poisson solver.
 * * @details This test script performs a Grid Convergence Study by solving the 2D 
 * Poisson equation on progressively refined grids (from n = 16 to n = 256).
 * It leverages a hybrid MPI + OpenMP approach, calculates the L2 error 
 * against the analytical exact solution, and logs the execution metrics.
 */

 /**
 * @brief Appends simulation results (metrics and errors) to a shared CSV file.
 * * @note This function must only be executed by Rank 0 in an MPI environment to
 * prevent write race conditions.
 * * @param size Total number of MPI processes used.
 * @param n Grid resolution (number of subdivisions per dimension).
 * @param L2_err The computed L2 norm error against the exact solution.
 * @param time The solver execution time measured in seconds via MPI_Wtime.
 */

void write_in_file(int size, int n, double L2_err, double time);

/**
 * @brief Main entry point for the grid refinement test suite.
 * * @param argc Program argument count.
 * @param argv Program argument vector.
 * @return int Execution status (0 for success).
 */

int main (int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    // 1. GetPot initialization
    GetPot command_line(argc, argv);
    std::string config_filename = command_line.follow("../parameters.pot", "-f");
    GetPot datafile(config_filename.c_str());

    // 2. Read parameters (the second value is the default in case of error)
    int max_iters = datafile("Jacobi_solver_parameters/max_iters", 10000);
    double tolerance = datafile("Jacobi_solver_parameters/tolerance", 1e-6);

    // 3. Read the forcing term and boundary condition expression as a string
    std::string force_expr = datafile("simulazione/forcing_term", "8*pi*pi*sin(2*pi*x)*sin(2*pi*y)");
    std::string bc_expr = datafile("simulazione/boundary_condition", "0.0");
    std::string uex_expr = datafile("simulazione/boundary_condition", "sin(2*pi*x)*sin(2*pi*y)");

    // Control print only for Rank 0
    if (rank == 0) {
        std::cout << "=== Parameters read from " << config_filename << " ===" << std::endl;
        std::cout << "Maximum iterations: " << max_iters << std::endl;
        std::cout << "Tolerance: " << tolerance << std::endl;
        std::cout << "Forcing term: " << force_expr << std::endl;
        std::cout << "Boundary condition: " << bc_expr << std::endl;
        std::cout << "Total MPI processes: " << size << std::endl;
        std::cout << "========================================" << std::endl;
    }

    utils::Function force(force_expr);
    utils::Function bc(bc_expr);
    utils::Function u_exact(uex_expr);

    MPI_Comm comm = MPI_COMM_WORLD;

    double x_i = 0.0, y_i = 0.0;
    double x_f = 1.0, y_f = 1.0;

    // Convergence test
    for (int k = 4; k <= 8; ++k) {

        int n = std::pow(2, k); // n = 16, 32, 64, 128, 256
        double h = (x_f - x_i) / (n - 1);

        if(rank == 0) std::cout << "RUNNING " << "Cores: " << size << " | Grid: " << n << "x" << n << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();

        // Execute the solver with dynamic parameters
        parallel_jacobi::JacobiSolver solver(n, x_i, y_i, x_f, y_f, force, bc, comm);
        solver.solve(max_iters, tolerance);

        MPI_Barrier(MPI_COMM_WORLD); 
        double end_time = MPI_Wtime();

        // Only Rank 0 
        if (rank == 0) {
            auto U = solver.get_global_matrix();
            double L2_err = utils::compute_L2_error(U, n, h, x_i, y_i, u_exact);
            double time = (end_time - start_time);
            std::cout << "Time: " << time << " , L2_error: " << L2_err << std::endl;
            
            // CSV results
            write_in_file(size, n, L2_err, time);
            
            // Export vtk file
            std::stringstream ss;
            ss << "data/solution_" << size << "cores_" << n << "gridsize" << ".vtk";
            std::string filename = ss.str();
            utils::export_vtk(filename, U, n, x_i, y_i, h);
        }
    }

    MPI_Finalize();
    return 0;
}


void write_in_file(int size, int n, double L2_err, double time){
    
    // Defyning the path
    std::string dir_path = "data";
    std::string file_path = dir_path + "/results.csv";

    std::filesystem::create_directories(dir_path);

    bool is_new_file = !std::filesystem::exists(file_path);

    // Open the file in APPEND mode
    std::ofstream outfile(file_path, std::ios_base::app);

    if (outfile.is_open()) {
        if (is_new_file) {
            outfile << "Cores,GridSize_n,Time_s,L2_Error\n"; // Header
        }
        // Results
        outfile << size << "," << n << "," << time << "," << L2_err << "\n";
        outfile.close();
    } else {
        std::cerr << "Error: cannot open file " << file_path << "\n";
    }
}
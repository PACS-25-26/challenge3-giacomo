#include "jacobi_solver.hpp"
#include "vtk_exporter.hpp"
#include "GetPot"
#include "forcing_parser.hpp"
#include "utils.hpp"
#include <functional>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>

void write_in_file(double l2_err, double time, int size, int n);
    

int main (int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Retrieve the total number of MPI processes

    // 1. GetPot initialization
    GetPot command_line(argc, argv);
    std::string config_filename = command_line.follow("config.pot", "-f");
    GetPot datafile(config_filename.c_str());

    // 2. Read parameters (the second value is the default in case of error)
    int max_iters = datafile("Jacobi_solver_parameters/max_iters", 10000);
    double tolerance = datafile("Jacobi_solver_parameters/tolerance", 1e-6);

    // 3. Read the forcing term expression as a string
    std::string force_expr = datafile("simulazione/forcing_term", "8*pi*pi*sin(2*pi*x)*sin(2*pi*y)");

    // Control print only for Rank 0
    if (rank == 0) {
        std::cout << "=== Parameters read from " << config_filename << " ===" << std::endl;
        std::cout << "Maximum iterations: " << max_iters << std::endl;
        std::cout << "Tolerance: " << tolerance << std::endl;
        std::cout << "Forcing term: " << force_expr << std::endl;
        std::cout << "Total MPI processes: " << size << std::endl;
        std::cout << "========================================" << std::endl;
    }

    utils::ForcingFunction force(force_expr);

    std::function<double(double, double)> bc = [](double x, double y) { 
        return 0.0; 
    };

    auto u_exact = [](double x, double y) {
        return std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
    };

    MPI_Comm comm = MPI_COMM_WORLD;

    double x_i = 0.0, y_i = 0.0;
    double x_f = 1.0, y_f = 1.0;

    for (int k = 4; k <= 8; ++k) {

        int n = std::pow(2, k); // n = 16, 32, 64, 128, 256
        int num_elem = n * n;
        double h = (x_f - x_i) / (n - 1);

        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();

        // Pass the variable 'n' 
        parallel_jacobi::JacobiSolver solver(n, x_i, y_i, x_f, y_f, force, bc, comm);
    
        // Execute the solver with dynamic parameters
        solver.solve(max_iters, tolerance);

        MPI_Barrier(MPI_COMM_WORLD); 
        double end_time = MPI_Wtime();

        // Only Rank 0 
        if (rank == 0) {
            auto U = solver.get_global_matrix();
            double l2_err = parallel_jacobi::compute_L2_error(U, n, h , x_i, y_i, u_exact);
            double time = (end_time - start_time);
            std::cout << size << "," << n << "," << (end_time - start_time) << "," << l2_err << std::endl;

            write_in_file(l2_err, time, size, n);

        }
    }

    MPI_Finalize();
    return 0;
}


//!!! IN MAKE FILE METTERE DI TOGLIERE CARTELLA DATA PRIMA DI CICLO SU CORES
void write_in_file(double l2_err, double time, int size, int n){
    
    // Definiz il percorso e creaz cartella se non esiste
    std::string dir_path = "test/data";
    std::string file_path = dir_path + "/results.csv";

    std::filesystem::create_directories(dir_path);

    // Controlla se il file esiste già per capire se scrivere l'header
    bool is_new_file = !std::filesystem::exists(file_path);

    // Apri il file in modalità APPEND (aggiunge alla fine senza cancellare)
    std::ofstream outfile(file_path, std::ios_base::app);

    if (outfile.is_open()) {
        if (is_new_file) {
            outfile << "Cores,GridSize_n,Time_s,L2_Error\n"; // Header
        }
        // Scrive la riga dei risultati
        outfile << size << "," << n << "," << time << "," << l2_err << "\n";
        outfile.close();
    } else {
        std::cerr << "Errore: impossibile aprire il file " << file_path << "\n";
    }
}
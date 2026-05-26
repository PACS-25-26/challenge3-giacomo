#include "jacobi_solver.hpp"
#include "vtk_exporter.hpp"
#include <functional>
#include <cmath>
#include <iostream>

int main (int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 3)        
    {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <num_elements>  <num_tasks>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    
    int n = std::stoi(argv[1]);
    int num_tasks = std::stoi(argv[2]);
    
    std::function<double(double, double)> force = [](double x, double y) { 
        return 8 * M_PI * M_PI * sin(2 * M_PI * x) * sin(2 * M_PI * y); 
    };
    std::function<double(double, double)> bc = [](double x, double y) { 
        return 0.0; 
    };

    MPI_Comm comm = MPI_COMM_WORLD;

    
    double x_i = 0.0, y_i = 0.0;
    double x_f = 1.0, y_f = 1.0;

    parallel_jacobi::JacobiSolver solver(n, x_i, y_i, x_f, y_f, force, bc, comm);
    
    // Esegue il solver parallelo
    solver.solve(10000, 1e-6);

    // Solo il Rank 0 raccoglie ed esporta!
    if (rank == 0) {
        auto result = solver.get_global_matrix();
        std::cout << "Calcolo terminato! Matrice risultante: " << result.rows() << "x" << result.cols() << std::endl;
        
        // Calcolo dello spacing 'h'
        double h = (x_f - x_i) / (n - 1);
        
        // Chiamata all'esportatore VTK
        parallel_jacobi::export_vtk("solution.vtk", result, n, x_i, y_i, h);
    }

    MPI_Finalize();
    return 0;
}
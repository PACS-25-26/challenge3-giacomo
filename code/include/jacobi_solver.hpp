#pragma once

#ifndef JACOBI_SOLVER_HPP
#define JACOBI_SOLVER_HPP

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <functional>
#include <iostream>
#include <string>
#include <fstream>
#include <mpi.h>
#include <omp.h>
#include <Eigen/Dense>


/**
 * @file jacobi_solver.hpp
 * @brief Definition of the JacobiSolver class for solving Poisson/Laplace equations in parallel.
 */

/**
 * @namespace parallel_jacobi
 * @brief Namespace housing the parallel Jacobi solver implementation and related types.
 */

namespace parallel_jacobi{

    /**
     * @brief Type alias for a dynamically-sized Eigen matrix stored in Row-Major order.
     * * Row-major layout is highly recommended here to optimize memory locality 
     * during MPI halo exchanges (sending/receiving continuous row blocks).
     */

    using MatrixRowMaj = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    /**
     * @class JacobiSolver
     * @brief A parallel 2D Jacobi iterative solver utilizing hybrid MPI + OpenMP parallelization.
     * * This class solves elliptic partial differential equations (like the Poisson equation)
     * on a uniform 2D grid. The domain is decomposed horizontally (by row blocks) among 
     * MPI processes, while internal matrix operations within each process can be accelerated 
     * using OpenMP threads.
     */

    class JacobiSolver{
        
        private:
            // --- Grid and Domain Parameters ---
            int n;              ///< Number of internal grid points along one dimension.
            double h;           ///< Grid spacing step size ($h = \Delta x = \Delta y$).
            double x_iniz;      ///< Starting boundary coordinate along the X-axis ($x_{min}$).
            double y_iniz;      ///< Starting boundary coordinate along the Y-axis ($y_{min}$).
            double x_fin;       ///< Ending boundary coordinate along the X-axis ($x_{max}$).
            double y_fin;       ///< Ending boundary coordinate along the Y-axis ($y_{max}$).
            
            /** * @brief Forcing function $f(x, y)$ (right-hand side of the Poisson equation $\nabla^2 u = f$).
             */
            std::function<double(double, double)> f;
            /** * @brief Dirichlet boundary condition function $g(x, y)$ applied to the domain edges.
             */
            std::function<double(double, double)> g;

            // --- MPI Communication Parameters ---
            MPI_Comm comm;      ///< The MPI communicator handles parallel communications.
            int rank;           ///< Rank ID of the current MPI process.
            int size;           ///< Total number of processes in the communicator.
            int local_rows;     ///< Number of grid rows assigned strictly to this process (excluding halos).
            int up_neighbor;    ///< Rank of the MPI neighbor process managing the block directly above.
            int down_neighbor;  ///< Rank of the MPI neighbor process managing the block directly below.
            int start_i;        ///< Global row index mapping to the first local row of this process.
            int end_i;          ///< Global row index mapping to the last local row of this process.
            
            /** * @brief Array containing the number of elements to receive from each rank during gathering.
             */
            std::vector<int> sendcounts;
            /** * @brief Entry displacements in the global array for MPI gathering operations.
             */
            std::vector<int> displs;

           // --- Local and Global Matrices (Data Buffers) ---
            MatrixRowMaj U_global; ///< Global solution matrix. Only fully assembled and accessible on Rank 0.
            MatrixRowMaj U_loc0;   ///< Local grid buffer storing values from the current iteration step (including halos).
            MatrixRowMaj U_loc1;   ///< Local grid buffer storing newly calculated values for the next iteration step.

            // --- Private Helper Methods ---

            /**
             * @brief Initializes the global matrix with Dirichlet boundary conditions.
             * * Fills the borders of the grid utilizing the boundary condition function @p g.
             */
            void initialize_global_bc();

            /**
             * @brief Performs ghost cell/halo exchange between adjacent MPI processes.
             * * Communicates overlapping row boundaries concurrently with top/bottom neighbors 
             * to synchronize subdomain margins before a computation step.
             */
            void exchange_halos();
            
            /**
             * @brief Computes a single local Jacobi iteration step.
             * * Updates values from @p U_loc0 into @p U_loc1 based on the discrete 5-point Laplacian stencil.
             * This operation is typically accelerated locally using OpenMP parallel loops.
             */
            void compute_step();

            /**
             * @brief Computes the local absolute error, performs an MPI reduction, and swaps local buffers.
             * * Calculates the maximum difference ($L^\infty$ norm) between @p U_loc1 and @p U_loc0, 
             * synchronizes global error across all ranks, and updates @p U_loc0 for the next step.
             * @return The global maximum absolute error across the entire domain.
             */
            double compute_error_and_update();

        public:
            /**
             * @brief Constructor for the JacobiSolver class.
             * * Handles domain partitioning, determines neighbor ranks, establishes local 
             * subgrid structures, and prepares internal memory buffers.
             * * @param n Number of internal subdivisions per grid dimension.
             * @param x_i Starting X coordinate of the physical domain.
             * @param y_i Starting Y coordinate of the physical domain.
             * @param x_f Ending X coordinate of the physical domain.
             * @param y_f Ending Y coordinate of the physical domain.
             * @param force Callable functional object representing the RHS source term $f(x,y)$.
             * @param bc Callable functional object representing the Dirichlet boundary conditions $g(x,y)$.
             * @param communicator The active MPI communicator context allocated for this solver.
             */
            JacobiSolver(int n, double x_i, double y_i, double x_f, double y_f,
                               std::function<double(double, double)> force,
                               std::function<double(double, double)> bc,
                               MPI_Comm communicator);
            

            /**
             * @brief Executes the iterative Jacobi numerical solution loop.
             * * The loop alternates between halo swaps, stencil updates, and residual checks 
             * until either the solution converges below the specified threshold or maximum iterations are met.
             * * @param max_iters Maximum number of allowed iterative steps to prevent infinite looping.
             * @param tolerance Convergence criteria limit ($L^\infty$ residual error bound).
             */
            void solve(int max_iters, double tolerance);
            
            /**
             * @brief Gathers and retrieves the fully reconstructed global solution matrix.
             * * Triggers a collective `MPI_Gatherv` call to pool discrete sub-matrices from 
             * all worker ranks.
             * * @note The complete matrix assembly is only built on **Rank 0**. Calling this 
             * on other ranks will yield an uninitialized/empty matrix block.
             * @return The complete compiled MatrixRowMaj structure containing the global solution.
             */
            MatrixRowMaj get_global_matrix() const;
    };

}


#endif

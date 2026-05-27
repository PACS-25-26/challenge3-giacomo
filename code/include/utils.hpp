    #pragma once

    #ifndef UTILS_HPP
    #define UTILS_HPP

    #include <cmath>
    #include <limits>
    #include <stdexcept>
    #include <mpi.h>
    #include <omp.h>
    #include <Eigen/Dense>
    #include <vector>
    #include <functional>
    #include "jacobi_solver.hpp"

    /**
    * @file utils.hpp
    * @brief Mathematical utility functions and error metrics for verification.
    * * Provides the exact analytical solution, forcing terms, boundary conditions,
    * and validation functions used to benchmark the numerical solver.
     */

    namespace parallel_jacobi{

        /**
        * @brief Computes the analytical exact solution for the benchmark problem.
        * * The exact solution is defined as:
        * $$u(x, y) = \sin(2\pi x) \sin(2\pi y)$$
        * * @param x The spatial coordinate $x$.
        * @param y The spatial coordinate $y$.
        * @return The exact value $u(x,y)$ as a double.
        */

        double exact_solution(double x, double y){
            return std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
        }

        /**
        * @brief Computes the analytical forcing term (RHS) corresponding to the exact solution.
        * * Derived from the Poisson equation $-\nabla^2 u = f$, where applying the Laplacian 
        * to the `exact_solution` yields the source term:
        * $$f(x, y) = 8\pi^2 \sin(2\pi x) \sin(2\pi y)$$
        * * @param x The spatial coordinate $x$.
        * @param y The spatial coordinate $y$.
        *  @return The forcing term value $f(x,y)$ as a double.
        */
        double forcing_term(double x, double y){
            return 8.0 * M_PI * M_PI * std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
        }

        /**
        *  @brief Computes the Dirichlet boundary condition for the benchmark problem.
        * * Returns a homogeneous boundary value ($g(x, y) = 0.0$) along the domain perimeter.
        * * @param x The spatial coordinate $x$.
        * @param y The spatial coordinate $y$.
        * @return The boundary value as a double (always 0.0).
        */
        double boundary_condition(double x, double y){
            return 0.0;
        }

        /**
        * @brief Calculates the global discrete $L^2$ error norm of the numerical solution.
        * * Computes the root-mean-square difference between the calculated matrix values 
        * and the analytical function over all internal grid nodes:
        * ||e||_{L^2} = \sqrt{h \cdot \sum_{i,j} (U_{i,j} - u(x_i, y_j))^2}
        * * @note This function performs a local evaluation. If used in a parallel context, 
        * it should only be executed on the **reconstructed global matrix (Rank 0)**.
        * * @param U The fully assembled global solution matrix of type MatrixRowMaj.
        * @param n The total number of grid points along each 2D dimension.
        * @param h The uniform spatial discretization mesh step size ($h = \Delta x = \Delta y$).
        * @param x_iniz The physical starting boundary coordinate along the $X$-axis.
        * @param y_iniz The physical starting boundary coordinate along the $Y$-axis.
        * @param exact_u A functional reference to the analytical solution function.
        * @return The computed discrete $L^2$ error norm as a double.
        */
        double compute_L2_error(const MatrixRowMaj U,
                                int n, double h, double x_iniz, double y_iniz,
                                std::function<double(double, double)> exact_u){
            double sum = 0.0;
            
            for (int i = 1; i < n - 1; i++) {
                for (int j = 1; j < n - 1; j++) {
                    double x_val = x_iniz + j * h;
                    double y_val = y_iniz + i * h;
                    
                    double diff = U(i, j) - exact_u(x_val, y_val);
                    sum += (diff * diff);
                }
            }
            
            return std::sqrt(h * sum);
        }
    }


    #endif
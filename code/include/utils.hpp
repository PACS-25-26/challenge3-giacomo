#pragma once

#ifndef UTILS_HPP
#define UTILS_HPP

#include "muParser.h"
#include "jacobi_solver.hpp"

/**
 * @file utils.hpp
 * @brief Mathematical utility functions and error metrics for verification.
 */

/**
 * @namespace utils
 * @brief Namespace for utility classes and helper functions.
 */

namespace utils{

    /**
     * @class Function
     * @brief Functor to dinamically evaluate 2D mathematical expressions.
     * * This class uses the muParser library to interpret text strings
     * as mathematical functions f(x,y)
     */

    class Function {
    private:
        std::string expression;

        /**
         * @brief Private helper function to safely retrieve the current MPI rank.
         * * @return The MPI rank of the current process, or 0 if MPI is not initialized.
         */
        
        int get_mpi_rank() const {
            int rank = 0;
            int mpi_initialized = 0;
            MPI_Initialized(&mpi_initialized);
            if (mpi_initialized) {
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            }
            return rank;
        }

    public:
        /**
         * @brief Default constructor.
         * Initializes the function expression to a constant "0.0".
         */
        Function() : expression("0.0") {}
        /**
         * @brief Main constructor that initializes the function with a custom expression.
         * @param expr The string containing the mathematical expression (e.g., "sin(x)*cos(y)").
         */
        explicit Function(const std::string& expr) : expression(expr) {}

        /**
         * @brief Overload of the call operator to evaluate the function at given coordinates.
         * * This operator makes the class a callable Functor. It is thread-safe because 
         * it instantiates internal parser variables as `thread_local`, allowing multiple 
         * OpenMP threads to call it concurrently without race conditions.
         * * If muParser initialization fails, it aborts the entire MPI environment. If evaluation 
         * fails, it prints an error message and returns 0.0.
         * * @param x The spatial coordinate $x$.
         * @param y The spatial coordinate $y$.
         * @return The evaluated result of the mathematical expression as a double.
         */

        double operator()(double x, double y) const {
            thread_local mu::Parser local_parser;
            thread_local double local_x = 0.0;
            thread_local double local_y = 0.0;
            thread_local bool initialized = false;

            if (!initialized) {
                try {
                    local_parser.SetExpr(expression);
                    local_parser.DefineVar("x", &local_x);
                    local_parser.DefineVar("y", &local_y);
                    local_parser.DefineConst("pi", M_PI);
                    initialized = true;
                } catch (mu::Parser::exception_type &e) {
                    std::cerr << "[Rank " << get_mpi_rank() << "] muParser initialization error: " 
                              << e.GetMsg() << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }

            local_x = x;
            local_y = y;

            try {
                return local_parser.Eval();
            } catch (mu::Parser::exception_type &e) {
                std::cerr << "[Rank " << get_mpi_rank() << "] muParser evaluation error: " 
                          << e.GetMsg() << std::endl;
                return 0.0;
            }
        }
    };

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
    double compute_L2_error(const parallel_jacobi::MatrixRowMaj U,
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
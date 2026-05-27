#ifndef FORCING_PARSER_HPP
#define FORCING_PARSER_HPP

#include "muParser.h"
#include "mpi.h"
#include <string>
#include <iostream>
#include <cmath>

/**
 * @file forcing_parser.hpp
 * @brief Definition of the class ForcingFunction for parallel computation.
 */

/**
 * @namespace utils
 * @brief Namespace for utility classes and helper functions.
 */

namespace utils {

    /**
     * @class ForcingFunction
     * @brief Functor to dinamically evaluate 2D mathematical expressions.
     * * This class uses the muParser library to interpret text strings
     * as mathematical functions f(x,y)
     */

    class ForcingFunction {
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
        ForcingFunction() : expression("0.0") {}
        /**
         * @brief Main constructor that initializes the function with a custom expression.
         * @param expr The string containing the mathematical expression (e.g., "sin(x)*cos(y)").
         */
        explicit ForcingFunction(const std::string& expr) : expression(expr) {}

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

} // namespace utils

#endif // FORCING_PARSER_HPP
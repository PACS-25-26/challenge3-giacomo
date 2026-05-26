#ifndef FORCING_PARSER_HPP
#define FORCING_PARSER_HPP

#include "muParser.h"
#include "mpi.h"
#include <string>
#include <iostream>
#include <cmath>

namespace utils {

    class ForcingFunction {
    private:
        std::string expression;

        // Funzione helper privata per stampare il rank MPI corretto in caso di errore
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
        // Costruttore di default
        ForcingFunction() : expression("0.0") {}
        
        // Costruttore principale che riceve la stringa da GetPot
        explicit ForcingFunction(const std::string& expr) : expression(expr) {}

        // Overload dell'operatore () per rendere la classe un Functor utilizzabile da JacobiSolver
        double operator()(double x, double y) const {
            // Ogni thread OpenMP istanzia la propria copia locale di queste variabili statiche
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

            // Aggiorna in modo sicuro i valori per il thread corrente
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
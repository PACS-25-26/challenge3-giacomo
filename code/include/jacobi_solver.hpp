#pragma once

#ifndef JACOBI_SOLVER_HPP
#define JACOBI_SOLVER_HPP

#include <cmath>
#include <limits>
#include <stdexcept>
#include <mpi.h>
#include <omp.h>
#include <Eigen/Dense>
#include <vector>
#include <functional>

namespace parallel_jacobi{

    using MatrixRowMaj = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    class JacobiSolver{
        
        private:
            int n;
            double h;
            double x_iniz, y_iniz, x_fin, y_fin;
            std::function<double(double, double)> f;
            std::function<double(double, double)> g;

            MPI_Comm comm;
            int rank, size;
            int local_rows;
            int up_neighbor, down_neighbor;
            int start_i, end_i;
            std::vector<int> sendcounts;
            std::vector<int> displs;

            MatrixRowMaj U_global; 
            MatrixRowMaj U_loc0;
            MatrixRowMaj U_loc1;

            void initialize_global_bc();
            void exchange_halos();
            void compute_step();
            double compute_error_and_update();

        public:
            JacobiSolver(int n, double x_i, double y_i, double x_f, double y_f,
                               std::function<double(double, double)> force,
                               std::function<double(double, double)> bc,
                               MPI_Comm communicator);

            void solve(int max_iters, double tolerance);

            MatrixRowMaj get_global_matrix() const;
    };

}


#endif

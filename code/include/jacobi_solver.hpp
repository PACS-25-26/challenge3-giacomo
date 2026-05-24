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

    class JacobiSolver{
        
        private:
            int num_elem;
            double x_iniz;
            double y_iniz;
            double x_fin;
            double y_fin:
            std::function<double(double, double)> f;
            std::function<double(double, double)> g;

        public:
            JacobiSolver(int n, int rank, int size);

            void solve(int max_iters, double tolerance);
    };

}


#endif

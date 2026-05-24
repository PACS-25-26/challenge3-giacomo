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

namespace parallel_jacobi{

    double exact_solution(double x, double y){
        return std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
    }

    double forcing_term(double x, double y){
        return 8.0 * M_PI * M_PI * std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
    }

    double boundary_condition(double x, double y){
        return 0.0;
    }

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
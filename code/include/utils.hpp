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

namespace parallel_jacobi{

    double exact_solution(double x, double y);
    double forcing_term(double x, double y);
    double boundary_condition(double x, double y);
    double compute_L2_error(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& U,
                            int n, double h, double x_iniz, double y_iniz,
                            std::function<double(double, double)> exact_u);
}


#endif
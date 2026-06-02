#pragma once

#ifndef VTK_EXPORTER_HPP
#define VTK_EXPORTER_HPP

#include "jacobi_solver.hpp"


/**
 * @file vtk_exporter.hpp
 * @brief Utility function to export 2D simulation data to VTK Legacy format.
 * * Allows visualization of the calculated potential fields or system profiles 
 * using post-processing software such as ParaView or VisIt.
 */

namespace utils {

    /**
     * @brief Exports a 2D matrix solution to an ASCII VTK Legacy structured points file.
     * * This function creates a structured grid representation ($N \times N \times 1$) 
     * where the Z-dimension is flattened. It maps the row-major Eigen matrix data into the 
     * coordinate system expected by VTK format rules.
     * * @note VTK structures iterate over $X$ coordinates fastest, then $Y$, then $Z$. 
     * Given that the matrix row index $i$ represents spatial grid height ($Y$-axis) and 
     * column index $j$ represents width ($X$-axis), the internal loop correctly unrolls 
     * the sequential output stream to maintain physical spatial orientation.
     * * @param filename The output path and name of the `.vtk` file to create.
     * @param U The computed solution matrix of type MatrixRowMaj to write.
     * @param n The total number of grid resolution points along each 2D dimension ($N \times N$).
     * @param x_iniz The physical starting boundary coordinate along the $X$-axis.
     * @param y_iniz The physical starting boundary coordinate along the $Y$-axis.
     * @param h The uniform spatial discretization mesh spacing step size ($h = \Delta x = \Delta y$).
     */
    inline void export_vtk(const std::string& filename, 
                           const parallel_jacobi::MatrixRowMaj& U, 
                           int n, double x_iniz, double y_iniz, double h) {
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to create the file " << filename << "\n";
            return;
        }

        // 1. STANDARD HEADER for VTK Legacy Format
        file << "# vtk DataFile Version 3.0\n";
        file << "Solution Equazione Poisson - Parallel Jacobi\n";
        file << "ASCII\n";
        
        // 2. DATASET STRUCTURED_POINTS
        file << "DATASET STRUCTURED_POINTS\n";
        file << "DIMENSIONS " << n << " " << n << " 1\n"; 
        file << "ORIGIN " << x_iniz << " " << y_iniz << " 0.0\n";
        file << "SPACING " << h << " " << h << " 1.0\n";
        
        // 3. POINT_DATA with scalar field "solution"
        file << "POINT_DATA " << n * n << "\n";
        file << "SCALARS solution double 1\n";
        file << "LOOKUP_TABLE default\n";

        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                file << U(i, j) << "\n";
            }
        }

        file.close();
        std::cout << "VTK export completed successfully: " << filename << std::endl;
    }
}

#endif
#pragma once
#ifndef VTK_EXPORTER_HPP
#define VTK_EXPORTER_HPP

#include "jacobi_solver.hpp"
#include <iostream>
#include <fstream>
#include <string>

namespace parallel_jacobi {

    // Funzione inline per evitare definizioni multiple in fase di linking
    inline void export_vtk(const std::string& filename, 
                           const MatrixRowMaj& U, 
                           int n, double x_iniz, double y_iniz, double h) {
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Errore: Impossibile creare il file " << filename << "\n";
            return;
        }

        // 1. HEADER standard del formato VTK Legacy
        file << "# vtk DataFile Version 3.0\n";
        file << "Soluzione Equazione Poisson - Jacobi Parallelo\n";
        file << "ASCII\n";
        
        // 2. DEFINIZIONE DELLA GRIGLIA
        file << "DATASET STRUCTURED_POINTS\n";
        file << "DIMENSIONS " << n << " " << n << " 1\n"; // 1 sta per la dimensione Z (che per noi è piatta)
        file << "ORIGIN " << x_iniz << " " << y_iniz << " 0.0\n";
        file << "SPACING " << h << " " << h << " 1.0\n";
        
        // 3. DATI (SCALARS)
        file << "POINT_DATA " << n * n << "\n";
        file << "SCALARS soluzione double 1\n";
        file << "LOOKUP_TABLE default\n";

        // VTK si aspetta che i dati scorrano variando prima la X, poi la Y, poi la Z.
        // Nel tuo solver: indice i -> Y, indice j -> X.
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                file << U(i, j) << "\n";
            }
        }

        file.close();
        std::cout << "Esportazione VTK completata con successo: " << filename << std::endl;
    }
}

#endif
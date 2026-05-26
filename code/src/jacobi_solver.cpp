#include "jacobi_solver.hpp"
#include <iostream>

namespace parallel_jacobi{

    JacobiSolver::JacobiSolver(int n_, double x_i, double y_i, double x_f, double y_f,
                               std::function<double(double, double)> force,
                               std::function<double(double, double)> bc,
                               MPI_Comm communicator):
    n(n_), x_iniz(x_i), y_iniz(y_i), x_fin(x_f), y_fin(y_f), f(force), g(bc), comm(communicator)
    {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        h = (x_fin - x_iniz) / (n - 1);

        int base_rows = n / size;
        int remainder = n % size;
        local_rows = (rank < remainder) ? (base_rows + 1) : base_rows;

        up_neighbor   = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        down_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
        
        start_i = (rank == 0) ? 2 : 1; 
        end_i   = (rank == size - 1) ? local_rows - 1 : local_rows;

        sendcounts.resize(size, 0);
        displs.resize(size, 0);
        
        if (rank == 0){
            U_global.resize(n, n);
            initialize_global_bc(); 
            int offset = 0;
            for (int i = 0; i < size; i++) {
                if (i < remainder){
                    sendcounts[i] = (base_rows + 1) * n;
                }
                else{
                    sendcounts[i] = (base_rows) * n;
                }
                displs[i] = offset;
                offset += sendcounts[i];
            }
        }

        // CORREZIONE: Distribuiamo le tabelle di comunicazione a tutti i processi
        MPI_Bcast(sendcounts.data(), size, MPI_INT, 0, comm);
        MPI_Bcast(displs.data(), size, MPI_INT, 0, comm);

        // Definizione delle matrici locali (Tutti i processi)
        U_loc0 = MatrixRowMaj::Zero(local_rows + 2, n);
        U_loc1 = MatrixRowMaj::Zero(local_rows + 2, n);

        // Distribuzione dati iniziale (Tutti i processi partecipano)
        MPI_Scatterv(U_global.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, 
                     &U_loc0(1, 0), local_rows * n, MPI_DOUBLE, 0, comm);
    }

    void JacobiSolver::initialize_global_bc(){
        U_global.setZero();
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            U_global(0, j) = g(x_iniz + j * h, y_fin);
            U_global(n - 1, j) = g(x_iniz + j * h, y_iniz);
        }

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            U_global(i, 0) = g(x_iniz, y_iniz + i * h);
            U_global(i, n - 1) = g(x_fin, y_iniz + i * h);
        }
    }

    void JacobiSolver::exchange_halos() {
        // 1. SHIFT VERSO IL BASSO
        // Mando la mia ultima riga al vicino di sotto, e ricevo la riga dal vicino di sopra
        MPI_Sendrecv(&U_loc0(local_rows, 0), n, MPI_DOUBLE, down_neighbor, 0,
                     &U_loc0(0, 0),          n, MPI_DOUBLE, up_neighbor,   0,
                     comm, MPI_STATUS_IGNORE);

        // 2. SHIFT VERSO L'ALTO
        // Mando la mia prima riga al vicino di sopra, e ricevo la riga dal vicino di sotto
        MPI_Sendrecv(&U_loc0(1, 0),              n, MPI_DOUBLE, up_neighbor,   1,
                     &U_loc0(local_rows + 1, 0), n, MPI_DOUBLE, down_neighbor, 1,
                     comm, MPI_STATUS_IGNORE);
    }

    void JacobiSolver::compute_step() {
        double coeff = 0.25;
        #pragma omp parallel for 
        for (int i = start_i; i <= end_i; i++) {
            for (int j = 1; j < n - 1; j++) {
                int global_i = (displs[rank] / n) + (i - 1);                
                U_loc1(i, j) = coeff * (U_loc0(i - 1, j) + U_loc0(i + 1, j) + 
                                        U_loc0(i, j - 1) + U_loc0(i, j + 1) + 
                                        h * h * f(j * h + x_iniz, global_i * h + y_iniz) );
            }
        }
    }

    double JacobiSolver::compute_error_and_update() {
        double loc_sum = 0;
        #pragma omp parallel for reduction(+:loc_sum)
        for (int i = start_i; i <= end_i; i++) {
            for (int j = 1; j < n - 1; j++) {
                double diff = U_loc1(i, j) - U_loc0(i, j);
                loc_sum += (diff * diff);
                U_loc0(i, j) = U_loc1(i, j);
            }
        }
        return std::sqrt(h * loc_sum);
    }

    void JacobiSolver::solve(int max_iters, double tolerance) {
        int k = 0;
        int flag = 0;
        int flag_glob = 0;
        double err = tolerance + 1.0;

        while (!flag_glob && k < max_iters) {
            exchange_halos();
            compute_step();
            err = compute_error_and_update();
            
            if (err < tolerance)
                flag = 1;
                
            MPI_Allreduce(&flag, &flag_glob, 1, MPI_INT, MPI_PROD, comm);
            
            if (rank == 0 && k % 100 == 0) {
                std::cout << "Iterazione: " << k << " | Errore Corrente: " << err << std::endl;
            }
            k++;
        }

        if (rank == 0) std::cout << "Calcolo completato in " << k << " iterazioni. Avvio Gatherv..." << std::endl;
        
        // Raccolta dati finale: prendiamo i dati da U_loc0(1,0)
        MPI_Gatherv(&U_loc0(1, 0), local_rows * n, MPI_DOUBLE, 
                    U_global.data(), sendcounts.data(), displs.data(), 
                    MPI_DOUBLE, 0, comm);
                    
        if (rank == 0) std::cout << "Gatherv terminata con successo!" << std::endl;
    }

    MatrixRowMaj JacobiSolver::get_global_matrix() const {
        return U_global;
    }

} 
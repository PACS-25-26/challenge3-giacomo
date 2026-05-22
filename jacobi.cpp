#include <cmath>
#include <limits>
#include <stdexcept>
#include <eigen>
#include <mpi.h>
#include <omp.h>

using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixRowMaj; //defined eigen matrix as row major

//function for parallel jacobi iteration
MatrixRowMaj jacobi(double tol, int num_elem, int max_iters, double x_iniz, double y_iniz, double x_fin, double y_fin, std::function<double(double, double)> f, std::function<double(double, double)> g, MPI_Comm comm){
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double n = sqrt(num_elem);
    double h = (x_fin-x_iniz) / (n-1);

    //computation of local rows for each processor
    int base_rows = n / size;
    int remainder = n % size;
    if (rank < reminder){       //some processor will have a row more
        local_rows = base_rows+1;
    }
    else{
        local_rows = base_rows;
    }

    //intializazion of gloabl matrix and vectors for scatterv
    std::vector<int> sendcounts(size, 0);
    std::vector<int> displs(size, 0);
    MatrixRowMaj U_global;
    if (rank == 0){
        U_global.resize(n, n);
        initializeBC(g, U_global,x_iniz,y_iniz,n,h); //funct that initialize U_glob with correct BC
        int offset = 0;
        for (int i = 0; i < size; i++) {
            if (i<reminder){
                sendcounts[i] = (base_rows+1)*n;
            }
            else{
                sendcounts[i] = (base_rows)*n;
            }
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    //definition of local matrices
    MatrixRowMaj U_loc0(local_rows + 2, n);
    MatrixRowMaj U_loc1(local_rows + 2, n);
    U_loc0.setZero();
    U_loc1.setZero();

    //scatterv
    MPI_Scatterv(U_global.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, &U_local(1, 0), local_rows * n, MPI_DOUBLE, 
        0, comm);

    //neighbours definition for rows comunications
    int up_neighbor   = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    
    //first and last processor do not communicate with anyone
    int start_i = (rank == 0) ? 2 : 1; 
    int end_i   = (rank == size - 1) ? local_rows - 1 : local_rows;

    int k = 0;
    int flag=0;
    while (!flag && k<max_iters){
        //communication between processors, two steps to avoid deadlocks
        if (rank % 2 == 0) {
            MPI_Send(&U_loc0(local_rows, 0), n, MPI_DOUBLE, down_neighbor, 0, comm);
            MPI_Recv(&U_loc0(local_rows + 1, 0), n, MPI_DOUBLE, down_neighbor, 1, comm, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(&U_loc0(0, 0), n, MPI_DOUBLE, up_neighbor, 0, comm, MPI_STATUS_IGNORE);
            MPI_Send(&U_loc0(1, 0), n, MPI_DOUBLE, up_neighbor, 1, comm);
        }

        if (rank % 2 == 0) {
            MPI_Send(&U_loc0(1, 0), n, MPI_DOUBLE, up_neighbor, 0, comm);
            MPI_Recv(&U_loc0(0, 0), n, MPI_DOUBLE, up_neighbor, 1, comm, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(&U_loc0(local_rows + 1, 0), n, MPI_DOUBLE, down_neighbor, 0, comm, MPI_STATUS_IGNORE);
            MPI_Send(&U_loc0(local_rows, 0), n, MPI_DOUBLE, down_neighbor, 1, comm);
        }

     
        // jacobi iteration
        #pragma omp parallel for 
        for (int i = start_i; i <= end_i; i++) {
            for (int j = 1; j < n - 1; j++) {
                int global_i = (rank == 0 ? 0 : displs[rank]/n) + (i - 1); 
                
                U_loc1(i, j) = coeff * ( U_loc0(i - 1, j) + U_loc0(i + 1, j) + 
                                        U_loc0(i, j - 1) + U_loc0(i, j + 1) + 
                                        f(global_i+x_iniz, j+y_iniz) );
            }
        }

        //error and update
        k++;
        int loc_sum = 0;
        #pragma omp parallel for 
        for (int i = start_i; i <= end_i; i++) {
            for (int j = 1; j < n - 1; j++) {
                double diff = U_loc1(i, j) - U_loc0(i, j);
                loc_sum += (diff * diff); // Somma dei quadrati
                
                //update
                U_loc0(i, j) = U_loc1(i, j);
            }
        }
        double err = std::sqrt(h * loc_sum);
        if (err<tol){
            flag=1;
        }
        MPI_Allreduce(&flag, &flag, 1, MPI_DOUBLE, MPI_PROD, comm);

    }

    MPI_Gatherv(&U_local(1, 0), local_rows * n, MPI_DOUBLE, U_global.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,0, comm);

    return U_glob;
}

//function to construct global U0, according to bc by g
void initializeBC(std::function<double(double, double)> g, MatrixRowMaj& u, double x_iniz, double y_iniz, double n, double h){
    u.setZero();
    #pragma omp parallel for
    for (int j = 0; j < n ; j++) {
        u(0,j) = g(x_iniz+j*h, y_iniz+n*h);
        u(n-1,j) = g(x_iniz+j*h, y_iniz);
    }

    #pragma omp parallel for
    for (int i = 0; i < n ; i++) {
        u(i,0) = g(x_iniz, y_iniz+j*h);
        u(i,n-1) = g(x_iniz+n*h, y_iniz+j*h);
    }
}
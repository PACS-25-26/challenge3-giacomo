## CHALLENGE 3 - Meda, Mietto, Romi

Our code is divided into three main folders: 
* `include` contains the hpp files
* `src` contains all the cpp files including the main
* `test` contains the scripts, the source files, and the generated data for the scalability tests, along with the analysis reports.

Before running the code, the user can choose different values for some parameters of the problem (as num_elements, max_iters, tolerance, forcing term): it's enough to go in the file `parameters.pot`, change the values and then digit **make run**. It's also important that user's device has all the tools required, in particoular Eigen and muParser.  

Regarding the implementation of the code:
* `jacobi_solver.hpp` contains a class that is called **JacobiSolver** and deal with the parallel implementation of the Jacobi iteration method;
* `utils.hpp` contains the implementation of useful tools for other parts of the code, including a functor that using muParser dynamically evaluate string-based math expressions. It leverages thread_local instances for OpenMP compatibility and features MPI-aware error logging for parallel environments;
* `vtk_exporter.hpp` takes care of post-processing and the visualization of the solution. It produces a file 'solution.vtk', that can be open with ParaView;
* `jacobi_solver.cpp` contains the definitions of the functions declared in 'jacobi_solver.hpp'.
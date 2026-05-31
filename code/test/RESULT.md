## Results discussion

The file 'data/results.csv' contains the execution times (Time_s) with respect to grid size(GridSize_n) and number of cores used (Cores).

To evaluate the performance, we can observe the scaling (how the time decreases while keeping the problem size fixed and increasing the number of cores) on the heaviest test cases:
**Grid N = 128:**
  * 1 Core:** 3.35 s
  * 2 Cores:** 1.80 s
  * 4 Cores:** 1.20 s <br>
In this case, we observe good scalability. By doubling the cores from 1 to 2, the execution time is almost halved. Moving to 4 cores we observe a further improvement, even if less pronounced.
**Grid N = 256:**
 * 1 Core:** 31.24 s
 * 2 Cores:** 17.81 s
 * 4 Cores:** 17.62 s <br>
For the larger grid, the transition from 1 to 2 cores provides a good speedup, almost halving the times. However, using 4 cores does not bring significant improvements compared to 2 cores.

* ** Conclusions ** <br>
The code demonstrates good scaling when moving from a serial execution to a parallel one on 2 cores. However, by further increasing the number of processes (4 cores), a flattening of performance is noticeable (particularly on the 256 grid). This behavior may suggests that, in this specific configuration, the communication overhead between MPI processes becomes prevalent and cancels out the computational advantages derived from having more cores available.
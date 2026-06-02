## Results discussion

The file `data/results.csv` contains the execution times (Time_s) with respect to grid size (GridSize_n) and number of cores used (Cores).

To evaluate the performance, we can observe the scaling (how the time decreases while keeping the problem size fixed and increasing the number of cores) on the heaviest test cases:<br>

**Grid N = 128:**
  * 1 Core: 2.49 s
  * 2 Cores: 1.29 s
  * 4 Cores: 0.73 s <br>

In this case, we observe good scalability. By doubling the cores from 1 to 2, the execution time is almost halved. Moving to 4 cores we observe a further improvement, even if less pronounced.<br>

**Grid N = 256:**
 * 1 Core: 28.75 s
 * 2 Cores: 15.27 s
 * 4 Cores: 7.78 s <br>

For the larger grid, the transition from 1 to 2 cores provides a good speedup, almost halving the times. Using 4 cores does also bring significant improvements compared to 2 cores.


**Conclusions** <br>
The code demonstrates good scaling when moving from a serial execution to a parallel one on 2 cores. Even by further increasing the number of processes (4 cores), the performance remains good. This behavior may suggests that, in this specific configuration, the communication overhead between MPI processes does not become prevalent and underlines computational advantages derived from having more cores available. It is also noticable that the L2 error remains of the same order of magnitude in all settings. 
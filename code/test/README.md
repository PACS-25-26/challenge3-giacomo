## Compilation and Execution (Test Directory)

To run the accuracy and parallel scalability tests, you must compile and execute the code directly from within the `test` directory. The local `Makefile` is configured to automatically link the test source with the core solver located in the parent directory.

Open your terminal and navigate to the test folder: `cd code/test`.

To compile the specific test executable (run_test), simply run: `make`.

An automated Bash script is provided to handle the strong scaling tests across multiple core counts (1, 2, 4, 8 cores).
First, ensure the script has execution permissions:
`chmod +x run_scalability.sh`.
Then, execute it: `./run_scalability.sh`

Depending on your needs, you can perform a partial or full cleanup of the directory, with only object files: `make clean`, or also the data folder: `make distclean`
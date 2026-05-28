CXX = mpicxx
mkEigenInc = -I/usr/include/eigen3
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -I../include -I../include/GetPot -fopenmp $(mkEigenInc)
LDFLAGS = -fopenmp -lmuparser

SRC_DIR = ../src
TEST_DIR = .
OBJ_DIR = ./obj

EXEC = run_test
OBJS = $(OBJ_DIR)/jacobi_solver.o $(OBJ_DIR)/run_test.o

.PHONY: all
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)
	@echo "run_test compiled!"

$(OBJ_DIR)/jacobi_solver.o: $(SRC_DIR)/jacobi_solver.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/run_test.o: $(TEST_DIR)/run_test.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(EXEC) 
	@echo "Clean completed."

.PHONY: distclean
distclean: clean
	rm -rf ./data 
	@echo "Distclean completed."
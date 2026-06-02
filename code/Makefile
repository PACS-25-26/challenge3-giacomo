CXX = mpicxx
mkEigenInc = -I/usr/include/eigen3
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -Iinclude -fopenmp $(mkEigenInc)

LDFLAGS = -fopenmp
LDFLAGS += -lmuparser

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj

SRCS = $(wildcard $(SRC_DIR)/*.cpp)

OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

EXEC = jacobi_solver

.PHONY: all
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)
	@echo "Hybrid build (MPI + OpenMP) completed! run with mpirun."

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(EXEC)
	@echo "Clean completed: object file and executable removed."

.PHONY: run
run: $(EXEC)
	OMP_NUM_THREADS=2 mpirun -n 4 ./$(EXEC) -f parameters.pot
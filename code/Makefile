
CXX = mpicxx
mkEigenInc = -I/usr/include/eigen3
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -Iinclude -fopenmp $(mkEigenInc)

LDFLAGS = -fopenmp

# LDFLAGS += -lmuparser

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj

SRCS = $(wildcard $(SRC_DIR)/*.cpp)

OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

EXEC = jacobi_solver


# Regola di default
.PHONY: all
all: $(EXEC)

# Linking dell'eseguibile
$(EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)
	@echo "Build ibrida (MPI + OpenMP) completata! Esegui con mpirun."

# Compilazione dei singoli file oggetto
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Creazione della cartella obj se non esiste
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)


# Pulizia dei file compilati
.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(EXEC)
	@echo "File oggetto ed eseguibile rimossi."

.PHONY: run
run: $(EXEC)
	OMP_NUM_THREADS=2 mpirun -n 2 ./$(EXEC) 100 2
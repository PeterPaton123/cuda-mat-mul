CC = g++
NVCC = nvcc
CXXFLAGS = -I./src
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart  

SRC_DIR = src
OBJ_DIR = obj
TARGET = out/main

CPP_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
CU_SOURCES = $(SRC_DIR)/matmul.cu
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SOURCES))
CU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SOURCES))

# Default target
all: $(TARGET)

# Link objects into the final executable
$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CC) $^ $(LDFLAGS) -o $@

# Compile C++ source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(CXXFLAGS) -c $< -o $@

# Clean target to remove object files and the executable
clean:
	rm -f $(TARGET) $(CPP_OBJECTS) $(CU_OBJECTS)

.PHONY: all clean

#include "matrix.h"
#include <cuda.h>
#include <stdexcept>

__global__ void DeviceMatMul(int *a, int *b, int *c, int rows_c, int cols_c, int shared_dim) {
  // (id_x, id_y) is the coordinate in the output matrix this thread calculates, coordinate is a functionof thread id (determined by block index and thread index in the block).
  int id_x = blockIdx.x * blockDim.x + threadIdx.x;
  int id_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (id_x < rows_c && id_y < cols_c) {
    for (int k = 0; k < shared_dim; k++) {
      c[id_x * cols_c + id_y] += a[id_x * shared_dim + k] * b[k * cols_c + id_y];
    }
  }
}

Matrix *matMul(Matrix* mat_a, Matrix* mat_b) {
    if (mat_a->cols != mat_b->rows) {
        throw std::invalid_argument("Invalid dimensions for multiplication: matrix 1 cols = " + std::to_string(mat_a->cols) + ", matrix 2 rows = " + std::to_string(mat_b->rows));
    }

    // Pointers to the device memory address of the matrices
    int *d_mat_a, *d_mat_b, *d_mat_c; 

    if (cudaMalloc(&d_mat_a, sizeof(int) * mat_a->rows * mat_a->cols) != cudaSuccess) {
        throw std::runtime_error("Could not allocate memory on GPU");
     }
  
    if (cudaMalloc(&d_mat_b, sizeof(int) * mat_b->rows * mat_b->cols) != cudaSuccess) {
        cudaFree(d_mat_a);
        throw std::runtime_error("Could not allocate memory on GPU");
    }
  
    if (cudaMalloc(&d_mat_c, sizeof(int) * mat_a->rows * mat_b->cols) != cudaSuccess) {
        cudaFree(d_mat_a);
        cudaFree(d_mat_b);
        throw std::runtime_error("Could not allocate memory on GPU");
    }

    if (cudaMemcpy(d_mat_a, mat_a->data, sizeof(int) * mat_a->rows * mat_a->cols, cudaMemcpyHostToDevice) != cudaSuccess || cudaMemcpy(d_mat_b, mat_b->data, sizeof(int) * mat_b->rows * mat_b->cols, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_mat_a);
        cudaFree(d_mat_b);
        cudaFree(d_mat_c);
        throw std::runtime_error("Could not write to GPU"); 
    }

    if (cudaMemset(d_mat_c, 0, sizeof(int) * mat_a->rows * mat_b->cols) != cudaSuccess) {
        cudaFree(d_mat_a);
        cudaFree(d_mat_b);
        cudaFree(d_mat_c);
        throw std::runtime_error("Could not write to GPU");
    }

    dim3 blockSize(mat_a->rows / 16 + 1, mat_b->cols / 16 + 1);
    dim3 threadSize(16, 16);

    DeviceMatMul<<<blockSize, threadSize>>>(d_mat_a, d_mat_b, d_mat_c, mat_a->rows, mat_b->cols, mat_a->cols);
  
    // Output vector on host memory
    int *h_mat_c = new int[mat_a->cols * mat_b->cols];

    if (cudaMemcpy(h_mat_c, d_mat_c, sizeof(int) * mat_a->rows * mat_b->cols, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_mat_a);
        cudaFree(d_mat_b);
        cudaFree(d_mat_c);
        throw std::runtime_error("Could not write from GPU");
    }

    cudaFree(d_mat_a);
    cudaFree(d_mat_b);
    cudaFree(d_mat_c);

    return new Matrix{mat_a->rows, mat_b->cols, h_mat_c}; 
}

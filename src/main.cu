#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>

void logMat(int *matrix, int rows, int cols) {
  // std::cout << "[";
  for (int i = 0; i < rows; i++) {
    std::cout << "[";    
    for (int j = 0; j < cols - 1; j++) {
      std::cout << matrix[i * cols + j] << ", ";
    }
    std::cout << matrix[i * cols + cols - 1] << "]\n";
  }
  std::cout << std::endl;
  // std::cout << "]" << std::endl;
}
    

__global__ void MatMul(int *a, int *b, int *c, int rows_c, int cols_c, int shared_dim) {
  // (id_x, id_y) is the coordinate in the output matrix this thread calculates, coordinate is a functionof thread id (determined by block index and thread index in the block).
  int id_x = blockIdx.x * blockDim.x + threadIdx.x;
  int id_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (id_x < rows_c && id_y < cols_c) {
    for (int k = 0; k < shared_dim; k++) {
      c[id_x * cols_c + id_y] += a[id_x * shared_dim + k] * b[k * cols_c + id_y];
    }
  }
}

int main() {
  std::srand(time(NULL));
  int rows_a = 3;
  int cols_a = 4000;
  int rows_b = 4000;
  int cols_b = 3;
 
  // Array a and b stored on host, hence h_mat_a
  int *h_mat_a = new int[rows_a * cols_a];
  int *h_mat_b = new int[rows_b * cols_b]; 
  
  // Array a and b stored on device, hence d_mat_a
  int *d_mat_a, *d_mat_b, *d_mat_c;

  for (int i = 0; i < rows_a * cols_a; i++) {
    h_mat_a[i] = std::rand() % (rows_a * cols_a) + 1;
  }
 
  for (int i = 0; i < rows_b * cols_b; i++) {
    h_mat_b[i] = std::rand() % (rows_b * cols_b) + 1;
  }

  // TODO: Check cols_a == rows_b

  if (cudaMalloc(&d_mat_a, sizeof(int) * rows_a * cols_a) != cudaSuccess) {
    std::cout << "CUDA Malloc failed" << std::endl;
    return -1;
  }
  
  if (cudaMalloc(&d_mat_b, sizeof(int) * rows_b * cols_b) != cudaSuccess) {
    cudaFree(d_mat_a);    
    std::cout << "CUDA Malloc failed" << std::endl;
    return -1;  
  }
  
  if (cudaMalloc(&d_mat_c, sizeof(int) * rows_a * cols_b) != cudaSuccess) {
    cudaFree(d_mat_a);    
    cudaFree(d_mat_b);    
    std::cout << "CUDA Malloc failed" << std::endl;
    return -1;  
  }

  if (cudaMemset(d_mat_c, 0, sizeof(int) * rows_a * cols_b) != cudaSuccess) {
    std::cout << "Could not initialise output matrix on device" << std::endl;
    return -1;
  }

  if (cudaMemcpy(d_mat_a, h_mat_a, sizeof(int) * rows_a * cols_a, cudaMemcpyHostToDevice) != cudaSuccess || cudaMemcpy(d_mat_b, h_mat_b, sizeof(int) * rows_b * cols_b, cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cout << "Could not copy" << std::endl;
    cudaFree(d_mat_a);
    cudaFree(d_mat_b);
    return -1;
}

  dim3 blockSize(rows_a / 16 + 1, cols_b / 16 + 1);
  dim3 threadSize(16, 16);

  MatMul<<<blockSize, threadSize>>>(d_mat_a, d_mat_b, d_mat_c, rows_a, cols_b, cols_a);
  
  // Output vector on device memory
  int *h_mat_c = new int[rows_a * cols_b];

  if (cudaMemcpy(h_mat_c, d_mat_c, sizeof(int) * rows_a * cols_b, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cout << "Copy back to host from device failed!" << std::endl;
    delete []h_mat_a;
    delete []h_mat_b; 
    cudaFree(d_mat_a);
    cudaFree(d_mat_b);
  }

  /*
  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      for (int k = 0; k < cols_a; k++) {
        h_mat_c[i * cols_b + j] += h_mat_a[i * cols_a + k] * h_mat_b[k * cols_b + j];
      }
    }
  }
  */

  logMat(h_mat_a, rows_a, cols_a);
  logMat(h_mat_b, rows_b, cols_b);
  logMat(h_mat_c, rows_a, cols_b); 
  
  return 0;
}

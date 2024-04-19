#include "matrix.h"
#include <iostream>

void logMat(Matrix* matrix) {
    int rows = matrix-> rows;
    int cols = matrix -> cols;
    for (int i = 0; i < matrix->rows; i++) {
        std::cout << "[";    
        for (int j = 0; j < cols - 1; j++) {
            std::cout << matrix->data[i * cols + j] << ", ";
        }
        std::cout << matrix->data[i * cols + cols - 1] << "]\n";
    }
    std::cout << std::endl;
}

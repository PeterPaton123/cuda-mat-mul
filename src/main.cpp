#include "matrix.h"
#include "matmul.cuh"

int main() {
    Matrix* mat_1 = new Matrix{2, 2, new int[4]{1, 2, 3, 4}};
    Matrix* mat_2 = new Matrix{2, 2, new int[4]{1, 2, 3, 4}};

    logMat(matMul(mat_1, mat_2));
    return 0;
}
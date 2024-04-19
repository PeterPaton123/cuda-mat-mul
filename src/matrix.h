#pragma once

struct Matrix {
    int rows;
    int cols;
    int *data;
};

void logMat(Matrix* matrix);
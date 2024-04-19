#include "matrix.h"
#include "matmul.cuh"
#include "json_parser.hpp"

int main() {
    nlohmann::json j = {
        {"row_dim", 2},
        {"col_dim", 2},
        {"data", {1, 2, 3, 4}}
    };

    nlohmann::json j2 = {
        {"row_dim", 2},
        {"col_dim", 2},
        {"data", {1, 2, 3, 4}}
    };

    Matrix *mat_1 = parseJson(j);
    Matrix *mat_2 = parseJson(j2);
    
    logMat(matMul(mat_1, mat_2));
    return 0;
}
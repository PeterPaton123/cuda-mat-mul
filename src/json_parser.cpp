#include "json_parser.hpp"

Matrix* parseJson(nlohmann::json jsonObject) {
    int rows = jsonObject["row_dim"];
    int cols = jsonObject["col_dim"];

    if (jsonObject["data"].size() != static_cast<size_t>(rows * cols)) {
        throw std::invalid_argument("Dimensions specified in JSON do not match sturcture of data in JSON.");
    }

    int* matrixData = new int[rows * cols];
    for (int i = 0; i < rows * cols; i++) {
        matrixData[i] = jsonObject["data"][i];
    }

    return new Matrix{rows, cols, matrixData};
}
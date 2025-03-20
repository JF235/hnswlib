#include "hnswlib/hnswlib.h"
#include "npy.hpp"

int main() {

    // Read NPY
    const std::string path {"score_matrix.npy"};
    npy::npy_data<float> d = npy::read_npy<float>(path);
    std::vector<float> vecData = d.data;
    std::vector<size_t> shape = d.shape;
    bool fortran_order = d.fortran_order;

    // Shape loaded from NPY
    std::cout << "Shape: ";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i] << " ";
    }
    std::cout << "\n";
    
    // The struct that contains the table and the number of points.
    hnswlib::GrMatchParam param;
    param.num_points = shape[0];
    param.table = new float[param.num_points * param.num_points];
    for (size_t i = 0; i < param.num_points; i++) {
        for (size_t j = 0; j < param.num_points; j++) {
            param.table[i * param.num_points + j] = static_cast<float>(vecData[i * shape[1] + j]);
        }
    }
    hnswlib::GrMatchSpace space(&param);

    // Create a representation for my data.
    // As the distances are already computed, I can store just the indices.
    size_t *data = new size_t[param.num_points];
    for (size_t i = 0; i < param.num_points; i++) {
        data[i] = i;
    } 


    // Build the index
    size_t max_elements = param.num_points;
    size_t M = 3;
    size_t ef_construction = 10;
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Add data
    for (size_t i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i, i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (size_t i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";

    // Serialize index
    std::string hnsw_path = "hnsw.bin";
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;

    // Deserialize index and check recall
    alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    correct = 0;
    for (size_t i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    recall = (float)correct / max_elements;
    std::cout << "Recall of deserialized index: " << recall << "\n";

    delete[] data;
    delete alg_hnsw;
    delete[] param.table;
    std::cout << "End of program\n";
    return 0;
}
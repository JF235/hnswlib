#include "hnswlib/hnswlib.h"
#include "npy.hpp"

int main() {

    // Read NPY
    const std::string path {"table.npy"};
    npy::npy_data d = npy::read_npy<double>(path);
    std::vector<double> vecData = d.data;
    std::vector<unsigned long> shape = d.shape;
    bool fortran_order = d.fortran_order;
    
    // The struct that contains the table and the number of points.
    hnswlib::GrMatchParam param;
    param.num_points = shape[0];
    param.table = new float[param.num_points * param.num_points];
    for (int i = 0; i < param.num_points; i++) {
        for (int j = 0; j < param.num_points; j++) {
            param.table[i * param.num_points + j] = static_cast<float>(vecData[i * shape[1] + j]);
        }
    }
    hnswlib::GrMatchSpace space(&param);

    // Create a representation for my data.
    // As the distances are already computed, I can store just the indices.
    size_t *data = new size_t[param.num_points];
    for (int i = 0; i < param.num_points; i++) {
        data[i] = i;
    } 


    // Build the index
    int max_elements = param.num_points;
    int M = 3;
    int ef_construction = 3;
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Add data
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i, i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
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
    for (int i = 0; i < max_elements; i++) {
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
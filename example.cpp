#include "hnswlib/hnswlib.h"
#include "npy.hpp"

int main() {

    // Load data
    // std::mt19937 rng;
    // rng.seed(47);
    // std::uniform_real_distribution<> distrib_real;
    // float* data = new float[dim * max_elements];
    // for (int i = 0; i < dim * max_elements; i++) {
    //     data[i] = distrib_real(rng);
    // }

    // Read NPY
    const std::string path {"random_data.npy"};
    npy::npy_data d = npy::read_npy<double>(path);
    std::vector<double> vecData = d.data;
    std::vector<unsigned long> shape = d.shape;
    bool fortran_order = d.fortran_order;

    // Adapt data to float
    int max_elements = shape[0];
    int dim = shape[1];
    float* data = new float[max_elements * dim];
    for (int i = 0; i < max_elements; i++) {
        for (int j = 0; j < dim; j++) {
            data[i * dim + j] = static_cast<float>(vecData[i * dim + j]);
        }
    }


    // Initing index
    int M = 3;
    int ef_construction = 3; 
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
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
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    recall = (float)correct / max_elements;
    std::cout << "Recall of deserialized index: " << recall << "\n";

    delete[] data;
    delete alg_hnsw;
    return 0;
}
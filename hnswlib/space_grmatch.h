#pragma once
#include "hnswlib.h"

namespace hnswlib {

struct GrMatchParam {
    size_t num_points;
    float *table;
};

static float
GrMatch(const void *pVect1, const void *pVect2, const void *dist_func_param) {

    // Compute the griaule score between two points.


    // The dist_func_param is a pointer to a struct that contains the table and the number of points.


    // The table is a 2D array of size (num_points, num_points) where the entry
    // table[i][j] is the griaule match score between points i and j.
    
    // The values i and j are given by the pVect1 and pVect2 pointers.
    // The table is assumed to be in row-major order.

    size_t i = *((size_t *) pVect1);
    size_t j = *((size_t *) pVect2);

    GrMatchParam *param = (GrMatchParam *) dist_func_param;
    float *table = param->table;
    size_t num_points = param->num_points;

    if (i < 0 || i >= num_points || j < 0 || j >= num_points) {
        // Generate an error if the indices are out of bounds.
        throw std::out_of_range("Index out of bounds");
    }

    // Compute the griaule match score.
    float score = table[i * num_points + j];
    return score;
}

static float
GrMatchDistance(const void *pVect1, const void *pVect2, const void *dist_func_param) {
    // Turn the griaule match score into a distance score.
    // return 1.0f - GrMatch(pVect1, pVect2, dist_func_param);
    return 1.0f / (1.0f + GrMatch(pVect1, pVect2, dist_func_param));
}

class GrMatchSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    GrMatchParam *dist_func_param_;

 public:
    GrMatchSpace(GrMatchParam *param) {
        fstdistfunc_ = GrMatchDistance;
        data_size_ = sizeof(size_t);
        dist_func_param_ = param;
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return dist_func_param_;
    }

~GrMatchSpace() {}
};

}  // namespace hnswlib

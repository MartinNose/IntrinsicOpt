#pragma once
#include "iostream"
#include "trace.h"
#include "Eigen/Core"
#include "omp.h"
#include <vector>

using namespace std;
using namespace Eigen;



void point_sample_tet(Vector3d v0, Vector3d v1, Vector3d v2, Vector3d v3, RowVector4d &bc);

template<typename Particles>
void point_sample(MatrixXd &V, MatrixXi &T, Particles & P, double l) {
    cout << "V rows: " << V.rows() << " cols: " << V.cols() << endl;
    cout << "T rows: " << T.rows() << " cols: " << T.cols() << endl;

    double total_volume = 0;
    std::vector<double> volume(T.rows());

    #pragma omp parallel for reduction(+:total_volume)
    for (int i = 0; i < T.rows(); i++) {
        auto &&cur = T.row(i);

        Vector3d v0 = V.row(cur[0]);
        Vector3d v1 = V.row(cur[1]);
        Vector3d v2 = V.row(cur[2]);
        Vector3d v3 = V.row(cur[3]);

        volume[i] = igl::volume_single(v0, v1, v2, v3);
        total_volume += volume[i];
    }
    // Total number of points
    int n = ceil(total_volume / pow(l, 3));
    cout << "total volume: " << total_volume << " sampling " << n << " particles." << endl;

    
    for (int i = 0; i < T.rows(); i++) {
        // get num point
        int num = ceil((volume[i] / total_volume) * n);

        vector<RowVector4d> bcs(num);
        #pragma omp parallel for
        for (int j = 0; j < num; j++) {
            auto &&cur = T.row(i);
            Vector3d v0 = V.row(T(i,0));
            Vector3d v1 = V.row(T(i,1));
            Vector3d v2 = V.row(T(i,2));
            Vector3d v3 = V.row(T(i,3));

            RowVector4d bc;
            point_sample_tet(v0, v1, v2, v3, bc);

            if (abs(bc[0] + bc[1] + bc[2] + bc[3] - 1) > 0.000000001) {
                std::cout << i << "th cell with bc: " << bc << std::endl;
            }
            bcs[j] = bc; 
        }

        for (int j = 0; j < num; j++) {
            
            P.emplace_back(i, bcs[j]);
        }

        cout << "[" << i + 1 << "/" << T.rows() << "] cells sampled\n";
    }
    cout << "sampling done" << endl;
}

void point_sample_tet(Vector3d v0, Vector3d v1, Vector3d v2, Vector3d v3, RowVector4d &bc) {
    
    RowVector3d a, b, c;
    a = v1 - v0;
    b = v2 - v0;
    c = v3 - v0;

    double s = (double)rand() / (double)RAND_MAX;
    double t = (double)rand() / (double)RAND_MAX;
    double u = (double)rand() / (double)RAND_MAX;

    if (s + t > 1) {
        s = 1 - s;
        t = 1 - t;
    }

    double s_ = s;
    double t_ = t;
    double u_ = u;

    if (s_ + t_ + u_ > 1) {
        if (t_ + u_ > 1) {
            t = 1 - u_;
            u = 1 - s_ - t_;
        } else {
            s = 1 - t_ - u_;
            u = s_ + t_ + u_ - 1;
        }
    }

    RowVector3d p = s * a + t * b + u * c + v0.transpose();

    igl::barycentric_coordinates(p, v0.transpose(), v1.transpose(), v2.transpose(), v3.transpose(), bc);
}
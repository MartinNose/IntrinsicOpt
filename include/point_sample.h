#pragma once
#include "iostream"
#include "MeshTrace/trace.h"
#include "Eigen/Core"
#include "omp.h"
#include <vector>
#include <map>
#include <random>

using namespace std;
using namespace Eigen;

void point_sample_tet(Vector3d v0, Vector3d v1, Vector3d v2, Vector3d v3, RowVector4d &bc);

template<typename Particles>
void point_sample(const MatrixXd &V, const MatrixXi &T, const MatrixXi &TF, Particles &P, double l,  map<vector<int>, int> out_face_map) {
    cout << "V rows: " << V.rows() << " cols: " << V.cols() << endl;
    cout << "T rows: " << T.rows() << " cols: " << T.cols() << endl;

    vector<int> num_in_tet(T.rows());
    map<int, bool> surface;
    for (int i = 0; i < TF.rows(); i++) {
        vector<int> key {TF.row(i)[0], TF.row(i)[1], TF.row(i)[2]};
        sort(key.begin(), key.end());
        for (int j = 0; j < 3; j++) {
            int vi = TF.row(i)[j];
            if (surface.find(vi) != surface.end()) continue;
            surface[vi] = true;
            RowVector3d bc = Vector3d::Zero();
            bc[j] = 1.0;
            P.emplace_back(i, bc, MESHTRACE::POINT);

            num_in_tet[out_face_map[key]]++;
        }
    }

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
    int total = ceil(total_volume / pow(l, 3) * 8);
    int n = total - P.size(); // to add
    cout << "Boundary points: " << P.size() << endl;
    cout << "total volume: " << total_volume << " sampling " << n << " particles." << endl;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> distribution(0, 1);
    std::vector<double> possible(volume.size());
    for (int i = 0; i < volume.size(); i++) {
        possible[i] = total * volume[i] / total_volume - num_in_tet[i];
    }
    std::discrete_distribution<size_t> dist6(possible.begin(),
                                            possible.end());

    for (int i = 0; i < n; i++) {
        // get num point
        // int num = ceil((volume[i] / total_volume) * n);
        int tet;
        RowVector4d bc;
        bool on_boundary;
        do {
            tet = dist6(rng);
            auto &&cur = T.row(tet);
            Vector3d v[4];
            v[0] = V.row(T(tet,0));
            v[1] = V.row(T(tet,1));
            v[2] = V.row(T(tet,2));
            v[3] = V.row(T(tet,3));

            point_sample_tet(v[0], v[1], v[2], v[3], bc);

            on_boundary = false;
            for (int j = 0; j < 4; j++) {
                vector<Vector3d> f;
                vector<int> key;
                for (int k = 0; k < 4; k++) {
                    if (j == k) continue;
                    f.push_back(v[k]);
                    key.push_back(T(tet, k));
                }
                sort(key.begin(), key.end());
                double area = 0.5 * (f[1] - f[0]).cross(f[2] - f[0]).norm();
                double d = 3 * bc[j] * volume[tet] / area;

                if (out_face_map.find(key) != out_face_map.end() && d < 0.5 * l) {
                    on_boundary = true;
                    break;
                }
            }
        } while(on_boundary);
        if (abs(bc[0] + bc[1] + bc[2] + bc[3] - 1) > 0.000000001) {
            std::cout << tet << "th cell with bc: " << bc << std::endl;
        }
        P.emplace_back(tet, bc, MESHTRACE::FREE);
        if (i % 1000 == 0) cout << "[" << i << "/" << n << "] inner points sampled." << endl; 
    }
    cout << "sampling done. " << P.size() << " particles sampled." << endl;
    // TODO add sampling on surface;
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
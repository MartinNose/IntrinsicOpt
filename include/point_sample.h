#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
#pragma once
#include "iostream"
#include "MeshTrace/trace.h"
#include "MeshTrace/trace_manager.h"
#include "igl/face_areas.h"
#include "Eigen/Core"
#include <utility>
#include "omp.h"
#include <vector>
#include <map>
#include <random>
#include <algorithm>

using namespace std;
using namespace Eigen;
using namespace MESHTRACE;

void point_sample_tet(Vector3d v0, Vector3d v1, Vector3d v2, Vector3d v3, RowVector4d &bc);
void point_sample_tri(Vector3d v0, Vector3d v1, Vector3d v2, RowVector3d &bc);

template<typename Particles>
void point_sample_init(const MatrixXd &V, const MatrixXi &T, const MatrixXi &TF, Particles &P, double l, map<vector<int>, pair<int, int>> &out_face_map, MESHTRACE::MeshTraceManager<double> &meshtrace) {
    cout << "V rows: " << V.rows() << " cols: " << V.cols() << endl;
    cout << "T rows: " << T.rows() << " cols: " << T.cols() << endl;

    vector<int> num_in_tri(TF.rows(), 0);
    for (int i = 0; i < meshtrace.surface_point.size(); i++) {
        if (!meshtrace.surface_point[i]) continue;
        if (meshtrace.surface_point_adj_sharp_edges.find(i) 
            == meshtrace.surface_point_adj_sharp_edges.end())
                continue;
        if (meshtrace.surface_point_adj_sharp_edges[i].size() == 0) { // face
            int face_i = meshtrace.surface_point_adj_faces[i][0];
            RowVector3d bc;
            for (int j = 0; j < 3; j++) {
                bc[j] = (i == TF.row(face_i)[j]) ? 1. : 0.;
            }
            ParticleD p(face_i, bc, FACE);
            P.push_back(p);
            num_in_tri[face_i]++;
        } else if (meshtrace.surface_point_adj_sharp_edges[i].size() == 2) { // edge
            int ej = *(meshtrace.surface_point_adj_sharp_edges[i].begin());
            ParticleD p;
            p.cell_id = i;
            p.bc.resize(1, 4);
            p.bc[0] = 0.5;
            p.bc[1] = 0.5;
            p.bc[2] = (double)i;
            p.bc[3] = (double)ej;
            p.flag = EDGE;
            P.push_back(p);
            vector<int> edge = {min(i, ej), max(i, ej)};
            num_in_tri[get<0>(meshtrace.edge_tri_map[edge])]++;
        } else { // p
            ParticleD p;
            p.cell_id = i;
            p.bc = V.row(i);
            p.flag = POINT;
            P.push_back(p);
            int tri_i = meshtrace.surface_point_adj_faces[i][0];
            num_in_tri[tri_i]++;
        }
        
    }

    double total_area = 0;
    std::vector<double> area(TF.rows());

    #pragma omp parallel for reduction(+:total_area)
    for (int i = 0; i < TF.rows(); i++) {
        auto &&cur = TF.row(i);

        Vector3d v0 = V.row(cur[0]);
        Vector3d v1 = V.row(cur[1]);
        Vector3d v2 = V.row(cur[2]);

        area[i] = 0.5 * abs((v1-v0).cross(v2-v0).norm());
        total_area += area[i];
    }
    // Total number of points
    cout << "lattice: " << l << endl;
    int total = ceil(total_area / pow(l, 2) * 4);
    int n = max(total - (int)P.size(), 1); // to add
    cout << "Boundary points: " << P.size() << endl;
    cout << "total area: " << total_area << " sampling " << n << " particles." << endl;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> distribution(0, 1);
    std::vector<double> possible(area.size());
    for (int i = 0; i < area.size(); i++) {
        possible[i] = total * (area[i] / total_area) - num_in_tri[i];
    }
    std::discrete_distribution<size_t> dist6(possible.begin(),
                                            possible.end());

    vector<Vector3d> BCC = {
            Vector3d(l, 0, 0), Vector3d(-l, 0, 0), Vector3d(0, l, 0),
            Vector3d(0, -l, 0), Vector3d(0, 0, l), Vector3d(0, 0, -l)
    };
    for (int i = 0; i < n; i++) {
        int tri;
        RowVector3d bc;
        MESHTRACE::ParticleD p;
        p.flag = MESHTRACE::FACE;
        tri = dist6(rng);
        p.cell_id = tri;
        Vector3d v[4];
        v[0] = V.row(TF(tri,0));
        v[1] = V.row(TF(tri,1));
        v[2] = V.row(TF(tri,2));

        point_sample_tri(v[0], v[1], v[2], bc);
        p.bc = bc;

        if (abs(bc[0] + bc[1] + bc[2] - 1) > 0.000000001) {
            std::cout << tri << "th cell with bc: " << bc << std::endl;
        }
        P.push_back(p);
        if (i % 10000 == 0) cout << "[" << i << "/" << n << "] surface points sampled." << endl; 
    }
    cout << "sampling done. " << P.size() << " particles sampled." << endl;
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

void point_sample_tri(Vector3d v0, Vector3d v1, Vector3d v2, RowVector3d &bc) {
    RowVector3d a, b;
    a = v1 - v0;
    b = v2 - v0;

    double u = (double)rand() / (double)RAND_MAX;
    double t = (double)rand() / (double)RAND_MAX;

    if (u + t > 1) {
        u = 1 - u;
        t = 1 - t;
    }


    RowVector3d p = u * a + t * b + v0.transpose();

    igl::barycentric_coordinates(p, v0.transpose(), v1.transpose(), v2.transpose(), bc);
}
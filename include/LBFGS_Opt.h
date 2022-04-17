#pragma once
#include <iostream>
#include "Eigen/Core"
#include <vector>
#include "igl/AABB.h"
#include "kdtree.hpp"
#include "barycentric_to_cartesian.h"

using namespace Eigen;
using namespace std;
using namespace NNSearch;


void LBFGS_optimization(double l, 
        const MatrixXd &V, 
        const MatrixXi &T, 
        const MatrixX3d FF0, 
        const MatrixX3d FF1, 
        const MatrixX3d FF2, 
        vector<Particle<>> &P) {
    MatrixXd P_in_Cartesian(P.size(), 3);
    barycentric_to_cartesian(V, T, P, P_in_Cartesian);

    // Declare and Build AABB Tree    
    double sigma = 0.3 * l;
    int iteration_cnt = 0;

    while(iteration_cnt++ < 100) {
        auto func = [&] (const VectorXd &x, VectorXd &grad) {
            int N = x.size() / 3;
            MatrixXd points_mat(N ,3);
            for (int i = 0; i < N; i++) {
                points_mat.row(i) << x[i * 3 + 0], x[i * 3 + 1], x[i * 3 + 2];
            }
            
            const int leaf_size = 5;
            KDTree kdtree(points_mat, leaf_size);            

            double EN = 0;

            auto g_exp = [=](Vector3d pi) {
                double norm = pi.norm();
                return exp(-(norm * norm) / (2 * sigma * sigma));
            };

            for (int i = 0; i < N; i++) {
                Vector3d p = points_mat.row(i);
                // Update kd tree
                vector<int> pts_idx;
                vector<double> pts_dist;
                kdtree.radiusSearch(p, l, pts_idx, pts_dist);
                
                double EI = 0;
                Vector3d FI = Vector3d::Zero();
                for (int j = 0; j < pts_idx.size(); j++) {
                    if (pts_idx[j] == i) continue;
                    Vector3d pj = points_mat.row(pts_idx[j]);
                    
                    EI += compute_energy(p, pj);
                    FI += compute_grad(p, pj);
                }
                EN += EI;
                Vector3d gradient = FI; // TODO update FI depending on P[i].bc.cols()
                grad[i*3 + 0] = FI[0];
                grad[i*3 + 1] = FI[1];
                grad[i*3 + 2] = FI[2];
            }
            return EN;
        };

        VectorXd x = VectorXd::Zero(P.size() * 3);
        for (int i = 0; i < P_in_Cartesian.rows(); i++) {
            x[i * 3 + 0] = P_in_Cartesian(i, 0);
            x[i * 3 + 1] = P_in_Cartesian(i, 1);
            x[i * 3 + 2] = P_in_Cartesian(i, 2);
        }
        double fx;
        // LBFGS(fun, x, fx);
        // Update P using trace
    }
}
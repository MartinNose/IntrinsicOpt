#pragma once
#include <iostream>
#include "Eigen/Core"
#include <vector>
#include "igl/AABB.h"
#include "kdtree.hpp"
#include "barycentric_to_cartesian.h"
#include "LBFGSpp/LBFGS.h"
#include "MeshTrace/trace_manager.h"

using namespace Eigen;
using namespace std;
using namespace NNSearch;
using namespace LBFGSpp;
using namespace MESHTRACE;

void LBFGS_optimization(double l, 
        vector<Particle<>> &P,
        MeshTraceManager<double>& meshtrace) {

    LBFGSParam<double> param;
    param.epsilon = 1e-5;
    param.max_iterations = 100;
    param.max_linesearch = 100;
    // param.max_linesearch = 100;
    LBFGSSolver<double> solver(param);

    vector<Vector3d> BCC = {
        Vector3d(l, 0, 0), Vector3d(-l, 0, 0), Vector3d(0, l, 0), 
        Vector3d(0, -l, 0), Vector3d(0, 0, l), Vector3d(0, 0, -l)
    };
    
    MatrixXd P_in_Cartesian(P.size(), 3);

    // Iteration Factors
    double sigma = 0.3 * l;
    int iteration_cnt = 0;
    int max_iteration = 100;

    while(iteration_cnt++ < 10) {
        cout << "----------------------------------\n" << iteration_cnt << "/ 100 iteration" << endl;
        meshtrace.to_cartesian(P, P_in_Cartesian);
        auto func = [&] (const VectorXd &x, VectorXd &grad) {
            static int call_cnt = 0;
            cout << ++call_cnt << " th called" << endl;;
            int N = x.size() / 3;
            MatrixXd points_mat(N ,3);
            for (int i = 0; i < N; i++) {
                points_mat.row(i) << x[i * 3 + 0], x[i * 3 + 1], x[i * 3 + 2];
            }
            
            const int leaf_size = 5;
            KDTree kdtree(points_mat, leaf_size);            

            double EN = 0;

            auto g_exp = [=](Vector3d v) {
                double norm = v.norm();
                return exp(-(norm * norm) / (2 * sigma * sigma));
            };

            #pragma omp parallel for reduction(+ : EN)
            for (int i = 0; i < N; i++) {
                Particle<> particle = P[i];
                Vector3d pi = points_mat.row(i);
                // Update kd tree
                vector<int> pts_idx;
                vector<double> pts_dist;
                kdtree.radiusSearch(pi, 1.5 * l, pts_idx, pts_dist);
                
                double EI = 0;
                Vector3d FI = Vector3d::Zero();
                for (int j = 0; j < pts_idx.size(); j++) {
                    if (pts_idx[j] == i) continue;

                    Vector3d pj = points_mat.row(pts_idx[j]);

                    EI += g_exp(pi - pj);
                    FI += (pi - pj) / (sigma * sigma) * g_exp(pi - pj);
                    // for (Vector3d h : BCC) {
                    //     EI += - 1.0 / 6.0 * g_exp(pi - pj - (pi + h));
                    //     FI += - 1.0 / 6.0 * (pi - pj - (pi + h)) / (sigma * sigma) * g_exp(pi - pj - (pi + h)); 
                    // }
                }
                if (particle.flag != POINT) {
                    EN += EI;
                }
                Vector3d gradient = FI;

                meshtrace.project(particle, gradient);

                grad[i*3 + 0] = -gradient[0];
                grad[i*3 + 1] = -gradient[1];
                grad[i*3 + 2] = -gradient[2];
            }
            cout << "E = " << EN << " gnorm: " << grad.norm() << endl;
            return EN;
        };

        VectorXd x = VectorXd::Zero(P.size() * 3);
        for (int i = 0; i < P_in_Cartesian.rows(); i++) {
            x[i * 3 + 0] = P_in_Cartesian(i, 0);
            x[i * 3 + 1] = P_in_Cartesian(i, 1);
            x[i * 3 + 2] = P_in_Cartesian(i, 2);
        }
        double fx;
        cout << "LBFGS Begin" << endl;
        int niter = solver.minimize(func, x, fx);
        cout << "LBFGS Done" << endl;
        // cout << "x = \n" << x.transpose() << endl;
        cout << "f(x) = " << fx << endl;

        for (auto [fx, gnorm] : solver.energy_history) {
            std::cout << fx << ", " << gnorm << std::endl;
        }

        for (int i = 0; i < P.size(); i++) {
            Vector3d displacement;
            displacement[0] = x[i * 3 + 0] - P_in_Cartesian(i, 0);      
            displacement[1] = x[i * 3 + 1] - P_in_Cartesian(i, 1);      
            displacement[2] = x[i * 3 + 2] - P_in_Cartesian(i, 2);      
            meshtrace.tracing(P[i], displacement);
        }
    }
}
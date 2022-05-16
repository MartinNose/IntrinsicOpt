#pragma once
#include <iostream>
#include "Eigen/Core"
#include <vector>
#include "igl/AABB.h"
#include "barycentric_to_cartesian.h"
#include "KDTreeVectorOfVectorsAdaptor.h"
#include "LBFGSpp/LBFGS.h"
#include "MeshTrace/trace_manager.h"
#include <cmath>

using namespace Eigen;
using namespace std;
using namespace LBFGSpp;
using namespace MESHTRACE;

using kdtree_t =
KDTreeVectorOfVectorsAdaptor<std::vector<Eigen::Vector3d>, double>;

void LBFGS_optimization(double l,
        vector<Particle<>> &PV,
        MeshTraceManager<double>& meshtrace,
        MatrixXd *debug_test = nullptr) {
    // LBFGS Routine
    LBFGSParam<double> param;
    param.epsilon = 1e-15;
    param.max_iterations = 10;
    param.max_linesearch = 100;
    LBFGSSolver<double> solver(param);

    vector<Vector3d> BCC = {
            Vector3d(l, 0, 0), Vector3d(-l, 0, 0), Vector3d(0, l, 0),
            Vector3d(0, -l, 0), Vector3d(0, 0, l), Vector3d(0, 0, -l)
    };

    // Iteration Factors
    double sigma = 0.3 * l;

    auto func = [&] (const VectorXd &x, VectorXd &grad) {
        int N = x.size() / 3;
        vector<Vector3d> points_vec(N);
        for (int i = 0; i < N; i++) {
            points_vec[i][0] = x[i * 3 + 0];
            points_vec[i][1] = x[i * 3 + 1];
            points_vec[i][2] = x[i * 3 + 2];
        }

        kdtree_t kdtree(3, points_vec, 25);

        nanoflann::SearchParams params;
        params.sorted = false;

        auto g_exp = [=](const Vector3d &v) {
            double norm = v.norm();
            return std::exp(-(norm * norm) / (2 * sigma * sigma));
        };

        double EN = 0;
//            #pragma omp parallel for reduction(+ : EN) // NOLINT(openmp-use-default-none)
        for (int i = 0; i < N; i++) {
            Particle<> particle = PV[i];
            Vector3d pi = points_vec[i];

            std::vector<std::pair<size_t, double>> ret_matches;

            double n_r = 1.5 * l;
            kdtree.index->radiusSearch(&pi[0], n_r * n_r, ret_matches, params);

            MatrixXd test(ret_matches.size(), 3);

            Vector3d fia = Vector3d::Zero();
            for (int j = 0; j < ret_matches.size(); j++) {
                if (ret_matches[j].first == i) continue;

                Vector3d pj = points_vec[ret_matches[j].first];
                Vector3d vij = pi - pj;
                double g_exp_1 = g_exp(vij);
                double ENN = 0;
                Vector3d fij = Vector3d::Zero();
                for (const Vector3d &h: BCC) {
                    double g_exp_0 = g_exp(vij - h);
                    ENN -= g_exp_0;
                    fij -= (vij - h) / (sigma * sigma) * g_exp_0;
                }
                fij /= 6.;
                ENN /= 6.;
                Vector3d f = fij + vij / (sigma * sigma) * g_exp_1;
                fia += f;
                if (particle.flag != POINT) {
                    EN += ENN + g_exp_1;
                }
            }
            meshtrace.project(particle, fia);

            grad[i * 3 + 0] = -fia[0];
            grad[i * 3 + 1] = -fia[1];
            grad[i * 3 + 2] = -fia[2];
        }
//            cout << "E = " << EN << " gnorm: " << grad.norm() << " ";
//            cout << x.transpose() << endl;
        return EN;
    };

    MatrixXd P_in_Cartesian;
    meshtrace.to_cartesian(PV, P_in_Cartesian);

    VectorXd x = VectorXd::Zero(PV.size() * 3);
    for (int i = 0; i < P_in_Cartesian.rows(); i++) {
        x[i * 3 + 0] = P_in_Cartesian(i, 0);
        x[i * 3 + 1] = P_in_Cartesian(i, 1);
        x[i * 3 + 2] = P_in_Cartesian(i, 2);
    }
    double fx;
    cout << "LBFGS Begin" << endl;
    int niter = solver.minimize(func, x, fx);
    cout << "LBFGS Done" << endl;
    cout << "f(x) = " << fx << endl;
    cout << "x = " << x.transpose() << endl;
    if (isnan(x[0])) {
        throw runtime_error("LBFGS Failed due to nan solution");
    }

    for (auto [fx_h, gnorm_h] : solver.energy_history) {
        std::cout << fx_h << ", " << gnorm_h << std::endl;
    }

    if (debug_test) (*debug_test).resize(PV.size(), 3);
    for (int i = 0; i < PV.size(); i++) {
        Vector3d displacement;
        displacement[0] = x[i * 3 + 0] - P_in_Cartesian(i, 0);
        displacement[1] = x[i * 3 + 1] - P_in_Cartesian(i, 1);
        displacement[2] = x[i * 3 + 2] - P_in_Cartesian(i, 2);
        cout << "start tracing. cell id: " << PV[i].cell_id << " bc: " <<PV[i].bc << " coord: " << P_in_Cartesian.row(i) << endl;
        cout << "d: " << displacement.transpose() << endl;
        meshtrace.tracing(PV[i], displacement);

        if (debug_test) {
            (*debug_test).row(i)[0] = x[i * 3 + 0];
            (*debug_test).row(i)[1] = x[i * 3 + 1];
            (*debug_test).row(i)[2] = x[i * 3 + 2];
        }
    }
}

void LBFGS_init(double l,
                        vector<Particle<>> &PV,
                        MeshTraceManager<double>& meshtrace,
                        MatrixXd *debug_test = nullptr) {
    // LBFGS Routine
    LBFGSParam<double> param;
    param.epsilon = 1e-15;
    param.max_iterations = 10;
    param.max_linesearch = 100;
    LBFGSSolver<double> solver(param);

    vector<Vector3d> BCC = {
            Vector3d(l, 0, 0), Vector3d(-l, 0, 0), Vector3d(0, l, 0),
            Vector3d(0, -l, 0), Vector3d(0, 0, l), Vector3d(0, 0, -l)
    };

    // Iteration Factors
    double sigma = 0.3 * l;

    auto func = [&] (const VectorXd &x, VectorXd &grad) {
        int N = x.size() / 3;
        vector<Vector3d> points_vec(N);
        for (int i = 0; i < N; i++) {
            points_vec[i][0] = x[i * 3 + 0];
            points_vec[i][1] = x[i * 3 + 1];
            points_vec[i][2] = x[i * 3 + 2];
        }

        kdtree_t kdtree(3, points_vec, 25);

        nanoflann::SearchParams params;
        params.sorted = false;

        auto g_exp = [=](const Vector3d &v) {
            double norm = v.norm();
            return std::exp(-(norm * norm) / (2 * sigma * sigma));
        };

        double EN = 0;
//            #pragma omp parallel for reduction(+ : EN) // NOLINT(openmp-use-default-none)
        for (int i = 0; i < N; i++) {
            Particle<> particle = PV[i];
            Vector3d pi = points_vec[i];

            std::vector<std::pair<size_t, double>> ret_matches;

            double n_r = 1.5 * l;
            kdtree.index->radiusSearch(&pi[0], n_r * n_r, ret_matches, params);

            MatrixXd test(ret_matches.size(), 3);

            Vector3d fia = Vector3d::Zero();
            for (int j = 0; j < ret_matches.size(); j++) {
                if (ret_matches[j].first == i) continue;

                Vector3d pj = points_vec[ret_matches[j].first];
                Vector3d vij = pi - pj;
                double g_exp_1 = g_exp(vij);
                Vector3d f = vij / (sigma * sigma) * g_exp_1;
                fia += f;
                if (particle.flag != POINT) {
                    EN += g_exp_1;
                }
            }
            meshtrace.project(particle, fia);

            grad[i * 3 + 0] = -fia[0];
            grad[i * 3 + 1] = -fia[1];
            grad[i * 3 + 2] = -fia[2];
        }
//            cout << "E = " << EN << " gnorm: " << grad.norm() << " ";
//            cout << x.transpose() << endl;
        return EN;
    };

    MatrixXd P_in_Cartesian;
    meshtrace.to_cartesian(PV, P_in_Cartesian);

    VectorXd x = VectorXd::Zero(PV.size() * 3);
    for (int i = 0; i < P_in_Cartesian.rows(); i++) {
        x[i * 3 + 0] = P_in_Cartesian(i, 0);
        x[i * 3 + 1] = P_in_Cartesian(i, 1);
        x[i * 3 + 2] = P_in_Cartesian(i, 2);
    }
    double fx;
    cout << "LBFGS Begin" << endl;
    int niter = solver.minimize(func, x, fx);
    cout << "LBFGS Done" << endl;
    cout << "f(x) = " << fx << endl;
    cout << "x = " << x.transpose() << endl;
    if (isnan(x[0])) {
        throw runtime_error("LBFGS Failed due to nan solution");
    }

    for (auto [fx_h, gnorm_h] : solver.energy_history) {
        std::cout << fx_h << ", " << gnorm_h << std::endl;
    }

    if (debug_test) (*debug_test).resize(PV.size(), 3);
    for (int i = 0; i < PV.size(); i++) {
        Vector3d displacement;
        displacement[0] = x[i * 3 + 0] - P_in_Cartesian(i, 0);
        displacement[1] = x[i * 3 + 1] - P_in_Cartesian(i, 1);
        displacement[2] = x[i * 3 + 2] - P_in_Cartesian(i, 2);
        cout << "start tracing. cell id: " << PV[i].cell_id << " bc: " <<PV[i].bc << " coord: " << P_in_Cartesian.row(i) << endl;
        cout << "d: " << displacement.transpose() << endl;
        meshtrace.tracing(PV[i], displacement);

        if (debug_test) {
            (*debug_test).row(i)[0] = x[i * 3 + 0];
            (*debug_test).row(i)[1] = x[i * 3 + 1];
            (*debug_test).row(i)[2] = x[i * 3 + 2];
        }
    }
}

#pragma once
#include <iostream>
#include "Eigen/Core"
#include <vector>
#include <ctime>
#include "igl/AABB.h"
#include "barycentric_to_cartesian.h"
#include "KDTreeVectorOfVectorsAdaptor.h"
#include "LBFGSpp/LBFGS.h"
#include "MeshTrace/trace_manager.h"
#include <cmath>
#include <eigen_binaryIO.h>

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
    param.epsilon = 1e-6;
    param.max_iterations = 10;
    // param.max_linesearch = 100;
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
    #pragma omp parallel for reduction(+ : EN) // NOLINT(openmp-use-default-none)
        for (int i = 0; i < N; i++) {
            Particle<> particle = PV[i];
            Vector3d pi = points_vec[i];
            if (particle.flag == POINT) {
                grad[i * 3 + 0] = 0.;
                grad[i * 3 + 1] = 0.;
                grad[i * 3 + 2] = 0.; 
                continue;
            }
            std::vector<std::pair<size_t, double>> ret_matches;

            double n_r = 1.5 * l;
            kdtree.index->radiusSearch(&pi[0], n_r * n_r, ret_matches, params);

            Vector3d fia = Vector3d::Zero();
            for (int j = 0; j < ret_matches.size(); j++) {
                if (ret_matches[j].first == i) continue;
                Vector3d n_0 = meshtrace.tri_normal.row(meshtrace.get_tri_id(particle));
                ParticleD par_j = PV[ret_matches[j].first];
                Vector3d n_1 = meshtrace.tri_normal.row(meshtrace.get_tri_id(par_j));
                if (n_0.dot(n_1) < 0) continue;

                Vector3d pj = points_vec[ret_matches[j].first];

                Matrix3d FF;
                meshtrace.get_mid_frame(particle, par_j, FF);

                Matrix3d B = FF.inverse();

                Vector3d vij = pi - pj;
                double g_exp_1 = g_exp(B * vij);

                double ENN = 0;
                Vector3d fij = Vector3d::Zero();

                int mark;
                double max = -1.;
                for (int i = 0; i < 3; i++) {
                    double cos = abs(n_0.dot(FF.col(i)));
                    if (cos > max) {
                        max = cos;
                        mark = i;
                    }
                }

                for (int i = 0; i < 6; i++) {
                    if (i / 2 == mark) continue;
                    double g_exp_0 = g_exp(B * vij - BCC[i]);
                    ENN -= g_exp_0;
                    fij -= (B * vij - BCC[i]) / (sigma * sigma) * g_exp_0;
                }
                fij *= 0.25;
                ENN /= 0.25;
                Vector3d f = FF * fij + vij / (sigma * sigma) * g_exp_1;
                fia += f;
                EN += ENN + g_exp_1;
            }
            if (particle.flag != FREE) meshtrace.project(particle, fia);
            
            grad[i * 3 + 0] = -fia[0];
            grad[i * 3 + 1] = -fia[1];
            grad[i * 3 + 2] = -fia[2];
        }
        std::cout << "E: " << EN << " gnorm: " << grad.norm() << std::endl;
        return EN;
    };

    MatrixXd P_in_Cartesian;
    meshtrace.to_cartesian(PV, P_in_Cartesian);

    VectorXd x = VectorXd::Zero(PV.size() * 3);

    #pragma omp parallel for // NOLINT(openmp-use-default-none)
    for (int i = 0; i < P_in_Cartesian.rows(); i++) {
        x[i * 3 + 0] = P_in_Cartesian(i, 0);
        x[i * 3 + 1] = P_in_Cartesian(i, 1);
        x[i * 3 + 2] = P_in_Cartesian(i, 2);
    }
    double fx;
    std::cout << "LBFGS Begin" << std::endl;
    int niter = solver.minimize(func, x, fx);
    std::cout << "LBFGS Done" << std::endl;
    std::cout << "f(x) = " << fx << std::endl;
    // cout << "x = " << x.transpose() << endl;
    if (isnan(x[0])) {
        throw runtime_error("LBFGS Failed due to nan solution");
    }

    for (auto [fx_h, gnorm_h] : solver.energy_history) {
        std::cout << fx_h << ", " << gnorm_h << std::endl;
    }
    write_binary("p_orig.dat", P_in_Cartesian);
    write_binary("x.dat", x);

    #pragma omp parallel for // NOLINT(openmp-use-default-none)
    for (int i = 0; i < PV.size(); i++) {
        Vector3d displacement;
        displacement[0] = x[i * 3 + 0] - P_in_Cartesian(i, 0);
        displacement[1] = x[i * 3 + 1] - P_in_Cartesian(i, 1);
        displacement[2] = x[i * 3 + 2] - P_in_Cartesian(i, 2);
//        cout << "start tracing. cell id: " << PV[i].cell_id << " bc: " <<PV[i].bc << " coord: " << P_in_Cartesian.row(i) << endl;
//        cout << "d: " << displacement.transpose() << endl;
        // MESHTRACE::FLAG flag = PV[i].flag;

        Vector3d target {x[i*3+0], x[i*3+1], x[i*3+2]};

        // if (PV[i].flag == FREE) {
        //     int tet = meshtrace.in_element(target);
        //     if (tet != -1) {
        //         RowVector4d bc;
        //         igl::barycentric_coordinates(target.transpose(), 
        //             meshtrace.V.row(meshtrace.TT.row(tet)[0]), meshtrace.V.row(meshtrace.TT.row(tet)[1]), 
        //             meshtrace.V.row(meshtrace.TT.row(tet)[2]), meshtrace.V.row(meshtrace.TT.row(tet)[3]), 
        //         bc);
        //         if (bc.minCoeff() > -BARYCENTRIC_BOUND) {
        //             ParticleD inserted(tet, bc, FREE);
        //             PV[i] = inserted;
        //             continue;
        //         } 
        //     }
        // } 
        
        meshtrace.tracing(PV[i], displacement);

        // if (flag == MESHTRACE::FREE && PV[i].flag == MESHTRACE::FREE) {
        //     Vector3d target {x[i*3+0], x[i*3+1], x[i*3+2]};
        //     int tet = meshtrace.in_element(target);
        //     if (tet != PV[i].cell_id) {
        //         cout << "expected: " << tet << " " << target.transpose() << endl;
        //         meshtrace.to_cartesian(PV[i], target);
        //         cout << "actually: " << PV[i].cell_id << " " << target.transpose() <<  endl;
        //     }
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
#pragma omp parallel for reduction(+ : EN) // NOLINT(openmp-use-default-none)
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
        meshtrace.tracing(PV[i], displacement);

        if (debug_test) {
            (*debug_test).row(i)[0] = x[i * 3 + 0];
            (*debug_test).row(i)[1] = x[i * 3 + 1];
            (*debug_test).row(i)[2] = x[i * 3 + 2];
        }
    }
}

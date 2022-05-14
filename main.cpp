#include <iostream>
#include <cstdlib>
#include <exception>
#include <igl/opengl/glfw/Viewer.h>
#include "Eigen/Core"
#include "eigen_binaryIO.h"
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/frame_field_deformer.h>
#include <igl/frame_to_cross_field.h>
#include <igl/jet.h>
#include <igl/local_basis.h>
#include <igl/readDMAT.h>
#include <igl/readOBJ.h>
#include <igl/readMSH.h>
#include <igl/rotate_vectors.h>
#include <igl/copyleft/comiso/nrosy.h>
#include <igl/copyleft/comiso/miq.h>
#include <igl/copyleft/comiso/frame_field.h>
#include "readVTK.h"
#include <igl/opengl/glfw/Viewer.h>
#include <igl/PI.h>
#include "MeshTrace/trace.h"
#include "MeshTrace/trace_manager.h"
#include "point_sample.h"
#include "barycentric_to_cartesian.h"
#include "omp.h"
#include "LBFGS_Opt.h"
#include "surface_mesh.h"
#include "trivial_case.h"
#include "KDTreeVectorOfVectorsAdaptor.h"

// Input frame field constraints

using namespace std;
using namespace Eigen;

using kdtree_t =
KDTreeVectorOfVectorsAdaptor<std::vector<Eigen::Vector3d>, double>;

#define datapath "/Users/liujunliang/Documents/Codes/IntrinsicOpt/dataset/"

Eigen::MatrixXd V;
Eigen::MatrixXi T;
Eigen::MatrixXi TF;
Eigen::MatrixXd trace_points;
Eigen::MatrixXd debug_point_a;
Eigen::MatrixXd debug_point_b;

Eigen::MatrixXd random_points;
Eigen::MatrixXd delete_insert_points;
Eigen::MatrixXd optimized_points;

Eigen::MatrixXd debug_point[9];

Eigen::MatrixXd diFix;
Eigen::MatrixXd diFace;
Eigen::MatrixXd diFree;
Eigen::MatrixXd ranFix;
Eigen::MatrixXd ranFace;
Eigen::MatrixXd ranFree;
Eigen::MatrixXd optFix;
Eigen::MatrixXd optFace;
Eigen::MatrixXd optFree;

MatrixXd B;
MatrixXd N;
double l;

void stepCallback(const ParticleD& target, double stepLen, double total) {
    static unsigned int cnt = 0;
    cout << "---------------------" << endl;
    cout << cnt++ << "th Particle Position: \nCell_id: " << target.cell_id << endl;
    cout << "BC: " << target.bc << endl;

    int col = target.bc.cols();
    Eigen::Matrix<double, Dynamic, 3> Cell(col, 3);
    for (int i = 0; i < col; i++) {
        Cell.row(i) = V.row(T.row(target.cell_id)(i));
    }

    Eigen::Vector3d endPoint = target.bc * Cell;

    cout << "new Start Point: " << endPoint.transpose() << endl;

    trace_points.conservativeResize(trace_points.rows() + 1, 3);
    trace_points.row(trace_points.rows() - 1) << endPoint.transpose();

    cout << "Current step length: " << stepLen << " Total traveled length: " << total << endl;
    debug_point_a.conservativeResize(debug_point_a.rows() + 6,3);
    debug_point_b.conservativeResize(debug_point_b.rows() + 6,3);

    debug_point_a.row(debug_point_a.rows() - 1) << Cell.row(0);
    debug_point_a.row(debug_point_a.rows() - 2) << Cell.row(0);
    debug_point_a.row(debug_point_a.rows() - 3) << Cell.row(0);
    debug_point_a.row(debug_point_a.rows() - 4) << Cell.row(1);
    debug_point_a.row(debug_point_a.rows() - 5) << Cell.row(1);
    debug_point_a.row(debug_point_a.rows() - 6) << Cell.row(2);

    debug_point_b.row(debug_point_a.rows() - 1) << Cell.row(1);
    debug_point_b.row(debug_point_a.rows() - 2) << Cell.row(2);
    debug_point_b.row(debug_point_a.rows() - 3) << Cell.row(3);
    debug_point_b.row(debug_point_a.rows() - 4) << Cell.row(2);
    debug_point_b.row(debug_point_a.rows() - 5) << Cell.row(3);
    debug_point_b.row(debug_point_a.rows() - 6) << Cell.row(3);

}

void callback(const ParticleD& target, double stepLen, double total) {
        stepCallback(target, stepLen, total);
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier) {
    using namespace std;
    using namespace Eigen;

    if (key >= '1' && key <= '9') {
        viewer.data().clear();
        viewer.data().set_mesh(V, TF);
        viewer.data().set_points((debug_point[key - '0']), RowVector3d(0, 0, 0.5));
    }
    return false;
    if (key == '1') {
        viewer.data().set_points(ranFix, RowVector3d(0, 0, 0.5));
    } else if (key == '2') {
        viewer.data().set_points(ranFace, RowVector3d(0, 0.5, 0));
    } else if (key == '3') {
        viewer.data().set_points(ranFree, RowVector3d(0.5, 0, 0));
    } else if (key == '4') {
        viewer.data().set_points(diFix, RowVector3d(0, 0, 0.5));
    } else if (key == '5') {
        viewer.data().set_points(diFace, RowVector3d(0, 0.5, 0));
    } else if (key == '6') {
        viewer.data().set_points(diFree, RowVector3d(0.5, 0, 0));
    } else if (key == '7') {
        viewer.data().set_points(optFix, RowVector3d(0, 0, 0.5));
    } else if (key == '8') {
        viewer.data().set_points(optFace, RowVector3d(0, 0.5, 0));
    } else if (key == '9') {
        viewer.data().set_points(optFree, RowVector3d(0.5, 0, 0));
    }    
    return false;
}

void log() {
    igl::opengl::glfw::Viewer viewer;
    viewer.callback_key_down = &key_down;
    key_down(viewer,'1',0);
    viewer.launch();
}

int main(int argc, char* argv[]) {
//    readVTK(datapath "l1-poly-dat/hex/kitty/orig.tet.vtk", V, T);
    // readVTK(datapath "trivial.vtk", V, T);
    create_trivial_case(V, T, 5, 0.1);

    MatrixXd FF0T = MatrixXd::Zero(T.rows(), 3);
    MatrixXd FF1T = MatrixXd::Zero(T.rows(), 3);
    MatrixXd FF2T = MatrixXd::Zero(T.rows(), 3);

    FF0T.col(0) = MatrixXd::Constant(T.rows(), 1, 1.0);
    FF1T.col(1) = MatrixXd::Constant(T.rows(), 1, 1.0);
    FF2T.col(2) = MatrixXd::Constant(T.rows(), 1, 1.0);

    auto[out_face_map, sharp_edge_map] = get_surface_mesh(V, T, TF);

    MatrixXd FF0F, FF1F;

    MatrixXd particle_dump;

    igl::barycenter(V, TF, B);

    igl::per_face_normals(V, TF, N);

    FF0F.resize(TF.rows(), 3);
    FF1F.resize(TF.rows(), 3);

    RowVector3d X{1.0, 0, 0};
    RowVector3d Z{0, 0, 1.0};

    for (int i = 0; i < TF.rows(); i++) {
        RowVector3d n = N.row(i);
        RowVector3d f = n.cross(Z);
        if (f.norm() == 0) {
            f = n.cross(X);
        }
        f.normalize();
        FF0F.row(i) = f;
        FF1F.row(i) = f.cross(n);
    }

    MeshTraceManager<double> meshtrace(V, T, TF, FF0T, FF1T, FF2T, FF0F, FF1F, out_face_map);


    vector<ParticleD> PV;
    for (int i = 0; i < T.rows(); i++) {
        if (i % 5 != 1) continue;
        if (i/125%5 != 2) continue;
        ParticleD p;
        p.cell_id = i;
        p.bc.resize(1, 4);
        p.bc << 0.25, 0.25, 0.25, 0.25;
        p.flag = MESHTRACE::FREE;
        PV.push_back(p);
    }

    // l = igl::avg_edge_length(V, T);

//    l = 0.15000000001;
    l = 0.08;
//    cin >> l;

    MatrixXd points_mat;
    meshtrace.to_cartesian(PV, points_mat);

    vector<Vector3d> points_vec;
    for (int i = 0; i < points_mat.rows(); i++) {
        Vector3d t = points_mat.row(i);
        points_vec.push_back(t);
    }

   // LBFGS Routine
    LBFGSParam<double> param;
    param.epsilon = 1e-6;
    param.max_iterations = 10;
    param.max_linesearch = 100;
    LBFGSSolver<double> solver(param);

    vector<Vector3d> BCC = {
            Vector3d(l, 0, 0), Vector3d(-l, 0, 0), Vector3d(0, l, 0),
            Vector3d(0, -l, 0), Vector3d(0, 0, l), Vector3d(0, 0, -l)
    };



    // Iteration Factors
    double sigma = 0.15 * l;
    int iteration_cnt = 0;
    int max_iteration = 1;


    while(iteration_cnt++ < max_iteration) {
        cout << "----------------------------------\n" << iteration_cnt << "th LBFGS iteration" << endl;
        MatrixXd P_in_Cartesian;
        meshtrace.to_cartesian(PV, P_in_Cartesian);
        debug_point[3] = P_in_Cartesian;
        auto func = [&] (const VectorXd &x, VectorXd &grad) {
//            static int call_cnt = 0;
//            cout << ++call_cnt << " th called" << endl;;
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

            auto g_exp = [=](const Vector3d& v) {
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

                grad[i*3 + 0] = -fia[0];
                grad[i*3 + 1] = -fia[1];
                grad[i*3 + 2] = -fia[2];
            }
//            cout << "E = " << EN << " gnorm: " << grad.norm() << " ";
//            cout << x.transpose() << endl;
            return EN;
        };

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
            throw runtime_error("LBFGS Failed");
        }

        for (auto [fx, gnorm] : solver.energy_history) {
            std::cout << fx << ", " << gnorm << std::endl;
        }

        MatrixXd test1;
        test1.resize(PV.size(), 3);
        for (int i = 0; i < PV.size(); i++) {
//            cout << "dealing with " << i << "/" << PV.size() << " particles" << endl;
            Vector3d displacement;
            test1.row(i)[0] = x[i * 3 + 0];
            test1.row(i)[1] = x[i * 3 + 1];
            test1.row(i)[2] = x[i * 3 + 2];

            displacement[0] = x[i * 3 + 0] - P_in_Cartesian(i, 0);
            displacement[1] = x[i * 3 + 1] - P_in_Cartesian(i, 1);
            displacement[2] = x[i * 3 + 2] - P_in_Cartesian(i, 2);
            cout << "start tracing. cell id: " << PV[i].cell_id << " bc: " <<PV[i].bc << " coord: " << P_in_Cartesian.row(i) << endl;
            cout << "d: " << displacement.transpose() << endl;
            meshtrace.tracing(PV[i], displacement);
        }
        cout << test1;
        debug_point[2] = test1;
        meshtrace.to_cartesian(PV, trace_points);
        debug_point[1] = trace_points;
    }



    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, TF);
    viewer.callback_key_down = &key_down;

    viewer.data().set_points(trace_points, RowVector3d{0, 0, 1});
    viewer.launch();

    return 0;
}
//
//int backup (){
//
//    bool use_cache = false;
//
//    if (!use_cache) {
//        bool pass = false;
//        while (!pass) {
//        try {
//            A.resize(0);
//            point_sample(V, T, TF, A, l, out_face_map);
//            meshtrace.to_cartesian(A, ranFix, ranFace, ranFree);
//            cout << "cartesian fix: " << ranFix.rows() << endl;
//            meshtrace.to_cartesian(A, random_points);
////            log(); return 0;
////            meshtrace.particle_insert_and_delete(A, 0.8 * l, l);
//            int iteration_cnt = 0;
//            while (iteration_cnt++ < 1) {
//                cout << "============ " << iteration_cnt << "th iteration==================" << endl;
//                bool flag = meshtrace.particle_insert_and_delete(A, 0.8 * l, l);
//                meshtrace.to_cartesian(A, delete_insert_points);
//                meshtrace.to_cartesian(A, diFix, diFace, diFree);
//                Eigen::write_binary("random.dat", random_points);
//                Eigen::write_binary("di.dat", delete_insert_points);
//
//                particle_dump.resize(A.size(), 6);
//                for (int i = 0; i < A.size(); i++) {
//                    particle_dump(i, 0) = A[i].flag;
//                    particle_dump(i, 1) = A[i].cell_id;
//                    if (A[i].flag == FREE) {
//                        particle_dump(i, 2) = A[i].bc[0];
//                        particle_dump(i, 3) = A[i].bc[1];
//                        particle_dump(i, 4) = A[i].bc[2];
//                        particle_dump(i, 5) = A[i].bc[3];
//                    } else {
//                        particle_dump(i, 2) = A[i].bc[0];
//                        particle_dump(i, 3) = A[i].bc[1];
//                        particle_dump(i, 4) = A[i].bc[2];
//                    }
//                }
//
//                Eigen::write_binary("Particle.dat", particle_dump);
//                if (flag) break;
//                LBFGS_optimization(l, A, meshtrace);
//                meshtrace.to_cartesian(A, optimized_points);
//                meshtrace.to_cartesian(A, optFix, optFace, optFree);
//                Eigen::write_binary("opt.dat", optimized_points);
//                pass = true;
//            }
//        } catch (exception& e) {
//            cout << e.what() << endl;
//            pass = false;
//        }
//        }
//        meshtrace.to_cartesian(A, optimized_points);
//        meshtrace.to_cartesian(A, optFix, optFace, optFree);
//        Eigen::write_binary("opt.dat", optimized_points);
//
//    } else {
//        Eigen::read_binary("random.dat", random_points);
//        Eigen::read_binary("di.dat", delete_insert_points);
//        Eigen::read_binary("Particle.dat", particle_dump);
//        A = vector<ParticleD>(particle_dump.rows());
//        for (int i = 0; i < particle_dump.rows(); i++) {
//            cout << particle_dump.row(i) << endl;
//            A[i].flag = (MESHTRACE::FLAG)particle_dump(i, 0);
//            A[i].cell_id = (int)particle_dump(i, 1);
//            if (A[i].flag == FREE) {
//                A[i].bc.resize(1, 4);
//                A[i].bc[0] = particle_dump(i, 2);
//                A[i].bc[1] = particle_dump(i, 3);
//                A[i].bc[2] = particle_dump(i, 4);
//                A[i].bc[3] = particle_dump(i, 5);
//            } else {
//                A[i].bc.resize(1, 3);
//                A[i].bc[0] = particle_dump(i, 2);
//                A[i].bc[1] = particle_dump(i, 3);
//                A[i].bc[2] = particle_dump(i, 4);
//            }
//        }
//        meshtrace.to_cartesian(A, diFix, diFace, diFree);
//
//
//        LBFGS_optimization(l, A, meshtrace);
//        meshtrace.to_cartesian(A, optimized_points);
//        meshtrace.to_cartesian(A, optFix, optFace, optFree);
//        Eigen::write_binary("opt.dat", optimized_points);
//    }
//
//
//
//
//
//    // MatrixX3d points;
//    // RowVector4d bc;
//    // bc << 0.25, 0.25, 0.25, 0.25;
//    // Particle<double> s {0, bc};
//    // Vector2d angle;
//    // angle << igl::PI/4, -igl::PI/4;
//    // meshtrace.tet_trace.tracing(1.0, s, -angle, callback);
//
//    // viewer.data().set_points(trace_points, RowVector3d(0, 0, 0.82745098));
//    // viewer.data().add_edges(trace_points.block(0, 0, trace_points.rows() - 1, 3),
//    //                         trace_points.block(1, 0, trace_points.rows() - 1, 3),
//    //                         Eigen::RowVector3d(1.0, 0, 0));
//    // viewer.data().add_edges(debug_point_a, debug_point_b, Eigen::RowVector3d(0, 1, 0));
//
//    // viewer.data().add_edges(B - 0.05 * l * FF0F, B + 0.05 * l * FF0F, Eigen::RowVector3d(0, 1, 0));
//    // viewer.data().add_edges(B - 0.05 * l * FF1F, B + 0.05 * l * FF1F, Eigen::RowVector3d(1, 0, 0));
//}

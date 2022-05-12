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

// Input frame field constraints

using namespace std;
using namespace Eigen;

#define datapath "/Users/liujunliang/Documents/Codes/IntrinsicOpt/dataset/"

Eigen::MatrixXd V;
Eigen::MatrixXi T;
Eigen::MatrixXi TF;
Eigen::MatrixX3d trace_points;
Eigen::MatrixX3d debug_point_a;
Eigen::MatrixX3d debug_point_b;

Eigen::MatrixXd random_points;
Eigen::MatrixXd delete_insert_points;
Eigen::MatrixXd optimized_points;

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
    }
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
    create_trivial_case(V, T, 10, 0.1);

    MatrixXd FF0T = MatrixXd::Zero(T.rows(), 3);
    MatrixXd FF1T = MatrixXd::Zero(T.rows(), 3);
    MatrixXd FF2T = MatrixXd::Zero(T.rows(), 3);

    FF0T.col(0) = MatrixXd::Constant(T.rows(), 1, 1.0);
    FF1T.col(1) = MatrixXd::Constant(T.rows(), 1, 1.0);
    FF2T.col(2) = MatrixXd::Constant(T.rows(), 1, 1.0);

    auto [out_face_map, sharp_edge_map] = get_surface_mesh(V, T, TF);

    MatrixXd FF0F, FF1F;

    MatrixXd particle_dump;

    igl::barycenter(V, TF, B);

    igl::per_face_normals(V, TF, N);

    FF0F.resize(TF.rows(), 3);
    FF1F.resize(TF.rows(), 3);

    RowVector3d X {1.0, 0, 0};
    RowVector3d Z {0, 0, 1.0};

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
    vector<ParticleD> A;

    // l = igl::avg_edge_length(V, T);
    l = 0.1;
    MeshTraceManager<double> meshtrace(V, T, TF, FF0T, FF1T, FF2T, FF0F, FF1F, out_face_map);

    bool use_cache = false;
    
    if (!use_cache) {
        bool pass = false;
        while (!pass) {
        try {
            A.resize(0);
            point_sample(V, T, TF, A, l, out_face_map);
            meshtrace.to_cartesian(A, ranFix, ranFace, ranFree);
            cout << "cartesian fix: " << ranFix.rows() << endl;
            meshtrace.to_cartesian(A, random_points);
//            log(); return 0;
//            meshtrace.particle_insert_and_delete(A, 0.8 * l, l);
            int iteration_cnt = 0;
            while (iteration_cnt++ < 1) {
                cout << "============ " << iteration_cnt << "th iteration==================" << endl;
                bool flag = meshtrace.particle_insert_and_delete(A, 0.8 * l, l);
                meshtrace.to_cartesian(A, delete_insert_points);
                meshtrace.to_cartesian(A, diFix, diFace, diFree);
                Eigen::write_binary("random.dat", random_points);
                Eigen::write_binary("di.dat", delete_insert_points);

                particle_dump.resize(A.size(), 6);
                for (int i = 0; i < A.size(); i++) {
                    particle_dump(i, 0) = A[i].flag;
                    particle_dump(i, 1) = A[i].cell_id;
                    if (A[i].flag == FREE) {
                        particle_dump(i, 2) = A[i].bc[0];
                        particle_dump(i, 3) = A[i].bc[1];
                        particle_dump(i, 4) = A[i].bc[2];
                        particle_dump(i, 5) = A[i].bc[3];
                    } else {
                        particle_dump(i, 2) = A[i].bc[0];
                        particle_dump(i, 3) = A[i].bc[1];
                        particle_dump(i, 4) = A[i].bc[2];
                    }
                }

                Eigen::write_binary("Particle.dat", particle_dump);
                if (flag) break;
                LBFGS_optimization(l, A, meshtrace);
                meshtrace.to_cartesian(A, optimized_points);
                meshtrace.to_cartesian(A, optFix, optFace, optFree);
                Eigen::write_binary("opt.dat", optimized_points);
                pass = true;
            }
        } catch (exception& e) {
            cout << e.what() << endl;
            pass = false;
        }
        }
        meshtrace.to_cartesian(A, optimized_points);
        meshtrace.to_cartesian(A, optFix, optFace, optFree);
        Eigen::write_binary("opt.dat", optimized_points);

    } else {
        Eigen::read_binary("random.dat", random_points);
        Eigen::read_binary("di.dat", delete_insert_points);
        Eigen::read_binary("Particle.dat", particle_dump);
        A = vector<ParticleD>(particle_dump.rows());
        for (int i = 0; i < particle_dump.rows(); i++) {
            cout << particle_dump.row(i) << endl;
            A[i].flag = (MESHTRACE::FLAG)particle_dump(i, 0);
            A[i].cell_id = (int)particle_dump(i, 1);
            if (A[i].flag == FREE) {
                A[i].bc.resize(1, 4);
                A[i].bc[0] = particle_dump(i, 2);
                A[i].bc[1] = particle_dump(i, 3);
                A[i].bc[2] = particle_dump(i, 4);
                A[i].bc[3] = particle_dump(i, 5);
            } else {
                A[i].bc.resize(1, 3);
                A[i].bc[0] = particle_dump(i, 2);
                A[i].bc[1] = particle_dump(i, 3);
                A[i].bc[2] = particle_dump(i, 4);
            }
        }
        meshtrace.to_cartesian(A, diFix, diFace, diFree);


        LBFGS_optimization(l, A, meshtrace);
        meshtrace.to_cartesian(A, optimized_points);
        meshtrace.to_cartesian(A, optFix, optFace, optFree);
        Eigen::write_binary("opt.dat", optimized_points);
    }

    log(); return 0;

    

    // MatrixX3d points;
    // RowVector4d bc;
    // bc << 0.25, 0.25, 0.25, 0.25;
    // Particle<double> s {0, bc};
    // Vector2d angle;
    // angle << igl::PI/4, -igl::PI/4;
    // meshtrace.tet_trace.tracing(1.0, s, -angle, callback);

    // viewer.data().set_points(trace_points, RowVector3d(0, 0, 0.82745098));
    // viewer.data().add_edges(trace_points.block(0, 0, trace_points.rows() - 1, 3),
    //                         trace_points.block(1, 0, trace_points.rows() - 1, 3),
    //                         Eigen::RowVector3d(1.0, 0, 0));
    // viewer.data().add_edges(debug_point_a, debug_point_b, Eigen::RowVector3d(0, 1, 0));

    // viewer.data().add_edges(B - 0.05 * l * FF0F, B + 0.05 * l * FF0F, Eigen::RowVector3d(0, 1, 0));
    // viewer.data().add_edges(B - 0.05 * l * FF1F, B + 0.05 * l * FF1F, Eigen::RowVector3d(1, 0, 0));
}

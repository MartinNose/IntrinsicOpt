#include <iostream>
#include <cstdlib>
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
#include <unistd.h>
#include "omp.h"
#include "LBFGS_Opt.h"
#include "surface_mesh.h"

// Input frame field constraints

using namespace std;

#define datapath "/Users/liujunliang/Documents/Codes/IntrinsicOpt/dataset/"

Eigen::MatrixXd V;
Eigen::MatrixXi T;
igl::opengl::glfw::Viewer viewer;
Eigen::MatrixX3d trace_points;
Eigen::MatrixX3d debug_point_a;
Eigen::MatrixX3d debug_point_b;


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

int main(int, char**) {

    readVTK(datapath "l1-poly-dat/hex/rod/orig.tet.vtk", V, T);

    MatrixXd FF0T = MatrixXd::Zero(T.rows(), 3);
    MatrixXd FF1T = MatrixXd::Zero(T.rows(), 3);
    MatrixXd FF2T = MatrixXd::Zero(T.rows(), 3);

    FF0T.col(0) = MatrixXd::Constant(T.rows(), 1, 1.0);
    FF1T.col(1) = MatrixXd::Constant(T.rows(), 1, 1.0);
    FF2T.col(2) = MatrixXd::Constant(T.rows(), 1, 1.0);

    MatrixXi TF;

    auto [out_face_map, sharp_edge_map] = get_surface_mesh(V, T, TF);

    MatrixXd FF0F, FF1F;

    MatrixXd B;
    MatrixXd N;

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

    double l = igl::avg_edge_length(V, T);
    // double l = igl::avg_edge_length(V, T);
    vector<ParticleD> A;
    point_sample(V, T, TF, A, l, out_face_map);
    MeshTraceManager<double> meshtrace(V, T, TF, FF0T, FF1T, FF2T, FF0F, FF1F);

    Eigen::MatrixXd points;
    meshtrace.to_cartesian(A, points);

    
    LBFGS_optimization(l, A, meshtrace);

    // // MatrixX3d points;
    // RowVector4d bc;
    // bc << 0.25, 0.25, 0.25, 0.25;
    // Particle<double> s {0, bc};
    // Vector2d angle;
    // angle << igl::PI/4, -igl::PI/4;
    // meshtrace.tracing(1.0, s, angle, callback);

    // viewer.data().set_points(trace_points, RowVector3d(0, 0, 0.82745098));
    // viewer.data().add_edges(trace_points.block(0, 0, trace_points.rows() - 1, 3),
    //                         trace_points.block(1, 0, trace_points.rows() - 1, 3),
    //                         Eigen::RowVector3d(1.0, 0, 0));
    // viewer.data().add_edges(debug_point_a, debug_point_b, Eigen::RowVector3d(0, 1, 0));

    // viewer.data().add_edges(B - 0.05 * l * FF0F, B + 0.05 * l * FF0F, Eigen::RowVector3d(0, 1, 0));
    // viewer.data().add_edges(B - 0.05 * l * FF1F, B + 0.05 * l * FF1F, Eigen::RowVector3d(1, 0, 0));
    // viewer.data().add_edges(B, B - 0.1 * l * N, Eigen::RowVector3d(1, 0, 0));

    viewer.data().set_points(points.block(V.size(), 0, A.size() - V.size(), 3), RowVector3d(0, 0, 0.82745098));

    
    viewer.data().set_mesh(V, TF);
    viewer.data().set_face_based(true);

    viewer.launch();
}

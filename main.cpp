#include <iostream>
#include "Eigen/Core"
#include <igl/barycenter.h>
#include "readVTK.h"
#include <igl/PI.h>
#include "MeshTrace/trace.h"
#include "MeshTrace/trace_manager.h"
#include "point_sample.h"
#include "LBFGS_Opt.h"
#include "surface_mesh.h"
#include "trivial_case.h"
#include "read_zyz.h"

// Input frame field constraints

using namespace std;
using namespace Eigen;

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

Eigen::MatrixXd debug_point[10];

Eigen::MatrixXd diFix;
Eigen::MatrixXd diFace;
Eigen::MatrixXd diFree;
Eigen::MatrixXd ranFix;
Eigen::MatrixXd ranFace;
Eigen::MatrixXd ranFree;
Eigen::MatrixXd optFix;
Eigen::MatrixXd optFace;
Eigen::MatrixXd optFree;

int boundary_count;

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


int main(int argc, char* argv[]) {
//    readVTK(datapath "l1-poly-dat/hex/kitty/orig.tet.vtk", V, T);
    // readVTK(datapath "trivial.vtk", V, T);
//    int col_of_cube;
//    cin >> col_of_cube;
//    col_of_cube = 5;
//    create_trivial_case(V, T, col_of_cube, 0.1);

    readVTK("/Users/liujunliang/Documents/Codes/IntrinsicOpt/dataset/joint.vtk", V, T);

    MatrixXd FF0T;
    MatrixXd FF1T;
    MatrixXd FF2T;

    read_zyz("/Users/liujunliang/Documents/Codes/IntrinsicOpt/dataset/mesh.zyz", FF0T, FF1T, FF2T);
    assert(FF0T.rows() == T.rows() && FF1T.rows() == T.rows() && FF2T.rows() == T.rows());

    auto[out_face_map, sharp_edge_map, surface_point] = get_surface_mesh(V, T, TF);

    boundary_count = 0;
    for (int i = 0; i < surface_point.size(); i++) {
        if (surface_point[i]) boundary_count++;
    }

    MatrixXd FF0F, FF1F;

    MatrixXd particle_dump;

    igl::barycenter(V, TF, B);

    igl::per_face_normals(V, TF, N);

    FF0F.resize(TF.rows(), 3);
    FF1F.resize(TF.rows(), 3); // TODO project ff to surface

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

    MeshTraceManager<double> meshtrace(V, T, TF, FF0T, FF1T, FF2T, FF0F, FF1F, out_face_map, surface_point);

    vector<ParticleD> PV;
//    for (int i = 0; i < T.rows(); i++) {
//        if (i % 5 != 1) continue;
////        if (i/5 / (col_of_cube * col_of_cube) != 2) continue;
//        ParticleD p;
//        p.cell_id = i;
//        p.bc.resize(1, 4);
//        p.bc << 0.25, 0.25, 0.25, 0.25;
//        p.flag = MESHTRACE::FREE;
//        PV.push_back(p);
//    }

//    l = 0.15000000001;
//    l = 0.10000000001;
//    cin >> l;
//    l = 0.8 * igl::avg_edge_length(V, T);

    l = 0.030802;

    point_sample_init(V, T, TF, PV, l, out_face_map, meshtrace);
    meshtrace.to_cartesian(PV, debug_point[0]);

    int debug_cnt = 0;
    meshtrace.particle_insert_and_delete(PV, 1.5 * l, l);
    meshtrace.to_cartesian(PV, debug_point[1]);

    LBFGS_init(l, PV, meshtrace, &(debug_point[(debug_cnt++) % 9 + 1]));
    meshtrace.to_cartesian(PV, debug_point[2]);

    for (int i = 0; i < 10; i++) {
        if (meshtrace.particle_insert_and_delete(PV, 1.5 * l, l)) {
            break;
        };
        meshtrace.to_cartesian(PV, debug_point[3]);

        LBFGS_optimization(l, PV, meshtrace, &(debug_point[4]));
        meshtrace.to_cartesian(PV, debug_point[5]);

        int removed = meshtrace.remove_boundary(PV, 0.5 * l);
        meshtrace.to_cartesian(PV, debug_point[6]);

        point_sample(V, T, TF, PV, l, out_face_map, meshtrace, removed);
        meshtrace.to_cartesian(PV, debug_point[7]);
    }

    meshtrace.particle_insert_and_delete(PV, 1.5 * l, l);
    meshtrace.to_cartesian(PV, debug_point[8]);

    meshtrace.remove_boundary(PV, 0.5 * l);
    meshtrace.to_cartesian(PV, debug_point[9]);

    // TODO Remove boundary

    return 0;
}

#include <iostream>
#include "Eigen/Core"
#include <igl/barycenter.h>
#include "read_vtk.h"
#include "read_lattice.h"
#include <igl/PI.h>
#include "MeshTrace/trace.h"
#include "MeshTrace/trace_manager.h"
#include "point_sample.h"
#include "LBFGS_Opt.h"
#include "surface_mesh.h"
#include "trivial_case.h"
#include "read_zyz.h"
#include <cstdlib>

using namespace std;
using namespace Eigen;

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

int main(int argc, char* argv[]) { // input tet_mesh, frame, lattice, out_put_file
    bool debug_mode = false;
    // if (argc != 6 && argc != 5 && argc != 3) {
    //     cout << "Usage: \nIntrinsicOpt {input.tet.vtk} {frame.zyz} {lattice.txt} {output_path} [if_debug]" << endl;
    //     cout << "IntrinsicOpt {num_of_trivial_case_col} {lattice}" << endl;
    //     return 0;
    // }

    MatrixXd FF0T;
    MatrixXd FF1T;
    MatrixXd FF2T;

    MatrixXd FF0F;
    MatrixXd FF1F;

    read_vtk("/home/martinnose/HexDom/tmp/tmp/mesh.vtk", V, T);
    cout << "V: " << V.rows() << " T: " << T.rows() << endl;
    read_zyz("/home/martinnose/HexDom/tmp/tmp/mesh.zyz", FF0T, FF1T, FF2T);
    cout << "read " << FF0T.rows() << "mat" << endl;
    l = read_lattice("/home/martinnose/HexDom/tmp/tmp/mesh_length.txt");
    cout << "lattice: " << l << endl;

    // if (argc == 5 || argc == 6) {
    //     debug_mode = false;
    //     cout << "reading " << argv[1] << endl;
    //     read_vtk(argv[1], V, T);
    //     cout << "V: " << V.rows() << " T: " << T.rows() << endl;
    //     cout << "reading " << argv[2] << endl;
    //     read_zyz(argv[2], FF0T, FF1T, FF2T);
    //     cout << "read " << FF0T.rows() << "mat" << endl;
    //     cout << "reading " << argv[3] << endl;
    //     l = read_lattice(argv[3]);
    //     cout << "lattice: " << l << endl;
    // } else {
    //     debug_mode = true;
    //     create_trivial_case(V, T, atoi(argv[1]), 0.1);
    //     l = atof(argv[2]);
    //     FF0T = MatrixXd::Zero(T.rows(), 3);
    //     FF1T = MatrixXd::Zero(T.rows(), 3);
    //     FF2T = MatrixXd::Zero(T.rows(), 3);
    //     FF0T.col(0) = MatrixXd::Constant(T.rows(), 1, 1.0);
    //     FF1T.col(1) = MatrixXd::Constant(T.rows(), 1, 1.0);
    //     FF2T.col(2) = MatrixXd::Constant(T.rows(), 1, 1.0);
    // }

    write_vtk_points("xxx.vtk", V);
    if (argc == 5 || argc == 6) write_vtk_points(argv[4], debug_point[9]);

    auto[out_face_map, sharp_edge_map, surface_point] = get_surface_mesh(V, T, TF);

    igl::barycenter(V, TF, B);
    igl::per_face_normals(V, TF, N);

    assert(FF0T.rows() == T.rows() && FF1T.rows() == T.rows() && FF2T.rows() == T.rows());

    FF0F.resize(TF.rows(), 3);
    FF1F.resize(TF.rows(), 3);

    for (auto const &[key, val]: out_face_map) {
        int tri = val.first;
        int tet = val.second;
        Vector3d n = N.row(tri).normalized();

        Vector3d ff[3];
        ff[0] = FF0T.row(tet);
        ff[1] = FF1T.row(tet);
        ff[2] = FF2T.row(tet);

        ff[0] -= ff[0].dot(n) * n;
        ff[1] -= ff[1].dot(n) * n;
        ff[2] -= ff[2].dot(n) * n;

        Vector3d tmp;
        for (int i = 0; i < 2; i++) {
            int max_index = i;
            for (int j = i + 1; j < 3; j++) {
                if (ff[j].norm() > ff[max_index].norm()) max_index = j;
            }
            if (max_index != i) {
                tmp = ff[i]; ff[i] = ff[max_index]; ff[max_index] = tmp;
            }
        }

        FF0F.row(tri) = ff[0];
        FF1F.row(tri) = ff[1];
    }
    
    MeshTraceManager<double> meshtrace(V, T, TF, FF0T, FF1T, FF2T, FF0F, FF1F, out_face_map, surface_point);

    vector<ParticleD> PV;

    point_sample_init(V, T, TF, PV, l, out_face_map, meshtrace);
    if (debug_mode) meshtrace.to_cartesian(PV, debug_point[0]);

    int debug_cnt = 0;
    meshtrace.particle_insert_and_delete(PV, 1.5 * l, l);
    if (debug_mode) meshtrace.to_cartesian(PV, debug_point[1]);

    LBFGS_init(l, PV, meshtrace, debug_mode ? &(debug_point[(debug_cnt++) % 9 + 1]) : nullptr);

    if (debug_mode) meshtrace.to_cartesian(PV, debug_point[2]);

    for (int i = 0; i < 10; i++) {
        cout << "iteration: " << i + 1 << endl;
        if (meshtrace.particle_insert_and_delete(PV, 1.5 * l, l)) {
            break;
        };
        if (debug_mode) meshtrace.to_cartesian(PV, debug_point[3]);

        LBFGS_optimization(l, PV, meshtrace, debug_mode ? &(debug_point[4]) : nullptr);
        if (debug_mode) meshtrace.to_cartesian(PV, debug_point[5]);

        int removed = meshtrace.remove_boundary(PV, 0.5 * l);
        if (debug_mode) meshtrace.to_cartesian(PV, debug_point[6]);

        point_sample(V, T, TF, PV, l, out_face_map, meshtrace, removed);
        if (debug_mode) meshtrace.to_cartesian(PV, debug_point[7]);
    }

    meshtrace.particle_insert_and_delete(PV, 1.5 * l, l);
    if (debug_mode) meshtrace.to_cartesian(PV, debug_point[8]);

    meshtrace.remove_boundary(PV, 0.5 * l);
    meshtrace.to_cartesian(PV, debug_point[9]);

    // if (argc == 5 || argc == 6) 
    write_vtk_points("/home/martinnose/HexDom/tmp/tmp/mesh_points.vtk", debug_point[9]);
    write_binary("inner_points", debug_point[9]);


    if (argc == 3) {
        // write all to vtk
    }

    return 0;
}

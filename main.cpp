#include <iostream>
#include "Eigen/Core"
#include <igl/barycenter.h>
#include <igl/avg_edge_length.h>
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
#include <ctime>

using namespace std;
using namespace Eigen;

Eigen::MatrixXd V;
Eigen::MatrixXi T;
Eigen::MatrixXi TF;
Eigen::MatrixXd trace_points;
Eigen::MatrixXd debug_point_a;
Eigen::MatrixXd debug_point_b;

vector<Eigen::MatrixXd> debug_points;


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

double time_insert_delete, time_lbfgs;

int main(int argc, char* argv[]) { // input tet_mesh, frame, lattice, out_put_file
    time_insert_delete = 0;
    time_lbfgs = 0;
    double cur_time;

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

    if (argc == 6) {
        debug_mode = false;
        cout << "reading " << argv[1] << endl;
        read_vtk(argv[1], V, T);
        cout << "V: " << V.rows() << " T: " << T.rows() << endl;
        cout << "reading " << argv[2] << endl;
        read_zyz(argv[2], FF0T, FF1T, FF2T);
        cout << "read " << FF0T.rows() << "mat" << endl;
        cout << "reading " << argv[3] << endl;
        l = read_lattice(argv[3]);
        cout << "lattice: " << l << endl;
    } else if (argc == 1) {
        debug_mode = true;
        create_trivial_case(V, T, 5, 0.1);
        l = 0.1;
        FF0T = MatrixXd::Zero(T.rows(), 3);
        FF1T = MatrixXd::Zero(T.rows(), 3);
        FF2T = MatrixXd::Zero(T.rows(), 3);
        FF0T.col(0) = MatrixXd::Constant(T.rows(), 1, 1.0);
        FF1T.col(1) = MatrixXd::Constant(T.rows(), 1, 1.0);
        FF2T.col(2) = MatrixXd::Constant(T.rows(), 1, 1.0);
    }

    auto[out_face_map, sharp_edge_map, surface_point] = get_surface_mesh(V, T, TF);

    FF0F.resize(TF.rows(), 3);
    FF1F.resize(TF.rows(), 3);

    igl::barycenter(V, TF, B);
    igl::per_face_normals(V, TF, N);

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

    boundary_count = 0;
    for (int i = 0; i < surface_point.size(); i++) {
        if (surface_point[i]) boundary_count++;
    }

    assert(FF0T.rows() == T.rows() && FF1T.rows() == T.rows() && FF2T.rows() == T.rows());

    MeshTraceManager<double> meshtrace(V, T, TF, FF0T, FF1T, FF2T, FF0F, FF1F, out_face_map, surface_point);

    // ParticleD p;
    // p.cell_id = 8410;
    // p.bc = RowVector3d(0.0084215667506007952, 0.21193537216738012, 0.77964306108201908);
    // p.flag = FACE;
    // Vector3d di {0.67720975144341555, -0.0033049319753077153, 0.73570129631219094};
    // di *= -1. * 0.030800546378627988;
    // meshtrace.tracing(p, di);
    vector<ParticleD> PV;

    point_sample_init(V, T, TF, PV, l, out_face_map, meshtrace);
    MatrixXd temp;
    meshtrace.to_cartesian(PV, temp);
    debug_points.push_back(temp);

    int debug_cnt = 0;
    cur_time = std::clock();
    meshtrace.particle_insert_and_delete(PV, 1.5 * l, l);
    time_insert_delete += (std::clock() - cur_time) / (double) CLOCKS_PER_SEC;
    meshtrace.to_cartesian(PV, temp);
    debug_points.push_back(temp);

    LBFGS_init(l, PV, meshtrace, debug_mode ? &(debug_points[(debug_cnt++) % 9 + 1]) : nullptr);

    meshtrace.to_cartesian(PV, temp);
    debug_points.push_back(temp);

    for (int i = 0; i < 10; i++) {
        cout << "iteration: " << i + 1 << endl;
        cur_time = std::clock();
        bool flag = meshtrace.particle_insert_and_delete(PV, 1.5 * l, l);
        time_insert_delete += (std::clock() - cur_time) / (double) CLOCKS_PER_SEC;
        if (flag) {
            break;
        };

        cur_time = std::clock();
        LBFGS_optimization(l, PV, meshtrace, debug_mode ? &(debug_points[4]) : nullptr);
        time_lbfgs += (std::clock() - cur_time) / (double) CLOCKS_PER_SEC;
        
        meshtrace.to_cartesian(PV, temp);
        debug_points.push_back(temp);
    }

    cur_time = std::clock();
    meshtrace.particle_insert_and_delete(PV, 1.5 * l, l);
    time_insert_delete += (std::clock() - cur_time) / (double) CLOCKS_PER_SEC;
    meshtrace.to_cartesian(PV, temp); debug_points.push_back(temp);

    cout << "Pipeline Execution Done, insert delete time: " << time_insert_delete << "s, lbfgs time: " << time_lbfgs << "s." <<endl;

    // meshtrace.remove_boundary(PV, 0.5 * l);
    // meshtrace.to_cartesian(PV, temp); debug_points.push_back(temp);

    // if (argc == 5 || argc == 6) 
    MatrixXd inner;
    MatrixXd surface;

    vector<Particle<>> PV_inner;
    vector<Particle<>> PV_surface;

    for (int i = 0; i < PV.size(); i++) {
        if (PV[i].flag == MESHTRACE::FREE) PV_inner.push_back(PV[i]);
        else PV_surface.push_back(PV[i]);
    }

    meshtrace.to_cartesian(PV_inner, inner);
    meshtrace.to_cartesian(PV_surface, surface);

    write_vtk_points("/home/ubuntu/HexDom/tmp/tmp/mesh_points.vtk", inner);
    write_vtk_points("/home/ubuntu/HexDom/tmp/tmp/mesh_surface_points.vtk", surface);
    
    for (int i = 0; i < debug_points.size(); i++) {
        write_vtk_points("/home/ubuntu/HexDom/tmp/cube/particles_" + to_string(i) + ".vtk", debug_points[i]);
    }
    
    // write_binary(argv[5], surface);
    if (argc == 6) {
        cout << "write " << surface.size() << " points to " << argv[4] << endl;
        write_matrix_with_binary(argv[4], surface.transpose());
        cout << "write " << inner.size() << " inner points to " << argv[5] << endl;
        write_matrix_with_binary(argv[5], inner.transpose());
    }
    

    return 0;
}

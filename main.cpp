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
#include <igl/rotate_vectors.h>
#include <igl/copyleft/comiso/nrosy.h>
#include <igl/copyleft/comiso/miq.h>
#include <igl/copyleft/comiso/frame_field.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/PI.h>
#include "point_sample.h"

Eigen::MatrixXd V;
Eigen::MatrixXi T;

Eigen::MatrixXd B;

// Input frame field constraints
Eigen::VectorXi b;
Eigen::MatrixXd bc1;
Eigen::MatrixXd bc2;

// Interpolated frame field
Eigen::MatrixXd FF1, FF2;

igl::opengl::glfw::Viewer viewer;

using namespace std;

int main(int, char**) {
    std::string datapath = "./dataset/example1/";
    igl::readOBJ(datapath + "bumpy-cube.obj", V, T);
    Eigen::MatrixXd temp;
    igl::readDMAT(datapath + "bumpy-cube.dmat",temp);

    cout << "test" << endl;

    vector<ParticleD3> A;
    point_sample<Eigen::MatrixXd, Eigen::MatrixXi, vector<ParticleD3>>(V, T, A);

    cout << "------------" << endl;
    // Interpolate the frame field
    Eigen::read_binary((datapath + "FF1.dat").c_str(), FF1);
    Eigen::read_binary((datapath + "FF2.dat").c_str(), FF2);
    cout << "V rows: " << V.rows() << " cols: " << V.cols() << endl;
    cout << "T rows: " << T.rows() << " cols: " << T.cols() << endl;

    cout << "FF1 rows: " << FF1.rows() << " cols: " << FF1.cols() << endl;
    cout << "FF2 rows: " << FF2.rows() << " cols: " << FF2.cols() << endl;

    igl::barycenter(V, T, B);

    double global_scale =  .05*igl::avg_edge_length(V, T);

    viewer.data().set_mesh(V, T);
    viewer.data().set_face_based(true);

    viewer.data().add_edges(B - global_scale*FF1, B + global_scale*FF1 ,Eigen::RowVector3d(0.3, 1, 0.3));
    viewer.data().add_edges(B - global_scale*FF2, B + global_scale*FF2, Eigen::RowVector3d(0.3, 0.3, 1));

    viewer.launch();
}

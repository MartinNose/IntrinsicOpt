#include "iostream"
#include "trace.h"
#include "Eigen/Core"

using namespace std;

typedef MeshTrace<double, 3>::Particle ParticleD3;

template<class Matrix1, class Matrix2, typename Particles>
void point_sample(Matrix1& V, Matrix2& T, Particles & P) {
    cout << "V rows: " << V.rows() << " cols: " << V.cols() << endl;
    cout << "T rows: " << T.rows() << " cols: " << T.cols() << endl;

    int cell_id = 1;
    Eigen::Matrix<double, 1, 3> bc;

    bc << 0.1, 0.1, 0.8;

    P.emplace_back(1, bc);
};
#pragma once
#include <Eigen/Core>

using namespace Eigen;

// V #tet_p by 3
// T #tet by 4
// ti #p by 1
// B #p by 4
// P #p by 3
void barycentric_to_cartesian(const MatrixXd &V, const MatrixXi &T, const MatrixXi &ti, const MatrixXd &B, MatrixX3d &P) {
    std::cout << "Converting from barycentric to cartecian" << std::endl;
    P.resize(ti.rows(), 3);
    #pragma omp parallel for
    for (int i = 0; i < ti.rows(); i++) {
        RowVector4i tet_i = T.row(ti(i, 0));
        RowVector4d bc = B.row(i);
        P.row(i) = bc[0] * V.row(tet_i[0]) + bc[1] * V.row(tet_i[1]) + bc[2] * V.row(tet_i[2]) + bc[3] * V.row(tet_i[3]);
    }
    std::cout << "Convertion done. " << ti.rows() << " particles total.\n----------------------------------" << std::endl;
}
#pragma once
#include <Eigen/Core>
#include "MeshTrace/trace.h"

using namespace Eigen;
using namespace MESHTRACE;

// V #tet_p by 3
// T #tet by 4
// ti #p by 1
// B #p by 4
// P #p by 3
void barycentric_to_cartesian(const MatrixXd &V, const MatrixXi &T, const vector<ParticleD> &A, MatrixXd &P) {
    std::cout << "Converting from barycentric to cartecian" << std::endl;

    P.resize(A.size(), 3);
    #pragma omp parallel for
    for (int i = 0; i < A.size(); i++) {
        RowVector4i tet_i = T.row(A[i].cell_id);
        RowVector4d bc = A[i].bc;
        P.row(i) = bc[0] * V.row(tet_i[0]) + bc[1] * V.row(tet_i[1]) + bc[2] * V.row(tet_i[2]) + bc[3] * V.row(tet_i[3]);
    }
    std::cout << "Convertion done. " << A.size() << " particles total.\n----------------------------------" << std::endl;
}
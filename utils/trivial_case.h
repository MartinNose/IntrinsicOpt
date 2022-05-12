#include "Eigen/Core"

using namespace Eigen;

void create_trivial_case(MatrixXd &V, MatrixXi &T, int m = 10, double l = 0.1) {
    int n = m + 1;
    V.resize(n * n * n, 3);
    Vector3d x {l, 0, 0};
    Vector3d y {0, l, 0};
    Vector3d z {0, 0, l};
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                V.row(i * n * n + j * n + k) = i * z + j * x + k * y;
            }
        }
    }
    T.resize(m * m * m * 5, 4);
    bool mark = false;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < m; k++) {
                int c0 = i * n * n + j * n + k;
                int c1 = c0 + n;
                int c2 = c0 + 1;
                int c3 = c1 + 1;
                int c4 = c0 + n * n;
                int c5 = c4 + n;
                int c6 = c4 + 1;
                int c7 = c5 + 1;
                if (mark) {
                    T.row((i * m * m + j * m + k) * 5 + 0) << c0, c1, c2, c4;
                    T.row((i * m * m + j * m + k) * 5 + 1) << c1, c2, c4, c7;
                    T.row((i * m * m + j * m + k) * 5 + 2) << c3, c2, c1, c7;
                    T.row((i * m * m + j * m + k) * 5 + 3) << c1, c4, c5, c7;
                    T.row((i * m * m + j * m + k) * 5 + 4) << c6, c7, c4, c2;
                } else {
                    T.row((i * m * m + j * m + k) * 5 + 0) << c0, c6, c4, c5;
                    T.row((i * m * m + j * m + k) * 5 + 1) << c0, c1, c3, c5;
                    T.row((i * m * m + j * m + k) * 5 + 2) << c0, c3, c2, c6;
                    T.row((i * m * m + j * m + k) * 5 + 3) << c0, c6, c5, c3;
                    T.row((i * m * m + j * m + k) * 5 + 4) << c3, c5, c7, c6;
                }
                mark = !mark;
            }
            if (m % 2 == 0) mark = !mark;
        }
        if (m % 2 == 0) mark = !mark;
    }
}
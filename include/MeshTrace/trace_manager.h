#pragma once
#include <iostream>
#include "MeshTrace/trace.h"
#include <igl/per_face_normals.h>
#include "Eigen/Core"
#include <math.h>

namespace MESHTRACE {
template<typename Scalar = double>
class MeshTraceManager {
private:
    MeshTrace<Scalar, 4> tet_trace;
    MeshTrace<Scalar, 3> tri_trace;
    MatrixXd tri_normal;
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &VT;
    const Eigen::Matrix<int, Eigen::Dynamic, 4> &TT;
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &FF0T;
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &FF1T;
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &FF2T;
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &VF;
    const Eigen::Matrix<int, Eigen::Dynamic, 3> &TF;
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &FF0F;
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &FF1F;
public:
    MeshTraceManager(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_VT,
        const Eigen::Matrix<int, Eigen::Dynamic, 4> &_TT,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_FF0T,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_FF1T,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_FF2T,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_VF,
        const Eigen::Matrix<int, Eigen::Dynamic, 3> &_TF,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_FF0F,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_FF1F
    ) : tet_trace(_VT, _TT, _FF0T, _FF1T, _FF2T), tri_trace(_VF, _TF, _FF0F, _FF1F), 
        VT(_VT), TT(_TT), FF0T(_FF0T), FF1T(_FF1T), FF2T(_FF2T),
        VF(_VF), TF(_TF), FF0F(_FF0F), FF1F(_FF1F) {
        igl::per_face_normals(VF, TF, tri_normal);
    }


    bool tracing(Particle<> &p, const Vector3d &v) {
        auto foo = [](const Particle<>& target, double stepLen, double total) {
            cout << "Current step length: " << stepLen << " Total traveled length: " << total << endl;
        };
        if (p.flag == FREE) {
                // direction to theta and phi
            Matrix<Scalar, 2, 1> direct;
            Vector3d ff0, ff1, ff2, n;
            ff0 = FF0T.row(p.cell_id);
            ff1 = FF1T.row(p.cell_id);
            ff2 = FF2T.row(p.cell_id);
            n = ff0.cross(ff1).normalized();
            Vector3d direction_cmp0 = v - (v.dot(n) * n);
            direct(0, 0) = acos(direction_cmp0.normalized().dot(ff0.normalized()));
            if (ff0.cross(direction_cmp0).dot(ff2) < 0) {
                direct(0, 0) = 2 * igl::PI - direct(0, 0);
            }
            direct(1, 0) = acos(direction_cmp0.normalized().dot(v.normalized()));
            if (v.dot(ff2) < 0) {
                direct(1, 0) = -direct(1, 0);
            }
            bool res = tet_trace.tracing(v.norm(), p, direct, foo);
            if (p.flag == FACE) {
                auto [v0, v1, v2] = tet_trace.get_face(p);
                p.cell_id = tri_trace.find_face(v0, v1, v2);
            }
            return res;
        } else if (p.flag == FACE) {
            Vector3d ff0 = FF0F.row(p.cell_id);
            Vector3d ff1 = FF1F.row(p.cell_id);
            Vector3d n = tri_normal.row(p.cell_id);
            n.normalize();
            Vector3d new_v = v - v.dot(n) * n;

            double theta = acos(new_v.normalized().dot(ff0));

            return tri_trace.tracing(new_v.norm(), p, theta, foo);
        } else if (p.flag == EDGE) {
            // TODO 1d support;
            return false;
        } else if (p.flag == POINT) {
            return true;
        }
    }

    inline void project(const Particle<>& p, Vector3d &v) {
            if (p.flag == FREE) {
                return;
            } else if (p.flag == FACE) {
                auto [f0, f1, f2] = tet_trace.get_face(p);
                Vector3d n = (f2 - f0).cross(f1 - f0).normalized();
                v = v - v.dot(n)*n;
                return;
            } else if (p.flag == EDGE) {
                int edge[2];
                int idx = 0;
                for (int i = 0; i < 4; i++) {
                    if (p.bc[i] < BARYCENTRIC_BOUND) {
                        continue;
                    }
                    if (idx == 3) {
                        cerr << "logic error: Face particle doesn't meet constraints." << endl;
                        cout << p.cell_id << endl; 
                        cout << p.bc << endl;
                        exit(-1);
                    }
                    edge[idx++] = i;
                }
                Vector3d v0 = VF.row(TF.row(p.cell_id)[edge[0]]);
                Vector3d v1 = VF.row(TF.row(p.cell_id)[edge[1]]);

                Vector3d e = (v1 - v0).normalized();
                v = v.dot(e) * e;
                return; 
            } else if (p.flag == POINT) {
                v = Vector3d::Zero();
                return;
            }
    }
    
    void to_cartesian(const vector<ParticleD> &A, MatrixXd &P) {
        std::cout << "Converting from barycentric to cartecian" << std::endl;

        P.resize(A.size(), 3);
        #pragma omp parallel for
        for (int i = 0; i < A.size(); i++) {
            if (A[i].flag == FREE) {
                RowVector4i tet_i = TT.row(A[i].cell_id);
                RowVector4d bc = A[i].bc;
                P.row(i) = bc[0] * VT.row(tet_i[0]) + bc[1] * VT.row(tet_i[1]) + bc[2] * VT.row(tet_i[2]) + bc[3] * VT.row(tet_i[3]);
            } else if (A[i].flag == FACE) {
                RowVector3i tri_i = TF.row(A[i].cell_id);
                RowVector3d bc = A[i].bc;
                P.row(i) = bc[0] * VF.row(tri_i[0]) + bc[1] * VF.row(tri_i[1]) + bc[2] * VF.row(tri_i[2]);
            } else if (A[i].flag == EDGE) {
                P.row(i) = A[i].bc;//TODO
            } 
            else if (A[i].flag == POINT) {
                P.row(i) = A[i].bc;
            } 
        }
        std::cout << "Convertion done. " << A.size() << " particles total.\n----------------------------------" << std::endl;
    }
};
}
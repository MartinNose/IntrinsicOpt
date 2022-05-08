#pragma once
#include <iostream>
#include "MeshTrace/trace.h"
#include <igl/per_face_normals.h>
#include "Eigen/Core"
#include <math.h>

#include "kdtree.hpp"

namespace MESHTRACE {
template<typename Scalar = double>
class MeshTraceManager {
private:
    MeshTrace<Scalar, 4> tet_trace;
    MeshTrace<Scalar, 3> tri_trace;
    MatrixXd tri_normal;
public:
    const Eigen::MatrixX<Scalar> V;
    const Eigen::MatrixXi TT;
    const Eigen::MatrixXi TF;
    const Eigen::MatrixX<Scalar> FF0T;
    const Eigen::MatrixX<Scalar> FF1T;
    const Eigen::MatrixX<Scalar> FF2T;
    const Eigen::MatrixX<Scalar> FF0F;
    const Eigen::MatrixX<Scalar> FF1F;
    MeshTraceManager(
        const Eigen::MatrixX<Scalar> &_V,
        const Eigen::MatrixXi &_TT,
        const Eigen::MatrixXi &_TF,
        const Eigen::MatrixX<Scalar> &_FF0T,
        const Eigen::MatrixX<Scalar> &_FF1T,
        const Eigen::MatrixX<Scalar> &_FF2T,
        const Eigen::MatrixX<Scalar> &_FF0F,
        const Eigen::MatrixX<Scalar> &_FF1F
    ) : tet_trace(_V, _TT, _FF0T, _FF1T, _FF2T), tri_trace(_V, _TF, _FF0F, _FF1F), 
        V(_V), TT(_TT), TF(_TF), FF0T(_FF0T), FF1T(_FF1T), FF2T(_FF2T),
        FF0F(_FF0F), FF1F(_FF1F) {
        igl::per_face_normals(V, TF, tri_normal);
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
            
            return false;
        } else if (p.flag == POINT) {
            return true;
        } else {
            cerr << "illegal flag type" <<endl;
            exit(-1);
        }
    }

    inline void project(const Particle<>& p, Vector3d &v) {
            if (p.flag == FREE) {
                return;
            } else if (p.flag == FACE) {
                Vector3d f0 = V.row(TF.row(p.cell_id)[0]);
                Vector3d f1 = V.row(TF.row(p.cell_id)[1]);
                Vector3d f2 = V.row(TF.row(p.cell_id)[2]);
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
                Vector3d v0 = V.row(TF.row(p.cell_id)[edge[0]]);
                Vector3d v1 = V.row(TF.row(p.cell_id)[edge[1]]);

                Vector3d e = (v1 - v0).normalized();
                v = v.dot(e) * e;
                return; 
            } else if (p.flag == POINT) {
                v = Vector3d::Zero();
                return;
            }
    }
    
    void const to_cartesian(const vector<ParticleD> &A, MatrixXd &P) {
        std::cout << "Converting from barycentric to cartecian" << std::endl;
        P.resize(A.size(), 3);
        #pragma omp parallel for
        for (int i = 0; i < A.size(); i++) {
            cout << i << "th/" << A.size() <<" particle converting" << endl;
            if (A[i].flag == FREE) {
                RowVector4i tet_i = TT.row(A[i].cell_id);
                RowVector4d bc = A[i].bc;
                P.row(i) = bc[0] * V.row(tet_i[0]) + bc[1] * V.row(tet_i[1]) + bc[2] * V.row(tet_i[2]) + bc[3] * V.row(tet_i[3]);
            } else if (A[i].flag == FACE) {
                RowVector3i tri_i = TF.row(A[i].cell_id);
                RowVector3d bc = A[i].bc;
                P.row(i) = bc[0] * V.row(tri_i[0]) + bc[1] * V.row(tri_i[1]) + bc[2] * V.row(tri_i[2]);
            } else if (A[i].flag == EDGE) {
                P.row(i) = A[i].bc;//TODO 1d support
            } 
            else if (A[i].flag == POINT) {
                RowVector3i tri_i = TF.row(A[i].cell_id);
                RowVector3d bc = A[i].bc;
                P.row(i) = bc[0] * V.row(tri_i[0]) + bc[1] * V.row(tri_i[1]) + bc[2] * V.row(tri_i[2]);
            } 
        }
        std::cout << "Convertion done. " << A.size() << " particles total.\n----------------------------------" << std::endl;
    }

    void particle_insert_and_delete(vector<ParticleD> &P, double n_r, double lattice) {
        std::cout << "Executing particle deletion scheme" << endl;
        
        double d_particle = 0.5 * lattice;
        double d_edge = 0.75 * lattice;
        double d_quad = 0.9 * lattice;
        double d_hex = 0.9 * lattice;

        MatrixXd points_mat;

        to_cartesian(P, points_mat);
        NNSearch::KDTree kdtree(points_mat, 5);

        vector<ParticleD> new_particles;
        vector<bool> removed(P.size(), false);
        for (int i = 0; i < P.size(); i++) {
            if (P[i].flag == POINT) {
                new_particles.push_back(P[i]);
                continue;
            }

            Vector3d p = points_mat.row(i);
            vector<double> D;
            vector<int> pts_idx;
            vector<double> pts_dist;
            kdtree.radiusSearch(p, n_r, pts_idx, pts_dist);

            for (int j = 0; j < pts_idx.size(); j++) {
                if (!removed[pts_idx[j]] && pts_idx[j] != i) {
                    D.push_back(pts_dist[j]);
                }
            }
            sort(D.begin(), D.end());

            bool flag = false;

            if (D[0] < d_particle) flag = true;
            if (D.size() > 2 && (D[0] + D[1]) / 2 < d_edge) flag = true;
            if (D.size() > 4) {
                double sum = 0;
                for (int j = 0; j < 4; j++) {
                    sum += D[j];
                }
                sum /= 4;
                if (sum < d_quad) flag = true;
            }
            if (D.size() > 8) {
                double sum = 0;
                for (int j = 0; j < 8; j++) {
                    sum += D[j];
                }
                sum /= 8;
                if (sum < d_hex) flag = true;
            }
            if (flag) {
                removed[i] = true;
            } else {
                new_particles.push_back(P[i]);
            }
        }

        int d_cnt = P.size() - new_particles.size();
        cout << "deleted " << d_cnt << " particles." << endl;
        MatrixXd new_points_mat;
        to_cartesian(new_particles, new_points_mat);

        NNSearch::KDTree new_kdtree(new_points_mat, 5);

        vector<ParticleD> C;
        for (int i = 0; i < new_particles.size(); i++) {
            if (new_particles[i].flag != FREE) continue;
            Vector3d ff[3];
            ff[0] = FF0T.row(new_particles[i].cell_id);
            ff[1] = FF1T.row(new_particles[i].cell_id);
            ff[2] = FF2T.row(new_particles[i].cell_id);

            vector<ParticleD> candidates(6, new_particles[i]);

            for (int j = 0; j < 3; j++) {
                tracing(candidates[j * 2 + 0], lattice * ff[j]);
                tracing(candidates[j * 2 + 1], -lattice * ff[j]);
            }

            MatrixXd candidates_mat;
            to_cartesian(candidates, candidates_mat);
            for (int j = 0; j < 6; j++) {
                if (candidates[j].flag != FREE) continue;
                Vector3d c_p = candidates_mat.row(j);

                vector<int> pts_idx;
                vector<double> pts_dist;
                new_kdtree.radiusSearch(c_p, lattice * 3, pts_idx, pts_dist); 

                vector<double> D;
                for (int k = 0; k < pts_idx.size() && D.size() <= 8; k++) {
                    if (removed[pts_idx[k]]) continue;
                    D.push_back(sqrt(pts_dist[k]));
                }

                if (D.size() >= 1 && D[0] < 0.7 * lattice) {// TODO locall lattice
                    continue;
                }

                double min_d = 1e20;
                MatrixXd C_mat;
                to_cartesian(C, C_mat);
                for (int k = 0; k < C.size(); k++) {
                    min_d = min(min_d, (C_mat.row(k) - c_p).norm());
                }

                if (min_d < 0.8 * lattice) // local lattice
                    continue;

                new_particles.push_back(candidates[j]);
                C.push_back(candidates[k]);                
            }
        }

        P = new_particles;
        return C.size() == d_cnt;
    }
};
}
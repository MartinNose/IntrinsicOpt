#pragma once
#include <iostream>
#include "MeshTrace/trace.h"
#include <igl/per_face_normals.h>
#include <igl/barycentric_coordinates.h>
#include <igl/doublearea.h>
#include "Eigen/Core"
#include <cmath>
#include <algorithm>
#include <map>
#include <vector>
#include <ctime>
#include <functional>
#include "KDTreeVectorOfVectorsAdaptor.h"

namespace MESHTRACE {

inline int vertex_of_face[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};

using namespace Eigen;
template<typename Scalar = double>
class MeshTraceManager {
private:
    void convert_to_face(Particle<> &p) {
        assert(p.flag == FACE && p.bc.cols() == 4);
        using namespace igl;
        using namespace std;
        vector<int> face(3);

        RowVector3d p_cartesian;
        p_cartesian[0] = p.bc[0];
        p_cartesian[1] = p.bc[1];
        p_cartesian[2] = p.bc[2];

        int not_face = (int)p.bc[3];

        RowVector4i tet = TT.row(p.cell_id);
        int index = 0;
        for (int i = 0; i < 4; i++) {
            if (tet[i] == not_face) continue;
            face[index++] = tet[i];
        }

        vector<int> key = face;
        sort(key.begin(), key.end());

        assert(out_face_map.find(key) != out_face_map.end() && "out_face_map must contains all out_face");
        assert(p.cell_id == out_face_map[key].second);
        p.cell_id = out_face_map[key].first;

        Vector3i tri = TF.row(p.cell_id);
        RowVector3d bc;
        barycentric_coordinates(p_cartesian, V.row(tri[0]), V.row(tri[1]), V.row(tri[2]), bc);

        assert(bc.minCoeff() > -BARYCENTRIC_BOUND);

        p.flag = FACE;
        p.bc.resize(1, 3);
        p.bc << bc;
    }
public:
    MeshTrace<Scalar, 4> tet_trace;
    MeshTrace<Scalar, 3> tri_trace;
    MatrixXd tri_normal;
    MatrixXd per_vertex_normals;

public:
    const Eigen::MatrixX<Scalar> V;
    const Eigen::MatrixXi TT;
    const Eigen::MatrixXi TF;
    const Eigen::MatrixX<Scalar> FF0T;
    const Eigen::MatrixX<Scalar> FF1T;
    const Eigen::MatrixX<Scalar> FF2T;
    const Eigen::MatrixX<Scalar> FF0F;
    const Eigen::MatrixX<Scalar> FF1F;

    std::map<vector<int>, vector<int>> adjacent_map;

    std::map<vector<int>, pair<int, int>> out_face_map; // map[triangle] = [id_in_TF, id_in_TT}
    std::vector<bool> surface_point;
    int anchor_cnt;


    MeshTraceManager(
        const Eigen::MatrixX<Scalar> &_V,
        const Eigen::MatrixXi &_TT,
        const Eigen::MatrixXi &_TF,
        const Eigen::MatrixX<Scalar> &_FF0T,
        const Eigen::MatrixX<Scalar> &_FF1T,
        const Eigen::MatrixX<Scalar> &_FF2T,
        const Eigen::MatrixX<Scalar> &_FF0F,
        const Eigen::MatrixX<Scalar> &_FF1F,
        std::map<vector<int>, pair<int, int>> &_out_face_map,
        std::vector<bool> & _surface_point
    ) : tet_trace(_V, _TT, _FF0T, _FF1T, _FF2T, _surface_point), tri_trace(_V, _TF, _FF0F, _FF1F),
        V(_V), TT(_TT), TF(_TF), FF0T(_FF0T), FF1T(_FF1T), FF2T(_FF2T),
        FF0F(_FF0F), FF1F(_FF1F), out_face_map(_out_face_map), surface_point(_surface_point) {
        igl::per_face_normals(V, TF, tri_normal);
        igl::per_vertex_normals(V, TF, per_vertex_normals);
        anchor_cnt = 0;
        for(int i = 0; i < surface_point.size(); i++) {
            if (surface_point[i]) anchor_cnt++;
        }
    }

    bool particle_insert_and_delete(vector<ParticleD> &P, double n_r, double lattice) {
        using kdtree_t =
        KDTreeVectorOfVectorsAdaptor<std::vector<Eigen::Vector3d>, double>;
        std::cout << "Executing particle deletion scheme" << endl;

        const vector<Vector3d> BCC = {
                Vector3d(lattice, 0, 0), Vector3d(-lattice, 0, 0), Vector3d(0, lattice, 0),
                Vector3d(0, -lattice, 0), Vector3d(0, 0, lattice), Vector3d(0, 0, -lattice)
        };

        // todo local lattice

        double d_particle = 0.5 * lattice;
        double d_edge = 0.75 * lattice;
        double d_quad = 0.9 * lattice;
        double d_hex = 0.9 * lattice;

        MatrixXd points_mat;
        vector<Vector3d> points_vec(P.size());

        to_cartesian(P, points_mat); // todo directly get vec
        for (int i = 0; i < points_mat.rows(); i++) {
            Vector3d temp = points_mat.row(i);
            points_vec[i] = temp;
        }

        kdtree_t kdtree(3, points_vec, 25);
        nanoflann::SearchParams params;
        params.sorted = true;

        vector<ParticleD> new_particles;
        vector<bool> removed(P.size());
        std::fill(removed.begin(), removed.end(), false);
        for (int i = 0; i < P.size(); i++) {
            if (P[i].flag == POINT) {
                new_particles.push_back(P[i]);
                continue;
            }
            Vector3d p = points_vec[i];
            vector<double> D;

            std::vector<std::pair<size_t, double>> ret_matches;
            kdtree.index->radiusSearch(&p[0], n_r * n_r, ret_matches, params);

            for (int j = 0; j < ret_matches.size() && D.size() <= 8; j++) {
                if (!removed[ret_matches[j].first] && ret_matches[j].first != i) {
                    D.push_back(sqrt(ret_matches[j].second));
                }
            }

            bool delete_condition = false;
            if (!D.empty()) {
                vector<double> sum(8, 0);

                for (int j = 0; j < 8 && j < D.size(); ++j) {
                    if (j == 0) {
                        sum[j] = D[j];
                    } else {
                        sum[j] = D[j] + sum[j - 1];
                    }
                }

                if (sum[0] < d_particle) delete_condition = true;
                if (D.size() >= 2 && 0.5 * sum[1] < d_edge) delete_condition = true;
                if (D.size() >= 4 && 0.25 * sum[3] < d_quad) delete_condition = true;
                if (D.size() >= 8 && 0.125 * sum[7] < d_hex) delete_condition = true;
            }

            if (delete_condition) {
                removed[i] = true;
            } else {
                new_particles.push_back(P[i]);
            }
        }

        int d_cnt = P.size() - new_particles.size();
        cout << "Deleted " << d_cnt << " particles" << endl;
        cout << "Executing particle insertion" << endl;

        vector<ParticleD> candidates;
        MatrixXd candi_mat;
        for (int i = 0; i < new_particles.size(); i++) {
            if (i % 1000 == 0) cout << "Particle Insert: Visiting " << i << "/" <<  new_particles.size() << " particles" << endl;
            if (new_particles[i].flag != FREE) continue;
            Vector3d ff[3];
            ff[0] = FF0T.row(new_particles[i].cell_id);
            ff[1] = FF1T.row(new_particles[i].cell_id);
            ff[2] = FF2T.row(new_particles[i].cell_id);

            for (int t = -1, k = 0; k < 6; k++) {
                t*=-1;
                ParticleD candidate = new_particles[i];//todo local lattice
                if(tracing(candidate, t * lattice * ff[k/2]) && candidate.flag == FREE) {
                    Vector3d v;
                    to_cartesian(candidate, v);
                    std::vector<std::pair<size_t, double>> ret_matches;
                    kdtree.index->radiusSearch(&v[0], n_r * n_r, ret_matches, params);

                    vector<double> D;
                    for (int j = 0; j < ret_matches.size() && D.size() <= 8; j++) {
                        if (!removed[ret_matches[j].first] && ret_matches[j].first != i) {
                            D.push_back(sqrt(ret_matches[j].second));
                        }
                    }

                    if (!D.empty() && D[0] < 0.7 * lattice) continue;

//                    double min_d = 1e20;
//                    for (auto & candi : candidates) {
//                        Vector3d c_v;
//                        to_cartesian(candi, c_v);
//                        min_d = min(min_d, (c_v - v).norm());
//                    }

                    double min_d = 1e20;

                    if (candi_mat.rows() != 0) {
                        Eigen::RowVectorXd row = v.transpose(); // the row you want to replicate
                        Eigen::MatrixXd Mat(row.colwise().replicate(candi_mat.rows()));
                        Mat = Mat - candi_mat;
                        auto arr = Mat.array() * Mat.array();
                        auto col = arr.col(0) + arr.col(1) + arr.col(2);
                        min_d = sqrt(col.minCoeff());
                    }


                    if (min_d < 0.8 * lattice) continue;

                    if (on_boundary(candidate, 0.5 * lattice)) continue;

                    candi_mat.conservativeResize(candi_mat.rows() + 1, 3);
                    candi_mat.row(candi_mat.rows() - 1) = v.transpose();
                    candidates.push_back(candidate);
                    new_particles.push_back(candidate);
                }
            }
        }

        cout << "deleted " << d_cnt << " particles." << "Inserted " << candidates.size() << " particles." << endl;
        cout << "-------------------------------" << endl;

        P = new_particles;
        return candidates.size() == d_cnt;
    }

    bool tracing(Particle<> &p, const Vector3d &v) {
        auto foo = [](const Particle<>& target, double stepLen, double total) {
//             cout << "Current step length: " << stepLen << " Total traveled length: " << total << endl;
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
            double re = direction_cmp0.normalized().dot(ff0.normalized());
            re = std::max(re, -1.);
            re = std::min(re, 1.);
            direct(0, 0) = acos(re);
            if (ff0.cross(direction_cmp0).dot(ff2) < 0) {
                direct(0, 0) = 2 * igl::PI - direct(0, 0);
            }
            re = direction_cmp0.normalized().dot(v.normalized());
            re = std::max(re, -1.);
            re = std::min(re, 1.);
            direct(1, 0) = acos(re);
            if (v.dot(ff2) < 0) {
                direct(1, 0) = -direct(1, 0);
            }
            bool res = tet_trace.tracing(v.norm(), p, direct, foo);
            if (p.flag == FACE) {
                convert_to_face(p);
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
    
    void to_cartesian(const vector<ParticleD> &A, MatrixXd &P) {
        P.resize(A.size(), 3);
        for (int i = 0; i < A.size(); i++) {
            if (A[i].flag == FREE) {
                RowVector4i tet_i = TT.row(A[i].cell_id);
                RowVector4d bc = A[i].bc;
                P.row(i) = bc[0] * V.row(tet_i[0]) + bc[1] * V.row(tet_i[1]) + bc[2] * V.row(tet_i[2]) + bc[3] * V.row(tet_i[3]);
            } else if (A[i].flag == FACE) {
                RowVector3i tri_i = TF.row(A[i].cell_id);
                RowVector3d bc = A[i].bc;
                P.row(i) = bc[0] * V.row(tri_i[0]) + bc[1] * V.row(tri_i[1]) + bc[2] * V.row(tri_i[2]);
            } else if (A[i].flag == EDGE) {
                //TODO 1d support
                RowVector4i tet_i = TT.row(A[i].cell_id);
                RowVector4d bc = A[i].bc;
                P.row(i) = bc[0] * V.row(tet_i[0]) + bc[1] * V.row(tet_i[1]) + bc[2] * V.row(tet_i[2]) + bc[3] * V.row(tet_i[3]);
            }
            else if (A[i].flag == POINT) {
                if (A[i].bc.cols() == 4) {
                    RowVector4i tet_i = TT.row(A[i].cell_id);
                    RowVector4d bc = A[i].bc;
                    P.row(i) = bc[0] * V.row(tet_i[0]) + bc[1] * V.row(tet_i[1]) + bc[2] * V.row(tet_i[2]) + bc[3] * V.row(tet_i[3]);
                } else {
                    RowVector3i tri_i = TF.row(A[i].cell_id);
                    RowVector3d bc = A[i].bc;
                    P.row(i) = bc[0] * V.row(tri_i[0]) + bc[1] * V.row(tri_i[1]) + bc[2] * V.row(tri_i[2]);
                }
            } 
        }
    }

    void to_cartesian(const ParticleD &p, Vector3d &v) {
        if (p.flag == FREE) {
            RowVector4i tet_i = TT.row(p.cell_id);
            RowVector4d bc = p.bc;
            v = (bc[0] * V.row(tet_i[0]) + bc[1] * V.row(tet_i[1]) + bc[2] * V.row(tet_i[2]) + bc[3] * V.row(tet_i[3])).transpose();
        } else if (p.flag == FACE) {
            RowVector3i tri_i = TF.row(p.cell_id);
            RowVector3d bc = p.bc;
            v = (bc[0] * V.row(tri_i[0]) + bc[1] * V.row(tri_i[1]) + bc[2] * V.row(tri_i[2])).transpose();
        } else if (p.flag == EDGE) {
            //TODO 1d support
            RowVector4i tet_i = TT.row(p.cell_id);
            RowVector4d bc = p.bc;
            v = (bc[0] * V.row(tet_i[0]) + bc[1] * V.row(tet_i[1]) + bc[2] * V.row(tet_i[2]) + bc[3] * V.row(tet_i[3])).transpose();
        }
        else if (p.flag == POINT) {
            if (p.bc.cols() == 4) {
                RowVector4i tet_i = TT.row(p.cell_id);
                RowVector4d bc = p.bc;
                v = (bc[0] * V.row(tet_i[0]) + bc[1] * V.row(tet_i[1]) + bc[2] * V.row(tet_i[2]) + bc[3] * V.row(tet_i[3])).transpose();
            } else {
                RowVector3i tri_i = TF.row(p.cell_id);
                RowVector3d bc = p.bc;
                v = (bc[0] * V.row(tri_i[0]) + bc[1] * V.row(tri_i[1]) + bc[2] * V.row(tri_i[2])).transpose();
            }
        }
    }

    void const to_cartesian(const vector<ParticleD> &A, MatrixXd &PFix, MatrixXd &PFace, MatrixXd &PFree) {
        PFix.resize(0, 3);
        PFree.resize(0, 3);
        PFace.resize(0, 3);
        for (int i = 0; i < A.size(); i++) {
            if (A[i].flag == FREE) {
                RowVector4i tet_i = TT.row(A[i].cell_id);
                RowVector4d bc = A[i].bc;
                PFree.conservativeResize(PFree.rows() + 1, 3);
                PFree.row(PFree.rows() - 1) = bc[0] * V.row(tet_i[0]) + bc[1] * V.row(tet_i[1]) + bc[2] * V.row(tet_i[2]) + bc[3] * V.row(tet_i[3]);
            } else if (A[i].flag == FACE) {
                RowVector3i tri_i = TF.row(A[i].cell_id);
                RowVector3d bc = A[i].bc;
                PFace.conservativeResize(PFace.rows() + 1, 3);
                PFace.row(PFace.rows() - 1) = bc[0] * V.row(tri_i[0]) + bc[1] * V.row(tri_i[1]) + bc[2] * V.row(tri_i[2]);
            } else if (A[i].flag == EDGE) {
                // PFree.row(i) = A[i].bc;//TODO 1d support
            } 
            else if (A[i].flag == POINT) {
                RowVector3i tri_i = TF.row(A[i].cell_id);
                RowVector3d bc = A[i].bc;
                PFix.conservativeResize(PFix.rows() + 1, 3);
                PFix.row(PFix.rows() - 1) = bc[0] * V.row(tri_i[0]) + bc[1] * V.row(tri_i[1]) + bc[2] * V.row(tri_i[2]);
            } 
        }
    }

    int remove_boundary(vector<ParticleD> &PV, double threshold, bool remove_anchor = false) {
        vector<ParticleD> new_P;
        for (int i = 0; i < PV.size(); i++) {
            if (PV[i].flag == POINT) {
                if (remove_anchor) continue;
                if (i >= anchor_cnt) continue;
            }

            if (PV[i].flag == FACE || PV[i].flag == EDGE) continue;
            if (PV[i].flag == FREE) {
                if (on_boundary(PV[i], threshold)) continue;
            }

            new_P.push_back(PV[i]);
        }
        int result = PV.size() - new_P.size();
        PV = new_P;
        return result;
    }

    bool on_boundary(ParticleD p, double threshold) {
        if (p.flag != FREE) return true;
        Vector4i tet = TT.row(p.cell_id);
        Vector3d vp;
        to_cartesian(p, vp);
        for (int i = 0; i < 4; i++) { // face
            vector<int> key(3);
            key[0] = tet[vertex_of_face[i][0]];
            key[1] = tet[vertex_of_face[i][1]];
            key[2] = tet[vertex_of_face[i][2]];
            sort(key.begin(), key.end());
            Vector3d v3 = V.row(tet[i]);
            if (out_face_map.find(key) != out_face_map.end()) {
                Vector3d v0 = V.row(key[0]);
                Vector3d v1 = V.row(key[1]);
                Vector3d v2 = V.row(key[2]);
                double area = (v0 - v1).cross(v2 - v1).norm() * 0.5;
                double volume = abs(igl::volume_single(v0, v1, v2, v3)) * p.bc[i];
                double distance = 3 * volume / area;
                assert(!isnan(distance));
                if (distance < threshold) return true;
            }
            if (surface_point[tet[i]]) {
                Vector3d n = per_vertex_normals.row(tet[i]);
                n.normalize();
                if (abs((vp - v3).dot(n)) < threshold) return true;
            }
        }

        return false;
    }
};
} // namespace MESHTRACE
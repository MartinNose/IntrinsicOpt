#pragma once
#include <iostream>
#include "MeshTrace/trace.h"
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/barycentric_coordinates.h>
#include <igl/doublearea.h>
#include <igl/AABB.h>
#include <igl/in_element.h>
#include "Eigen/Core"
#include <cmath>
#include <algorithm>
#include <map>
#include <vector>
#include <set>
#include <ctime>
#include <functional>
#include "KDTreeVectorOfVectorsAdaptor.h"

namespace MESHTRACE {

inline int vertex_of_face[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
inline int edge_of_triangle[3][2] = {{1, 2}, {0, 2}, {0, 1}};
inline Vector3d BCC[6] = {
    Vector3d(1., 0, 0), Vector3d(-1., 0, 0), Vector3d(0, 1., 0),
    Vector3d(0, -1., 0), Vector3d(0, 0, 1.), Vector3d(0, 0, -1.)
};

using namespace Eigen;
template<typename Scalar = double>
class MeshTraceManager {
private:
    igl::AABB<Eigen::MatrixXd, 3> tree;
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
    
    MatrixXd frame_field;
    std::map<vector<int>, vector<int>> adjacent_map;

    std::map<vector<int>, pair<int, int>> out_face_map; // map[triangle] = [id_in_TF, id_in_TT}
    std::vector<bool> surface_point; // TODO change to set for performance improvement
    std::map<vector<int>, tuple<int, int, bool>> edge_tri_map; // map[{vi,vj} = {face_i, face_j, if_sharp}
    std::map<int, std::vector<int>> surface_point_adj_faces;
    std::map<int, std::set<int>> surface_point_adj_sharp_edges;
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
        std::vector<bool> _surface_point, double sharp_threshold = 0.64278760968 // cos(50deg)
    ) : tet_trace(_V, _TT, _TF, _FF0T, _FF1T, _FF2T, _surface_point),
        tri_trace(_V, _TF, _FF0F, _FF1F),
        V(_V), TT(_TT), TF(_TF), FF0T(_FF0T), FF1T(_FF1T), FF2T(_FF2T),
        FF0F(_FF0F), FF1F(_FF1F), out_face_map(_out_face_map), surface_point(_surface_point) {
        igl::per_face_normals(V, TF, tri_normal);
        igl::per_vertex_normals(V, TF, per_vertex_normals);
        anchor_cnt = 0;
        for(auto && i : surface_point) {
            if (i) anchor_cnt++; // TODO consider remove
        }
        
        for (int i = 0; i < TF.rows(); i++) {
            Vector3i tri = TF.row(i);
            for (auto & j : edge_of_triangle) {
                vector<int> edge = {tri[j[0]], tri[j[1]]};
                sort(edge.begin(), edge.end());
                if (edge_tri_map.find(edge) == edge_tri_map.end()) {
                    edge_tri_map[edge] = make_tuple(i, i, false);
                } else {
                    assert(get<0>(edge_tri_map[edge]) == get<1>(edge_tri_map[edge]) && "one edge should only be visited twice");
                    get<1>(edge_tri_map[edge]) = i;
                    bool if_sharp = tri_normal.row(get<0>(edge_tri_map[edge])).dot(tri_normal.row(i)) <= sharp_threshold;
                    get<2>(edge_tri_map[edge]) = if_sharp;
                    if (if_sharp) {
                        if (surface_point_adj_sharp_edges.find(edge[0]) != surface_point_adj_sharp_edges.end()) {
                            surface_point_adj_sharp_edges[edge[0]].insert(edge[1]);
                        } else {
                            surface_point_adj_sharp_edges[edge[0]] = {edge[1]};
                        }
                        if (surface_point_adj_sharp_edges.find(edge[1]) != surface_point_adj_sharp_edges.end()) {
                            surface_point_adj_sharp_edges[edge[1]].insert(edge[0]);
                        } else {
                            surface_point_adj_sharp_edges[edge[1]] = {edge[0]};
                        }
                    }
                }
            }
            for (int k = 0; k < 3; k++) {
                if (surface_point_adj_faces.find(tri[k]) != surface_point_adj_faces.end()) {
                    surface_point_adj_faces[tri[k]].push_back(i); 
                } else {
                    surface_point_adj_faces[tri[k]] = {i};
                }
            }
        }
        cout << "building vertex_adj_faces done" << endl;
        cout << "building vertex_adj_sharp_edges done" << endl;
        cout << "building edge_tri_map done" << endl;

        tree.init(V, TT);

        frame_field.resize(3, TT.rows() * 3);
        for (int i = 0; i < TT.rows(); i++) {
            frame_field.col(i * 3 + 0) = FF0T.row(i).transpose();
            frame_field.col(i * 3 + 1) = FF1T.row(i).transpose();
            frame_field.col(i * 3 + 2) = FF2T.row(i).transpose();
        }
        tri_trace.vertex_adj_faces = surface_point_adj_faces;
        tri_trace.vertex_adj_sharp_edges = surface_point_adj_sharp_edges;
        tri_trace.edge_tri_map = edge_tri_map;
        
        tet_trace.vertex_adj_faces = surface_point_adj_faces;
        tet_trace.vertex_adj_sharp_edges = surface_point_adj_sharp_edges;
        tet_trace.edge_tri_map = edge_tri_map;
    }

    int in_element(Vector3d p) {
        VectorXi I;
        igl::in_element(V, TT, p.transpose(), tree, I);
        if (I.size() > 0 && I[0] >= 0) {
            return I[0];
        } else {
            return -1;
        }
    }

    bool particle_insert_and_delete(vector<ParticleD> &P, double n_r, double lattice) {
        using kdtree_t =
        KDTreeVectorOfVectorsAdaptor<std::vector<Eigen::Vector3d>, double>;
        std::cout << "Executing particle deletion scheme" << endl;

        // todo local lattice

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
                int index = ret_matches[j].first;
                if (!removed[index] && index != i && 
                    P[i].flag  >= P[index].flag) {
                    D.push_back(sqrt(ret_matches[j].second));
                }
            }


            double a = FF0T.row(P[i].cell_id).norm();
            double b = FF1T.row(P[i].cell_id).norm();
            double c = FF2T.row(P[i].cell_id).norm();

            double local_lattice = (a + b + c - max(a, max(b, c))) / 2. * lattice;
            double d_particle = 0.5 * local_lattice;
            double d_edge = 0.75 * local_lattice;
            double d_quad = 0.9 * local_lattice;
            double d_hex = 0.9 * local_lattice;

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
            if (i % 5000 == 0) cout << "Particle Insert: Visiting " << i << "/" <<  new_particles.size() << " particles" << endl;
            if (new_particles[i].flag == FREE) {
                Vector3d ff[3];
                ff[0] = FF0T.row(new_particles[i].cell_id);
                ff[1] = FF1T.row(new_particles[i].cell_id);
                ff[2] = FF2T.row(new_particles[i].cell_id);
                double a = ff[0].norm();
                double b = ff[1].norm();
                double c = ff[2].norm();

                double local_lattice = (a + b + c - max(a, max(b, c))) / 2. * lattice;

                double t = -1.;
                for (int k = 0; k < 6; k++) {
                    t*=-1.;
                    ParticleD candidate = new_particles[i];//todo local lattice
                    Vector3d v;
                    to_cartesian(candidate, v);
                    Vector3d c = v + t * local_lattice * ff[k/2];
                    if (ff[k/2].norm() < 0.7) continue;

                    int tet = in_element(c); // use trace when model has pits
                
                    if(tet != -1) {
                        std::vector<std::pair<size_t, double>> ret_matches;
                        kdtree.index->radiusSearch(&c[0], n_r * n_r, ret_matches, params);

                        vector<double> D;
                        for (int j = 0; j < ret_matches.size() && D.size() < 1; j++) {
                            if (!removed[ret_matches[j].first]) {
                                D.push_back(sqrt(ret_matches[j].second));
                            }
                        }

                        if (!D.empty() && D[0] < 0.7 * local_lattice) continue;

                        double min_d = 1e20;
                        if (candi_mat.rows() != 0) {
                            Eigen::RowVectorXd row = c.transpose();
                            Eigen::MatrixXd Mat(row.colwise().replicate(candi_mat.rows()));
                            Mat = Mat - candi_mat;
                            auto arr = Mat.array() * Mat.array();
                            auto col = arr.col(0) + arr.col(1) + arr.col(2);
                            min_d = sqrt(col.minCoeff());
                        }

                        if (min_d < 0.8 * local_lattice) continue;

                        RowVector4d bc;
                        igl::barycentric_coordinates(c.transpose(), 
                            V.row(TT.row(tet)[0]), V.row(TT.row(tet)[1]), V.row(TT.row(tet)[2]), V.row(TT.row(tet)[3]), 
                            bc);
                        if (bc.minCoeff() < -BARYCENTRIC_BOUND) continue;
                        ParticleD inserted(tet, bc, FREE);
                        
                        if (on_surface(inserted, 0.5 * local_lattice)) continue;

                        candi_mat.conservativeResize(candi_mat.rows() + 1, 3);
                        candi_mat.row(candi_mat.rows() - 1) = c.transpose();
                        candidates.push_back(inserted);
                        new_particles.push_back(inserted);
                    }
                }
            } else if (new_particles[i].flag == FACE) {
                Vector3d ff[2];
                ff[0] = FF0F.row(new_particles[i].cell_id);
                ff[1] = FF1F.row(new_particles[i].cell_id);
                double a = ff[0].norm();
                double b = ff[1].norm();

                double local_lattice = (a + b) / 2. * lattice;

                double t = -1.;
                for (int k = 0; k < 4; k++) {
                    t*=-1.;
                    ParticleD candidate = new_particles[i];//todo local lattice
                    tracing(candidate, t * local_lattice * ff[k/2]);
                    
                    Vector3d c;
                    to_cartesian(candidate, c);
                
                    std::vector<std::pair<size_t, double>> ret_matches;
                    kdtree.index->radiusSearch(&c[0], n_r * n_r, ret_matches, params);

                    vector<double> D;
                    for (int j = 0; j < ret_matches.size() && D.size() < 1; j++) {
                        if (!removed[ret_matches[j].first] && 
                            candidate.flag >= P[ret_matches[j].first].flag) {
                            D.push_back(sqrt(ret_matches[j].second));
                        }
                    }

                    if (!D.empty() && D[0] < 0.7 * local_lattice) continue;

                    double min_d = 1e20;
                    if (candi_mat.rows() != 0) {
                        Eigen::RowVectorXd row = c.transpose();
                        Eigen::MatrixXd Mat(row.colwise().replicate(candi_mat.rows()));
                        Mat = Mat - candi_mat;
                        auto arr = Mat.array() * Mat.array();
                        auto col = arr.col(0) + arr.col(1) + arr.col(2);
                        min_d = sqrt(col.minCoeff());
                    }

                    if (min_d < 0.8 * local_lattice) continue;

                    candi_mat.conservativeResize(candi_mat.rows() + 1, 3);
                    candi_mat.row(candi_mat.rows() - 1) = c.transpose();
                    candidates.push_back(candidate);
                    new_particles.push_back(candidate);
                }
            } else if (new_particles[i].flag == EDGE) {
                pair<int, int> edge_index = new_particles[i].get_edge();
                int e_i = edge_index.first;
                int e_j = edge_index.second;

                Vector3d vi = V.row(e_i);
                Vector3d vj = V.row(e_j);

                Vector3d edge_v = (vj - vi).normalized();

                double t = -1.;
                for (int k = 0; k < 1; k++) {
                    t*=-1.;
                    ParticleD candidate = new_particles[i];//todo local lattice
                    tracing(candidate, t * lattice * edge_v);
                    
                    Vector3d c;
                    to_cartesian(candidate, c);
                
                    std::vector<std::pair<size_t, double>> ret_matches;
                    kdtree.index->radiusSearch(&c[0], n_r * n_r, ret_matches, params);

                    vector<double> D;
                    for (int j = 0; j < ret_matches.size() && D.size() < 1; j++) {
                        if (!removed[ret_matches[j].first] && 
                            candidate.flag >= P[ret_matches[j].first].flag) {
                            D.push_back(sqrt(ret_matches[j].second));
                        }
                    }

                    if (!D.empty() && D[0] < 0.7 * lattice) continue;

                    double min_d = 1e20;
                    if (candi_mat.rows() != 0) {
                        Eigen::RowVectorXd row = c.transpose();
                        Eigen::MatrixXd Mat(row.colwise().replicate(candi_mat.rows()));
                        Mat = Mat - candi_mat;
                        auto arr = Mat.array() * Mat.array();
                        auto col = arr.col(0) + arr.col(1) + arr.col(2);
                        min_d = sqrt(col.minCoeff());
                    }

                    if (min_d < 0.8 * lattice) continue;

                    candi_mat.conservativeResize(candi_mat.rows() + 1, 3);
                    candi_mat.row(candi_mat.rows() - 1) = c.transpose();
                    candidates.push_back(candidate);
                    new_particles.push_back(candidate);
                }     
            } else if (new_particles[i].flag == POINT) {
                // do nothing
            }
        }

        cout << "Deleted " << d_cnt << " particles." << "Inserted " << candidates.size() << " particles." << endl;
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
            
            if (p.flag == STEP) {
                p.flag = FACE;
            } else if (p.flag == FACE) {
                convert_to_face(p);
            } else if (p.flag == EDGE) {
                auto [ei, ej] = p.get_edge();
                vector<int> edge = {min(ei, ej), max(ei, ej)};
                assert(edge_tri_map.find(edge) != edge_tri_map.end());
                auto tuple = edge_tri_map[edge];
                if (!get<2>(tuple)) { // not sharp edge, change to surface
                    p.flag = FACE;
                    double t = p.bc[0];
                    p.cell_id = get<0>(tuple); // new tri
                    p.bc.resize(1, 3);
                    p.bc = RowVector3d::Zero();
                    for (int i = 0; i < 3; i++) {
                        if (TF.row(p.cell_id)[i] == ei) {
                            p.bc[i] = t;
                        } else if (TF.row(p.cell_id)[i] == ej) {
                            p.bc[i] = 1 - t;
                        } else {
                            p.bc[i] = 0;
                        }
                    }
                    assert(p.bc[0] + p.bc[1] + p.bc[2] == 1);
                }
            }
            return res; // EDGE will be p {cell_id: one face,
        } else if (p.flag == FACE) {
            Vector3d ff0 = FF0F.row(p.cell_id);
            Vector3d ff1 = FF1F.row(p.cell_id);
            Vector3d n = tri_normal.row(p.cell_id);
            n.normalize();
            Vector3d new_v = v - v.dot(n) * n;

            if (new_v.norm() < 1e-6) return true;

            double cos_theta = new_v.normalized().dot(ff0);
            cos_theta = min(1., cos_theta);
            cos_theta = max(-1., cos_theta);
            double theta = acos(cos_theta);

            if (ff0.cross(new_v).dot(n) < -BARYCENTRIC_BOUND) {
                theta = 2 * igl::PI - theta;
            }
            assert(abs(p.bc[0] + p.bc[1] + p.bc[2] - 1.) < BARYCENTRIC_BOUND);

            return tri_trace.tracing(new_v.norm(), p, theta, foo);
        } else if (p.flag == EDGE) {
            double distance = v.norm();
            Vector3d displacement = v;

            while(distance > 1e-6) {
                Vector3d start = p.get_edge_coord(V);

                pair<int, int> edge_index = p.get_edge();
                int e_i = edge_index.first;
                int e_j = edge_index.second;

                Vector3d vi = V.row(e_i);
                Vector3d vj = V.row(e_j);

                Vector3d edge_v = (vj - vi).normalized();

                distance = displacement.dot(edge_v);

                if (distance < 0) {
                    e_i = edge_index.second;
                    e_j = edge_index.first;
                    vi = V.row(e_i);
                    vj = V.row(e_j);
                    edge_v *= -1.;
                    distance = -distance;
                }

                if (distance < 1e-6) break;
                displacement = distance * edge_v;

                // towards e_j
                if (distance < (vj - start).norm()) { // inside the edge
                    start += displacement;
                    double t = (start - vi).norm() / (vj - vi).norm();
                    p.bc[0] = 1 - t;
                    p.bc[1] = t;
                    p.bc[2] = e_i;
                    p.bc[3] = e_j;
                    distance  -= distance;
                } else {
                    assert(surface_point_adj_sharp_edges.find(e_j) != surface_point_adj_sharp_edges.end());
                    assert(surface_point_adj_sharp_edges[e_j].size() != 0);
                    int e_next;
                    if (surface_point_adj_sharp_edges[e_j].size() == 2) {
                        for (int iter : surface_point_adj_sharp_edges[e_j]) {
                            if (iter == e_i) continue;
                            e_next = iter;
                        }
                        // p with edge e_j e_next
                        distance -= (vj - start).norm();
                        displacement = distance * edge_v;
                        p.bc[0] = 1.;
                        p.bc[1] = 0.;
                        p.bc[2] = (double) e_j;
                        p.bc[3] = (double) e_next;
                    } else { // make p fixed point
                        p.flag = POINT;
                        p.bc.resize(1, 3);
                        p.bc = vj.transpose();
                        return true;
                    }
                }
            }
            return true;
        } else if (p.flag == POINT) {
            return true;
        }
        assert(false && "Illegal Flag");
        cerr << "tracing: failed" << endl;
        exit(-1);
    }

    inline void project(const Particle<>& p, Vector3d &v) {
        if (p.flag == FREE) {
            return;
        } else if (p.flag == FACE) {
            Vector3d n = tri_normal.row(p.cell_id);
            v = v - v.dot(n)*n;
            return;
        } else if (p.flag == EDGE) { // TODO change to explicitly show edge index
            auto [ei, ej] = p.get_edge();
            Vector3d e = V.row(ei) - V.row(ej);
            v = v.dot(e) * e;
            return; 
        } else if (p.flag == POINT) {
            v = Vector3d::Zero();
            return;
        }
    }
    
    void to_cartesian(const vector<ParticleD> &A, MatrixXd &P) { // TODO Upate
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
                Vector3d v = A[i].get_edge_coord(V);
                P.row(i) = v.transpose();
            }
            else if (A[i].flag == POINT) {
                P.row(i) = V.row(A[i].cell_id);
            } 
        }
    }

    void to_cartesian(const ParticleD &p, Vector3d &v) { // TODO Updating
        if (p.flag == FREE) {
            RowVector4i tet_i = TT.row(p.cell_id);
            RowVector4d bc = p.bc;
            v = (bc[0] * V.row(tet_i[0]) + bc[1] * V.row(tet_i[1]) + bc[2] * V.row(tet_i[2]) + bc[3] * V.row(tet_i[3])).transpose();
        } else if (p.flag == FACE) {
            RowVector3i tri_i = TF.row(p.cell_id);
            RowVector3d bc = p.bc;
            v = (bc[0] * V.row(tri_i[0]) + bc[1] * V.row(tri_i[1]) + bc[2] * V.row(tri_i[2])).transpose();
        } else if (p.flag == EDGE) {
            v = p.get_edge_coord(V);
        }
        else if (p.flag == POINT) {
            v = p.bc;
        }
    }

    void const to_cartesian(const vector<ParticleD> &A, MatrixXd &PFix, MatrixXd &PEdge, MatrixXd &PFace, MatrixXd &PFree) {
        PFix.resize(0, 3);
        PFree.resize(0, 3);
        PEdge.resize(0, 3);
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
                Vector3d v = A[i].get_edge_coord(V);
                PEdge.row(i) = v.transpose();
            }
            else if (A[i].flag == POINT) {
                Vector3d v = A[i].bc;
                PFix.conservativeResize(PFix.rows() + 1, 3);
                PFix.row(PFix.rows() - 1) = v.transpose();
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
                if (on_surface(PV[i], threshold)) continue;
            }

            new_P.push_back(PV[i]);
        }
        int result = PV.size() - new_P.size();
        PV = new_P;
        return result;
    }

    bool on_surface(ParticleD p, double threshold) {
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
        // TODO Check n-ring neighbour if average length of this cell is less than the threshold;
        return false;
    }

    bool on_edge(ParticleD p, double threshold) {
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
        // TODO Check n-ring neighbour if average length of this cell is less than the threshold;

        return false;
    }

    bool on_corner(ParticleD p, double threshold) {
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
        // TODO Check n-ring neighbour if average length of this cell is less than the threshold;

        return false;
    }

    size_t get_tet_id(const ParticleD &p) {
        if (p.flag == FREE) {
            return p.cell_id;
        }
        if (p.flag == FACE) {
            vector<int> face {TF.row(p.cell_id)[0], TF.row(p.cell_id)[1], TF.row(p.cell_id)[2]};
            sort(face.begin(), face.end());
            assert(out_face_map.find(face) != out_face_map.end());
            return out_face_map[face].second;
        }
        if (p.flag == EDGE) {
            auto [ei, ej] = p.get_edge();
            vector<int> edge = {min(ei, ej), max(ei, ej)};
            int face_i = get<0>(edge_tri_map[edge]);
            vector<int> face {TF.row(face_i)[0], TF.row(face_i)[1], TF.row(face_i)[2]};
            sort(face.begin(), face.end());
            assert(out_face_map.find(face) != out_face_map.end());
            return out_face_map[face].second;
        }
        if (p.flag == POINT) {
            Vector3i tri = TF.row(surface_point_adj_faces[p.cell_id][0]);
            vector<int> face {tri[0], tri[1], tri[2]};
            sort(face.begin(), face.end());
            assert(out_face_map.find(face) != out_face_map.end());
            return out_face_map[face].second;
        }
        cerr << "invalid flag" << endl;
        exit(-1);
    }

    void get_frame(int tet, Matrix3d &ff) {
        ff.col(0) = FF0T.row(tet).transpose();
        ff.col(1) = FF1T.row(tet).transpose();
        ff.col(2) = FF2T.row(tet).transpose();
    }

    void get_mid_frame(const Vector3d &vi, const Vector3d &vj, Matrix3d &ff) {
        int tet = in_element((vi + vj) * 0.5);
        if (tet != -1) {
            ff = frame_field.block(0, tet * 3, 3, 3);
            return;
        } else {
            ff = MatrixXd::Identity(3, 3);
        }
    }

    int get_tri_id(const ParticleD &p) {
       if (p.flag == FACE) {
            return p.cell_id;
        }
        if (p.flag == EDGE) {
            auto [ei, ej] = p.get_edge();
            vector<int> edge = {min(ei, ej), max(ei, ej)};
            int face_i = get<0>(edge_tri_map[edge]);
            return face_i;
        }
        if (p.flag == POINT) {
            return surface_point_adj_faces[p.cell_id][0];
        }
        cerr << "invalid flag" << endl;
        exit(-1);
    } 

    void get_mid_frame(const ParticleD &pi, const ParticleD &pj, Matrix3d &ff) {
        ParticleD tmp = pi;
        Vector3d vi, vj;

        to_cartesian(pi, vi);
        to_cartesian(pj, vj);
        
        tracing(tmp, 0.5 * (vj - vi));
        get_frame(get_tet_id(tmp), ff);
    }

};
} // namespace MESHTRACE
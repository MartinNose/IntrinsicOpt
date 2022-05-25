#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <cmath>
#include <igl/segment_segment_intersect.h>
#include <igl/per_face_normals.h>
#include <igl/barycentric_coordinates.h>
#include <vector>
#include <tuple>
#include <map>
#include <set>
#include <utility>

#define EPSILON 1e-13
#define BARYCENTRIC_BOUND 1e-10
//  `DIM` = 3: tri mesh
//  `DIM` = 4: tet mesh

using namespace Eigen;
using namespace std;

namespace MESHTRACE {

enum FLAG { STEP, POINT, EDGE, FACE, FREE };

// Usage: MeshTrace<double, 3>::tracing(...);
template <typename Scalar = double>
struct Particle {
    size_t cell_id;
    Eigen::RowVectorXd bc;
    FLAG flag;
    // omit the rest common functions like constructor, assignment operator,
    // etc. Please complete.
    Particle() {
        cell_id = -1;
        bc.resize(1, 3);
        flag = FREE;
    }
    Particle(size_t _cell_id, const RowVector4<Scalar> &_bc, FLAG _flag = FREE) : cell_id(_cell_id), flag(_flag) {
        bc.resize(1, _bc.cols());
        bc.row(0) << _bc;
    }
    Particle(size_t _cell_id, const RowVector3<Scalar> &_bc, FLAG _flag = FACE) : cell_id(_cell_id), flag(_flag) {
        bc.resize(1, _bc.cols());
        bc.row(0) << _bc;
        flag = _flag;
    }
    pair<int, int> get_edge() const {
        assert(flag == EDGE && "get_edge: particle must on an edge");
        int ei = int(bc[2]);
        int ej = int(bc[3]);
        return make_pair(ei, ej);
    }
    Vector3d get_edge_coord(const MatrixXd& V) const {
        Vector3d vi = V.row(int(bc[2]));
        Vector3d vj = V.row(int(bc[3]));
        return bc[0] * vi + bc[1] * vj;
    }
    Vector3d get_vertex() const {
        assert(flag == POINT && "get_vertex: particle must be a fixed point");
        Vector3d v;
        v[0] = bc[0];
        v[1] = bc[1];
        v[2] = bc[2];
        return v;
    }
};

template <typename DerivedB>
Particle<> create_particle(const size_t cell_id, const Eigen::MatrixBase <DerivedB> bc, const FLAG flag = FREE) {
    if (bc.cols() == 4) {
        RowVector4d tmp;
        tmp << bc;
        return Particle<>(cell_id, tmp, flag);
    } else if (bc.cols() == 3) {
        RowVector3d tmp;
        tmp << bc;
        return Particle<>(cell_id, tmp, flag);
    } else {
        cerr << "create_particled: Unsupported input" << endl;
        exit(-1);
    }
}

using ParticleD = Particle<double>;

template<typename Scalar, int DIM = 4>
class MeshTrace {
private:
    static_assert(DIM == 3 || DIM == 4, "DIM must be 3 or 4");
    // using Direction =
    // typename std::conditional<DIM == 3,
    //         Scalar,                     // theta
    //         Eigen::Matrix <Scalar, 2, 1> // pitch, yaw
    // >::type;

    using VecDIM = Eigen::Matrix<Scalar, DIM, 1>;
    using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

    const Eigen::MatrixX<Scalar> &V;
    const Eigen::MatrixXi &T;
    const Eigen::MatrixX<Scalar> &FF0;
    const Eigen::MatrixX<Scalar> &FF1;
    const Eigen::MatrixX<Scalar> &FF2;
    Eigen::MatrixX<Scalar> N;
public:
    const Eigen::MatrixXi& TF;
    std::map<int, std::vector<int>> vertex_adj_faces;
    std::map<int, std::set<int>> vertex_adj_sharp_edges;
    std::map<vector<int>, tuple<int, int, bool>> edge_tri_map; // map[{vi,vj} = {face_i, face_j, if_sharp}

    vector<int> findAdjacentFace(int face_id, vector<int> index) {
        if (index.size() == 2) {            
            if (index[0] > index[1]) {
                int tmp = index[0]; index[0] = index[1]; index[1] = tmp;
            }
            assert(edge_tri_map.find(index) != edge_tri_map.end());
            auto t = edge_tri_map[index];
            if (get<2>(t)) {
                return vector<int>{-1};
            } else {
                return vector<int>{get<0>(t) == face_id ? get<1>(t) : get<0>(t)};
            }
        } else if (index.size() == 1) {
            assert(vertex_adj_faces.find(index[0]) != vertex_adj_faces.end());
            return vertex_adj_faces[index[0]];
        }
        assert(false && "error request");
    }

    vector<int> findAdjacentCell(int cell_id, vector<int> index) {
        assert(!index.empty());
        sort(index.begin(), index.end());
        vector<int> result;
        if (index.size() == 3) {
            if (face_adjacent_map.find(index) == face_adjacent_map.end()) {
                cerr << "Adjacent cell record not found in face_adjacent_map" << endl;
                cerr << "Cell id: " << cell_id << " Face index: " << index[0] << index[1] << index[2] << endl;
                exit(-1);
            }
            auto pair = face_adjacent_map[index];
            if (pair.second == -1) return result;
            if (pair.first == cell_id) {
                result.push_back(pair.second);
                return result;
            } else if (pair.second == cell_id) {
                result.push_back(pair.first);
                return result;
            } else {
                cerr << "Error in face_adjacent_map" << endl;
                cerr << "Cell id: " << cell_id << " Face index: " << index[0] << index[1] << index[2] << endl;
                cerr << "Pair: " << pair.first << pair.second << endl;
                exit(-1);
            }
        } else if (index.size() == 2) { // edge
            auto key = make_pair(index[0], index[1]);
            if (edge_adjacent_map.find(key) == edge_adjacent_map.end()) {
                cerr << "Adjacent cell record not found in edge_adjacent_map" << endl;
                cerr << "Cell id: " << cell_id << " edge index: " << index[0] << index[1] << endl;
                exit(-1);
            }
            return edge_adjacent_map[key];
        } else if (index.size() == 1) { // point
            return vertice_adjacent_map[index[0]];
        } else {
            cerr << "findAdjacentCell: Invalid index" << endl;
            exit(-1);
        }
    }
    
    void find_cut(const Vector3d &s, const Vector3d &e, 
                const Vector3d &v0, const Vector3d &v1, const Vector3d &v2,
                Vector3d& cut) {
        double u, t;
        bool if_cut;

        Vector3d target;
        Vector3d n = (v0 - v1).cross(v0 - v2).normalized();
        Vector3d displacement = e - s;
        target = s + (displacement - displacement.dot(n) * n) * 1000;

        if (igl::segment_segment_intersect(s, target - s, v0, v1 - v0, u, t, 1e-10)) {
            cut = v0 + t * (v1 - v0);
            return;
        } else if (igl::segment_segment_intersect(s, target - s, v1, v2 - v1, u, t, 1e-10)) {
            cut = v1 + t * (v2 - v1);
            return;
        } else if (igl::segment_segment_intersect(s, target - s, v2, v0 - v2, u, t, 1e-10)) {
            cut = v2 + t * (v0 - v2);
            return;
        }
        assert(false && "cut not found");
    }

    // Get the theta of a and b with sign
    Scalar get_theta(Vec3 a, Vec3 b, Vec3 n) {
        a.normalize();
        b.normalize();
        n.normalize();
        Scalar res = acos(a.dot(b));
        if (a.cross(b).dot(n) > 0) {
            res = -res;
        }
        return res;
    }

    // trace a particle Scalar, in the mesh with the given
    // direction, `callback` is called when the particleScalar, 
    // crosses a cell boundary or the travel is done.
    //

//    template <typename F>
//    inline bool traceStep(Scalar distance, Direction direction, unsigned int cell_id, const unsigned edgeIndex[2],
//                          const Scalar CutCoord1D, Scalar total, F &callback);

    bool find_joint(const Vector3d &start, const Vector3d &end, const Vector3d &f0, const Vector3d &f1, const Vector3d &f2, Vector3d &joint) {
        Vector3d face_n = (f0 - f1).cross(f0-f2).normalized();
        if (face_n.dot(start - f0) * face_n.dot(end - f0) < BARYCENTRIC_BOUND || // Ch eck if start and end are at different side of face
            face_n.dot(start - f1) * face_n.dot(end - f1) < BARYCENTRIC_BOUND ||
            face_n.dot(start - f2) * face_n.dot(end - f2) < BARYCENTRIC_BOUND) {
            double ds = abs(face_n.dot(start - f0));
            double de = abs(face_n.dot(end - f0));
            Vector3d joint_t;
            if (ds > de) joint_t = start + (ds / (ds + de)) * (end - start);
            else joint_t = end + (de / (ds + de)) * (start - end);
            RowVector3d bc;
            igl::barycentric_coordinates(joint_t.transpose(), f0.transpose(), f1.transpose(), f2.transpose(), bc);
            if (bc.minCoeff() >= -BARYCENTRIC_BOUND) {
                joint = joint_t;
                return true;
            }
        }
        return false;
    }

    int compute_new_p(int cell, int f0, int f1, int f2, const Vector3d & joint, RowVector4d & bc) {
        vector<int> face_i {f0, f1, f2};
        sort(face_i.begin(), face_i.end());
        vector<int> candidates = findAdjacentCell(cell, face_i);
        RowVector4<Scalar> bc_joint_tet;
        if (candidates.empty()) {
            return -1;
        }
        int new_cell = candidates[0];

        Vector4i new_t = T.row(new_cell);
        Vec3 v0_new = V.row(new_t[0]);
        Vec3 v1_new = V.row(new_t[1]);
        Vec3 v2_new = V.row(new_t[2]);
        Vec3 v3_new = V.row(new_t[3]);
        RowVector4d temp_bc;
        igl::barycentric_coordinates(joint.transpose(),
                                     v0_new.transpose(), v1_new.transpose(),
                                     v2_new.transpose(), v3_new.transpose(),
                                     temp_bc);
        assert(temp_bc.minCoeff() > -BARYCENTRIC_BOUND);
        bc = temp_bc;
        return new_cell;
    }

    int compute_new_p(int cell, int f0, int f1, const Vector3d &end_point) {
        vector<int> edge_i {f0, f1};
        sort(edge_i.begin(), edge_i.end());
        vector<int> candidates = findAdjacentCell(cell, edge_i);
        int new_cell = -1;

        Vector3d edge[2] = {V.row(f0), V.row(f1)};
        for (int i = 0; i < candidates.size(); i++) {
            if (candidates[i] == cell) continue;
            Vector4i candi_tet = T.row(candidates[i]);
            Vec3 extra[2];
            int extra_cnt = 0;
            for (int j = 0; j < 4; j++) {
                if (candi_tet[j] == f0 || candi_tet[j] == f1) continue;
                extra[extra_cnt++] = V.row(candi_tet[j]);
            }
            assert(extra_cnt == 2);
            Vector3d n0 = (edge[1] - edge[0]).cross(extra[0] - edge[1]);
            Vector3d n1 = (edge[1] - edge[0]).cross(extra[1] - edge[1]);
            Vector3d n2 = (edge[1] - edge[0]).cross(end_point - edge[1]);

            if (n2.cross(n0).dot(n1.cross(n2)) > 0) {
                return candidates[i];
            }
        }
        return -1;
    }

    void get_direction(int tet, const Matrix<Scalar, 2, 1> &direction, Vector3d &direct) {
        Vec3 ff0 = FF0.row(tet).transpose();
        Vec3 ff1 = FF1.row(tet).transpose();
        Vec3 ff2 = FF2.row(tet).transpose();

        Matrix3<Scalar> A = AngleAxis<Scalar>(direction[0], ff2.normalized()).toRotationMatrix();

        direct = (A * ff0).normalized();
        
        A = AngleAxis<Scalar>(direction[1], direct.cross(ff2).normalized()).toRotationMatrix();

        direct = (A * direct).normalized();
    }


public:
    std::map<vector<int>, std::pair<int, int>> face_adjacent_map; // vi vj vk -> tet1 tet2
    std::map<std::pair<int, int>, vector<int>> edge_adjacent_map; // ei ej -> tet1, 2 ....
    std::vector<vector<int>> vertice_adjacent_map; // vi -> tet ....
    std::vector<bool> surface_point; // if ith point in V is on the surface

    template <typename F>
    inline bool traceStep(Scalar distance, int last_cell, Particle<Scalar> &start, double direction, Scalar total, Vector3<Scalar> ff, F &callback) {
        if (start.bc.minCoeff() < -BARYCENTRIC_BOUND) {
            cerr << "Invalid bc" << start.bc << endl;
            assert(start.bc.minCoeff() >= -BARYCENTRIC_BOUND);
        };
        int cur_cell = start.cell_id;
        if (distance < 1e-6) return true;
        Eigen::Matrix<int, 3, 1> cell_i = T.row(start.cell_id);
        Eigen::Matrix<Scalar, 3, 3> Cell;
        
        for (int i = 0; i < 3; i++) {
            Cell.col(i) = V.row(cell_i[i]);
        }

        Eigen::Matrix<Scalar, 1, 3> BC {start.bc};

        // Computing the local coordinate of the displacement
        Vec3 alpha = ff;
        Vec3 beta = FF1.row(start.cell_id);

        if (alpha.cross(beta).norm() < EPSILON) {
            beta = FF0.row(start.cell_id);
            if (alpha.cross(beta).norm() < EPSILON) {
                std::cerr << "Illegal Frame Field at " << start.cell_id <<": \n" << FF0.row(start.cell_id) << "\n" << FF1.row(start.cell_id) << std::endl;
                return false;
            }
        }

        Vec3 e0 = (Cell.col(1) - Cell.col(0));
        Vec3 e1 = (Cell.col(2) - Cell.col(1));

        if (alpha.cross(beta).dot(e0.cross(e1)) < 0) {
            beta = -beta;
        }

        Vec3 normal = N.row(start.cell_id).normalized();

        Matrix<Scalar, 3, 3> A = AngleAxis<Scalar>(direction, normal).toRotationMatrix();

        Vec3 direct = (A * alpha).normalized();

        Vec3 displacement = distance * direct;

        Vec3 startPoint = Cell * BC.transpose();
        Vec3 endPoint = startPoint + displacement;

        RowVector3d endPointB;
        igl::barycentric_coordinates(endPoint.transpose(), V.row(cell_i[0]), V.row(cell_i[1]), V.row(cell_i[2]), endPointB);

        // Get the coefficient of the local coordinate of the target point
        Scalar b0(endPointB(0));
        Scalar b1(endPointB(1));
        Scalar b2(endPointB(2));

        Matrix<double, Dynamic, 3> temp(1,3);

        if (endPointB.minCoeff() >= -BARYCENTRIC_BOUND) { // the target point is inside the triangle
            start.bc << endPointB;
            callback(start, distance, total + distance);
            return true;
        }
        
        int edges[3][2]{ {1, 2}, {2, 0}, {0, 1} };

        vector<int> neg_idx;
        vector<int> pos_idx;
        for (int i = 0; i < 3; i++) {
            if(endPointB(i) < -BARYCENTRIC_BOUND) neg_idx.push_back(i);
            else pos_idx.push_back(i);
        }

        if (neg_idx.size() == 1) { // joint is at the edge
            vector<int> edge = {cell_i[pos_idx[0]], cell_i[pos_idx[1]]};
            vector<int> res = findAdjacentFace(start.cell_id, edge);
            if (res[0] == -1) { // SHARP EDGE
                double u, t;
                Vector3d e0 = V.row(edge[0]);
                Vector3d e1 = V.row(edge[1]);
                assert(igl::segment_segment_intersect(startPoint, endPoint - startPoint, e0, e1 - e0, u, t, 1e-10));
                start.bc.resize(1, 4);
                start.bc[0] = 1 - t;
                start.bc[1] = t;
                start.bc[2] = (double) edge[0];
                start.bc[3] = (double) edge[1];
                start.flag = EDGE;
                return true; // sharp edge
            } else {
                Vector3d v0 = V.row(cell_i[pos_idx[0]]);
                Vector3d v1 = V.row(cell_i[pos_idx[1]]);
                double u, t;
                assert(igl::segment_segment_intersect(startPoint, endPoint - startPoint, v0, v1 - v0, u, t, 1e-10));
                Vector3d joint = v0 + t * (v1 - v0);
                start.cell_id = res[0];
                Vector3i new_tri = T.row(res[0]);
                RowVector3d newbc;
                for (int i = 0; i < 3; i++) {
                    if (new_tri[i] == cell_i[pos_idx[0]]) newbc[i] = 1 - t;
                    else if (new_tri[i] == cell_i[pos_idx[1]]) newbc[i] = t; 
                    else newbc[i] = 0;
                }
                start.bc = newbc;

                Vector3d n = N.row(start.cell_id);
                ff.normalize();
                ff = ff - ff.dot(n) * n;

                Vec3 newFF[4] = {FF0.row(start.cell_id) ,FF1.row(start.cell_id)};
                newFF[2] = -newFF[0];
                newFF[3] = -newFF[1];

                Scalar max = -1.;
                Vector3d new_ff;
                for (auto & curff : newFF) {
                    Scalar cos = ff.dot(curff.normalized());
                    if (cos > max) {
                        max = cos;
                        new_ff = curff;
                    }
                }
                // cut on edge
                if (start.cell_id == last_cell) {
                    start.bc = RowVector3d(0., 0., 0.);
                    int joint_idx;
                    if ((v1 - v0).dot(endPoint - startPoint) > 0) {
                        joint_idx = pos_idx[1];
                    } else {
                        joint_idx = pos_idx[0];
                    }
                    start.bc[joint_idx] = 1.;
                    return traceStep(distance - (V.row(cell_i[joint_idx]) - startPoint.transpose()).norm(), cur_cell, start, direction, total, ff, callback);
                }
                return traceStep(distance - (joint - startPoint).norm(), cur_cell, start, direction, total, new_ff, callback);;
            }
        } else if (neg_idx.size() == 2) { // crossing vertex
            Vector3d v = V.row(cell_i[pos_idx[0]]);
            if (((endPoint - startPoint).normalized()).cross((v - startPoint).normalized()).norm() < 1e-6) {
                int sharp_cnt = vertex_adj_sharp_edges[cell_i[pos_idx[0]]].size();
                if (sharp_cnt == 1 || sharp_cnt >= 3) { // encountering vertex
                    start.bc.resize(1, 3);
                    start.bc = V.row(cell_i[pos_idx[0]]);
                    start.flag = POINT;
                    start.cell_id = cell_i[pos_idx[0]];
                    return true;
                } else if (sharp_cnt == 2) {
                    start.bc.resize(1, 4);
                    start.bc[0] = 1.;
                    start.bc[1] = 0;
                    start.bc[2] = (double) cell_i[pos_idx[0]];
                    start.bc[3] = (double) *(vertex_adj_sharp_edges[cell_i[pos_idx[0]]].begin());
                    start.flag = EDGE;
                    return true;
                }
                vector<int> candi = findAdjacentFace(start.cell_id, vector<int>{cell_i[pos_idx[0]]});
                for (int j = 0; j < candi.size(); j++) {
                    vector<Vector3d> e;
                    int id;
                    for (int k = 0; k < 3; k++) {
                        if (cell_i[pos_idx[0]] != T.row(candi[j])[k]) {
                            e.push_back(V.row(T.row(candi[j])[k]));
                        }
                        else id = k;
                    }

                    Vector3d n = N.row(candi[j]).normalized();
                    Vector3d origin = endPoint - v;
                    origin = origin - origin.dot(n) * n;;

                    if ((origin.cross(e[0] - v)).dot((e[1] - v).cross(origin)) < -BARYCENTRIC_BOUND) continue;
                    start.cell_id = candi[j];
                    start.bc = RowVector3d::Zero();
                    start.bc[id] = 1.;

                    ff.normalize();
                    ff = ff - ff.dot(n) * n;

                    Vec3 newFF[4] = {FF0.row(start.cell_id) ,FF1.row(start.cell_id)};
                    newFF[2] = -newFF[0];
                    newFF[3] = -newFF[1];

                    Scalar max = -1.;
                    Vector3d new_ff;
                    for (auto & curff : newFF) {
                        Scalar cos = ff.dot(curff.normalized());
                        if (cos > max) {
                            max = cos;
                            new_ff = curff;
                        }
                    }
                    //crossing vertex
                    if (start.cell_id == last_cell) {
                        start.cell_id = cell_i[pos_idx[0]];
                        start.bc = v.transpose();
                        start.flag = POINT;
                        return true;
                    }
                    return traceStep(distance - (endPoint - v).norm(), cur_cell, start, direction, total, new_ff, callback);
                }
            } else {
                Vector3d v = V.row(cell_i[pos_idx[0]]);
                for (int i = 0; i < 2; i++) {
                    Vector3d v1 = V.row(cell_i[neg_idx[i]]);
                    vector<int> edge = {cell_i[pos_idx[0]], cell_i[neg_idx[i]]};
                    double u, t;
                    if (!igl::segment_segment_intersect(startPoint, endPoint - startPoint, v, v1 - v, u, t, 1e-10)) continue;
                    
                    vector<int> res = findAdjacentFace(start.cell_id, edge);
                    if (res[0] == -1) { // SHARP EDGE
                        double u, t;
                        assert(igl::segment_segment_intersect(startPoint, endPoint - startPoint, v, v1 - v, u, t, 1e-10));
                        start.bc.resize(1, 4);
                        start.bc[0] = 1 - t;
                        start.bc[1] = t;
                        start.bc[2] = (double) cell_i[pos_idx[0]];
                        start.bc[3] = (double) cell_i[neg_idx[i]];
                        start.flag = EDGE;
                        // sharp edge
                        return true;
                    } else if (res[0] == last_cell) { // near edge case
                        start.bc = RowVector3d(0., 0., 0.);
                        start.bc[pos_idx[0]] = 1.;
                        return traceStep(distance - (v - startPoint).norm(), cur_cell, start, direction, total, ff, callback);
                    } else {
                        start.cell_id = res[0];
                        Vector3i new_tri = T.row(res[0]);
                        RowVector3d newbc;
                        for (int j = 0; j < 3; j++) {
                            if (new_tri[j] == cell_i[pos_idx[0]]) newbc[j] = 1 - t;
                            else if (new_tri[j] == cell_i[neg_idx[i]]) newbc[j] = t; 
                            else newbc[j] = 0;
                        }
                        start.bc = newbc;

                        Vector3d n = N.row(start.cell_id);
                        ff.normalize();
                        ff = ff - ff.dot(n) * n;

                        Vec3 newFF[4] = {FF0.row(start.cell_id) ,FF1.row(start.cell_id)};
                        newFF[2] = -newFF[0];
                        newFF[3] = -newFF[1];

                        Vector3d new_ff;
                        Scalar max = -1.;
                        for (auto & curff : newFF) {
                            Scalar cos = ff.dot(curff.normalized());
                            if (cos > max) {
                                max = cos;
                                new_ff = curff;
                            }
                        }
                        // crossing edge
                        return traceStep(distance - (endPoint - v).norm(), cur_cell, start, direction, total, new_ff, callback);
                    } 
                }
            }
        }
        assert(false && "impossible situation!");
    }

    Matrix3d last_ff;

    template <typename F>
    inline bool traceStep(Scalar distance, int last_cell, Particle<Scalar> &start, Matrix <Scalar, 2, 1> direction, Scalar total, F &callback) {
        if (start.bc.minCoeff() < -BARYCENTRIC_BOUND) {
            cerr << "Invalid bc" << start.bc << endl;
            assert(start.bc.minCoeff() >= -BARYCENTRIC_BOUND);
        };
        if (distance < 1e-6) return true;
        int cur_cell = start.cell_id;
        Matrix3<Scalar> ff;
        ff.col(0) << FF0.row(start.cell_id)[0], FF0.row(start.cell_id)[1], FF0.row(start.cell_id)[2];
        ff.col(1) << FF1.row(start.cell_id)[0], FF1.row(start.cell_id)[1], FF1.row(start.cell_id)[2];
        ff.col(2) << FF2.row(start.cell_id)[0], FF2.row(start.cell_id)[1], FF2.row(start.cell_id)[2];

        if (last_cell != -1) {
            if (last_ff.col(0).dot(ff.col(0)) < 0) ff.col(0) = -ff.col(0);
            if (last_ff.col(1).dot(ff.col(1)) < 0) ff.col(1) = -ff.col(1);
            if (last_ff.col(2).dot(ff.col(2)) < 0) ff.col(2) = -ff.col(2);
        } else {
            last_ff = ff;
        }

        Eigen::Matrix<int, 4, 1> cell_i = T.row(start.cell_id);

        Vec3 v[4];
        for (int i = 0; i < 4; i++) {
            v[i] = V.row(cell_i[i]);
        }

        Eigen::Matrix<Scalar, 1, DIM> BC;
        BC << start.bc;

        // Computing the local coordinate of the displacement
        Vec3 ff0 = ff.col(0);
        Vec3 ff1 = ff.col(1);
        Vec3 ff2 = ff.col(2);

        Matrix3<Scalar> A = AngleAxis<Scalar>(direction(0, 0), ff2.normalized()).toRotationMatrix();

        Vec3 direct = (A * ff0).normalized();
        
        A = AngleAxis<Scalar>(direction(1, 0), direct.cross(ff2).normalized()).toRotationMatrix();

        direct = (A * direct).normalized();

        Vec3 displacement = distance * direct;

        //cout << "d_trans: " << displacement.transpose() << endl;

        Vec3 startPoint = BC[0] * v[0] +  BC[1] * v[1] +  BC[2] * v[2] +  BC[3] * v[3];
        Vec3 end_point = startPoint + displacement;

        RowVector4<Scalar> endPointB;
        igl::barycentric_coordinates(end_point.transpose(), v[0].transpose(), v[1].transpose(), v[2].transpose(), v[3].transpose(), endPointB);

        // Get the coefficient of the local coordinate of the target point
        double b0(endPointB(0));
        double b1(endPointB(1));
        double b2(endPointB(2));
        double b3(endPointB(3));

        vector<int> neg_eb_idx;
        vector<int> pos_eb_idx;

        for (int i = 0; i < 4; i++) {
            if (endPointB(i) < 0) neg_eb_idx.push_back(i);
            else pos_eb_idx.push_back(i);
        }

        sort(pos_eb_idx.begin(), pos_eb_idx.end());
        sort(neg_eb_idx.begin(), neg_eb_idx.begin());

        int new_cell;
        vector<int> candidates;
        if (neg_eb_idx.empty()) { // the target point is inside the triangle
            start.bc = endPointB;
            callback(start, distance, total + distance);
            return true;
        } else if (neg_eb_idx.size() == 1) { // joint is on the face
            Vec3 vs[3] = {v[pos_eb_idx[0]], v[pos_eb_idx[1]], v[pos_eb_idx[2]]};
            vector<int> face_i ={cell_i[pos_eb_idx[0]], cell_i[pos_eb_idx[1]], cell_i[pos_eb_idx[2]]};

            Vec3 face_n = (vs[0] - vs[1]).cross(vs[0]-vs[2]).normalized();
            Vector3d joint;
            bool found = find_joint(startPoint, end_point, vs[0], vs[1], vs[2], joint);

            assert(found && "joint has to be on the face");

            RowVector4<Scalar> bc_joint_tet;

            new_cell = compute_new_p(start.cell_id,
                                     cell_i[pos_eb_idx[0]], cell_i[pos_eb_idx[1]], cell_i[pos_eb_idx[2]],
                                     joint, bc_joint_tet);
            if (new_cell == -1) {
                igl::barycentric_coordinates(joint.transpose(), v[0].transpose(), v[1].transpose(), v[2].transpose(), v[3].transpose(), bc_joint_tet);
                start.bc[0] = joint[0];
                start.bc[1] = joint[1];
                start.bc[2] = joint[2];
                start.bc[3] = (double) cell_i[neg_eb_idx[0]];
                start.flag = FACE;

                callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
                return true;
            } else if (new_cell == last_cell) {
                Vector3d cut;
                find_cut(startPoint, end_point, vs[0], vs[1], vs[2], cut);
                RowVector4d new_bc;
                igl::barycentric_coordinates(cut.transpose(), v[0].transpose(), v[1].transpose(), v[2].transpose(), v[3].transpose(), new_bc);
                start.bc = new_bc;
                return traceStep(distance - (cut - startPoint).norm(), last_cell, start, direction, total, callback);
            }
            assert(bc_joint_tet.minCoeff() > -BARYCENTRIC_BOUND && "joint on face");
            start.cell_id = new_cell;
            start.bc.row(0) << bc_joint_tet;
            callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
            double cur_step = (joint - startPoint).norm();
            if (cur_step < 1e-6) return true;
            return traceStep(distance - (joint - startPoint).norm(), cur_cell, start, direction, total, callback);
        } else if (neg_eb_idx.size() == 2) { // endpoint is on the other side of the edge
            vector<int> face_i ={cell_i[pos_eb_idx[0]], cell_i[pos_eb_idx[1]]};
            Vec3 edge[2] = {v[pos_eb_idx[0]], v[pos_eb_idx[1]]};
            // if joint on the edge

            Vector3d face_joint_1, face_joint_2;
            bool found_1, found_2;
            found_1 = find_joint(startPoint, end_point,
                                 v[pos_eb_idx[0]], v[pos_eb_idx[1]], v[neg_eb_idx[0]],
                                 face_joint_1);
            found_2 = find_joint(startPoint, end_point,
                                 v[pos_eb_idx[0]], v[pos_eb_idx[1]], v[neg_eb_idx[1]],
                                 face_joint_2);

            RowVector4<Scalar> bc_joint_tet;
            if (found_1 && found_2) {
                double u, t;
                if (igl::segment_segment_intersect(startPoint, end_point - startPoint, edge[0], edge[1] - edge[0], u, t, BARYCENTRIC_BOUND)) { // joint is on the edge
                    Vector3d joint = edge[0] + t * (edge[1] - edge[0]);

                    new_cell = compute_new_p(start.cell_id,
                                             cell_i[pos_eb_idx[0]], cell_i[pos_eb_idx[1]],
                                             end_point);

                    RowVector4d new_bc;
                    if (new_cell == -1) {
                        igl::barycentric_coordinates(joint.transpose(), v[0].transpose(), v[1].transpose(), v[2].transpose(), v[3].transpose(), new_bc);
                        start.bc[0] = 1 - t;
                        start.bc[1] = t;
                        start.bc[2] = (double)cell_i[pos_eb_idx[0]];
                        start.bc[3] = (double)cell_i[pos_eb_idx[1]];
                        start.flag = EDGE;
                        callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
                        return true;
                    } else if (new_cell == last_cell) {
                        RowVector4d new_bc = RowVector4d::Zero();
                        if ((end_point - startPoint).dot(edge[1] - edge[0]) > 0) {
                            new_bc[pos_eb_idx[1]] = 1.;
                            joint = edge[1];
                        } else {
                            new_bc[pos_eb_idx[0]] = 1.;
                            joint = edge[0];
                        }
                        start.bc = new_bc;
                        return traceStep(distance - (joint - startPoint).norm(), last_cell, start, direction, total, callback); 
                    } else {
                        Vector4i new_t = T.row(new_cell);
                        Vec3 v0_new = V.row(new_t[0]); Vec3 v1_new = V.row(new_t[1]);
                        Vec3 v2_new = V.row(new_t[2]); Vec3 v3_new = V.row(new_t[3]);
                        igl::barycentric_coordinates(joint.transpose(), v0_new.transpose(), v1_new.transpose(), v2_new.transpose(), v3_new.transpose(), new_bc);
                        assert(new_bc.minCoeff() > -BARYCENTRIC_BOUND);
                        start.cell_id = new_cell;
                        start.bc.row(0) << new_bc;
                        callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
                        return traceStep(distance - (joint - startPoint).norm(), cur_cell, start, direction, total, callback);
                   }
                }
            }


            if (found_1 != found_2) { // joint on face
                new_cell = compute_new_p(start.cell_id,
                                         cell_i[pos_eb_idx[0]],
                                         cell_i[pos_eb_idx[1]],
                                         found_1 ? cell_i[neg_eb_idx[0]] : cell_i[neg_eb_idx[1]],
                                         found_1 ? face_joint_1 : face_joint_2,
                                         bc_joint_tet);
                Vector3d joint = (found_1) ? face_joint_1 : face_joint_2;
                if (new_cell == -1) {
                    igl::barycentric_coordinates(joint.transpose(), v[0].transpose(), v[1].transpose(), v[2].transpose(), v[3].transpose(), bc_joint_tet);
                    start.bc[0] = joint[0];
                    start.bc[1] = joint[1];
                    start.bc[2] = joint[2];
                    start.bc[3] = (double) cell_i[neg_eb_idx[found_1 ? 1 : 0]];

                    start.flag = FACE;

                    callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
                    return true;
                } else if (new_cell == last_cell) {
                    Vector3d cut;
                    int third_v = found_1 ? neg_eb_idx[0] : neg_eb_idx[1];
                    find_cut(startPoint, end_point, v[pos_eb_idx[0]], v[pos_eb_idx[1]], v[third_v], cut);
                    RowVector4d new_bc;
                    igl::barycentric_coordinates(cut.transpose(), v[0].transpose(), v[1].transpose(), v[2].transpose(), v[3].transpose(), new_bc);
                    start.bc = new_bc;
                    return traceStep(distance - (cut - startPoint).norm(), last_cell, start, direction, total, callback);
                }
                if (bc_joint_tet.minCoeff() > -BARYCENTRIC_BOUND) {
                    Vector3d new_direct;
                    get_direction(new_cell, direction, new_direct);
                    Vector3d face_n = (v[pos_eb_idx[0]] - joint).cross(v[pos_eb_idx[1]] - joint).normalized();
                    if (direct.dot(face_n) * new_direct.dot(face_n) < 0) {
                        callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
                        return true;
                    }
                    
                    start.cell_id = new_cell;
                    start.bc.row(0) << bc_joint_tet;
                    callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
                    return traceStep(distance - (joint - startPoint).norm(), cur_cell, start, direction, total, callback);
                }
            }
            assert(!found_1 && !found_2);
            assert(false && "edge condition wrong");
        } else if (neg_eb_idx.size() == 3) { // joint is on the vertex
            vector<int> face_i ={cell_i[pos_eb_idx[0]]};
            Vector3d vertex = V.row(face_i[0]);

            Vector3d ev = vertex - end_point;
            Vector3d vs = startPoint - vertex;
            Vector3d es = startPoint - end_point;

            double dist;

            if (ev.norm() > vs.norm()) {
                dist = ev.cross(es).squaredNorm() / es.squaredNorm();
            } else {
                dist = vs.cross(es).squaredNorm() / es.squaredNorm();
            }
            if (dist <= BARYCENTRIC_BOUND * BARYCENTRIC_BOUND) { // joint is on the vertex
                candidates = findAdjacentCell(start.cell_id, face_i);
                new_cell = -1;

                int joint_idx = cell_i[pos_eb_idx[0]];
                Vec3 joint = v[pos_eb_idx[0]];

                if (surface_point[joint_idx]) {
                    RowVector4d new_bc = RowVector4d::Zero();
                    new_bc[pos_eb_idx[0]] = 1.;

                    int face = vertex_adj_faces[joint_idx][0];

                    start.cell_id = face;
                    start.bc.resize(1, 3);
                    start.bc = RowVector3d::Zero();

                    for (int i = 0; i < 3; i++) {
                        if (TF.row(face)[i] == joint_idx) {
                            start.bc[i] = 1.;
                        }
                    }
                    start.flag = STEP;
                    callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
                    return true;
                }

                RowVector4d temp_bc;
                for (int i = 0; i < candidates.size(); i++) {
                    if (candidates[i] == start.cell_id) continue;

                    Vec3 new_v[4];
                    Vector4i candi_tet = T.row(candidates[i]);
                    int new_cell_v_idx = -1;
                    for (int j = 0; j < 4; j++) {
                        if (candi_tet[j] == joint_idx) {
                            new_cell_v_idx = j;
                        }
                        new_v[j] = V.row(candi_tet[j]);
                    }
                    assert(new_cell_v_idx != -1);

                    igl::barycentric_coordinates(end_point.transpose(), new_v[0].transpose(), new_v[1].transpose(), new_v[2].transpose(), new_v[3].transpose(), temp_bc);

                    bool ifnan = false;
                    int neg_count = 0;
                    for (int j = 0; j < 4; j++) {
                        if (isnan(temp_bc[j])) {
                            ifnan = true;
                            break;
                        }
                        if (temp_bc[j] < 0) neg_count++;
                    }
                    if (ifnan) continue;

                    if (neg_count == 0 || (neg_count == 1 && temp_bc[new_cell_v_idx] < 0)) {
                        igl::barycentric_coordinates(joint.transpose(), new_v[0].transpose(), new_v[1].transpose(), new_v[2].transpose(), new_v[3].transpose(), temp_bc);
                        assert(temp_bc.minCoeff() > -BARYCENTRIC_BOUND);
                        start.cell_id = candidates[i];
                        start.bc << temp_bc;
                        callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
                        return traceStep(distance - (joint - startPoint).norm(), cur_cell, start, direction, total, callback);
                    }
                }
            }

            // joint is on the face adjacent to v

            for (int i = 0; i < 2; i++) {
                for (int j = i + 1; j < 3; j++) {
                    Vector3d joint;
                    Vector3d face0 = V.row(cell_i[neg_eb_idx[i]]);
                    Vector3d face1 = V.row(cell_i[neg_eb_idx[j]]);

                    bool found = find_joint(startPoint, end_point, vertex, face0, face1, joint);
                    if (!found) continue;

                    RowVector4<Scalar> bc_joint_tet;

                    new_cell = compute_new_p(start.cell_id,
                                             cell_i[pos_eb_idx[0]], cell_i[neg_eb_idx[i]], cell_i[neg_eb_idx[j]],
                                             joint,
                                             bc_joint_tet);
                    if (new_cell == -1) {
                        igl::barycentric_coordinates(joint.transpose(), v[0].transpose(), v[1].transpose(), v[2].transpose(), v[3].transpose(), bc_joint_tet);
                        start.bc[0] = joint[0];
                        start.bc[1] = joint[1];
                        start.bc[2] = joint[2];
                        int face_index;
                        for (int k = 0; k < 3; ++k) {
                            if (k == i || k == j) continue;
                            face_index = k;
                        }
                        start.bc[3] = (double) cell_i[neg_eb_idx[face_index]];
                        start.flag = FACE;

                        callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
                        return true;
                    }

                    if (new_cell == last_cell) {
                        start.bc = RowVector4d::Zero();
                        start.bc[pos_eb_idx[0]] = 1.;
                        return traceStep(distance - (v[pos_eb_idx[0]] - startPoint).norm(), last_cell, start, direction, total, callback);
                    }

                    start.cell_id = new_cell;
                    start.bc.row(0) << bc_joint_tet;
                    callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
                    return traceStep(distance - (joint - startPoint).norm(), cur_cell, start, direction, total, callback);
                }
            }

        }
        cout << "trace failed " << "start_point: " << startPoint.transpose() << "bc: " << start.bc <<
             " end_point: " << end_point.transpose() << " bc: " << endPointB << endl;
        exit(-1);
    }

    MeshTrace(const Eigen::MatrixX<Scalar> &_V,
        const Eigen::MatrixXi &_T,
        const Eigen::MatrixX<Scalar> &_FF0,
        const Eigen::MatrixX<Scalar> &_FF1
    ): V(_V), T(const_cast<Eigen::MatrixXi &>(_T)), 
        FF0(_FF0), FF1(_FF1), FF2(_FF1), TF(_T) {
            igl::per_face_normals(V, T, N);
    }
    
    MeshTrace(const Eigen::MatrixX<Scalar> &_V, const Eigen::MatrixXi &_T,  const Eigen::MatrixXi &_TF,
              const Eigen::MatrixX<Scalar> &_FF0, const Eigen::MatrixX<Scalar> &_FF1, const Eigen::MatrixX<Scalar> &_FF2,
              vector<bool> _surface_point
              ):
        V(_V), T(_T), TF(_TF), FF0(_FF0), FF1(_FF1), FF2(_FF2),
        surface_point(_surface_point) {
            vertice_adjacent_map.resize(V.rows());
            for (int i = 0; i < T.rows(); i++) {
                Vector4i tet = T.row(i);
                for (int j = 0; j < 4; j++) {
                    vector<int> key {tet[0], tet[1], tet[2], tet[3]};
                    key.erase(key.begin() + j);
                    sort(key.begin(), key.end());
                    if (face_adjacent_map.find(key) != face_adjacent_map.end()) {
                        face_adjacent_map[key].second = i;
                    } else {
                        face_adjacent_map[key] = make_pair(i, -1);
                    }
                    for (int k = j + 1; k < 4; k++) {
                        int e0 = tet[j];
                        int e1 = tet[k];
                        auto edge = make_pair(min(e0, e1), max(e0, e1));
                        if (edge_adjacent_map.find(edge) != edge_adjacent_map.end()) {
                            edge_adjacent_map[edge].push_back(i);
                        } else {
                            edge_adjacent_map[edge] = vector<int> {i};
                        }
                    }
                    vertice_adjacent_map[tet[j]].push_back(i);
                }
            }
        }

    // return `true` if the travel is done,
    // `false` otherwise (e.g. hit on the mesh boundary).
    template <typename F>
//     requires std::is_invocable<F,
//         Particle,Scalar,  // current position
//         double,   // current step traveled length
//         double    // total traveled length
//     > // remove the comment if your compiler support c++20 concepts.
    inline bool tracing(Scalar distance, Particle<double> &start, Matrix <Scalar, 2, 1> direction, F &callback) {
        return traceStep<F>(distance, -1, start, direction, 0, callback);
    }
    template <typename F>
    inline bool tracing(Scalar distance, Particle<double> &start, double direction, F &callback) {
        return traceStep<F>(distance, -1, start, direction, 0, Vector3<Scalar>(FF0.row(start.cell_id)),callback);
    }
};
}

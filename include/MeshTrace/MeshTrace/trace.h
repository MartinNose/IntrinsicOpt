#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <concepts>
#include <iostream>
#include <cmath>
#include <igl/segment_segment_intersect.h>
#include <igl/per_face_normals.h>
#include <igl/barycentric_coordinates.h>
#include <vector>
#include <tuple>
#include <map>
#include <utility>

#define EPSILON 1e-13
#define BARYCENTRIC_BOUND 1e-14
//  `DIM` = 3: tri mesh
//  `DIM` = 4: tet mesh

using namespace Eigen;
using namespace std;

namespace MESHTRACE {

enum FLAG { POINT, STEP, EDGE, FACE, FREE };

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
    const Eigen::MatrixX<Scalar> N;
    std::map<vector<int>, std::pair<int, int>> face_adjacent_map;
    std::map<std::pair<int, int>, vector<int>> edge_adjacent_map;
    std::vector<vector<int>> vertice_adjacent_map;

    bool findAdjacentCell(int cell_id, const int *edge, int *new_cell) {
        for (int i = 0; i < T.rows(); i++) {
            int hit_count = 0;
            for (int j = 0; j < 3; j++) {
                if (T(i, j) == edge[0] || T(i, j) == edge[1]) hit_count++;
            }
            if (hit_count == 2 && i != cell_id) {
                *new_cell = i;
                return true;
            }
        }
        return false;
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

public:
    template <typename F>
    inline bool traceStep(Scalar distance, Particle<Scalar> &start, double direction, Scalar total, Vector3<Scalar> ff, F &callback) {
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

        Vec3 normal = alpha.cross(beta).normalized();

        Vec3 initGlobal = FF0.row(start.cell_id);

        Matrix<Scalar, 3, 3> A = AngleAxis<Scalar>(direction, normal).toRotationMatrix();

        Vec3 direct = (A * alpha).normalized();

        Vec3 displacement = distance * direct;

        Vec3 startPoint = Cell * BC.transpose();
        Vec3 endPoint = startPoint + displacement;

        Vec3 endPointB = Cell.inverse() * endPoint;

        // Get the coefficient of the local coordinate of the target point
        Scalar b0(endPointB(0, 0));
        Scalar b1(endPointB(1, 0));
        Scalar b2(endPointB(2, 0));

        Matrix<double, Dynamic, 3> temp(1,3);

        if (b0 >= 0 && b1 >= 0 && b0 + b1 <= 1) { // the target point is inside the triangle
            start.bc << endPointB.transpose();
            callback(start, distance, total + distance);
            return true;
        } else {
            int edges[3][3]{ {0, 1, 2}, {1, 2, 0}, {2, 0, 1} };

            for (auto & i : edges) {
                if (start.bc(i[2]) == 0) continue;

                double u, t;
                int vi0 = cell_i(i[0]);
                int vi1 = cell_i(i[1]);
                Vec3 v0 = Cell.col(i[0]);
                Vec3 v1 = Cell.col(i[1]);

                if (igl::segment_segment_intersect(startPoint, endPoint - startPoint, v0, v1 - v0, u, t, EPSILON)) {
                    // two segments cross each other

                    if (t < EPSILON) {
                        std::cerr << "Encountering Crossing vertex case" << std::endl;
                        return false;
                    }

                    std:: cout << "u: " << u << " t: " << t << std::endl;

                    int newCellId;

                    int edge_tmp[2] =  {vi0, vi1};
                    if (!findAdjacentCell(start.cell_id, edge_tmp, &newCellId)) {
                        std::cerr << "Encountering not finding adjacent face case" << std::endl;
                        return false;
                    }

                    Eigen::Matrix<int, 3, 1> newcell_i = T.row(newCellId);
                    Eigen::Matrix<Scalar, 1, 3> bc;
                    Eigen::Matrix3<Scalar> newCell;
                    for (int j = 0; j < 3; j++) {
                        int vertexIndex = newcell_i(j);
                        if (vertexIndex == vi0) {
                            bc(0, j) = 1 - t;
                        } else if (vertexIndex == vi1 ) {
                            bc(0, j) = t;
                        } else {
                            bc(0,j) = 0;
                        }
                    }

                    start.cell_id = newCellId;
                    start.bc.row(0) << bc.row(0);
                    Scalar traveledDistance = u * (endPoint - startPoint).norm();

                    std::cout << "*************************" << std::endl;
                    std::cout << "startPoint: " << startPoint.transpose() << std::endl;
                    std::cout << "endPoint: " << (startPoint + u * (endPoint - startPoint)).transpose() << std::endl;

                    callback(start, traveledDistance, total);
                    if (traveledDistance < EPSILON) {
                        return true;
                    }
                    Vec3 edgeDirect = (v1 - v0).normalized();

                    Vec3 newFF[4] = {FF0.row(start.cell_id) ,FF1.row(start.cell_id)};
                    newFF[2] = -newFF[0];
                    newFF[3] = -newFF[1];

                    Scalar min = 3;
                    Vec3 new_ff = newFF[0];
                    Scalar theta_0 = get_theta(ff, edgeDirect, normal);

                    for (auto & j : newFF) {
                        Vec3 cur_ff = j.normalized();
                        Vec3 new_normal = N.row(start.cell_id);
                        Scalar cur_theta = get_theta(cur_ff, edgeDirect, new_normal);

                        if (abs(theta_0 - cur_theta) < min) {
                            min = abs(theta_0 - cur_theta);
                            new_ff = j;
                        }
                    }

                    Matrix<Scalar, Dynamic, 3> new_ff_mark(2, 3);
                    Vec3 barycenter = (Cell.col(0) + Cell.col(1) + Cell.col(2)) / 3;
                    new_ff_mark.row(0) = barycenter;
                    new_ff_mark.row(1) = new_ff;
                    callback(start, traveledDistance, total);
                    return traceStep(distance - traveledDistance, start, direction, total + traveledDistance, new_ff, callback);
                }
            }
            std::cerr << "Error Case" << std::endl;
            return false;
        }
    }

    template <typename F>
    inline bool traceStep(Scalar distance, Particle<Scalar> &start, Matrix <Scalar, 2, 1> direction, Scalar total, F &callback) {
        Matrix3<Scalar> ff;
        ff.col(0) << FF0.row(start.cell_id)[0], FF0.row(start.cell_id)[1], FF0.row(start.cell_id)[2];
        ff.col(1) << FF1.row(start.cell_id)[0], FF1.row(start.cell_id)[1], FF1.row(start.cell_id)[2];
        ff.col(2) << FF2.row(start.cell_id)[0], FF2.row(start.cell_id)[1], FF2.row(start.cell_id)[2];
        
        Eigen::Matrix<int, 4, 1> cell_i = T.row(start.cell_id);

        Vec3 v[4];
        for (int i = 0; i < 4; i++) {
            v[i] = V.row(cell_i[i]);
        }

        Eigen::Matrix<Scalar, 1, DIM> BC;
        BC << start.bc;

        // Computing the local coordinate of the displacement
        Vec3 ff0 = ff.row(0);
        Vec3 ff1 = ff.row(1);
        Vec3 ff2 = ff.row(2);

        Matrix3<Scalar> A = AngleAxis<Scalar>(direction(0, 0), ff2.normalized()).toRotationMatrix();

        Vec3 direct = (A * ff0).normalized();
        
        A = AngleAxis<Scalar>(direction(1, 0), direct.cross(ff2).normalized()).toRotationMatrix();

        direct = (A * direct).normalized();

        Vec3 displacement = distance * direct;

        cout << "d_trans: " << displacement.transpose() << endl;

        Vec3 startPoint = BC[0] * v[0] +  BC[1] * v[1] +  BC[2] * v[2] +  BC[3] * v[3];
        Vec3 endPoint = startPoint + displacement;

        RowVector4<Scalar> endPointB;
        igl::barycentric_coordinates(endPoint.transpose(), v[0].transpose(), v[1].transpose(), v[2].transpose(), v[3].transpose(), endPointB);

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
            
            assert(face_n.dot(startPoint - vs[0]) * face_n.dot(endPoint - vs[0]) < -BARYCENTRIC_BOUND ||
                   face_n.dot(startPoint - vs[1]) * face_n.dot(endPoint - vs[1]) < -BARYCENTRIC_BOUND ||
                   face_n.dot(startPoint - vs[2]) * face_n.dot(endPoint - vs[2]) < -BARYCENTRIC_BOUND);
            
            double ds = abs(face_n.dot(startPoint - vs[0]));
            double de = abs(face_n.dot(endPoint - vs[0]));
            Vec3 joint = startPoint + (ds / (ds + de)) * (endPoint - startPoint);
            RowVector3<Scalar> bc_joint_face;
            igl::barycentric_coordinates(joint.transpose(), vs[0].transpose(), vs[1].transpose(), vs[2].transpose(), bc_joint_face);

            assert(abs(bc_joint_face[0] + bc_joint_face[1] + bc_joint_face[2] - 1) <= BARYCENTRIC_BOUND);
            assert(bc_joint_face[0] >= -BARYCENTRIC_BOUND);
            assert(bc_joint_face[1] >= -BARYCENTRIC_BOUND);
            assert(bc_joint_face[2] >= -BARYCENTRIC_BOUND);

            candidates = findAdjacentCell(start.cell_id, face_i);
            RowVector4<Scalar> bc_joint_tet;
            if (candidates.empty()) {
                igl::barycentric_coordinates(joint.transpose(), v[0].transpose(), v[1].transpose(), v[2].transpose(), v[3].transpose(), bc_joint_tet);
                start.bc << bc_joint_tet;
                start.flag = FACE;

                callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
                return true;
            }
            new_cell = candidates[0];

            Vector4i new_t = T.row(new_cell);
            Vec3 v0_new = V.row(new_t[0]);
            Vec3 v1_new = V.row(new_t[1]);
            Vec3 v2_new = V.row(new_t[2]);
            Vec3 v3_new = V.row(new_t[3]);
            igl::barycentric_coordinates(joint.transpose(), v0_new.transpose(), v1_new.transpose(), v2_new.transpose(), v3_new.transpose(), bc_joint_tet);
            start.cell_id = new_cell;
            start.bc.row(0) << bc_joint_tet;
            callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
            return traceStep(distance - (joint - startPoint).norm(), start, direction, total, callback);
        } else if (neg_eb_idx.size() == 2) { // joint is on the edge
            vector<int> face_i ={cell_i[pos_eb_idx[0]], cell_i[pos_eb_idx[1]]};
            vector<int> candidate_cell = findAdjacentCell(start.cell_id, face_i);
            new_cell = -1;
            Vec3 edge[2] = {v[pos_eb_idx[0]], v[pos_eb_idx[1]]};
            int edge_idx[2] = {cell_i[pos_eb_idx[0]], cell_i[pos_eb_idx[1]]};
            for (int i = 0; i < candidate_cell.size(); i++) {
                if (candidates[i] == start.cell_id) continue;
                Vector4i candi_tet = T.row(candidates[i]);
                Vec3 extra[2];
                int extra_cnt = 0;
                for (int j = 0; j < 4; j++) {
                    if (candi_tet[j] == edge_idx[0] || candi_tet[j] == edge_idx[1]) continue;
                    extra[extra_cnt++] = V.row(candi_tet[i]);
                }
                assert(extra_cnt == 3);
                Vector3d n0 = (edge[1] - edge[0]).cross(extra[0] - edge[1]);
                Vector3d n1 = (edge[1] - edge[0]).cross(extra[1] - edge[1]);
                Vector3d n2 = (edge[1] - edge[0]).cross(endPoint - edge[1]);

                if (n2.cross(n0).dot(n1.cross(n2)) > 0) {
                    new_cell = candidates[i];
                    break;
                }
            }
            // compute joint on edge
            Vector3d n = (edge[1] - edge[0]).cross(endPoint - startPoint).normalized();
            Vector3d nn = n.cross(edge[1] - edge[0]).normalized();
            double de = (endPoint - edge[0]).dot(nn);
            double ds = (startPoint - edge[0]).dot(nn);

            Vector3d joint;
            if (de > ds) {
                joint = endPoint + (de / (de + ds)) * (startPoint - endPoint);
            } else {
                joint = startPoint + (ds / (de + ds)) * (endPoint - startPoint);
            }

            RowVector4d new_bc;
            if (new_cell == -1) {
                igl::barycentric_coordinates(joint.transpose(), v[0].transpose(), v[1].transpose(), v[2].transpose(), v[3].transpose(), new_bc);

                start.bc << new_bc;

                start.flag = EDGE;
                callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
                return true;
                // make joint
            } else {
                Vector4i new_t = T.row(new_cell);
                Vec3 v0_new = V.row(new_t[0]);
                Vec3 v1_new = V.row(new_t[1]);
                Vec3 v2_new = V.row(new_t[2]);
                Vec3 v3_new = V.row(new_t[3]);
                igl::barycentric_coordinates(joint.transpose(), v0_new.transpose(), v1_new.transpose(), v2_new.transpose(), v3_new.transpose(), new_bc);
                start.cell_id = new_cell;
                start.bc.row(0) << new_bc;
                callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
                return traceStep(distance - (joint - startPoint).norm(), start, direction, total, callback);
            }
        } else if (neg_eb_idx.size() == 3) { // joint is on the vertex
            vector<int> face_i ={cell_i[pos_eb_idx[0]]};
            candidates = findAdjacentCell(start.cell_id, face_i);
            new_cell = -1;

            int joint_idx = cell_i[pos_eb_idx[0]];
            Vec3 joint = v[pos_eb_idx[0]];

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

                igl::barycentric_coordinates(endPoint.transpose(), new_v[0].transpose(), new_v[1].transpose(), new_v[2].transpose(), new_v[3].transpose(), temp_bc);

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
                    start.cell_id = candidates[i];
                    start.bc << temp_bc;
                    callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
                    return traceStep(distance - (joint - startPoint).norm(), start, direction, total, callback);
                }
            }

            // joint is the endpoint
            RowVector4d new_bc = RowVector4d::Zero();
            new_bc[pos_eb_idx[0]] = 1.;

            start.bc << new_bc;
            start.flag = POINT;
            callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
            return true;
        } else {
            cout << "wrong case" << endl;
            exit(-1);
        }
    }

    MeshTrace(const Eigen::MatrixX<Scalar> &_V,
              const Eigen::MatrixXi &_T,
              const Eigen::MatrixX<Scalar> &_FF0,
              const Eigen::MatrixX<Scalar> &_FF1
              ): // tri_mesh version
        V(_V),
        T(const_cast<Eigen::MatrixXi &>(_T)),
        FF0(_FF0),
        FF1(_FF1),
        FF2(_FF1) {}
    
        MeshTrace(const Eigen::MatrixX<Scalar> &_V,
              const Eigen::MatrixXi &_T,
              const Eigen::MatrixX<Scalar> &_FF0,
              const Eigen::MatrixX<Scalar> &_FF1,
              const Eigen::MatrixX<Scalar> &_FF2
              ):
        V(_V),
        T(_T),
        FF0(_FF0),
        FF1(_FF1),
        FF2(_FF2) {
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
        return traceStep<F>(distance, start, direction, 0, callback);
    }
    template <typename F>
    inline bool tracing(Scalar distance, Particle<double> &start, double direction, F &callback) {
        return traceStep<F>(distance, start, direction, 0, Vector3<Scalar>(FF0.row(start.cell_id)),callback);

    }
};
}
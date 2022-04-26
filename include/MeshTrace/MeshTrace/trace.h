#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <concepts>
#include <iostream>
#include <math.h>
#include <igl/segment_segment_intersect.h>
#include <igl/per_face_normals.h>
#include <igl/barycentric_coordinates.h>
#include <vector>
#include <tuple>

#define EPSILON 1e-13
#define BARYCENTRIC_BOUND 1e-15
//  `DIM` = 3: tri mesh
//  `DIM` = 4: tet mesh

using namespace Eigen;
using namespace std;


namespace MESHTRACE {

enum FLAG { POINT, STEP, EDGE, FACE, FREE };

//
// Usage: MeshTrace<double, 3>::tracing(...);

template <typename Scalar = double>
struct Particle {
    size_t cell_id;
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> bc;
    FLAG flag;
    // omit the rest common functions like constructor, assignment operator,
    // etc. Please complete.
    Particle(size_t _cell_id, RowVector4<Scalar> &_bc, FLAG _flag = FREE) : cell_id(_cell_id), flag(_flag) {
        bc.resize(1, _bc.cols());
        bc.row(0) << _bc;
    }
    Particle(size_t _cell_id, RowVector3<Scalar> &_bc, FLAG _flag = FACE) : cell_id(_cell_id), flag(_flag) {
        bc.resize(1, _bc.cols());
        bc.row(0) << _bc;
        flag = FREE;
    }
    template <typename DerivedB> static
    Particle<Scalar> create(size_t cell_id, Eigen::MatrixBase <DerivedB> bc, FLAG flag = FREE) {
        VectorXd tmp;
        tmp << bc;
        return Particle(cell_id, tmp, flag);
    }
};

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

    using VecDIM = Eigen::Matrix<Scalar, DIM, 1> ;
    using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &V;
    Eigen::Matrix<int, Eigen::Dynamic, DIM> &T;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &FF0;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &FF1;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &FF2;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 3> N;


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

    bool findAdjacentCell(int cell_id, int f0, int f1, int f2, int *new_cell){
        vector<int> f {f0, f1, f2};
        sort(f.begin(), f.end());
        for (int i = 0; i < T.rows(); i++) {
            Vector4i tv = T.row(i);
            vector<int> t (tv.data(), tv.data() + 4);
            sort(t.begin(), t.end());
            if (includes(t.begin(), t.end(), f.begin(), f.end()) && i != cell_id) {
                *new_cell = i;
                return true;
            }
        }
        return false;
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
        ff.row(0) << FF0.row(start.cell_id);
        ff.row(1) << FF1.row(start.cell_id);
        ff.row(2) << FF2.row(start.cell_id);
        
        Eigen::Matrix<int, 4, 1> cell_i = T.row(start.cell_id);
        Vec3 v0 = V.row(cell_i[0]);
        Vec3 v1 = V.row(cell_i[1]);
        Vec3 v2 = V.row(cell_i[2]);
        Vec3 v3 = V.row(cell_i[3]);

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

        Vec3 startPoint = BC[0] * v0 +  BC[1] * v1 +  BC[2] * v2 +  BC[3] * v3;
        Vec3 endPoint = startPoint + displacement;

        RowVector4<Scalar> endPointB;
        igl::barycentric_coordinates(endPoint.transpose(), v0.transpose(), v1.transpose(), v2.transpose(), v3.transpose(), endPointB);

        // Get the coefficient of the local coordinate of the target point
        double b0(endPointB(0));
        double b1(endPointB(1));
        double b2(endPointB(2));
        double b3(endPointB(3));

        if (b0 >= 0 && b1 >= 0 && b2 >= 0 && b3 >= 0) { // the target point is inside the triangle
            start.bc = endPointB;
            callback(start, distance, total + distance);
            return true;
        } else {
            for (int i = 0; i < 4; i++) {
                Vec3 vs[3];
                int vi[3];
                int t = 0;
                for (int j = 0; j < 4; j++) {
                    if (j == i) continue;
                    vs[t] = V.row(cell_i(j));
                    vi[t] = cell_i(j);
                    t++;
                }
                if (BC[i] < EPSILON) continue;
                Vec3 face_n = (vs[0] - vs[1]).cross(vs[0]-vs[2]).normalized();
                if (face_n.dot(startPoint - vs[0]) * face_n.dot(endPoint - vs[0]) > 0) continue;
                double ds = abs(face_n.dot(startPoint - vs[0]));
                double de = abs(face_n.dot(endPoint - vs[0]));
                Vec3 joint = startPoint + (ds / (ds + de)) * (endPoint - startPoint);
                RowVector3<Scalar> bc_joint_face;
                igl::barycentric_coordinates(joint.transpose(), vs[0].transpose(), vs[1].transpose(), vs[2].transpose(), bc_joint_face);
                if (bc_joint_face.minCoeff() < 0) continue;
                int new_cell;
                RowVector4<Scalar> bc_joint_tet;
                if (!findAdjacentCell(start.cell_id, vi[0], vi[1], vi[2], &new_cell)) {
                    start.bc.resize(1, 3);
                    start.bc.row(0) << bc_joint_face;
                    start.flag = FACE;

                    callback(start, (joint - startPoint).norm(), total + (joint - startPoint).norm());
                    return true;
                }
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
            }
            cerr << "Wrong case" << endl;
            return false; 
        }
    }

    MeshTrace(const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_V,
              const Eigen::Matrix<int, Eigen::Dynamic, 3> &_T,
              const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_FF0,
              const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_FF1
              ):
        V((Eigen::Matrix<double, Eigen::Dynamic, 3> &) _V),
        T(const_cast<Eigen::Matrix<int, Eigen::Dynamic, 3> &>(_T)),
        FF0((Eigen::Matrix<double, Eigen::Dynamic, 3> &) _FF0),
        FF1((Eigen::Matrix<double, Eigen::Dynamic, 3> &) _FF1),
        FF2((Eigen::Matrix<double, Eigen::Dynamic, 3> &) _FF1) {
            igl::per_face_normals(V, T, N);
        }
    
        MeshTrace(const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_V,
              const Eigen::Matrix<int, Eigen::Dynamic, DIM> &_T,
              const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_FF0,
              const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_FF1,
              const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_FF2
              ):
        V((Eigen::Matrix<double, Eigen::Dynamic, 3> &) _V),
        T(const_cast<Eigen::Matrix<int, Eigen::Dynamic, 4> &>(_T)),
        FF0((Eigen::Matrix<double, Eigen::Dynamic, 3> &) _FF0),
        FF1((Eigen::Matrix<double, Eigen::Dynamic, 3> &) _FF1),
        FF2((Eigen::Matrix<double, Eigen::Dynamic, 3> &) _FF2) {}

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

    tuple<Vector3d, Vector3d, Vector3d> get_face(Particle<> p) {
        int face[3];
        int idx = 0;
        int unchosen = -1;
        for (int i = 0; i < 4; i++) {
            if (p.bc[i] < BARYCENTRIC_BOUND) {
                continue;
                unchosen = i;
            }
            face[idx++] = i;
        }
        if (idx == 4 || unchosen == -1) {
            cerr << "logic error: Face particle doesn't meet constraints." << endl;
            cout << p.cell_id << endl; 
            cout << p.bc << endl;
        }
        Vector3d v0 = V.row(T.row(p.cell_id)[face[0]]);
        Vector3d v1 = V.row(T.row(p.cell_id)[face[1]]);
        Vector3d v2 = V.row(T.row(p.cell_id)[face[2]]);
        Vector3d v3 = V.row(T.row(p.cell_id)[face[unchosen]]);
        if ((v1 - v0).cross(v2 - v1).dot(v0 - v3) < 0) {
            return make_tuple(v0, v2, v1);
        } else {
            return make_tuple(v0, v1, v2);
        }
    }

    // Only can called when DIM == 3
    int find_face(Vector3d v0, Vector3d v1, Vector3d v2) {
        // TODO
    }
};
}
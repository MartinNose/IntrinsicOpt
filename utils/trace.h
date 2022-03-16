#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <concepts>
#include <iostream>
#include <math.h>
#include <igl/segment_segment_intersect.h>
#include <igl/per_face_normals.h>

#define EPSILON 0.0000001
//  `DIM` = 3: tri mesh
//  `DIM` = 4: tet mesh

using namespace Eigen;

enum FLAG { POINT, STEP, EDGE };

extern FLAG FLAG;
//
// Usage: MeshTrace<double, 3>::tracing(...);
template<typename Scalar, int DIM = 4>
class MeshTrace {
private:
    static_assert(DIM == 3 || DIM == 4, "DIM must be 3 or 4");
    using Direction =
    typename std::conditional<DIM == 3,
            Scalar,                     // theta
            Eigen::Matrix < Scalar, 3, 1> // phi, theta, psi
    >::type;

    using VecDIM = Eigen::Matrix<Scalar, DIM, 1> ;
    using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;


    Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &V;
    Eigen::Matrix<int, Eigen::Dynamic, DIM> &T;
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

    // trace a particle in the mesh with the given
    // direction, `callback` is called when the particle
    // crosses a cell boundary or the travel is done.
    //


//    template <typename F>
//    inline bool traceStep(Scalar distance, Direction direction, unsigned int cell_id, const unsigned edgeIndex[2],
//                          const Scalar CutCoord1D, Scalar total, F &callback);

public:
    struct Particle {
        size_t cell_id;
        Eigen::Matrix<Scalar, 1, DIM> bc;
        // omit the rest common functions like constructor, assignment operator,
        // etc. Please complete.
        Particle(size_t _cell_id, Eigen::Matrix<Scalar, 1, DIM> _bc) : cell_id(_cell_id), bc(_bc) {}
    };

    template <typename F>
    //     requires std::is_invocable<F,
    //         Particle, // current position
    //         double,   // current step traveled length
    //         double    // total traveled length
    //     > // remove the comment if your compiler support c++20 concepts.
    inline bool traceStep(Scalar distance, const Particle &start, Direction direction, Scalar total, Vector3<Scalar> ff, F &callback) {
        Eigen::Matrix<int, 3, 1> vertexIndices = T.row(start.cell_id);
        Eigen::Matrix<Scalar, 3, DIM> Cell;
        for (int i = 0; i < DIM; i++) {
            Cell.col(i) = V.row(vertexIndices[i]);
        }

        Eigen::Matrix<Scalar, 1, 3> BC {start.bc};

        // Computing the local coordinate of the displacement
        Vec3 alpha = ff;
        Vec3 beta = FF2.row(start.cell_id);

        if (alpha.cross(beta).norm() < EPSILON) {
            beta = FF1.row(start.cell_id);
            if (alpha.cross(beta).norm() < EPSILON) {
                std::cerr << "Illegal Frame Field at " << start.cell_id <<": \n" << FF1.row(start.cell_id) << "\n" << FF2.row(start.cell_id) << std::endl;
                return false;
            }
        }

        Vec3 e0 = (Cell.col(1) - Cell.col(0));
        Vec3 e1 = (Cell.col(2) - Cell.col(1));

        if (alpha.cross(beta).dot(e0.cross(e1)) < 0) {
            beta = -beta;
        }

        Vec3 normal = alpha.cross(beta).normalized();

        Vec3 initGlobal = FF1.row(start.cell_id);

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
            Particle res {start.cell_id, endPointB};
            callback(res, distance, total + distance, STEP, temp);
            return true;
        } else {
            int edges[3][3]{ {0, 1, 2}, {1, 2, 0}, {2, 0, 1} };

            for (auto & i : edges) {
                if (start.bc(i[2]) == 0) continue;

                double u, t;
                int vi0 = vertexIndices(i[0]);
                int vi1 = vertexIndices(i[1]);
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

                    Eigen::Matrix<int, 3, 1> newVertexIndices = T.row(newCellId);
                    Eigen::Matrix<Scalar, 1, DIM> bc;
                    Eigen::Matrix3<Scalar> newCell;
                    for (int j = 0; j < 3; j++) {
                        int vertexIndex = newVertexIndices(j);
                        if (vertexIndex == vi0) {
                            bc(0, j) = 1 - t;
                        } else if (vertexIndex == vi1 ) {
                            bc(0, j) = t;
                        } else {
                            bc(0,j) = 0;
                        }
                    }

                    Particle cut(newCellId, bc);
                    Scalar traveledDistance = u * (endPoint - startPoint).norm();

                    std::cout << "*************************" << std::endl;
                    std::cout << "startPoint: " << startPoint.transpose() << std::endl;
                    std::cout << "endPoint: " << (startPoint + u * (endPoint - startPoint)).transpose() << std::endl;

                    callback(cut, traveledDistance, total, STEP,temp);
                    if (traveledDistance < EPSILON) {
                        return true;
                    }
                    Vec3 edgeDirect = (v1 - v0).normalized();


                    Vec3 newFF[4] = {FF1.row(cut.cell_id) ,FF2.row(cut.cell_id)};
                    newFF[2] = -newFF[0];
                    newFF[3] = -newFF[1];

                    Scalar min = 3;
                    Vec3 new_ff = newFF[0];
                    Scalar theta_0 = get_theta(ff, edgeDirect, normal);

                    if (cut.cell_id == 38811) {
                        std::cout << "debug" << std::endl;
                    }
                    for (auto & j : newFF) {
                        Vec3 cur_ff = j.normalized();
                        Vec3 new_normal = N.row(cut.cell_id);
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
                    callback(cut, traveledDistance, total, EDGE, new_ff_mark);
                    return traceStep(distance - traveledDistance, cut, direction, total + traveledDistance, new_ff, callback);
                }
            }
            std::cerr << "Error Case" << std::endl;
            return false;
        }
    }

MeshTrace(const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_V,
              const Eigen::Matrix<int, Eigen::Dynamic, DIM> &_T,
              const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_FF1,
              const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &_FF2
              ):
        V((Eigen::Matrix<double, Eigen::Dynamic, 3> &) _V),
        T(const_cast<Eigen::Matrix<int, Eigen::Dynamic, 3> &>(_T)),
        FF1((Eigen::Matrix<double, Eigen::Dynamic, 3> &) _FF1),
        FF2((Eigen::Matrix<double, Eigen::Dynamic, 3> &) _FF2)
        {
            igl::per_face_normals(V, T, N);
        }

    // return `true` if the travel is done,
    // `false` otherwise (e.g. hit on the mesh boundary).
    template <typename F>
//     requires std::is_invocable<F,
//         Particle, // current position
//         double,   // current step traveled length
//         double    // total traveled length
//     > // remove the comment if your compiler support c++20 concepts.
    inline bool tracing(Scalar distance, const Particle &start,
            Direction direction, F &callback) {
        return traceStep<F>(distance, start, direction, 0, Vector3<Scalar>(FF1.row(start.cell_id)),callback);
    }
};

#pragma once

#include <map>
#include <vector>
#include <algorithm>

template <typename T>
inline std::pair<T, T> sort2(T a, T b) {
    if (a < b)
        return {a, b};
    else
        return {b, a};
}
template <typename T>
inline std::tuple<T, T, T> sort3(T a, T b, T c) {
    if (a < b && a < c) {
        std::tie(b, c) = sort2(b, c);
        return {a, b, c};
    } else if (b < a && b < c) {
        std::tie(a, c) = sort2(a, c);
        return {b, a, c};
    } else {
        std::tie(a, b) = sort2(a, b);
        return {c, a, b};
    }
}

std::tuple<map<vector<int>, pair<int, int>>, map<vector<int>, int>> get_surface_mesh(const MatrixXd &V, const MatrixXi &TT, MatrixXi &TF) {
    assert(TT.cols() == 4);
    TF.resize(TT.rows(), 3);

    // map a sorted face index to the (tri, tet) it belongs to.
    map<vector<int>, pair<int, int>> out_face_map;
    // map a sorted edge index to the face it belongs to. 
    map<vector<int>, int> sharp_edge_map;

    for (int i = 0; i < TT.rows(); i++) {
        Vector4i tet_i = TT.row(i);
        for(int j = 0; j < 4; j++) {
            vector<int> key {tet_i[0], tet_i[1], tet_i[2], tet_i[3]};
            int unselected = j;
            key.erase(key.begin() + j);
            sort(key.begin(), key.end());
            if (out_face_map.find(key) != out_face_map.end()) {
                out_face_map[key] = make_pair(-1, -1);
            } else {
                out_face_map[key] = make_pair(i, j);
            }
        }
    }

    auto iter = out_face_map.begin();
    while(iter != out_face_map.end()) {
        if (iter->second.first == -1) {
            iter = out_face_map.erase(iter);
        } else {
            iter++;
        }
    }

    TF.resize(out_face_map.size(), 3);
    int cnt = 0;
    for (auto const &[key, val]: out_face_map) {
        Vector3d v0 = V.row(key[0]);
        Vector3d v1 = V.row(key[1]);
        Vector3d v2 = V.row(key[2]);
        Vector3d v3 = V.row(TT.row(val.first)[val.second]);
        if ((v1 - v0).cross(v2 - v0).dot(v0 - v3) > 0) {
            TF.row(cnt) << key[0], key[1], key[2];
        } else {
            TF.row(cnt) << key[0], key[2], key[1];
        }
        out_face_map[key] = make_pair(cnt, val.first);
        cnt++;
    }
    // Define sharp edges
    return make_tuple(out_face_map, sharp_edge_map);
}
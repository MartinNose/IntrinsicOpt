#pragma once
#include <iostream>
#include <fstream>
#include "Eigen/Core"

using namespace Eigen;
using namespace std;

void read_vtk(const std::string& path, MatrixXd &V, MatrixXi &T) {
    cout << "Reading input from " << path << endl;
    std::ifstream input_s(path);

    int V_num;
    int T_num;
    // Only for vtk tet file
    for (string line; getline(input_s, line);) {
        if (line.substr(0, 6) == "POINTS") {
            istringstream ls(line);
            string tmp;
            ls >> tmp >> tmp;
            V_num = stoi(tmp);
            break;
        }
    } 

    V.resize(V_num, 3);

    for (int i = 0; i < V_num; i++) {
        string line;
        getline(input_s, line);
        istringstream ls(line);
        string tmp0, tmp1, tmp2;
        ls >> tmp0 >> tmp1 >> tmp2;
        
        V.row(i) << stof(tmp0), stof(tmp1), stof(tmp2);
        
    }


    for (string line; getline(input_s, line);) {
        if (line.substr(0, 5) == "CELLS") {
            istringstream ls(line);
            string tmp;
            ls >> tmp >> tmp;

            T_num = stoi(tmp);
            break;
        }
    } 

    T.resize(T_num, 4);

    for (int i = 0; i < T_num; i++) {
        string line;
        getline(input_s, line);
        istringstream ls(line);
        string tmp, tmp0, tmp1, tmp2, tmp3;
        ls >> tmp >> tmp0 >> tmp1 >> tmp2 >> tmp3;

        if (tmp != "4") {
            cerr << "Unsupported mesh, only *.tet.vtk" << endl;
            exit(-1);
        }
        
        T.row(i) << stoi(tmp0), stoi(tmp1), stoi(tmp2), stoi(tmp3);
    }

} 

void write_vtk_points(const std::string& path, MatrixXd &V) {
    std::ofstream out(path);
    out << "# vtk DataFile Version 2.0\n"
            "TET\n"
            "ASCII\n"

            "DATASET UNSTRUCTURED_GRID\n";
    int n = V.rows();
    out << "POINTS " << n << " double\n";
    for (int i = 0; i < n; i++) {
        out << V.row(i)[0] << " " << V.row(i)[1] << " " << V.row(i)[2] << "\n";
    }
    out << "CELLS " << n << " " << 2 * n << "\n";
    for (int i = 0; i < n; i++) {
        out << "1 " << i << "\n";
    }
    out << "CELL_TYPES " << n << "\n";
    for (int i = 0; i < n; i++) {
        out << "1\n";
    }
}
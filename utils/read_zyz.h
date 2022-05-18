#include <iostream>
#include <string>
#include <Eigen/Core>
#include "maxima_funcs.h"

using namespace Eigen;
using namespace std;

template<typename T>
std::istream& binary_read(std::istream& stream, T& value){
    return stream.read(reinterpret_cast<char*>(&value), sizeof(T));
}

bool read_zyz(const string &filename, MatrixXd &FF0, MatrixXd &FF1, MatrixXd &FF2) {
    std::ifstream inf(filename, std::ios::binary);
    if (inf.fail()) {
        std::cerr << "# [ IO ] cannot read matrix from file: "
                  << filename << std::endl;
        return false;
    }

    size_t r, c;
    binary_read(inf, r);
    binary_read(inf, c);

    MatrixXd zyz(r, c);

    for (int i = 0; i < c; i++) {
        for (int j = 0; j < r; j++) {
            binary_read(inf, zyz(j,i));
        }
    }

    FF0.resize(zyz.cols(), 3);
    FF1.resize(zyz.cols(), 3);
    FF2.resize(zyz.cols(), 3);

    for (int i = 0; i < zyz.cols(); i++) {
        double tt[7];
        tt[1] = cos(zyz.col(i)[0]);
        tt[2] = cos(zyz.col(i)[1]);
        tt[3] = cos(zyz.col(i)[2]);
        tt[4] = sin(zyz.col(i)[0]);
        tt[5] = sin(zyz.col(i)[2]);
        tt[6] = sin(zyz.col(i)[1]);
        MatrixXd temp(3,3);
        temp(0,0) = tt[1]*tt[2]*tt[3]-tt[4]*tt[5];
        temp(0,1) = (-tt[1]*tt[5])-tt[2]*tt[3]*tt[4];
        temp(0,2) = tt[3]*tt[6];
        temp(1,0) = tt[1]*tt[2]*tt[5]+tt[3]*tt[4];
        temp(1,1) = tt[1]*tt[3]-tt[2]*tt[4]*tt[5];
        temp(1,2) = tt[5]*tt[6];
        temp(2,0) = -tt[1]*tt[6];
        temp(2,1) = tt[4]*tt[6];
        temp(2,2) = tt[2];
        FF0.row(i) = temp.col(0).transpose();
        FF1.row(i) = temp.col(1).transpose();
        FF2.row(i) = temp.col(2).transpose();
    }
    return true;
}
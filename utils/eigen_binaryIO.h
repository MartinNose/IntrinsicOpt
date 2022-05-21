#ifndef MESHTRACE_EIGEN_BINARYIO_H
#define MESHTRACE_EIGEN_BINARYIO_H
#include <string>
#include <fstream>

namespace Eigen{
    template <typename M>
        void write_matrix_with_binary(const std::string& filename, const M& mat)
    {
        std::ofstream out(filename, std::ios::binary);
        size_t rows = mat.rows();
        size_t cols = mat.cols();
        out.write((char*) (&rows), sizeof(size_t));
        out.write((char*) (&cols), sizeof(size_t));
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                out.write((char*) (&mat(i,j)), sizeof(double));
            }
        }
    }

    template<class Matrix>
    void write_binary(const char* filename, const Matrix& matrix){
        std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
        typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
        out.write((char*) (&rows), sizeof(typename Matrix::Index));
        out.write((char*) (&cols), sizeof(typename Matrix::Index));
        out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
        out.close();
    }
    template<class Matrix>
    void read_binary(const char* filename, Matrix& matrix){
        std::ifstream in(filename, std::ios::in | std::ios::binary);
        typename Matrix::Index rows=0, cols=0;
        in.read((char*) (&rows),sizeof(typename Matrix::Index));
        in.read((char*) (&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
        in.close();
    }
} // Eigen::

#endif
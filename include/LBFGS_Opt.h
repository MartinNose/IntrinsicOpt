#pragma once
#include <iostream>
#include "Eigen/Core"
#include <vector>

using namespace Eigen;
using namespace std;

template<typename Scalar>
void LBFGS_optimization(Scalar l, 
        const MatrixXd &V, 
        const MatrixXi &T, 
        const MatrixX3d FF0, 
        const MatrixX3d FF1, 
        const MatrixX3d FF2, 
        vector<Particle<>> &P) {
    
    


}
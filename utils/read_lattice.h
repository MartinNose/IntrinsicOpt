#pragma once
#include <iostream>
#include <fstream>
#include <cstdlib>

double read_lattice(const std::string& path) {
    std::ifstream input;
    input.open(path);
    double res;
    input >> res;
    input.close();
    return res;
}
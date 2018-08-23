//
// Created by Sergei Troshin on 20/08/2018.
//

#include <iostream>
#include <armadillo>
#include <vector>
#include <assert.h>
#include "linalg.h"
#include <fstream>
#include<set>
#include <string>
#include <deque>
#include <ctime>
#include <string.h>
#define ARMA_USE_HDF5

#define __FILE_KARTAN__ "/Users/istar/ClionProjects/gauss/kartan.txt"
#define __FILE_WEIGHTS__ "/Users/istar/ClionProjects/gauss/weights"

std::deque<arma::Mat<int> > basis(int ndim) {
    std::deque<arma::Mat<int> > a;
    for (int i = 0; i < ndim; i++) {
        arma::Mat<int> x = arma::Mat<int>(1, ndim, arma::fill::zeros);
        x(i) = 1;
        a.push_back(x);
    }
    return a;
}

int main3() {
    const char * ROOT_SYS_NAME = "f4"; //
    auto kartan = load(__FILE_KARTAN__, ROOT_SYS_NAME);
    auto weights = load(__FILE_WEIGHTS__, ROOT_SYS_NAME);
    auto base = basis(kartan.n_rows);
    auto s = spread(base, kartan);
    for (auto & it : s)
        it.print("");
    cout << s.size();
    return 0;
}
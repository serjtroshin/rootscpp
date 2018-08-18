//
// Created by Sergei Troshin on 11/08/2018.
//

#ifndef GAUSS_LINALG_H
#define GAUSS_LINALG_H

//
// Created by Sergei Troshin on 11/08/2018.
//

#include <iostream>
#include <armadillo>
#include <vector>
#include <assert.h>

#ifndef GAUSS_LINALG_H_H
#define GAUSS_LINALG_H_H

using std::pair;
using std::cout;


struct gauss_meta {
    int rank;
    std::vector<std::pair<int, int> > main_elems;
    std::vector<int> free_cols;
    gauss_meta() {
        rank = 0;
        main_elems = std::vector<std::pair<int, int> >();
        free_cols = std::vector<int>();
    };
};

struct c_unique {
    int current;
    c_unique() {current=0;}
    int operator()() {return current++;}
};

struct gauss_meta
gauss(arma::Mat<int> & mt);

//  struct gauss_meta enhanced_gauss(arma::Mat<int> & A, bool already_gauss=false, struct gauss_meta * meta_=nullptr);
//  arma::Mat<int> solve(arma::Mat<int> & A);
arma::Mat<int> solve_fsr(arma::Mat<int> & A, struct gauss_meta * meta_);

#endif //GAUSS_LINALG_H_H


#endif //GAUSS_LINALG_H

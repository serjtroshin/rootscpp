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
#include <deque>
#include <set>


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
struct cmp {
    bool operator()(const arma::Mat<int> &a, const arma::Mat<int> &b) const {
        assert(a.n_cols == b.n_cols);
        assert(a.n_rows == b.n_rows);
        for (int i = 0; i < a.n_rows * a.n_cols; ++i) {
            if (a(i) != b(i))
                return a(i) < b(i);
        }
        return 0;
    }
};

arma::Mat<int> load(std::string file, std::string ROOT_SYS_NAME);
std::set<arma::Mat<int>, cmp> spread(std::deque<arma::Mat<int> > & d, arma::Mat<int> & kartan);

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
pair<std::vector<int>, int> gcd(const std::vector<int> ar);

#endif //GAUSS_LINALG_H_H


#endif //GAUSS_LINALG_H

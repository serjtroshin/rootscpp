#include <iostream>
#include <armadillo>
#include <vector>
#include <assert.h>
#include "linalg.h"
#include <algorithm>

using std::pair;
using std::cout;

/*
 *
 *
e6
2 0 -1 0 0 0; 0 2 0 -1 0 0; -1 0 2 -1 0 0; 0 -1 -1 2 -1 0; 0 0 0 -1 2 -1; 0 0 0 0 -1 2
e7
2 0 -1 0 0 0 0; 0 2 0 -1 0 0 0; -1 0 2 -1 0 0 0; 0 -1 -1 2 -1 0 0; 0 0 0 -1 2 -1 0; 0 0 0 0 -1 2 -1; 0 0 0 0 0 -1 2
e8
2 0 -1 0 0 0 0 0; 0 2 0 -1 0 0 0 0; -1 0 2 -1 0 0 0 0; 0 -1 -1 2 -1 0 0 0; 0 0 0  -1 2 -1 0 0; 0 0 0 0 -1 2 -1 0; 0 0 0 0 0 -1 2 -1; 0 0 0 0 0 0 -1 2
 */


/*
 * weights
 *
e6
4 3 5 6 4 2; 1 2 2 3 2 1; 5 6 10 12 8 4; 4 6 8 12 10 5; 2 4 4 6 4 2; 4 6 8 12 8 4; 2 3 4 6 5 4

 */

inline int sgn(int a) {
    if (a == 0)
        return 0;
    return a > 0 ? 1 : -1;
}
pair<int, int> mod(int a, int b)
{
    assert(b != 0);
    int r = a % b;
    int d = a / b;
    return r >= 0 ? std::pair<int, int>(d, r) : std::pair<int, int>(d - sgn(b), r + std::abs(b));
}

pair<pair<int, int>, int>
gcd2(int a, int b) {
    auto res = mod(a, b);
    int d = res.first;
    int m = res.second;
    arma::Mat<int> c1 = {0, 1};
    arma::Mat<int> c2 = {1, -d};
    a = b;
    b = m;
    while (m != 0) {
        res = mod(a, b);
        d = res.first;
        m = res.second;
        auto tmp = c2;
        //  arma::Mat<int> d_ = {d};
        c2 = c1 - d * c2;
        c1 = tmp;
        a = b;
        b = m;
    }
    return {{c1(0,0), c1(0,1)}, a};
};

pair<std::vector<int>, int> gcd(const std::vector<int> ar) {
    assert(ar.size() > 1);
    std::vector<int> linear(ar.size());
    std::vector<pair<int, int> > linear_prev;
    int gcd_ = ar[0];
    for (int i = 0; i < ar.size() - 1; ++i) {
        auto cur = gcd2(gcd_, ar[i + 1]);
        gcd_ = cur.second;
        linear_prev.emplace_back(cur.first);
    }
    int mult = 1;
    int cur = 1;
    for (int i = linear.size() - 2; i >= 0; i--) {
        mult *= cur;
        linear[i + 1] = mult * linear_prev[i].second;
        cur = linear_prev[i].first;
    }
    linear[0] = mult * linear_prev[0].first;
    return {linear, gcd_};
};

int lcd(int a, int b) {
    return a * b / gcd2(a, b).second;
}

void normalize(arma::Mat<int> & mt) {
    for (int i = 0; i < mt.n_rows; i++) {
        std::vector<int> rr;
        for (int j = 0; j < mt.n_cols; j++) {
            if (mt(i, j) != 0)
                rr.push_back(mt(i, j));
        }
        if (rr.size() > 1) {
            int gg = gcd(rr).second;
            mt.row(i) /= gg;

        } else if (rr.size() == 1) {
            mt.row(i) /= rr[0];
        }
    }
}


struct gauss_meta
gauss(arma::Mat<int> & mt) {
    auto n = mt.n_rows;
    auto m = mt.n_cols;
    int r = 0; // main row
    auto meta = gauss_meta();
    for (int i = 0; i < m; ++i) {
        std::vector<int> unders, indexes;
        for (int j = r; j < n; j++) {
            if (mt(j, i) != 0) {
                indexes.push_back(j);
                unders.push_back(mt(j, i));
            }
        }
        if (unders.size() == 0) {
            meta.free_cols.push_back(i);
            continue;
        } else if (unders.size() == 1) {
            mt.swap_rows(r, indexes[0]);
            //mt.row(r).swap(mt.row(indexes[0]));
            meta.main_elems.push_back({r, i});
            r++;
            continue;
        }
        auto linear_gcd = gcd(unders);
        auto linear = linear_gcd.first;
        int notz = 0;
        for (int j = 0; j < linear.size(); j++) {
            if (linear[j] != 0)
                notz = indexes[j];
        }
        auto gcd_ = linear_gcd.second;
        auto gcd_vector = arma::Mat<int>(1, m, arma::fill::zeros);
        for (int j = 0; j < unders.size(); ++j) {
            if (linear[j] != 0)
                gcd_vector += mt.row(indexes[j]) * linear[j];
        }
        mt.row(notz) = gcd_vector;
        mt.swap_rows(r, notz);
        for (int j = r + 1; j < n; ++j) {
            if (mt(j, i) != 0) {
                mt.row(j) -= (mt(j,i) / mt(r,i)) * mt.row(r);
            }
        }
        meta.main_elems.push_back({r, i});
        r++;
    }
    normalize(mt);
    meta.rank = meta.main_elems.size();
    //  mt.print("mt");
    return meta;
}

struct gauss_meta enhanced_gauss(arma::Mat<int> & A_, bool already_gauss=false, struct gauss_meta * meta_=nullptr) {
    struct gauss_meta meta;
    if (!already_gauss)
        meta = gauss(A_);
    else
        meta = *meta_;
    arma::Mat<long long> A = arma::Mat<long long>(A_.n_rows, A_.n_cols);
    for (int i = 0; i < A_.n_rows; i++) {
        for (int j = 0; j < A_.n_cols; j++) {
            A(i, j) = (long long)A_(i, j);
        }
    }
    for (auto & it : meta.main_elems) {
        int i = it.first;
        int j = it.second;
        for (int k = i - 1; k >= 0; k--) {
            if (A(k, j) != 0) {
                int _gcd = gcd2(A(i, j), A(k, j)).second;
                A.row(k) = A.row(k) * (A(i, j) / _gcd) - A.row(i) * (A(k, j) / _gcd);
                //  A.print("ii");
            }
        }
    }
    for (int i = 0; i < A.n_rows; i++) {
        for (int j = 0; j < A.n_cols; j++) {
            assert(abs(A_(i, j)) < (long long)INT_MAX);
            A_(i, j) = (int)A(i, j);
        }
    }
    //  A.print("A");
    return meta;
}

arma::Mat<int> solve_fsr(arma::Mat<int> & A, struct gauss_meta * meta_=nullptr) {
    auto n = A.n_rows;
    auto m = A.n_cols;
    arma::Mat<int> fund_sys = arma::Mat<int>();
    struct gauss_meta meta;
    if (meta_ == nullptr)
        meta = enhanced_gauss(A);
    else {
        meta = *meta_;
        meta = enhanced_gauss(A, true, &meta);
    }
    std::map<int, int> main_col;
    for (auto & it : meta.main_elems) {
        main_col[it.second] = it.first;
    }
    int cur_main_col = 0;
    int lcd_ = 1;
    for (int col = 0; col < m; ++col) {
        if (main_col.find(col) == main_col.end()) {
            auto new_vector = arma::Mat<int>(1, m, arma::fill::zeros);
            new_vector(col) = lcd_;
            int j = 0;
            while (j < meta.main_elems.size() && meta.main_elems[j].second < col) {
                new_vector(meta.main_elems[j].second) = -lcd_ / A(meta.main_elems[j].first, meta.main_elems[j].second) * A(j, col);
                j++;
            }
            std::vector<int> g;
            for (int j = 0; j < m; j++) {
                if (new_vector(j) != 0)
                    g.push_back(new_vector(j));
            }
            if (g.size() > 1) {
                new_vector /= gcd(g).second;
            } else if (g.size() == 1) {
                new_vector /= g[0];
            }
            fund_sys.insert_rows(0, new_vector);
        } else {
            lcd_ = lcd(lcd_, A(main_col[col], col));
        }
    }
    return fund_sys;
}


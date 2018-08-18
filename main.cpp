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
std::map<std::string, long long> weil_size = {
        {"b3", 48},
        {"b4", 384},
        {"b5", 3840},
        {"f4", 1152},
        {"e6", 51840}
};


c_unique UniqueNumber;

arma::Mat<int> load(std::string file, std::string ROOT_SYS_NAME) {
    std::ifstream f;
    f.open(file);
    std::string s;
    while (std::getline(f, s)) {
        if (s == ROOT_SYS_NAME) {
            std::getline(f, s);
            return arma::Mat<int>(s);
        }
    }
    f.close();
    return arma::Mat<int>(s);
}

arma::Mat<int> reflect(arma::Mat<int> & a, int basis_root_id, arma::Mat<int> & kartan) {
    auto b = arma::Mat<int>(1, a.n_cols, arma::fill::zeros);
    b(basis_root_id) = arma::dot(a, kartan.col(basis_root_id));
    return a - b;
}
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

auto spread(std::deque<arma::Mat<int> > & d, arma::Mat<int> & kartan) {
    std::set<arma::Mat<int>, cmp> s;
    while (!d.empty()) {
        arma::Mat<int> fr = d.front();
        for (int i = 0; i < kartan.n_cols; i++) {
            auto r = reflect(fr, i, kartan);
            if (s.find(r) == s.end()) {
                s.insert(r);
                d.push_back(r);
            }
        }
        d.pop_front();
    }
    return s;
}

bool f(arma::Mat<int> & mt, arma::Mat<int> * res, int rank) {
    static int counter = 0;
    counter++;
    if (counter % 100000 == 0)
        std::cout << counter << '\n';
    // mt.print("mt");
    struct gauss_meta meta = gauss(mt);
    if (meta.rank == rank) {
        *res = solve_fsr(mt, &meta);
        return 1;
    }
    return 0;
}

void myfunction (int i) {
    std::cout << i << ' ';
}

void save(arma::Mat<int> & res, const char * file) {
    //  res.save(file, arma::arma_ascii);
}

void write_to_file(arma::Mat<int> & res, std::ofstream & file) {
    for (int i = 0; i < res.n_cols; ++i) {
        file << res(i) << ' ';
    }
    file << '\n';
}



std::string to_str(arma::Mat<int> & col) {
    std::string s;
    for (int i = 0; i < col.n_rows; i++) {
        int elem = col(i);
        if (elem < 0) {
            s += "-";
            s += -elem + '0';
        } else {
            s += elem + '0';
        }
        s += " ";
    }
    return s;
}
int num_unique_cols(arma::Mat<int> & cols) {
    std::set<std::string> st;
    for (int i = 0; i < cols.n_cols; i++) {
        arma::Mat<int> col = cols.col(i);
        st.insert(to_str(col));
    }
    //for (auto & it : st) {
    //    std::cout << it << ' ';
    //}
    return st.size();
}













int main1(int argc, char ** argv)
{
    clock_t begin = clock();
    const char * ROOT_SYS_NAME = "b5";
    //arma::Mat<int> A = {{2, 4, 6, 10, 6, 4}, {5, 3, 7, 9, 8, 4}, {5, 6, 7, 9, 8, 4}, {5, 6, 7, 12, 8, 4}, {5, 6, 10, 12, 8, 4}};

    //auto fss = solve_fsr(A, nullptr);
    //fss.print("ФСР");


    auto kartan = load(__FILE_KARTAN__, ROOT_SYS_NAME);
    auto weights = load(__FILE_WEIGHTS__, ROOT_SYS_NAME);
    kartan.print("kartan");
    weights.print("weights");
    std::deque<arma::Mat<int> > d;
    for (int i = 0; i < weights.n_rows; i++)
        d.push_back(weights.row(i));
    auto s = spread(d, kartan);
    std::cout << "S size: " << s.size() << '\n';

    // -----------------
    std::cout << "check: \n";
    long long w_sz = weil_size[ROOT_SYS_NAME];
    for (int i = 0; i < weights.n_rows; ++i) {
        std::deque<arma::Mat<int> > d;
        d.push_back(weights.row(i));
        auto s = spread(d, kartan);
        std::cout << "orbit size: " << s.size() << '\n';
        assert(w_sz % s.size() == 0); // Порядок элемента делит порядок группы Вейля
    }
    // ------------------
    int rank = kartan.n_cols - 1;

    long long to_solve = 1;
    int ff = 1;
    for (int i = 0, j = s.size(), f = 1; i < rank; i++, j--, f *= (i + 1)) {
        to_solve *= j;
        ff = f;
    }
    cout << "to_solve " << to_solve / ff << '\n';



    int n = s.size();
    int r = rank;
    std::vector<arma::Mat<int> > sv(s.begin(), s.end());
    arma::Mat<int> S; // column vector
    for (auto & it : sv)
        S.insert_rows(0, it);
    S.print("S");
    std::vector<int> myints(r);
    std::vector<int>::iterator first = myints.begin(), last = myints.end();

    std::generate(first, last, UniqueNumber);
    arma::Mat<int> final;
    arma::Mat<int> res;
    arma::Mat<int> sub;
    for (auto & it : myints)
        sub.insert_rows(0, S.row(it));

    int ans = 0;
    std::ofstream fout;
    fout.open(ROOT_SYS_NAME);
    sub.print("sub before");
    if (f(sub, &res, rank)) {
        ans++;
        //res.print("ans");
        //final.insert_rows(0, res);
        //final.insert_rows(0, -res);
        write_to_file(res, fout);
        res = -res;
        write_to_file(res, fout);
    }
    cout << '\n';
    while((*first) != n-r){
        std::vector<int>::iterator mt = last;

        while (*(--mt) == n-(last-mt));
        (*mt)++;
        while (++mt != last) *mt = *(mt-1)+1;
        //std::for_each(first, last, myfunction);
        //cout << '\n';
        //  do something with next

        arma::Mat<int> sub;
        for (auto & it : myints)
            sub.insert_rows(0, S.row(it));
        // sub.print("sub before");
        // cout << '\n';
        if (f(sub, &res, rank)) {
            // sub.print("sub");
            ans++;
            //final.insert_rows(0, res);
            //final.insert_rows(0, -res);
            if (arma::any(arma::vectorise(sub * res.t()))) { exit(1); }
            // res.print("res");
            write_to_file(res, fout);
            res = -res;
            write_to_file(res, fout);
        }
        //cout << "\n\n";

    }
    cout << "ans " <<  ans << ' ';

    // final = final.t();
    // final.print("final");
    //auto unique = final(arma::find_unique(final));
    std::cout << "UNIQUE: " << num_unique_cols(final) << '\n';

    save(final, ROOT_SYS_NAME);
    fout.close();
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "TIME secs: " << elapsed_secs << '\n';
}
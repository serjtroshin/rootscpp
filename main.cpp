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
#define __FILE_GRAMM__ "/Users/istar/ClionProjects/gauss/gramm"
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

std::set<arma::Mat<int>, cmp> spread(std::deque<arma::Mat<int> > & d, arma::Mat<int> & kartan) {
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
std::set<arma::Mat<int>, cmp> s_to_grammed_s(std::set<arma::Mat<int>, cmp> & s, arma::Mat<int> & gramm) {
    std::set<arma::Mat<int>, cmp> new_s;
    arma::Mat<int> m;
    for (auto & it : s) {
        m.insert_rows(0, it);
    }
    m = m * gramm;
    for (int i = 0; i < m.n_rows; ++i) {
        arma::Mat<int> mm = m.row(i);
        new_s.insert(mm);
    }
    return new_s;
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

auto reduce(std::set<arma::Mat<int>, cmp> s) {
    std::set<arma::Mat<int>, cmp> s_new;
    std::copy_if(s.begin(), s.end(), std::inserter(s_new, s_new.end()), [](auto & elem){return
            ! arma::all(arma::vectorise(elem > 0)) and
            ! arma::all(arma::vectorise(elem < 0));});
    std::set<arma::Mat<int>, cmp> s_new_1;
    for (auto & it : s_new) {
        arma::Mat<int> nit = -it;
        if (s_new_1.find(nit) == s_new_1.end()) {
            s_new_1.insert(it);
        }
    }
    return s_new_1;
}


std::vector<int> as_list(const arma::Mat<int> & a) {
    std::vector<int> vec;
    for (int i = 0; i < a.n_rows * a.n_cols; i++) {
        vec.push_back(a(i));
    }
    return vec;
}



bool has_collinear(std::set<arma::Mat<int>, cmp> & final_s) {
    std::set<arma::Mat<int>, cmp> test;
    for (auto & it : final_s) {
        auto vec = as_list(it);
        std::vector<int> to_copy;
        std::copy_if(vec.begin(), vec.end(), std::back_inserter(to_copy), [](int i){return i != 0;});
        int _gcd;
        if (to_copy.size() == 1)
            _gcd = to_copy[0];
        else
            _gcd = gcd(to_copy).second;
        if (abs(_gcd) != 1) {
            arma::Mat<int> normed = it / _gcd;
            if (final_s.find(normed) != final_s.end())
                return 1;
        }
    }
    return 0;
}

void run(const char * ROOT_SYS_NAME) {
    cout << "\n\n\nSTARTED " << ROOT_SYS_NAME << '\n';
    clock_t begin = clock();
    //arma::Mat<int> A = {{2, 4, 6, 10, 6, 4}, {5, 3, 7, 9, 8, 4}, {5, 6, 7, 9, 8, 4}, {5, 6, 7, 12, 8, 4}, {5, 6, 10, 12, 8, 4}};

    //auto fss = solve_fsr(A, nullptr);
    //fss.print("ФСР");


    auto kartan = load(__FILE_KARTAN__, ROOT_SYS_NAME);
    auto weights = load(__FILE_WEIGHTS__, ROOT_SYS_NAME);
    auto gramm = load(__FILE_GRAMM__, ROOT_SYS_NAME);
    kartan.print("kartan");
    weights.print("weights");
    gramm.print("gramm");
    (weights * kartan).print("kartan * weights");
    std::deque<arma::Mat<int> > d;
    for (int i = 0; i < weights.n_rows; i++)
        d.push_back(weights.row(i));
    auto s = spread(d, kartan);
    std::cout << "S size: " << s.size() << '\n';
    for (auto & it : s)
        it.print("");
    // -----------------

    // update by Авдеев Р.C.
    // reduce S
    s = reduce(s);
    std::cout << "reduced size of S: " << s.size() << '\n';

    // -----------------
    /*
    std::cout << "check: \n";
    long long w_sz = weil_size[ROOT_SYS_NAME];
    for (int i = 0; i < weights.n_rows; ++i) {
        std::deque<arma::Mat<int> > d;
        d.push_back(weights.row(i));
        auto s = spread(d, kartan);
        std::cout << "orbit size: " << s.size() << '\n';
        assert(w_sz % s.size() == 0); // Порядок элемента делит порядок группы Вейля
    }
    */
    // ------------------
    int rank = kartan.n_cols - 1;

    long long to_solve = 1;
    int ff = 1;
    for (int i = 0, j = s.size(), f = 1; i < rank; i++, j--, f *= (i + 1)) {
        to_solve *= j;
        ff = f;
    }
    cout << "to_solve " << to_solve / ff << '\n';

    // !!!!!!!!!!!!
    // IF the basis is of prime vector -> use gramm matrix for SoLE
    s = s_to_grammed_s(s, gramm);
    //

    //  iterate over all subsets of vectors from S and find ortogonal vector (one from characteristic set)
    int n = s.size();
    int r = rank;
    std::vector<arma::Mat<int> > sv(s.begin(), s.end());
    arma::Mat<int> S; // column vector
    for (auto & it : sv)
        S.insert_rows(0, it);
    //  S.print("S");
    std::vector<int> myints(r);
    std::vector<int>::iterator first = myints.begin(), last = myints.end();

    std::generate(first, last, UniqueNumber);
    std::set<arma::Mat<int>, cmp> final;
    arma::Mat<int> res;
    arma::Mat<int> sub;
    for (auto & it : myints)
        sub.insert_rows(0, S.row(it));

    int ans = 0;
    std::ofstream fout;
    fout.open(ROOT_SYS_NAME);
    //sub.print("sub before");
    if (f(sub, &res, rank)) {
        ans++;
        //res.print("ans");
        final.insert(res);
        final.insert(-res);
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
            final.insert(res);
            final.insert(-res);
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

    std::cout << "UNIQUE: " << final.size() << '\n';

    //  for (auto & it : final) {it.print();}


    std::deque<arma::Mat<int> > final_d(final.begin(), final.end());
    auto final_s = spread(final_d, kartan);
    std::cout << "final size " << final_s.size() << '\n';
    // проверка на коллинеарность
    {
        if (has_collinear(final_s))
            exit(1);
    }
    //----------

    //for (auto & it : final_s) {it.print();}
    //save(final, ROOT_SYS_NAME);
    fout.close();
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "TIME secs: " << elapsed_secs << '\n';
}

int main(int argc, char ** argv)
{
    //arma::Mat<int> gramm = {{4, -2, 0, 0}, {-2, 4, -2, 0}, {0, -2, 2, -1}, {0, 0, -1, 2}};
    run("f4");
}
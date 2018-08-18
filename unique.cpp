//
// Created by Sergei Troshin on 18/08/2018.
//

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <set>

int main(int argc, char ** argv) {
    char * pathin = argv[1];
    std::ifstream in;
    in.open(pathin);
    std::string s;
    int cnt = 0;
    std::set<std::string> ar;
    while (std::getline(in, s)) {
                cnt++;
                ar.insert(s);
                if (cnt % 10000000 == 0) {
                    std::cout << cnt << ' ' << ar.size() << '\n';
                }
    }
    std::cout << ar.size();
}
#include <iostream>
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>
#include <streambuf>
#include <chrono>
#include <random>
#include <immintrin.h>
#include <thread>
#include <string>
#include <sstream>  // Add this line

using namespace std;

// Function to read matrix (of type Data_Type) from text file
template <typename Data_Type>
tuple<Data_Type*, int, int> Read_Data(string path) {
    ifstream t(path);
    string str((istreambuf_iterator<char>(t)), istreambuf_iterator<char>());

    auto in_float = [](char ch) { return ('0' <= ch && ch <= '9') || (ch == '.'); };
    int rows = 0, cols = 0;

    for (int i = 0; i < str.size(); i++)
        if (str[i] == '\n') rows++;

    for (int i1 = 0, i2 = 0; i2 < str.size() && str[i2] != '\n'; ) {
        for (i1 = i2; !in_float(str[i1]) && i1 < str.size(); i1++) {}
        for (i2 = i1; in_float(str[i2]) && i2 < str.size(); i2++) {}
        if (i1 != i2) cols++;
    }

    Data_Type* p = new Data_Type[rows * cols];

    for (int i1 = 0, i2 = 0, j = 0; i2 < str.size(); ) {
        for (i1 = i2; !in_float(str[i1]) && i1 < str.size(); i1++) {}
        for (i2 = i1; in_float(str[i2]) && i2 < str.size(); i2++) {}
        if (i1 != i2) p[j++] = stof(str.substr(i1, i2 - i1));
    }

    return { p, rows, cols };
}

int main(int argc, const char* argv[]) {
    //double * data_matrix[800][800];
    auto path = "C:/Users/lembergdan/Desktop/Data/data_struct(1000x1000).txt";
    auto [data_matrix, rows, cols] = Read_Data<double>(path);

    int i = 5;
    int j = 10;
    cout << data_matrix[i * cols + j] << endl;
    cout << "rows = " << rows << endl;
    cout << "cols = " << cols << endl;
    return 0;
}

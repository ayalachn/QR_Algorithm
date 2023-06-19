#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <tuple>
#include <sstream>  
#include <chrono>
#include <random>
#include <immintrin.h>
#include <thread>
#include <valarray>

using namespace std;
using slice = std::slice;
typedef std::valarray<double> Vector;

long double Get_Time() { // get time in seconds
    using chrono::high_resolution_clock;
    auto t = high_resolution_clock::now();
    auto nanosec = t.time_since_epoch();
    return nanosec.count() / 1000000000.0;
}

/////////////////////////////////////////////////////////////////////////////////////////
///                         CLASS MATRIX                                    /////////////
/////////////////////////////////////////////////////////////////////////////////////////
class Matrix {
public:
    double* p;
    int rows;
    int cols;

    bool isTriangular = false;

    Matrix() : p(nullptr), rows(0), cols(0) {}
    Matrix(int rows_, int cols_) : p(new double[rows_ * cols_]), rows(rows_), cols(cols_) {}
    Matrix(int rows_, int cols_, double val) : p(new double[rows_ * cols_]), rows(rows_), cols(cols_) {
        for (int i = 0; i < rows * cols; i++)
            p[i] = val;
    }
    Matrix(int rows_, int cols_, double a, double b) : p(new double[rows_ * cols_]), rows(rows_), cols(cols_) { // Random matrix a(i,j) ~ U(a, b)
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(a, b);

        for (int i = 0; i < rows * cols; i++) p[i] = dis(gen);
    }

    Matrix(int rows_, int cols_, string val) : p(new double[rows_ * cols_]), rows(rows_), cols(cols_) { // identity
        int i, j;
        if (val == "I") {
            for (i = 0; i < rows_; i++) {
                for (j = 0; j < cols_; j++) {
                    if (i != j)
                        *(p + i * cols_ + j) = 0;
                    else
                        *(p + i * cols_ + j) = 1;
                }
            }
        }
    }
    Matrix(string path) {
        ifstream t(path);
        string str((istreambuf_iterator<char>(t)), istreambuf_iterator<char>());

        auto in_float = [](char ch) { return ('0' <= ch && ch <= '9') || (ch == '.'); };
        int rows_ = 0, cols_ = 0;

        for (int i = 0; i < str.size(); i++)
            if (str[i] == '\n') rows_++;

        for (int i1 = 0, i2 = 0; i2 < str.size() && str[i2] != '\n'; ) {
            for (i1 = i2; !in_float(str[i1]) && i1 < str.size(); i1++) {}
            for (i2 = i1; in_float(str[i2]) && i2 < str.size(); i2++) {}
            if (i1 != i2) cols_++;
        }

        rows = rows_; cols = cols_;
        p = new double[rows * cols];

        for (int i1 = 0, i2 = 0, j = 0; i2 < str.size(); ) {
            for (i1 = i2; !in_float(str[i1]) && i1 < str.size(); i1++) {}
            for (i2 = i1; in_float(str[i2]) && i2 < str.size(); i2++) {}
            if (i1 != i2) p[j++] = stof(str.substr(i1, i2 - i1));
        }
    }
    Matrix(Matrix& m) : p(new double[m.rows * m.cols]), rows(m.rows), cols(m.cols) {
        for (int i = 0; i < rows * cols; i++)
            p[i] = m.p[i];
    }

    Matrix(const Matrix& m) : p(new double[m.rows * m.cols]), rows(m.rows), cols(m.cols) {
        for (int i = 0; i < rows * cols; i++)
            p[i] = m.p[i];
    }
    
    Matrix(Matrix&& m) : rows(m.rows), cols(m.cols) {
        p = m.p;
        m.p = nullptr;
    }

    friend bool eq(Matrix& a, Matrix& b) { return a.rows == b.rows && a.cols == b.cols; }
    friend bool eq(Matrix&& a, Matrix&& b) { return a.rows == b.rows && a.cols == b.cols; }
    friend bool eq(Matrix&& a, Matrix& b) { return a.rows == b.rows && a.cols == b.cols; }
    friend bool eq(Matrix& a, Matrix&& b) { return a.rows == b.rows && a.cols == b.cols; }
    void setIsTriangular(bool isTriangular) {
        this->isTriangular = isTriangular;
    }
    Matrix& operator = (Matrix& m) {
        if (p == m.p) return *this;
        if (eq(*this, m)) {
            for (int i = 0; i < rows * cols; i++)
                p[i] = m.p[i];
        }
        else {
            delete[] p;
            p = nullptr;
            rows = m.rows;
            cols = m.cols;
            p = new double[rows * cols];
            for (int i = 0; i < rows * cols; i++)
                p[i] = m.p[i];
        }
        return *this;
    }
    Matrix& operator = (Matrix&& m) {
        if (p == m.p) return *this;
        p = m.p;
        rows = m.rows;
        cols = m.cols;
        m.p = nullptr;
        return *this;
    }
    
    Matrix getSubMatrix(int start_row, int start_col, int end_row, int end_col) {
        // Input validation
        if (start_row < 0 || start_row >= rows || start_col < 0 || start_col >= cols ||
            end_row <= start_row || end_row > rows || end_col <= start_col || end_col > cols) {
            // Return an empty matrix or throw an exception to indicate invalid input
            return Matrix();
        }

        // Calculate the dimensions of the submatrix
        int subMatrixRows = end_row - start_row;
        int subMatrixCols = end_col - start_col;

        // Create the submatrix using initializer list
        Matrix subMatrix(subMatrixRows, subMatrixCols);

        // Copy elements from the original matrix to the submatrix
        for (int i = 0; i < subMatrixRows; ++i) {
            memcpy(&subMatrix.p[i * subMatrixCols], &p[(start_row + i) * cols + start_col], subMatrixCols * sizeof(double));
        }

        return subMatrix;
    }
    void add2Diag(double u) {
        for (int i = 0; i < rows; ++i)
            p[i * cols + i] += u;
    }
    void setIdentity() {
        for (int i=0 ; i<rows ; i++)
            for (int j = 0; j < cols; j++) {
                if (i == j) p[i * cols + j] = 1;
                else p[i * cols + j] = 0;
            }
    }
    
    void setSubMatrix(const Matrix& submatrix, int start_row, int start_col) {
        const int submatrixrows = submatrix.rows;
        const int submatrixcols = submatrix.cols;

        for (int i = 0; i < submatrixrows; ++i) {
            const int rowoffset = (start_row + i) * cols;
            const int submatrixrowoffset = i * submatrixcols;

            memcpy(&p[rowoffset + start_col], &submatrix.p[submatrixrowoffset], submatrixcols * sizeof(double));
        }
    }
    
    ~Matrix() {
        if (p) {
            delete[] p;
            p = nullptr;
        }
    }
    double& operator () (int i, int j) {
        if (0 <= i && i < rows && 0 <= j && j < cols)
            return p[i * cols + j];
        cerr << "Error of index in operator ()." << endl;
        return p[0];
    }
    /////////////////////////////// transpose ///////////////////////////////
    Matrix t() {
        Matrix tr(cols, rows);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                tr(j, i) = (*this)(i, j);
        return move(tr);
    }
    /////////////////////////////////// + ///////////////////////////////////
    friend Matrix operator + (Matrix& a, Matrix& b) {
        if (!eq(a, b)) {
            cerr << "Error of matrix size in operator +." << endl;
            return Matrix();
        }

        Matrix s(a.rows, a.cols);
        for (int i = 0; i < a.rows * a.cols; i++)
            s.p[i] = a.p[i] + b.p[i];
        return move(s);
    }
    friend Matrix operator + (Matrix&& a, Matrix&& b) {
        if (!eq(a, b)) {
            cerr << "Error of matrix size in operator +." << endl;
            return Matrix();
        }
        for (int i = 0; i < b.rows * b.cols; i++)
            b.p[i] += a.p[i];
        return move(b);
    }
    friend Matrix operator + (Matrix&& a, Matrix& b) { return move(b) + move(a); }
    friend Matrix operator + (Matrix& a, Matrix&& b) { return move(a) + move(b); }
    Matrix& operator += (Matrix&& m) {
        if (!eq(*this, m)) {
            cerr << "Error of matrix size in operator +=." << endl;
            return *this;
        }
        for (int i = 0; i < rows * cols; i++)
            p[i] += m.p[i];
        return *this;
    }
    Matrix& operator += (Matrix& m) { return operator+=(move(m)); }
    /////////////////////////////////// - ///////////////////////////////////
    friend Matrix operator - (Matrix& a, Matrix& b) {
        if (!eq(a, b)) {
            cerr << "Error of matrix size in operator -." << endl;
            return Matrix();
        }
        Matrix s(a.rows, a.cols);
        for (int i = 0; i < a.rows * a.cols; i++)
            s.p[i] = a.p[i] - b.p[i];
        return move(s);
    }
    friend Matrix operator - (Matrix&& a, Matrix&& b) {
        if (!eq(a, b)) {
            cerr << "Error of matrix size in operator -." << endl;
            return Matrix();
        }
        for (int i = 0; i < a.rows * a.cols; i++)
            a.p[i] -= b.p[i];
        return move(a);
    }
    friend Matrix operator - (Matrix&& a, Matrix& b) { return move(a) - move(b); }
    friend Matrix operator - (Matrix& a, Matrix&& b) {
        if (!eq(a, b)) {
            cerr << "Error of matrix size in operator -." << endl;
            return Matrix();
        }
        for (int i = 0; i < a.rows * a.cols; i++)
            b.p[i] = a.p[i] - b.p[i];
        return move(b);
    }
    Matrix& operator -= (Matrix&& m) {
        if (!eq(*this, m)) {
            cerr << "Error of matrix size in operator -=." << endl;
            return *this;
        }
        for (int i = 0; i < rows * cols; i++)
            p[i] -= m.p[i];
        return *this;
    }
    Matrix& operator -= (Matrix& m) { return operator-=(move(m)); }
    /////////////////////////////////// * ///////////////////////////////////
    friend Matrix operator * (Matrix& a, double b) {
        Matrix prod(a.rows, a.cols);
        for (int i = 0; i < a.rows * a.cols; i++)
            prod.p[i] = a.p[i] * b;
        return move(prod);
    }
    friend Matrix operator * (Matrix&& a, double b) {
        for (int i = 0; i < a.rows * a.cols; i++)
            a.p[i] *= b;
        return move(a);
    }
    Matrix& operator *= (Matrix&& m) {
        if (!eq(*this, m)) {
            cerr << "Error of matrix size in operator *." << endl;
            return *this;
        }
        for (int i = 0; i < rows * cols; i++)
            p[i] *= m.p[i];
        return *this;
    }
    Matrix& operator *= (Matrix& m) { return operator*=(move(m)); }

    friend Matrix operator * (double b, Matrix& a) { return a * b; }
    friend Matrix operator * (double b, Matrix&& a) { return a * b; }
    friend Matrix operator * (Matrix&& a, Matrix&& b) {
        if (a.cols != b.rows) {
            cerr << "Error of matrix size in operator *." << endl;
            return Matrix();
        }
        Matrix ret;
        if (a.isTriangular == true) {


            ret.rows = a.rows;
            ret.cols = b.cols;
            ret.p = new double[a.cols * a.cols];

            for (int i = 0; i < a.cols ; ++i) {
                for (int j = 0; j < a.cols ; ++j) {
                    double sum = 0.0;
                    for (int l = i; l < a.cols; l++) {
                        sum += a.p[i*a.cols + l] * b.p[l*b.cols +j];
                    }
                    ret.p[i*ret.cols + j] = sum;
                }
            }
            return ret;
        }
        ret.p = Tools::mult_thread_padd(a.rows, a.p, b.p, a.cols, b.cols, b.cols, Tools::dim_th, Tools::n_th);
        ret.rows = a.rows;
        ret.cols = b.cols;
        return ret;
    }
    friend Matrix operator * (Matrix& a, Matrix& b) { return move(a) * move(b); }
    friend Matrix operator * (Matrix&& a, Matrix& b) { return move(a) * move(b); }
    friend Matrix operator * (Matrix& a, Matrix&& b) { return move(a) * move(b); }

    friend ostream& operator << (ostream& out, Matrix&& m) {
        for (int i = 0; i < m.rows - 1; i++) {
            for (int j = 0; j < m.cols; j++)
                out << m(i, j) << "\t";
            out << endl;
        }
        for (int j = 0; j < m.cols; j++)
            out << m(m.rows - 1, j) << "\t";
        return out;
    }
    friend ostream& operator << (ostream& out, Matrix& m) {
        return out << move(m);
    }
    void toString() {
        std::cout << "\n";
        // Print the matrix in a table-like format
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << std::setw(8) << std::setprecision(4) << std::fixed << p[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }
    void multiplyToTheRight(Matrix& b) { // this * b
        double *ret = Tools::mult_thread_padd(rows, p, b.p, cols, b.cols, b.cols, Tools::dim_th, Tools::n_th);
        delete[] p;
        p = nullptr;
        p = ret;

    }
    /////////////////////////////////// Matrix Operation Tools Class ///////////////////////////////////
    struct Tools {
        // Zero initialization of the block (m x n) in the matrix ("c" - start of the block, ldc - namber of colums in the matrix)
        static void init_c(int m, int n, double* c, int ldc)
        {
            for (int i = 0; i < m; i++, c += ldc)
                for (int j = 0; j < n; j += 4)
                    _mm256_storeu_pd(c + j, _mm256_setzero_pd());
        }

        // Multiplication of (6 x k) block of "a" and (k x 8) block of "b" ("b" - reordered) and streing it to (6 x 8) block in "c"
        static void kernel(int k, const double* a, const double* b, double* c, int lda, int ldb, int ldc)
        {
            __m256d a0, a1, b0, b1;

            __m256d c00 = _mm256_setzero_pd();    __m256d c01 = _mm256_setzero_pd();
            __m256d c10 = _mm256_setzero_pd();    __m256d c11 = _mm256_setzero_pd();
            __m256d c20 = _mm256_setzero_pd();    __m256d c21 = _mm256_setzero_pd();
            __m256d c30 = _mm256_setzero_pd();    __m256d c31 = _mm256_setzero_pd();
            __m256d c40 = _mm256_setzero_pd();    __m256d c41 = _mm256_setzero_pd();
            __m256d c50 = _mm256_setzero_pd();    __m256d c51 = _mm256_setzero_pd();

            const int offset0 = lda * 0;         const int offset3 = lda * 3;
            const int offset1 = lda * 1;         const int offset4 = lda * 4;
            const int offset2 = lda * 2;         const int offset5 = lda * 5;

            for (int i = 0; i < k; i++)
            {
                b0 = _mm256_loadu_pd(b + 0);                  b1 = _mm256_loadu_pd(b + 4);

                a0 = _mm256_broadcast_sd(a + offset0);        a1 = _mm256_broadcast_sd(a + offset1);

                c00 = _mm256_fmadd_pd(a0, b0, c00);           c10 = _mm256_fmadd_pd(a1, b0, c10);
                c01 = _mm256_fmadd_pd(a0, b1, c01);           c11 = _mm256_fmadd_pd(a1, b1, c11);

                a0 = _mm256_broadcast_sd(a + offset2);        a1 = _mm256_broadcast_sd(a + offset3);

                c20 = _mm256_fmadd_pd(a0, b0, c20);           c30 = _mm256_fmadd_pd(a1, b0, c30);
                c21 = _mm256_fmadd_pd(a0, b1, c21);           c31 = _mm256_fmadd_pd(a1, b1, c31);

                a0 = _mm256_broadcast_sd(a + offset4);        a1 = _mm256_broadcast_sd(a + offset5);

                c40 = _mm256_fmadd_pd(a0, b0, c40);           c50 = _mm256_fmadd_pd(a1, b0, c50);
                c41 = _mm256_fmadd_pd(a0, b1, c41);           c51 = _mm256_fmadd_pd(a1, b1, c51);

                b += ldb; a++;
            }
            _mm256_storeu_pd(c + 0, _mm256_add_pd(c00, _mm256_loadu_pd(c + 0)));
            _mm256_storeu_pd(c + 4, _mm256_add_pd(c01, _mm256_loadu_pd(c + 4)));
            c += ldc;
            _mm256_storeu_pd(c + 0, _mm256_add_pd(c10, _mm256_loadu_pd(c + 0)));
            _mm256_storeu_pd(c + 4, _mm256_add_pd(c11, _mm256_loadu_pd(c + 4)));
            c += ldc;
            _mm256_storeu_pd(c + 0, _mm256_add_pd(c20, _mm256_loadu_pd(c + 0)));
            _mm256_storeu_pd(c + 4, _mm256_add_pd(c21, _mm256_loadu_pd(c + 4)));
            c += ldc;
            _mm256_storeu_pd(c + 0, _mm256_add_pd(c30, _mm256_loadu_pd(c + 0)));
            _mm256_storeu_pd(c + 4, _mm256_add_pd(c31, _mm256_loadu_pd(c + 4)));
            c += ldc;
            _mm256_storeu_pd(c + 0, _mm256_add_pd(c40, _mm256_loadu_pd(c + 0)));
            _mm256_storeu_pd(c + 4, _mm256_add_pd(c41, _mm256_loadu_pd(c + 4)));
            c += ldc;
            _mm256_storeu_pd(c + 0, _mm256_add_pd(c50, _mm256_loadu_pd(c + 0)));
            _mm256_storeu_pd(c + 4, _mm256_add_pd(c51, _mm256_loadu_pd(c + 4)));
        }

        // Reordering of (k x 16) block of B
        static void reorder(int k, const double* b, int ldb, double* b_tmp)
        {
            for (int i = 0; i < k; i++, b += ldb, b_tmp += 8)
            {
                _mm256_storeu_pd(b_tmp + 0, _mm256_loadu_pd(b + 0));
                _mm256_storeu_pd(b_tmp + 4, _mm256_loadu_pd(b + 4));
            }
        }

        // Product of matrices A (m x k) and B (k x n)
        static void mult(int m, int k, int n, const double* a, const double* b, double* c, int lda, int ldb, int ldc)
        {
            double* b_tmp = new double[k * 8];

            for (int j = 0; j < n; j += 8)
            {
                reorder(k, b + j, ldb, b_tmp);
                for (int i = 0; i < m; i += 6)
                {
                    init_c(6, 8, c + i * ldc + j, ldc);
                    kernel(k, a + i * lda, b_tmp, c + i * ldc + j, lda, 8, ldc);
                }
            }

            delete[] b_tmp;
            b_tmp = nullptr;
        }

        static double* mult_thread(int m, const double* a, const double* b, int lda, int ldb, int ldc, int dim_thread = dim_th, int n_thread = n_th) {
            int m_t;
            try {
                thread* t = new thread[n_thread];
                double* c = new double[m * ldc];

                switch (dim_thread) {
                case 0:
                    m_t = m / n_thread;
                    for (int i = 0; i < n_thread; i++)
                        t[i] = thread([&, i]() { mult(m_t, lda, ldc, a + i * m_t * lda, b, c + i * m_t * ldc, lda, ldb, ldc); });
                    break;
                case 1:
                    m_t = ldc / n_thread;
                    for (int i = 0; i < n_thread; i++)
                        t[i] = thread([&, i]() { mult(m, lda, m_t, a, b + i * m_t, c + i * m_t, lda, ldb, ldc); });
                    break;
                default:
                    delete[] t;
                    delete[] c;
                    cerr << "Error in parameter 'dim_thread' in function 'mult_thread'." << endl;
                    return nullptr;
                }

                for (int i = 0; i < n_thread; i++)
                    t[i].join();

                delete[] t;
                t = nullptr;
                return c;
            }
            catch (const exception& e) {
                cerr << "Allocation failed: " << e.what() << endl;
                return nullptr;
            }
        }

        static double* padd_mat(const double* a, int m, int n, int new_m, int new_n) {
            try {
                double* p = new double[new_m * new_n];
                int t = 0;

                for (int i = 0, j; i < m; i++) {
                    for (j = 0; j < n; j++)
                        p[t++] = a[i * n + j];
                    for (; j < new_n; j++)
                        p[t++] = 0;
                }

                for (; t < new_m * new_n; t++)
                    p[t] = 0;

                return p;
            }
            catch (const exception& e) {
                cerr << "Allocation failed: " << e.what() << endl;
                return nullptr;
            }
        }

        static double* unpadd_mat(const double* a, int m, int n, int new_m, int new_n) {
            try {
                double* p = new double[new_m * new_n];

                if (a == nullptr) {
                    return nullptr;
                }

                for (int i = 0, j = 0, t = 0; i < new_m; i++, j += (n - new_n)) {
                    for (int k = 0; k < new_n; k++, j++, t++) {
                        p[t] = a[j];
                    }
                }

                return p;
            }
            catch (const exception& e) {
                cerr << "Allocation failed: " << e.what() << endl;
                return nullptr;
            }
        }


        static double* mult_thread_padd(int m, const double* a, const double* b, int lda, int ldb, int ldc, int dim_thread = dim_th, int n_thread = n_th) {
            int c, m_new, lda_new, ldb_new, ldc_new;

            switch (dim_thread) {
            case 0:
                c = 6 * n_thread;
                lda_new = (lda % 8 == 0) ? lda : (lda / 8) * 8 + 8;
                ldb_new = (ldb % 8 == 0) ? ldb : (ldb / 8) * 8 + 8;
                ldc_new = (ldc % 8 == 0) ? ldc : (ldc / 8) * 8 + 8;
                m_new = (m % c == 0) ? m : (m / c) * c + c;
                break;
            case 1:
                c = 8 * n_thread;
                lda_new = (lda % 8 == 0) ? lda : (lda / 8) * 8 + 8;
                ldb_new = (ldb % c == 0) ? ldb : (ldb / c) * c + c;
                ldc_new = (ldc % c == 0) ? ldc : (ldc / c) * c + c;
                m_new = (m % 6 == 0) ? m : (m / 6) * 6 + 6;
                break;
            default:
                cerr << "Error in parametr 'dim_thread' in function 'mult_thread_padd'." << endl;
                return nullptr;
            }

            double* a_padd = nullptr, * b_padd = nullptr, * c_padd = nullptr, * ret = nullptr;
            bool is_a_padd = m_new != m || lda_new != lda;
            bool is_b_padd = lda_new != lda || ldb_new != ldb;

            if (is_a_padd) a_padd = padd_mat(a, m, lda, m_new, lda_new);
            if (is_b_padd) b_padd = padd_mat(b, lda, ldb, lda_new, ldb_new);

            if (is_a_padd && is_b_padd) {

                c_padd = mult_thread(m_new, a_padd, b_padd, lda_new, ldb_new, ldc_new, dim_thread, n_thread);

                ret = unpadd_mat(c_padd, m_new, ldc_new, m, ldc);
                delete[] a_padd;
                delete[] b_padd;
                delete[] c_padd;
            }
            if (is_a_padd && !is_b_padd) {
                c_padd = mult_thread(m_new, a_padd, b, lda_new, ldb_new, ldc_new, dim_thread, n_thread);

                ret = unpadd_mat(c_padd, m_new, ldc_new, m, ldc);
                delete[] a_padd;
                delete[] c_padd;
            }
            if (!is_a_padd && is_b_padd) {
                c_padd = mult_thread(m_new, a, b_padd, lda_new, ldb_new, ldc_new, dim_thread, n_thread);

                ret = unpadd_mat(c_padd, m_new, ldc_new, m, ldc);
                delete[] b_padd;
                delete[] c_padd;
            }
            if (!is_a_padd && !is_b_padd) {
                ret = mult_thread(m_new, a, b, lda_new, ldb_new, ldc_new, dim_thread, n_thread);
            }
            return ret;
        }
        static int n_th;
        static int dim_th;
    };
};
int Matrix::Tools::n_th = 8;
int Matrix::Tools::dim_th = 1;


double VdotProduct(const Vector& v) {
    if (v.size() == 1)
        return v[0] * v[0];
    return v[0] * v[0] + v[1] * v[1];
}
void houseHolder(Vector* x, double* c) {
    double xNorm = std::sqrt(VdotProduct(*x));

    (*x)[0] += (*x)[0] < 0 ? -xNorm : xNorm;

    *c = 2.0 / VdotProduct(*x);
}
void updateR(const Vector v, const double c, Matrix& R, const size_t starting_index) {
    if (starting_index == R.rows - 1) {
        return;
    }
    double dd21, dd22;
    for (size_t  i = starting_index; i < R.cols ; ++i) {
        dd21 = R.p[starting_index * R.cols + i]; dd22 = R.p[(starting_index + 1) * R.cols + i];
        R.p[starting_index*R.cols + i] -= c * (v[0] * v[0] * dd21 + v[0] * v[1] * dd22);                // update row 0 of subR
        R.p[(starting_index+1)*R.cols + i] -= c * (v[1] * v[1] * dd22 + v[0] * v[1] * dd21);      // update row 1 of subR
    }
}

void updateQ(const Vector v, const double c, Matrix& Q, const size_t starting_col) {
    if (v.size() == 1) {
        for (int i = 0; i < Q.rows ; ++i) { // iterate rows
            Q.p[starting_col + i * Q.cols] -= - c * Q.p[i * Q.cols + starting_col] * v[0] * v[0];
        }
        return;
    }
    double dd21, dd22;
    for (size_t  i = 0; i < Q.rows; ++i) {
        dd21 = Q.p[Q.cols * i + starting_col]; dd22 = Q.p[Q.cols * i + starting_col + 1];
        Q.p[starting_col + i*Q.cols] -= c * (v[0] * v[0] * dd21 + v[0] * v[1] * dd22);                     // update column 0
        Q.p[starting_col + 1 + i*Q.cols] -= c * (v[1] * v[1] * dd22 + v[0] * v[1] * dd21);     // update column 1
    }
}

void computeQR(Matrix &R, Matrix &Q) {
    double c;
    int n = R.rows;

    for (size_t  j = 0; j < n-1; ++j) {
        Vector x(n - j);
        for (size_t  i = j, k = 0; i < n; ++i, ++k)
            x[k] = R.p[i * n + j];
        houseHolder(&x, &c); // compute Householder reflector vector
        updateR(x, c, R, j);  // apply Householder transformation to eliminate entries below the diagonal in the jth column
        updateQ(x, c, Q, j);
    }
}

void getDiagonal1AbsMin(const Matrix& A, double* min, int* pos) {
    int n = A.rows;
    *min = std::abs(A.p[1]);
    for (int i = 0; i < n - 1; ++i) {
        if (std::abs(A.p[i * n + i + 1]) < *min) {
            *min = A.p[i * n + i + 1];
            *pos = i;
        }
    }
}void getDiagonal1AbsMax(const Matrix& A, double* max, int* pos) {
    int n = A.rows;
    *max = std::abs(A.p[1]);
    for (int i = 0; i < n - 1; ++i) {
        if (std::abs(A.p[i * n + i + 1]) > *max) {
            *max = A.p[i * n + i + 1];
            *pos = i;
        }
    }
}

std::tuple<Vector, Matrix> my_eigen_recursive(Matrix& A, const double epsilon = 1e-6) {
    int n = A.rows;
    Matrix Q(n, n); 
    Matrix eigenvectors(n, n, "I");
    double u = 0.0; // for shift
    double min_diag_A = 100.0;
    int diag_arr_position = 0;

    if (n == 1) {
        Vector a(1);
        a[0] = A.p[0];
        return std::make_tuple(a, Matrix(n,n,"I"));
    }

    // QR iteration   
    while (abs(min_diag_A) > epsilon) {

        // compute QR factorization for A-uI
        A.add2Diag(-u);
        Q.setIdentity();
        computeQR(A, Q);

        // compute A = RQ + uI
        A.multiplyToTheRight(Q);
        A.add2Diag(u);

        eigenvectors.multiplyToTheRight(Q);

        u = A.p[(n - 1) * A.cols + n - 1]; // next shift

        getDiagonal1AbsMin(A, &min_diag_A, &diag_arr_position);
    }

    // get the submatrices for recursive call
    Matrix upper_mat = A.getSubMatrix(0, 0, diag_arr_position + 1, diag_arr_position + 1);
    Matrix low_mat = A.getSubMatrix(diag_arr_position + 1, diag_arr_position + 1, A.rows, A.cols);
    delete[] A.p;
    A.p = nullptr;

    // recursive call to improve performance
    auto [eigenvalues_upper, eigenvector_upper] = my_eigen_recursive(upper_mat,  epsilon);
    delete[] upper_mat.p;
    upper_mat.p = nullptr;
    auto [eigenvalues_lower, eigenvector_lower] = my_eigen_recursive(low_mat, epsilon);

    delete[] low_mat.p;
    low_mat.p = nullptr;

    // concat result eigenvalues
    Vector concatenated_eigenvalues(eigenvalues_upper.size() + eigenvalues_lower.size());
    concatenated_eigenvalues[slice(0, eigenvalues_upper.size(), 1)] = eigenvalues_upper;
    concatenated_eigenvalues[slice(eigenvalues_upper.size(), eigenvalues_lower.size(), 1)] = eigenvalues_lower;

    // concatinate v1 and v2 to a single matrix
    Matrix concateV1V2(eigenvector_upper.rows + eigenvector_lower.rows, eigenvector_upper.cols + eigenvector_lower.cols, 0.0);
    concateV1V2.setSubMatrix(eigenvector_upper, 0, 0);
    concateV1V2.setSubMatrix(eigenvector_lower, eigenvector_upper.rows, eigenvector_upper.cols);
    
    // compute eigenvectors
    eigenvectors.multiplyToTheRight(concateV1V2);
    return std::make_tuple(concatenated_eigenvalues, eigenvectors);
}
std::tuple<Vector, Matrix> my_eigen(Matrix& A, const double epsilon = 1e-6) {
    // Initialize matrices I & Q to be equal to the identity matrix 
    int n = A.rows;
    Matrix I(n, n, "I"); // identity matrix
    Matrix Q(n, n); // identity matrix
    Matrix eigenvectors(n,n, "I");
    //Matrix R(n, n);
   // R.setIsTriangular(true);
    double u = 0.0; // for shift
    double max_diag_A = 100.0;
    int diag_arr_position = 0;
   int iterations = 0;

    if (n == 1) {
        Vector a(1);
        a[0] = A.p[0];
        return std::make_tuple(a, I);
    }

    // QR iteration
    while (abs(max_diag_A) > epsilon) {

        // compute A-u*I
        
        A.add2Diag(-u);
        // compute QR factorization
       // cout << "\nA:" << endl;
        //A.toString();
        Q.setIdentity();
        computeQR(A, Q);

        // compute A = RQ
        //A = R;
        //A.setIsTriangular(true);
       // delete[] A.p;
        //A.p = nullptr;
        A.multiplyToTheRight(Q);
        //A = A*Q;
       // cout << "\nR:" << endl;
       // cout << "\nQ:" << endl;
        //Q.toString();

        // compute A+u*I
        A.add2Diag(u);
       // cout << "\nA after:" << endl;
       // A.toString
        eigenvectors.multiplyToTheRight(Q);
        //eigenvectors = eigenvectors * Q;
       // cout << "\neigenvectors:" << endl;
       // eigenvectors.toString();
        u = A.p[(n - 1) * A.cols + n - 1];
       // A.toString();
        getDiagonal1AbsMax(A, &max_diag_A, &diag_arr_position);

        ++iterations;
       // if (iterations == 1000)
          //  break;
    }
    //delete[]  Q.p;
    ////delete[] R.p;
    //Q.p = nullptr;
    //  R.p = nullptr;

    Vector eigenvalues(A.rows);
    for (int i = 0; i < A.rows; ++i)
        eigenvalues[i] = A.p[i * A.cols + i];

    return std::make_tuple(eigenvalues, eigenvectors);
}
Matrix GetColumnVector(const Matrix A,  const int start_row, const int col) {
    Matrix v(A.rows - start_row, 1);
    int n = A.cols;
    for (int i = start_row, x = 0; i < A.rows; ++i, ++x) {
        v.p[x] = A.p[i * n + col];
    }
    return v;
}

double GetVecNorm(const Matrix x) {
    double norm = 0;

    for (int i = 0; i < x.rows; i++)
        norm += x.p[i] * x.p[i];

    return sqrt(norm);
}

// Computes the Hessenberg form of a symmetric matrix A using Householder reflections.
tuple<Matrix, Matrix> HessenbergForm(Matrix H, const double epsilon) {
    int n = H.rows;
    Matrix v;
    Matrix vvT;
    Matrix Q(n, n, "I");
    double sign;
    for (int k = 0; k < n - 2; ++k) {
        v = GetColumnVector(H, k + 1, k);
        sign = (v.p[0] < 0 ? -1 : 1);
        v.p[0] += sign * GetVecNorm(v);
        v = v * (1 / GetVecNorm(v));
        vvT = v * v.t();

        // H[k+1:, k:] -= 2.0 * np.outer(v, v @ H[k+1:, k:])
        Matrix subH1 = H.getSubMatrix(k+1, k, H.rows, H.cols);
        subH1 -= 2 * vvT * subH1;
        H.setSubMatrix(subH1, k+1, k);

        // H[:, k + 1 : ] -= 2.0 * np.outer(H[:, k + 1 : ] @ v, v)
        Matrix subH2 = H.getSubMatrix(0, k+1, H.rows, H.cols);
        subH2 -= 2 *subH2 * vvT;
        H.setSubMatrix(subH2, 0, k+1);

        // Q[:, k+1:] -= 2.0 * np.outer(Q[:, k+1:] @ v, v)
        Matrix subQ = Q.getSubMatrix(0, k + 1, Q.rows, Q.cols);
        subQ -= 2 * subQ * vvT;
        Q.setSubMatrix(subQ, 0, k + 1);

        delete[] subH1.p;
        subH1.p = nullptr;
        delete[] subH2.p;
        subH2.p = nullptr;
        delete[] subQ.p;
        subQ.p = nullptr;
        delete[] v.p;
        v.p = nullptr;
        delete[] vvT.p;
        vvT.p = nullptr;
       
    }
    // Create the mask for tridiagonal elements
    std::vector<std::vector<bool>> mask(n, std::vector<bool>(n, false));
    for (int i = 0; i < n; ++i) {
        mask[i][i] = true;
        if (i + 1 < n) {
            mask[i][i + 1] = true;
            mask[i + 1][i] = true;
        }
    }

    // Assign zeros to the masked elements
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; j++) {
            if (!mask[i][j] && std::abs(H.p[i * n + j]) < epsilon) {
                H.p[i * n + j] = 0.0;
            }
        }
    }

    return { H, Q };
}

int main() {
    Matrix A("inv_matrix(800 x 800).txt");
    //int sub_n = 400;

    //Matrix a = A.getSubMatrix(0, 0, sub_n, sub_n);

    printf("\nConverting matrix to Hessenberg form...\n");
    auto [HessenbergMat, H] = HessenbergForm(A, 1e-6);
    delete[] A.p;
    A.p = nullptr;
    long double tt, duration;
    printf("\nComputing eigenvalues & eigenvectors using QR method...\n");
    tt = Get_Time();
    auto [eigenvalues, eigenvectors] = my_eigen_recursive(HessenbergMat, 1e-3);
    //auto [Q, R] = computeQR(a, a.rows);
    eigenvectors = H * eigenvectors;
    duration = Get_Time();
    duration -= tt;
    cout << "\nmy_eigen: " << duration << endl;
    //printf("\nmy_eigen_recursive: %lf\n", duration);
   // 
   // std::cout << "\nEigenvalues:" << std::endl;
   // for (double eigenvalue : eigenvalues) {
   //     std::cout << eigenvalue << std::endl;
   //}

   // std::cout << "\nEigenvectors:" << std::endl;
   // eigenvectors.toString();
    
    return 0;
}

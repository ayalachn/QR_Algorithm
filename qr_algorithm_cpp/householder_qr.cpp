#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>  // Add this line
#include <iostream>
#include <cmath>
//#include <vector>
#include <tuple>
#include <sstream>  // Add this line
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
#include <valarray>
//#include <inner_product>
#include <iostream>
#include <cmath>
#include <vector>


#include <iostream>
#include <cmath>
#include <vector>

#include <iostream>
#include <cmath>
#include <vector>

using namespace std;
typedef std::valarray<double> Vector;
//typedef std::valarray<valarray<double>> Matrix;
typedef std::vector<Vector> Matrix;

void printMatrix(const Matrix& mat, int n) {
    
    for (const Vector& row : mat) {
        for (double element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
}

Matrix multiplyMatrix(const Matrix& A, const Matrix& B) {
    Matrix result = Matrix(A[0].size(), Vector(B[1].size(), 0.0));
    int n = A.size();
    int m = B[0].size();
    int p = B.size();
    //result.resize(n, Vector(m, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < p; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}
/*
double dotProduct(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::runtime_error("Vectors must have the same size");
    }

    return std::inner_product(std::begin(v1), std::end(v1), std::begin(v2), 0.0);
}
*/
double VdotProduct(const Vector& v) {
    if (v.size() == 1)
        return v[0] * v[0];
    return v[0] * v[0] + v[1] * v[1];
}

void houseHolder(Vector* x, Vector* v, double* c) {
    // Copy vector x into v
    *v = *x;

    (*v)[0] += std::signbit((*x)[0]) ? -1.0 * std::sqrt(VdotProduct(*x)) : std::sqrt(VdotProduct(*x));

    *c = 2.0 / VdotProduct(*v);
}
void updateR(Vector v, double c, Matrix& R, int n, int starting_index) {
    if (starting_index == n - 1) {
        R[starting_index][starting_index] -= c * (v[0] * v[0] * R[starting_index][starting_index]);
        return;
    }

    // copy only what's relevant: R's first two rows.
    Matrix dd2 = Matrix(2, Vector(n- starting_index, 0.0));
    for (int i = starting_index, x=0; i < starting_index + 2; i++, x++) {
        for (int j = starting_index, y=0; j < n ; j++, y++) {
            dd2[x][y] = R[i][j];
        }
    }

    for (int i = 0; i < n - starting_index; i++) {
        R[starting_index][starting_index + i] -= c * (v[0] * v[0] * dd2[0][i] + v[0] * v[1] * dd2[1][i]);
        R[starting_index + 1][starting_index + i] -= c * ((v[0] * v[1]) * dd2[0][i] + (v[1] * v[1] * dd2[1][i]));
    }
}

void updateQ(Vector v, double c, Matrix& Q, int n, int starting_col) {
    if (v.size() == 1) {
        for (int i = 0; i < n; i++) { // iterate rows
            Q[i][starting_col] = Q[i][starting_col] - c * Q[i][starting_col] * v[0] * v[0];
        }
        return;
    }

    // copy only what's relevant: Q's first two columns.
    Matrix dd2 = Matrix(n, Vector(2, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <2;j++) {
            dd2[i][j] = Q[i][starting_col + j];
        }
    }
    for (int i = 0; i < n; i++) {
        Q[i][starting_col] -= c * (dd2[i][0] * v[0] * v[0] + dd2[i][1] * (v[0] * v[1]));
        Q[i][starting_col + 1] -= c * (dd2[i][0] * v[0] * v[1] + dd2[i][1] * (v[1] * v[1]));
    }
}
void computeQR(const Matrix& A, Matrix& Q, Matrix& R) {
    int n = A.size();
    Q = Matrix(n, Vector(n, 0.0));
    R = Matrix(n, Vector(n, 0.0));
    Vector v;
    double c;

    // Q = identity matrix
    for (int i = 0; i < n; i++) {
        Q[i][i] = 1;
    }

    // R = copy of A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; ++j) {
            R[i][j] = A[i][j];
        }
    }

    // compute QR factorization
    for (int j = 0; j < n; ++j) {
        Vector x = Vector(n-j, 1.0);
        for (int i = j, k = 0; i < n; i++, k++)
            x[k] = R[i][j];
        houseHolder(&x, &v, &c); // comoute Householder reflector vector
        updateR(v, c, R, n, j);  // Apply Householder transformation to eliminate entries below the diagonal in the jth column
        updateQ(v, c, Q, n, j);
    }
}

void multiplyScalarWithMatrix(double scalar, Matrix& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] *= scalar;
        }
    }
}
void getDiagonalAbsMin(Matrix& A, double *min, int *pos) {
    int n = A.size();
    *min = std::abs(A[0][0]);
    for (int i = 0; i < n ; i++) {
        if (std::abs(A[i][i]) < *min) {
            *min = A[i][i];
            *pos = i;
        }
    }
}
void copyMatrix(const Matrix& source, Matrix& destination) {
    int rows = source.size();
    int cols = source[0].size();

    destination.resize(rows);
    for (int i = 0; i < rows; ++i) {
        destination[i].resize(cols);
        for (int j = 0; j < cols; ++j) {
            destination[i][j] = source[i][j];
        }
    }
}

Matrix concatenateArrays(const Matrix& arr1, const Matrix& arr2) {
    if (arr1.size() != arr2.size()) {
        std::cerr << "Error: Arrays must have the same number of rows." << std::endl;
        return Matrix();
    }

    Matrix result;
    for (size_t i = 0; i < arr1.size(); ++i) {
        Vector row = arr1[i];
        std::vector<double> arr2_row(std::begin(arr2[i]), std::end(arr2[i]));
        row.resize(row.size() + arr2_row.size());
        std::copy(arr2_row.begin(), arr2_row.end(), std::begin(row) + arr1[i].size());
        result.push_back(row);
    }

    return result;
}

tuple<Vector, Matrix> my_eigen_recursive(Matrix& A, double epsilon=1e-6) {
    int n = A.size();
    Matrix I = Matrix(n, Vector(n, 0.0));
    Matrix Q = Matrix(n, Vector(n, 0.0));
    Matrix R = Matrix(n, Vector(n, 0.0));
    Matrix uI = Matrix(n, Vector(n, 0.0));
    Matrix eigenvectors = Matrix(n, Vector(n, 0.0));
    //Vector eigenvalues = Vector(n, 0.0);
    double u = 1.0; // for shift
    double min_diag_A = 100.0;
    int diag_arr_position=0;

    // Initialize matrices I & Q to be equal to the identity matrix 
    for (int i = 0; i < n; i++) {
        I[i][i] = 1.0;
        Q[i][i] = 1.0;
        eigenvectors[i][i] = 1.0;
    }

    if (n == 1) {
        Vector firstCol;
        for (int i = 0; i < n; n++)
            firstCol[i] = A[i][0];
        return {firstCol,I}; // return A[:,0], I
    }


    // QR iteration
    while (min_diag_A > epsilon) {
        copyMatrix(I, uI);
        multiplyScalarWithMatrix(u, uI);
        computeQR(A, Q, R);
        A = multiplyMatrix(R, Q);
        u = A[n - 1][n - 1];
        eigenvectors = multiplyMatrix(eigenvectors, Q);
        getDiagonalAbsMin(A, &min_diag_A, &diag_arr_position);
        printf("\n\nA:\n");
        printMatrix(A);
    }

    Matrix upper_mat(A.begin(), A.begin() + diag_arr_position + 1);
    Matrix low_mat(A.begin() + diag_arr_position + 1, A.end());

    auto [eigenvalues_upper, eigenvector_upper] = my_eigen_recursive(upper_mat, epsilon);
    auto [eigenvalues_lower, eigenvector_lower] = my_eigen_recursive(low_mat, epsilon);

    Vector concatenated_eigenvalues(eigenvalues_upper.size() + eigenvalues_lower.size());
    std::copy(std::begin(eigenvalues_upper), std::end(eigenvalues_upper), std::begin(concatenated_eigenvalues));
    std::copy(std::begin(eigenvalues_lower), std::end(eigenvalues_lower), std::begin(concatenated_eigenvalues) + eigenvalues_lower.size());

    Matrix v1;
    v1.reserve(eigenvector_upper.size() + eigenvector_lower.size());
    v1.insert(v1.end(), eigenvector_upper.begin(), eigenvector_upper.end());
    v1.resize(v1.size(), Vector(eigenvector_upper[0].size(), 0.0));

    Matrix v2;
    v2.reserve(eigenvector_upper.size() + eigenvector_lower.size());
    v2.resize(eigenvector_upper.size(), Vector(eigenvector_lower[0].size(), 0.0));
    v2.insert(v2.end(), eigenvector_lower.begin(), eigenvector_lower.end());

    eigenvectors = multiplyMatrix(eigenvectors, concatenateArrays(v1, v2));

    return { concatenated_eigenvalues, eigenvectors};
}
Vector outer(Vector vector) { // vector * vector.T
    // Compute the transpose of the vector
    Vector transpose = vector;
    int n = vector.size();
    // Create a matrix to store the result
    Vector result(vector.size() * vector.size());

    // Compute the matrix multiplication of vector and its transpose
    for (std::size_t i = 0; i < vector.size(); ++i) {
        for (std::size_t j = 0; j < vector.size(); ++j) {
            result[i * n + j] = vector[i] * transpose[j];
        }
    }

    return result;
}
void IdentityMatrix(Vector &Q, int n) { // Q must be initialized to zeros...
    Q.resize(n * n);
    for (int i = 0; i < n; i++)
        Q[i*n + i] = 1.0;
}

Vector GetColumnVector(Matrix A, int start_row, int col) {
    Vector col_vec = Vector(A.size() - start_row, 0.0);

    for (int i = start_row, x = 0; i < A.size(); i++, x++) {
        col_vec[x] = A[i][col];
    }

    return col_vec;
}
Vector GetColumnVector(Vector A, int orig_col_size, int start_row, int col) {
    Vector col_vec(orig_col_size - start_row);

    for (int i = start_row, x = 0; i < orig_col_size; i++, x++) {
        col_vec[x] = A[i * orig_col_size + col];
    }

    return col_vec;
}
double GetVetNorm(Vector x) {
    double norm = 0;

    for (int i = 0; i < x.size(); i++)
        norm += x[i] * x[i];

    return sqrt(norm);
}
// Computes the Hessenberg form of a symmetric matrix A using Householder reflections.
tuple<Matrix, Vector> HessenbergForm(Matrix A, double epsilon) {
    int n = A.size();
    int nn = n * n;
    Vector x, v;
    Vector vvT;
    Vector H = Vector(A.size()* A.size());
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            H[i * n + j] = A[i][j];
    Vector Q(nn, 0.0);
    IdentityMatrix(Q, n);

    for (int k = 0; k < n - 2; ++k) {
        x = GetColumnVector(H, n, k + 1, k);
        v = x;
        v[0] += (x[0] < 0 ? -1 : 1) * GetVetNorm(x);
        v = v / GetVetNorm(v);
        int vn = v.size();
        vvT = outer(v);

        for (int i = k + 1, x = 0; i < n-1; i++, x++) {
            for (int j = k, y = 0; j < n-1; j++, y++) {
                H[(i)*n + j] = H[(i)*n + j] - 2.0 * vvT[x * vn + y] * H[i * n + j];
            }
        }
        for (int i = 0, x = 0; i < n-1; i++, x++) {
            for (int j = k + 1, y = 0; j < n; j++, y++) {
                H[i * n + j] -= 2.0 * H[i * n + j] * vvT[x * vn + y];
                Q[i * n + j] -= 2.0 * Q[i * n + j] * vvT[x * vn + y];
            }
        }
    }
    // Create the mask for tridiagonal elements
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
        for (int j = 0; j < n; ++j) {
            if (!mask[i][j] && std::abs(H[i * n + j]) < epsilon) {
                H[i * n + j] = 0.0;
            }
        }
    }
    Matrix Hessenberg = Matrix(H.size(), H);
    return { Hessenberg, Q };
}
int main() {
    // Example usage
    Matrix A = {
        {1, 4, 2},
        {4, 2, 1},
        {2, 1, 4}
    };

   // Vector eigenvalues;
 //   Matrix eigenvectors;
    auto [HessenbergMat, H] = HessenbergForm(A, 1e-6);
    printMatrix(HessenbergMat);
    auto [eigenvalues, eigenvectors] = my_eigen_recursive(HessenbergMat, 1e-6);

    std::cout << "Eigenvalues:" << std::endl;
    for (double eigenvalue : eigenvalues) {
        std::cout << eigenvalue << std::endl;
    }

    std::cout << "Eigenvectors:" << std::endl;
    printMatrix(eigenvectors);

    return 0;
}

/*
class Matrix {
public:
    float* p;
    int rows;
    int cols;

    //static int n_th;
    //static int dim_th;

    Matrix() : p(nullptr), rows(0), cols(0) {}
    Matrix(int rows_, int cols_) : p(new float[rows_ * cols_]), rows(rows_), cols(cols_) {}
    Matrix(int rows_, int cols_, float val) : p(new float[rows_ * cols_]), rows(rows_), cols(cols_) {
        for (int i = 0; i < rows * cols; i++)
            p[i] = val;
    }
    Matrix(int rows_, int cols_, float a, float b) : p(new float[rows_ * cols_]), rows(rows_), cols(cols_) { // Random matrix a(i,j) ~ U(a, b)
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(a, b);

        for (int i = 0; i < rows * cols; i++) p[i] = dis(gen);
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
        p = new float[rows * cols];

        for (int i1 = 0, i2 = 0, j = 0; i2 < str.size(); ) {
            for (i1 = i2; !in_float(str[i1]) && i1 < str.size(); i1++) {}
            for (i2 = i1; in_float(str[i2]) && i2 < str.size(); i2++) {}
            if (i1 != i2) p[j++] = stof(str.substr(i1, i2 - i1));
        }
    }
    Matrix(Matrix& m) : p(new float[m.rows * m.cols]), rows(m.rows), cols(m.cols) {
        for (int i = 0; i < rows * cols; i++)
            p[i] = m.p[i];
    }
    Matrix(Matrix&& m) : rows(m.rows), cols(m.cols) {
        p = m.p;
        m.p = nullptr;
    }
    void IdentityMatrix() {
        for (int i = 0; i < cols; ++i)
            p[i + i * cols] = 1.0;
    }
    friend bool eq(Matrix& a, Matrix& b) { return a.rows == b.rows && a.cols == b.cols; }
    friend bool eq(Matrix&& a, Matrix&& b) { return a.rows == b.rows && a.cols == b.cols; }
    friend bool eq(Matrix&& a, Matrix& b) { return a.rows == b.rows && a.cols == b.cols; }
    friend bool eq(Matrix& a, Matrix&& b) { return a.rows == b.rows && a.cols == b.cols; }

    Matrix& operator = (Matrix& m) {
        if (p == m.p) return *this;
        if (eq(*this, m)) {
            for (int i = 0; i < rows * cols; i++)
                p[i] = m.p[i];
        }
        else {
            delete[] p;
            rows = m.rows;
            cols = m.cols;
            p = new float[rows * cols];
            for (int i = 0; i < rows * cols; i++)
                p[i] = m.p[i];
        }
        return *this;
    }
    Matrix& operator = (Matrix&& m) {
        if (p == m.p) return *this;
        p = m.p;
        m.p = nullptr;
        return *this;
    }
    ~Matrix() {
        if (p) delete[] p;
    }
    float& operator () (int i, int j) {
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
    friend Matrix operator * (Matrix& a, float b) {
        Matrix prod(a.rows, a.cols);
        for (int i = 0; i < a.rows * a.cols; i++)
            prod.p[i] = a.p[i] * b;
        return move(prod);
    }
    friend Matrix operator * (Matrix&& a, float b) {
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

    friend Matrix operator * (float b, Matrix& a) { return a * b; }
    friend Matrix operator * (float b, Matrix&& a) { return a * b; }
    friend Matrix operator * (Matrix&& a, Matrix&& b) {
        if (a.cols != b.rows) {
            cerr << "Error of matrix size in operator *." << endl;
            return Matrix();
        }
        Matrix ret;
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

    /////////////////////////////////// Matrix Operation Tools Class ///////////////////////////////////
    struct Tools {
        // Zero initialization of the block (m x n) in the matrix ("c" - start of the block, ldc - namber of colums in the matrix)
        static void init_c(int m, int n, float* c, int ldc)
        {
            for (int i = 0; i < m; i++, c += ldc)
                for (int j = 0; j < n; j += 8)
                    _mm256_storeu_ps(c + j, _mm256_setzero_ps());
        }

        // Multiplication of (6 x k) block of "a" and (k x 16) block of "b" ("b" - reordered) and streing it to (6 x 16) block in "c"
        static void kernel(int k, const float* a, const float* b, float* c, int lda, int ldb, int ldc)
        {
            __m256 a0, a1, b0, b1;

            __m256 c00 = _mm256_setzero_ps();    __m256 c01 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps();    __m256 c11 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps();    __m256 c21 = _mm256_setzero_ps();
            __m256 c30 = _mm256_setzero_ps();    __m256 c31 = _mm256_setzero_ps();
            __m256 c40 = _mm256_setzero_ps();    __m256 c41 = _mm256_setzero_ps();
            __m256 c50 = _mm256_setzero_ps();    __m256 c51 = _mm256_setzero_ps();

            const int offset0 = lda * 0;         const int offset3 = lda * 3;
            const int offset1 = lda * 1;         const int offset4 = lda * 4;
            const int offset2 = lda * 2;         const int offset5 = lda * 5;

            for (int i = 0; i < k; i++)
            {
                b0 = _mm256_loadu_ps(b + 0);                  b1 = _mm256_loadu_ps(b + 8);

                a0 = _mm256_broadcast_ss(a + offset0);        a1 = _mm256_broadcast_ss(a + offset1);

                c00 = _mm256_fmadd_ps(a0, b0, c00);           c10 = _mm256_fmadd_ps(a1, b0, c10);
                c01 = _mm256_fmadd_ps(a0, b1, c01);           c11 = _mm256_fmadd_ps(a1, b1, c11);

                a0 = _mm256_broadcast_ss(a + offset2);        a1 = _mm256_broadcast_ss(a + offset3);

                c20 = _mm256_fmadd_ps(a0, b0, c20);           c30 = _mm256_fmadd_ps(a1, b0, c30);
                c21 = _mm256_fmadd_ps(a0, b1, c21);           c31 = _mm256_fmadd_ps(a1, b1, c31);

                a0 = _mm256_broadcast_ss(a + offset4);        a1 = _mm256_broadcast_ss(a + offset5);

                c40 = _mm256_fmadd_ps(a0, b0, c40);           c50 = _mm256_fmadd_ps(a1, b0, c50);
                c41 = _mm256_fmadd_ps(a0, b1, c41);           c51 = _mm256_fmadd_ps(a1, b1, c51);

                b += ldb; a++;
            }
            _mm256_storeu_ps(c + 0, _mm256_add_ps(c00, _mm256_loadu_ps(c + 0)));
            _mm256_storeu_ps(c + 8, _mm256_add_ps(c01, _mm256_loadu_ps(c + 8)));
            c += ldc;
            _mm256_storeu_ps(c + 0, _mm256_add_ps(c10, _mm256_loadu_ps(c + 0)));
            _mm256_storeu_ps(c + 8, _mm256_add_ps(c11, _mm256_loadu_ps(c + 8)));
            c += ldc;
            _mm256_storeu_ps(c + 0, _mm256_add_ps(c20, _mm256_loadu_ps(c + 0)));
            _mm256_storeu_ps(c + 8, _mm256_add_ps(c21, _mm256_loadu_ps(c + 8)));
            c += ldc;
            _mm256_storeu_ps(c + 0, _mm256_add_ps(c30, _mm256_loadu_ps(c + 0)));
            _mm256_storeu_ps(c + 8, _mm256_add_ps(c31, _mm256_loadu_ps(c + 8)));
            c += ldc;
            _mm256_storeu_ps(c + 0, _mm256_add_ps(c40, _mm256_loadu_ps(c + 0)));
            _mm256_storeu_ps(c + 8, _mm256_add_ps(c41, _mm256_loadu_ps(c + 8)));
            c += ldc;
            _mm256_storeu_ps(c + 0, _mm256_add_ps(c50, _mm256_loadu_ps(c + 0)));
            _mm256_storeu_ps(c + 8, _mm256_add_ps(c51, _mm256_loadu_ps(c + 8)));
        }

        // Reordering of (k x 16) block of B
        static void reorder(int k, const float* b, int ldb, float* b_tmp)
        {
            for (int i = 0; i < k; i++, b += ldb, b_tmp += 16)
            {
                _mm256_storeu_ps(b_tmp + 0, _mm256_loadu_ps(b + 0));
                _mm256_storeu_ps(b_tmp + 8, _mm256_loadu_ps(b + 8));
            }
        }

        // Product of matrices A (m x k) and B (k x n)
        static void mult(int m, int k, int n, const float* a, const float* b, float* c, int lda, int ldb, int ldc)
        {
            float* b_tmp = new float[k * 16];

            for (int j = 0; j < n; j += 16)
            {
                reorder(k, b + j, ldb, b_tmp);
                for (int i = 0; i < m; i += 6)
                {
                    init_c(6, 16, c + i * ldc + j, ldc);
                    kernel(k, a + i * lda, b_tmp, c + i * ldc + j, lda, 16, ldc);
                }
            }

            delete[] b_tmp;
        }

        // Multithreaded product of matrices A (m x k) and B (k x n)
        static float* mult_thread(int m, const float* a, const float* b, int lda, int ldb, int ldc, int dim_thread = dim_th, int n_thread = n_th) {
            int m_t;
            thread* t = new thread[n_thread];
            float* c = new float[m * ldc];

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
                cerr << "Error in parametr 'dim_thread' in function 'mult_thread'." << endl;
                return nullptr;
            }
            for (int i = 0; i < n_thread; i++)
                t[i].join();

            return c;
        }

        static float* padd_mat(const float* a, int m, int n, int new_m, int new_n) {
            float* p = new float[new_m * new_n];
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

        static float* unpadd_mat(const float* a, int m, int n, int new_m, int new_n) {
            float* p = new float[new_m * new_n];

            for (int i = 0, j = 0, t = 0; i < new_m; i++, j += (n - new_n))
                for (int k = 0; k < new_n; k++, j++, t++)
                    p[t] = a[j];

            return p;
        }

        static float* mult_thread_padd(int m, const float* a, const float* b, int lda, int ldb, int ldc, int dim_thread = dim_th, int n_thread = n_th) {
            int c, m_new, lda_new, ldb_new, ldc_new;

            switch (dim_thread) {
            case 0:
                c = 6 * n_thread;
                lda_new = (lda % 16 == 0) ? lda : (lda / 16) * 16 + 16;
                ldb_new = (ldb % 16 == 0) ? ldb : (ldb / 16) * 16 + 16;
                ldc_new = (ldc % 16 == 0) ? ldc : (ldc / 16) * 16 + 16;
                m_new = (m % c == 0) ? m : (m / c) * c + c;
                break;
            case 1:
                c = 16 * n_thread;
                lda_new = (lda % 16 == 0) ? lda : (lda / 16) * 16 + 16;
                ldb_new = (ldb % c == 0) ? ldb : (ldb / c) * c + c;
                ldc_new = (ldc % c == 0) ? ldc : (ldc / c) * c + c;
                m_new = (m % 6 == 0) ? m : (m / 6) * 6 + 6;
                break;
            default:
                cerr << "Error in parametr 'dim_thread' in function 'mult_thread_padd'." << endl;
                return nullptr;
            }

            float* a_padd = nullptr, * b_padd = nullptr, * c_padd = nullptr, * ret = nullptr;
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
///int Matrix::Tools::n_th = 8;
//int Matrix::Tools::dim_th = 1;


float* mult(int M, int K, int N, float* A, float* B) {

    float* C = new float[M * N];
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] = 0;

    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
    return C;

}
/*
class Matrix {
public:
    Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows, std::vector<double>(cols)) {}

    int rows() const {
        return rows_;
    }

    int cols() const {
        return cols_;
    }

    double& operator()(int row, int col) {
        return data_[row][col];
    }

    const double& operator()(int row, int col) const {
        return data_[row][col];
    }

private:
    int rows_;
    int cols_;
    std::vector<std::vector<double>> data_;
};

Matrix ReadMatrixFromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file " << path << std::endl;
        return Matrix(0, 0);
    }

    std::vector<std::vector<double>> matrixData;
    std::string line;
    int rows = 0, cols = 0;

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::istringstream iss(line);
        double value;

        while (iss >> value) {
            row.push_back(value);
        }

        if (!row.empty()) {
            matrixData.push_back(row);
            ++rows;
            if (cols == 0) {
                cols = row.size();
            }
            else if (row.size() != cols) {
                std::cerr << "Error: Inconsistent number of columns at row " << rows << std::endl;
                return Matrix(0, 0);
            }
        }
    }

    file.close();

    Matrix matrix(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = matrixData[i][j];
        }
    }

    return matrix;
}
void PrintMatrix(const Matrix& matrix) {
    int n = matrix.cols;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << matrix.p[i * matrix.cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

double VecNorm(valarray<float> vec) {
    return std::sqrt((vec * vec).sum());
}

valarray<float> GetColumnVector(Matrix mat, int row_num, int col_num) {
    valarray<float> col_vec;
    for (int i = row_num; i < mat.cols; ++i) {
        col_vec[i] = mat.p[i*mat.cols + col_num];
    }
    return col_vec;
}
*/

/*
int main() {
   // std::string filePath = "inv_matrix(800 x 800).txt";
   // Matrix matrix = ReadMatrixFromFile(filePath);
    //double data[3][3] = { {0.1411, 0.9489, 0.3635},{0.9489, 0.8513, 0.376},{0.3635, 0.376, 0.7033 } };

    Matrix matrix("inv_matrix(800 x 800).txt");
   // matrix.rows = 3;
   // matrix.cols = 3;
    // Check if the matrix was successfully read
    if (matrix.rows == 0 || matrix.cols == 0) {
        std::cerr << "Error: Failed to read matrix from file" << std::endl;
        return 1;
    }
    std::cout << "Loaded matrix successfully, matrix has " << matrix.cols << " cols and " << matrix.rows << " rows" << std::endl;
    PrintMatrix(matrix);
    
    // convert matrix to Hessenberg form
    HessenbergForm(matrix, 1e-6);
    PrintMatrix(matrix);
    return 0;
}
*/
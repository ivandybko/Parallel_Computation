#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>


using namespace std;

const int n = 4;
const int b = 2;
const int max_block_size = b;

void fill(
    std::vector<double>& A,
    std::vector<double>& B)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            A[i * n + j] = cos(i + j);
            B[i * n + j] = sin(i - j);
        }
}


void fill(
    std::vector<double>& A)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            A[i * n + j] = cos(i + j);
        }
}


void zero(std::vector<double>& A)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i * n + j] = 0.0;
}


void DisplayFlattenMatrix(const vector<double>& A) {
    int n = sqrt(A.size());

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << A[n * i + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

// A: m x n
void DisplayFlattenMatrix(const vector<double>& A, int m, int n) {

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << A[n * i + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}


void FlattenMatrixDiff(const vector<double>& A, const vector<double>& B, vector<double>& res) {
    int n = A.size();
    for (int i = 0; i < n; i++) {
        res[i] = A[i] - B[i];
    }
}


void FlattenMatrixSum(const vector<double>& A, const vector<double>& B, vector<double>& res) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        res[i] = A[i] + B[i];
    }
}

void LU(vector<double>& A) {
    int n = sqrt(A.size());
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            A[j * n + i] /= A[i * n + i];
        }

        for (int j = i + 1; j < n; j++) {
            for (int k = i + 1; k < n; k++) {
                A[j * n + k] -= A[j * n + i] * A[i * n + k];
            }
        }
    }
}

void FindLowerTriangleMatrix(vector<double>& L, const vector<double>& U, const vector<double>& A, int n_local) {
    // U: d x d, L: n-d x d, A: n-d x d,
    // int d{static_cast<int>(sqrt(U.size()))}; 
    int nd{n_local -b }; // nd=n-d
    double sum{ 0.0 };
    for (int i = 0; i < nd; i++) {
        for (int j = 0; j < b; j++) {
            // L[i][j] = A[i][j] - sum( L[i][k] * U[k][j],k=0, j-1)
            sum = 0.0;
            for (int k = 0; k <= j - 1; k++) {
                sum += L[i * b + k] * U[k * b + j];
            }
            if (fabs(U[j * b + j]) < 1e-12) {
                cout << "Warning: Zero pivot in FindLowerTriangleMatrix at (" << j << "," << j << ")" << endl;
            }
            L[i * b + j] = (A[i * b + j] - sum) / U[j * b + j];
        }
    }
}

void FindUpperTriangleMatrix(const vector<double>& L, vector<double>& U, const vector<double>& A) {
    // L: b x b (lower triangular), U: b x (n-b), A: b x (n-b)
    // U[i][j] = (A[i][j] - sum(L[i][k] * U[k][j], k=0, i-1)) / L[i][i]
    int rows = b;
    int cols = U.size() / b;
    double sum = 0.0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum = 0.0;
            for (int k = 0; k < i; k++) {
                sum += L[i * b + k] * U[k * cols + j];
            }
            U[i * cols + j] = (A[i * cols + j] - sum);
        }
    }
}


// L: m x k, U: k x n, result: m x n
void MultiplyLU(const vector<double>& L, const vector<double>& U, vector<double>& result, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int p = 0; p < k; p++) {
                result[i * n + j] += L[i * k + p] * U[p * n + j];
            }
        }
    }
}


// A: m x l, B: l x n, res: m x n
void MultiplyAB(const vector<double>& A, const vector<double>& B, int m, int l, int n, vector<double>& res) {
    fill(res.begin(), res.end(), 0.0);
    for (int i = 0; i < m; ++i)
        for (int k = 0; k < l; ++k)
            for (int j = 0; j < n; ++j)
                res[i * n + j] += A[i * l + k] * B[k * n + j];
}


void LU_bl(vector<double>& A, int n) {

    if (n <= max_block_size){
        DisplayFlattenMatrix(A);
        LU(A);
        return;
    }

    vector<double> A11(b * b), A12(b * (n - b)), A21((n - b) * b), A22((n - b) * (n - b));
    // A11: b * b, A12: b * (n-b), A21: (n-b) * b, A22: (n-b)*(n-b)

    // заполняем блоки
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < b; j++) {
            A11[i * b + j] = A[i * n + j];
        }
        for (int j = 0; j < n - b; j++) {
            A12[i * (n - b) + j] = A[i * n + j + b];
        }
    }
    for (int i = 0; i < n - b; i++) {
        for (int j = 0; j < b; j++) {
            A21[i * b + j] = A[(i + b) * n + j];
        }
        for (int j = 0; j < n - b; j++) {
            A22[i * (n - b) + j] = A[(i + b) * n + j + b];
        }
    }

    // Вычисляем LU для A11
    LU(A11);
    // Получаем A12 = L11 * U12
    std::vector<double> U12(b * (n - b));


    FindUpperTriangleMatrix(A11, U12, A12);

    // Получаем A21 = L21 * U11

    vector<double> L21(b * (n - b));
    FindLowerTriangleMatrix(L21, A11, A21, n);

    // Преобразовываем матрицу A22
    vector<double> A22_mul((n - b) * (n - b), 0.0);
    MultiplyAB(L21, U12, n - b, b, n - b, A22_mul);
    FlattenMatrixDiff(A22, A22_mul, A22);

    LU_bl(A22, n - b);

    // cout << "A11" << endl;
    // DisplayFlattenMatrix(A11);
    // cout << "A12" << endl;
    // DisplayFlattenMatrix(A12,b, n-b);
    // cout << "A21" << endl;
    // DisplayFlattenMatrix(A21, n-b, b);
    // cout << "A22" << endl;
    // DisplayFlattenMatrix(A22);
    // cout << "local n = " << n << endl;
    // все собрать в одну матрицу 



    for (int i = 0; i < b; i++) {

        //A11
        for (int j = 0; j < b; j++) {
            A[i * n + j] = A11[i * b + j];
        }
        //A12
        for (int j = 0; j < n - b; j++) {
            A[i * n + j + b] = U12[i * (n - b) + j];
        }
    }

    for (int i = 0; i < n - b; i++) {

        //A21
        for (int j = 0; j < b; j++) {
            A[(i + b) * n + j] = L21[i * b + j];
        }

        //A22
        for (int j = 0; j < n - b; j++) {
            A[(i + b) * n + j + b] = A22[i * (n - b) + j];
        }
    }

}

void MultiplyLU(const vector<double>& L, const vector<double>& U, vector<double>& result) {
    int size = sqrt(L.size());
    MultiplyLU(L, U, result, size, size, size);
}


int main()
{
    vector<double> A(n * n);
    fill(A);

    //DisplayFlattenMatrix(A);

    //LU(A);
    //cout << "Simple LU:" << endl;

    //DisplayFlattenMatrix(A);

    //fill(A);

    //LU_bl(A, n);
    //cout << "Block LU" << endl;
    //DisplayFlattenMatrix(A);

    // vector<double> A({ 1,2,-3,4,5,6,-7,8,9,-10,11,-12,13,14, 15, 16 });

    DisplayFlattenMatrix(A);
    cout << "Simple LU" << endl;
    LU(A);
    DisplayFlattenMatrix(A);
    // A = { 1,2,-3,4,5,6,-7,8,9,-10,11,-12,13,14, 15, 16 };
    fill(A);
    LU_bl(A, n);
    cout << "Block LU" << endl;
    DisplayFlattenMatrix(A);

    //     vector<double> result (2*3,0);
    //     FindUpperTriangleMatrix(L, U, A);
    //     MultiplyAB(L, U, 3, 3, 2, result);
    //     cout << "Matrix A:" << endl;
    //     cout << "Matrix LU:" << endl;
    //     DisplayFlattenMatrix(result, 2, 3);
    //     cout << "Matrix U:" << endl;
    //     DisplayFlattenMatrix(U, 3, 2);
    //     // DisplayFlattenMatrix(L, 2, 3);
}
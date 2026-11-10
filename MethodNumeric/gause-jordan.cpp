#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace std;
#define Matrix vector<vector<double>>
int N = 10;


Matrix genran(int rows, int cols, double minVal, double maxVal);
Matrix trian(Matrix* , Matrix*  , int);
double det(Matrix , int);
void afficher(Matrix  , int);
int main() {

    Matrix A = genran(N , N , 0 , 100)
    int rows = 10, cols = 10;
    Matrix U(N, vector<int>(N, 0));   
    Matrix L(N, vector<int>(N, 0));   
    trian(U* , L* , N);
    afficher(U , N);
    cout << "################" << endl;
    afficher(L , N);
    cout << "determinant : " << det(B , N) << endl;

       return 0;
}


Matrix trian(Matrix *A , Matrix *L, int N){

    double alpha = 0;
    for (int c = 0; c < N-1 ; c++){
        if(A[c][c] == 0){
            cout << "error!" << endl;
            return A;
        }


        for(int l=c+1 ; l < N ; l++) {
                alpha = - A[l][c] / A[c][c];
                for (int j =c; j < N ; j++){
                    A[l][j] = A[l][j] + alpha * A[c][j];
                }
        }
    }


    return A;
}


void afficher(Matrix A, int N){
    
 for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout <<fixed << setprecision(2) << setw(10) << A[i][j];
        }
        cout << "\n";
    }


}

double det(Matrix A , int N){
    double ans = 1;
    int sign = 1;
    for (int i = 0; i < N; i++){
        sign *=-1;
        ans *=A[i][i]; 
    }
    return ans;
}


Matrix genran(int rows, int cols, double minVal, double maxVal) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(minVal, maxVal);

    Matrix matrix(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dist(gen);
        }
    }
    return matrix;
}   

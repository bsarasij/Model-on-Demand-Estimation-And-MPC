#include "Header.h"

//void main()
//{
//    int ny = 2;
//    int nu = 3;
//    int ll = 0;
//    Matrix <double, 2, 2> NA;
//    Matrix <double, 2, 3> NB;
//    Matrix <double, 2, 3> NK;
//    Matrix <double, 20, 5> z;
//    cout << "initialize" << endl;
//    NA << 2, 2,
//        2, 2;
//    NB << 3, 3, 3,
//        3, 3, 3;
//    NK << 1, 1, 1,
//        1, 1, 1;
//
//    z << 1, 2, 3, 4, 5,
//        6, 7, 8, 9, 10,
//        11, 12, 13, 14, 15,
//        16, 17, 18, 19, 20,
//        21, 22, 23, 24, 25,
//        26, 27, 28, 29, 30,
//        31, 32, 33, 34, 35,
//        36, 37, 38, 39, 40,
//        41, 42, 43, 44, 45,
//        46, 47, 48, 49, 50,
//        51, 52, 53, 54, 55,
//        56, 57, 58, 59, 60,
//        61, 62, 63, 64, 65,
//        66, 67, 68, 69, 70,
//        71, 72, 73, 74, 75,
//        76, 77, 78, 79, 80,
//        81, 82, 83, 84, 85,
//        86, 87, 88, 89, 90,
//        91, 92, 93, 94, 95,
//        96, 97, 98, 99, 100;
//
//    int si = max((NA + MatrixXd::Ones(NA.rows(), NA.cols())).maxCoeff(), (NB + NK).maxCoeff());
//    MatrixXd x(13, 1);
//
//    VectorXd samp;  // Empty sample range (use full range)
//    int n = z.rows();
//    VectorXd nr;
//    if (samp.size() == 0) {
//        nr.resize(n - si);
//        iota(nr.begin(), nr.end(), si);
//    }
//    else if (*min_element(samp.begin(), samp.end()) < si || *max_element(samp.begin(), samp.end()) > n) {
//        cout << "Validation range out of bounds. Ignoring!" << endl;
//        nr.resize(n - si);
//        iota(nr.begin(), nr.end(), si);
//    }
//    else {
//        nr = samp;
//    }
//
//    for (int ii : nr) {
//        cout << "ii = " << ii << endl;
//        cout << " " << endl;
//        for (int ll = 0; ll < ny; ++ll) {
//
//            for (int k = 0, i = 0; k < ny; ++k) {
//
//                for (int na = 0; na < NA(ll, k); ++na) {
//                    if (ii - na - 1 >= 0) {
//                        x(i++) = z(ii - na - 1, k);
//                    }
//                }
//            }
//
//            //cout << x << endl;
//            int offset = NA.row(ll).sum();
//            for (int k = 0, i = 0; k < nu; ++k) {
//                for (int nb = 0; nb < NB(ll, k); ++nb) {
//                    if (ii - NK(ll, k) - nb - 1 >= 0) {
//                        x(offset + (i++)) = z(ii - (int)NK(ll, k) - nb, ny + k);
//                    }
//                }
//
//            }
//            cout << x <<endl;
//            cout << " " << endl;
//        }
//    }
//
//}

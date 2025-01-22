#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <cstring>  
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <numeric>

using namespace std;
using namespace Eigen;

// Define struct to hold the output S
struct LocPolResult {
    double var;           // Variance estimate
    double conf;          // Confidence interval
    VectorXd k;           // Number of nearest neighbors for each evaluation
    VectorXd h;           // Bandwidth values used
    VectorXd gof;         // Goodness-of-fit values
    VectorXd Y_selected;  // Y values
    MatrixXd X_selected;  // X values
    int kopt;             // Optimal number of nearest neighbors
    int dim;              // Dimension of the regressor
    VectorXd resid;       // Residuals (Y - Y_est)
};

tuple<VectorXd, MatrixXd, VectorXd, VectorXd>
normsort(const VectorXd& Y,
    const MatrixXd& X,
    MatrixXd M = MatrixXd(),
    VectorXd V = VectorXd(),
    int kmax = -1);

VectorXd call_kernel_function(const char* kernel_type, const VectorXd& u);

// Function to construct the local design matrix
tuple<MatrixXd, int> mkdesign(const MatrixXd& X, int pol_ord) {
    int d = X.rows();
    int n = X.cols();
    MatrixXd Xx;

    switch (pol_ord) {
    case 0:
        Xx = MatrixXd::Ones(1, n);
        break;
    case 1:
        Xx = MatrixXd(d + 1, n);
        Xx.row(0) = RowVectorXd::Ones(n);
        Xx.block(1, 0, d, n) = X;
        break;
    case 2: {
        vector<int> i, j;
        for (int row = 0; row < d; ++row) {
            for (int col = 0; col <= row; ++col) {
                i.push_back(row);
                j.push_back(col);
            }
        }
        int num_quadratic_terms = i.size();
        Xx = MatrixXd(1 + d + num_quadratic_terms, n);
        Xx.row(0) = RowVectorXd::Ones(n);
        Xx.block(1, 0, d, n) = X;

        for (size_t k = 0; k < i.size(); ++k) {
            Xx.row(d + 1 + k) = X.row(j[k]).array() * X.row(i[k]).array();
        }
        break;
    }
    default:
        throw invalid_argument("The polynomial order must be either 0, 1, or 2.");
    }

    int npar = Xx.rows();
    return make_tuple(Xx, npar);
}

// locpol function
tuple<VectorXd, double, LocPolResult> locpol(
    VectorXd Y, MatrixXd X, const VectorXd& x, int pol_ord = 2, MatrixXd M = MatrixXd(),
    const char* kern = "gaussian", const char* minmethod = "globmin",
    VectorXd v = VectorXd(), vector<int> ks = {}, vector<int> ind = {}
) {
    int d = X.rows(), n = X.cols();

    if (Y.size() != n || x.size() != d) {
        throw invalid_argument("Wrong data format. Dimensions of Y, X, or x do not match.");
    }

    if (M.size() == 0) {
        M = MatrixXd::Identity(d, d).cwiseQuotient(X.rowwise().squaredNorm().asDiagonal().toDenseMatrix());
    }

    if (v.size() == 0) {
        v = VectorXd::Ones(n);
    }

    int kmin = ks.empty() ?  10 : ks[0];
    int kmax = ks.empty() ? min(300, n) : min((int)ks.size() > 1 ? ks[1] : ks[0], n);

    if (ind.empty()) {
        ind = vector<int>(d);
        iota(ind.begin(), ind.end(), 0);
    }

    MatrixXd X_centered = X.colwise() - x;

    VectorXd dist;
    tie(Y, X_centered, dist, v) = normsort(Y, X_centered, M, v, kmax);

    int npar;
    MatrixXd Xx;
    tie(Xx, npar) = mkdesign(X_centered(ind, all), pol_ord);
    //cout << Xx.block(0,0,7,5) << endl;
    kmin = max({ 2 * npar - 1, pol_ord + 3, kmin });

    double alpha =  (1.0 + 0.3 / d);

    int N = (kmax > kmin) ? ceil(log(dist(kmax - 1) / dist(kmin - 1)) / log(alpha)) : 1;
    if (kmax < kmin) {
        throw invalid_argument("KMAX must be larger than or equal to KMIN.");
    }

    MatrixXd locdata(2, N);
    MatrixXd B = MatrixXd::Zero(npar, N);
    VectorXd GOF(N), variances(N), normH(N);

    int k = kmin;
    double h = dist(kmin - 1);

    for (int i = 0; i < N; ++i) {
        locdata(0, i) = k;
        locdata(1, i) = h;

        VectorXd weights = call_kernel_function(kern, dist.head(k) / dist(k - 1));

        SparseMatrix<double> W(k, k), V(k, k);
        for (int j = 0; j < k; ++j) {
            W.insert(j, j) = weights[j];
            V.insert(j, j) = v[j];
        }


        MatrixXd Xxk = Xx.leftCols(k);


        MatrixXd sqrtVW = (V.cwiseSqrt() * W.cwiseSqrt()).toDense();

        HouseholderQR<MatrixXd> qr(sqrtVW * Xxk.transpose());
        MatrixXd Q = qr.householderQ();
        MatrixXd R = qr.matrixQR().triangularView<Upper>().toDenseMatrix();
        MatrixXd H = R.colPivHouseholderQr().solve(Q.topRows(k).transpose().eval()) * sqrtVW;

       

        VectorXd bhat = H * Y.head(k);
        B.col(i) = bhat;

        VectorXd residuals = Y.head(k) - Xxk.transpose() * bhat;
        double ws = weights.sum();
        variances[i] = (residuals.array().square() * weights.array()).sum() / abs((ws - Xxk.rows()));

        GOF[i] = (residuals.array().square() * weights.array()).sum() / ws;

        h *= alpha;
        k = (dist.array() <= h).count();
    }

    // Minimization method section
    int minind;
    double min_GOF;
    //cout << GOF << " " << "idx: " << minind << " " << "val: " << min_GOF << endl;
    if (strcmp(minmethod, "globmin") == 0) {
        min_GOF = GOF.minCoeff(&minind);
    }
    else {
        throw runtime_error("Unknown loss minimizer");
    }
    double variance = variances[minind];
    double crit = 1.96;
    double conf = crit * sqrt(variance) * normH[minind];

    // Prepare output struct S
    LocPolResult S;
    S.var = variance;
    S.conf = conf;
    S.k = locdata.row(0);
    S.h = locdata.row(1);
    S.gof = GOF;
    S.kopt = locdata(0, minind);
    S.Y_selected = Y.head(S.kopt);
    S.X_selected = X(Eigen::all, Eigen::seq(0, S.kopt - 1));
    S.dim = d;
    //cout << (B.col(minind).transpose() * Xx.leftCols(S.kopt)).rows() << endl;
    S.resid = S.Y_selected - (B.col(minind).transpose() * Xx.leftCols(S.kopt)).transpose();

    return make_tuple(B.col(minind), locdata(1, minind), S);
}


// // Code to test the locpol function
//int main() {
//    // Example input data
//    int num_samples = 654;
//    int num_features = 8;
//
//    // Generate random data for Y and X
//    VectorXd Y = VectorXd::Random(num_samples);         // Random output vector
//    MatrixXd X = MatrixXd::Random(num_features, num_samples);  // Random input matrix
//    VectorXd x = VectorXd::Random(num_features);        // Query point
//
//    // Polynomial order
//    int pol_ord = 1;
//
//    // Optional inputs
//    MatrixXd M = MatrixXd::Identity(num_features, num_features);  // Identity matrix for weights
//    VectorXd v = VectorXd::Ones(num_samples);  // Uniform variance weights
//    vector<int> ks = { 30, 50 };  // Range of nearest neighbors
//
//    try {
//        // Call the local polynomial regression function
//        VectorXd B_opt;
//        double h_opt;
//        LocPolResult S;
//        std::tie(B_opt, h_opt, S) = locpol(Y, X, x, pol_ord, M, "gaussian", "fpe", "globmin", v, ks);
//
//        // Display the results
//        cout << "Optimal B (coefficients):\n" << B_opt.transpose() << endl;
//        cout << "Optimal bandwidth h: " << h_opt << endl;
//        cout << "Variance estimate: " << S.var << endl;
//        cout << "Confidence interval: " << S.conf << endl;
//        cout << "Optimal number of nearest neighbors: " << S.kopt << endl;
//        cout << "Selected Y values:\n" << S.Y_selected.transpose() << endl;
//        cout << "Residuals:\n" << S.resid.transpose() << endl;
//    }
//    catch (const std::exception& e) {
//        cerr << "Error: " << e.what() << endl;
//    }
//
//    return 0;
//}
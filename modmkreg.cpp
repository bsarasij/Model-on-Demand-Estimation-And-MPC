#include "Header.h"

// Function to compute standard deviation of a vector
VectorXd computeStdDev(const MatrixXd& mat) {
    VectorXd stddev(mat.rows());
    for (int i = 0; i < mat.rows(); ++i) {
        double mean = mat.row(i).mean();
        double sum_sq_diff = (mat.row(i).array() - mean).square().sum();
        stddev(i) = sqrt(sum_sq_diff / mat.cols());
    }
    return stddev;
}

RegressionResult modmkreg(const MatrixXd& z, const vector<int>& nn, double ve = 0.0) {
    int n = z.rows();
    int m = z.cols();

    int ny = 1;  // Number of outputs (for our case, ny = 1)
    int nu = m - ny;  // Number of inputs (for our case, nu = 3-1 = 2)

    // Validate dimensions of nn
     if (2 * m - ny != nn.size()) {
        throw invalid_argument("Incorrect number of orders specified!");
    }

    // Split the nn vector into NA, NB, NK components
    vector<int> NA(nn.begin(), nn.begin() + ny);
    vector<int> NB(nn.begin() + ny, nn.begin() + ny + nu);
    vector<int> NK(nn.begin() + ny + nu, nn.begin() + ny + 2 * nu);

    // Compute start index
    int si = *max_element(NA.begin(), NA.end()) + 1;
    for (size_t i = 0; i < NB.size(); ++i) {
        si = max(si, NB[i] + NK[i]);
    }

    int max_reg = accumulate(NA.begin(), NA.end(), 0) + accumulate(NB.begin(), NB.end(), 0);

    if (n <= si) {
        throw runtime_error("Not enough data.");
    }

    

    //VectorXd y = z.block(si - 1, 0, n - si + 1, ny).transpose().eval();
    //cout << z.block(si - 1, 0, n - si + 1, ny).eval() << endl;
    VectorXd y = z.middleRows(si - 1, n - si + 1).leftCols(ny);

    // Pre-allocate regressor matrices
    vector<MatrixXd> phi(ny, MatrixXd::Zero(max_reg, n - si + 1));
    vector<MatrixXd> M(ny, MatrixXd::Zero(max_reg, max_reg));

    for (int ll = 0; ll < ny; ++ll) {
        int regs = accumulate(NA.begin(), NA.end(), 0) + accumulate(NB.begin(), NB.end(), 0);
        for (int ii = si; ii <= n; ++ii) {
            int jj = ii - si;

            // Handle output regressors
            int index = 0;
            for (int kk = 0; kk < ny; ++kk) {
                for (int p = 0; p < NA[kk]; ++p) {
                    phi[ll](index++, jj) = z(ii - p - 1, kk);
                }
            }
            
            // Handle input regressors
            for (int kk = 0; kk < nu; ++kk) {
                for (int p = 0; p < NB[kk]; ++p) {
                    phi[ll](index++, jj) = z(ii - NK[kk] - p, kk + ny);
                }
            }
        }
        
        // Default scaling matrix (inverse covariance)
        MatrixXd phiT = phi[ll].transpose();
        VectorXd stddev = computeStdDev(phiT);
        for (int i = 0; i < regs; ++i) {
            M[ll](i, i) = 1.0 / (stddev(i) * stddev(i));
        }
        /*cout << phi[ll](all,0)<< endl;
        cout << z.block(0, 0, 10, 3) << endl;*/
    }

    VectorXd v;

    return { y, phi, M, v };
}

//int main() {
//    // Example dataset (800 rows, 3 columns)
//    MatrixXd z(800, 3);
//    z.setRandom(); // Filling with random values
//
//    vector<int> nn = { 1, 1, 1, 1, 1 };  // Example ARX model
//
//    try {
//        RegressionResult result = modmkreg(z, nn, 0.0);
//
//        cout << "Output vector (y):\n" << result.y.transpose() << "\n";
//        cout << "Regressor matrix (phi):\n" << result.phi[0] << "\n";
//        cout << "Scaling matrix (M):\n" << result.M[0] << "\n";
//    }
//    catch (const exception& e) {
//        cerr << "Error: " << e.what() << "\n";
//    }
//
//    return 0;
//}

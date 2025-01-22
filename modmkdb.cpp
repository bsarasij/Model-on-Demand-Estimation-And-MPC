#include <vector>
#include <cmath>
#include <tuple>
#include <Eigen/Dense>
#include <iostream>


using namespace Eigen;
using namespace std;

struct EstimationData {
    Eigen::VectorXd Y;      // Output of estimation data set 
    Eigen::MatrixXd X;  // Regressor of estimation data set 
    Eigen::VectorXd var;    // Variance
    Eigen::MatrixXd M;      // Scaling Matrix
    Eigen::MatrixXd nn;        // Model order [na nb nk]
    int size;                   // Size of the dataset
    double T;                   // Sampling interval
};

struct RegressionResult {
    VectorXd y;
    vector<MatrixXd> phi;
    vector<MatrixXd> M;
    VectorXd v;
};

// Function Declaration for modmkreg
RegressionResult modmkreg(const Eigen::MatrixXd& z, const std::vector<int>& nn, double ve);

EstimationData modmkdb(const Eigen::MatrixXd& z, const std::vector<int>& nn, double T = 1.0, double ve = 0.0) {
    // Ensure ve is non-negative
    ve = std::abs(ve);

    // Call the modmkreg function to populate Y, X, var, and M
    Eigen::VectorXd Y, var;
    Eigen::MatrixXd X, M;

    RegressionResult result = modmkreg(z, nn, ve);

    MatrixXd eigen_nn(1,nn.size());
    for (int i = 0; i < nn.size(); ++i) {
        eigen_nn(0,i) = nn[i];
    }


    EstimationData Z;
    Z.Y = result.y;
    Z.X = result.phi[0];
    Z.var = result.v;
    Z.M = result.M[0];
    Z.nn = eigen_nn;
    Z.size = X.size()==0 ? 0 : X.cols();

    // Check if T is a valid numeric value (default to 1 if invalid)
    Z.T = (std::isnan(T) || !std::isfinite(T)) ? 1.0 : T;

    return Z;
}

//int main() {
//    // Example input data matrix (800 samples, 3 columns - output + 2 inputs)
//    int n = 800;  // Number of samples
//    int m = 3;    // Number of columns (1 output + 2 inputs)
//
//    MatrixXd z = MatrixXd::Random(n, m);  // Generate random data for testing
//
//    // ARX model orders and delays [0, 1, 1, 0, 0] (for our case)
//    vector<int> nn = { 1, 1, 1, 1, 1};
//
//    // Sampling time
//    double T = 1.0;  // Default sampling time
//
//    // Noise variance estimate
//    double ve = 0.01;  // Some small noise value
//
//    // Call the modmkdb function with test inputs
//    EstimationData result = modmkdb(z, nn, T, ve);
//
//    // Print results to verify correctness
//    cout << "Output Y (size: " << result.Y.size() << "):" << endl;
//    for (size_t i = 0; i < result.Y.size(); ++i) {
//        cout << result.Y[i] << " ";
//    }
//    cout << "\n\n";
//
//    cout << "Regressor matrix X (size: " << result.X.size() << " x "
//        << result.X.cols() << "):" << endl;
//    for (size_t i = 0; i < result.X.rows(); ++i) {
//        for (size_t j = 0; j < result.X.cols(); ++j) {
//            cout << result.X(i,j) << " ";
//        }
//        cout << endl;
//    }
//
//    cout << "\nNoise variance vector (size: " << result.var.size() << "):" << endl;
//    for (double v : result.var) {
//        cout << v << " ";
//    }
//    cout << "\n\n";
//
//    cout << "Matrix M (size: " << result.M.size() << " x "
//        << result.M.cols() << "):" << endl;
//    for (size_t i = 0; i < result.M.rows(); ++i) {
//        for (size_t j = 0; j < result.M.cols(); ++j) {
//            cout << result.M(i,j) << " ";
//        }
//        cout << endl;
//    }
//
//    cout << "\nModel order nn: ";
//    for (int val : result.nn) {
//        cout << val << " ";
//    }
//    cout << endl;
//
//    cout << "Size of regressor matrix X: " << result.size << endl;
//    cout << "Sampling interval T: " << result.T << endl;
//
//    return 0;
//}

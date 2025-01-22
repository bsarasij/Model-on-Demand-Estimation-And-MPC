#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <algorithm>
#include <numeric>

using namespace std;
using namespace Eigen;

// Function to normalize a matrix
MatrixXd normalizeMatrix(const MatrixXd& M) {
    double normVal = M.norm();  // Frobenius norm
    if (normVal == 0) {
        throw runtime_error("Matrix norm is zero, cannot normalize.");
    }
    return M / normVal;
}

// Custom sorting function based on computed distances
vector<int> sortIndices(const VectorXd& dist) {
    vector<int> indices(dist.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&dist](int i, int j) {
        return dist(i) < dist(j);
        });
    return indices;
}

// Main normsort function
tuple<VectorXd, MatrixXd, VectorXd, VectorXd> normsort(
    const VectorXd& Y,
    const MatrixXd& X,
    MatrixXd M = MatrixXd(),
    VectorXd V = VectorXd(),
    int kmax = -1
) {
    int dim = X.rows();
    int n = X.cols();

    // Default M to identity if not provided
    if (M.size() == 0) {
        M = MatrixXd::Identity(dim, dim);
    }

    // Normalize the matrix M
    M = normalizeMatrix(M);

    // Default kmax to number of columns (n)
    if (kmax == -1) {
        kmax = n;
    }

    // Compute distances using M-norm: sqrt(1' * (X .* (M * X)))
    VectorXd dist = ((X.array()).cwiseProduct((M * X).array())).colwise().sum().cwiseSqrt();

    // Sort indices based on distance
    vector<int> sortedIndices = sortIndices(dist);

    // Take the top kmax elements
    sortedIndices.resize(kmax);

    // Sort distance vector based on sorted indices
    VectorXd sortedDist(kmax);
    for (int i = 0; i < kmax; ++i) {
        sortedDist(i) = dist(sortedIndices[i]);
    }

    // Select sorted Y values and corresponding columns from X
    VectorXd Y_sorted(kmax);
    MatrixXd X_sorted(dim, kmax);
    for (int i = 0; i < kmax; ++i) {
        Y_sorted(i) = Y(sortedIndices[i]);
        X_sorted.col(i) = X.col(sortedIndices[i]);
    }

    // Select values from V if provided
    VectorXd V_sorted;
    if (V.size() > 0) {
        V_sorted.resize(kmax);
        for (int i = 0; i < kmax; ++i) {
            V_sorted(i) = V(sortedIndices[i]);
        }
    }
    else {
        V_sorted = VectorXd::Zero(kmax);
    }

    return make_tuple(Y_sorted, X_sorted, sortedDist, V_sorted);
}

// //main
//int main() {
//    //  example data
//    int dim = 3;
//    int n = 10;
//    VectorXd Y = VectorXd::LinSpaced(n, 1, 10);
//    MatrixXd X = MatrixXd::Random(dim, n);
//    MatrixXd M = MatrixXd::Identity(dim, dim);
//    VectorXd V = VectorXd::Random(n);
//
//    // Call normsort
//    try {
//        VectorXd Y_sorted, dist, V_sorted;
//        MatrixXd X_sorted;
//        tie(Y_sorted, X_sorted, dist, V_sorted) = normsort(Y, X, M, V, 5);
//
//        // Print results
//        cout << "Sorted Y:\n" << Y_sorted.transpose() << endl;
//        cout << "Sorted X:\n" << X_sorted << endl;
//        cout << "Distances:\n" << dist.transpose() << endl;
//        cout << "Sorted V:\n" << V_sorted.transpose() << endl;
//    }
//    catch (const exception& e) {
//        cerr << "Error: " << e.what() << endl;
//    }
//
//    return 0;
//}

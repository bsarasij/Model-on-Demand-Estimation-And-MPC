#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace Eigen;

// Function to generate random data in the range [-1, 1]
double random_value() {
    return static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
}

// Function to generate ARX dataset and return it as an Eigen matrix
MatrixXd generate_arx_dataset() {
    const int num_samples = 100;  // Fixed number of samples
    MatrixXd dataset(num_samples, 3); // 3 columns: Y (output), u1, u2 (inputs)

    VectorXd Y(num_samples);
    VectorXd U1(num_samples);
    VectorXd U2(num_samples);

    // Initialize with random values
    for (int i = 0; i < 4; ++i) {
        Y(i) = random_value();
        U1(i) = random_value();
        U2(i) = random_value();
    }

    // ARX model coefficients.
    double a1 = 0.2, a2 = -0.1;  // Coefficients for Y
    double b1 = 0.9, b2 = -0.3;  // Coefficients for U1
    double c1 = 0.7, c2 = 0.1;   // Coefficients for U2
    double d1 = 0.5, d2 = -0.1;  // Current input coefficients

    // Generate the ARX time series data
    for (int t = 2; t < num_samples; ++t) {
        U1(t) = random_value();
        U2(t) = random_value();

        Y(t) = a1 * Y(t - 1) + a2 * Y(t - 2)
             + b1 * U1(t - 1) + b2 * U1(t - 2)
             + c1 * U2(t - 1) + c2 * U2(t - 2)
             + d1 * U1(t) + d2 * U2(t);
    }

    // Store the data in Eigen matrix format
    for (int i = 0; i < num_samples; ++i) {
        dataset(i, 0) = Y(i);
        dataset(i, 1) = U1(i);
        dataset(i, 2) = U2(i);
    }

    return dataset;
}

//int main() {
//    srand(static_cast<unsigned>(time(0)));  // Seed random number generator
//
//    // Generate the ARX dataset
//    MatrixXd dataset = generate_arx_dataset();
//
//    // Print first 10 samples
//    cout << "First 10 samples of the generated dataset:\n";
//    cout << dataset.topRows(10) << endl;
//
//    return 0;
//}

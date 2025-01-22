#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <stdexcept>
#include <cmath>
#include <optional> // For std::optional
#include <utility>  // For std::pair 
#include <numeric>  // For iota

using namespace std;
using namespace Eigen;

// structure to hold components of modmkdb
struct EstimationData {
    Eigen::VectorXd Y;      // Output of estimation data set (1x800)
    Eigen::MatrixXd X;  // Regressor of estimation data set (2x800*1)
    Eigen::VectorXd var;    // Variance
    Eigen::MatrixXd M;      // Model data
    Eigen::MatrixXd nn;        // Model order [na nb nk]
    int size;                   // Size of the dataset
    double T;                   // Sampling interval
};

// Define a struct to hold all outputs of modcmp_simplified
struct ModCmpOutputs {
    MatrixXd yho;             // Predicted outputs
    MatrixXd kopto;           // Optimal bandwidths
    MatrixXd confl;           // Confidence lower bounds
    MatrixXd confh;           // Confidence upper bounds
    MatrixXd beta_estimate;   // Beta coefficients
};

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
    VectorXd resid;       // Residuals
};


// Compute the difference of two sets
void setdiff(const vector<int>& set1, const vector<int>& set2, vector<int>& result) {
    for (int x : set1) {
        if (find(set2.begin(), set2.end(), x) == set2.end()) {
            result.push_back(x);
        }
    }
}
// Declaration for local polynomial generator
std::tuple<Eigen::VectorXd, double, LocPolResult> locpol(
    Eigen::VectorXd Y,
    Eigen::MatrixXd X,
    const Eigen::VectorXd& x,
    int pol_ord,
    Eigen::MatrixXd M,
    const char* kern,
    const char* minmethod,
    Eigen::VectorXd v,
    std::vector<int> ks,
    std::vector<int> ind
);


bool valueExists(const Eigen::MatrixXd& mat, double value) {
    return (mat.array() == value).any();
}

// Function implementation
ModCmpOutputs modcmp(MatrixXd& z, const EstimationData& Z, int M_opt,
    const vector<int>& km, MatrixXd Sc_opt,
    const char* guimm, const VectorXd& samp, int p = 1) {

    // Argument checking
    int n = z.rows();    // Number of samples in validation dataset
    int m = z.cols();    // Columns: 1 output + N inputs
    int ny = Z.nn.rows(); // Number of outputs
    int nu = (Z.nn.cols() - ny) / 2; // Number of inputs

    if (m != ny + nu) {
        throw invalid_argument("Validation dataset format is incorrect.");
    }

    VectorXd var;
    if (Z.var.size() == 0) {
        var = VectorXd::Ones(ny);
    }
    else {
        var = Z.var;
    }

    if (var.size() != ny) {
        throw invalid_argument("Length of variance vector is incorrect.");
    }
    //cout << numeric_limits<int>::max() << endl;
    // M is the prediction horizon
    int M = (M_opt == 1) ? M_opt : INT_MAX;

    MatrixXd Sc = Sc_opt.size() > 0 ? Sc_opt : Z.M;
    //pair<string, double> gof = gof_opt.size()>0 ? gof_opt.value() : make_pair("CP", log(Z.size));

    //  Theta is always zero since ind1 is empty
    RowVectorXd theta = RowVectorXd::Zero(1); // Always zero

    // Initialize output matrices
    int reg_num = (p == 1) ? Z.X.rows() : Z.X.rows() + Z.X.rows() * (Z.X.rows() + 1) / 2;
    MatrixXd yho = MatrixXd::Zero(n, ny);
    MatrixXd kopto = MatrixXd::Zero(n, ny);
    MatrixXd confl = MatrixXd::Zero(n, ny);
    MatrixXd confh = MatrixXd::Zero(n, ny);
    MatrixXd beta_estimate = MatrixXd::Zero(reg_num + 1, n);

    // Extract NA, NB, and NK from Z.nn
    MatrixXd NA = Z.nn.leftCols(ny);
    MatrixXd NB = Z.nn.middleCols(ny, nu);
    MatrixXd NK = Z.nn.rightCols(nu);

    int si = max((NA + MatrixXd::Ones(NA.rows(), NA.cols())).maxCoeff(), (NB + NK).maxCoeff());

    // Determine sampling range (nr)

    VectorXd nr;
    if (samp.size() == 0) {
        nr.resize(n - si);
        iota(nr.begin(), nr.end(), si);
    }
    else if (*min_element(samp.begin(), samp.end()) < si || *max_element(samp.begin(), samp.end()) > n) {
        std::cout << "Validation range out of bounds. Ignoring!" << endl;
        nr.resize(n - si);
        iota(nr.begin(), nr.end(), si);
    }
    else {
        nr = samp;
    }

    //std::cout << z.block(0,0,10,3) << endl;
    //  Main Loop
    if (M == 1 || M == numeric_limits<int>::max()) {
        for (int ii : nr) {
            for (int ll = 0; ll < ny; ++ll) {
                int regs = NA.row(ll).sum() + NB.row(ll).sum();
                VectorXd x = VectorXd::Zero(regs);

                // Build regressor
                for (int k = 0, x_ind = 0; k < ny; ++k) {
                    for (int na = 0; na < NA(ll, k); ++na) {
                        if (ii - na - 1 >= 0) {
                            x(x_ind++) = z(ii - na - 1, k);
                            //std::cout << z(ii - na - 1, k) << endl;

                        }
                    }
                }
                /*std::cout << "b" << endl;
                std::cout << x << endl;*/

                int offset = NA.row(ll).sum();
                for (int k = 0, x_ind = 0; k < nu; ++k) {
                    for (int nb = 0; nb < NB(ll, k); ++nb) {
                        if (ii - NK(ll, k) - nb - 1 >= 0) {
                            x(offset + (x_ind++)) = z(ii - (int)NK(ll, k) - nb, ny + k);
                            //cout << ii - (int)NK(ll, k) - nb << endl;
                        }
                    }
                }

                Eigen::VectorXd beta;
                double hopt;
                LocPolResult S;

                //cout << x << endl;
                //cout << " " << endl;
                // Perform local polynomial regression
                {
                    tie(beta, hopt, S) = locpol(Z.Y, Z.X, x, p, Sc, "gaussian", guimm, {}, km, {});
                }
                //std::cout << " a" << endl;

                /*cout << beta << endl;
                cout << " " << endl;*/
                // Update outputs
                yho(ii, ll) = beta(0);
                beta_estimate.col(ii) = beta;
                kopto(ii, ll) = S.kopt;
                confl(ii, ll) = beta(0) - S.conf;
                confh(ii, ll) = beta(0) + S.conf;

                //if (M == numeric_limits<int>::max()) {
                //    z(ii, ll) = yho(ii, ll); // Replace value in pure simulation
                //}
            }
        }
    }



    // 8. Return outputs
    return { yho, kopto, confl, confh, beta_estimate };
}


//int main() {
//    // Define test data for the validation dataset (z)
//    MatrixXd z(20, 3);
//    z << 1, 2, 3,
//        4, 5, 6,
//        7, 8, 9,
//        10, 11, 12,
//        13, 14, 15,
//        16, 17, 18,
//        19, 20, 21,
//        22, 23, 24,
//        25, 26, 27,
//        28, 29, 30,
//        31, 32, 33,
//        34, 35, 36,
//        37, 38, 39,
//        40, 41, 42,
//        43, 44, 45,
//        46, 47, 48,
//        49, 50, 51,
//        52, 53, 54,
//        55, 56, 57,
//        58, 59, 60;
//    MatrixXd nn(1,5);
//    nn << 2, 2, 2, 1, 1;
//    // Define the EstimationData data (training dataset information)
//    EstimationData Z;
//    Z.Y = VectorXd::LinSpaced(20, 1, 10);  // Outputs
//    Z.X = MatrixXd::Random(6, 20);         // Regressors
//    Z.var = VectorXd::Ones(1);             // Variance
//    Z.M = MatrixXd::Identity(6, 6);        // Scaling matrix
//    Z.nn = nn;           // Model structure
//    Z.size = Z.X.cols();                   // Number of samples in training set
//
//    // Optional inputs for modcmp
//    MatrixXd M_opt = MatrixXd();  // No optimal prediction horizon
//    vector<int> km = { 5, 50 };    // Range of neighbors for locpol
//    MatrixXd Sc_opt = MatrixXd(); // No scaling matrix provided
//    const char* guimm = "globmin";  // Minimization method
//    VectorXd samp;  // Empty sample range (use full range)
//    int p = 1;  // Polynomial order
//
//    try {
//        // Call the modcmp function
//        ModCmpOutputs result = modcmp(z, Z, M_opt, km, Sc_opt, guimm, samp, p);
//
//        // Print the results
//        cout << "Predicted Outputs (yho):\n" << result.yho << "\n\n";
//        cout << "Optimal Bandwidths (kopto):\n" << result.kopto << "\n\n";
//        cout << "Confidence Lower Bounds (confl):\n" << result.confl << "\n\n";
//        cout << "Confidence Upper Bounds (confh):\n" << result.confh << "\n\n";
//        cout << "Beta Coefficients (beta_estimate):\n" << result.beta_estimate << "\n\n";
//    }
//    catch (const std::exception& e) {
//        cerr << "Error: " << e.what() << endl;
//    }
//
//    return 0;
//}
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

// Define the Z structure
struct Z_Struct {
    VectorXd Y;        // Output vector (1xN)
    MatrixXd X;        // Input matrix (NxN)
    VectorXd var;      // Variance vector
    MatrixXd M;        // Scaling matrix
    MatrixXd nn;       // Structure matrix (NA, NB, NK)
    int size;          // Size of dataset (used for log(Z.size) in `gof`)
};

// Define the output structure for locpol
struct LocpolOutput {
    VectorXd beta; // Regression coefficients
    double hopt;   // Optimal bandwidth
    struct {
        int kopt;      // Optimal neighborhood size
        double conf;   // Confidence radius
        double var;    // Variance of the estimate
    } S;
};

// Define a struct to hold all outputs of modcmp_simplified
struct ModCmpOutputs {
    MatrixXd yho;             // Predicted outputs
    MatrixXd kopto;           // Optimal bandwidths
    MatrixXd confl;           // Confidence lower bounds
    MatrixXd confh;           // Confidence upper bounds
    MatrixXd beta_estimate;   // Beta coefficients
};

// Helper function: Compute the difference of two sets
void setdiff(const vector<int>& set1, const vector<int>& set2, vector<int>& result) {
    for (int x : set1) {
        if (find(set2.begin(), set2.end(), x) == set2.end()) {
            result.push_back(x);
        }
    }
}

// Placeholder locpol function
LocpolOutput locpol(const VectorXd& Y, const MatrixXd& X, const VectorXd& x, int p, const MatrixXd& Sc,
    const pair<string, double>& gof, const string& guimm, const VectorXd& var,
    const vector<int>& km, const vector<int>& ind2) {
    // Placeholder implementation
    LocpolOutput output;
    output.beta = VectorXd::Constant(X.rows(), 0.5); // Example beta
    output.hopt = 1.0; // Example hopt
    output.S.kopt = 10; // Example kopt
    output.S.conf = 0.1; // Example confidence radius
    output.S.var = 0.05; // Example variance
    return output;
}


bool valueExists(const Eigen::MatrixXd& mat, double value) {
    return (mat.array() == value).any();
}

// Function implementation
ModCmpOutputs modcmp_simplified(const MatrixXd& z, const Z_Struct& Z, MatrixXd M_opt,
    const vector<int>& km, MatrixXd Sc_opt,
    pair<string, double> gof_opt,
    const string& guimm, const vector<int>& samp, int p = 1) {
    // 1. Argument checking
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

    // 2. Default values for inputs
    int M = M_opt.size()>0 ? M_opt.value() : numeric_limits<int>::max();
    //MatrixXd Sc = Sc_opt.size() > 0 ? Sc_opt.value() : Z.M;
    //pair<string, double> gof = gof_opt.size()>0 ? gof_opt.value() : make_pair("CP", log(Z.size));

    // 3. Theta is always zero since ind1 is empty
    RowVectorXd theta = RowVectorXd::Zero(1); // Always zero

    // 4. Initialize output matrices
    int reg_num = (p == 1) ? Z.X.rows() : Z.X.rows() + Z.X.rows() * (Z.X.rows() + 1) / 2;
    MatrixXd yho = MatrixXd::Zero(n, ny);
    MatrixXd kopto = MatrixXd::Zero(n, ny);
    MatrixXd confl = MatrixXd::Zero(n, ny);
    MatrixXd confh = MatrixXd::Zero(n, ny);
    MatrixXd beta_estimate = MatrixXd::Zero(reg_num + 1, n);

    // 5. Extract NA, NB, and NK from Z.nn
    MatrixXd NA = Z.nn.leftCols(ny);
    MatrixXd NB = Z.nn.middleCols(ny, nu);
    MatrixXd NK = Z.nn.rightCols(nu);

    int si = max((NA + MatrixXd::Ones(NA.rows(), NA.cols())).maxCoeff(), (NB + NK).maxCoeff());

    // 6. Determine sampling range (nr)
    vector<int> nr;
    if (samp.empty()) {
        nr.resize(n - si + 1);
        iota(nr.begin(), nr.end(), si);
    }
    else if (*min_element(samp.begin(), samp.end()) < si || *max_element(samp.begin(), samp.end()) > n) {
        cout << "Validation range out of bounds. Ignoring!" << endl;
        nr.resize(n - si + 1);
        iota(nr.begin(), nr.end(), si);
    }
    else {
        nr = samp;
    }

    // 7. Main Loop
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
                        }
                    }
                }
                int offset = NA.row(ll).sum();
                for (int k = 0,x_ind = 0; k < nu; ++k) {
                    for (int nb = 0; nb < NB(ll, k); ++nb) {
                        if (ii - NK(ll, k) - nb - 1 >= 0) {
                            x(offset + (x_ind++)) = z( ii - (int)NK(ll, k) - nb - 1, ny + k);
                        }
                    }
                }

                // Perform local polynomial regression
                LocpolOutput locpol_result = locpol(Z.Y, Z.X, x, p, Sc, gof, guimm, var, km, {});

                // Update outputs
                yho(ii, ll) = locpol_result.beta(0);
                beta_estimate.col(ii) = locpol_result.beta;
                kopto(ii, ll) = locpol_result.S.kopt;
                confl(ii, ll) = locpol_result.beta(0) - locpol_result.S.conf;
                confh(ii, ll) = locpol_result.beta(0) + locpol_result.S.conf;

                if (M == numeric_limits<int>::max()) {
                    z(ii, ll) = yho(ii, ll); // Replace value in pure simulation
                }
            }
        }
    }

    // 8. Return outputs
    return { yho, kopto, confl, confh, beta_estimate };
}



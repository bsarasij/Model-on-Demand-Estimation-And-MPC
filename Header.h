#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <optional> // For std::optional
#include <utility>  // For std::pair 
#include <numeric>  // For iota
#include <algorithm>
#include <tuple>
#include <cstring>  
#include <Eigen/Sparse>
#include <cstdlib>
#include <ctime>


#define _USE_MATH_DEFINES


using namespace std;
using namespace Eigen;


using namespace std;
using namespace Eigen;

struct EstimationData {
    Eigen::VectorXd Y;      // Output of estimation data set 
    Eigen::MatrixXd X;  // Regressor of estimation data set
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
    MatrixXd beta_estimate;   // regerssor coefficients
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

struct RegressionResult {
    VectorXd y;
    vector<MatrixXd> phi;
    vector<MatrixXd> M;
    VectorXd v;
};


// Declaration for normsort
tuple<VectorXd, MatrixXd, VectorXd, VectorXd>
normsort(const VectorXd& Y,
    const MatrixXd& X,
    MatrixXd M = MatrixXd(),
    VectorXd V = VectorXd(),
    int kmax = -1);

//Declaration for kernel weight generator
VectorXd call_kernel_function(const char* kernel_type, const VectorXd& u);

// Function Declaration for Data Generation for Estimation and Validation
MatrixXd generate_arx_dataset();
EstimationData modmkdb(const Eigen::MatrixXd& z, const std::vector<int>& nn, double T, double ve);
ModCmpOutputs modcmp(MatrixXd& iodatav, const EstimationData& Z, int M,
    const vector<int>& km, MatrixXd Sc,
    const char* guimm, const VectorXd& samp, int P);



// Function Declaration for modmkreg
RegressionResult modmkreg(const Eigen::MatrixXd& z, const std::vector<int>& nn, double ve);


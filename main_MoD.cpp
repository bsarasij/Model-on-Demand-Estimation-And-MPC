#include "Header.h"

int main() {

    //ARX model orders and delays 
    vector<int> nn = { 2, 2, 2, 1, 1};
   
    MatrixXd iodata; // Estimation Dataset
    MatrixXd iodatav; // Validation Dataset 
    
    srand(static_cast<unsigned>(time(0)));  // Seed random number generator

    // Generate the ARX dataset
    iodata = generate_arx_dataset();
    iodatav = generate_arx_dataset();

    double T = 1.0; //sampling time

    // Noise variance estimate
    double ve = 0.01;  // Some small noise value

    // Call the modmkdb function to generate the estimation regressor space
    EstimationData Z = modmkdb(iodata, nn, T, ve);

    int M = INT_MAX; // infinite prediction horizon
    vector<int> km = { 5, 50 };    // Range of neighborhood size for local polynomial modeling

    int P = 1; // Polynomial Order
    MatrixXd Sc = Z.M; // Scaling Matrix
    const char* guimm = "globmin";  // Minimization method
    VectorXd samp;  // Empty sample range (use full range)

    try {
        // Call the modcmp function
        ModCmpOutputs result = modcmp(iodatav, Z, M, km, Sc, guimm, samp, P);

        // Print the results
        MatrixXd true_est(100,2);
        true_est.col(0) << result.yho;
        true_est.col(1) = iodatav.block(0,0,all,1);
        cout << "Predicted Outputs (yho) vs True Output:\n" << true_est << "\n\n";

        cout << "Optimal Bandwidths (kopto):\n" << result.kopto << "\n\n";
        cout << "Confidence Lower Bounds (confl):\n" << result.confl << "\n\n";
        cout << "Confidence Upper Bounds (confh):\n" << result.confh << "\n\n";
        cout << "Beta Coefficients (beta_estimate):\n" << result.beta_estimate.block(0,0,7,10) << "\n\n";
    }
    catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
    }

}
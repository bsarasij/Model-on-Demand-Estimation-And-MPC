#include "Header.h"

// Gaussian 
VectorXd gaussian_kernel(const VectorXd& u) {
    return (-0.5 * u.array().square()).exp() / sqrt(2 * M_PI);
}

// Epanechnikov 
VectorXd epanechnikov_kernel(const VectorXd& u) {
    VectorXd result = VectorXd::Zero(u.size());
    for (int i = 0; i < u.size(); ++i) {
        if (std::abs(u[i]) <= 1) {
            result[i] = 0.75 * (1 - u[i] * u[i]);
        }
    }
    return result;
}

// Uniform 
VectorXd uniform_kernel(const VectorXd& u) {
    VectorXd result = VectorXd::Zero(u.size());
    for (int i = 0; i < u.size(); ++i) {
        if (std::abs(u[i]) <= 1) {
            result[i] = 0.5;
        }
    }
    return result;
}

// Triangular
VectorXd triangular_kernel(const VectorXd& u) {
    VectorXd result = VectorXd::Zero(u.size());
    for (int i = 0; i < u.size(); ++i) {
        if (std::abs(u[i]) <= 1) {
            result[i] = 1 - std::abs(u[i]);
        }
    }
    return result;
}

// Biweight 
VectorXd biweight_kernel(const VectorXd& u) {
    VectorXd result = VectorXd::Zero(u.size());
    for (int i = 0; i < u.size(); ++i) {
        if (std::abs(u[i]) <= 1) {
            result[i] = 15.0 / 16.0 * pow(1 - u[i] * u[i], 2);
        }
    }
    return result;
}

// Triweight 
VectorXd triweight_kernel(const VectorXd& u) {
    VectorXd result = VectorXd::Zero(u.size());
    for (int i = 0; i < u.size(); ++i) {
        if (std::abs(u[i]) <= 1) {
            result[i] = 35.0 / 32.0 * pow(1 - u[i] * u[i], 3);
        }
    }
    return result;
}

// kernel function 
VectorXd call_kernel_function(const char* kernel_type, const VectorXd& u) {
    if (strcmp(kernel_type, "gaussian") == 0) {
        return gaussian_kernel(u);
    }
    else if (strcmp(kernel_type, "epanechnikov") == 0) {
        return epanechnikov_kernel(u);
    }
    else if (strcmp(kernel_type, "uniform") == 0) {
        return uniform_kernel(u);
    }
    else if (strcmp(kernel_type, "triangular") == 0) {
        return triangular_kernel(u);
    }
    else if (strcmp(kernel_type, "biweight") == 0) {
        return biweight_kernel(u);
    }
    else if (strcmp(kernel_type, "triweight") == 0) {
        return triweight_kernel(u);
    }
    else {
        throw std::invalid_argument("Unknown kernel type: " + std::string(kernel_type));
    }
}

//int main() {
//    // Define input values for kernel function
//    VectorXd u(7);
//    u << -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5;
//
//    // List of kernel types to test
//    const char* kernel_types[] = {
//        "gaussian",
//        "epanechnikov",
//        "uniform",
//        "triangular",
//        "biweight",
//        "triweight",
//        "unknown"  // Invalid kernel type for error testing
//    };
//
//    // Loop through each kernel type and test the function
//    for (const char* kernel : kernel_types) {
//        try {
//            cout << "Kernel Type: " << kernel << endl;
//            VectorXd result = call_kernel_function(kernel, u);
//            cout << "Result: " << result.transpose() << endl << endl;
//        }
//        catch (const std::exception& e) {
//            cout << "Error: " << e.what() << endl << endl;
//        }
//    }
//
//    return 0;
//}
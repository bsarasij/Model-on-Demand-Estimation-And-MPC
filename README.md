# Model-on-Demand-Estimation (MoD)

<p aligh = "center">
  <image src = "https://github.com/user-attachments/assets/c82796fa-11e1-42ce-8215-af695b7946ed">
</p>


<br>
<br>

## Algorithm for MoD
<br>
<p aligh = "center">
  <image src = "https://github.com/user-attachments/assets/3da861a9-2df9-4e8a-98f8-2651a3c6e6ea">
  </p>


<br>
<br>

## Bias-Variance Tradeoff in MoD
<br>
<p aligh = "center">
  <image src = "https://github.com/user-attachments/assets/2cbc7e1d-825e-4dc4-beea-47ba1acb4771">
  </p>

## MoD in conjunction with MPC
<br>
<p aligh = "center">
  <image src = "https://github.com/user-attachments/assets/5be2e748-5a40-481b-8058-8478cf452914">
</p>

## Function Description
<br>
Main File is used to run the algorithm. It uses a random time series generator to create synthetic data for testing the algorithm.
modmkdb - Takes the input-output data and sends to modmkreg to create the regressor matrices
modmkreg - Generates the regressors
modcmp - Wrapper around locpol
locpol - Local polynomial generator. Returns coefficients of the time series model, the prediction, and the neighborhood size at each time instant.
call_kernel_function - Provides an array of kernel weights to choose from.
normsort - sorts the distance of the training regressor datapoints from the current operating point,



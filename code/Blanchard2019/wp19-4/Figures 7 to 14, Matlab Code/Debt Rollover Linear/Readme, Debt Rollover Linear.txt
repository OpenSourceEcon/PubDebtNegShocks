To generate the figures for this model:

1. Run the file "RfApproximationLinear" in the folder "Step 1 - Rf Approximation, Linear", which generates the approximating function for Rf. The file will save the coefficients of the approximating polynomial (as well as other model parameters) in a matrix called RfmatrixPolyFitLargeCubedLinear53.

(Notice that you must specify the specific calibration you want in Line 142. Alternatively, you could replace the calibration number with "1:column", which would calculate the polynomials for all calibrations, but that would take many days to complete.

Copy the matrix and paste it into the folder "Step 2 - Debt Rollover Simulation, Linear".

2. (Run the file "SimulationDRLinear" in folder "Step 2 - Debt Rollover Simulation, Linear", which loads the saved matrix and computes simulations for the utility effects of a debt rollover, generating the corresponding graphs. Notice that you must specify the calibration number in Line 21).
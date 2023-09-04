# economist_application

This is the repository for the first coding assignment for The Economist application.

There are two files:

  1) main_script.py: Contains calls for necessary functions from econ_app_utils.py, sets seed and script settings. Everything can be run from just this file, however I have included commenting and reasoning for the choice of model etc. in the econ_app_utils.py file.
  
  2) econ_app_utils.py: Contains the functions that actually do the work.

        a) etl_data(): Used for cleaning data and exporting the polls.csv.
     
        b) fit_splines(): Cubic polynomial spline model used to fit the timeseries -- discussion of this is included in the function.
     
        c) plot_predictions(): Plots results of fit_splines().
     
        d) export_trend(): Exports a CSV with the trends.
     
        e) plot_diagnostics(): A quick trace plot of the MCMC chains as a sanity check.

As mentioned, I went with a cubic polynomial basis spline model for the fit. I discuss it further in the script, but this seemed like a good compromise between the moving average and Dirichlet mixture models mentioned in the problem statement.

Packages:
Python v3.11.4,
pandas,
numpy,
matplotlib,
seaborn,
warnings,
bambi,
arviz,
logging,
datetime

In case it doesn't run, I have included the results from my last run in the test_results folder.

Thanks for your consideration!

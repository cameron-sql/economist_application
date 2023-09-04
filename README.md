# economist_application

This is the repository for the first coding assignment for The Economist application.

There are two files:

  main_script.py: Contains calls for necessary functions from econ_app_utils, sets seed and script settings.
  
  econ_app_utils.py: Contains the functions that actually do the work.
    etl_data(): Used for cleaning data and exporting the polls.csv.
    fit_splines(): Cubic polynomial spline model used to fit the timeseries -- discussion of this is included in the function.
    plot_predictions(): Plots results of fit_splines().
    export_trend(): Exports a CSV with the trends.
    plot_diagnostics(): A quick trace plot of the MCMC chains as a sanity check.

As mentioned, I went with a cubic polynomial basis spline model for the fit. I discuss it further in the script, but this seemed like a good 
compromise between the moving average and Dirichlet mixture models mentioned in the problem statement.

I have included the results from my last run in the test_results folder.

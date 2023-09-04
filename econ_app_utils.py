# Economist helper functions

import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import pandas as pd
import arviz as az
import datetime as dt
import bambi as bmb

np.random.seed(3332)

def etl_data(html_str):
    """
    INPUT: string, html for webpage containing data to scrape
    OUTPUT: df_list, a list of dataframes where each candidate has their own dataframe
        df_resample, a copy of the original dataframe with resampling to fill in missing dates

    This function takes the html string and ETLs the data for use in the fitting process.
    """

    # Read webpage
    df_list = pd.read_html(html_str)
    df = df_list[0]

    # Split up dataframe
    df_ind = df[['Date','Pollster','Sample']]
    df_cand = df.drop(columns=['Date','Pollster','Sample']).copy()

    #df_ind = df.iloc[:,: 3]
    #df_cand = df.iloc[:,-(len(cols) - len(df_ind.columns)):]

    # Remove non-numeric symbols and cast results between [0,1]
    df_cand = df_cand.replace('%','',regex=True).replace('**',np.nan)
    df_cand = df_cand.astype(float)
    df_cand = df_cand / 100

    # Merge back
    df_merge = df_ind.merge(df_cand, right_index=True, left_index=True)
    df_merge.Date = pd.to_datetime(df_merge.Date)

    # Resample with pandas to get the missing days, assign a dummy index for each day
    df_resample = df_merge.set_index('Date').resample('D').mean('Date').reset_index()
    df_resample = df_resample.reset_index().rename(columns={'index':'count_idx'})

    # Merge back once more to get the sample size and export the result
    df_polls = df_merge[['Date','Sample','Pollster']].merge(df_resample, on='Date', how='left').drop(columns='count_idx')
    df_polls.to_csv('polls.csv',index=False)

    # Melt the candidate columns for use in graphing / fitting
    df_melt_rs = pd.melt(df_resample, id_vars=['Date','count_idx'], value_vars=df_cand.columns, var_name='Candidate', value_name='Proportion')

    # Create color palette and labels for graph
    sunlight_cat = ["#156B90", "#9a3e25", "#708259","#bd8f22" ,"#842854","#ba5f06","#0f8c79","#bd2d28","#A0B700","#f2da57","#8e6c8a","#7abfcc", "#f3a126"]
    levels_rs, categories_rs = pd.factorize(df_melt_rs['Candidate'])
    colors_rs = [sunlight_cat[i] for i in levels_rs]
    handles_rs = [matplotlib.patches.Patch(color=sunlight_cat[i], label=c) for i, c in enumerate(categories_rs)]

    # Create a quick graph of the data
    df_melt_rs.plot.scatter(x='Date',y='Proportion',c=colors_rs)
    plt.legend(handles=handles_rs,loc='upper right')
    plt.title('All Candidates')
    plt.savefig('all_candidates.pdf',bbox_inches="tight")

    # Initialize list and append each dataframe to it, grouped by candidate
    df_list = []

    for cand in df_cand.columns:
        df_list.append(df_melt_rs.groupby('Candidate').get_group(cand).dropna())

    return df_list, df_resample



def fit_splines(data, num_knots=11):
    """
    INPUT: data, a list of dataframes (1 for each candidate) - output from etl_data()
        num_knots, the number of knots for the basis spline fit 
    OUTPUT: knot_list, a list of numpy arrays of the knot values (1 for each candidate)
        model_list, a list of bambi model objects (1 for each candidate)
        fit_list, a list of xarray inference data objects (1 for each candidate)

    This function takes the dataframe list and creates / fits a cubic polynomial basis spline model
    using BaMbi (backended by PyMC).
    """

    # Initialize knot lists and create the knots for the inference
    knot_list = []
    iknots_list = []

    for i in range(len(data)):
        knot_list.append(np.quantile(data[i]['count_idx'], np.linspace(0, 1, num_knots)))

    for i in range(len(knot_list)):
        iknots_list.append(knot_list[i][1:-1])

    for i in range(len(knot_list)):
        exec(f'iknots_{i} = knot_list[i][1:-1]')

    """
    MODEL: CUBIC BASIS SPLINES

    I went with cubic basis splines for the trend for a couple of reasons:
        - In the problem statement, it mentions that the type of model is not important as long as it follows the trend.
        Splines seemed like a good fit for this because they show the trend without touching on mechanisms, cause, etc.,
        but they still manage to capture uncertainty information.
        - I was recently reading over Richard McElreath's Statistical Rethinking textbook and enjoyed his example
        using the Cherry Blossoms, so I figured I would give it a go on my own!

    I define a simple model as below, where the proportion on a given day goes as a normal distribution. The mean 
    of the normal distribution is the intercept plus a sum of the weights * basis_functions for all basis functions.

    y ~ N(mu, sigma)
    mu = a + sum(w*B)

    y: day-to-day proportion
    a: intercept
    w: common (weights)
    B: basis functions

    For priors, I am going with the standard uninformative choices -- this should be fine
    considering that the fit looks easy and we are simply looking to fit the data and not 
    necessarily say anything mechanistic / causal about it. 
    """

    priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
        "common": bmb.Prior("Normal", mu=0, sigma=1), 
        "sigma": bmb.Prior("Exponential", lam=1)
    }

    model_list = []

    # Define the model for each candidate with their own knots
    # Include an intercept term and set the degree of the basis polynomials to 3
    for i in range(len(data)):
        model_list.append(bmb.Model(f'Proportion ~ bs(count_idx, knots=iknots_{i}, intercept=True, degree=3)', data=data[i], priors=priors, dropna=True))

    fit_list = []

    # Fit the model -- my laptop isn't great, so I am going with a low amount of draws / chains
    for i in range(len(model_list)):
        fit_list.append(model_list[i].fit(inference_method='mcmc',draws=1000,tune=1000,chains=2,cores=1,idata_kwargs={"log_likelihood": True}, random_seed=3332))

    return knot_list, model_list, fit_list

def plot_scatter(data, figsize=(6, 4)):
    """
    INPUT: data, a list of dataframes (1 for each candidate) - output from etl_data()
    OUTPUT: axes object for plotting

    This is a helper function for plot_predictions.
    """
    _, ax = plt.subplots(figsize=figsize)
    ax.scatter(data["count_idx"], data["Proportion"], alpha=0.5, s=30)
    ax.set_title("Proportion of Vote: " + data.Candidate.unique().item())
    ax.set_xlabel("count_idx")
    ax.set_ylabel("Proportion")
    return ax

def plot_knots(knots, ax):
    """
    INPUT: knots, numpy array of knot values
    ax, matplotlib axes objext
    OUTPUT: axes object for plotting

    This is a helper function for plot_predictions.
    """
    for knot in knots:
        ax.axvline(knot, color="0.1", alpha=0.4)
    return ax

def plot_predictions(data, idata, model, knots, dates):
    """
    INPUT: data, a list of dataframes (1 for each candidate) - output from etl_data()
        idata, inference data object list (1 for each candidate) - output from fit_splines()
        model, bambi model object list (1 for each candidate) - output from fit_splines()
        knots, list of numpy arrays of knots (1 for each candidate) - output from fit_splines()
        dates, set of dates to use for x-axis
    OUTPUT:
        Graphs for each candidate

    This function takes the output from fit_splines and plots the results.
    """

    # Create a test dataset with observations spanning the whole range
    new_data = pd.DataFrame({"count_idx": np.linspace(data.count_idx.min(), data.count_idx.max(), num=len(dates))})

    # Make predictions
    model.predict(idata, data=new_data)

    # Extract these predictions
    posterior_stacked =  az.extract_dataset(idata)
    y_hat = posterior_stacked["Proportion_mean"]

    # Calculate mean
    y_hat_mean = y_hat.mean("sample")

    # Compute 94% credible intervals
    hdi_data = np.quantile(y_hat, [0.03, 0.97], axis=1)

    # Plot obserevd data
    ax = plot_scatter(data)
    
    # Plot predicted line
    ax.plot(new_data["count_idx"], y_hat_mean, color="#9a3e25")
    
    # Plot CI
    ax.fill_between(new_data["count_idx"], hdi_data[0], hdi_data[1], alpha=0.3, color="#9a3e25")
    
    # Add knots
    plot_knots(knots, ax)
    ax.set_xticklabels(dates.Date.dt.date, rotation = 45)
    ax.set_xlabel('Date')
 
    return ax

def export_trend(data, idata, model, dates):
    """
    INPUT: data, a list of dataframes (1 for each candidate) - output from etl_data()
        idata, inference data object list (1 for each candidate) - output from fit_splines()
        model, bambi model object list (1 for each candidate) - output from fit_splines()
        dates, set of dates to use for the CSV export
    OUTPUT: Exported CSV file with trend for each candidate

    This function takes the output from fit_splines and organizes the data as requested from the
    problem statement.
    """
    
    # Create a test dataset with observations spanning the whole range
    new_data = pd.DataFrame({"count_idx": np.linspace(dates.count_idx.min(), dates.count_idx.max(), num=len(dates))})

    # Run prediction with 94% CI
    posterior_stacked = []
    y_hat = []
    y_hat_mean = []
    hdi_data = []
    for i in range(len(idata)):
        model[i].predict(idata[i], data=new_data)
        posterior_stacked.append(az.extract_dataset(idata[i]))
        y_hat.append(posterior_stacked[i]["Proportion_mean"])
        y_hat_mean.append(y_hat[i].mean('sample'))
        hdi_data.append(np.quantile(y_hat[i], [0.03, 0.97], axis=1))

    df_trend_list = []

    # Organize the data into a dataframe for each candidate and store in a list
    for i in range(len(idata)):
        df_trend_list.append(pd.DataFrame({'Date':dates['Date'],
                        f'{data[i].Candidate.unique()[0]}_y_hat_mean':y_hat_mean[i],
                        f'{data[i].Candidate.unique()[0]}_lower_bound':hdi_data[i][0],
                        f'{data[i].Candidate.unique()[0]}_upper_bound':hdi_data[i][1]}))

    # Concat the dataframe list and drop the repeated columns
    df_concat = pd.concat(df_trend_list,axis=1)
    df_concat = df_concat.loc[:,~df_concat.columns.duplicated()].copy()

    # Export
    df_concat.to_csv('trends.csv',index=False)

    return

def plot_diagnostics(idata):
    """
    INPUT: idata, inference data object list (1 for each candidate) - output from fit_splines()
    OUTPUT: Graphs with param distributions / traces

    This function calls ArVizs trace plot to do a quick sanity check on the MCMC process.
    """
    az.plot_trace(idata)
    plt.tight_layout();
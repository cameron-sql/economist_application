# Economist helper functions

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az
import datetime as dt

# Economist helper function

def plot_scatter(data, figsize=(6, 4)):
    _, ax = plt.subplots(figsize=figsize)
    ax.scatter(data["count_idx"], data["Proportion"], alpha=0.5, s=30)
    ax.set_title("Proportion of Vote: " + data.Candidate.unique().item())
    ax.set_xlabel("count_idx")
    ax.set_ylabel("Proportion")
    return ax

def plot_knots(knots, ax):
    for knot in knots:
        ax.axvline(knot, color="0.1", alpha=0.4)
    return ax

def plot_predictions(data, idata, model, knots, dates):
    # Create a test dataset with observations spanning the whole range of year
    new_data = pd.DataFrame({"count_idx": np.linspace(data.count_idx.min(), data.count_idx.max(), num=len(dates))})

    # Predict the day of first blossom
    model.predict(idata, data=new_data)

    posterior_stacked =  az.extract_dataset(idata)
    # Extract these predictions
    y_hat = posterior_stacked["Proportion_mean"]

    # Compute the mean of the predictions, plotted as a single line.
    y_hat_mean = y_hat.mean("sample")

    # Compute 94% credible intervals for the predictions, plotted as bands
    hdi_data = np.quantile(y_hat, [0.03, 0.97], axis=1)

    # Plot obserevd data
    ax = plot_scatter(data)
    
    # Plot predicted line
    ax.plot(new_data["count_idx"], y_hat_mean, color="#9a3e25")
    
    # Plot credibility bands
    ax.fill_between(new_data["count_idx"], hdi_data[0], hdi_data[1], alpha=0.3, color="#9a3e25")
    
    # Add knots
    plot_knots(knots, ax)
    ax.set_xticklabels(dates.Date.dt.date, rotation = 45)
 
    return ax

def export_trend(data, idata, model, dates):
    
    # Create a test dataset with observations spanning the whole range of year
    new_data = pd.DataFrame({"count_idx": np.linspace(dates.count_idx.min(), dates.count_idx.max(), num=len(dates))})

    # Predict
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

    for i in range(len(idata)):
        df_trend_list.append(pd.DataFrame({'Date':dates['Date'],
                        f'{data[i].Candidate.unique()[0]}_y_hat_mean':y_hat_mean[i],
                        f'{data[i].Candidate.unique()[0]}_lower_bound':hdi_data[i][0],
                        f'{data[i].Candidate.unique()[0]}_upper_bound':hdi_data[i][1]}))

    df_concat = pd.concat(df_trend_list,axis=1)

    df_concat = df_concat.loc[:,~df_concat.columns.duplicated()].copy()

    df_concat.to_csv('trends.csv',index=False)

    return
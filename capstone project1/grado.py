# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 20:00:51 2024

@author: win10
"""
import gradio as gr
import itertools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

def generate_custom_diagnostics_plots(residuals):
    # Standardized Residuals
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))
    
    # Plot 1: Standardized Residuals
    axs[0, 0].plot(standardized_residuals, label='Standardized Residuals')
    axs[0, 0].set_title('Standardized Residuals')
    axs[0, 0].legend(loc='best')
    
   # Plot 2: Histogram with Estimated Density and Standard Normal Distribution
    sns.histplot(standardized_residuals, kde=True, ax=axs[0, 1], stat='density', color='blue',binwidth=0.9, label='hist')
    
    # Plot KDE
    sns.kdeplot(standardized_residuals, ax=axs[0, 1], color='green', label='KDE')
    
    
    # Overlay standard normal distribution
    xmin, xmax = axs[0, 1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, 0, 1)  # Standard normal distribution (mean=0, std=1)
    axs[0, 1].plot(x, p, 'k--', label='N(0,1)', linewidth=2)
    
    axs[0, 1].set_xlim(-3, 3)
   
    axs[0, 1].set_title('Histogram plus Estimated Density and N(0,1)')
    axs[0, 1].legend(loc='best')
    
    # Plot 3: Normal Q-Q Plot
    stats.probplot(standardized_residuals, dist="norm", plot=axs[1, 0])
    axs[1, 0].set_title('Normal Q-Q Plot')
    
    # Plot 4: ACF and PACF
    lags = min(15, len(residuals) - 1)
    acf_values = acf(residuals, nlags=lags)
    pacf_values = pacf(residuals, nlags=lags)
    
    axs[1, 1].stem(range(len(acf_values)), acf_values, linefmt='b-', markerfmt='bo', basefmt='b-', label='ACF')
    axs[1, 1].stem(range(len(pacf_values)), pacf_values, linefmt='r-', markerfmt='ro', basefmt='r-', label='PACF')
    axs[1, 1].set_title('ACF and PACF of Residuals')
    axs[1, 1].legend(loc='best')
    
    plt.tight_layout()
    fig.savefig('plots/custom_diagnostics_plots.png')
    plt.close(fig)

def forecast(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    y = data['gdp yoy'].resample('MS').mean()
    
    # Create a directory to save plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    if len(y) < 36:  # Changed from 24 to 36 as per your requirement
        fig, ax = plt.subplots()
        y.plot(ax=ax, label='Original')
        ax.set_title('Not enough data points for seasonal decomposition and SARIMA modeling')
        ax.legend(loc='best')
        fig.savefig('plots/original_plot.png')
        return {
            "ADF Test": None,
            "Best ARIMA Model": None,
            "Mean Squared Error": None,
            "Root Mean Squared Error": None,
            "Forecast": None,
            "Confidence Interval": None,
            "Results Summary": None
        }, 'plots/original_plot.png', None, None, None
    
    # Decomposition
    decomposition = seasonal_decompose(y, model='additive')
    fig, ax = plt.subplots(4, 1, figsize=(12, 8))
    
    ax[0].plot(y, label='Original')
    ax[0].legend(loc='best')
    
    ax[1].plot(decomposition.trend, label='Trend')
    ax[1].legend(loc='best')
    
    ax[2].plot(decomposition.seasonal, label='Seasonal')
    ax[2].legend(loc='best')
    
    ax[3].plot(decomposition.resid, label='Residual')
    ax[3].legend(loc='best')
    
    plt.tight_layout()
    fig.savefig('plots/decomposition_plot.png')
    
    # ADF test
    result = adfuller(y.dropna())
    adf_test_result = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': {key: value for key, value in result[4].items()}
    }
    
    # Grid search for best ARIMA model
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    best_aic = float('inf')
    best_param = None
    best_seasonal_param = None
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y, order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False, 
                                                enforce_invertibility=False)
                results = mod.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_param = param
                    best_seasonal_param = param_seasonal
            except:
                continue
    
    if best_param is None or best_seasonal_param is None:
        mod = sm.tsa.ARIMA(y, order=(1, 1, 1))
        results = mod.fit()
    else:
        mod = sm.tsa.statespace.SARIMAX(y,
                                        order=best_param,
                                        seasonal_order=best_seasonal_param,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        results = mod.fit(disp=False)
    
    results_summary = results.summary().tables[1].as_html()
    
    # Diagnostics Plot Generation
    diagnostics_fig_path = 'plots/diagnostics_plot.png'
    try:
        num_lags = min(10, len(results.resid) // 2)
        if len(results.resid) > num_lags:
            generate_custom_diagnostics_plots(results.resid)
            diagnostics_fig_path = 'plots/custom_diagnostics_plots.png'
        else:
            diagnostics_fig_path = None
            print("Not enough residuals to generate diagnostics plot.")
            # Generate alternative diagnostics plots
            generate_custom_diagnostics_plots(results.resid)
    except Exception as e:
        print(f"Error generating diagnostics plot with results.plot_diagnostics(): {e}")
        diagnostics_fig_path = None
        # Generate alternative diagnostics plots
        generate_custom_diagnostics_plots(results.resid)
    
    # Forecast
    start_date = '2020-01-01'
    pred = results.get_prediction(start=pd.to_datetime(start_date), dynamic=False)
    pred_ci = pred.conf_int()
    
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    y[start_date:].plot(ax=ax2, label='Observed')
    pred.predicted_mean.plot(ax=ax2, label='One-step ahead Forecast', alpha=.7)
    ax2.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('gdp yoy')
    ax2.legend()
    
    fig2.savefig('plots/one_step_ahead_forecast_plot.png')
    
    y_forecasted = pred.predicted_mean
    y_truth = y[start_date:]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    rmse = np.sqrt(mse)
    
    forecast_steps = 13
    pred_uc = results.get_forecast(steps=forecast_steps)
    pred_ci = pred_uc.conf_int()
    
    fig3, ax3 = plt.subplots(figsize=(14, 7))
    y.plot(ax=ax3, label='Observed')
    pred_uc.predicted_mean.plot(ax=ax3, label='Forecast')
    ax3.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('gdp yoy')
    ax3.legend()
    
    fig3.savefig('plots/future_forecast_plot.png')
    
    forecast_values = pred_uc.predicted_mean
    forecast_conf_int = pred_ci
    
    return {
        "ADF Test": adf_test_result,
        "Best ARIMA Model": f"SARIMAX: {best_param} x {best_seasonal_param}12 - AIC:{best_aic}",
        "Mean Squared Error": mse,
        "Root Mean Squared Error": rmse,
        "Forecast": forecast_values.to_json(),
        "Confidence Interval": forecast_conf_int.to_json(),
        "Results Summary": results_summary
    }, 'plots/decomposition_plot.png', diagnostics_fig_path, 'plots/one_step_ahead_forecast_plot.png', 'plots/future_forecast_plot.png'

# Define Gradio interface
inputs = gr.Dataframe(
    type="pandas",
    value=[["2020-01-01",-1.1],["2020-02-01",0.1],["2020-03-01",0.1],["2020-04-01",-83.4],
           ["2020-05-01",-83.4],["2020-06-01",-83.4],["2020-07-01",-20.1],["2020-08-01",-16.9],["2020-09-01",-20.1],
           ["2020-10-01",1.6],["2020-11-01",1.6],["2020-12-01",1.6],["2021-01-01",9.4],["2021-02-01",9.4],
           ["2021-03-01",9.4],["2021-04-01",116.8],["2021-05-01",116.8],["2021-06-01",116.8],["2021-07-01",39.0],
           ["2021-08-01",39.0],["2021-09-01",39.0],["2021-10-01",43.9],["2021-11-01",43.9],["2021-12-01",43.9],
           ["2022-01-01",34.3],["2022-02-01",34.3],["2022-03-01",34.3],["2022-04-01",27.6],["2022-05-01",27.6],
           ["2022-06-01",27.6],["2022-07-01",28.0],["2022-08-01",28.0],["2022-09-01",28.0],["2022-10-01",16.1],
           ["2022-11-01",16.1],["2022-12-01",16.1],["2023-01-01",20.1],["2023-02-01",20.1],["2023-03-01",20.1],
           ["2023-04-01",8.2],["2023-05-01",8.2],["2023-06-01",8.2],["2023-07-01",10.3]],
    headers=["Date", "gdp yoy"]
)

outputs = [
    gr.Json(label="Results"),
    gr.Image(label="Decomposition Plot"),
    gr.Image(label="Custom Diagnostics Plot"),
    gr.Image(label="One-step Ahead Forecast Plot"),
    gr.Image(label="Future Forecast Plot")
]

# Launch Gradio Interface
gr.Interface(forecast, inputs, outputs).launch()























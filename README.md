# Stock Price Predictor

## Overview

The **Stock Price Predictor** is a web application built with **Streamlit** that allows users to forecast stock prices using the **Facebook Prophet** model and predict stock trends with a **Random Forest Classifier**. It uses real-time stock data from Yahoo Finance and offers customizable model parameters to make forecasts and predictions.

The application walks users through a step-by-step process, including selecting a stock, setting a date range, and customizing prediction parameters.

### Key Features:
- **Stock Price Forecasting**: Predict future stock prices using the Prophet time-series model.
- **Trend Prediction**: Classify future stock price trends (up or down) using a Random Forest Classifier.
- **Performance Metrics**: Evaluate Prophet's cross-validation metrics, including RMSE, and assess the accuracy of the Random Forest Classifier with a classification report.
- **Data Visualization**: Interactive visualizations of actual vs. predicted stock prices, profit predictions, and trend forecasts using Plotly.

---

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [App Workflow](#app-workflow)
4. [Features](#features)
5. [Technologies Used](#technologies-used)
6. [Acknowledgments](#acknowledgments)

---

## Installation

To run this application locally, follow the steps below.

### Prerequisites
- Python 3.7 or higher
- Recommended to create a virtual environment before installing the dependencies.

### Setup

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
   pip install -r requirements.txt
   streamlit run app.py

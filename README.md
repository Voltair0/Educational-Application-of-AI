# Didactic Application of AI: Wine Quality Prediction

## Overview
This repository contains a school project focused on the practical application of Artificial Intelligence and Machine Learning. The primary goal of this project is to build and evaluate predictive models to determine wine quality based on chemical properties, serving as a "didactic" (educational) exploration of the standard machine learning pipeline.

## Project Structure

The project consists of the following key Python scripts:

*   **`preprocess.py`**: Handles the data loading, cleaning, and preprocessing steps. This prepares the raw wine quality dataset for model training (e.g., handling missing values, scaling features, train/test splitting).
*   **`count.py`**: A utility script likely used for Exploratory Data Analysis (EDA), counting target class distributions, or data summarization.
*   **`wine_quality.py`**: The main or baseline machine learning script used to train and evaluate initial models on the wine dataset.
*   **`XGBoost_wine_quality.py`**: A specialized script that implements the **XGBoost** (eXtreme Gradient Boosting) algorithm to achieve higher predictive accuracy on the wine quality dataset.

*(Note: `__pycache__` directories contain compiled Python bytecode files like `preprocess.cpython-313.pyc` and are generated automatically when the scripts are run.)*

## Prerequisites

To run these scripts, you will need Python 3 installed (the project seems to run on Python 3.13 based on the cache files), along with several data science libraries. 

You can typically install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn xgboost
```

## How to Run

1. **Preprocess the Data**: Run the preprocessing script to prepare your dataset (if it runs as a standalone script).
   ```bash
   python preprocess.py
   ```
2. **Explore the Data**: 
   ```bash
   python count.py
   ```
3. **Train the Models**: 
   Run the baseline model:
   ```bash
   python wine_quality.py
   ```
   Run the advanced XGBoost model:
   ```bash
   python XGBoost_wine_quality.py
   ```

## Dataset
*Note: Make sure to place your `winequality.csv` (or similarly named dataset) in the root directory before running the scripts if it is not automatically downloaded by the code.*

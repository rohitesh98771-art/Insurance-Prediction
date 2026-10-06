# Insurance Prediction

A command-line tool that learns from an insurance CSV file and predicts the yearly insurance charges for a new person.

## What it does

- Asks for the path of your CSV file (or opens a file picker if you press Enter)
- Cleans the data: tidies text values, reports missing values, removes duplicate rows
- Trains three models (Linear Regression, Random Forest, Gradient Boosting) and keeps the most accurate one
- Shows charts: model accuracy, predicted vs actual, most important inputs, and average charges by category
- Lets you enter new details (age, sex, bmi, children, smoker, region) and returns the predicted charges, a likely range, and the monthly cost
- Saves the trained model as `insurance_model.joblib` so it can be reused next time

## Requirements

- Python 3.8 or newer
- pandas
- scikit-learn
- joblib
- matplotlib

Install them with:

```
pip install pandas scikit-learn joblib matplotlib
```

## How to run

```
python Insurance_Prediction.py
```

Then follow the prompts:

1. Enter the path to your CSV file
2. Wait for the model to train and view the charts
3. Choose **1) New prediction** and enter the new person's details

## Expected CSV format

The CSV needs a numeric column with the charges (named `charges`, `premium`, `cost` or similar) plus input columns. Example:

```
age,sex,bmi,children,smoker,region,charges
19,female,27.9,0,yes,southwest,16884.924
18,male,33.77,1,no,southeast,1725.5523
```

## Note

Predictions are estimates based on the data you provide. They are not official insurance quotes.

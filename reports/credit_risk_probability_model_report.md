# Credit Risk Probability Model for Alternative Data

## Executive Summary

I built a customer-level credit-risk probability model using Xente transaction data. Since the dataset does not include true loan repayment or default outcomes, the model predicts a proxy Probability of Default (PD) label derived from customer RFMS behavior and observed fraud history.

The trained Random Forest model achieved ROC-AUC 0.9985 and F1 0.9518 on the held-out test set. This means the model separates the engineered proxy risk classes very well. It does not prove real default-risk performance until the proxy is validated against actual repayment outcomes.

## Data

- Raw rows: 95,662 transactions
- Customers: 3,742
- Source fields: transaction amount/value, product, provider, channel, timestamp, pricing strategy, and fraud flag
- Modeling grain: one row per customer

## Target Engineering

The dataset has no direct default label, so I created `is_high_risk`:

- `1` if a customer belongs to the weakest RFMS cluster;
- `1` if the customer has at least one observed fraud transaction;
- `0` otherwise.

RFMS uses recency, frequency, and monetary behavior to identify weaker customer segments. `FraudResult` is not used as a model feature, only as a target-construction signal.

## Feature Engineering

Features include:

- activity: transaction count, active days, transactions per active day;
- monetary behavior: total value, average value, median amount, maximum value, standard deviation;
- account-flow behavior: credit/debit transaction ratios and net amount per transaction;
- channel/product breadth: unique products, categories, providers, and channels;
- temporal behavior: recency, customer age, average hour, weekend ratio;
- dominant categorical behavior: most common product category, channel, provider, and pricing strategy.

## Modeling

I compared Logistic Regression and Random Forest using stratified cross-validation and ROC-AUC selection. Numeric features are median-imputed and standardized. Categorical features are mode-imputed and one-hot encoded.

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8678 | 0.6633 | 0.6065 | 0.7318 |
| Random Forest | 0.9985 | 0.9518 | 0.9655 | 0.9385 |

Best model: Random Forest.

## Deployment

The project includes a FastAPI app with:

- `GET /health`
- `GET /model/info`
- `POST /predict`
- `POST /predict/batch`

The model artifact is saved at `models/credit_risk_pd_model.joblib`.

## Caveats

This is a strong coursework-ready implementation, but the proxy target is not a substitute for real default data. Before production use, validate the score against repayment, delinquency, charge-off, or collections outcomes; calibrate PD probabilities; add fairness testing; and monitor drift across channels, products, and customer segments.

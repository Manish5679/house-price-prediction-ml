# House Price Prediction Using Machine Learning

## Business Problem
A real-estate business needs a data-driven way to estimate house selling prices using property features.

## Objective
Build and deploy a regression model that predicts house price from property details.

## Dataset
- Source: Kaggle Housing Prices Dataset
- File: `Housing.csv`
- Rows: 545
- Columns: 13
- Target Variable: `price`

## Methodology
Data loading → Data cleaning → EDA → Feature preparation → Train/test split → Model training → Model evaluation → Model saving → Streamlit app → Deployment.

## Models Used
- Linear Regression
- Random Forest Regressor

## Model Performance
| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 970,043 | 1,754,318,687,331 | 1,324,507 | 0.653 |
| Random Forest Regressor | 1,012,520 | 1,932,088,148,256 | 1,389,996 | 0.618 |

## Final Model
Linear Regression was selected because it produced the better R² score and lower error values on the test set.

## Business Insights
Area, bathrooms, stories, parking and bedrooms show positive relationships with house price. Furnished houses and houses with air conditioning generally have higher average prices.

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository Structure
```text
house_price_project/
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
├── data/
│   └── Housing.csv
├── notebooks/
│   └── model_development.ipynb
└── images/
    └── dashboard.png
```

## Deployment
The application is deployed using Streamlit Community Cloud.

## Links
- GitHub Repository: https://github.com/Manish5679/house-price-prediction-ml
- Live Streamlit App: https://house-price-prediction-ml-uhfetoflkcghelnkgaphhs.streamlit.app/

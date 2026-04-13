# Task 1 — Multiple Linear Regression Model

# I created a 50 record housing dataset with area, bedrooms and age as features.
# Each feature got its own coefficient — area adds to price, bedrooms push it -
# higher, and older houses bring it slightly down. That matches real life.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

np.random.seed(42)
n = 50

area_sqft    = np.random.randint(500, 3000, n)
num_bedrooms = np.random.randint(1, 6, n)
age_years    = np.random.randint(1, 30, n)

price_lakhs = (
    area_sqft * 0.05
    + num_bedrooms * 8
    - age_years * 0.3
    + np.random.normal(0, 5, n)
)

df = pd.DataFrame({
    'area_sqft':    area_sqft,
    'num_bedrooms': num_bedrooms,
    'age_years':    age_years,
    'price_lakhs':  price_lakhs
})

X = df[['area_sqft', 'num_bedrooms', 'age_years']]
y = df['price_lakhs']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", round(model.intercept_, 2))
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {round(coef, 4)}")

y_pred = model.predict(X_test)
print("\nActual vs Predicted (first 5):")
print(f"{'Actual':<15} {'Predicted'}")
print("-" * 30)
for actual, predicted in zip(y_test[:5], y_pred[:5]):
    print(f"{round(actual, 2):<15} {round(predicted, 2)}")


# Task 2 — MAE, RMSE and R²

# MAE of 3.32 means my predictions are off by around 3.32 lakhs on average.
# RMSE of 4.21 is a bit higher which means a few predictions had larger errors.
# R² of 0.9865 tells me the model explains 98.65% of the price variation -
# that's a strong result for a simple linear model.

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"  MAE  : {round(mae, 2)} lakhs")
print(f"  RMSE : {round(rmse, 2)} lakhs")
print(f"  R²   : {round(r2, 4)}")


# Task 3 — Residuals Histogram

# A residual is the gap between the actual price and what the model predicted.
# My histogram shows most errors sitting between -2 and +4 lakhs near zero.
# There are a couple of outliers around -8 and +6 but they are isolated.
# Overall the spread is fairly balanced around zero - the model isn't
# consistently overshooting or undershooting in one direction.

residuals = y_test.values - y_pred

plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=10, color='steelblue', edgecolor='black')
plt.title('Histogram of Residuals')
plt.xlabel('Residual (Actual - Predicted) in Lakhs')
plt.ylabel('Frequency')
plt.axvline(0, color='red', linestyle='--', label='Zero Error Line')
plt.legend()
plt.tight_layout()
plt.savefig('residuals_histogram.png')
plt.show()
print("Histogram saved as residuals_histogram.png")
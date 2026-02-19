import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = {
    'sqft': [1500, 2000, 2500, 1200, 1800, 3000, 2200, 1100, 2700, 1900],
    'bedrooms': [3, 4, 4, 2, 3, 5, 4, 2, 4, 3],
    'age': [10, 5, 1, 20, 15, 2, 8, 25, 3, 12],
    'price': [300000, 450000, 550000, 250000, 350000, 650000, 480000, 200000, 590000, 380000]
}
df = pd.DataFrame(data)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "KNN (Neighbors)": KNeighborsRegressor(n_neighbors=3),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel='linear')
}

comparison_list = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    comparison_list.append({
        "Model_Name": name,
        "MAE": mae,
        "RMSE": rmse
    })

results_df = pd.DataFrame(comparison_list)

results_df.to_csv('model_results.csv', index=False)

print("-" * 30)
print("CSV EXPORTED SUCCESSFULLY: model_results.csv")
print("-" * 30)
print(results_df)

plt.bar(results_df['Model_Name'], results_df['RMSE'])
plt.title('RMSE Comparison')
plt.show()
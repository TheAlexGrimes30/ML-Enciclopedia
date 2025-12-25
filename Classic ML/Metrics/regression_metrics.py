from typing import Dict

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

X, y = make_regression(
    n_samples=500,
    n_features=5,
    n_informative=3,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Функция для вычисления кастомных метрик для регрессии
    :param y_true: np.ndarray - действительные метки
    :param y_pred: np.ndarray - предсказанные метки
    :return:
    """

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_custom = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    mse_custom = np.mean((y_true - y_pred) ** 2)
    rmse_custom = np.sqrt(mse_custom)
    mae_custom = np.mean(np.abs((y_true - y_pred)))
    mape_custom = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)))
    smape_custom = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))

    return {
        'R²': r2_custom,
        'MSE': mse_custom,
        'RMSE': rmse_custom,
        'MAE': mae_custom,
        'MAPE': mape_custom,
        'SMAPE': smape_custom
    }

sklearn_metrics = {
    'R²': r2_score(y_test, y_pred),
    'MSE': mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
    'MAE': mean_absolute_error(y_test, y_pred),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred)
}

print("\n" + "=" * 60)
print("МЕТРИКИ (КАСТОМНЫЕ РАСЧЕТЫ):")
print("=" * 60)

custom_metrics = calculate_regression_metrics(y_test, y_pred)
for metric, value in custom_metrics.items():
    print(f"{metric:<2}: {value:.4f}")

print("\n" + "=" * 60)
print("МЕТРИКИ (Sklearn РАСЧЕТЫ):")
print("=" * 60)
for metric, value in sklearn_metrics.items():
    print(f"{metric:<2}: {value:.4f}")




from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

X, y = make_regression(
    n_samples=5000,
    n_features=20,
    n_informative=15,
    noise=10.0,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

cat_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function="RMSE",
    verbose=False,
    random_state=42
)

cat_model.fit(X_train, y_train)
y_pred_cat = cat_model.predict(X_test)

print("CatBoost Regressor")
print("MSE:", mean_squared_error(y_test, y_pred_cat))
print("R2:", r2_score(y_test, y_pred_cat))
print()

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost Regressor")
print("MSE:", mean_squared_error(y_test, y_pred_xgb))
print("R2:", r2_score(y_test, y_pred_xgb))
print()

lgbm_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="regression",
    random_state=42
)

lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_test)

print("LightGBM Regressor")
print("MSE:", mean_squared_error(y_test, y_pred_lgbm))
print("R2:", r2_score(y_test, y_pred_lgbm))

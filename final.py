import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

def load_and_prepare_data(ages_path, data_path):
    
    ages = pd.read_csv(ages_path)
    data = pd.read_csv(data_path)

    
    data.rename(columns={'Unnamed: 0': 'Sample Accession'}, inplace=True)

    
    data_cleaned = data.drop(data.columns[1], axis=1)

    
    merged_data = pd.merge(ages, data_cleaned, on='Sample Accession')

    
    X = merged_data.drop(columns=['Sample Accession', 'Age'])
    y = merged_data['Age']

    
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)  

    return X, y

def train_models(X_train, y_train):
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)

    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)

    return {'Random Forest': rf_model, 'Gradient Boosting': gb_model, 'XGBoost': xgb_model}

def evaluate_models(models, X_test, y_test):
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MAE': mae, 'R2': r2}
    return results

def main():
    
    X, y = load_and_prepare_data('Ages.csv', 'data.csv')

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    models = train_models(X_train, y_train)

    
    results = evaluate_models(models, X_test, y_test)


    for name, metrics in results.items():
        print(f"Model: {name}")
        print(f"Mean Absolute Error: {metrics['MAE']}")
        print(f"R2 Score: {metrics['R2']}\n")

if _name_ == "_main_":
    main()
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product

def train_xgb(df, feature_columns):
    """Trains an XGBoost classifier on the given feature set."""
    X = df[feature_columns]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return acc

def feature_selection_experiment(df):
    """Tests all feature combinations and logs their accuracy."""
    feature_names = [col for col in df.columns if col != 'target']
    combinations = list(product([0, 1], repeat=len(feature_names)))
    
    results = []
    for combo in combinations:
        selected_features = [feature_names[i] for i in range(len(combo)) if combo[i] == 1]
        if not selected_features:
            continue  # Skip empty feature selection
        
        acc = train_xgb(df, selected_features)
        results.append(tuple(combo) + (acc,))
    
    columns = feature_names + ['Fit']
    results_df = pd.DataFrame(results, columns=columns)
    print("\nFeature Selection Results:")
    print(results_df)
    return results_df

if __name__ == "__main__":
    file_path = "venv/data/banana.tsv"
    df = pd.read_csv(file_path, sep='\t')
    feature_selection_experiment(df)
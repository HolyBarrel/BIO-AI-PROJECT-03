import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product

def train_xgb(df, feature_columns, use_gpu=False):
    """Trains an XGBoost classifier on the given feature set."""
    X = df[feature_columns]
    y = df['target']

    # Ensure class labels start from 0
    y = y - y.min()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert data to DMatrix (recommended by XGBoost)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # XGBoost model configuration
    params = {
        'eval_metric': 'logloss', 
        'objective': 'multi:softmax', 
        'num_class': len(y.unique())
    }

    # Set the device parameter instead of gpu_id
    if use_gpu:
        params['device'] = 'cuda:0'

    model = xgb.train(params, dtrain)
    
    y_pred = model.predict(dtest)
    acc = accuracy_score(y_test, y_pred)
    
    return acc

def feature_selection_experiment(df, target_column='target', ignored_features=[], print_per=1000, use_gpu=False, show_lookup=True, show_best=True):
    """Tests all feature combinations and logs their accuracy, printing progress at every `print_per` steps."""
    feature_names = [col for col in df.columns if col != target_column and col not in ignored_features]
    combinations = list(product([0, 1], repeat=len(feature_names)))

    total_combinations = len(combinations)
    results = []
    best_combination = None
    best_accuracy = -float('inf')

    for i, combo in enumerate(combinations):
        selected_features = [feature_names[j] for j in range(len(combo)) if combo[j] == 1]
        if not selected_features:
            continue  # Skip empty feature selection
        
        acc = train_xgb(df, selected_features, use_gpu)
        results.append(tuple(combo) + (acc,))

        # Update the best combination if the current accuracy is higher
        if acc > best_accuracy:
            best_accuracy = acc
            best_combination = selected_features

        # Print progress every `print_per` iterations
        if (i + 1) % print_per == 0 or (i + 1) == total_combinations:
            percent_done = (i + 1) / total_combinations * 100
            print(f"Processed {i+1}/{total_combinations} combinations ({percent_done:.2f}% done)")

    columns = feature_names + ['Fit']
    results_df = pd.DataFrame(results, columns=columns)

    # Print the best combination and its fitness score
    if(show_lookup):
        print("\nFeature Selection Results:")
        print(results_df)
    if(show_best):
        print("\nBest Feature Set and Accuracy:")
        print(f"Best Features: {best_combination}")
        print(f"Accuracy: {best_accuracy}")

    return results_df

if __name__ == "__main__":
    # breast_cancer_wisconsin_original
    # wine_quality_combined
    # banana
    file_root = "preprocessing/data/"
    file_name = "wine_quality_combined"
    file_path = file_root + file_name + ".tsv"
    df = pd.read_csv(file_path, sep='\t')
    
    feature_selection_experiment(df, ignored_features=['wine_type'], print_per=100, use_gpu=False)

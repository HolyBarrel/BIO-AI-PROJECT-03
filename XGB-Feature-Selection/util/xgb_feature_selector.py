import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product

class FeatureSelectionXGB:
    def __init__(self, df_file_name, target_column='target', ignored_features=[], test_size=0.2, use_gpu=False, print_per=1000, show_lookup=True, show_best=True, save_to_csv=False, output_location='output'):
        """
        Initializes the class with the dataframe and parameters for feature selection experiment.
        """
        self.df_file_name = df_file_name
        self.target_column = target_column
        self.ignored_features = ignored_features
        self.test_size = test_size
        self.use_gpu = use_gpu
        self.print_per = print_per
        self.show_lookup = show_lookup
        self.show_best = show_best
        self.save_to_csv = save_to_csv
        self.output_location = output_location

        # Load the dataset
        self.df = pd.read_csv(f"XGB-Feature-Selection/data/{self.df_file_name}.tsv", sep='\t')

    def train_xgb(self, feature_columns):
        """Trains an XGBoost classifier on the given feature set."""
        X = self.df[feature_columns]
        y = self.df[self.target_column]

        # Ensure class labels start from 0
        y = y - y.min()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        
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
        if self.use_gpu:
            params['device'] = 'cuda:0'

        model = xgb.train(params, dtrain)
        
        y_pred = model.predict(dtest)
        acc = accuracy_score(y_test, y_pred)
        
        return acc
    
    def save_results_to_csv(self, results_df):
        # Create output folders
        if not os.path.exists(self.output_location):
            os.makedirs(self.output_location)

        # Save file to output folder
        results_df.to_csv(self.output_location + "/" + self.df_file_name, index=False)
        print(f"\nResults saved to {self.output_location}")



    def feature_selection_experiment(self):
        """Tests all feature combinations and logs their accuracy, printing progress at every `print_per` steps."""
        feature_names = [col for col in self.df.columns if col != self.target_column and col not in self.ignored_features]
        combinations = list(product([0, 1], repeat=len(feature_names)))

        total_combinations = len(combinations)
        results = []
        best_combination = None
        best_accuracy = -float('inf')

        for i, combo in enumerate(combinations):
            selected_features = [feature_names[j] for j in range(len(combo)) if combo[j] == 1]
            if not selected_features:
                continue  # Skip empty feature selection
            
            acc = self.train_xgb(selected_features)
            results.append(tuple(combo) + (acc,))

            # Update the best combination if the current accuracy is higher
            if acc > best_accuracy:
                best_accuracy = acc
                best_combination = selected_features

            # Print progress every `print_per` iterations
            if (i + 1) % self.print_per == 0 or (i + 1) == total_combinations:
                percent_done = (i + 1) / total_combinations * 100
                print(f"Processed {i+1}/{total_combinations} combinations ({percent_done:.2f}% done)")

        columns = feature_names + ['Fit']
        results_df = pd.DataFrame(results, columns=columns)

        # Print the best combination and its fitness score
        if self.show_lookup:
            print("\nFeature Selection Results:")
            print(results_df)
        if self.show_best:
            print("\nBest Feature Set and Accuracy:")
            print(f"Best Features: {best_combination}")
            print(f"Accuracy: {best_accuracy}")

        # Save to CSV if requested
        if self.save_to_csv:
            self.save_results_to_csv(results_df)

        return results_df

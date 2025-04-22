import os
import math
import pandas as pd
import itertools
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# how many rows to buffer before appending to CSV
BATCH_SIZE = 100

# Column names
col_cleaveland_heart = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
col_zoo = [
    "animal_name", "hair", "feathers", "eggs", "milk", "airborne",
    "aquatic", "predator", "toothed", "backbone", "breathes",
    "venomous", "fins", "legs", "tail", "domestic", "catsize", "target"
]

# Load data
df_cleaveland_heart = pd.read_csv(
    "test_data/processed.cleveland.data", header=None, names=col_cleaveland_heart
).replace("?", 0)
df_zoo = pd.read_csv("test_data/zoo.data", header=None, names=col_zoo).replace("?", 0)

# Split X/y
X_cleaveland_heart = df_cleaveland_heart.drop(columns=["target"])
y_cleaveland_heart = df_cleaveland_heart["target"]

X_zoo = df_zoo.drop(columns=["animal_name", "target"])
y_zoo = df_zoo["target"]

datasets = {
    "heart_disease": (X_cleaveland_heart, y_cleaveland_heart),
    "zoo":           (X_zoo, y_zoo)
}

def eval_combination(X, y, combination):
    selected = list(combination)
    def one_seed(seed):
        clf = RandomForestClassifier(
            n_estimators=30,
            random_state=seed,
            n_jobs=-1
        )
        clf.fit(X[selected], y)
        return clf.score(X[selected], y)

    accuracies = Parallel(n_jobs=-1)(
        delayed(one_seed)(seed) for seed in range(30)
    )
    avg_acc = sum(accuracies) / len(accuracies)
    row = {feat: (feat in selected) for feat in X.columns}
    row["Loss"] = 1.0 - avg_acc
    return row

if __name__ == "__main__":
    os.makedirs("output_test", exist_ok=True)

    for name, (X, y) in datasets.items():
        feature_names = list(X.columns)
        n_features = len(feature_names)

        # total count of non-empty subsets = 2^n - 1
        total_combinations = sum(
            math.comb(n_features, r) for r in range(1, n_features + 1)
        )
        # generator for all non-empty subsets
        combinations = itertools.chain.from_iterable(
            itertools.combinations(feature_names, r)
            for r in range(1, n_features + 1)
        )

        output_csv = os.path.join("output_test", f"random_forest_{name}.csv")
        # figure out how many rows we've already written
        if os.path.exists(output_csv):
            done_df = pd.read_csv(output_csv)
            start_idx = len(done_df)
        else:
            start_idx = 0

        # progress bar that starts at what we've done
        pbar = tqdm(total=total_combinations,
                    initial=start_idx,
                    desc=f"{name} combinations")

        buffer = []
        for idx, comb in enumerate(combinations):
            pbar.update(1)
            if idx < start_idx:
                continue  # skip already‐done

            buffer.append(eval_combination(X, y, comb))

            if len(buffer) >= BATCH_SIZE:
                # append to CSV
                df_batch = pd.DataFrame(buffer)
                df_batch.to_csv(
                    output_csv,
                    mode="a",
                    header=not os.path.exists(output_csv),
                    index=False
                )
                buffer.clear()

        # flush any leftovers
        if buffer:
            df_batch = pd.DataFrame(buffer)
            df_batch.to_csv(
                output_csv,
                mode="a",
                header=not os.path.exists(output_csv),
                index=False
            )
        pbar.close()
        print(f"Saved {output_csv}")

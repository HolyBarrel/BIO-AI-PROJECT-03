import os
import math
import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from cuml.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score
import cupy as cp
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# reproducibility constants
DATA_SPLIT_SEED = 123
RF_SEED         = 456

# how many rows to buffer before appending to CSV
BATCH_SIZE = 100

# --- 1) load your two pandas datasets as before ---
df_cleaveland_heart = pd.read_csv(
    "XGB-Feature-Selection/test_data/processed.cleveland.data", header=None,
    names=[
        "age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","target"
    ]
).replace("?", 0)
X_heart = df_cleaveland_heart.drop(columns="target").astype(float)
y_heart = df_cleaveland_heart["target"].astype(int).values

df_zoo = pd.read_csv("XGB-Feature-Selection/test_data/zoo.data", header=None,
    names=[
        "animal_name","hair","feathers","eggs","milk","airborne",
        "aquatic","predator","toothed","backbone","breathes",
        "venomous","fins","legs","tail","domestic","catsize","target"
    ]
).replace("?", 0)
X_zoo = df_zoo.drop(columns=["animal_name","target"]).astype(float)
y_zoo = df_zoo["target"].astype(int).values

df_letter_recognition = pd.read_csv(
    "XGB-Feature-Selection/test_data/letter-recognition.data", header=None,
    names=[
        "letter","x-box","y-box","width","high","onpix",
        "x-bar","y-bar","x2bar","y2bar","xybar","x2ybr",
        "xy2br","x-ege","xegvy","y-ege","yegvx"]
).replace("?", 0)

X_letter = df_letter_recognition.drop(columns=["letter"]).astype(float)
y_letter = df_letter_recognition["letter"].values  # Keep as string for classification

datasets = {
    "heart_disease": (X_heart, y_heart),
    "zoo":           (X_zoo,    y_zoo),
    "letter_recognition": (X_letter, y_letter)
}

# --- print any existing best rows (as before) ---
os.makedirs("XGB-Feature-Selection/output_test", exist_ok=True)
for name in datasets:
    output_csv = os.path.join("XGB-Feature-Selection/output_test", f"random_forest_{name}.csv")
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        best_row = df.loc[df["Loss"].idxmin()]
        print(f"Best row for {name}:")
        print(best_row)

# number of worker processes
n_procs = max(1, multiprocessing.cpu_count() - 1)

for name, (X_df, y) in datasets.items():
    X_vals = X_df.values
    feature_names = list(X_df.columns)
    n_features = X_vals.shape[1]

    # --- fixed train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_vals, y, test_size=0.3, random_state=DATA_SPLIT_SEED
    )

    # --- build bit‐mask combinations ---
    masks = np.arange(1, 1 << n_features, dtype=np.uint64)
    n_bytes = (n_features + 7) // 8
    bytes_view = (
        masks
        .astype(f'>u{n_bytes}')
        .view(np.uint8)
        .reshape(-1, n_bytes)
    )
    bits = np.unpackbits(bytes_view, axis=1)[:, -n_features:]
    comb_indices = [list(np.nonzero(row)[0]) for row in bits]
    total = len(comb_indices)

    # --- prepare output CSV & resume info ---
    output_csv = os.path.join("XGB-Feature-Selection/output_test", f"random_forest_{name}.csv")
    if os.path.exists(output_csv):
        done_rows = len(pd.read_csv(output_csv))
        write_header = False
    else:
        done_rows = 0
        write_header = True

    # --- initialize RNG for forest seeds (30 draws) ---
    bitgen = np.random.MT19937(RF_SEED)
    rng = np.random.Generator(bitgen)
    forest_seeds = rng.integers(
        low=0,
        high=np.iinfo(np.int32).max,
        size=30,
        dtype=np.int32
    )

    # --- define GPU-based evaluator ---
    def eval_comb(idx_list):
        scores = []
        X_train_gpu = cp.asarray(X_train[:, idx_list])
        X_test_gpu = cp.asarray(X_test[:, idx_list])
        y_train_gpu = cp.asarray(y_train)
        y_test_gpu = cp.asarray(y_test)

        for seed in forest_seeds:
            clf = RandomForestClassifier(
                n_estimators=30,
                random_state=int(seed),
                n_streams=1
            )
            clf.fit(X_train_gpu, y_train_gpu)
            y_pred = clf.predict(X_test_gpu)
            acc = accuracy_score(y_test_gpu, y_pred)
            scores.append(acc)

        loss = 1.0 - np.mean(scores)
        row = {feat: (i in idx_list) for i, feat in enumerate(feature_names)}
        row["Loss"] = loss
        return row

    # --- batch + parallel loop with progress bar ---
    pbar = tqdm(total=total, desc=f"{name} combos", initial=done_rows)
    for start in range(done_rows, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_idxs = comb_indices[start:end]

        results = Parallel(
            n_jobs=n_procs,
            verbose=0
        )(delayed(eval_comb)(idxs) for idxs in batch_idxs)

        pd.DataFrame(results).to_csv(
            output_csv,
            mode="a",
            header=write_header,
            index=False
        )
        write_header = False

        pbar.update(len(batch_idxs))

    pbar.close()
    print(f"→ Saved {output_csv}")

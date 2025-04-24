import os
import math
import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# reproducibility constants
DATA_SPLIT_SEED = 123
RF_SEED         = 456

# how many rows to buffer before appending to CSV
BATCH_SIZE = 100


# --- 1) load your two pandas datasets as before ---
df_cleaveland_heart = pd.read_csv(
    "test_data/processed.cleveland.data", header=None,
    names=[
        "age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","target"
    ]
).replace("?", 0)
X_heart = df_cleaveland_heart.drop(columns="target")
y_heart = df_cleaveland_heart["target"].values

df_zoo = pd.read_csv("test_data/zoo.data", header=None,
    names=[
        "animal_name","hair","feathers","eggs","milk","airborne",
        "aquatic","predator","toothed","backbone","breathes",
        "venomous","fins","legs","tail","domestic","catsize","target"
    ]
).replace("?", 0)
X_zoo = df_zoo.drop(columns=["animal_name","target"])
y_zoo = df_zoo["target"].values


df_letter_recognition = pd.read_csv(
    "test_data/letter-recognition.data", header=None,
    names=[
        "letter","x-box","y-box","width","high","onpix",
        "x-bar","y-bar","x2bar","y2bar","xybar","x2ybr",
        "xy2br","x-ege","xegvy","y-ege","yegvx"]
).replace("?", 0)

X_letter = df_letter_recognition.drop(columns=["letter"])
y_letter = df_letter_recognition["letter"].values



datasets = {
    "heart_disease": (X_heart, y_heart),
    "zoo":           (X_zoo,    y_zoo),
    "letter_recognition": (X_letter, y_letter)
}


# --- print any existing best rows (as before) ---
os.makedirs("output_test", exist_ok=True)
for name in datasets:
    output_csv = os.path.join("output_test", f"random_forest_{name}.csv")
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        best_row = df.loc[df["Loss"].idxmin()]
        print(f"Best row for {name}:")
        print(best_row)


# number of worker processes
n_procs = max(1, multiprocessing.cpu_count() - 1)


for name, (X_df, y) in datasets.items():
    # convert to numpy once
    X_vals = X_df.values
    feature_names = list(X_df.columns)
    n_features = X_vals.shape[1]

    # --- 1) fixed train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_vals, y, test_size=0.3, random_state=DATA_SPLIT_SEED
    )

    # --- 2) build bit‐mask combinations once ---
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

    # --- 3) prepare output CSV & resume info ---
    output_csv = os.path.join("output_test", f"random_forest_{name}.csv")
    if os.path.exists(output_csv):
        done_rows   = len(pd.read_csv(output_csv))
        write_header = False
    else:
        done_rows   = 0
        write_header = True

    # --- 4) initialize RNG for forest seeds (30 draws) ---
    bitgen    = np.random.MT19937(RF_SEED)        # Mersenne Twister seeded 456
    rng       = np.random.Generator(bitgen)
    forest_seeds = rng.integers(
        low=0,
        high=np.iinfo(np.int32).max,
        size=30,
        dtype=np.int32
    )

    # --- 5) define your evaluator (one mask → one dict row) ---
    def eval_comb(idx_list):
        scores = []
        for seed in forest_seeds:
            clf = RandomForestClassifier(
                n_estimators=30,
                random_state=int(seed),  # drawn reproducibly from RF_SEED
                n_jobs=1
            )
            clf.fit(X_train[:, idx_list], y_train)
            scores.append(clf.score(X_test[:, idx_list], y_test))
        loss = 1.0 - np.mean(scores)
        row = { feat: (i in idx_list)
                for i, feat in enumerate(feature_names) }
        row["Loss"] = loss
        return row

    # --- 6) batch + parallel loop with progress bar ---
    pbar = tqdm(total=total, desc=f"{name} combos", initial=done_rows)
    for start in range(done_rows, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_idxs = comb_indices[start:end]

        # dispatch this batch to worker processes
        results = Parallel(
            n_jobs=n_procs,
            verbose=0
        )(delayed(eval_comb)(idxs) for idxs in batch_idxs)

        # append to CSV
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

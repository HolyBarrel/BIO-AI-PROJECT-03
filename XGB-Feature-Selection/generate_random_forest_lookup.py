import h5py
import pandas as pd

filename = 'test_data/5-heart-c_rf_mat.h5'

with h5py.File(filename, 'r') as f:
    # Confirm the dataset is there
    print("Available datasets:", list(f.keys()))      # -> ['data']
    
    # Read the entire dataset into a NumPy array
    data_arr = f['data'][()]                         # same as f['/data'][()]
    print("Loaded array shape:", data_arr.shape)
    print("Loaded array dtype:", data_arr.dtype)

# Convert to a pandas DataFrame (if it’s 2D/tabular)
df = pd.DataFrame(data_arr)
print(df.head())

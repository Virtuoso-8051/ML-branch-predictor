import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import time
import gc
import warnings

warnings.filterwarnings('ignore')

print("\n--- INITIATING FULL DATASET EXTRACTION ---")
start_time = time.time()

# THE LIMITER IS GONE. Loading the entire 665MB file.
print(" -> Loading raw CSV into memory (This may take a moment)...")
df = pd.read_csv('branch_data.csv')

row_count = len(df)
print(f" -> Successfully loaded {row_count:,} rows.")

print(" -> Engineering Features (RAM usage spiking)...")
df['PC_int'] = df['PC'].apply(lambda x: int(str(x), 16))
df['Target_int'] = df['Target'].apply(lambda x: int(str(x), 16))

df['History_1'] = df.groupby('PC')['Taken'].shift(1)
df['History_2'] = df.groupby('PC')['Taken'].shift(2)
df = df.dropna()

print(" -> DOWNCASTING: Compressing memory footprint...")
df['Taken'] = df['Taken'].astype(np.int8)
df['History_1'] = df['History_1'].astype(np.int8)
df['History_2'] = df['History_2'].astype(np.int8)

# Now that we've engineered the new columns, we don't need the Hex strings anymore
# We drop them and run Garbage Collection to forcefully free up your RAM
df.drop(columns=['PC', 'Target'], inplace=True)
gc.collect() 

X = df[['PC_int', 'Target_int', 'History_1', 'History_2']]
y = df['Taken']

print(" -> Splitting data into 80% Training / 20% Testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Free up the massive original dataframe from RAM now that we have our splits
del df 
gc.collect()

print("\n--- IGNORING TREES, PLANTING A FOREST (XGBOOST) ---")
print(" -> CPU Cores engaging. Training model on millions of rows...")
model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\n--- ADMINISTERING FINAL EXAM ---")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

end_time = time.time()
total_time = end_time - start_time

print("\n==========================================")
print("        FINAL PIPELINE COMPLETE!          ")
print(f"    Total Rows Processed: {row_count:,}  ")
print(f"    Final Predictor Accuracy: {accuracy * 100:.2f}%  ")
print(f"    Total Run Time: {total_time:.2f} seconds")
print("==========================================\n")

print("\n -> Exporting trained model to disk...")
model.save_model("branch_predictor_brain.json")
print(" -> Model saved successfully!")
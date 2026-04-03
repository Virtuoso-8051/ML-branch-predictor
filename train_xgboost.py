import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import time
import warnings

# Suppress minor warnings
warnings.filterwarnings('ignore')

print("\n--- PHASE 1: SECURING MEMORY & LOADING 1M ROWS ---")
start_time = time.time()

# We are scaling up 10x to 1,000,000 rows
df = pd.read_csv('branch_data.csv', nrows=1000000)

print(" -> Engineering Features...")
df['PC_int'] = df['PC'].apply(lambda x: int(str(x), 16))
df['Target_int'] = df['Target'].apply(lambda x: int(str(x), 16))

# Build the 2-bit history
df['History_1'] = df.groupby('PC')['Taken'].shift(1)
df['History_2'] = df.groupby('PC')['Taken'].shift(2)
df = df.dropna()

print(" -> DOWNCASTING: Shrinking RAM footprint by 80%...")
# Force 0s and 1s into 8-bit memory slots instead of 64-bit
df['Taken'] = df['Taken'].astype(np.int8)
df['History_1'] = df['History_1'].astype(np.int8)
df['History_2'] = df['History_2'].astype(np.int8)

X = df[['PC_int', 'Target_int', 'History_1', 'History_2']]
y = df['Taken']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- PHASE 2: IGNORING TREES, PLANTING A FOREST (XGBOOST) ---")
print(" -> Training 100 parallel XGBoost trees. Fans might spin up...")
# n_jobs=-1 tells XGBoost to use EVERY core your CPU has
model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\n--- PHASE 3: ADMINISTERING FINAL EXAM ---")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

end_time = time.time()
total_time = end_time - start_time

print("\n==========================================")
print("        XGBOOST PIPELINE SUCCESS!         ")
print(f"    Predictor Accuracy: {accuracy * 100:.2f}%  ")
print(f"    Total Run Time:     {total_time:.2f} seconds")
print("==========================================\n")
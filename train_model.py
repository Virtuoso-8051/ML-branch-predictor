import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings

# Suppress minor Pandas warnings to keep the terminal clean
warnings.filterwarnings('ignore')

print("\n--- PHASE 1: LOADING DATA ---")
# We load 100,000 rows for our first test so it runs in seconds, not hours.
df = pd.read_csv('branch_data.csv', nrows=100000)

# Convert Hex strings to Integers (ML models only do math, they can't read '0x400b')
df['PC_int'] = df['PC'].apply(lambda x: int(str(x), 16))
df['Target_int'] = df['Target'].apply(lambda x: int(str(x), 16))

print("--- PHASE 2: FEATURE ENGINEERING (THE SECRET WEAPON) ---")
# Here we recreate the "2-Bit History" from your handwritten notes!
# We group by the PC, and 'shift' the Taken column down. 
# This creates new columns asking: "Was this exact branch taken 1 step ago? 2 steps ago?"
df['History_1'] = df.groupby('PC')['Taken'].shift(1)
df['History_2'] = df.groupby('PC')['Taken'].shift(2)

# Drop the first few rows that don't have a history yet (they will be NaN)
df = df.dropna()

print("--- PHASE 3: SPLITTING DATA (80/20) ---")
# Features (X): What the model uses to guess
X = df[['PC_int', 'Target_int', 'History_1', 'History_2']]
# Label (y): The actual answer
y = df['Taken']

# Hide 20% of the data for the final exam
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- PHASE 4: TRAINING THE DECISION TREE ---")
# We use a Decision Tree with a max depth of 12 so it stays fast and lightweight
model = DecisionTreeClassifier(max_depth=12, random_state=42)
model.fit(X_train, y_train)

print("--- PHASE 5: ADMINISTERING FINAL EXAM ---")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n==========================================")
print("          PIPELINE SUCCESS!               ")
print(f"      AI Predictor Accuracy: {accuracy * 100:.2f}%  ")
print("==========================================\n")
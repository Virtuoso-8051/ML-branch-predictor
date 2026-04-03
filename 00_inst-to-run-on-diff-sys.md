1. Re-Compile the C++ Code
"g++ main_predictor.cpp -o ai_test -O3
g++ trace_simulator.cpp -o trace_sim -O3"

2. Replicate the ML Environment
"pip install pandas scikit-learn m2cgen xgboost==1.7.6"

3. Run the Pipeline
"python train_final.py

python extract_brain.py

./trace_sim"
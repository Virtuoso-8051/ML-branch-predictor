# ML Branch Predictor (V2 Architecture)
**Authors:** Anurag Raj & Aditi Chauhan 

> Replaces a classical CPU 2-bit branch predictor with a Machine Learning model. Uses Intel Pin to intercept C++ execution traces, trains an XGBoost model on an adversarial payload, and transpiles the AI weights into bare-metal C++ for a zero-dependency hardware simulator.

## What This Project Does
Modern CPUs use legacy hardware state-machines (like 2-bit saturating counters) to predict branch outcomes. This project proves that a lightweight Machine Learning model can mathematically outperform hardware predictors, especially on adversarial or complex periodic code.

**The Pipeline:**
1. **Intel Pin** intercepts a running C++ program (`beast_target.cpp`) and records exactly 5 features per branch, including an 8-bit local history window.
2. **XGBoost** trains on ~2.78 million rows using a strict chronological split to prevent time-travel data leakage.
3. **m2cgen** converts the trained model into a massive block of pure hardcoded C++ `if-else` statements (`ai_predictor.h`).
4. **C++ Simulators** replay the trace through both a true Bimodal 2-Bit predictor and the transpiled AI predictor to compare accuracies.
5. A **Streamlit Dashboard** provides a dynamic, visual UI for execution and analysis.

---

## The Results
Our model was tested against a highly volatile, adversarial C++ payload designed specifically to induce hardware aliasing and break bimodal hysteresis.

| Predictor Type | Evaluation Environment | Accuracy |
|---|---|---|
| **Classical Bimodal 2-Bit Counter** | Bare-Metal C++ Simulator | **71.12%** |
| **XGBoost AI Model** | Python (Post-Chronological Split) | **99.97%** |
| **Transpiled AI Model** | Bare-Metal C++ Simulator | **95.18%** |

> **Conclusion:** By leveraging an 8-bit local history window, the AI successfully reverse-engineers periodic mathematical traps (Mod-3, Mod-2, LCGs) that permanently cripple legacy 2-bit state machines.

---

## Project Structure

```text
ml_branch_predictor/
├── beast_target.cpp             # Adversarial C++ payload (Triggers hardware aliasing)
├── BranchDataGen.cpp            # Intel Pin tool — records 5-column trace to CSV
├── test_2bit.cpp                # Hardware simulator for the 2-bit baseline
├── train_final.py               # XGBoost chronological training engine
├── extract_brain.py             # Converts model → ai_predictor.h via m2cgen
├── trace_simulator.cpp          # Bare-metal C++ AI simulator
├── ai_predictor.h               # Auto-generated prediction logic (Zero Dependencies)
├── branch_predictor_brain.json  # Saved XGBoost model weights
└── app.py                       # Streamlit interactive frontend
```

---

## ⚙️ Phase 1: Trace Extraction (C++ & Intel Pin)

### Prerequisites
* Linux or WSL (Windows Subsystem for Linux)
* `g++` compiler
* Intel Pin Toolkit

### Step 1 — Set PIN_ROOT
Extract Intel Pin and map the environment variable. Add it permanently so it survives terminal restarts:
```bash
export PIN_ROOT=/home/yourname/pin-3.28-98749-g6643ecee5-gcc-linux
echo 'export PIN_ROOT=/home/yourname/pin-3.28-98749-g6643ecee5-gcc-linux' >> ~/.bashrc
source ~/.bashrc
```

### Step 2 — Compile the Target & Pin Tool
```bash
mkdir ~/ml_branch_predictor
cd ~/ml_branch_predictor

# Compile the adversarial payload
g++ beast_target.cpp -o beast_target

# Compile the Intel Pin extraction tool
cp BranchDataGen.cpp $PIN_ROOT/source/tools/MyPinTool/
cd $PIN_ROOT/source/tools/MyPinTool
mkdir -p obj-intel64
make obj-intel64/BranchDataGen.so
```

### Step 3 — Generate Branch Trace & Test Baseline
Run the target through Pin to generate `branch_data.csv` (Hardcapped at 5M branches to prevent disk I/O bottlenecks).
```bash
cd ~/ml_branch_predictor
$PIN_ROOT/pin -t $PIN_ROOT/source/tools/MyPinTool/obj-intel64/BranchDataGen.so -- ./beast_target

# Compile and run the legacy 2-bit hardware simulator
g++ test_2bit.cpp -o test_2bit
./test_2bit
```

---

## 🧠 Phase 2: ML Training & Transpilation

### Step 1 — Python & Conda Setup
If you don't have Conda, download the installer and initialize it:
```bash
cd ~
wget [https://repo.anaconda.com/archive/Anaconda3-2024.05-Linux-x86_64.sh](https://repo.anaconda.com/archive/Anaconda3-2024.05-Linux-x86_64.sh)
bash Anaconda3-2024.05-Linux-x86_64.sh
source ~/.bashrc
```
Create the ML environment:
```bash
cd ~/ml_branch_predictor
conda create -n ml_branch python=3.10 -y
conda activate ml_branch
conda install pandas scikit-learn numpy -y
pip install xgboost==2.0.3
```
> **Note:** We are using XGBoost 2.0+ along with a custom locally patched version of `m2cgen` to handle the new AST JSON list structure and bypass legacy pinning requirements.

### Step 2 — Train the AI Engine
```bash
python train_final.py
```
*This script uses a chronological timeline split to prevent data leakage. The model is saved automatically to `branch_predictor_brain.json`.*

### Step 3 — Extract the Brain to Bare-Metal C++
```bash
python extract_brain.py
```
*This converts the JSON weights into `ai_predictor.h` — a fully self-contained C++ namespace wrapper.*

### Step 4 — Run the AI Hardware Simulator
```bash
g++ trace_simulator.cpp -o trace_sim -O3
./trace_sim
```

---

## 📊 Streamlit Dashboard & UI

The frontend provides an interactive environment to test custom C++ code, view dynamic accuracy graphs, and analyze branch logic without touching the terminal.

### Install UI Dependencies
```bash
conda activate ml_branch
pip install streamlit plotly
```

### Run the Dashboard
```bash
cd ~/ml_branch_predictor
streamlit run app.py
```
Open your browser at `http://localhost:8501`. 

> ⚠️ **Note:** It may be possible that for traces which have millions of rows, the model takes around 5-10 minutes to give the results. Please don't refresh the page or press any buttons on the screen until it finishes processing.

### Dashboard Features
* **Live Execution:** Paste any C++ program. The app compiles it, injects Intel Pin, generates the trace, and runs both simulators automatically.
* **Trace Analytics:** Upload existing `branch_data.csv` files for rapid inference.
* **Visualizations:** Compare 2-Bit vs AI accuracy via rolling charts, analyze branch distribution pies, and view the hot-branch history table.

---

## Key Architecture Concepts

* **The 8-Bit Local History:** The secret to the AI's success. Instead of only looking at the immediate branch, Intel Pin masks an 8-bit integer tracking the last 8 outcomes. The AI uses this to decode complex sequences.
* **Data-Leakage Prevention:** We do not shuffle our train/test split. Time flows in one direction in a CPU. We train on the first 80% of the trace and predict the unseen final 20%.
* **m2cgen Transpilation:** We completely drop all ML libraries (like scikit or xgboost) for deployment. The model is converted into raw, inline C++ code so it can be evaluated exactly like a physical hardware logic gate.

### CSV Trace Format (V2)
```text
PC,Target,IsBackward,LocalHistory,Taken
4198954,4198960,0,170,1
4198954,4198960,0,85,0
...
```

---
### Dashboard Previews

*(Add Screenshot 1: Streamlit Dashboard Home / Code Input here)*

*(Add Screenshot 2: Visualizations & Trace Results here)*

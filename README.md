# ML Branch Predictor

> Replaces algorithmic CPU branch prediction with Machine Learning. Uses Intel Pin to intercept C++ traces, trains an XGBoost model on 31.8M instructions, and transpiles the AI weights into bare-metal C++ for a dependency-free hardware simulator.
1. **Accuracy** - 86.76%
2. **Model** - XGBoost
3. **Tool** - Intel PinTool
4. **UI**- Streamlit Dashboard



##  What this project does

Most CPUs use hand-crafted algorithms (like TAGE) to predict branch outcomes. This project asks: **can a machine learning model do the same job?**

The pipeline:
1. **Intel Pin** intercepts a running C++ program and records every branch decision (31.8M rows)
2. **XGBoost** trains on that data, learning branch patterns
3. **m2cgen** converts the trained model into pure hardcoded C++ `if-else` statements — zero runtime dependencies
4. A **C++ simulator** replays the trace through the AI predictor and measures accuracy
5. A **Streamlit dashboard** lets you paste any C++ code and see live results

---

## Results

| Predictor | Accuracy |
|---|---|
| ML model — trained + tested on sorting | **86.76%** |
| ML model — tested on unseen code | **83.83%** |
| 2-bit saturating counter (classical) | ~84–87% |


> The model generalises to unseen workloads with only a **2.93% accuracy drop** — it learned transferable branch behaviour patterns, not just sorting-specific ones.

---

##  Project structure

```
ml_branch_predictor/
├── beast_target.cpp             # Branch-heavy C++ program for training data
├── BranchDataGen.cpp            # Intel Pin tool — records branches to CSV
├── train_model.py               # Quick decision tree on 100k rows (sanity check)
├── train_xgboost.py             # XGBoost on 1M rows
├── train_final.py               # Full XGBoost training on all 31.8M rows
├── extract_brain.py             # Converts model → ai_predictor.h via m2cgen
├── trace_simulator.cpp          # C++ simulator that replays trace through AI
├── ai_predictor.h               # Auto-generated bare-metal C++ prediction logic
├── branch_predictor_brain.json  # Saved XGBoost model weights
└── app.py                       # Streamlit dashboard
```

---

## ⚙️ Setup

### Prerequisites

- Linux or WSL (Windows Subsystem for Linux)
- `g++` compiler
- Intel Pin ([download from Intel](https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-binary-instrumentation-tool-downloads.html))
- Anaconda or Python 3.10+

---

### Step 1 — Set PIN_ROOT

```bash
export PIN_ROOT=/home/yourname/pin-3.28-98749-g6643ecee5-gcc-linux
```

Add it permanently so it survives terminal restarts:

```bash
echo 'export PIN_ROOT=/home/yourname/pin-3.28-...' >> ~/.bashrc
source ~/.bashrc
```

---

### Step 2 — Create project folder

```bash
mkdir ~/ml_branch_predictor
cd ~/ml_branch_predictor
```

---

### Step 3 — Compile the target program

```bash
g++ beast_target.cpp -o beast_target
```

---

### Step 4 — Compile the Pin tool

```bash
cp BranchDataGen.cpp $PIN_ROOT/source/tools/MyPinTool/
cd $PIN_ROOT/source/tools/MyPinTool
mkdir -p obj-intel64
make obj-intel64/BranchDataGen.so
```

---

### Step 5 — Generate branch trace

```bash
cd ~/ml_branch_predictor
$PIN_ROOT/pin -t $PIN_ROOT/source/tools/MyPinTool/obj-intel64/BranchDataGen.so -- ./beast_target
```

Wait for:
```
--- BRANCH TRACE COMPLETE ---
Data successfully saved to branch_data.csv
```

Optionally preview the first 15 rows:
```bash
head -n 15 branch_data.csv
```

---

##  ML Training

### Step 1 — Install Anaconda

```bash
cd ~
wget https://repo.anaconda.com/archive/Anaconda3-2024.05-Linux-x86_64.sh
bash Anaconda3-2024.05-Linux-x86_64.sh
source ~/.bashrc
```

If you see a terms & conditions error:
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

---

### Step 2 — Create environment and install dependencies

```bash
cd ~/ml_branch_predictor
conda create -n ml_branch python=3.10 -y
conda activate ml_branch
conda install pandas scikit-learn numpy -y
pip install xgboost==1.7.6
```

> ⚠️ XGBoost must be pinned to **1.7.6** — newer versions store split values as lists which breaks m2cgen transpilation.

---

### Step 3 — Run training

| Script | Description |
|---|---|
| `train_model.py` | Decision tree on 100k rows — quick sanity check |
| `train_xgboost.py` | XGBoost on 1M rows |
| `train_final.py` | Full XGBoost on all 31.8M rows (**close all other apps**) |

```bash
python train_final.py
```

The model is saved automatically to `branch_predictor_brain.json`.

---

## Transpile Model to C++

### Step 1 — Install m2cgen

```bash
pip install m2cgen
```

### Step 2 — Extract the brain

```bash
python extract_brain.py
```

This converts `branch_predictor_brain.json` → `ai_predictor.h` — a fully self-contained C++ header with zero external dependencies.

### Step 3 — Compile and run the simulator

```bash
g++ trace_simulator.cpp -o trace_sim -O3
./trace_sim
```

Expected output:
```
==========================================
 SIMULATION COMPLETE!
 Total Branches: 31,800,000+
 Correct Guesses: ...
 C++ AI Accuracy: 86.76%
==========================================
```

---

## Streamlit Dashboard

### Install dependencies

```bash
pip install streamlit plotly pandas numpy
```

### Run the app

```bash
cd ~/ml_branch_predictor
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### Dashboard pages

| Page | What it does |
|---|---|
| Run C++ Code | Paste any C++ program — app compiles, runs Pin, generates trace, runs both predictors automatically |
| Upload CSV | Upload an existing `branch_data.csv` and run predictions |
| Visualizations | Rolling accuracy chart, branch distribution pie, hot branches table, taken rate table |

> PIN_ROOT is auto-detected from your environment. If the sidebar shows blank, paste your Pin path manually.<br>
>⚠️ It may be possible that for traces which have millions of rows, the model takes around 5-10 minutes to give the results, so please don't refresh the page or press any button on the screen till that time
---

##  Running on another machine

This project is not plug-and-play. The other person needs to:

| Requirement | What to do |
|---|---|
| Intel Pin | Download from Intel website and extract |
| `BranchDataGen.so` | Compile the Pin tool (Step 4 above) — **cannot be copied from another machine** |
| `g++` | `sudo apt install g++` |
| Python packages | `pip install streamlit plotly pandas numpy xgboost==1.7.6 m2cgen` |
| PIN_ROOT | Set in sidebar or export in `~/.bashrc` |



##  Key concepts

**Branch prediction** — CPUs speculatively execute instructions before knowing the outcome of an `if` statement. A wrong guess flushes the pipeline (10–20 cycle penalty).

**Intel Pin** — a dynamic binary instrumentation tool that injects code around any instruction while a program runs, recording PC address, target address, and taken/not-taken outcome for every branch.

**XGBoost** — gradient boosting algorithm that builds trees sequentially, each correcting the mistakes of the previous one. Parallelises the heavy math across CPU cores.

**m2cgen** — transpiles a trained ML model into pure language code (C++, Python, Java, etc.) with no ML library dependencies.

**Hot branches** — branch instructions that execute the most times. Getting these right has the biggest impact on overall accuracy since mispredicting them costs millions of cycles.

---

##  How the CSV trace looks

```
PC,Target,Taken
0x401234,0x401240,1
0x401234,0x401240,1
0x401234,0x401240,0
0x401267,0x401280,1
...
```

Each row = one branch instruction that executed. The same PC appears many times because it's inside a loop.

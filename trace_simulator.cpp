#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <iomanip>
#include "ai_predictor.h" // The AI Brain

using namespace std;

// The 2-bit history state
struct BranchHistory {
    double last_outcome = 0.0;
    double second_last_outcome = 0.0;
};

// The History Table
unordered_map<long long, BranchHistory> history_table;

// Function 1: Ask the AI for a prediction
bool get_ai_prediction(long long pc, long long target) {
    BranchHistory& hist = history_table[pc];

    double input[4];
    input[0] = (double)pc;
    input[1] = (double)target;
    input[2] = hist.last_outcome;
    input[3] = hist.second_last_outcome;

    double output[2] = {0.0, 0.0};
    predict_branch_ai(input, output);

    return output[1] > 0.5;
}

// Function 2: Update the history table
void update_history(long long pc, bool actual_taken) {
    BranchHistory& hist = history_table[pc];
    hist.second_last_outcome = hist.last_outcome;
    hist.last_outcome = actual_taken ? 1.0 : 0.0;
}

int main() {
    cout << "\n--- INITIATING C++ HARDWARE SIMULATOR ---" << endl;
    
    ifstream file("branch_data_test.csv");
    string line;

    if (!file.is_open()) {
        cout << "Error: Could not open branch_data_test.csv. Make sure it is in the same folder!" << endl;
        return 1;
    }

    // Skip the CSV header row ("PC,Target,Taken")
    getline(file, line);

    long long total_branches = 0;
    long long correct_predictions = 0;

    cout << " -> Booting AI Brain..." << endl;
    cout << " -> Parsing 665MB Trace File (Millions of instructions)..." << endl;

    // Read the file line by line
    while (getline(file, line)) {
        stringstream ss(line);
        string pc_str, target_str, taken_str;

        // Parse CSV columns
        getline(ss, pc_str, ',');
        getline(ss, target_str, ',');
        getline(ss, taken_str, ',');

        // Convert Hex strings (like "0x4000") to long long integers natively
        long long pc = stoull(pc_str, nullptr, 16);
        long long target = stoull(target_str, nullptr, 16);
        bool actual_taken = stoi(taken_str);

        // 1. AI makes its prediction BEFORE the branch executes
        bool predicted = get_ai_prediction(pc, target);
        
        // 2. Tally the score
        if (predicted == actual_taken) {
            correct_predictions++;
        }

        // 3. Update the history table with the actual result
        update_history(pc, actual_taken);
        total_branches++;

        // Print progress every 5 million rows so we know it is running
        if (total_branches % 5000000 == 0) {
            cout << "    ...Processed " << total_branches << " branches..." << endl;
        }
    }

    // Calculate final accuracy
    double accuracy = ((double)correct_predictions / total_branches) * 100.0;

    cout << "\n==========================================" << endl;
    cout << "          SIMULATION COMPLETE!            " << endl;
    cout << "  Total Branches:  " << total_branches << endl;
    cout << "  Correct Guesses: " << correct_predictions << endl;
    cout << "  C++ AI Accuracy: " << fixed << setprecision(2) << accuracy << "%" << endl;
    cout << "==========================================\n" << endl;

    return 0;
}
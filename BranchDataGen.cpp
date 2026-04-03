#include "pin.H"
#include <iostream>
#include <fstream>

using namespace std;

ofstream OutFile;

VOID RecordBranch(VOID * ip, VOID * target, BOOL taken) {
    // Outputs a clean, comma-separated row for the Pandas DataFrame
    OutFile << ip << "," << target << "," << taken << "\n";
}

VOID Instruction(INS ins, VOID *v) {
    // We only want to log conditional branches (like if/else statements)
    // INS_HasFallThrough ensures it's a branch that actually makes a choice
    if (INS_IsBranch(ins) && INS_HasFallThrough(ins)) {
        INS_InsertCall(
            ins, IPOINT_BEFORE, (AFUNPTR)RecordBranch,
            IARG_INST_PTR,
            IARG_BRANCH_TARGET_ADDR,
            IARG_BRANCH_TAKEN,
            IARG_END);
    }
}

VOID Fini(INT32 code, VOID *v) {
    OutFile.close();
    cout << "\n--- BRANCH TRACE COMPLETE ---\n";
    cout << "Data successfully saved to branch_data.csv\n";
    cout << "-----------------------------\n";
}

int main(int argc, char * argv[]) {
    if (PIN_Init(argc, argv)) return -1;
    
    OutFile.open("branch_data.csv");
    // Write the CSV header so Python knows what the columns are
    OutFile << "PC,Target,Taken\n"; 
    
    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);
    PIN_StartProgram();
    
    return 0;
}
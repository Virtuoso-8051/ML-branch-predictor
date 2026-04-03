#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            // This condition creates massive amounts of branch data
            if (arr[j] > arr[j + 1]) { 
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) { 
        int pivot = arr[high];
        int i = (low - 1);
        for (int j = low; j <= high - 1; j++) {
            // Another highly variable branch
            if (arr[j] < pivot) { 
                i++;
                swap(arr[i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[high]);
        int pi = i + 1;
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    // Generate an array of 5,000 random numbers
    int dataSize = 5000;
    vector<int> data1(dataSize);
    vector<int> data2(dataSize);
    
    for(int i = 0; i < dataSize; i++) {
        int val = rand() % 10000;
        data1[i] = val;
        data2[i] = val;
    }
    
    cout << "Executing Beast Mode Targets...\n";
    bubbleSort(data1);
    quickSort(data2, 0, data2.size() - 1);
    
    cout << "Execution Complete. Sorting finished.\n";
    return 0;
}
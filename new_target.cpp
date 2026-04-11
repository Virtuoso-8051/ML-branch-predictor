#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

int binarySearch(vector<int>& arr, int target) {
    int low = 0, high = arr.size() - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}

void matrixMultiply(vector<vector<int>>& A, vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                if (A[i][k] > 0 && B[k][j] > 0)
                    C[i][j] += A[i][k] * B[k][j];
}

int main() {
    // Binary search on sorted array
    int size = 10000;
    vector<int> arr(size);
    for (int i = 0; i < size; i++) arr[i] = i * 2;
    for (int i = 0; i < 50000; i++)
        binarySearch(arr, rand() % (size * 2));

    // Matrix multiplication
    int n = 100;
    vector<vector<int>> A(n, vector<int>(n)), B(n, vector<int>(n)), C(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    matrixMultiply(A, B, C, n);

    cout << "New target complete!\n";
    return 0;
}

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
using namespace std;
const int X = 1024;
size_t threadsPerBlock;
size_t numberOfBlocks;

void A_reset(float **A)
{
    for (int i = 0; i < X; i++)
    {
        for (int j = 0; j < i; j++)
            A[i][j] = 0;
        A[i][i] = 1.0;
        for (int j = i + 1; j < X; j++)
            A[i][j] = rand();
    }
    for (int k = 0; k < X; k++)
        for (int i = k + 1; i < X; i++)
            for (int j = 0; j < X; j++)
                A[i][j] += A[k][j];
}

//串行算法:
void normal(int n, float **A)
{
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
        {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

//除法:
__global__ void division_kernel(float **A, int k, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index+k+1; i < n ; i += stride)
    {
        float element = A[k][k];
        float temp = A[k][i];
        A[k][i] = (float)temp / element;
    }
    return;
}

//消元:
__global__ void eliminate_kernel(float **A, int k, int N)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tx == 0)
        A[k][k] = 1;  
    int row = k + 1 + blockIdx.x; 
    while (row < N)
    {
        int tid = threadIdx.x;
        while (k + 1 + tid < N)
        {
            int col = k + 1 + tid;
            float temp_1 = A[row][col];
            float temp_2 = A[row][k];
            float temp_3 = A[k][col];
            A[row][col] = temp_1 - temp_2 * temp_3;
            tid = tid + blockDim.x;
        }
        __syncthreads(); 
        if (threadIdx.x == 0)
        {
            A[row][k] = 0;
        }
        row += gridDim.x;
    }
    return;
}

//并行算法
void sp(int n, float **A)
{
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    threadsPerBlock = 32;
    numberOfBlocks = 32 * numberOfSMs;
    size_t size = n * n * sizeof(float);
    float **A_d;
    cudaMalloc((void **)&A_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);

    for (int k = 0; k < n; k++)
    {
        division_kernel<<<numberOfBlocks, threadsPerBlock>>>(A_d, k, n); 
        cudaDeviceSynchronize();      
        eliminate_kernel<<<numberOfBlocks, threadsPerBlock>>>(A_d, k, n);
        cudaDeviceSynchronize();
        // ret = cudaGetLastError();
        // if (ret != cudaSuccess)
        // {
        //     printf("eliminate_kernel failed, %s\n", cudaGetErrorString(ret));
        // }
    }
    cudaMemcpy(A, A_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);

}
int main()
{
    float **A = new float *[X];
    for(int i = 0; i < X; i++)
        A[i] = new float[X];
    int step = 64;
    clock_t start ,finish_1, finish_2;
    for (int i = step; i <= X; i += step)
    {
        //串行
        A_reset(A);
        start = clock();
        normal(i, A);
        finish_1 = clock();
        float time_1 = ( finish_1 - start)/float (CLOCKS_PER_SEC);
        //并行
        A_reset(A);
        start = clock();
        sp(i, A);
        finish_2 = clock();
        float time_2 = ( finish_2 - start)/float (CLOCKS_PER_SEC);
        cout<<fixed << setprecision(6);
        cout<< time_1 << "    " << time_2 << endl;
    }
    return 0;
}
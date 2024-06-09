#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
using namespace std;
const int X = 1024;
size_t threadsPerBlock;
size_t numberOfBlocks;

void A_reset(float *A)
{
    for (int i = 0; i < X; i++)
    {
        for (int j = 0; j < i; j++)
            A[i*X+j] = 0;
        A[i*X+i] = 1.0;
        for (int j = i + 1; j < X; j++)
            A[i*X+j] = rand();
    }
    for (int k = 0; k < X; k++)
        for (int i = k + 1; i < X; i++)
            for (int j = 0; j < X; j++)
                A[i*X+j] += A[k*X+j];
}

// 串行:
void normal(int n, float *A)
{
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
        {
            A[k*n+j] /= A[k*n+k];
        }
        A[k*n+k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                A[i*n+j] -= A[i*n+k] * A[k*n+j];
            }
            A[i*n+k] = 0;
        }
    }
}

//除法:
__global__ void division_kernel(float *A, int k, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index+k+1; i < n ; i += stride)
    {
        float element = A[k*n+k];
        float temp = A[k*n+i];
        A[k*n+i] = (float)temp / element;
    }
    return;
}

//消元:
__global__ void eliminate_kernel(float *A, int k, int N)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tx == 0)
        A[k*N+k] = 1;  //对角线元素设为 1
    int row = k + 1 + blockIdx.x; //每个块负责一行
    while (row < N)
    {
        int tid = threadIdx.x;
        while (k + 1 + tid < N)
        {
            int col = k + 1 + tid;
            float temp_1 = A[row*N+col];
            float temp_2 = A[row*N+k];
            float temp_3 = A[k*N+col];
            A[row*N+col] = temp_1 - temp_2 * temp_3;
            tid = tid + blockDim.x;
        }
        __syncthreads(); //块内同步
        if (threadIdx.x == 0)
        {
            A[row*N+k] = 0;
        }
        row += gridDim.x;
    }
    return;
}

//并行算法
void sp(int n, float *A)
{
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    threadsPerBlock = 32;
    numberOfBlocks = 32 * numberOfSMs;

    for (int k = 0; k < n; k++)
    {
        division_kernel<<<numberOfBlocks, threadsPerBlock>>>(A, k, n); 
        cudaDeviceSynchronize();  
        eliminate_kernel<<<numberOfBlocks, threadsPerBlock>>>(A, k, n); 
        cudaDeviceSynchronize();
    }

}
int main()
{
    
    size_t size = X * X * sizeof(float);
    float *A;
    cudaMallocManaged(&A, size);
    
    int step = 64;
    clock_t start ,finish_1,finish_2;
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
    cudaFree(A);
    cout << "hello" << endl;
    return 0;
}
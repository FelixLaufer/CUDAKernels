/**
* Efficient sum reduction kernel.
* Reduces float array *in_data to the sum of its elements and stores result in *out_result.
* Usage: sum_eduction<<<grid, block>>> (size, data_in_dev, out_result_dev)
* where block = 1024 (or 512, depending on device) and grid = (size + block - 1).
* Requires computing capability >= 3.0 (Kepler) due to __shfl_down operations.
**/
__global__ void sum_reduction(const int size, const float *in_data, float *out_result)
{
    __shared__ float temp[32];

    int tid = threadIdx.x;
    int i = tid + blockIdx.x * blockDim.x;

    float value = (i < size) ? in_data[i] : 0.0;

    for (int i = 16; i > 0; i /= 2)
        value += __shfl_down(value, i);

    if (tid % 32 == 0)
        temp[tid / 32] = value;

    __syncthreads();

    int warps = blockDim.x / 32;

    if (tid < warps)
    {
        value = temp[tid];
        for (int i = warps / 2; i > 0; i /= 2)
            value += __shfl_down(value, i);

        if (tid == 0)
            atomicAdd(out_result, value);
    }   
}

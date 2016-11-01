#define TILE_DIM 32
#define UNROLL 8

/**
* Matrix transpose kernel 
* matrix dimensions mxn must be a multiple of TILE_DIM
* Usage: matrix_transpose <<<grid, block>>> (matrix_dev, matrix_transposed_dev, m, n)
* where block = dim3(TILE_DIM, UNROLL) 
* and grid = dim3((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM)
* optimal TILE_DIM and UNROLL are dependent on the device
* Settings 780Ti: TILEDIM 32, UNROLL 8
* Settings 960M: TILEDIM 16, UNROLL 4 or 8
**/
__global__ void matrix_transpose(const float *idata, float *odata, const int m, const int n)
{
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int i = 0; i < TILE_DIM; i += UNROLL)
        tile[threadIdx.y+i][threadIdx.x] = idata[(y+i)*n + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int i = 0; i < TILE_DIM; i += UNROLL)
        odata[(y+i)*m + x] = tile[threadIdx.x][threadIdx.y+i];
}

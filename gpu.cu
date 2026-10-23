#include <cuda_runtime.h>
#include <iostream>

__global__ void first_compute(int* resultat)
{
    *resultat = 42;
}

bool test_gpu()
{
    int* result_gpu = nullptr;

    auto status = cudaMalloc(
        reinterpret_cast<void**>(&result_gpu),
        sizeof(int)
    );

    if (status != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(status) << std::endl;
        return false;
    }

    first_compute<<<1, 1>>>(result_gpu);
    status = cudaGetLastError();

    int resultat_cpu = 0;

    if (status == cudaSuccess)
    {
        status = cudaMemcpy(
            &resultat_cpu,
            result_gpu,
            sizeof(int),
            cudaMemcpyDeviceToHost
        );
    }

    cudaFree(result_gpu);

    if (status != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(status) << std::endl;
        return false;
    }

    return resultat_cpu == 42;
}
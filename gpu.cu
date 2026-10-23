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

__global__ void predictions_batch(
    const double* entries,
    const double* weights,
    const double* biais,
    int* predictions,
    int nb_entrees,
    int nb_exemples,
    int nb_eleves)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nb_exemples * nb_eleves)
        return;

    const int eleve = index / nb_exemples;
    const int exemple = index % nb_exemples;

    double sum = 0.0;

    for (int i = 0; i < nb_entrees; ++i)
    {
        sum += weights[eleve * nb_entrees + i]
            * entries[exemple * nb_entrees + i];
    }

    predictions[index] =
        sum + biais[eleve] < 0.0 ? -1 : 1;
}

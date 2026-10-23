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

__global__ void update_weights_batch(
    const double* entries,
    const int* targets,
    const int* predictions,
    double* weights,
    double* biais,
    float eta,
    int nb_entrees,
    int nb_exemples,
    int nb_eleves)
{
    const int nb_parametres = nb_entrees + 1;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nb_eleves * nb_parametres)
        return;

    const int eleve = index / nb_parametres;
    const int parametre = index % nb_parametres;

    double delta = 0.0;

    for (int exemple = 0; exemple < nb_exemples; ++exemple)
    {
        const int prediction =
            predictions[eleve * nb_exemples + exemple];

        const float correction =
            eta * static_cast<float>(targets[exemple] - prediction);

        if (parametre < nb_entrees)
        {
            delta += correction * static_cast<float>(
                entries[exemple * nb_entrees + parametre]
            );
        }
        else
        {
            delta += correction;
        }
    }

    if (parametre < nb_entrees)
        weights[eleve * nb_entrees + parametre] += delta;
    else
        biais[eleve] += delta;
}

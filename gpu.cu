#include <cuda_runtime.h>

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace
{
    void check_cuda(cudaError_t status, const char* operation)
    {
        if (status != cudaSuccess)
            throw runtime_error(
                string(operation) + " : " + cudaGetErrorString(status));
    }

    template <typename T>
    class DeviceBuffer
    {
    public:
        T* data = nullptr;

        explicit DeviceBuffer(size_t count)
        {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data),
                                  count * sizeof(T)), "cudaMalloc");
        }

        ~DeviceBuffer()
        {
            if (data != nullptr)
                cudaFree(data);
        }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    };
}

__global__ void first_compute(int* resultat)
{
    *resultat = 42;
}

bool test_gpu()
{
    try
    {
        DeviceBuffer<int> resultat_gpu(1);
        first_compute<<<1, 1>>>(resultat_gpu.data);
        check_cuda(cudaGetLastError(), "Launching first compute");
        check_cuda(cudaDeviceSynchronize(), "Executing first_compute");

        int resultat_cpu = 0;
        check_cuda(cudaMemcpy(&resultat_cpu, resultat_gpu.data, sizeof(int),
                              cudaMemcpyDeviceToHost), "Reading GPU test");
        return resultat_cpu == 42;
    }
    catch (const exception& error)
    {
        cerr << "Test CUDA : " << error.what() << endl;
        return false;
    }
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
        sum += weights[eleve * nb_entrees + i]
            * entries[exemple * nb_entrees + i];

    predictions[index] = sum + biais[eleve] < 0.0 ? -1 : 1;
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
        const int prediction = predictions[eleve * nb_exemples + exemple];
        const float correction =
            eta * static_cast<float>(targets[exemple] - prediction);
        if (parametre < nb_entrees)
            delta += correction * static_cast<float>(
                entries[exemple * nb_entrees + parametre]);
        else
            delta += correction;
    }

    if (parametre < nb_entrees)
        weights[eleve * nb_entrees + parametre] += delta;
    else
        biais[eleve] += delta;
}

__global__ void update_weights_online(
    const double* entries,
    const int* targets,
    double* weights,
    double* biases,
    const size_t* iterations,
    float eta,
    int nb_entrees,
    int nb_exemples,
    int nb_eleves)
{
    const int eleve = static_cast<int>(blockIdx.x);
    const unsigned int tid = threadIdx.x;

    if (eleve >= nb_eleves
        || iterations[eleve] != static_cast<size_t>(-1))
        return;

    extern __shared__ double scratch[];

    double* student_weights = weights
        + static_cast<size_t>(eleve) * nb_entrees;

    for (int exemple = 0; exemple < nb_exemples; ++exemple)
    {
        const double* input = entries
            + static_cast<size_t>(exemple) * nb_entrees;

        double sum = 0.0;

        for (size_t i = tid;
             i < static_cast<size_t>(nb_entrees);
             i += blockDim.x)
        {
            sum += student_weights[i] * input[i];
        }

        scratch[tid] = sum;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2;
             stride > 0; stride /= 2)
        {
            if (tid < stride)
                scratch[tid] += scratch[tid + stride];

            __syncthreads();
        }

        if (tid == 0)
        {
            const int prediction =
                scratch[0] + biases[eleve] < 0.0 ? -1 : 1;

            scratch[blockDim.x] =
                eta * static_cast<float>(
                    targets[exemple] - prediction
                );
        }

        __syncthreads();

        const double correction = scratch[blockDim.x];

        if (correction != 0.0)
        {
            for (size_t i = tid;
                 i < static_cast<size_t>(nb_entrees);
                 i += blockDim.x)
            {
                student_weights[i] += correction * input[i];
            }

            if (tid == 0)
                biases[eleve] += correction;
        }
        __syncthreads();
    }
}

void learn_gpu(
    const vector<double>& entries,
    const vector<int>& targets,
    vector<double>& weights,
    vector<double>& biases,
    vector<size_t>& iterations,
    int nb_entrees,
    float eta,
    size_t max_epochs,
    bool online)
{
    const size_t int_max =
        numeric_limits<int>::max();
    if (nb_entrees <= 0 || static_cast<size_t>(nb_entrees) >= int_max
        || targets.empty() || biases.empty())
        throw invalid_argument("Null or void GPU dimensions");

    const size_t d = static_cast<size_t>(nb_entrees);
    const size_t p = targets.size();
    const size_t s = biases.size();

    // verify int for kernels
    if (p > int_max / d || s > int_max / p || s > int_max / (d + 1))
        throw invalid_argument("Dimensions too big for kernel");
    if (entries.size() != p * d || weights.size() != s * d)
        throw invalid_argument("GPU batch incompatible size");
    for (int target : targets)
        if (target != -1 && target != 1)
            throw invalid_argument("Targets must be -1 or 1");

    const int nb_exemples = static_cast<int>(p);
    const int nb_eleves = static_cast<int>(s);
    const size_t prediction_count = s * p;

    DeviceBuffer<double> gpu_entries(entries.size());
    DeviceBuffer<int> gpu_targets(targets.size());
    DeviceBuffer<double> gpu_weights(weights.size());
    DeviceBuffer<double> gpu_biases(biases.size());
    DeviceBuffer<int> gpu_predictions(prediction_count);
    DeviceBuffer<size_t> gpu_iterations(s);

    check_cuda(cudaMemcpy(gpu_entries.data, entries.data(),
                          entries.size() * sizeof(double), cudaMemcpyHostToDevice),
               "Entry copies");
    check_cuda(cudaMemcpy(gpu_targets.data, targets.data(),
                          targets.size() * sizeof(int), cudaMemcpyHostToDevice),
               "Target copies");
    check_cuda(cudaMemcpy(gpu_weights.data, weights.data(),
                          weights.size() * sizeof(double), cudaMemcpyHostToDevice),
               "Weights copies");
    check_cuda(cudaMemcpy(gpu_biases.data, biases.data(),
                          biases.size() * sizeof(double), cudaMemcpyHostToDevice),
               "Biais copies");

    constexpr unsigned int threads = 256;
    const unsigned int prediction_blocks = static_cast<unsigned int>((prediction_count + threads - 1) / threads);
    const unsigned int update_blocks = static_cast<unsigned int>((s * (d + 1) + threads - 1) / threads);

    const size_t failure = numeric_limits<size_t>::max();
    iterations.assign(s, failure);
    vector<int> predictions(prediction_count);

    // verify initial state and state after last update
    for (size_t epoch = 0; ; ++epoch)
    {
        predictions_batch<<<prediction_blocks, threads>>>(
            gpu_entries.data, gpu_weights.data, gpu_biases.data,
            gpu_predictions.data, nb_entrees, nb_exemples, nb_eleves
        );
        check_cuda(cudaGetLastError(), "Launching predictions");
        check_cuda(cudaMemcpy(predictions.data(), gpu_predictions.data,
                              prediction_count * sizeof(int), cudaMemcpyDeviceToHost),
                   "Reading predictions");

        bool all_converged = true;
        for (size_t student = 0; student < s; ++student)
        {
            if (iterations[student] != failure)
                continue;
            bool correct = true;
            for (size_t example = 0; example < p; ++example)
            {
                if (predictions[student * p + example] != targets[example])
                {
                    correct = false;
                    break;
                }
            }
            if (correct)
                iterations[student] = epoch;
            else
                all_converged = false;
        }

        if (all_converged || epoch == max_epochs)
            break;

        // student converged stays null
        if (online)
        {
            check_cuda(
                cudaMemcpy(
                    gpu_iterations.data,
                    iterations.data(),
                    s * sizeof(size_t),
                    cudaMemcpyHostToDevice
                ),
                "Copy of online convergence"
            );

            update_weights_online<<<
                nb_eleves,
                threads,
                (threads + 1) * sizeof(double)
                >>>(
                    gpu_entries.data,
                    gpu_targets.data,
                    gpu_weights.data,
                    gpu_biases.data,
                    gpu_iterations.data,
                    eta,
                    nb_entrees,
                    nb_exemples,
                    nb_eleves
                );

            check_cuda(
                cudaGetLastError(),
                "Launching online update"
            );
        }
        else
        {
            update_weights_batch<<<update_blocks, threads>>>(
                gpu_entries.data,
                gpu_targets.data,
                gpu_predictions.data,
                gpu_weights.data,
                gpu_biases.data,
                eta,
                nb_entrees,
                nb_exemples,
                nb_eleves
            );

            check_cuda(
                cudaGetLastError(),
                "Launching batch update"
            );
        }
    }

    check_cuda(cudaMemcpy(weights.data(), gpu_weights.data,
                          weights.size() * sizeof(double), cudaMemcpyDeviceToHost),
               "Reading weights");
    check_cuda(cudaMemcpy(biases.data(), gpu_biases.data,
                          biases.size() * sizeof(double), cudaMemcpyDeviceToHost),
               "Reading biais");
}

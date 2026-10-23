#include <iostream>
#include <ranges>
#include <map>
#include <random>
#include <vector>
#include <stdexcept>
#include <thread>
#include <cmath>
#include <matplot/matplot.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <filesystem>
#include <chrono>
#include <limits>
#ifdef HAS_CUDA
#include <cuda_runtime_api.h>
#endif
#ifdef HAS_CUDA_KERNELS
bool test_gpu();

void learn_batch_gpu(
    const std::vector<double>& entries,
    const std::vector<int>& targets,
    std::vector<double>& weights,
    std::vector<double>& biases,
    std::vector<std::size_t>& iterations,
    int nb_entrees,
    float eta,
    std::size_t max_epochs);
#endif

using namespace std;
using namespace matplot;

int heaviside(const double x)
{
    return x < 0 ? -1 : 1;
}

double sum(const vector<double>& weights, const vector<int>& input)
{
    double sum = 0;
    for (int i = 0; i < weights.size(); i++)
        sum += weights[i] * input[i];

    return sum;
}

vector<double> add(const vector<double>& weights, const vector<double>& delta)
{
    vector<double> result;
    for (int i = 0; i < weights.size(); i++)
        result.push_back(weights[i] + delta[i]);
    return result;
}

bool equals(const vector<int>& a, const vector<int>& b)
{
    if (a.size() != b.size()) return false;
    for (int i = 0; i < a.size(); i++)
        if (a[i] != b[i]) return false;
    return true;
}

class Perceptron
{
    vector<double> weights;
    float eta;
    double biais;
    map<vector<int>, int> learn_data;

public:
    enum LearningType
    {
        ONLINE,
        BATCH
    };

    [[nodiscard]] const vector<double>& get_weights() const
    {
        return this->weights;
    }

    Perceptron()
    {
        this->weights = {};
        this->eta = 1.0;
        this->biais = 1.0;
        this->learn_data = {};
    }

    Perceptron(const vector<double>& weights, const float eta, const double biais,
               const map<vector<int>, int>& learn_data) :
        learn_data(learn_data)
    {
        this->weights = weights;
        this->eta = eta;
        this->biais = biais;
    }

    Perceptron(const vector<double>& weights, const float eta, double biais)
    {
        this->weights = weights;
        this->eta = eta;
        this->biais = biais;
        this->learn_data = {};
    }

    Perceptron(const int n, const float eta, const double biais,
               const map<vector<int>, int>& learn_data) :
        learn_data(learn_data)
    {
        vector<double> init(n);

        random_device rd;
        mt19937 gen(rd());

        uniform_real_distribution weight_dist(-1.0, 1.0);

        for (double& w : init) { w = weight_dist(gen); }

        this->weights = init;
        this->eta = eta;
        this->biais = biais;
    }

    Perceptron(const int n, const float eta, const double biais)
    {
        vector<double> init(n);

        random_device rd;
        mt19937 gen(rd());

        uniform_real_distribution weight_dist(-1.0, 1.0);

        for (double& w : init) { w = weight_dist(gen); }

        this->weights = init;
        this->eta = eta;
        this->biais = biais;
        this->learn_data = {};
    }

    [[nodiscard]] int output(const vector<int>& input) const
    {
        if (input.size() != this->weights.size())
        {
            throw invalid_argument("Input size does not match number of input weights.");
        }

        return heaviside(sum(this->weights, input) + this->biais);
    }

    [[nodiscard]] bool verify() const
    {
        // building output vector for learn data (correct version)
        vector<int> output_vector;
        for (const auto& val : this->learn_data | views::values)
        {
            output_vector.push_back(val);
        }

        // building current output vector (might be false)
        vector<int> current_output;
        for (const auto& key : this->learn_data | views::keys)
        {
            current_output.push_back(this->output(key));
        }

        if (equals(current_output, output_vector)) return true;
        return false;
    }

private:
    size_t learn_incremental(const size_t max_epochs)
    {
        for (size_t epoch = 0; epoch < max_epochs; epoch++)
        {
            if (verify()) return epoch;
            for (const auto& [fst, snd] : this->learn_data)
            {
                auto x = fst;
                const int y = this->output(fst);

                if (const int t = snd; y != t)
                {
                    const double correction = this->eta * static_cast<float>(t - y);

                    for (size_t i = 0; i < this->weights.size(); ++i)
                        this->weights[i] += correction * x[i];
                    this->biais += correction;
                }
            }
        }
        return verify() ? max_epochs : -1;
    }

    size_t learn_batch(const size_t max_epochs)
    {
        for (size_t epoch = 0; epoch < max_epochs; epoch++)
        {
            if (verify()) return epoch;
            vector delta(this->weights.size(), 0.0);
            double delta_biais = 0;

            for (const auto& [fst, snd] : this->learn_data)
            {
                auto x = fst;
                const int y = this->output(fst);

                if (const int t = snd; y != t)
                {
                    vector<double> delta_local;
                    for (const auto x_prime : x)
                    {
                        delta_local.push_back(this->eta * static_cast<float>(t - y) * static_cast<float>(x_prime));
                    }
                    delta = add(delta, delta_local);
                    delta_biais += this->eta * static_cast<float>(t - y);
                }
            }

            this->weights = add(this->weights, delta);
            this->biais += delta_biais;
        }
        return verify() ? max_epochs : -1;
    }

public:
    size_t learn(const LearningType type_, const size_t max_epochs = 1000)
    {
        switch (type_)
        {
        case BATCH:
            return learn_batch(max_epochs);
        default:
            return learn_incremental(max_epochs);
        }
    }
};

map<vector<int>, int> create_prof_dataset(const int n, const int p, Perceptron& prof, const float eta = 0.5)
{
    map<vector<int>, int> ls_dataset = {};

    random_device rd;
    mt19937 gen(rd());

    prof = Perceptron(n + 1, eta, 0.0);

    for (int i = 0; i < p; i++)
    {
        vector<int> vec;
        vec.push_back(1);
        for (int j = 0; j < n; j++)
        {
            vec.push_back(static_cast<int>(gen()) % 2);
        }
        ls_dataset.emplace(vec, prof.output(vec));
    }
    return ls_dataset;
}

struct Result
{
    Perceptron p;
    size_t iterations{};
    double recouvrement{};
};

double recouvrement(const vector<double>& a, const vector<double>& b)
{
    if (a.size() != b.size()) throw invalid_argument("Wrong number of input sizes");

    double ww = 0;

    for (int i = 0; i < a.size(); i++) ww += a[i] * b[i];

    double norm_a = 0;
    double norm_b = 0;

    for (int i = 0; i < a.size(); i++)
    {
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a == 0.0 || norm_b == 0.0)
        throw invalid_argument("Undefined : null vector");

    return cos(ww / (sqrt(norm_a) * sqrt(norm_b)));
}

vector<Result> learn_all(const map<vector<int>, int>& dataset, const int n, const Perceptron& prof, const float eta,
                         Perceptron::LearningType type = Perceptron::ONLINE, bool use_gpu = false)
{
    vector<thread> threads;
    threads.reserve(20);

    vector<Result> results(20);
    vector<exception_ptr> errors(20);

    const auto& prof_weights = prof.get_weights();

    vector<vector<double>> initial_weights(20);

    for (int i = 0; i < 20; ++i)
    {
        Perceptron initial(n + 1, eta, 0.0);
        initial_weights[i] = initial.get_weights();
    }

    const int nb_entrees = n + 1;
    const int nb_exemples = static_cast<int>(dataset.size());
    const int nb_eleves = static_cast<int>(initial_weights.size());

    vector<double> flat_entries;
    vector<int> targets;

    flat_entries.reserve(
        static_cast<size_t>(nb_exemples) * nb_entrees
    );
    targets.reserve(nb_exemples);

    for (const auto& [input, target] : dataset)
    {
        flat_entries.insert(
            flat_entries.end(),
            input.begin(),
            input.end()
        );

        targets.push_back(target);
    }

    // weights of each student
    vector<double> flat_weights;

    flat_weights.reserve(
        static_cast<size_t>(nb_eleves) * nb_entrees
    );

    for (const auto& weights : initial_weights)
    {
        flat_weights.insert(
            flat_weights.end(),
            weights.begin(),
            weights.end()
        );
    }

    // each student initiated with a biais of 0.0
    vector biases(nb_eleves, 0.0);
#ifdef HAS_CUDA_KERNELS
    if (use_gpu && type == Perceptron::BATCH)
    {
        try
        {
            vector<size_t> iterations;

            learn_batch_gpu(
                flat_entries, targets, flat_weights, biases,
                iterations, nb_entrees, eta, 1000
            );

            for (int i = 0; i < nb_eleves; ++i)
            {
                vector<double> weights(
                    flat_weights.begin() + i * nb_entrees,
                    flat_weights.begin() + (i + 1) * nb_entrees
                );

                results[i] = {
                    Perceptron(weights, eta, biases[i], dataset),
                    iterations[i],
                    recouvrement(prof_weights, weights)
                };
            }

            return results;
        }
        catch (const exception& error)
        {
            cerr << "CUDA : " << error.what()
                << " -- retour a l'apprentissage CPU."
                << endl;
        }
    }
#endif

    for (int i = 0; i < 20; i++)
    {
        threads.emplace_back([&, i, n, eta, type]()
        {
            try
            {
                auto eleve = Perceptron(initial_weights[i], eta, 0.0, dataset);
                // ReSharper disable once CppDFAUnusedValue
                // ReSharper disable once CppDFAUnreadVariable
                const size_t it = eleve.learn(type);

                const double r = recouvrement(
                    prof_weights, eleve.get_weights()
                );

                results[i] = {eleve, it, r};
            }
            catch (...)
            {
                errors[i] = current_exception();
            }
        });
    }

    for (thread& t : threads) t.join();

    for (const auto& error : errors)
    {
        if (error)
            rethrow_exception(error);
    }

    return results;
}

using Results =
map<pair<int, int>, pair<double, double>>;

// one panel for each methode with it and r in each cell
void plot_panel(
    const Results& results,
    int position,
    const string& method,
    double eta,
    const vector<int>& ns,
    const vector<int>& ps)
{
    // color set to track R
    vector colors(
        ps.size(), vector<double>(ns.size())
    );

    for (size_t row = 0; row < ps.size(); ++row)
    {
        for (size_t col = 0; col < ns.size(); ++col)
        {
            colors[row][col] =
                results.at({ns[col], ps[row]}).second;
        }
    }

    subplot(1, 2, position);

    auto heatmap = matplot::heatmap(colors);
    heatmap->always_hide_labels(true);

    hold(on);

    colormap({
        {1.00, 0.95, 0.85},
        {0.95, 0.95, 0.95},
        {0.65, 0.85, 1.00}
    });

    caxis({-1.0, 1.0});
    colorbar();
    xticks({1, 2, 3, 4, 5, 6});
    yticks({1, 2, 3, 4, 5});

    xticklabels({"2", "10", "100", "500", "1000", "5000"});
    yticklabels({"10", "100", "200", "500", "1000"});

    xlabel("N");
    ylabel("P");

    ostringstream title;
    title << method << " | eta = " << eta
        << " | couleur = R moyen";
    matplot::title(title.str());

    for (size_t row = 0; row < ps.size(); ++row)
    {
        for (size_t col = 0; col < ns.size(); ++col)
        {
            const auto& [it, r] =
                results.at({ns[col], ps[row]});

            const auto x = static_cast<double>(col + 1);
            const auto y = static_cast<double>(row + 1);

            ostringstream it_label;
            it_label << "IT=" << fixed
                << setprecision(1) << it;

            ostringstream r_label;
            r_label << "R=" << fixed
                << setprecision(3) << r;

            // heatmaps vertical exis is reversed
            // y - 0.15 above center
            auto top = text(x, y - 0.15, it_label.str());
            top->alignment(labels::center);
            top->font_size(9);

            auto bottom = text(x, y + 0.15, r_label.str());
            bottom->alignment(labels::center);
            bottom->font_size(9);
        }
    }

    xlim({0.5, 6.5});
    ylim({0.5, 5.5});

    hold(off);
}

using Times = map<pair<int, int>, double>;

void plot_time_panel(
    const Times& times,
    int position,
    const string& method,
    double eta,
    const vector<int>& ns,
    const vector<int>& ps,
    double max_milliseconds)
{
    vector colors(
        ps.size(), vector<double>(ns.size())
    );

    for (size_t row = 0; row < ps.size(); ++row)
        for (size_t col = 0; col < ns.size(); ++col)
            colors[row][col] = times.at({ns[col], ps[row]});

    subplot(1, 2, position);

    auto heatmap = matplot::heatmap(colors);
    heatmap->always_hide_labels(true);

    hold(on);

    colormap({
        {1.00, 0.98, 0.90},
        {1.00, 0.85, 0.65},
        {0.95, 0.60, 0.45}
    });

    caxis({0.0, max_milliseconds});
    colorbar();

    vector<double> x_ticks, y_ticks;
    vector<string> x_labels, y_labels;

    for (size_t col = 0; col < ns.size(); ++col)
    {
        x_ticks.push_back(static_cast<double>(col + 1));
        x_labels.push_back(to_string(ns[col]));
    }

    for (size_t row = 0; row < ps.size(); ++row)
    {
        y_ticks.push_back(static_cast<double>(row + 1));
        y_labels.push_back(to_string(ps[row]));
    }

    xticks(x_ticks);
    yticks(y_ticks);
    xticklabels(x_labels);
    yticklabels(y_labels);

    xlabel("N");
    ylabel("P");

    ostringstream title;
    title << method << " | eta = " << eta
        << " | temps en millisecondes";
    matplot::title(title.str());

    for (size_t row = 0; row < ps.size(); ++row)
    {
        for (size_t col = 0; col < ns.size(); ++col)
        {
            ostringstream label;
            label << fixed << setprecision(3)
                << colors[row][col] << " ms";

            auto text = matplot::text(
                static_cast<double>(col + 1),
                static_cast<double>(row + 1),
                label.str()
            );

            text->alignment(labels::center);
            text->font_size(9);
        }
    }

    xlim({0.5, static_cast<double>(ns.size()) + 0.5});
    ylim({0.5, static_cast<double>(ps.size()) + 0.5});

    hold(off);
}

void plot_results(
    const Results& batch_averages,
    const Results& online_averages,
    const double eta,
    const vector<int>& ns,
    const vector<int>& ps)
{
    const auto fig = figure(true);
    fig->width(1500);
    fig->height(650);

    plot_panel(batch_averages, 0, "Batch", eta, ns, ps);
    plot_panel(online_averages, 1, "Online", eta, ns, ps);

    filesystem::create_directories("plots");

    ostringstream filename;
    filename << "plots/resultats_eta_" << eta;

    fig->save(filename.str() + ".png");

    // to visualize in the app :
    // show();

    cout << "Printed heatmaps for ETA " << eta << endl;
}

void plot_time_results(
    const Times& batch_times,
    const Times& online_times,
    const double eta,
    const vector<int>& ns,
    const vector<int>& ps
)
{
    if (ns.size() < 2 || ps.size() < 2)
        throw invalid_argument(
            "Ces heatmaps demandent au moins deux valeurs par axe."
        );

    double max_milliseconds = 0.0;

    for (int n : ns)
    {
        for (int p : ps)
        {
            for (double t : {
                     batch_times.at({n, p}),
                     online_times.at({n, p})
                 })
            {
                if (!isfinite(t) || t < 0.0)
                    throw invalid_argument("Temps invalide.");

                max_milliseconds = std::max(max_milliseconds, t);
            }
        }
    }

    if (max_milliseconds == 0.0) max_milliseconds = 1.0;

    const auto fig = figure(true);
    fig->width(1500);
    fig->height(650);

    plot_time_panel(batch_times, 0, "Batch", eta, ns, ps, max_milliseconds);
    plot_time_panel(online_times, 1, "Online", eta, ns, ps, max_milliseconds);

    filesystem::create_directories("plots/time");

    ostringstream filename;
    filename << "plots/time/time_eta_" << eta;

    fig->save(filename.str() + ".png");
}

bool gpu_available()
{
#ifdef HAS_CUDA
    int nombre_gpu = 0;

    if (const auto status = cudaGetDeviceCount(&nombre_gpu);
        status != cudaSuccess)
    {
        cout << "CUDA unavailable : "
            << cudaGetErrorString(status) << endl;
        return false;
    }

    if (nombre_gpu == 0)
        return false;

    cudaDeviceProp proprietes{};

    if (const auto info_status = cudaGetDeviceProperties(&proprietes, 0);
        info_status != cudaSuccess)
    {
        cout << "Cannot read GPU : "
            << cudaGetErrorString(info_status) << endl;
        return false;
    }

    cout << "GPU detected : " << proprietes.name << endl;
    return true;
#else
    return false;
#endif
}

int main()
{
    const bool use_gpu = gpu_available();

    cout << "CUDA available : "
        << (use_gpu ? "oui" : "non")
        << endl;
#ifdef HAS_CUDA_KERNELS
    if (use_gpu)
    {
        const bool test_ok = test_gpu();

        cout << "Test d'execution GPU : "
            << (test_ok ? "OK" : "ECHEC")
            << endl;
    }
#endif

    // Exercice 0 avec le perceptron basique

    cout << "Question 0" << endl;

    // testing AND
    const map<vector<int>, int> input_and = {
        {{0, 0}, -1},
        {{0, 1}, -1},
        {{1, 0}, -1},
        {{1, 1}, 1},
    };

    Perceptron and_(2, 0.5, 0.0, input_and);
    and_.learn(Perceptron::BATCH);

    cout << and_.output({0, 1}) << endl;

    // testing OR
    const map<vector<int>, int> input_or = {
        {{0, 0}, -1},
        {{0, 1}, 1},
        {{1, 0}, 1},
        {{1, 1}, 1},
    };
    Perceptron or_(2, 0.5, 0.0, input_or);
    or_.learn(Perceptron::BATCH);
    cout << or_.output({0, 1}) << endl;

    //testing XOR
    const map<vector<int>, int> input_xor = {
        {{0, 0}, -1},
        {{0, 1}, 1},
        {{1, 0}, 1},
        {{1, 1}, -1},
    };
    Perceptron xor_(2, 0.5, 0.0, input_xor);
    xor_.learn(Perceptron::BATCH);
    cout << xor_.output({0, 0}) << endl;

    // Exercice 1 avec perceptron professeur
    // voir la fonction create_prof_dataset

    // Exercice 2
    // voir la fonction learning_all

    // Exercice 3

    for (vector<float> etas = {0.5, 0.25, 0.05}; float eta : etas)
    {
        vector ns = {2, 10, 100, 500, 1000, 5000};
        vector ps = {10, 100, 200, 500, 1000};

        constexpr int nb_tirages = 100;

        // compute averages
        // (n, p) -> (it, recouvrement)
        Results online_averages;
        Results batch_averages;
        map<pair<int, int>, double> online_time_results;
        map<pair<int, int>, double> batch_time_results;

        auto prof = Perceptron();

        for (auto n : ns)
        {
            for (auto p : ps)
            {
                const auto key = pair(n, p);
                auto& [it_o, reco_o] = online_averages[key];
                auto& [it_b, reco_b] = batch_averages[key];

                // failures to converge before max epoch reached
                int fail_o = 0;
                int fail_b = 0;

                for (int tirage = 0; tirage < nb_tirages; ++tirage)
                {
                    auto ds = create_prof_dataset(n, p, prof);

                    // ONLINE
                    auto start = chrono::steady_clock::now();
                    const auto online = learn_all(ds, n, prof, eta, Perceptron::ONLINE);
                    auto end = chrono::steady_clock::now();

                    online_time_results[key] +=
                        // convert to ms
                        chrono::duration<double, milli>(end - start).count() / nb_tirages;

                    for (const auto& r : online)
                    {
                        if (r.iterations == static_cast<size_t>(-1))
                        {
                            ++fail_o;
                            continue;
                        }

                        it_o += r.iterations;
                        reco_o += r.recouvrement;
                    }

                    // BATCH
                    start = chrono::steady_clock::now();
                    const auto batch = learn_all(ds, n, prof, eta, Perceptron::BATCH, true);
                    end = chrono::steady_clock::now();

                    batch_time_results[key] += chrono::duration<double, milli>(end - start).count() / nb_tirages;
                    for (const auto& r : batch)
                    {
                        if (r.iterations == static_cast<size_t>(-1))
                        {
                            ++fail_b;
                            continue;
                        }

                        it_b += r.iterations;
                        reco_b += r.recouvrement;
                    }
                }
                constexpr double total = nb_tirages * 20.0;

                const double succes_o = total - fail_o;
                const double succes_b = total - fail_b;

                // no node has converged
                constexpr double nan = numeric_limits<double>::quiet_NaN();

                it_o = succes_o > 0 ? it_o / succes_o : nan;
                reco_o = succes_o > 0 ? reco_o / succes_o : nan;

                it_b = succes_b > 0 ? it_b / succes_b : nan;
                reco_b = succes_b > 0 ? reco_b / succes_b : nan;

                cout << "eta=" << eta << " N=" << n << " P=" << p
                    << " | echecs online : " << fail_o << "/" << total
                    << " (" << 100.0 * fail_o / total << "%)"
                    << " | echecs batch : " << fail_b << "/" << total
                    << " (" << 100.0 * fail_b / total << "%)"
                    << endl;
            }
        }

        // generating plots
        plot_results(batch_averages, online_averages, eta, ns, ps); //iterations and recouvrement

        // running time
        plot_time_results(batch_time_results, online_time_results, eta, ns, ps);
    }


    return 0;
}

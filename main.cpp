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

using namespace std;
using namespace matplot;

int heaviside(const double x)
{
    return x > 0 ? 1 : 0;
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

    vector<double> get_weights()
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

    bool verify()
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

vector<Result> learn_all(const map<vector<int>, int>& dataset, const int n, Perceptron prof, const float eta,
                         Perceptron::LearningType type = Perceptron::ONLINE)
{
    vector<thread> threads;
    threads.reserve(20);

    vector<Result> results(20);
    vector<exception_ptr> errors(20);

    const auto prof_weights = prof.get_weights();

    for (int i = 0; i < 20; i++)
    {
        threads.emplace_back([&, i, n, eta, type]()
        {
            try
            {
                auto eleve = Perceptron(n + 1, eta, 0.0, dataset);
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
    vector<int> ns,
    vector<int> ps)
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

    std::filesystem::create_directories("plots");

    std::ostringstream filename;
    filename << "plots/resultats_eta_" << eta;

    fig->save(filename.str() + ".png");

    // to visualize in the app :
    // show();
}

int main()
{
    // Exercice 0 avec le perceptron basique

    cout << "Question 0" << endl;

    // testing AND
    const map<vector<int>, int> input_and = {
        {{0, 0}, 0},
        {{0, 1}, 0},
        {{1, 0}, 0},
        {{1, 1}, 1},
    };

    Perceptron and_(2, 0.5, 0.0, input_and);
    and_.learn(Perceptron::BATCH);

    cout << and_.output({0, 1}) << endl;

    // testing OR
    const map<vector<int>, int> input_or = {
        {{0, 0}, 0},
        {{0, 1}, 1},
        {{1, 0}, 1},
        {{1, 1}, 1},
    };
    Perceptron or_(2, 0.5, 0.0, input_or);
    or_.learn(Perceptron::BATCH);
    cout << or_.output({0, 1}) << endl;

    //testing XOR
    const map<vector<int>, int> input_xor = {
        {{0, 0}, 0},
        {{0, 1}, 1},
        {{1, 0}, 1},
        {{1, 1}, 0},
    };
    Perceptron xor_(2, 0.5, 0.0, input_xor);
    xor_.learn(Perceptron::BATCH);
    cout << xor_.output({0, 1}) << endl;

    // Exercice 1 avec perceptron professeur
    // voir la fonction create_prof_dataset

    // Exercice 2
    // voir la fonction learning_all

    // Exercice 3

    for (vector<float> etas = {0.5, 0.25, 0.05}; float eta : etas)
    {
        auto start = chrono::steady_clock::now();

        vector ns = {2, 10, 100, 500, 1000, 5000};
        vector ps = {10, 100, 200, 500, 1000};

        map<pair<int, int>, vector<Result>> online_results;
        map<pair<int, int>, vector<Result>> batch_results;

        auto prof = Perceptron();

        for (auto n : ns)
        {
            for (auto p : ps)
            {
                auto ds = create_prof_dataset(n, p, prof);

                // ONLINE
                online_results.emplace(pair(n, p), learn_all(ds, n, prof, eta, Perceptron::ONLINE));
                // BATCH
                batch_results.emplace(pair(n, p), learn_all(ds, n, prof, eta, Perceptron::BATCH));
            }
        }

        auto end = chrono::steady_clock::now();

        const double milliseconds = chrono::duration<double, milli>(end - start).count();

        cout << "Eta " << eta << " : " << milliseconds << " ms" << endl;

        // compute averages
        // (n, p) -> (it, recouvrement)
        map<pair<int, int>, pair<double, double>> online_averages;
        map<pair<int, int>, pair<double, double>> batch_averages;

        for (auto n : ns)
        {
            for (auto p : ps)
            {
                double reco_o = 0;
                double it_o = 0;
                double reco_b = 0;
                double it_b = 0;

                for (const auto& r : online_results[pair(n, p)])
                {
                    reco_o += r.recouvrement;
                    it_o += r.iterations;
                }

                online_averages.emplace(pair(n, p), pair(it_o / 20, reco_o / 20));

                for (const auto& r : batch_results[pair(n, p)])
                {
                    reco_b += r.recouvrement;
                    it_b += r.iterations;
                }

                batch_averages.emplace(pair(n, p), pair(it_b / 20, reco_b / 20));
            }
        }

        // generating plots
        plot_results(batch_averages, online_averages, eta, ns, ps);
    }


    return 0;
}

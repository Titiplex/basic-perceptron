#include <iostream>
#include <ranges>
#include <map>
#include <random>
#include <vector>
#include <stdexcept>
#include <thread>

using namespace std;

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
    bool learn_incremental()
    {
        for (size_t epoch = 0; epoch < 1000; epoch++)
        {
            if (verify()) return true;
            for (const auto& [fst, snd] : this->learn_data)
            {
                auto x = fst;
                const int y = this->output(fst);

                if (const int t = snd; y != t)
                {
                    vector<double> delta(this->weights.size());
                    for (const auto x_prime : x)
                    {
                        delta.push_back(this->eta * static_cast<float>(t - y) * static_cast<float>(x_prime));
                    }
                    this->weights = add(this->weights, delta);
                    this->biais += this->eta * static_cast<float>(t - y);
                }
            }
        }
        return verify();
    }

    bool learn_batch()
    {
        for (size_t epoch = 0; epoch < 1000; epoch++)
        {
            if (verify()) return true;
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
        return verify();
    }

public:
    bool learn(const LearningType type_)
    {
        switch (type_)
        {
        case BATCH:
            return learn_batch();
        default:
            return learn_incremental();
        }
    }
};

map<vector<int>, int> create_prof_dataset(const int n, const int p)
{
    map<vector<int>, int> ls_dataset = {};

    random_device rd;
    mt19937 gen(rd());

    const auto prof = Perceptron(n + 1, 0.5, 0.0);

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

vector<Perceptron> learn_all(const map<vector<int>, int>& dataset, const int n, Perceptron::LearningType type)
{
    vector<thread> threads;
    threads.reserve(20);

    vector<Perceptron> results;

    for (int i = 0; i < 20; i++)
    {
        threads.emplace_back([&, type]()
        {
            auto eleve = Perceptron(n, 0.5, 0.0, dataset);
            eleve.learn(type);
            results.push_back(eleve);
        });
    }

    for (thread& t : threads) t.join();

    return results;
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


    return 0;
}

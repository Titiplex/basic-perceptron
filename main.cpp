#include <iostream>
#include <ranges>
#include <map>
#include <vector>
#include <stdexcept>

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
    Perceptron(const vector<double>& weights, const float eta, double biais,
               const map<vector<int>, int>& learn_data) :
        learn_data(learn_data)
    {
        this->weights = weights;
        this->eta = eta;
        this->biais = biais;
    }

    Perceptron(int n, const float eta, double biais,
               const map<vector<int>, int>& learn_data) :
        learn_data(learn_data)
    {
        const vector init(n, 0.0);
        this->weights = init;
        this->eta = eta;
        this->biais = biais;
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

    bool learn_incremental()
    {
        for (size_t epoch = 0; epoch < 1000; epoch++)
        {
            if (verify()) return true;
            for (const auto& [fst, snd] : this->learn_data)
            {
                auto x = fst;
                const int y = this->output(fst);
                const int t = snd;

                if (y != t)
                {
                    vector delta(this->weights.size(), 0.0);
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
                const int t = snd;

                if (y != t)
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
};

int main()
{
    // testing AND
    map<vector<int>, int> input_and = {
        {{0, 0}, 0},
        {{0, 1}, 0},
        {{1, 0}, 0},
        {{1, 1}, 1},
    };

    Perceptron and_(2, 0.5, 0.0, input_and);
    and_.learn_batch();

    cout << and_.output({0, 1}) << endl;

    // testing OR
    map<vector<int>, int> input_or = {
        {{0, 0}, 0},
        {{0, 1}, 1},
        {{1, 0}, 1},
        {{1, 1}, 1},
    };
    Perceptron or_(2, 0.5, 0.0, input_or);
    or_.learn_batch();
    cout << or_.output({0, 1}) << endl;

    //testing XOR
    map<vector<int>, int> input_xor = {
        {{0, 0}, 0},
        {{0, 1}, 1},
        {{1, 0}, 1},
        {{1, 1}, 0},
    };
    Perceptron xor_(2, 0.5, 0.0, input_xor);
    xor_.learn_batch();
    cout << xor_.output({0, 1}) << endl;

    return 0;
}

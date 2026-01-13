#include "generar_individuo_copy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int POP_SIZE = 286;
constexpr int GENERATIONS = 300;
constexpr int M = 4;
constexpr int NEIGHBOR_SIZE = 15;
constexpr double P_CROSS = 0.6;
constexpr double P_MUT_EXT = 0.4;
constexpr double P_MUT_CONTR = 0.4;
constexpr double P_MUT_REPL = 0.4;
constexpr double THETA_PBI = 5.0;
constexpr double MUT_PROB_MIN = 0.3;
constexpr double MUT_PROB_MAX = 0.8;
constexpr int LOG_INTERVAL = 1;
constexpr int HV_SAMPLES = 20000;
constexpr std::array<double, M> HV_REF_POINT = {2.0, 2.0, 2.0, 2.0};
constexpr std::array<const char*, M> METRIC_NAMES = {
    "jaccard",
    "cosine",
    "phi",
    "kappa"
};

using Obj = std::array<double, M>;

struct Individual {
    moead::Chromosome c1;
    moead::Chromosome c2;
    Obj obj;
    Obj f;
    Obj metrics;
};

struct RuleKey {
    moead::Chromosome c1;
    moead::Chromosome c2;

    bool operator==(const RuleKey& other) const {
        return c1 == other.c1 && c2 == other.c2;
    }
};

struct RuleKeyHash {
    std::size_t operator()(const RuleKey& key) const {
        std::size_t seed = 0;
        std::hash<int> hasher;
        for (int v : key.c1) {
            seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        for (int v : key.c2) {
            seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

static double rand_real(std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

static int rand_int(int min_val, int max_val, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(min_val, max_val);
    return dist(rng);
}

static std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped.push_back(ch);
        }
    }
    escaped += "\"";
    return escaped;
}

static std::vector<std::vector<int>> construir_vecindarios(
    const std::vector<std::vector<double>>& pesos,
    int k) {
    std::size_t n = pesos.size();
    std::vector<std::vector<int>> vecinos(n);

    for (std::size_t i = 0; i < n; ++i) {
        std::vector<std::pair<double, int>> dist;
        dist.reserve(n);
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t d = 0; d < pesos[i].size(); ++d) {
                double diff = pesos[i][d] - pesos[j][d];
                sum += diff * diff;
            }
            dist.emplace_back(std::sqrt(sum), static_cast<int>(j));
        }
        std::sort(dist.begin(), dist.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) {
                return a.first < b.first;
            }
            return a.second < b.second;
        });
        int take = std::min(static_cast<int>(n), k);
        vecinos[i].reserve(take);
        for (int t = 0; t < take; ++t) {
            vecinos[i].push_back(dist[t].second);
        }
    }

    return vecinos;
}

static Obj evaluar_individuo(
    const moead::Chromosome& crom1,
    const moead::Chromosome& crom2,
    const moead::DataSet& df,
    std::unordered_map<RuleKey, Obj, RuleKeyHash>& eval_cache) {
    RuleKey key{crom1, crom2};
    auto it = eval_cache.find(key);
    if (it != eval_cache.end()) {
        return it->second;
    }
    int a_c, a_no_c, no_a_c, no_a_no_c;
    std::tie(a_c, a_no_c, no_a_c, no_a_no_c) = moead::contar_casos_regla(df, crom1, crom2);

    Obj result = {
        moead::jaccard(a_c, a_no_c, no_a_c, no_a_no_c),
        moead::cosine(a_c, a_no_c, no_a_c, no_a_no_c),
        moead::phi(a_c, a_no_c, no_a_c, no_a_no_c),
        moead::kappa(a_c, a_no_c, no_a_c, no_a_no_c)
    };
    eval_cache.emplace(std::move(key), result);
    return result;
}

static Obj obj_from_metrics(const Obj& metrics) {
    return metrics;
}

static double sum_obj(const Obj& obj) {
    double sum = 0.0;
    for (double v : obj) {
        sum += v;
    }
    return sum;
}

static std::pair<std::vector<std::pair<moead::Chromosome, moead::Chromosome>>, std::vector<Obj>>
generar_poblacion_valida(
    const moead::DataSet& df,
    int pop_size,
    std::mt19937& rng,
    std::unordered_map<RuleKey, Obj, RuleKeyHash>& eval_cache) {
    std::vector<std::pair<moead::Chromosome, moead::Chromosome>> poblacion;
    std::vector<Obj> objs;
    poblacion.reserve(pop_size);
    objs.reserve(pop_size);

    while (static_cast<int>(poblacion.size()) < pop_size) {
        auto individuo = moead::generar_individuo(moead::DEFAULT_ALLELES_CROM2.size(), rng);
        auto metrics = evaluar_individuo(individuo.first, individuo.second, df, eval_cache);
        auto vals = obj_from_metrics(metrics);
        poblacion.push_back(std::move(individuo));
        objs.push_back(vals);
    }

    return {poblacion, objs};
}

static std::pair<std::vector<int>, std::vector<Obj>>
asignar_soluciones_a_pesos(
    const std::vector<std::vector<double>>& pesos,
    const std::vector<std::pair<moead::Chromosome, moead::Chromosome>>& poblacion,
    const moead::DataSet& df,
    std::mt19937& rng,
    std::unordered_map<RuleKey, Obj, RuleKeyHash>& eval_cache) {
    std::vector<Obj> objs;
    objs.reserve(poblacion.size());
    for (const auto& ind : poblacion) {
        auto metrics = evaluar_individuo(ind.first, ind.second, df, eval_cache);
        objs.push_back(obj_from_metrics(metrics));
    }

    int n_w = static_cast<int>(pesos.size());
    int n_p = static_cast<int>(poblacion.size());
    std::vector<int> asignacion;
    asignacion.reserve(n_w);

    if (n_p >= n_w) {
        std::vector<int> indices(n_p);
        for (int i = 0; i < n_p; ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), rng);
        asignacion.assign(indices.begin(), indices.begin() + n_w);
    } else {
        for (int i = 0; i < n_w; ++i) {
            asignacion.push_back(rand_int(0, n_p - 1, rng));
        }
    }

    return {asignacion, objs};
}

static std::pair<std::string, std::string> decode_rule(
    const moead::Chromosome& c1,
    const moead::Chromosome& c2) {
    std::vector<std::string> ante;
    std::vector<std::string> cons;

    for (std::size_t idx = 0; idx < c1.size(); ++idx) {
        int g1 = c1[idx];
        int g2 = c2[idx];
        if (g1 == 0) {
            continue;
        }
        auto it = moead::VALUE_MAP[idx].find(g2);
        if (it == moead::VALUE_MAP[idx].end()) {
            continue;
        }
        std::string literal = moead::COLUMN_ORDER[idx] + "=" + it->second;
        if (g1 == 1) {
            ante.push_back(literal);
        } else {
            cons.push_back(literal);
        }
    }

    auto join = [](const std::vector<std::string>& parts) {
        if (parts.empty()) {
            return std::string("(vacio)");
        }
        std::ostringstream oss;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            if (i > 0) {
                oss << " ^ ";
            }
            oss << parts[i];
        }
        return oss.str();
    };

    return {join(ante), join(cons)};
}

static bool dominates(const Obj& a, const Obj& b) {
    bool strictly = false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] > b[i]) {
            return false;
        }
        if (a[i] < b[i]) {
            strictly = true;
        }
    }
    return strictly;
}

static std::vector<int> frente_pareto_indices(const std::vector<Obj>& objs) {
    std::vector<int> indices;
    int n = static_cast<int>(objs.size());
    for (int i = 0; i < n; ++i) {
        bool dominated = false;
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            if (dominates(objs[j], objs[i])) {
                dominated = true;
                break;
            }
        }
        if (!dominated) {
            indices.push_back(i);
        }
    }
    return indices;
}

static double hipervolumen(const std::vector<Obj>& objs, const Obj& ref_point) {
    if (objs.empty()) {
        return 0.0;
    }

    Obj min_vals = objs[0];
    Obj max_vals = objs[0];
    for (const auto& obj : objs) {
        for (std::size_t i = 0; i < M; ++i) {
            min_vals[i] = std::min(min_vals[i], obj[i]);
            max_vals[i] = std::max(max_vals[i], obj[i]);
        }
    }

    Obj ref;
    for (std::size_t i = 0; i < M; ++i) {
        ref[i] = std::max(ref_point[i], max_vals[i] + 1e-9);
    }

    double volume = 1.0;
    for (std::size_t i = 0; i < M; ++i) {
        double span = ref[i] - min_vals[i];
        if (span <= 0.0) {
            return 0.0;
        }
        volume *= span;
    }

    std::mt19937 rng(12345);
    std::vector<std::uniform_real_distribution<double>> dists;
    dists.reserve(M);
    for (std::size_t i = 0; i < M; ++i) {
        dists.emplace_back(min_vals[i], ref[i]);
    }

    int dominated_count = 0;
    Obj sample{};
    for (int s = 0; s < HV_SAMPLES; ++s) {
        for (std::size_t i = 0; i < M; ++i) {
            sample[i] = dists[i](rng);
        }
        bool dominated = false;
        for (const auto& obj : objs) {
            bool dominates_sample = true;
            for (std::size_t i = 0; i < M; ++i) {
                if (obj[i] > sample[i]) {
                    dominates_sample = false;
                    break;
                }
            }
            if (dominates_sample) {
                dominated = true;
                break;
            }
        }
        if (dominated) {
            dominated_count += 1;
        }
    }

    return volume * (static_cast<double>(dominated_count) / static_cast<double>(HV_SAMPLES));
}

struct Stats {
    double hv = 0.0;
    Obj min_vals = {};
    Obj max_vals = {};
    Obj mean_vals = {};
    Obj std_vals = {};
};

static Stats resumen_estadisticas(
    const std::vector<Obj>& objs,
    const Obj& hv_ref) {
    Stats stats;
    if (objs.empty()) {
        return stats;
    }

    stats.min_vals = objs[0];
    stats.max_vals = objs[0];
    for (const auto& obj : objs) {
        for (std::size_t i = 0; i < M; ++i) {
            stats.min_vals[i] = std::min(stats.min_vals[i], obj[i]);
            stats.max_vals[i] = std::max(stats.max_vals[i], obj[i]);
            stats.mean_vals[i] += obj[i];
        }
    }

    for (std::size_t i = 0; i < M; ++i) {
        stats.mean_vals[i] /= static_cast<double>(objs.size());
    }

    for (const auto& obj : objs) {
        for (std::size_t i = 0; i < M; ++i) {
            double diff = obj[i] - stats.mean_vals[i];
            stats.std_vals[i] += diff * diff;
        }
    }

    for (std::size_t i = 0; i < M; ++i) {
        stats.std_vals[i] = std::sqrt(stats.std_vals[i] / static_cast<double>(objs.size()));
    }

    stats.hv = hipervolumen(objs, hv_ref);
    return stats;
}

static void ajustar_probabilidades(
    const std::unordered_map<std::string, int>& mut_successes,
    std::unordered_map<std::string, double>& mut_probs) {
    int total = 0;
    for (const auto& kv : mut_successes) {
        total += kv.second;
    }
    if (total == 0) {
        return;
    }
    double avg = total / static_cast<double>(mut_successes.size());
    double step = 0.05;
    for (auto& kv : mut_probs) {
        auto it = mut_successes.find(kv.first);
        if (it == mut_successes.end()) {
            continue;
        }
        if (it->second < avg) {
            kv.second = std::min(MUT_PROB_MAX, kv.second + step);
        } else if (it->second > avg) {
            kv.second = std::max(MUT_PROB_MIN, kv.second - step);
        }
    }
}

static double scalarizacion_pbi(
    const Obj& f,
    const std::vector<double>& w,
    const Obj& z,
    double theta) {
    double norm_w = 0.0;
    for (double v : w) {
        norm_w += v * v;
    }
    norm_w = std::sqrt(norm_w);
    if (norm_w == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    Obj w_unit{};
    for (std::size_t i = 0; i < M; ++i) {
        w_unit[i] = w[i] / norm_w;
    }

    double d1 = 0.0;
    for (std::size_t i = 0; i < M; ++i) {
        d1 += (f[i] - z[i]) * w_unit[i];
    }
    d1 = std::abs(d1);

    Obj proj{};
    for (std::size_t i = 0; i < M; ++i) {
        proj[i] = z[i] + d1 * w_unit[i];
    }

    double d2 = 0.0;
    for (std::size_t i = 0; i < M; ++i) {
        double diff = f[i] - proj[i];
        d2 += diff * diff;
    }
    d2 = std::sqrt(d2);

    return d1 + (theta * d2);
}

static std::pair<int, int> seleccionar_padres(
    int idx,
    const std::vector<std::vector<int>>& vecinos,
    int pop_size,
    std::mt19937& rng,
    double prob_vecindario = 0.9) {
    const auto& vec = vecinos[idx];
    std::vector<int> fuera;
    fuera.reserve(pop_size);
    for (int j = 0; j < pop_size; ++j) {
        if (std::find(vec.begin(), vec.end(), j) == vec.end()) {
            fuera.push_back(j);
        }
    }

    const std::vector<int>& pool = (rand_real(rng) < prob_vecindario || fuera.empty()) ? vec : fuera;
    if (pool.size() == 1) {
        return {pool[0], pool[0]};
    }

    int i1 = rand_int(0, static_cast<int>(pool.size() - 1), rng);
    int i2 = i1;
    while (i2 == i1) {
        i2 = rand_int(0, static_cast<int>(pool.size() - 1), rng);
    }
    return {pool[i1], pool[i2]};
}

static std::vector<std::tuple<moead::Chromosome, moead::Chromosome, Obj>>
generar_hijos(
    const Individual& p1,
    const Individual& p2,
    const moead::DataSet& df,
    std::unordered_map<std::string, double>& mut_probs,
    std::unordered_map<std::string, int>& mut_successes,
    std::unordered_map<std::string, int>& mut_attempts,
    std::unordered_map<RuleKey, Obj, RuleKeyHash>& eval_cache,
    std::mt19937& rng) {
    moead::Chromosome c1a = p1.c1;
    moead::Chromosome c2a = p1.c2;
    moead::Chromosome c1b = p2.c1;
    moead::Chromosome c2b = p2.c2;

    moead::Chromosome h1_c1;
    moead::Chromosome h1_c2;
    moead::Chromosome h2_c1;
    moead::Chromosome h2_c2;

    if (rand_real(rng) < P_CROSS) {
        auto hijos = moead::recombinacion_n_puntos(c1a, c2a, c1b, c2b, 4, rng);
        h1_c1 = std::move(hijos.first.first);
        h1_c2 = std::move(hijos.first.second);
        h2_c1 = std::move(hijos.second.first);
        h2_c2 = std::move(hijos.second.second);
    } else {
        h1_c1 = c1a;
        h1_c2 = c2a;
        h2_c1 = c1b;
        h2_c2 = c2b;
    }

    std::vector<std::pair<moead::Chromosome, moead::Chromosome>> hijos = {
        {h1_c1, h1_c2},
        {h2_c1, h2_c2}
    };

    std::vector<std::tuple<moead::Chromosome, moead::Chromosome, Obj>> hijos_mutados;
    for (auto& child : hijos) {
        auto c1 = child.first;
        auto c2 = child.second;
        auto metrics_actual = evaluar_individuo(c1, c2, df, eval_cache);
        auto obj_actual = obj_from_metrics(metrics_actual);
        double score_actual = sum_obj(obj_actual);

        if (rand_real(rng) < mut_probs["ext"]) {
            mut_attempts["ext"] += 1;
            auto mut = moead::mutacion_extension(c1, c2, rng);
            auto metrics_mut = evaluar_individuo(mut.first, mut.second, df, eval_cache);
            auto obj_mut = obj_from_metrics(metrics_mut);
            double score_mut = sum_obj(obj_mut);
            if (score_mut < score_actual) {
                mut_successes["ext"] += 1;
                c1 = std::move(mut.first);
                c2 = std::move(mut.second);
                obj_actual = obj_mut;
                metrics_actual = metrics_mut;
                score_actual = score_mut;
            }
        }

        if (rand_real(rng) < mut_probs["contr"]) {
            mut_attempts["contr"] += 1;
            auto mut = moead::mutacion_contraccion(c1, c2, rng);
            auto metrics_mut = evaluar_individuo(mut.first, mut.second, df, eval_cache);
            auto obj_mut = obj_from_metrics(metrics_mut);
            double score_mut = sum_obj(obj_mut);
            if (score_mut < score_actual) {
                mut_successes["contr"] += 1;
                c1 = std::move(mut.first);
                c2 = std::move(mut.second);
                obj_actual = obj_mut;
                metrics_actual = metrics_mut;
                score_actual = score_mut;
            }
        }

        if (rand_real(rng) < mut_probs["repl"]) {
            mut_attempts["repl"] += 1;
            auto mut = moead::mutacion_reemplazo(c1, c2, rng);
            auto metrics_mut = evaluar_individuo(mut.first, mut.second, df, eval_cache);
            auto obj_mut = obj_from_metrics(metrics_mut);
            double score_mut = sum_obj(obj_mut);
            if (score_mut < score_actual) {
                mut_successes["repl"] += 1;
                c1 = std::move(mut.first);
                c2 = std::move(mut.second);
                obj_actual = obj_mut;
                metrics_actual = metrics_mut;
                score_actual = score_mut;
            }
        }

        hijos_mutados.emplace_back(c1, c2, metrics_actual);
    }

    return hijos_mutados;
}

static void guardar_reglas(const std::vector<Individual>& pop, const std::string& path) {
    std::ofstream out(path);
    out << "antecedente,consecuente,jaccard,cosine,phi,kappa\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& ind : pop) {
        auto decoded = decode_rule(ind.c1, ind.c2);
        out << csv_escape(decoded.first) << ","
            << csv_escape(decoded.second) << ","
            << ind.metrics[0] << ","
            << ind.metrics[1] << ","
            << ind.metrics[2] << ","
            << ind.metrics[3] << "\n";
    }
}

static void guardar_invalid_children(
    const std::unordered_map<RuleKey, int, RuleKeyHash>& invalid_counts,
    const std::string& path) {
    std::ofstream out(path);
    out << "antecedente,consecuente,contador\n";
    for (const auto& kv : invalid_counts) {
        auto decoded = decode_rule(kv.first.c1, kv.first.c2);
        out << csv_escape(decoded.first) << ","
            << csv_escape(decoded.second) << ","
            << kv.second << "\n";
    }
}

static void guardar_logs(
    const std::vector<std::pair<int, Stats>>& logs,
    const std::string& path) {
    std::ofstream out(path);
    out << "hv";
    for (const auto& name : METRIC_NAMES) {
        out << ",min_" << name
            << ",max_" << name
            << ",mean_" << name
            << ",std_" << name;
    }
    out << ",generacion\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& entry : logs) {
        int gen = entry.first;
        const auto& stats = entry.second;
        out << stats.hv;
        for (std::size_t i = 0; i < M; ++i) {
            out << "," << stats.min_vals[i]
                << "," << stats.max_vals[i]
                << "," << stats.mean_vals[i]
                << "," << stats.std_vals[i];
        }
        out << "," << gen << "\n";
    }
}

static void guardar_mutaciones(
    const std::vector<std::unordered_map<std::string, double>>& mutation_log,
    const std::string& path) {
    std::ofstream out(path);
    out << "generacion,exitos_ext,exitos_contr,exitos_repl,prob_ext,prob_contr,prob_repl\n";
    out << std::fixed << std::setprecision(6);
    int gen = 1;
    for (const auto& row : mutation_log) {
        out << gen << ","
            << static_cast<int>(row.at("exitos_ext")) << ","
            << static_cast<int>(row.at("exitos_contr")) << ","
            << static_cast<int>(row.at("exitos_repl")) << ","
            << row.at("prob_ext") << ","
            << row.at("prob_contr") << ","
            << row.at("prob_repl") << "\n";
        gen += LOG_INTERVAL;
    }
}

static std::tuple<std::vector<Individual>, Obj,
                  std::unordered_map<RuleKey, int, RuleKeyHash>,
                  std::vector<std::pair<int, Stats>>,
                  std::vector<std::unordered_map<std::string, double>>>
moead_run(
    const moead::DataSet& df,
    int seed,
    const std::vector<std::vector<double>>& weights,
    const std::vector<std::vector<int>>& vecindarios) {
    std::mt19937 rng(seed);
    std::unordered_map<RuleKey, Obj, RuleKeyHash> eval_cache;

    auto init_pop = generar_poblacion_valida(df, POP_SIZE, rng, eval_cache);
    auto asignacion = asignar_soluciones_a_pesos(weights, init_pop.first, df, rng, eval_cache);

    std::vector<Individual> pop;
    pop.reserve(asignacion.first.size());
    for (int idx : asignacion.first) {
        const auto& indiv = init_pop.first[idx];
        auto metrics = evaluar_individuo(indiv.first, indiv.second, df, eval_cache);
        auto obj = obj_from_metrics(metrics);
        pop.push_back({indiv.first, indiv.second, obj, obj, metrics});
    }

    Obj ideal = pop.front().f;
    for (const auto& ind : pop) {
        for (std::size_t i = 0; i < M; ++i) {
            ideal[i] = std::min(ideal[i], ind.f[i]);
        }
    }

    std::unordered_map<RuleKey, int, RuleKeyHash> invalid_counts;
    std::vector<std::pair<int, Stats>> log_rows;

    std::unordered_map<std::string, double> mut_probs = {
        {"ext", std::min(MUT_PROB_MAX, std::max(MUT_PROB_MIN, P_MUT_EXT))},
        {"contr", std::min(MUT_PROB_MAX, std::max(MUT_PROB_MIN, P_MUT_CONTR))},
        {"repl", std::min(MUT_PROB_MAX, std::max(MUT_PROB_MIN, P_MUT_REPL))}
    };

    std::unordered_map<std::string, int> mut_successes = {
        {"ext", 0},
        {"contr", 0},
        {"repl", 0}
    };

    std::unordered_map<std::string, int> mut_attempts = {
        {"ext", 0},
        {"contr", 0},
        {"repl", 0}
    };

    std::vector<std::unordered_map<std::string, double>> mutation_log;

    for (int gen = 0; gen < GENERATIONS; ++gen) {
        if (gen % 10 == 0) {
            std::cout << gen << "\n";
        }
        for (std::size_t i = 0; i < pop.size(); ++i) {
            auto parents = seleccionar_padres(static_cast<int>(i), vecindarios, static_cast<int>(pop.size()), rng);
            auto hijos = generar_hijos(
                pop[parents.first],
                pop[parents.second],
                df,
                mut_probs,
                mut_successes,
                mut_attempts,
                eval_cache,
                rng);

            std::vector<std::tuple<moead::Chromosome, moead::Chromosome, Obj>> valid_children;
            for (const auto& hijo : hijos) {
                const auto& metrics_h = std::get<2>(hijo);
                auto obj_h = obj_from_metrics(metrics_h);
                bool invalid = false;
                for (double v : obj_h) {
                    if (v == 3.0) {
                        invalid = true;
                        break;
                    }
                }
                if (invalid) {
                    RuleKey key{std::get<0>(hijo), std::get<1>(hijo)};
                    invalid_counts[key] += 1;
                    continue;
                }
                valid_children.push_back(hijo);
            }

            if (valid_children.empty()) {
                continue;
            }

            for (const auto& hijo : valid_children) {
                const auto& hijo_c1 = std::get<0>(hijo);
                const auto& hijo_c2 = std::get<1>(hijo);
                const auto& metrics_hijo = std::get<2>(hijo);
                auto obj_hijo = obj_from_metrics(metrics_hijo);
                auto f_hijo = obj_hijo;

                for (std::size_t k = 0; k < M; ++k) {
                    ideal[k] = std::min(ideal[k], f_hijo[k]);
                }

                for (int j : vecindarios[i]) {
                    double g_old = scalarizacion_pbi(pop[j].f, weights[j], ideal, THETA_PBI);
                    double g_new = scalarizacion_pbi(f_hijo, weights[j], ideal, THETA_PBI);
                    if (g_new <= g_old) {
                        pop[j] = {hijo_c1, hijo_c2, obj_hijo, f_hijo, metrics_hijo};
                    }
                }
            }
        }

        ajustar_probabilidades(mut_successes, mut_probs);
        if ((gen + 1) % LOG_INTERVAL == 0 || gen == GENERATIONS - 1) {
            std::vector<Obj> objs_cur;
            objs_cur.reserve(pop.size());
            for (const auto& ind : pop) {
                objs_cur.push_back(ind.obj);
            }

            auto pareto_idx = frente_pareto_indices(objs_cur);
            std::vector<Obj> pareto_objs;
            pareto_objs.reserve(pareto_idx.size());
            for (int idx : pareto_idx) {
                pareto_objs.push_back(objs_cur[idx]);
            }

            auto stats = resumen_estadisticas(pareto_objs, HV_REF_POINT);
            log_rows.emplace_back(gen + 1, stats);

            std::unordered_map<std::string, double> row;
            row["exitos_ext"] = static_cast<double>(mut_successes["ext"]);
            row["exitos_contr"] = static_cast<double>(mut_successes["contr"]);
            row["exitos_repl"] = static_cast<double>(mut_successes["repl"]);
            row["prob_ext"] = mut_probs["ext"];
            row["prob_contr"] = mut_probs["contr"];
            row["prob_repl"] = mut_probs["repl"];
            mutation_log.push_back(std::move(row));
        }
    }

    return {pop, ideal, invalid_counts, log_rows, mutation_log};
}

}  

int main() {
    try {
        auto df = moead::read_csv("discretized_dataset_2.csv");
        std::vector<int> seeds = {2,3,5,7,11,13,17,19,23,29};

        auto weights = moead::generar_pesos_das_dennis(M, POP_SIZE);
        auto vecindarios = construir_vecindarios(weights, NEIGHBOR_SIZE);

        struct RunInfo {
            int seed;
            double hv;
            std::string reglas_path;
        };

        std::vector<RunInfo> runs_info;

        for (int seed : seeds) {
            std::cout << "--- Semilla " << seed << " ---\n";
            auto result = moead_run(df, seed, weights, vecindarios);
            auto& pop_final = std::get<0>(result);
            auto& ideal_final = std::get<1>(result);
            auto& invalid_counts = std::get<2>(result);
            auto& logs = std::get<3>(result);
            auto& mutation_log = std::get<4>(result);

            std::cout << "Ideal final (objetivos minimizados): ";
            for (std::size_t i = 0; i < M; ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << ideal_final[i];
            }
            std::cout << "\n";
            std::cout << "Primer individuo final obj: ";
            for (std::size_t i = 0; i < M; ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << pop_final.front().obj[i];
            }
            std::cout << "\n";

            if (!invalid_counts.empty()) {
                std::string invalid_path = "invalid_children_seed" + std::to_string(seed) + ".csv";
                guardar_invalid_children(invalid_counts, invalid_path);
                std::cout << "Reglas invalidas guardadas en " << invalid_path << "\n";
            }

            std::vector<Obj> objs_final;
            objs_final.reserve(pop_final.size());
            for (const auto& ind : pop_final) {
                objs_final.push_back(ind.obj);
            }
            auto pareto_idx = frente_pareto_indices(objs_final);
            std::vector<Obj> pareto_objs;
            pareto_objs.reserve(pareto_idx.size());
            for (int idx : pareto_idx) {
                pareto_objs.push_back(objs_final[idx]);
            }
            double hv_val = hipervolumen(pareto_objs, HV_REF_POINT);
            std::cout << "Hipervolumen (ref [";
            for (std::size_t i = 0; i < M; ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << HV_REF_POINT[i];
            }
            std::cout << "]): " << hv_val << "\n";

            if (!logs.empty()) {
                std::string log_path = "logs_seed" + std::to_string(seed) + ".csv";
                guardar_logs(logs, log_path);
            }

            std::string reglas_path = "reglas_seed" + std::to_string(seed) + ".csv";
            guardar_reglas(pop_final, reglas_path);

            if (!mutation_log.empty()) {
                std::string mut_path = "mutations_seed" + std::to_string(seed) + ".csv";
                guardar_mutaciones(mutation_log, mut_path);
                std::cout << "Mutaciones (exitos y probabilidades) guardadas en " << mut_path << "\n";
            }

            runs_info.push_back({seed, hv_val, reglas_path});
        }

        if (!runs_info.empty()) {
            std::sort(runs_info.begin(), runs_info.end(), [](const RunInfo& a, const RunInfo& b) {
                return a.hv < b.hv;
            });
            std::size_t med_idx = runs_info.size() / 2;
            const auto& med_run = runs_info[med_idx];
            std::cout << "Semilla mediana por HV: " << med_run.seed
                      << " (hv=" << med_run.hv << ")\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    return 0;
}

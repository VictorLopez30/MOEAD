#include "generar_individuo.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace moead {

const std::vector<std::vector<int>> DEFAULT_ALLELES_CROM2 = {
    {1, 2, 3, 4, 5},  // freq ceros
    {1, 2, 3, 4, 5},  // freq unos
    {1, 2, 3, 4, 5},  // freq dos
    {1, 2, 3, 4, 5},  // width
    {1, 2},           // pasos
    {1, 2},           // simetria
    {1, 2, 3, 4, 5},  // actividad
    {1, 2, 3, 4, 5},  // entropia
    {1, 2},           // transitoriedad
    {1, 2, 3, 4, 5},  // sensitividad
    {1, 2, 3, 4, 5}   // time entropy
};

const std::vector<std::unordered_map<int, std::string>> VALUE_MAP = {
    {{1, "1"}, {2, "2"}, {3, "3"}, {4, "4"}, {5, "5"}},
    {{1, "1"}, {2, "2"}, {3, "3"}, {4, "4"}, {5, "5"}},
    {{1, "1"}, {2, "2"}, {3, "3"}, {4, "4"}, {5, "5"}},
    {{1, "1"}, {2, "2"}, {3, "3"}, {4, "4"}, {5, "5"}},
    {{1, "1"}, {2, "2"}},
    {{1, "1"}, {2, "2"}},
    {{1, "1"}, {2, "2"}, {3, "3"}, {4, "4"}, {5, "5"}},
    {{1, "1"}, {2, "2"}, {3, "3"}, {4, "4"}, {5, "5"}},
    {{1, "1"}, {2, "2"}},
    {{1, "1"}, {2, "2"}, {3, "3"}, {4, "4"}, {5, "5"}},
    {{1, "1"}, {2, "2"}, {3, "3"}, {4, "4"}, {5, "5"}}
};

const std::vector<std::string> COLUMN_ORDER = {
    "freq ceros",
    "freq unos",
    "freq dos",
    "width",
    "pasos",
    "simetria",
    "actividad",
    "entropia",
    "transitoriedad",
    "sensitividad",
    "time entropy"
};

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        char ch = line[i];
        if (ch == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (ch == ',' && !in_quotes) {
            fields.push_back(field);
            field.clear();
        } else {
            field.push_back(ch);
        }
    }
    fields.push_back(field);
    return fields;
}

DataSet read_csv(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("No se pudo abrir el archivo: " + path);
    }

    DataSet df;
    std::string line;
    if (!std::getline(input, line)) {
        return df;
    }

    df.columns = split_csv_line(line);
    std::unordered_map<std::string, std::size_t> column_index;
    for (std::size_t i = 0; i < df.columns.size(); ++i) {
        column_index[df.columns[i]] = i;
    }

    std::vector<std::size_t> order_indices;
    order_indices.reserve(COLUMN_ORDER.size());
    for (const auto& col : COLUMN_ORDER) {
        auto it = column_index.find(col);
        if (it == column_index.end()) {
            throw std::runtime_error("Columna faltante en CSV: " + col);
        }
        order_indices.push_back(it->second);
    }

    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        auto fields = split_csv_line(line);
        if (fields.size() < df.columns.size()) {
            fields.resize(df.columns.size());
        }
        std::vector<std::string> ordered;
        ordered.reserve(COLUMN_ORDER.size());
        for (auto idx : order_indices) {
            ordered.push_back(fields[idx]);
        }
        df.rows.push_back(std::move(ordered));
    }

    df.row_count = df.rows.size();
    df.value_masks.assign(COLUMN_ORDER.size(), {});
    if (df.row_count == 0) {
        df.full_mask.clear();
        df.tail_mask = 0;
        return df;
    }

    std::size_t words = (df.row_count + 63) / 64;
    df.full_mask.assign(words, ~static_cast<std::uint64_t>(0));
    std::size_t rem = df.row_count % 64;
    if (rem == 0) {
        df.tail_mask = ~static_cast<std::uint64_t>(0);
    } else {
        df.tail_mask = (static_cast<std::uint64_t>(1) << rem) - 1;
        df.full_mask.back() = df.tail_mask;
    }

    for (std::size_t row_idx = 0; row_idx < df.rows.size(); ++row_idx) {
        const auto& row = df.rows[row_idx];
        std::size_t word = row_idx / 64;
        std::size_t bit = row_idx % 64;
        std::uint64_t mask_bit = static_cast<std::uint64_t>(1) << bit;
        for (std::size_t col = 0; col < row.size(); ++col) {
            auto& map = df.value_masks[col];
            auto it = map.find(row[col]);
            if (it == map.end()) {
                it = map.emplace(row[col], std::vector<std::uint64_t>(words, 0)).first;
            }
            it->second[word] |= mask_bit;
        }
    }

    return df;
}

static int rand_int(int min_val, int max_val, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(min_val, max_val);
    return dist(rng);
}

static int rand_choice(const std::vector<int>& options, std::mt19937& rng) {
    if (options.empty()) {
        throw std::runtime_error("No hay opciones para elegir");
    }
    std::uniform_int_distribution<std::size_t> dist(0, options.size() - 1);
    return options[dist(rng)];
}

static void set_gene(
    Chromosome& crom1,
    Chromosome& crom2,
    std::size_t idx,
    int value,
    const std::vector<std::vector<int>>& alelos_crom2,
    std::mt19937& rng) {
    crom1[idx] = value;
    if (value == 0) {
        crom2[idx] = 0;
        return;
    }
    const auto& opciones = alelos_crom2[idx];
    if (opciones.empty()) {
        throw std::runtime_error("posicion de alelos_crom2 sin valor");
    }
    crom2[idx] = rand_choice(opciones, rng);
}

static int popcount64(std::uint64_t value) {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(value));
#else
    return __builtin_popcountll(value);
#endif
}

static std::vector<std::uint64_t> build_mask(
    const DataSet& df,
    const Chromosome& crom1,
    const Chromosome& crom2,
    int target) {
    if (df.row_count == 0) {
        return {};
    }

    std::vector<std::uint64_t> mask = df.full_mask;
    bool has_literal = false;
    for (std::size_t idx = 0; idx < crom1.size(); ++idx) {
        if (crom1[idx] != target) {
            continue;
        }
        has_literal = true;
        auto map_it = VALUE_MAP[idx].find(crom2[idx]);
        if (map_it == VALUE_MAP[idx].end()) {
            return std::vector<std::uint64_t>(mask.size(), 0);
        }
        const std::string& value = map_it->second;
        const auto& masks = df.value_masks[idx];
        auto it = masks.find(value);
        if (it == masks.end()) {
            return std::vector<std::uint64_t>(mask.size(), 0);
        }
        const auto& val_mask = it->second;
        for (std::size_t i = 0; i < mask.size(); ++i) {
            mask[i] &= val_mask[i];
        }
    }

    if (!has_literal) {
        return df.full_mask;
    }
    return mask;
}

static bool ensure_regla(
    Chromosome& crom1,
    Chromosome& crom2,
    const std::vector<std::vector<int>>& alelos_crom2,
    std::mt19937& rng) {
    bool tiene_1 = std::any_of(crom1.begin(), crom1.end(), [](int v) { return v == 1; });
    bool tiene_2 = std::any_of(crom1.begin(), crom1.end(), [](int v) { return v == 2; });

    if (tiene_1 && tiene_2) {
        return true;
    }

    std::vector<std::size_t> zeros;
    std::vector<std::size_t> unos;
    std::vector<std::size_t> doses;
    for (std::size_t i = 0; i < crom1.size(); ++i) {
        if (crom1[i] == 0) {
            zeros.push_back(i);
        } else if (crom1[i] == 1) {
            unos.push_back(i);
        } else if (crom1[i] == 2) {
            doses.push_back(i);
        }
    }

    if (!tiene_1 && !tiene_2) {
        std::vector<std::size_t> candidates = zeros;
        if (candidates.size() < 2) {
            candidates.resize(crom1.size());
            for (std::size_t i = 0; i < crom1.size(); ++i) {
                candidates[i] = i;
            }
        }
        if (candidates.size() < 2) {
            return false;
        }
        std::shuffle(candidates.begin(), candidates.end(), rng);
        std::size_t i1 = candidates[0];
        std::size_t i2 = candidates[1];
        set_gene(crom1, crom2, i1, 1, alelos_crom2, rng);
        set_gene(crom1, crom2, i2, 2, alelos_crom2, rng);
        return true;
    }

    if (!tiene_1) {
        std::size_t target_idx = crom1.size();
        if (!zeros.empty()) {
            target_idx = zeros[rand_int(0, static_cast<int>(zeros.size() - 1), rng)];
        } else if (doses.size() > 1) {
            target_idx = doses[rand_int(0, static_cast<int>(doses.size() - 1), rng)];
        }
        if (target_idx == crom1.size()) {
            return false;
        }
        set_gene(crom1, crom2, target_idx, 1, alelos_crom2, rng);
        return true;
    }

    if (!tiene_2) {
        std::size_t target_idx = crom1.size();
        if (!zeros.empty()) {
            target_idx = zeros[rand_int(0, static_cast<int>(zeros.size() - 1), rng)];
        } else if (unos.size() > 1) {
            target_idx = unos[rand_int(0, static_cast<int>(unos.size() - 1), rng)];
        }
        if (target_idx == crom1.size()) {
            return false;
        }
        set_gene(crom1, crom2, target_idx, 2, alelos_crom2, rng);
        return true;
    }

    return true;
}

std::pair<Chromosome, Chromosome> generar_individuo(
    std::size_t tam_cromosoma,
    std::mt19937& rng) {
    return generar_individuo(tam_cromosoma, rng, DEFAULT_ALLELES_CROM2);
}

std::pair<Chromosome, Chromosome> generar_individuo(
    std::size_t tam_cromosoma,
    std::mt19937& rng,
    const std::vector<std::vector<int>>& alelos_crom2) {
    if (tam_cromosoma != alelos_crom2.size()) {
        throw std::runtime_error("tam_cromosoma debe coincidir con alelos_crom2");
    }

    Chromosome crom1;
    Chromosome crom2;
    crom1.reserve(tam_cromosoma);
    crom2.reserve(tam_cromosoma);

    for (const auto& opciones : alelos_crom2) {
        int gen1 = rand_int(0, 2, rng);
        int gen2 = 0;
        if (gen1 != 0) {
            gen2 = rand_choice(opciones, rng);
        }
        crom1.push_back(gen1);
        crom2.push_back(gen2);
    }

    ensure_regla(crom1, crom2, alelos_crom2, rng);
    return {crom1, crom2};
}

std::pair<Chromosome, Chromosome> mutacion_extension(
    const Chromosome& crom1,
    const Chromosome& crom2,
    std::mt19937& rng) {
    return mutacion_extension(crom1, crom2, rng, DEFAULT_ALLELES_CROM2);
}

std::pair<Chromosome, Chromosome> mutacion_extension(
    const Chromosome& crom1,
    const Chromosome& crom2,
    std::mt19937& rng,
    const std::vector<std::vector<int>>& alelos_crom2) {
    if (crom1.size() != crom2.size()) {
        throw std::runtime_error("Los cromosomas deben tener el mismo tamano");
    }
    if (crom1.size() != alelos_crom2.size()) {
        throw std::runtime_error("alelos_crom2 debe coincidir en longitud");
    }

    Chromosome nuevo_c1 = crom1;
    Chromosome nuevo_c2 = crom2;

    std::vector<std::size_t> indices_inactivos;
    for (std::size_t i = 0; i < nuevo_c1.size(); ++i) {
        if (nuevo_c1[i] == 0) {
            indices_inactivos.push_back(i);
        }
    }
    if (indices_inactivos.empty()) {
        return {nuevo_c1, nuevo_c2};
    }

    std::size_t idx = indices_inactivos[rand_int(0, static_cast<int>(indices_inactivos.size() - 1), rng)];
    nuevo_c1[idx] = rand_int(1, 2, rng);
    const auto& opciones = alelos_crom2[idx];
    if (opciones.empty()) {
        throw std::runtime_error("posicion de alelos_crom2 sin valor");
    }
    nuevo_c2[idx] = rand_choice(opciones, rng);

    return {nuevo_c1, nuevo_c2};
}

std::pair<Chromosome, Chromosome> mutacion_contraccion(
    const Chromosome& crom1,
    const Chromosome& crom2,
    std::mt19937& rng) {
    return mutacion_contraccion(crom1, crom2, rng, DEFAULT_ALLELES_CROM2);
}

std::pair<Chromosome, Chromosome> mutacion_contraccion(
    const Chromosome& crom1,
    const Chromosome& crom2,
    std::mt19937& rng,
    const std::vector<std::vector<int>>& alelos_crom2) {
    if (crom1.size() != crom2.size()) {
        throw std::runtime_error("Los cromosomas deben tener el mismo tamano");
    }

    std::vector<std::size_t> indices_activos;
    for (std::size_t i = 0; i < crom1.size(); ++i) {
        if (crom1[i] != 0) {
            indices_activos.push_back(i);
        }
    }
    if (indices_activos.empty()) {
        return {crom1, crom2};
    }

    std::shuffle(indices_activos.begin(), indices_activos.end(), rng);

    for (std::size_t idx : indices_activos) {
        Chromosome nuevo_c1 = crom1;
        Chromosome nuevo_c2 = crom2;
        set_gene(nuevo_c1, nuevo_c2, idx, 0, alelos_crom2, rng);
        if (ensure_regla(nuevo_c1, nuevo_c2, alelos_crom2, rng)) {
            return {nuevo_c1, nuevo_c2};
        }
    }

    return {crom1, crom2};
}

std::pair<Chromosome, Chromosome> mutacion_reemplazo(
    const Chromosome& crom1,
    const Chromosome& crom2,
    std::mt19937& rng) {
    return mutacion_reemplazo(crom1, crom2, rng, DEFAULT_ALLELES_CROM2);
}

std::pair<Chromosome, Chromosome> mutacion_reemplazo(
    const Chromosome& crom1,
    const Chromosome& crom2,
    std::mt19937& rng,
    const std::vector<std::vector<int>>& alelos_crom2) {
    if (crom1.size() != crom2.size()) {
        throw std::runtime_error("Los cromosomas deben tener el mismo tamano");
    }
    if (crom1.size() != alelos_crom2.size()) {
        throw std::runtime_error("alelos_crom2 debe coincidir en longitud");
    }

    std::vector<std::size_t> activos;
    for (std::size_t i = 0; i < crom1.size(); ++i) {
        if (crom1[i] != 0) {
            activos.push_back(i);
        }
    }
    if (activos.empty()) {
        return {crom1, crom2};
    }

    std::size_t idx = activos[rand_int(0, static_cast<int>(activos.size() - 1), rng)];
    const auto& opciones = alelos_crom2[idx];
    if (opciones.empty()) {
        return {crom1, crom2};
    }

    std::vector<int> opciones_nuevas;
    opciones_nuevas.reserve(opciones.size());
    for (int val : opciones) {
        if (val != crom2[idx]) {
            opciones_nuevas.push_back(val);
        }
    }
    if (opciones_nuevas.empty()) {
        return {crom1, crom2};
    }

    Chromosome nuevo_c1 = crom1;
    Chromosome nuevo_c2 = crom2;
    nuevo_c2[idx] = rand_choice(opciones_nuevas, rng);
    return {nuevo_c1, nuevo_c2};
}

std::pair<std::pair<Chromosome, Chromosome>, std::pair<Chromosome, Chromosome>>
recombinacion_n_puntos(
    const Chromosome& padre1_c1,
    const Chromosome& padre1_c2,
    const Chromosome& padre2_c1,
    const Chromosome& padre2_c2,
    int n_puntos,
    std::mt19937& rng) {
    return recombinacion_n_puntos(padre1_c1, padre1_c2, padre2_c1, padre2_c2, n_puntos, rng, DEFAULT_ALLELES_CROM2);
}

std::pair<std::pair<Chromosome, Chromosome>, std::pair<Chromosome, Chromosome>>
recombinacion_n_puntos(
    const Chromosome& padre1_c1,
    const Chromosome& padre1_c2,
    const Chromosome& padre2_c1,
    const Chromosome& padre2_c2,
    int n_puntos,
    std::mt19937& rng,
    const std::vector<std::vector<int>>& alelos_crom2) {
    if (padre1_c1.size() != padre1_c2.size() || padre2_c1.size() != padre2_c2.size()) {
        throw std::runtime_error("Cada padre debe tener cromosomas de igual longitud");
    }
    if (padre1_c1.size() != padre2_c1.size()) {
        throw std::runtime_error("Los cromosomas de ambos padres deben tener la misma longitud");
    }
    if (padre1_c1.size() != alelos_crom2.size()) {
        throw std::runtime_error("alelos_crom2 debe coincidir en longitud con los cromosomas");
    }

    std::size_t l = padre1_c1.size();
    if (l < 2) {
        Chromosome h1_c1 = padre1_c1;
        Chromosome h1_c2 = padre1_c2;
        Chromosome h2_c1 = padre2_c1;
        Chromosome h2_c2 = padre2_c2;
        ensure_regla(h1_c1, h1_c2, alelos_crom2, rng);
        ensure_regla(h2_c1, h2_c2, alelos_crom2, rng);
        return {{h1_c1, h1_c2}, {h2_c1, h2_c2}};
    }

    int n = std::max(1, std::min(n_puntos, static_cast<int>(l - 1)));
    std::vector<int> puntos;
    puntos.reserve(n + 1);
    std::vector<int> candidates;
    candidates.reserve(l - 1);
    for (int i = 1; i < static_cast<int>(l); ++i) {
        candidates.push_back(i);
    }
    std::shuffle(candidates.begin(), candidates.end(), rng);
    for (int i = 0; i < n; ++i) {
        puntos.push_back(candidates[i]);
    }
    std::sort(puntos.begin(), puntos.end());
    puntos.push_back(static_cast<int>(l));

    Chromosome h1_c1;
    Chromosome h1_c2;
    Chromosome h2_c1;
    Chromosome h2_c2;
    h1_c1.reserve(l);
    h1_c2.reserve(l);
    h2_c1.reserve(l);
    h2_c2.reserve(l);

    int src = 0;
    for (std::size_t i = 0; i < puntos.size(); ++i) {
        int punto = puntos[i];
        if (i % 2 == 0) {
            h1_c1.insert(h1_c1.end(), padre1_c1.begin() + src, padre1_c1.begin() + punto);
            h1_c2.insert(h1_c2.end(), padre1_c2.begin() + src, padre1_c2.begin() + punto);
            h2_c1.insert(h2_c1.end(), padre2_c1.begin() + src, padre2_c1.begin() + punto);
            h2_c2.insert(h2_c2.end(), padre2_c2.begin() + src, padre2_c2.begin() + punto);
        } else {
            h1_c1.insert(h1_c1.end(), padre2_c1.begin() + src, padre2_c1.begin() + punto);
            h1_c2.insert(h1_c2.end(), padre2_c2.begin() + src, padre2_c2.begin() + punto);
            h2_c1.insert(h2_c1.end(), padre1_c1.begin() + src, padre1_c1.begin() + punto);
            h2_c2.insert(h2_c2.end(), padre1_c2.begin() + src, padre1_c2.begin() + punto);
        }
        src = punto;
    }

    ensure_regla(h1_c1, h1_c2, alelos_crom2, rng);
    ensure_regla(h2_c1, h2_c2, alelos_crom2, rng);
    return {{h1_c1, h1_c2}, {h2_c1, h2_c2}};
}

std::tuple<int, int, int, int> contar_casos_regla(
    const DataSet& df,
    const Chromosome& crom1,
    const Chromosome& crom2) {
    if (crom1.size() != crom2.size()) {
        throw std::runtime_error("Los cromosomas deben tener la misma longitud");
    }
    if (crom1.size() != COLUMN_ORDER.size()) {
        throw std::runtime_error("La longitud de los cromosomas debe coincidir con las columnas");
    }

    if (!df.value_masks.empty()) {
        auto ante_mask = build_mask(df, crom1, crom2, 1);
        auto cons_mask = build_mask(df, crom1, crom2, 2);

        int a_c = 0;
        int a_no_c = 0;
        int no_a_c = 0;

        for (std::size_t i = 0; i < ante_mask.size(); ++i) {
            std::uint64_t a = ante_mask[i];
            std::uint64_t c = cons_mask[i];
            std::uint64_t not_a = ~a;
            std::uint64_t not_c = ~c;
            if (i + 1 == ante_mask.size()) {
                not_a &= df.tail_mask;
                not_c &= df.tail_mask;
            }
            a_c += popcount64(a & c);
            a_no_c += popcount64(a & not_c);
            no_a_c += popcount64(not_a & c);
        }

        int no_a_no_c = static_cast<int>(df.row_count) - a_c - a_no_c - no_a_c;
        return {a_c, a_no_c, no_a_c, no_a_no_c};
    }

    int a_c = 0;
    int a_no_c = 0;
    int no_a_c = 0;
    int no_a_no_c = 0;

    for (const auto& row : df.rows) {
        bool ante_match = true;
        bool cons_match = true;
        for (std::size_t idx = 0; idx < crom1.size(); ++idx) {
            int c1 = crom1[idx];
            if (c1 != 1 && c1 != 2) {
                continue;
            }
            auto map_it = VALUE_MAP[idx].find(crom2[idx]);
            if (map_it == VALUE_MAP[idx].end()) {
                throw std::runtime_error("Valor invalido en cromosoma");
            }
            const std::string& expected = map_it->second;
            if (c1 == 1 && row[idx] != expected) {
                ante_match = false;
            } else if (c1 == 2 && row[idx] != expected) {
                cons_match = false;
            }
            if (!ante_match && !cons_match) {
                break;
            }
        }

        if (ante_match && cons_match) {
            ++a_c;
        } else if (ante_match && !cons_match) {
            ++a_no_c;
        } else if (!ante_match && cons_match) {
            ++no_a_c;
        } else {
            ++no_a_no_c;
        }
    }

    return {a_c, a_no_c, no_a_c, no_a_no_c};
}

double casual_supp(int a_c, int a_no_c, int no_a_c, int no_a_no_c) {
    int total = a_c + a_no_c + no_a_c + no_a_no_c;
    if (total == 0) {
        return 3.0;
    }
    return (1.0 - ((a_c + no_a_no_c) / static_cast<double>(total))) * 2.0;
}

double casual_conf(int a_c, int a_no_c, int no_a_c, int no_a_no_c) {
    int total_a = a_c + a_no_c;
    int total_no_a = no_a_c + no_a_no_c;

    if (total_a == 0 || total_no_a == 0) {
        return 3.0;
    }

    double conf_pos = a_c / static_cast<double>(total_a);
    double conf_neg = no_a_no_c / static_cast<double>(total_no_a);

    return (1.0 - (0.5 * (conf_pos + conf_neg))) * 2.0;
}

double max_conf(int a_c, int a_no_c, int no_a_c, int no_a_no_c) {
    int total_a = a_c + a_no_c;
    int total_y = a_c + no_a_c;

    if (total_a == 0 || total_y == 0) {
        return 3.0;
    }

    double conf_xy = a_c / static_cast<double>(total_a);
    double conf_yx = a_c / static_cast<double>(total_y);

    return (1.0 - std::max(conf_xy, conf_yx)) * 2.0;
}

double jaccard(int a_c, int a_no_c, int no_a_c, int no_a_no_c) {
    int total = a_c + a_no_c + no_a_c + no_a_no_c;
    if (total == 0) {
        return 3.0;
    }

    int denom = a_c + a_no_c + no_a_c;
    if (denom == 0) {
        return 3.0;
    }

    return a_c / static_cast<double>(denom);
}

double cosine(int a_c, int a_no_c, int no_a_c, int no_a_no_c) {
    int total = a_c + a_no_c + no_a_c + no_a_no_c;
    if (total == 0) {
        return 3.0;
    }

    double p_x = (a_c + a_no_c) / static_cast<double>(total);
    double p_y = (a_c + no_a_c) / static_cast<double>(total);
    double denom = std::sqrt(p_x * p_y);
    if (denom == 0.0) {
        return 3.0;
    }
    return (1.0 - ((a_c / static_cast<double>(total)) / denom)) * 2.0;
}

double phi(int a_c, int a_no_c, int no_a_c, int no_a_no_c) {
    int total = a_c + a_no_c + no_a_c + no_a_no_c;
    if (total == 0) {
        return 3.0;
    }

    double p_x = (a_c + a_no_c) / static_cast<double>(total);
    double p_y = (a_c + no_a_c) / static_cast<double>(total);
    double p_xy = a_c / static_cast<double>(total);
    double denom = std::sqrt(p_x * (1.0 - p_x) * p_y * (1.0 - p_y));
    if (denom == 0.0) {
        return 3.0;
    }
    return 1.0 - ((p_xy - p_x * p_y) / denom);
}

double kappa(int a_c, int a_no_c, int no_a_c, int no_a_no_c) {
    int total = a_c + a_no_c + no_a_c + no_a_no_c;
    if (total == 0) {
        return 3.0;
    }

    double p_x = (a_c + a_no_c) / static_cast<double>(total);
    double p_y = (a_c + no_a_c) / static_cast<double>(total);
    double p_not_x = 1.0 - p_x;
    double p_not_y = 1.0 - p_y;
    double p_xy = a_c / static_cast<double>(total);
    double p_not_x_not_y = no_a_no_c / static_cast<double>(total);

    double num = p_xy + p_not_x_not_y - p_x * p_y - p_not_x * p_not_y;
    double denom = 1.0 - p_x * p_y - p_not_x * p_not_y;
    if (denom == 0.0) {
        return 3.0;
    }
    return 1.0 - (num / denom);
}

double rule_size_metric(const Chromosome& crom1, int max_elements) {
    int count = 0;
    for (int v : crom1) {
        if (v != 0) {
            count += 1;
        }
    }
    if (max_elements <= 0) {
        return 2.0;
    }
    double value = (static_cast<double>(max_elements) - count) / 6.0;
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 2.0) {
        return 2.0;
    }
    return value;
}

double support(int a_c, int a_no_c, int no_a_c, int no_a_no_c) {
    int total = a_c + a_no_c + no_a_c + no_a_no_c;
    if (total == 0) {
        return 3.0;
    }
    double supp = a_c / static_cast<double>(total);
    return (1.0 - supp) * 2.0;
}

double confidence(int a_c, int a_no_c, int no_a_c, int no_a_no_c) {
    int total_a = a_c + a_no_c;
    if (total_a == 0) {
        return 3.0;
    }
    double conf = a_c / static_cast<double>(total_a);
    return (1.0 - conf) * 2.0;
}

double leverage(int a_c, int a_no_c, int no_a_c, int no_a_no_c) {
    int total = a_c + a_no_c + no_a_c + no_a_no_c;
    if (total == 0) {
        return 3.0;
    }
    double p_x = (a_c + a_no_c) / static_cast<double>(total);
    double p_y = (a_c + no_a_c) / static_cast<double>(total);
    double p_xy = a_c / static_cast<double>(total);
    double lev = p_xy - (p_x * p_y);
    return (0.25 - lev) * 4.0;
}

double chi_cuadrada(int a_c, int a_no_c, int no_a_c, int no_a_no_c) {
    int total = a_c + a_no_c + no_a_c + no_a_no_c;
    if (total == 0) {
        return 3.0;
    }
    double a = static_cast<double>(a_c);
    double b = static_cast<double>(a_no_c);
    double c = static_cast<double>(no_a_c);
    double d = static_cast<double>(no_a_no_c);
    double denom = (a + b) * (c + d) * (a + c) * (b + d);
    if (denom == 0.0) {
        return 3.0;
    }
    double chi2 = (static_cast<double>(total) * (a * d - b * c) * (a * d - b * c)) / denom;
    return (1.0 / (1.0 + chi2)) * 2.0;
}

double coverage(int a_c, int a_no_c, int no_a_c, int no_a_no_c) {
    int total = a_c + a_no_c + no_a_c + no_a_no_c;
    if (total == 0) {
        return 3.0;
    }
    double cov = (a_c + a_no_c) / static_cast<double>(total);
    return (1.0 - cov) * 2.0;
}

static long long comb_ll(int n, int k) {
    if (k < 0 || k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    k = std::min(k, n - k);
    long long result = 1;
    for (int i = 1; i <= k; ++i) {
        result = result * (n - k + i) / i;
    }
    return result;
}

static void combinations_recursive(
    int n,
    int k,
    int start,
    std::vector<int>& current,
    std::vector<std::vector<int>>& out) {
    if (static_cast<int>(current.size()) == k) {
        out.push_back(current);
        return;
    }
    for (int i = start; i <= n - (k - static_cast<int>(current.size())); ++i) {
        current.push_back(i);
        combinations_recursive(n, k, i + 1, current, out);
        current.pop_back();
    }
}

std::vector<std::vector<double>> generar_pesos_das_dennis(int m, int n_vectores) {
    if (m <= 0) {
        throw std::runtime_error("m debe ser positivo");
    }
    if (n_vectores <= 0) {
        throw std::runtime_error("n_vectores debe ser positivo");
    }

    int H = 0;
    while (true) {
        long long combos = comb_ll(H + m - 1, m - 1);
        if (combos >= n_vectores) {
            break;
        }
        ++H;
    }

    std::vector<std::vector<int>> cuts_list;
    std::vector<int> current;
    combinations_recursive(H + m - 2, m - 1, 0, current, cuts_list);

    std::vector<std::vector<double>> pesos;
    pesos.reserve(cuts_list.size());

    for (const auto& cuts : cuts_list) {
        std::vector<int> indices;
        indices.reserve(m + 1);
        indices.push_back(-1);
        for (int c : cuts) {
            indices.push_back(c);
        }
        indices.push_back(H + m - 1);

        std::vector<double> parts;
        parts.reserve(m);
        for (int i = 0; i < m; ++i) {
            int val = indices[i + 1] - indices[i] - 1;
            parts.push_back(val / static_cast<double>(H));
        }
        pesos.push_back(std::move(parts));
    }

    return pesos;
}

}  // namespace moead

#ifdef GENERAR_INDIVIDUO_MAIN
int main() {
    try {
        auto df = moead::read_csv("discretized_dataset.csv");
        moead::Chromosome c1 = {2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 2};
        moead::Chromosome c2 = {1, 4, 1, 2, 2, 2, 1, 4, 2, 1, 3};
        auto counts = moead::contar_casos_regla(df, c1, c2);
        int a_c, a_no_c, no_a_c, no_a_no_c;
        std::tie(a_c, a_no_c, no_a_c, no_a_no_c) = counts;
        std::cout << "Conteos (a_c, a_no_c, no_a_c, no_a_no_c): "
                  << a_c << ", " << a_no_c << ", " << no_a_c << ", " << no_a_no_c << "\n";
        std::cout << "casual-supp: " << moead::casual_supp(a_c, a_no_c, no_a_c, no_a_no_c) << "\n";
        std::cout << "casual-conf: " << moead::casual_conf(a_c, a_no_c, no_a_c, no_a_no_c) << "\n";
        std::cout << "max-conf: " << moead::max_conf(a_c, a_no_c, no_a_c, no_a_no_c) << "\n";
        std::cout << "jaccard: " << moead::jaccard(a_c, a_no_c, no_a_c, no_a_no_c) << "\n";
        std::cout << "cosine: " << moead::cosine(a_c, a_no_c, no_a_c, no_a_no_c) << "\n";
        std::cout << "phi: " << moead::phi(a_c, a_no_c, no_a_c, no_a_no_c) << "\n";
        std::cout << "kappa: " << moead::kappa(a_c, a_no_c, no_a_c, no_a_no_c) << "\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
    return 0;
}
#endif

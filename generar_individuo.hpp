#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace moead {

using Chromosome = std::vector<int>;

struct DataSet {
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;
    std::vector<std::unordered_map<std::string, std::vector<std::uint64_t>>> value_masks;
    std::vector<std::uint64_t> full_mask;
    std::size_t row_count = 0;
    std::uint64_t tail_mask = 0;
};

extern const std::vector<std::vector<int>> DEFAULT_ALLELES_CROM2;
extern const std::vector<std::unordered_map<int, std::string>> VALUE_MAP;
extern const std::vector<std::string> COLUMN_ORDER;

DataSet read_csv(const std::string& path);

std::pair<Chromosome, Chromosome> generar_individuo(
    std::size_t tam_cromosoma,
    std::mt19937& rng);

std::pair<Chromosome, Chromosome> generar_individuo(
    std::size_t tam_cromosoma,
    std::mt19937& rng,
    const std::vector<std::vector<int>>& alelos_crom2);

std::pair<Chromosome, Chromosome> mutacion_extension(
    const Chromosome& crom1,
    const Chromosome& crom2,
    std::mt19937& rng);

std::pair<Chromosome, Chromosome> mutacion_extension(
    const Chromosome& crom1,
    const Chromosome& crom2,
    std::mt19937& rng,
    const std::vector<std::vector<int>>& alelos_crom2);

std::pair<Chromosome, Chromosome> mutacion_contraccion(
    const Chromosome& crom1,
    const Chromosome& crom2,
    std::mt19937& rng);

std::pair<Chromosome, Chromosome> mutacion_contraccion(
    const Chromosome& crom1,
    const Chromosome& crom2,
    std::mt19937& rng,
    const std::vector<std::vector<int>>& alelos_crom2);

std::pair<Chromosome, Chromosome> mutacion_reemplazo(
    const Chromosome& crom1,
    const Chromosome& crom2,
    std::mt19937& rng);

std::pair<Chromosome, Chromosome> mutacion_reemplazo(
    const Chromosome& crom1,
    const Chromosome& crom2,
    std::mt19937& rng,
    const std::vector<std::vector<int>>& alelos_crom2);

std::pair<std::pair<Chromosome, Chromosome>, std::pair<Chromosome, Chromosome>>
recombinacion_n_puntos(
    const Chromosome& padre1_c1,
    const Chromosome& padre1_c2,
    const Chromosome& padre2_c1,
    const Chromosome& padre2_c2,
    int n_puntos,
    std::mt19937& rng);

std::pair<std::pair<Chromosome, Chromosome>, std::pair<Chromosome, Chromosome>>
recombinacion_n_puntos(
    const Chromosome& padre1_c1,
    const Chromosome& padre1_c2,
    const Chromosome& padre2_c1,
    const Chromosome& padre2_c2,
    int n_puntos,
    std::mt19937& rng,
    const std::vector<std::vector<int>>& alelos_crom2);

std::tuple<int, int, int, int> contar_casos_regla(
    const DataSet& df,
    const Chromosome& crom1,
    const Chromosome& crom2);

double casual_supp(int a_c, int a_no_c, int no_a_c, int no_a_no_c);

double casual_conf(int a_c, int a_no_c, int no_a_c, int no_a_no_c);

double max_conf(int a_c, int a_no_c, int no_a_c, int no_a_no_c);

double jaccard(int a_c, int a_no_c, int no_a_c, int no_a_no_c);

double cosine(int a_c, int a_no_c, int no_a_c, int no_a_no_c);

double phi(int a_c, int a_no_c, int no_a_c, int no_a_no_c);

double kappa(int a_c, int a_no_c, int no_a_c, int no_a_no_c);

std::vector<std::vector<double>> generar_pesos_das_dennis(int m, int n_vectores);

}  // namespace moead

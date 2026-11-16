// Independent checks for SIMD lane ordering, inclusive rejection thresholds,
// and decimal conversion against libc's correctly rounded reference parser.
#include <cassert>
#include <random>
#define main minivectordb_program_main
#include "../src/minivectordb.cpp"
#undef main

static void decimal_cases() {
  std::mt19937_64 rng(13849);
  std::vector<std::string> numbers = {
      "0", "-0", "+0", ".5", "-.5", "1.", "0000.00125", "1e-40",
      "3.4028234663852886e38", "-1.1754943508222875e-38", "0x1.8p+2",
      "1.000000059604644775390625", "0.1234567890123456789", "1e-310",
      "nan", "inf", "1e400", "1e-400", "+-1", "+", "--1"};
  for (size_t i = 0; i < 30000; ++i) {
    std::string digits = std::to_string(rng() % 1000000000000000ull);
    size_t places = rng() % (digits.size() + 1);
    if (places) digits.insert(digits.size() - places, ".");
    if (rng() % 2) digits.insert(0, "-");
    numbers.push_back(digits);
  }
  for (const std::string &number : numbers) {
    const std::string input = "[" + number + "]";
    std::vector<float> parsed;
    std::string err;
    bool ok = parse_vector_string(input, parsed, err);
    char *end = nullptr;
    errno = 0;
    double reference = std::strtod(number.c_str(), &end);
    bool reference_ok = end != number.c_str() && *end == '\0' && errno != ERANGE &&
                        std::isfinite(reference) && std::fabs(reference) <= FLT_MAX;
    assert(ok == reference_ok);
    if (ok) {
      assert(parsed.size() == 1);
      float expected = (float)reference;
      assert(std::memcmp(&parsed[0], &expected, sizeof(float)) == 0);
    }
  }
}

static void lane_cases() {
  std::mt19937 rng(7349);
  for (size_t repeat = 0; repeat < 2000; ++repeat) {
    alignas(32) int8_t rows[8][128], transposed[1024];
    uint8_t query[128];
    int bias[8] = {}, exact[8] = {}, query_norm = 0;
    for (size_t j = 0; j < 128; ++j) {
      int q = repeat % 17 == 0 ? 63 : (int)(rng() % 127) - 63;
      query[j] = (uint8_t)(q + 63);
      query_norm += q * q;
      for (size_t row = 0; row < 8; ++row) {
        int x = repeat % 17 == 0 ? -63 : (int)(rng() % 127) - 63;
        rows[row][j] = (int8_t)x;
        transposed[(j / 4) * 32 + row * 4 + j % 4] = (int8_t)x;
        bias[row] += x * x + 126 * x;
        exact[row] += (q - x) * (q - x);
      }
    }
    for (size_t row = 0; row < 8; ++row) {
      int reference = 0;
      for (size_t j = 0; j < 128; ++j) reference += rows[row][j] * query[j];
      assert(quant_dot_lane128(transposed, row, query) == reference);
    }
#if defined(__x86_64__)
    if (__builtin_cpu_supports("avx2")) {
      for (size_t chosen = 0; chosen < 8; ++chosen) {
        for (int delta : {-1, 0, 1}) {
          int distances[8];
          int limit = exact[chosen] + delta;
          unsigned actual = quant_mask8_128(transposed, query, bias, query_norm, limit, distances);
          unsigned expected = 0;
          for (size_t row = 0; row < 8; ++row) {
            assert(distances[row] == exact[row]);
            if (exact[row] <= limit) expected |= 1u << row;
          }
          assert(actual == expected);
        }
      }
    }
#endif
  }
}

int main() {
  decimal_cases();
  lane_cases();
  std::cout << "PASS SIMD lane distances, inclusive masks and decimal parsing\n";
}

#include <cassert>
#include <random>
#define main minivectordb_program_main
#include "../src/minivectordb.cpp"
#undef main

int main() {
#if defined(__x86_64__)
  __builtin_cpu_init();
  const bool avx2 = __builtin_cpu_supports("avx2") != 0;
#endif
  std::mt19937 rng(7119);
  const size_t dimensions[] = {
      0, 1, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 95, 96, 97,
      127, 128, 129, 191, 192, 193, 255, 256, 257, 4095, 4096};
  for (size_t dim : dimensions) {
    for (size_t offset = 0; offset < 32; ++offset) {
      std::vector<int8_t> row(dim + offset);
      std::vector<uint8_t> query(dim + 31 - offset);
      int8_t *row_data = row.data();
      uint8_t *query_data = query.data();
      if (offset) row_data += offset;
      if (31 - offset) query_data += 31 - offset;
      for (int pattern = 0; pattern < 8; ++pattern) {
        for (size_t coord = 0; coord < dim; ++coord) {
          switch (pattern) {
            case 0:
              row_data[coord] = 63;
              query_data[coord] = 126;
              break;
            case 1:
              row_data[coord] = -63;
              query_data[coord] = 126;
              break;
            case 2:
              row_data[coord] = (coord / 32) % 2 ? -63 : 63;
              query_data[coord] = 126;
              break;
            case 3:
              row_data[coord] = (coord / 2) % 2 ? -63 : 63;
              query_data[coord] = 126;
              break;
            case 4:
              row_data[coord] = 63;
              query_data[coord] = 0;
              break;
            case 5:
              row_data[coord] = 0;
              query_data[coord] = 126;
              break;
            default:
              row_data[coord] = (int8_t)((int)(rng() % 127) - 63);
              query_data[coord] = (uint8_t)(rng() % 127);
              break;
          }
        }
        int64_t expected = 0;
        for (size_t coord = 0; coord < dim; ++coord)
          expected += (int64_t)row_data[coord] * query_data[coord];
        assert(quant_dot(row_data, query_data, dim) == expected);
#if defined(__x86_64__)
        if (avx2)
          assert(quant_dot_avx2(row_data, query_data, dim) == expected);
#endif
      }
    }
  }
  std::puts("PASS quant_dot_test: exact integer dots, extrema, tails, unaligned inputs");
}

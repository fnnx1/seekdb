// Focused ReadBuffer regression. This uses the real 128 MiB spill threshold.
// Example build (run separately from timed benchmarks):
// g++ -O2 -std=c++17 -pthread -fopenmp tests/vector_store_read_test.cpp -o /tmp/vector-store-test
// Override MVDB_VECTOR_STORE_SOURCE to test the repository implementation.
#include <atomic>
#include <cassert>
#include <cerrno>
#include <exception>
#include <thread>
#include <unistd.h>

static thread_local size_t injected_chunk_limit = 0;
static thread_local unsigned injected_interrupts = 0;
static thread_local bool injected_eof = false;
static thread_local bool injected_io_error = false;
static thread_local size_t intercepted_reads = 0;

static ssize_t vector_store_test_pread(int fd, void *data, size_t count,
                                       off_t offset) {
  ++intercepted_reads;
  if (injected_interrupts) {
    --injected_interrupts;
    errno = EINTR;
    return -1;
  }
  if (injected_eof) return 0;
  if (injected_io_error) { errno = EIO; return -1; }
  if (injected_chunk_limit && count > injected_chunk_limit)
    count = injected_chunk_limit;
  return ::pread(fd, data, count, offset);
}

#ifndef MVDB_VECTOR_STORE_SOURCE
#define MVDB_VECTOR_STORE_SOURCE "../src/minivectordb.cpp"
#endif
#define main minivectordb_program_main
#define pread vector_store_test_pread
#include MVDB_VECTOR_STORE_SOURCE
#undef pread
#undef main

static constexpr uint64_t memory_limit = 128ull * 1024 * 1024;

static float coordinate(uint64_t row, size_t column, uint64_t seed) {
  uint32_t bits = 0x3f000000u |
      (uint32_t)((row * 2654435761ull + column * 2246822519ull + seed * 3266489917ull) &
                 0x007fffffu);
  if ((row + column + seed) & 1) bits |= 0x80000000u;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static Value reference(uint64_t row, size_t dim) {
  Value value;
  value.type = VT_VECTOR;
  value.vdisk = true;
  value.vlen = (uint32_t)dim;
  value.voff = row * dim * sizeof(float);
  return value;
}

static Value owned(uint64_t row, size_t dim, uint64_t seed) {
  Value value;
  value.type = VT_VECTOR;
  value.v.resize(dim);
  for (size_t column = 0; column < dim; ++column)
    value.v[column] = coordinate(row, column, seed);
  return value;
}

static void check_vector(const float *actual, uint64_t row, size_t dim,
                         uint64_t seed) {
  for (size_t column = 0; column < dim; ++column) {
    float expected = coordinate(row, column, seed);
    assert(std::memcmp(actual + column, &expected, sizeof(float)) == 0);
  }
}

static void append(VectorStore &store, uint64_t row, size_t dim, uint64_t seed) {
  Value value = owned(row, dim, seed);
  Value saved = store.store(value);
  assert(saved.vdisk && saved.voff == row * dim * sizeof(float));
  assert(saved.vlen == dim);
}

static uint64_t fill_past_spill(VectorStore &store, size_t dim, uint64_t seed) {
  const uint64_t count = memory_limit / (dim * sizeof(float)) + 3;
  Value value;
  value.type = VT_VECTOR;
  value.v.resize(dim);
  for (uint64_t row = 0; row < count; ++row) {
    for (size_t column = 0; column < dim; ++column)
      value.v[column] = coordinate(row, column, seed);
    Value saved = store.store(value);
    assert(saved.voff == row * dim * sizeof(float));
    if ((row + 1) * dim * sizeof(float) <= memory_limit)
      assert(store.raw_base() != nullptr);
  }
  assert(store.raw_base() == nullptr);
  assert(store.mapped_bytes() == 0);
  return count;
}

static void compare(VectorStore &store, uint64_t row, size_t dim, uint64_t seed,
                    VectorStore::ReadBuffer &window, bool read_ahead = true) {
  const Value saved = reference(row, dim);
  const float *actual = store.get(saved, window, read_ahead);
  check_vector(actual, row, dim, seed);
  std::vector<float> direct;
  const float *baseline = store.get(saved, direct);
  assert(std::memcmp(actual, baseline, dim * sizeof(float)) == 0);
}

static void check_samples(VectorStore &store, uint64_t count, size_t dim,
                          uint64_t seed, VectorStore::ReadBuffer &window) {
  const uint64_t samples[] = {0, 1, count / 2, count - 2, count - 1};
  for (uint64_t row : samples)
    compare(store, row, dim, seed, window);
  uint64_t state = 7349347;
  for (size_t repeat = 0; repeat < 1500; ++repeat) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    compare(store, state % count, dim, seed, window);
  }
}

static void thread_checks(VectorStore &store, uint64_t count, size_t dim,
                          uint64_t seed) {
  constexpr size_t workers = 4;
  std::atomic<size_t> ready{0};
  std::exception_ptr errors[workers];
  std::thread threads[workers];
  for (size_t worker = 0; worker < workers; ++worker) {
    threads[worker] = std::thread([&, worker] {
      try {
        VectorStore::ReadBuffer window;
        ready.fetch_add(1, std::memory_order_release);
        while (ready.load(std::memory_order_acquire) != workers)
          std::this_thread::yield();
        const uint64_t start = count * worker / workers;
        for (uint64_t step = 0; step < 2000; ++step)
          compare(store, (start + step) % count, dim, seed, window);
        for (uint64_t step = 0; step < 300; ++step)
          compare(store, (start + step * 997) % count, dim, seed, window);
      } catch (...) { errors[worker] = std::current_exception(); }
    });
  }
  for (auto &thread : threads) thread.join();
  for (auto &error : errors) if (error) std::rethrow_exception(error);
}

static void one_dimension(const std::string &directory, size_t dim,
                          uint64_t seed) {
  VectorStore store;
  store.init(directory, "window_" + std::to_string(dim), (int)dim);
  VectorStore::ReadBuffer window;
  Value local = owned(0, dim, seed);
  check_vector(store.get(local, window), 0, dim, seed);
  uint64_t count = fill_past_spill(store, dim, seed);
  check_samples(store, count, dim, seed, window);

  // An odd dimension puts complete vectors across 64 KiB boundaries.
  if (dim & 1) {
    uint64_t crossing = UINT64_MAX;
    for (uint64_t row = 0; row < count; ++row) {
      const uint64_t within = row * dim * sizeof(float) % (64u * 1024u);
      if (within + dim * sizeof(float) > 64u * 1024u) {
        crossing = row;
        break;
      }
    }
    assert(crossing != UINT64_MAX);
    for (uint64_t row = crossing - 1; row <= crossing + 1; ++row)
      compare(store, row, dim, seed, window);
  }

  // A cached partial EOF window must expand when append creates new bytes.
  compare(store, count - 1, dim, seed, window);
  const uint64_t previous_end = count * dim * sizeof(float);
  assert(window.offset + window.valid == previous_end);
  append(store, count, dim, seed);
  compare(store, count, dim, seed, window);
  compare(store, count - 1, dim, seed, window);
  ++count;

  // Mixing direct reads with read-ahead must invalidate overwritten contents.
  compare(store, 0, dim, seed, window);
  compare(store, count / 2, dim, seed, window, false);
  assert(window.valid == 0);
  compare(store, 0, dim, seed, window);
  compare(store, count / 2, dim, seed, window);

  // Buffered pread must retry EINTR and combine arbitrary byte short reads.
  VectorStore::ReadBuffer interrupted;
  const size_t before_reads = intercepted_reads;
  injected_interrupts = 2;
  injected_chunk_limit = 137;
  check_vector(store.get(reference(17, dim), interrupted), 17, dim, seed);
  assert(injected_interrupts == 0);
  assert(intercepted_reads > before_reads + 3);
  injected_chunk_limit = 0;
  compare(store, 18, dim, seed, interrupted);

  // An unsuccessful refill must not expose stale valid data on a retry.
  for (bool eof : {false, true}) {
    VectorStore::ReadBuffer failed;
    injected_io_error = !eof;
    injected_eof = eof;
    bool error = false;
    try { store.get(reference(0, dim), failed); }
    catch (const SqlError &) { error = true; }
    injected_io_error = false;
    injected_eof = false;
    assert(error && failed.valid == 0);
    compare(store, 0, dim, seed, failed);
  }

  // A distinct store and a moved-from object may reuse a ReadBuffer safely.
  VectorStore moved(std::move(store));
  check_samples(moved, count, dim, seed, window);
  store.init(directory, "moved_from_" + std::to_string(dim), (int)dim);
  append(store, 0, dim, seed + 1);
  compare(store, 0, dim, seed + 1, window);
  compare(moved, 0, dim, seed, window);

  // Move assignment back into the original object invalidates prior windows.
  store = std::move(moved);
  check_samples(store, count, dim, seed, window);
  thread_checks(store, count, dim, seed);

  // An out-of-range request must fail before attempting a bogus cache hit.
  bool threw = false;
  try { store.get(reference(count + 1, dim), window); }
  catch (const SqlError &) { threw = true; }
  assert(threw);

  // Reinitialize the same object with the same dimension and changed bytes.
  // This forces a second real spill so an old file-backed window could match
  // offsets and owner address if release/init failed to change its epoch.
  compare(store, 0, dim, seed, window);
  store.init(directory, "reinitialized_" + std::to_string(dim), (int)dim);
  count = fill_past_spill(store, dim, seed + 7);
  compare(store, 0, dim, seed + 7, window);
  check_samples(store, count, dim, seed + 7, window);
  store.release();
}

int main() {
  char temporary[] = "/tmp/mvdb-read-buffer-test-XXXXXX";
  char *created = ::mkdtemp(temporary);
  assert(created);
  const std::string directory(created);
  try {
    one_dimension(directory, 128, 137);
    one_dimension(directory, 129, 937);
    fs::remove_all(directory);
  } catch (...) {
    fs::remove_all(directory);
    throw;
  }
  std::cout << "PASS real spill, bitwise reads, odd dimensions, append, mode "
               "switching, move, reinit and independent thread windows\n";
}

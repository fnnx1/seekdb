// Focused ownership checks for the compact cells, plus a real snapshot reload.
// Build this test with -fsanitize=address,undefined -fno-omit-frame-pointer.
// ASAN_OPTIONS=detect_leaks=1 /tmp/stored-value-test
#include <cassert>
#define main minivectordb_program_main
#include "../src/minivectordb.cpp"
#undef main

static void equal_value(const StoredValue &actual, const Value &expected) {
  Value got = actual;
  assert(got.type == expected.type);
  assert(got.vdisk == expected.vdisk);
  assert(got.vlen == expected.vlen);
  switch (expected.type) {
    case VT_BIGINT: assert(got.i == expected.i); break;
    case VT_DOUBLE:
      assert(std::memcmp(&got.d, &expected.d, sizeof(double)) == 0);
      break;
    case VT_VARCHAR: assert(got.s == expected.s); break;
    case VT_VECTOR:
      if (expected.vdisk) assert(got.voff == expected.voff);
      else assert(got.v == expected.v);
      break;
    default: assert(false);
  }
}

static void ownership_checks() {
  std::vector<Value> samples(8);
  samples[0].i = INT64_MIN;
  samples[1].i = INT64_MAX;
  samples[2].type = VT_DOUBLE;
  samples[2].d = -0.0;
  samples[3].type = VT_VARCHAR;
  samples[3].s = std::string(2048, 'x') + "向量数据库" + std::string("a\0b", 3);
  samples[4].type = VT_VARCHAR;
  samples[5].type = VT_VECTOR;
  samples[5].v = {0.0f, -1.25f, 2.5f};
  samples[5].vlen = 3;
  samples[6].type = VT_VECTOR;  // Owned empty vector still owns its container.
  samples[7].type = VT_VECTOR;
  samples[7].vdisk = true;
  samples[7].voff = UINT64_MAX - 128;
  samples[7].vlen = 128;

  for (const auto &value : samples) {
    StoredValue original(value);
    StoredValue copy(original);
    equal_value(copy, value);
    if (value.type == VT_VARCHAR) {
      original.str->append("changed");
      assert(copy.str != original.str);
    } else if (value.type == VT_VECTOR && !value.vdisk) {
      original.vec->push_back(9.0f);
      assert(copy.vec != original.vec);
    }
    equal_value(copy, value);  // Copies must own independent allocations.
    StoredValue moved(std::move(copy));
    equal_value(moved, value);
    Value reset = copy;
    assert(reset.type == VT_BIGINT && reset.i == 0);
    moved = moved;
    equal_value(moved, value);
    StoredValue *alias = &moved;
    moved = std::move(*alias);
    equal_value(moved, value);
    for (const auto &replacement : samples) {
      StoredValue assigned(replacement);
      assigned = moved;  // All 8 x 8 ownership/type transitions.
      equal_value(assigned, value);
      assigned = replacement;
      equal_value(assigned, replacement);
      StoredValue transfer(value);
      assigned = std::move(transfer);
      equal_value(assigned, value);
      transfer = replacement;
      equal_value(transfer, replacement);
    }
  }

  Row row;
  for (const auto &value : samples) row.vals.emplace_back(value);
  Row clone = row;
  row.vals.clear();
  for (size_t i = 0; i < samples.size(); ++i) equal_value(clone.vals[i], samples[i]);
  // Reallocation repeatedly moves string/vector owners and then destroys them.
  std::vector<Row> rows;
  for (int i = 0; i < 300; ++i) rows.push_back(clone);
  rows.erase(rows.begin() + 2, rows.begin() + 290);
  for (const auto &stored : rows)
    for (size_t i = 0; i < samples.size(); ++i) equal_value(stored.vals[i], samples[i]);
}

static void snapshot_checks() {
  char temporary[] = "/tmp/mvdb-stored-test-XXXXXX";
  char *created = ::mkdtemp(temporary);
  assert(created);
  const std::string directory(created);
  const std::string long_text = std::string(1024, 'z') + "中文";
  const std::string original = "最初的文本";
  std::string err;
  {
    DB db;
    assert(db.open(directory, err));
    db.exec_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(4096), v VECTOR(3));");
    db.exec_sql("INSERT INTO t VALUES (1,'" + original + "','[1.25,-2.5,3.75]'),"
                "(2,'" + long_text + "','[0,0,0]');");
    db.exec_sql("BEGIN;");
    db.exec_sql("UPDATE t SET s='temporary',v='[9,8,7]' WHERE id=2;");
    db.exec_sql("DELETE FROM t WHERE id=1;");
    db.exec_sql("ROLLBACK;");
    Result restored = db.exec_sql("SELECT id,s,v FROM t ORDER BY id;");
    assert(restored.rows.size() == 2);
    assert(restored.rows[0][1].s == original && restored.rows[1][1].s == long_text);
    assert(restored.rows[1][2].v == std::vector<float>({0, 0, 0}));
    db.close();
  }
  {
    DB replay;
    assert(replay.open(directory, err));  // Replay WAL, then persist a snapshot.
    assert(fs::file_size(fs::path(directory) / "snapshot.bin") > long_text.size());
    assert(fs::file_size(fs::path(directory) / "wal.bin") == 0);
    replay.close();
  }
  {
    DB snapshot;
    assert(snapshot.open(directory, err));  // Read that snapshot with no WAL.
    Result result = snapshot.exec_sql("SELECT id,s,v FROM t ORDER BY id;");
    assert(result.rows.size() == 2 && result.rows[0][0].i == 1 && result.rows[1][0].i == 2);
    assert(result.rows[0][1].s == original && result.rows[1][1].s == long_text);
    assert(result.rows[0][2].v == std::vector<float>({1.25f, -2.5f, 3.75f}));
    assert(result.rows[1][2].v == std::vector<float>({0, 0, 0}));
    snapshot.close();
  }
  fs::remove_all(directory);
}

int main() {
  ownership_checks();
  snapshot_checks();
  std::cout << "PASS compact stored value ownership and varchar/vector snapshot reload\n";
}

// MiniVectorDB — OceanBase 2026 database competition playoff (question 600081).
// Single-process, single-client, stdin/stdout SQL engine with exact L2 TopK,
// simple transactions, /data persistence and crash recovery.
//
// The engine is intentionally self-contained: the SQL parser, row store,
// vector search and transaction/WAL layers are implemented here.  Only the
// C++ standard library is used.
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------
static std::string trim(const std::string &s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace((unsigned char)s[a])) ++a;
  while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
  return s.substr(a, b - a);
}

static std::string upper(std::string s) {
  for (char &c : s) c = (char)std::toupper((unsigned char)c);
  return s;
}

static bool ieq(const std::string &a, const char *b) { return upper(a) == upper(std::string(b)); }

static bool starts_with_ci(const std::string &s, const char *p) {
  size_t n = std::strlen(p);
  if (s.size() < n) return false;
  for (size_t i = 0; i < n; ++i)
    if (std::toupper((unsigned char)s[i]) != std::toupper((unsigned char)p[i]))
      return false;
  return true;
}

static bool like_match(const std::string &s, const std::string &p) {
  size_t si = 0, pi = 0;
  size_t star = std::string::npos, match = 0;
  while (si < s.size()) {
    if (pi < p.size() && (p[pi] == '_' || p[pi] == s[si])) {
      ++si;
      ++pi;
    } else if (pi < p.size() && p[pi] == '%') {
      star = pi++;
      match = si;
    } else if (star != std::string::npos) {
      pi = star + 1;
      si = ++match;
    } else {
      return false;
    }
  }
  while (pi < p.size() && p[pi] == '%') ++pi;
  return pi == p.size();
}

struct SqlError : std::runtime_error {
  int code;
  std::string state;
  SqlError(int c, std::string st, const std::string &msg)
      : std::runtime_error(msg), code(c), state(std::move(st)) {}
};

static SqlError syntax(const std::string &m) { return SqlError(1064, "42000", m); }

// CRC32 (IEEE) over arbitrary byte ranges.
static uint32_t crc32_update(uint32_t crc, const void *data, size_t n) {
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      table[i] = c;
    }
    init = true;
  }
  const uint8_t *p = (const uint8_t *)data;
  for (size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
  return crc;
}

// Binary file helpers (little-endian host is assumed; the evaluation is x86_64).
static bool put_bytes(FILE *f, const void *p, size_t n) {
  return n == 0 || std::fwrite(p, 1, n, f) == n;
}
static bool get_bytes(FILE *f, void *p, size_t n) {
  return n == 0 || std::fread(p, 1, n, f) == n;
}
static bool put_u32(FILE *f, uint32_t v) { return put_bytes(f, &v, 4); }
static bool put_u64(FILE *f, uint64_t v) { return put_bytes(f, &v, 8); }
static bool put_i32(FILE *f, int32_t v) { return put_bytes(f, &v, 4); }
static bool put_u8(FILE *f, uint8_t v) { return put_bytes(f, &v, 1); }
static bool get_u32(FILE *f, uint32_t &v) { return get_bytes(f, &v, 4); }
static bool get_u64(FILE *f, uint64_t &v) { return get_bytes(f, &v, 8); }
static bool get_i32(FILE *f, int32_t &v) { return get_bytes(f, &v, 4); }
static bool get_u8(FILE *f, uint8_t &v) { return get_bytes(f, &v, 1); }

static bool put_str(FILE *f, const std::string &s) {
  if (s.size() > 0xFFFFFFFFu) return false;
  return put_u32(f, (uint32_t)s.size()) && put_bytes(f, s.data(), s.size());
}
static bool get_str(FILE *f, std::string &s) {
  uint32_t n = 0;
  if (!get_u32(f, n) || n > 64u * 1024u * 1024u) return false;
  s.resize(n);
  return get_bytes(f, n ? &s[0] : nullptr, n);
}

static void fsync_dir(const std::string &dir) {
  int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
  if (fd >= 0) {
    ::fsync(fd);
    ::close(fd);
  }
}

// ---------------------------------------------------------------------------
// Values, rows and tables
// ---------------------------------------------------------------------------
enum ValueType { VT_BIGINT = 0, VT_VARCHAR = 1, VT_VECTOR = 2, VT_DOUBLE = 3 };

struct Value {
  int type = VT_BIGINT;
  int64_t i = 0;
  double d = 0.0;
  std::string s;
  // Owned vector (SQL literal / query materialization).
  std::vector<float> v;
  // Stored vector reference: valid when vdisk is true; `v` stays empty.
  uint64_t voff = 0;
  uint32_t vlen = 0;
  bool vdisk = false;
};

struct Column {
  std::string name;
  int type = VT_BIGINT;
  int len = 0;  // varchar char limit or vector dimension
};

// File-backed vector storage.  The vector file is a derived cache: the
// snapshot/WAL remain the source of truth and rebuild it on startup.  Small
// files are mmap'd for query speed; after MMAP_LIMIT bytes the store switches
// to pread/pwrite so a large dataset does not inflate anonymous RSS.
class VectorStore {
 public:
  VectorStore() = default;
  ~VectorStore() { release(); }
  VectorStore(const VectorStore &) = delete;
  VectorStore &operator=(const VectorStore &) = delete;
  VectorStore(VectorStore &&o) noexcept { steal(o); }
  VectorStore &operator=(VectorStore &&o) noexcept {
    if (this != &o) {
      release();
      steal(o);
    }
    return *this;
  }

  void init(const std::string &dir, const std::string &table, int dim) {
    release();
    dim_ = dim;
    if (dim_ <= 0) throw SqlError(1105, "HY000", "invalid vector dimension");
    path_ = (fs::path(dir) / ("vec_" + hexname(table) + ".bin")).string();
    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd_ < 0)
      throw SqlError(1105, "HY000", "cannot create vector store " + path_);
    end_ = 0;
    mapped_ = 0;
    chunks_.clear();
  }

  // Copy an owned vector into the file and return a stored reference.
  Value store(const Value &v) {
    if (v.type != VT_VECTOR)
      throw SqlError(1366, "22007", "Incorrect vector value");
    if (v.vdisk) {
      if ((int)v.vlen != dim_)
        throw SqlError(1366, "22007", "Vector dimension mismatch");
      return v;
    }
    if ((int)v.v.size() != dim_)
      throw SqlError(1366, "22007", "Vector dimension mismatch");
    size_t bytes = (size_t)dim_ * sizeof(float);
    uint64_t off = end_;
    if (off + bytes <= kMmapLimit) {
      size_t ci = (size_t)(off / chunk_);
      size_t within = (size_t)(off % chunk_);
      while (chunks_.size() <= ci) {
        uint64_t coff = (uint64_t)chunks_.size() * chunk_;
        if (::ftruncate(fd_, (off_t)(coff + chunk_)) != 0)
          throw SqlError(1105, "HY000", "vector store ftruncate failed");
        void *p = ::mmap(nullptr, chunk_, PROT_READ | PROT_WRITE, MAP_SHARED,
                         fd_, (off_t)coff);
        if (p == MAP_FAILED)
          throw SqlError(1105, "HY000", "vector store mmap failed");
        chunks_.push_back(p);
      }
      std::memcpy((char *)chunks_[ci] + within, v.v.data(), bytes);
      mapped_ = std::max(mapped_, off + bytes);
    } else {
      ssize_t w = ::pwrite(fd_, v.v.data(), bytes, (off_t)off);
      if (w != (ssize_t)bytes)
        throw SqlError(1105, "HY000", "vector store write failed");
    }
    end_ = off + bytes;
    Value out;
    out.type = VT_VECTOR;
    out.vdisk = true;
    out.voff = off;
    out.vlen = (uint32_t)dim_;
    return out;
  }

  const float *get(const Value &v, std::vector<float> &scratch) const {
    size_t bytes = (size_t)dim_ * sizeof(float);
    if (!v.vdisk) {
      if ((int)v.v.size() != dim_)
        throw SqlError(1366, "22007", "Vector dimension mismatch");
      return v.v.data();
    }
    if (v.voff + bytes <= mapped_ && !chunks_.empty()) {
      size_t ci = (size_t)(v.voff / chunk_);
      size_t within = (size_t)(v.voff % chunk_);
      if (ci < chunks_.size())
        return (const float *)((const char *)chunks_[ci] + within);
    }
    scratch.resize((size_t)dim_);
    ssize_t r = ::pread(fd_, scratch.data(), bytes, (off_t)v.voff);
    if (r != (ssize_t)bytes)
      throw SqlError(1105, "HY000", "vector store read failed");
    return scratch.data();
  }

  Value materialize(const Value &v) const {
    if (!v.vdisk) return v;
    std::vector<float> scratch;
    const float *p = get(v, scratch);
    Value out;
    out.type = VT_VECTOR;
    out.v.assign(p, p + dim_);
    return out;
  }

  void release() {
    for (void *p : chunks_)
      if (p && p != MAP_FAILED) ::munmap(p, chunk_);
    chunks_.clear();
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    if (!path_.empty()) {
      ::unlink(path_.c_str());
      path_.clear();
    }
    end_ = 0;
    mapped_ = 0;
    dim_ = 0;
  }

 private:
  static constexpr uint64_t kMmapLimit = 64ull * 1024 * 1024;
  size_t chunk_ = 64ull * 1024 * 1024;
  int fd_ = -1;
  int dim_ = 0;
  uint64_t end_ = 0;
  uint64_t mapped_ = 0;
  std::vector<void *> chunks_;
  std::string path_;

  static std::string hexname(const std::string &s) {
    static const char *h = "0123456789abcdef";
    std::string r;
    for (unsigned char c : s) {
      r.push_back(h[c >> 4]);
      r.push_back(h[c & 15]);
    }
    if (r.empty()) r = "empty";
    return r;
  }

  void steal(VectorStore &o) noexcept {
    fd_ = o.fd_;
    dim_ = o.dim_;
    end_ = o.end_;
    mapped_ = o.mapped_;
    chunk_ = o.chunk_;
    chunks_ = std::move(o.chunks_);
    path_ = std::move(o.path_);
    o.fd_ = -1;
    o.dim_ = 0;
    o.end_ = 0;
    o.mapped_ = 0;
    o.chunks_.clear();
    o.path_.clear();
  }
};

struct Row {
  std::vector<Value> vals;
  bool deleted = false;
};

struct Table {
  std::string name;
  std::vector<Column> cols;
  int pk = -1;
  int vec = -1;
  std::vector<Row> rows;
  std::unordered_map<int64_t, size_t> pkmap;
  VectorStore vecs;
};

static std::string fmt_double(double d) {
  if (!std::isfinite(d)) return "NULL";
  char b[96];
  std::snprintf(b, sizeof(b), "%.6f", d);
  return b;
}

static std::string fmt_vector(const std::vector<float> &v) {
  std::string r = "[";
  char b[64];
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) r.push_back(',');
    if (!std::isfinite(v[i])) {
      r += "nan";
    } else {
      std::snprintf(b, sizeof(b), "%.7g", (double)v[i]);
      r += b;
    }
  }
  r.push_back(']');
  return r;
}

static std::string fmt_value(const Value &v) {
  switch (v.type) {
    case VT_BIGINT:
      return std::to_string(v.i);
    case VT_VARCHAR:
      return v.s;
    case VT_VECTOR:
      return v.vdisk ? std::string("[...]") : fmt_vector(v.v);
    case VT_DOUBLE:
      return fmt_double(v.d);
    default:
      return "";
  }
}

// Count Unicode codepoints in a valid UTF-8 string.  Invalid bytes count as one.
static size_t utf8_len(const std::string &s) {
  size_t n = 0;
  for (size_t i = 0; i < s.size();) {
    unsigned char c = (unsigned char)s[i];
    size_t step = 1;
    if ((c & 0x80) == 0) step = 1;
    else if ((c & 0xE0) == 0xC0) step = 2;
    else if ((c & 0xF0) == 0xE0) step = 3;
    else if ((c & 0xF8) == 0xF0) step = 4;
    if (i + step > s.size()) step = 1;
    i += step;
    ++n;
  }
  return n;
}

static bool parse_vector_string(const std::string &text, std::vector<float> &out,
                                std::string &err) {
  std::string s = trim(text);
  if (s.size() < 2 || s.front() != '[' || s.back() != ']') {
    err = "vector literal must be [v1,v2,...]";
    return false;
  }
  out.clear();
  size_t i = 1;
  while (i + 1 < s.size()) {
    while (i + 1 < s.size() && (std::isspace((unsigned char)s[i]) || s[i] == ',')) ++i;
    if (i + 1 >= s.size()) break;
    char *end = nullptr;
    errno = 0;
    double d = std::strtod(s.c_str() + i, &end);
    if (end == s.c_str() + i || errno == ERANGE) {
      err = "invalid vector component";
      return false;
    }
    if (!std::isfinite(d) || std::fabs(d) > FLT_MAX) {
      err = "vector component is not a finite float32";
      return false;
    }
    out.push_back((float)d);
    i = (size_t)(end - s.c_str());
    while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
    if (i < s.size() && s[i] == ',') ++i;
  }
  return true;
}

static int value_cmp(const Value &a, const Value &b) {
  if ((a.type == VT_BIGINT || a.type == VT_DOUBLE) &&
      (b.type == VT_BIGINT || b.type == VT_DOUBLE)) {
    double x = a.type == VT_BIGINT ? (double)a.i : a.d;
    double y = b.type == VT_BIGINT ? (double)b.i : b.d;
    if (x < y) return -1;
    if (x > y) return 1;
    return 0;
  }
  if (a.type == VT_VARCHAR && b.type == VT_VARCHAR) {
    if (a.s < b.s) return -1;
    if (a.s > b.s) return 1;
    return 0;
  }
  std::string x = fmt_value(a), y = fmt_value(b);
  if (x < y) return -1;
  if (x > y) return 1;
  return 0;
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------
struct Token {
  enum Kind { IDENT, NUMBER, STRING, SYMBOL, END } kind = END;
  std::string text;
};

class Lexer {
 public:
  explicit Lexer(const std::string &s) : s_(s) {}

  Token next() {
    skip();
    if (i_ >= s_.size()) return {Token::END, ""};
    char c = s_[i_];
    if (std::isalpha((unsigned char)c) || c == '_') {
      size_t b = i_;
      while (i_ < s_.size() &&
             (std::isalnum((unsigned char)s_[i_]) || s_[i_] == '_' || s_[i_] == '$'))
        ++i_;
      return {Token::IDENT, s_.substr(b, i_ - b)};
    }
    if (std::isdigit((unsigned char)c) ||
        (c == '.' && i_ + 1 < s_.size() && std::isdigit((unsigned char)s_[i_ + 1]))) {
      size_t b = i_;
      bool dot = false, exp = false;
      while (i_ < s_.size()) {
        char x = s_[i_];
        if (std::isdigit((unsigned char)x)) {
          ++i_;
        } else if (x == '.' && !dot && !exp) {
          dot = true;
          ++i_;
        } else if ((x == 'e' || x == 'E') && !exp) {
          exp = true;
          ++i_;
          if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
        } else {
          break;
        }
      }
      return {Token::NUMBER, s_.substr(b, i_ - b)};
    }
    if (c == '\'' || c == '"' || c == '`') {
      char q = c;
      ++i_;
      std::string out;
      while (i_ < s_.size()) {
        char x = s_[i_++];
        if (x == '\\' && q != '`' && i_ < s_.size()) {
          char e = s_[i_++];
          switch (e) {
            case 'n': out.push_back('\n'); break;
            case 't': out.push_back('\t'); break;
            case 'r': out.push_back('\r'); break;
            case '0': out.push_back('\0'); break;
            default: out.push_back(e); break;
          }
          continue;
        }
        if (x == q) {
          if (i_ < s_.size() && s_[i_] == q) {
            out.push_back(q);
            ++i_;
            continue;
          }
          break;
        }
        out.push_back(x);
      }
      return {q == '`' ? Token::IDENT : Token::STRING, out};
    }
    // multi-character symbols
    if (i_ + 1 < s_.size()) {
      std::string two = s_.substr(i_, 2);
      if (two == "<=" || two == ">=" || two == "<>" || two == "!=") {
        i_ += 2;
        return {Token::SYMBOL, two};
      }
    }
    ++i_;
    return {Token::SYMBOL, std::string(1, c)};
  }

 private:
  void skip() {
    for (;;) {
      while (i_ < s_.size() && std::isspace((unsigned char)s_[i_])) ++i_;
      if (i_ + 1 < s_.size() && s_[i_] == '-' && s_[i_ + 1] == '-' &&
          (i_ + 2 >= s_.size() || std::isspace((unsigned char)s_[i_ + 2]))) {
        while (i_ < s_.size() && s_[i_] != '\n') ++i_;
        continue;
      }
      if (i_ < s_.size() && s_[i_] == '#') {
        while (i_ < s_.size() && s_[i_] != '\n') ++i_;
        continue;
      }
      if (i_ + 1 < s_.size() && s_[i_] == '/' && s_[i_ + 1] == '*') {
        i_ += 2;
        while (i_ + 1 < s_.size() && !(s_[i_] == '*' && s_[i_ + 1] == '/')) ++i_;
        if (i_ + 1 < s_.size()) i_ += 2;
        continue;
      }
      break;
    }
  }

  const std::string &s_;
  size_t i_ = 0;
};


// ---------------------------------------------------------------------------
// Structured execution result and JSONL protocol helpers
// ---------------------------------------------------------------------------
struct Result {
  bool query = false;
  std::vector<std::string> columns;
  std::vector<std::vector<Value>> rows;
  bool has_affected = false;
  int64_t affected = 0;
};

static Value vstr(const std::string &s) {
  Value v;
  v.type = VT_VARCHAR;
  v.s = s;
  return v;
}

static std::string ok_text(int64_t n) {
  if (n == 1) return "Query OK, 1 row affected";
  return "Query OK, " + std::to_string(n) + " rows affected";
}

static std::string format_legacy(const Result &r) {
  if (r.query) {
    std::string out;
    for (size_t i = 0; i < r.columns.size(); ++i) {
      if (i) out.push_back('\t');
      out += r.columns[i];
    }
    out.push_back('\n');
    for (const auto &row : r.rows) {
      for (size_t i = 0; i < row.size(); ++i) {
        if (i) out.push_back('\t');
        out += fmt_value(row[i]);
      }
      out.push_back('\n');
    }
    return out;
  }
  if (r.has_affected) return ok_text(r.affected);
  return ok_text(0);
}

static std::string json_escape(const std::string &s) {
  std::string o;
  o.reserve(s.size() + 2);
  o.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\b': o += "\\b"; break;
      case '\f': o += "\\f"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char b[8];
          std::snprintf(b, sizeof b, "\\u%04x", c);
          o += b;
        } else {
          o.push_back((char)c);
        }
    }
  }
  o.push_back('"');
  return o;
}

static std::string json_double(double d) {
  if (!std::isfinite(d)) return "0";
  char b[64];
  std::snprintf(b, sizeof b, "%.9g", d);
  return b;
}

static std::string json_value(const Value &v) {
  switch (v.type) {
    case VT_BIGINT: return std::to_string(v.i);
    case VT_VARCHAR: return json_escape(v.s);
    case VT_DOUBLE: return json_double(v.d);
    case VT_VECTOR: {
      std::string o = "[";
      for (size_t i = 0; i < v.v.size(); ++i) {
        if (i) o.push_back(',');
        o += json_double(v.v[i]);
      }
      o.push_back(']');
      return o;
    }
  }
  return "null";
}

static std::string json_result_body(const Result &r) {
  std::string o;
  if (r.query) {
    o += ",\"columns\":[";
    for (size_t i = 0; i < r.columns.size(); ++i) {
      if (i) o.push_back(',');
      o += json_escape(r.columns[i]);
    }
    o += "],\"rows\":[";
    for (size_t i = 0; i < r.rows.size(); ++i) {
      if (i) o.push_back(',');
      o.push_back('[');
      for (size_t j = 0; j < r.rows[i].size(); ++j) {
        if (j) o.push_back(',');
        o += json_value(r.rows[i][j]);
      }
      o.push_back(']');
    }
    o.push_back(']');
  }
  if (r.has_affected) o += ",\"affected_rows\":" + std::to_string(r.affected);
  return o;
}

// Minimal JSON request parser for the JSONL protocol.
class JsonReader {
 public:
  explicit JsonReader(const std::string &s) : s_(s) {}

  void ws() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\r' || s_[i_] == '\n')) ++i_;
  }
  bool eat(char c) {
    ws();
    if (i_ < s_.size() && s_[i_] == c) { ++i_; return true; }
    return false;
  }
  bool parse_string(std::string &out) {
    ws();
    if (i_ >= s_.size() || s_[i_] != '"') return false;
    ++i_;
    out.clear();
    while (i_ < s_.size()) {
      unsigned char c = (unsigned char)s_[i_++];
      if (c == '"') return true;
      if (c != '\\') { out.push_back((char)c); continue; }
      if (i_ >= s_.size()) return false;
      char e = s_[i_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          if (i_ + 4 > s_.size()) return false;
          unsigned cp = 0;
          for (int k = 0; k < 4; ++k) {
            char h = s_[i_++];
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
            else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
            else return false;
          }
          if (cp >= 0xD800 && cp <= 0xDBFF && i_ + 6 <= s_.size() &&
              s_[i_] == '\\' && s_[i_ + 1] == 'u') {
            unsigned lo = 0;
            size_t save = i_;
            i_ += 2;
            bool ok = true;
            for (int k = 0; k < 4; ++k) {
              char h = s_[i_++];
              lo <<= 4;
              if (h >= '0' && h <= '9') lo |= (unsigned)(h - '0');
              else if (h >= 'a' && h <= 'f') lo |= (unsigned)(h - 'a' + 10);
              else if (h >= 'A' && h <= 'F') lo |= (unsigned)(h - 'A' + 10);
              else { ok = false; break; }
            }
            if (ok && lo >= 0xDC00 && lo <= 0xDFFF)
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            else
              i_ = save;
          }
          if (cp <= 0x7F) out.push_back((char)cp);
          else if (cp <= 0x7FF) {
            out.push_back((char)(0xC0 | (cp >> 6)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
          } else if (cp <= 0xFFFF) {
            out.push_back((char)(0xE0 | (cp >> 12)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
          } else {
            out.push_back((char)(0xF0 | (cp >> 18)));
            out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
          }
          break;
        }
        default: return false;
      }
    }
    return false;
  }
  bool parse_raw(std::string &out) {
    ws();
    size_t start = i_;
    if (i_ >= s_.size()) return false;
    if (s_[i_] == '"') {
      std::string tmp;
      if (!parse_string(tmp)) return false;
      out = s_.substr(start, i_ - start);
      return true;
    }
    while (i_ < s_.size()) {
      char c = s_[i_];
      if (c == ',' || c == '}' || c == ']' || c == ' ' || c == '\t' || c == '\r' || c == '\n') break;
      ++i_;
    }
    if (i_ == start) return false;
    out = s_.substr(start, i_ - start);
    return true;
  }
  bool skip_value() {
    ws();
    if (i_ >= s_.size()) return false;
    if (s_[i_] == '"') { std::string t; return parse_string(t); }
    if (s_[i_] == '{' || s_[i_] == '[') {
      char open = s_[i_++], close = open == '{' ? '}' : ']';
      int depth = 1;
      while (i_ < s_.size() && depth > 0) {
        char c = s_[i_++];
        if (c == '"') { --i_; std::string t; if (!parse_string(t)) return false; continue; }
        if (c == open) ++depth;
        else if (c == close) --depth;
      }
      return depth == 0;
    }
    std::string t;
    return parse_raw(t);
  }

 private:
  const std::string &s_;
  size_t i_ = 0;
};

struct JRequest {
  std::string id_raw;
  std::string op;
  std::string sql;
  std::string search_mode;
};

static bool parse_request(const std::string &line, JRequest &req, std::string &err) {
  JsonReader r(line);
  r.ws();
  if (!r.eat('{')) { err = "request must be a JSON object"; return false; }
  bool first = true;
  while (true) {
    r.ws();
    if (r.eat('}')) return true;
    if (!first) {
      if (!r.eat(',')) { err = "expected ','"; return false; }
      r.ws();
    }
    first = false;
    std::string key;
    if (!r.parse_string(key)) { err = "expected key"; return false; }
    if (!r.eat(':')) { err = "expected ':'"; return false; }
    if (key == "id") {
      if (!r.parse_raw(req.id_raw)) { err = "invalid id"; return false; }
    } else if (key == "op") {
      if (!r.parse_string(req.op)) { err = "invalid op"; return false; }
    } else if (key == "sql") {
      if (!r.parse_string(req.sql)) { err = "invalid sql"; return false; }
    } else if (key == "search_mode") {
      if (!r.parse_string(req.search_mode)) { err = "invalid search_mode"; return false; }
    } else {
      if (!r.skip_value()) { err = "invalid value"; return false; }
    }
  }
}

static std::string protocol_error_code(const SqlError &e) {
  if (e.code == 1062) return "duplicate_key";
  if (e.code == 1064) return "unsupported_sql";
  if (e.code == 1366) {
    std::string m = e.what();
    if (m.find("dimension") != std::string::npos || m.find("Vector") != std::string::npos)
      return "dimension_mismatch";
    return "invalid_value";
  }
  if (e.code == 1406) return "value_too_long";
  if (e.code == 1146) return "table_not_found";
  if (e.code == 1054) return "column_not_found";
  if (e.code == 1050) return "table_exists";
  if (e.code == 1568) return "transaction_conflict";
  return "sql_error";
}

// ---------------------------------------------------------------------------
// AST and parser
// ---------------------------------------------------------------------------
struct Expr;
using ExprP = std::shared_ptr<Expr>;

struct Expr {
  enum Kind { COL, LIT, L2, CMP, AND, OR, NOT, IN, BETWEEN } kind = LIT;
  std::string name;  // column name or comparison operator
  Value lit;
  std::vector<float> vec;
  ExprP a, b, c;
  std::vector<ExprP> list;
};

struct SelectItem {
  ExprP expr;
  std::string alias;
};

struct OrderItem {
  ExprP expr;
  bool asc = true;
};

struct Stmt {
  enum Kind {
    CREATE_TABLE,
    INSERT,
    SELECT,
    UPDATE,
    DELETE_,
    BEGIN,
    COMMIT,
    ROLLBACK,
    CREATE_INDEX,
    DROP_INDEX,
    DROP_TABLE,
    SHOW_TABLES,
    DESCRIBE_TABLE,
    TRUNCATE,
    OTHER
  } kind = OTHER;

  std::string table;
  std::vector<Column> cols;
  int pk = -1;
  int vec = -1;
  bool if_not_exists = false;
  bool if_exists = false;
  std::string like;
  std::string index_name, index_col;

  std::vector<std::string> insert_cols;
  std::vector<std::vector<ExprP>> insert_rows;

  std::vector<SelectItem> items;
  bool star = false;
  ExprP where;
  std::vector<OrderItem> order;
  int64_t limit = -1;
  int64_t offset = 0;

  std::vector<std::pair<std::string, ExprP>> sets;
};

class Parser {
 public:
  explicit Parser(const std::string &sql) : lex_(sql), tok_(lex_.next()) {}
  Stmt parse() { return parse_statement(); }

 private:
  Lexer lex_;
  Token tok_;

  void adv() { tok_ = lex_.next(); }
  bool is_kw(const char *k) const {
    return tok_.kind == Token::IDENT && upper(tok_.text) == k;
  }
  bool match_kw(const char *k) {
    if (is_kw(k)) {
      adv();
      return true;
    }
    return false;
  }
  void expect_kw(const char *k) {
    if (!match_kw(k)) throw syntax(std::string("expected ") + k);
  }
  bool is_sym(const char *s) const { return tok_.kind == Token::SYMBOL && tok_.text == s; }
  bool match_sym(const char *s) {
    if (is_sym(s)) {
      adv();
      return true;
    }
    return false;
  }
  void expect_sym(const char *s) {
    if (!match_sym(s)) throw syntax(std::string("expected '") + s + "'");
  }
  std::string ident() {
    if (tok_.kind != Token::IDENT) throw syntax("expected identifier");
    std::string s = tok_.text;
    adv();
    return s;
  }
  int64_t int_lit() {
    if (tok_.kind != Token::NUMBER) throw syntax("expected integer");
    std::string t = tok_.text;
    adv();
    char *end = nullptr;
    errno = 0;
    long long v = std::strtoll(t.c_str(), &end, 10);
    if (end == t.c_str() || *end != '\0' || errno == ERANGE)
      throw syntax("invalid integer");
    return v;
  }

  Stmt parse_statement() {
    Stmt st;
    if (tok_.kind == Token::END) return st;
    if (match_kw("CREATE")) {
      if (match_kw("DATABASE")) {
        if (match_kw("IF")) {
          expect_kw("NOT");
          expect_kw("EXISTS");
        }
        ident();
        st.kind = Stmt::OTHER;
        return st;
      }
      if (match_kw("TABLE")) return parse_create_table();
      if (match_kw("VECTOR")) {
        expect_kw("INDEX");
        st.kind = Stmt::CREATE_INDEX;
        if (match_kw("IF")) {
          expect_kw("NOT");
          expect_kw("EXISTS");
        }
        st.index_name = ident();
        expect_kw("ON");
        st.table = ident();
        expect_sym("(");
        st.index_col = ident();
        expect_sym(")");
        return st;
      }
      if (match_kw("INDEX")) {
        st.kind = Stmt::CREATE_INDEX;
        if (match_kw("IF")) {
          expect_kw("NOT");
          expect_kw("EXISTS");
        }
        st.index_name = ident();
        expect_kw("ON");
        st.table = ident();
        expect_sym("(");
        st.index_col = ident();
        expect_sym(")");
        return st;
      }
      throw syntax("unsupported CREATE statement");
    }
    if (match_kw("DROP")) {
      if (match_kw("DATABASE")) {
        if (match_kw("IF")) expect_kw("EXISTS");
        ident();
        st.kind = Stmt::OTHER;
        return st;
      }
      if (match_kw("INDEX")) {
        st.kind = Stmt::DROP_INDEX;
        if (match_kw("IF")) expect_kw("EXISTS");
        st.index_name = ident();
        expect_kw("ON");
        st.table = ident();
        return st;
      }
      expect_kw("TABLE");
      st.kind = Stmt::DROP_TABLE;
      if (match_kw("IF")) {
        expect_kw("EXISTS");
        st.if_exists = true;
      }
      st.table = ident();
      return st;
    }
    if (match_kw("INSERT")) return parse_insert();
    if (match_kw("SELECT")) return parse_select();
    if (match_kw("UPDATE")) return parse_update();
    if (match_kw("DELETE")) {
      expect_kw("FROM");
      st.kind = Stmt::DELETE_;
      st.table = ident();
      if (match_kw("WHERE")) st.where = parse_expr();
      parse_order_by(st);
      if (match_kw("LIMIT")) st.limit = int_lit();
      return st;
    }
    if (match_kw("BEGIN")) {
      st.kind = Stmt::BEGIN;
      return st;
    }
    if (match_kw("START")) {
      expect_kw("TRANSACTION");
      st.kind = Stmt::BEGIN;
      return st;
    }
    if (match_kw("COMMIT")) {
      st.kind = Stmt::COMMIT;
      return st;
    }
    if (match_kw("ROLLBACK")) {
      st.kind = Stmt::ROLLBACK;
      return st;
    }
    if (match_kw("DESCRIBE") || match_kw("DESC")) {
      st.kind = Stmt::DESCRIBE_TABLE;
      st.table = ident();
      return st;
    }
    if (match_kw("SHOW")) {
      if (match_kw("TABLES")) {
        st.kind = Stmt::SHOW_TABLES;
        if (match_kw("LIKE")) {
          if (tok_.kind != Token::STRING) throw syntax("expected string after LIKE");
          st.like = tok_.text;
          adv();
        }
        return st;
      }
      if (match_kw("COLUMNS")) {
        expect_kw("FROM");
        st.kind = Stmt::DESCRIBE_TABLE;
        st.table = ident();
        return st;
      }
      throw syntax("unsupported SHOW statement");
    }
    if (match_kw("TRUNCATE")) {
      match_kw("TABLE");
      st.kind = Stmt::TRUNCATE;
      st.table = ident();
      return st;
    }
    if (match_kw("SET") || match_kw("USE")) {
      // Session settings / USE are not part of the scored subset; accept them.
      while (tok_.kind != Token::END) adv();
      st.kind = Stmt::OTHER;
      return st;
    }
    // Unknown leading token: consume the rest so the REPL stays usable.
    throw syntax("unsupported statement");
  }

  Stmt parse_create_table() {
    Stmt st;
    st.kind = Stmt::CREATE_TABLE;
    if (match_kw("IF")) {
      expect_kw("NOT");
      expect_kw("EXISTS");
      st.if_not_exists = true;
    }
    st.table = ident();
    expect_sym("(");
    bool first = true;
    while (true) {
      if (tok_.kind == Token::END) throw syntax("unterminated CREATE TABLE");
      if (!first) {
        if (!match_sym(",")) break;
      }
      first = false;
      if (is_kw("PRIMARY")) {
        adv();
        expect_kw("KEY");
        expect_sym("(");
        std::string c = ident();
        expect_sym(")");
        for (size_t i = 0; i < st.cols.size(); ++i)
          if (ieq(st.cols[i].name, c.c_str())) st.pk = (int)i;
        if (st.pk < 0) throw syntax("PRIMARY KEY column not found");
        continue;
      }
      Column c;
      c.name = ident();
      if (tok_.kind != Token::IDENT) throw syntax("expected column type");
      std::string tp = upper(tok_.text);
      adv();
      if (tp == "BIGINT" || tp == "INT" || tp == "INTEGER") {
        c.type = VT_BIGINT;
        c.len = 0;
      } else if (tp == "VARCHAR") {
        expect_sym("(");
        c.len = (int)int_lit();
        expect_sym(")");
        c.type = VT_VARCHAR;
        if (c.len <= 0) throw syntax("VARCHAR length must be positive");
      } else if (tp == "VECTOR") {
        expect_sym("(");
        c.len = (int)int_lit();
        expect_sym(")");
        c.type = VT_VECTOR;
        if (c.len <= 0) throw syntax("VECTOR dimension must be positive");
      } else {
        throw syntax("unsupported column type " + tp);
      }
      while (true) {
        if (match_kw("PRIMARY")) {
          expect_kw("KEY");
          for (size_t i = 0; i < st.cols.size(); ++i)
            if (ieq(st.cols[i].name, c.name.c_str())) st.pk = (int)i;
          st.pk = (int)st.cols.size();
        } else if (match_kw("NOT")) {
          expect_kw("NULL");
        } else if (match_kw("UNIQUE")) {
          // accepted, not enforced beyond the primary key
        } else if (match_kw("AUTO_INCREMENT")) {
          // accepted for schema compatibility; explicit ids are still used
        } else {
          break;
        }
      }
      st.cols.push_back(c);
    }
    expect_sym(")");
    if (st.cols.empty()) throw syntax("table must have columns");
    int pk = st.pk;
    int vec = -1;
    int pkcount = 0;
    for (size_t i = 0; i < st.cols.size(); ++i) {
      if (st.cols[i].type == VT_VECTOR) {
        if (vec >= 0) throw syntax("at most one vector column");
        vec = (int)i;
      }
      if (st.cols[i].type == VT_BIGINT && pk < 0) {
        // no explicit PK yet
      }
    }
    if (pk < 0) throw syntax("a BIGINT PRIMARY KEY is required");
    for (size_t i = 0; i < st.cols.size(); ++i)
      if (st.cols[i].type == VT_BIGINT && (int)i == pk) ++pkcount;
    if (pkcount != 1) throw syntax("PRIMARY KEY must be a BIGINT column");
    st.vec = vec;
    return st;
  }

  Stmt parse_insert() {
    Stmt st;
    st.kind = Stmt::INSERT;
    expect_kw("INTO");
    st.table = ident();
    if (match_sym("(")) {
      do {
        st.insert_cols.push_back(ident());
      } while (match_sym(","));
      expect_sym(")");
    }
    expect_kw("VALUES");
    do {
      expect_sym("(");
      std::vector<ExprP> row;
      do {
        row.push_back(parse_primary());
      } while (match_sym(","));
      expect_sym(")");
      st.insert_rows.push_back(std::move(row));
    } while (match_sym(","));
    return st;
  }

  void parse_order_by(Stmt &st) {
    if (!match_kw("ORDER")) return;
    expect_kw("BY");
    do {
      OrderItem oi;
      oi.expr = parse_expr();
      if (match_kw("DESC")) oi.asc = false;
      else match_kw("ASC");
      st.order.push_back(std::move(oi));
    } while (match_sym(","));
  }

  Stmt parse_select() {
    Stmt st;
    st.kind = Stmt::SELECT;
    if (match_sym("*")) {
      st.star = true;
    } else {
      do {
        SelectItem it;
        it.expr = parse_expr();
        if (match_kw("AS")) {
          it.alias = ident();
        } else if (tok_.kind == Token::IDENT) {
          std::string u = upper(tok_.text);
          if (u != "FROM" && u != "WHERE" && u != "ORDER" && u != "LIMIT" &&
              u != "ASC" && u != "DESC" && u != "AND" && u != "OR") {
            it.alias = ident();
          }
        }
        st.items.push_back(std::move(it));
      } while (match_sym(","));
    }
    if (match_kw("FROM")) {
      st.table = ident();
      if (match_kw("WHERE")) st.where = parse_expr();
      parse_order_by(st);
    }
    if (match_kw("LIMIT")) {
      int64_t a = int_lit();
      if (match_sym(",")) {
        st.offset = a;
        st.limit = int_lit();
      } else if (match_kw("OFFSET")) {
        st.limit = a;
        st.offset = int_lit();
      } else {
        st.limit = a;
      }
      if (st.limit < 0 || st.offset < 0) throw syntax("LIMIT/OFFSET must be non-negative");
    }
    return st;
  }

  Stmt parse_update() {
    Stmt st;
    st.kind = Stmt::UPDATE;
    st.table = ident();
    expect_kw("SET");
    do {
      std::string c = ident();
      expect_sym("=");
      st.sets.emplace_back(c, parse_expr());
    } while (match_sym(","));
    if (match_kw("WHERE")) st.where = parse_expr();
    parse_order_by(st);
    if (match_kw("LIMIT")) st.limit = int_lit();
    return st;
  }

  ExprP parse_expr() { return parse_or(); }

  ExprP parse_or() {
    ExprP a = parse_and();
    while (match_kw("OR")) {
      auto e = std::make_shared<Expr>();
      e->kind = Expr::OR;
      e->a = a;
      e->b = parse_and();
      a = e;
    }
    return a;
  }

  ExprP parse_and() {
    ExprP a = parse_cmp();
    while (match_kw("AND")) {
      auto e = std::make_shared<Expr>();
      e->kind = Expr::AND;
      e->a = a;
      e->b = parse_cmp();
      a = e;
    }
    return a;
  }

  ExprP parse_cmp() {
    if (match_kw("NOT")) {
      auto e = std::make_shared<Expr>();
      e->kind = Expr::NOT;
      e->a = parse_cmp();
      return e;
    }
    ExprP a = parse_primary();
    if (match_kw("LIKE")) {
      auto e = std::make_shared<Expr>();
      e->kind = Expr::CMP;
      e->name = "LIKE";
      e->a = a;
      e->b = parse_primary();
      return e;
    }
    if (match_kw("IN")) {
      auto e = std::make_shared<Expr>();
      e->kind = Expr::IN;
      e->a = a;
      expect_sym("(");
      if (!match_sym(")")) {
        do {
          e->list.push_back(parse_primary());
        } while (match_sym(","));
        expect_sym(")");
      }
      return e;
    }
    if (match_kw("BETWEEN")) {
      auto e = std::make_shared<Expr>();
      e->kind = Expr::BETWEEN;
      e->a = a;
      e->b = parse_primary();
      expect_kw("AND");
      e->c = parse_primary();
      return e;
    }
    if (tok_.kind == Token::SYMBOL &&
        (tok_.text == "=" || tok_.text == "<>" || tok_.text == "!=" ||
         tok_.text == "<" || tok_.text == "<=" || tok_.text == ">" ||
         tok_.text == ">=")) {
      std::string op = tok_.text;
      adv();
      ExprP b = parse_primary();
      auto e = std::make_shared<Expr>();
      e->kind = Expr::CMP;
      e->name = op;
      e->a = a;
      e->b = b;
      return e;
    }
    return a;
  }

  ExprP parse_primary() {
    if (tok_.kind == Token::IDENT && ieq(tok_.text, "L2_DISTANCE")) {
      adv();
      expect_sym("(");
      std::string col = ident();
      expect_sym(",");
      std::vector<float> q;
      if (tok_.kind == Token::STRING) {
        std::string er;
        if (!parse_vector_string(tok_.text, q, er)) throw syntax(er);
        adv();
      } else if (match_sym("[")) {
        q = parse_number_vector();
      } else {
        throw syntax("expected vector literal");
      }
      expect_sym(")");
      auto e = std::make_shared<Expr>();
      e->kind = Expr::L2;
      e->name = col;
      e->vec = std::move(q);
      return e;
    }
    if (tok_.kind == Token::IDENT) {
      auto e = std::make_shared<Expr>();
      e->kind = Expr::COL;
      e->name = tok_.text;
      adv();
      return e;
    }
    if (tok_.kind == Token::NUMBER) {
      auto e = std::make_shared<Expr>();
      e->kind = Expr::LIT;
      std::string t = tok_.text;
      if (t.find('.') != std::string::npos || t.find('e') != std::string::npos ||
          t.find('E') != std::string::npos) {
        e->lit.type = VT_DOUBLE;
        e->lit.d = std::strtod(t.c_str(), nullptr);
      } else {
        errno = 0;
        char *end = nullptr;
        long long v = std::strtoll(t.c_str(), &end, 10);
        if (errno == ERANGE || end == t.c_str() || *end) throw syntax("invalid integer");
        e->lit.type = VT_BIGINT;
        e->lit.i = v;
      }
      adv();
      return e;
    }
    if (tok_.kind == Token::STRING) {
      auto e = std::make_shared<Expr>();
      e->kind = Expr::LIT;
      e->lit.type = VT_VARCHAR;
      e->lit.s = tok_.text;
      adv();
      return e;
    }
    if (is_sym("[")) {
      adv();
      auto e = std::make_shared<Expr>();
      e->kind = Expr::LIT;
      e->lit.type = VT_VECTOR;
      e->lit.v = parse_number_vector();
      return e;
    }
    if (match_sym("(")) {
      ExprP e = parse_expr();
      expect_sym(")");
      return e;
    }
    if (match_sym("-")) {
      ExprP e = parse_primary();
      if (e->kind == Expr::LIT && e->lit.type == VT_BIGINT) e->lit.i = -e->lit.i;
      else if (e->kind == Expr::LIT && e->lit.type == VT_DOUBLE) e->lit.d = -e->lit.d;
      else throw syntax("unary minus needs a numeric literal");
      return e;
    }
    throw syntax("unexpected token");
  }

  std::vector<float> parse_number_vector() {
    std::vector<float> v;
    if (match_sym("]")) return v;
    while (true) {
      bool neg = match_sym("-");
      if (tok_.kind != Token::NUMBER) throw syntax("expected vector number");
      char *end = nullptr;
      double d = std::strtod(tok_.text.c_str(), &end);
      if (end == tok_.text.c_str() || *end) throw syntax("invalid vector number");
      adv();
      if (!std::isfinite(d) || std::fabs(d) > FLT_MAX)
        throw syntax("vector component is not a finite float32");
      v.push_back((float)(neg ? -d : d));
      if (match_sym("]")) break;
      expect_sym(",");
    }
    return v;
  }
};

// ---------------------------------------------------------------------------
// Database engine
// ---------------------------------------------------------------------------
struct Undo {
  int op = 0;  // 1=insert, 2=update, 3=delete
  std::string table;
  int64_t pk = 0;
  size_t idx = 0;
  Row old;
};

class DB {
 public:
  DB() = default;
  ~DB() { close(); }

  bool open(const std::string &dir, std::string &err) {
    dir_ = dir;
    std::error_code ec;
    fs::create_directories(dir_, ec);
    if (!fs::exists(dir_)) {
      err = "cannot create data directory " + dir_;
      return false;
    }
    if (!load_snapshot(err)) return false;
    if (!replay_wal(err)) return false;
    recovering_ = false;
    if (!save_snapshot(err)) return false;
    truncate_wal_file();
    if (!open_wal_append(err)) return false;
    return true;
  }

  void close() {
    if (closed_) return;
    if (in_tx_) rollback_tx();
    closed_ = true;
    if (wal_) {
      std::fclose(wal_);
      wal_ = nullptr;
    }
    std::string err;
    // Keep the WAL if the snapshot could not be written; it is the only
    // durable copy of commits newer than the last snapshot.
    if (save_snapshot(err)) truncate_wal_file();
    for (auto &kv : tables_) kv.second.vecs.release();
  }

  // Execute one statement.  Returns a structured result for JSON/legacy output.
  // Throws SqlError for parse/execution errors.
  Result exec_sql(const std::string &raw, bool recovering = false) {
    Parser p(raw);
    Stmt st = p.parse();
    bool rec = recovering || recovering_;
    switch (st.kind) {
      case Stmt::BEGIN: {
        if (in_tx_) throw SqlError(1568, "25001", "transaction already in progress");
        in_tx_ = true;
        undo_.clear();
        tx_sql_.clear();
        return Result{};
      }
      case Stmt::COMMIT: {
        if (in_tx_) commit_tx();
        return Result{};
      }
      case Stmt::ROLLBACK: {
        if (in_tx_) rollback_tx();
        return Result{};
      }
      case Stmt::DESCRIBE_TABLE: {
        Table &t = table(st.table);
        Result r;
        r.query = true;
        r.columns = {"Field", "Type", "Null", "Key", "Default", "Extra"};
        for (size_t i = 0; i < t.cols.size(); ++i) {
          const Column &c = t.cols[i];
          std::string type;
          if (c.type == VT_BIGINT) type = "bigint";
          else if (c.type == VT_VARCHAR) type = "varchar(" + std::to_string(c.len) + ")";
          else if (c.type == VT_VECTOR) type = "vector(" + std::to_string(c.len) + ")";
          else type = "unknown";
          r.rows.push_back({vstr(c.name), vstr(type),
                            vstr((int)i == t.pk ? "NO" : "YES"),
                            vstr((int)i == t.pk ? "PRI" : ""),
                            vstr("NULL"), vstr("")});
        }
        return r;
      }
      case Stmt::SHOW_TABLES: {
        Result r;
        r.query = true;
        r.columns = {"Tables_in_test"};
        for (const auto &kv : tables_) {
          if (!st.like.empty() && !like_match(kv.first, st.like)) continue;
          r.rows.push_back({vstr(kv.first)});
        }
        return r;
      }
      case Stmt::TRUNCATE: {
        if (in_tx_) commit_tx();
        Table &t = table(st.table);
        if (!rec) wal_commit_one(raw);
        t.rows.clear();
        t.pkmap.clear();
        return Result{};
      }
      case Stmt::OTHER:
        return Result{};
      case Stmt::CREATE_TABLE:
        if (in_tx_) commit_tx();
        create_table(st, raw, rec);
        return Result{};
      case Stmt::DROP_TABLE:
        if (in_tx_) commit_tx();
        drop_table(st, raw, rec);
        return Result{};
      case Stmt::CREATE_INDEX:
        {
          auto &t = table(st.table);
          (void)t;
          col_index(t, st.index_col);
        }
        return Result{};
      case Stmt::DROP_INDEX:
        (void)table(st.table);
        return Result{};
      case Stmt::INSERT: {
        int64_t n = insert_rows(st, raw, rec);
        Result r;
        r.has_affected = true;
        r.affected = n;
        return r;
      }
      case Stmt::UPDATE: {
        int64_t n = update_rows(st, raw, rec);
        Result r;
        r.has_affected = true;
        r.affected = n;
        return r;
      }
      case Stmt::DELETE_: {
        int64_t n = delete_rows(st, raw, rec);
        Result r;
        r.has_affected = true;
        r.affected = n;
        return r;
      }
      case Stmt::SELECT:
        return select_rows(st);
    }
    return Result{};
  }

  bool recovering_ = false;

 private:
  std::string dir_;
  std::map<std::string, Table> tables_;
  bool in_tx_ = false;
  bool closed_ = false;
  std::vector<Undo> undo_;
  std::vector<std::string> tx_sql_;
  uint64_t seq_ = 0;
  uint64_t last_seq_ = 0;
  FILE *wal_ = nullptr;

  static std::string ok(int64_t n) {
    if (n == 1) return "Query OK, 1 row affected";
    return "Query OK, " + std::to_string(n) + " rows affected";
  }

  Table &table(const std::string &name) {
    auto it = tables_.find(name);
    if (it == tables_.end())
      throw SqlError(1146, "42S02", "Table '" + name + "' doesn't exist");
    return it->second;
  }

  int col_index(const Table &t, const std::string &name) {
    for (size_t i = 0; i < t.cols.size(); ++i)
      if (ieq(t.cols[i].name, name.c_str())) return (int)i;
    throw SqlError(1054, "42S22", "Unknown column '" + name + "'");
  }

  int64_t row_pk(const Table &t, const Row &r) const {
    return r.vals[(size_t)t.pk].i;
  }

  Value coerce(const Value &v, const Column &c) {
    if (c.type == VT_BIGINT) {
      if (v.type == VT_BIGINT) return v;
      if (v.type == VT_DOUBLE) {
        if (!std::isfinite(v.d) || v.d != std::floor(v.d) ||
            v.d < (double)std::numeric_limits<int64_t>::min() ||
            v.d > (double)std::numeric_limits<int64_t>::max())
          throw SqlError(1366, "22007", "Incorrect integer value");
        Value r;
        r.type = VT_BIGINT;
        r.i = (int64_t)v.d;
        return r;
      }
      if (v.type == VT_VARCHAR) {
        std::string t = trim(v.s);
        char *end = nullptr;
        errno = 0;
        long long x = std::strtoll(t.c_str(), &end, 10);
        if (!t.empty() && end && *end == '\0' && errno != ERANGE) {
          Value r;
          r.type = VT_BIGINT;
          r.i = (int64_t)x;
          return r;
        }
      }
      throw SqlError(1366, "22007", "Incorrect integer value");
    }
    if (c.type == VT_VARCHAR) {
      Value r;
      r.type = VT_VARCHAR;
      if (v.type == VT_VARCHAR) r.s = v.s;
      else if (v.type == VT_BIGINT) r.s = std::to_string(v.i);
      else if (v.type == VT_DOUBLE) r.s = fmt_double(v.d);
      else throw SqlError(1366, "22007", "Incorrect string value");
      if ((int)utf8_len(r.s) > c.len)
        throw SqlError(1406, "22001", "Data too long for column '" + c.name + "'");
      return r;
    }
    if (c.type == VT_VECTOR) {
      if (v.type == VT_VECTOR) {
        size_t n = v.vdisk ? (size_t)v.vlen : v.v.size();
        if ((int)n != c.len)
          throw SqlError(1366, "22007",
                         "Vector dimension mismatch for column '" + c.name + "'");
        if (!v.vdisk) {
          for (float x : v.v) {
            if (!std::isfinite(x))
              throw SqlError(1366, "22007", "Vector component is not finite");
          }
        }
        return v;
      }
      if (v.type == VT_VARCHAR) {
        std::vector<float> vv;
        std::string er;
        if (!parse_vector_string(v.s, vv, er))
          throw SqlError(1366, "22007", "Incorrect vector value: " + er);
        if ((int)vv.size() != c.len)
          throw SqlError(1366, "22007",
                         "Vector dimension mismatch for column '" + c.name + "'");
        for (float x : vv) {
          if (!std::isfinite(x))
            throw SqlError(1366, "22007", "Vector component is not finite");
        }
        Value out;
        out.type = VT_VECTOR;
        out.v = std::move(vv);
        return out;
      }
      throw SqlError(1366, "22007", "Incorrect vector value");
    }
    throw SqlError(1105, "HY000", "unsupported column type");
  }

  Value default_value(const Column &c) {
    Value v;
    v.type = c.type;
    if (c.type == VT_VECTOR) v.v.assign((size_t)c.len, 0.0f);
    return v;
  }

  Value eval(const ExprP &e, const Table &t, const Row &r) {
    if (!e) throw SqlError(1105, "HY000", "empty expression");
    switch (e->kind) {
      case Expr::LIT:
        return e->lit;
      case Expr::COL: {
        if (r.vals.empty())
          throw SqlError(1105, "HY000", "column reference not allowed in this context");
        int idx = col_index(t, e->name);
        return r.vals[(size_t)idx];
      }
      case Expr::L2: {
        if (r.vals.empty())
          throw SqlError(1105, "HY000", "column reference not allowed in this context");
        int idx = col_index(t, e->name);
        const Value &v = r.vals[(size_t)idx];
        if (v.type != VT_VECTOR)
          throw SqlError(1366, "22007", "L2_DISTANCE needs a VECTOR column");
        size_t n = v.vdisk ? (size_t)v.vlen : v.v.size();
        if (n != e->vec.size())
          throw SqlError(1366, "22007", "Vector dimension mismatch");
        std::vector<float> scratch;
        const float *p = t.vecs.get(v, scratch);
        double sum = 0;
        for (size_t i = 0; i < n; ++i) {
          double d = (double)p[i] - (double)e->vec[i];
          sum += d * d;
        }
        Value out;
        out.type = VT_DOUBLE;
        out.d = std::sqrt(sum);
        return out;
      }
      case Expr::IN: {
        Value a = eval(e->a, t, r);
        Value out;
        out.type = VT_BIGINT;
        out.i = 0;
        for (const auto &item : e->list) {
          Value b = eval(item, t, r);
          if (value_cmp(a, b) == 0) {
            out.i = 1;
            break;
          }
        }
        return out;
      }
      case Expr::BETWEEN: {
        Value a = eval(e->a, t, r), lo = eval(e->b, t, r), hi = eval(e->c, t, r);
        Value out;
        out.type = VT_BIGINT;
        out.i = (value_cmp(a, lo) >= 0 && value_cmp(a, hi) <= 0) ? 1 : 0;
        return out;
      }
      case Expr::CMP: {
        Value a = eval(e->a, t, r), b = eval(e->b, t, r);
        if (e->name == "LIKE") {
          Value out;
          out.type = VT_BIGINT;
          out.i = (a.type == VT_VARCHAR && b.type == VT_VARCHAR)
                      ? (like_match(a.s, b.s) ? 1 : 0)
                      : 0;
          return out;
        }
        int c = value_cmp(a, b);
        int tv = 0;
        if (e->name == "=") tv = (c == 0);
        else if (e->name == "!=" || e->name == "<>") tv = (c != 0);
        else if (e->name == "<") tv = (c < 0);
        else if (e->name == "<=") tv = (c <= 0);
        else if (e->name == ">") tv = (c > 0);
        else if (e->name == ">=") tv = (c >= 0);
        Value out;
        out.type = VT_BIGINT;
        out.i = tv;
        return out;
      }
      case Expr::AND: {
        Value out;
        out.type = VT_BIGINT;
        out.i = (eval(e->a, t, r).i != 0 && eval(e->b, t, r).i != 0);
        return out;
      }
      case Expr::OR: {
        Value out;
        out.type = VT_BIGINT;
        out.i = (eval(e->a, t, r).i != 0 || eval(e->b, t, r).i != 0);
        return out;
      }
      case Expr::NOT: {
        Value out;
        out.type = VT_BIGINT;
        out.i = (eval(e->a, t, r).i == 0);
        return out;
      }
    }
    throw SqlError(1105, "HY000", "bad expression");
  }

  bool where_matches(const ExprP &where, const Table &t, const Row &r) {
    if (!where) return true;
    Value v = eval(where, t, r);
    if (v.type == VT_BIGINT) return v.i != 0;
    if (v.type == VT_DOUBLE) return v.d != 0;
    return false;
  }

  // ---- WAL ----------------------------------------------------------------
  bool open_wal_append(std::string &err) {
    if (wal_) std::fclose(wal_);
    std::string p = (fs::path(dir_) / "wal.bin").string();
    wal_ = std::fopen(p.c_str(), "ab");
    if (!wal_) {
      err = "cannot open WAL " + p;
      return false;
    }
    return true;
  }

  void truncate_wal_file() {
    if (wal_) {
      std::fclose(wal_);
      wal_ = nullptr;
    }
    std::string p = (fs::path(dir_) / "wal.bin").string();
    FILE *f = std::fopen(p.c_str(), "wb");
    if (f) std::fclose(f);
  }

  bool wal_sync() {
    if (!wal_) return false;
    if (std::fflush(wal_) != 0) return false;
    return ::fsync(::fileno(wal_)) == 0;
  }

  bool wal_record(uint8_t type, const std::string &payload) {
    if (!wal_) return false;
    uint64_t sq = ++seq_;
    uint32_t len = (uint32_t)payload.size();
    uint32_t crc = 0;
    uint32_t c = 0xFFFFFFFFu;
    c = crc32_update(c, &sq, 8);
    c = crc32_update(c, &type, 1);
    c = crc32_update(c, payload.data(), payload.size());
    crc = c ^ 0xFFFFFFFFu;
    if (!put_u32(wal_, len) || !put_u64(wal_, sq) || !put_u32(wal_, crc) ||
        !put_u8(wal_, type) || !put_bytes(wal_, payload.data(), payload.size()))
      return false;
    return true;
  }

  void wal_commit_one(const std::string &raw) {
    std::string sql = raw;
    if (!sql.empty() && sql.back() != ';') sql.push_back(';');
    if (!wal_record(1, sql) || !wal_record(2, "") || !wal_sync())
      throw SqlError(1105, "HY000", "WAL write failed");
    last_seq_ = seq_;
  }

  void wal_commit_tx() {
    for (const auto &s : tx_sql_) {
      std::string sql = s;
      if (!sql.empty() && sql.back() != ';') sql.push_back(';');
      if (!wal_record(1, sql)) throw SqlError(1105, "HY000", "WAL write failed");
    }
    if (!wal_record(2, "") || !wal_sync())
      throw SqlError(1105, "HY000", "WAL write failed");
    last_seq_ = seq_;
  }

  // ---- snapshots -----------------------------------------------------------
  bool put_value(FILE *f, const Value &v) {
    if (!put_u8(f, (uint8_t)v.type)) return false;
    if (v.type == VT_BIGINT) return put_bytes(f, &v.i, 8);
    if (v.type == VT_DOUBLE) return put_bytes(f, &v.d, 8);
    if (v.type == VT_VARCHAR) return put_str(f, v.s);
    if (v.type == VT_VECTOR) {
      if (v.v.size() > 0xFFFFFFFFu) return false;
      if (!put_u32(f, (uint32_t)v.v.size())) return false;
      return put_bytes(f, v.v.data(), v.v.size() * sizeof(float));
    }
    return false;
  }

  bool get_value(FILE *f, Value &v) {
    uint8_t t = 0;
    if (!get_u8(f, t)) return false;
    v = Value();
    v.type = t;
    if (t == VT_BIGINT) return get_bytes(f, &v.i, 8);
    if (t == VT_DOUBLE) return get_bytes(f, &v.d, 8);
    if (t == VT_VARCHAR) return get_str(f, v.s);
    if (t == VT_VECTOR) {
      uint32_t n = 0;
      if (!get_u32(f, n) || n > 10000000u) return false;
      v.v.resize(n);
      return get_bytes(f, n ? v.v.data() : nullptr, (size_t)n * sizeof(float));
    }
    return false;
  }

  bool save_snapshot(std::string &err) {
    std::string tmp = (fs::path(dir_) / "snapshot.tmp").string();
    std::string dst = (fs::path(dir_) / "snapshot.bin").string();
    FILE *f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
      err = "cannot write snapshot";
      return false;
    }
    bool ok = true;
    const char magic[8] = {'M', 'V', 'D', 'B', '0', '0', '0', '1'};
    ok = ok && put_bytes(f, magic, 8);
    ok = ok && put_u64(f, last_seq_);
    ok = ok && put_u32(f, (uint32_t)tables_.size());
    for (auto &kv : tables_) {
      const Table &t = kv.second;
      ok = ok && put_str(f, t.name);
      ok = ok && put_u32(f, (uint32_t)t.cols.size());
      for (const auto &c : t.cols) {
        ok = ok && put_str(f, c.name);
        ok = ok && put_i32(f, c.type);
        ok = ok && put_i32(f, c.len);
      }
      ok = ok && put_i32(f, t.pk);
      ok = ok && put_i32(f, t.vec);
      uint64_t live = 0;
      for (const auto &r : t.rows)
        if (!r.deleted) ++live;
      ok = ok && put_u64(f, live);
      for (const auto &r : t.rows) {
        if (r.deleted) continue;
        for (size_t ci = 0; ci < r.vals.size(); ++ci) {
          const Value &v = r.vals[ci];
          if (v.type == VT_VECTOR && v.vdisk) {
            try {
              Value mv = t.vecs.materialize(v);
              ok = ok && put_value(f, mv);
            } catch (const std::exception &e) {
              ok = false;
            }
          } else {
            ok = ok && put_value(f, v);
          }
        }
      }
      if (!ok) break;
    }
    if (ok) {
      ok = std::fflush(f) == 0 && ::fsync(::fileno(f)) == 0;
    }
    std::fclose(f);
    if (!ok) {
      std::remove(tmp.c_str());
      err = "snapshot write failed";
      return false;
    }
    if (std::rename(tmp.c_str(), dst.c_str()) != 0) {
      err = "snapshot rename failed";
      return false;
    }
    fsync_dir(dir_);
    return true;
  }

  bool load_snapshot(std::string &err) {
    std::string p = (fs::path(dir_) / "snapshot.bin").string();
    FILE *f = std::fopen(p.c_str(), "rb");
    if (!f) return true;  // fresh data directory
    char magic[8];
    if (!get_bytes(f, magic, 8) || std::memcmp(magic, "MVDB0001", 8) != 0) {
      std::fclose(f);
      err = "invalid snapshot";
      return false;
    }
    uint64_t last = 0;
    uint32_t ntab = 0;
    if (!get_u64(f, last) || !get_u32(f, ntab) || ntab > 1000000u) {
      std::fclose(f);
      err = "invalid snapshot header";
      return false;
    }
    last_seq_ = last;
    tables_.clear();
    for (uint32_t ti = 0; ti < ntab; ++ti) {
      Table t;
      uint32_t nc = 0;
      if (!get_str(f, t.name) || !get_u32(f, nc) || nc > 100000u) {
        std::fclose(f);
        err = "invalid snapshot table";
        return false;
      }
      t.cols.resize(nc);
      for (auto &c : t.cols) {
        int32_t ty = 0, ln = 0;
        if (!get_str(f, c.name) || !get_i32(f, ty) || !get_i32(f, ln)) {
          std::fclose(f);
          err = "invalid snapshot column";
          return false;
        }
        c.type = ty;
        c.len = ln;
      }
      int32_t pk = -1, vec = -1;
      if (!get_i32(f, pk) || !get_i32(f, vec)) {
        std::fclose(f);
        err = "invalid snapshot table metadata";
        return false;
      }
      t.pk = pk;
      t.vec = vec;
      if (t.vec >= 0) {
        try {
          t.vecs.init(dir_, t.name, t.cols[(size_t)t.vec].len);
        } catch (const std::exception &e) {
          std::fclose(f);
          err = std::string("vector store init failed: ") + e.what();
          return false;
        }
      }
      uint64_t nrow = 0;
      if (!get_u64(f, nrow) || nrow > 1000000000ull) {
        std::fclose(f);
        err = "invalid snapshot row count";
        return false;
      }
      t.rows.reserve((size_t)std::min<uint64_t>(nrow, 1000000));
      for (uint64_t ri = 0; ri < nrow; ++ri) {
        Row r;
        r.vals.resize(nc);
        for (size_t ci = 0; ci < r.vals.size(); ++ci) {
          if (!get_value(f, r.vals[ci])) {
            std::fclose(f);
            err = "invalid snapshot row";
            return false;
          }
          if (t.cols[ci].type == VT_VECTOR) {
            try {
              r.vals[ci] = t.vecs.store(r.vals[ci]);
            } catch (const std::exception &e) {
              std::fclose(f);
              err = std::string("vector snapshot load failed: ") + e.what();
              return false;
            }
          }
        }
        if (t.pk >= 0 && (size_t)t.pk < r.vals.size()) {
          int64_t key = r.vals[(size_t)t.pk].i;
          if (t.pkmap.count(key)) {
            std::fclose(f);
            err = "duplicate primary key in snapshot";
            return false;
          }
          t.pkmap[key] = t.rows.size();
        }
        t.rows.push_back(std::move(r));
      }
      tables_[t.name] = std::move(t);
    }
    std::fclose(f);
    return true;
  }

  // WAL replay.  Complete transaction groups with seq > snapshot last_seq are
  // re-executed; a trailing incomplete record is ignored.
  bool replay_wal(std::string &err) {
    std::string p = (fs::path(dir_) / "wal.bin").string();
    FILE *f = std::fopen(p.c_str(), "rb");
    if (!f) return true;
    std::vector<std::string> buf;
    uint64_t maxseq = last_seq_;
    for (;;) {
      uint32_t len = 0, crc = 0;
      uint64_t sq = 0;
      uint8_t type = 0;
      if (std::fread(&len, 4, 1, f) != 1) break;
      if (std::fread(&sq, 8, 1, f) != 1) break;
      if (std::fread(&crc, 4, 1, f) != 1) break;
      if (std::fread(&type, 1, 1, f) != 1) break;
      if (len > 64u * 1024u * 1024u) break;
      std::string payload(len, '\0');
      if (len && std::fread(&payload[0], 1, len, f) != len) break;
      uint32_t c = 0xFFFFFFFFu;
      c = crc32_update(c, &sq, 8);
      c = crc32_update(c, &type, 1);
      c = crc32_update(c, payload.data(), payload.size());
      c ^= 0xFFFFFFFFu;
      if (c != crc) break;
      if (type == 1) {
        buf.push_back(payload);
      } else if (type == 2) {
        if (sq > last_seq_) {
          for (auto &sql : buf) {
            try {
              recovering_ = true;
              Result ignored = exec_sql(sql, true);
              (void)ignored;
            } catch (const SqlError &e) {
              std::cerr << "WAL replay error [" << e.code << "]: " << e.what()
                        << " sql=" << sql.substr(0, 200) << "\n";
            } catch (const std::exception &e) {
              std::cerr << "WAL replay exception: " << e.what() << "\n";
            }
          }
          maxseq = std::max(maxseq, sq);
        }
        buf.clear();
      }
    }
    std::fclose(f);
    last_seq_ = maxseq;
    seq_ = maxseq;
    recovering_ = false;
    return true;
  }

  // ---- DDL / DML -----------------------------------------------------------
  void create_table(const Stmt &st, const std::string &raw, bool recovering) {
    if (tables_.count(st.table)) {
      if (st.if_not_exists) return;
      throw SqlError(1050, "42S01", "Table '" + st.table + "' already exists");
    }
    std::set<std::string> names;
    for (const auto &c : st.cols) {
      std::string u = upper(c.name);
      if (names.count(u))
        throw SqlError(1060, "42S21", "Duplicate column name '" + c.name + "'");
      names.insert(u);
    }
    if (!recovering) {
      wal_commit_one(raw);
    }
    Table t;
    t.name = st.table;
    t.cols = st.cols;
    t.pk = st.pk;
    t.vec = st.vec;
    if (t.vec >= 0) t.vecs.init(dir_, t.name, t.cols[(size_t)t.vec].len);
    tables_[st.table] = std::move(t);
  }

  void drop_table(const Stmt &st, const std::string &raw, bool recovering) {
    auto it = tables_.find(st.table);
    if (it == tables_.end()) {
      if (st.if_exists) return;
      throw SqlError(1051, "42S02", "Unknown table '" + st.table + "'");
    }
    if (!recovering) wal_commit_one(raw);
    it->second.vecs.release();
    tables_.erase(it);
  }

  int64_t insert_rows(const Stmt &st, const std::string &raw, bool recovering) {
    Table &t = table(st.table);
    std::vector<int> target;
    if (st.insert_cols.empty()) {
      for (size_t i = 0; i < t.cols.size(); ++i) target.push_back((int)i);
    } else {
      std::set<int> seen;
      for (const auto &c : st.insert_cols) {
        int idx = col_index(t, c);
        if (seen.count(idx))
          throw SqlError(1110, "42000", "Column '" + c + "' specified twice");
        seen.insert(idx);
        target.push_back(idx);
      }
    }
    if (target.empty()) throw syntax("no insert columns");
    std::set<int64_t> newkeys;
    std::vector<Row> newrows;
    for (const auto &src : st.insert_rows) {
      if (src.size() != target.size())
        throw SqlError(1136, "21S01",
                       "Column count doesn't match value count");
      Row r;
      r.vals.resize(t.cols.size());
      for (size_t i = 0; i < t.cols.size(); ++i) r.vals[i] = default_value(t.cols[i]);
      for (size_t i = 0; i < target.size(); ++i) {
        Value v = eval(src[i], t, Row{});
        const Column &col = t.cols[(size_t)target[i]];
        Value cv = coerce(v, col);
        if (col.type == VT_VECTOR && !cv.vdisk) cv = t.vecs.store(cv);
        r.vals[(size_t)target[i]] = std::move(cv);
      }
      for (size_t i = 0; i < t.cols.size(); ++i) {
        if (t.cols[i].type == VT_VECTOR && !r.vals[i].vdisk)
          r.vals[i] = t.vecs.store(r.vals[i]);
      }
      int64_t key = r.vals[(size_t)t.pk].i;
      if (t.pkmap.count(key) || newkeys.count(key))
        throw SqlError(1062, "23000", "Duplicate entry '" + std::to_string(key) +
                                       "' for key 'PRIMARY'");
      newkeys.insert(key);
      newrows.push_back(std::move(r));
    }
    if (!in_tx_ && !recovering) wal_commit_one(raw);
    for (auto &r : newrows) {
      int64_t key = r.vals[(size_t)t.pk].i;
      size_t idx = t.rows.size();
      t.pkmap[key] = idx;
      if (in_tx_) undo_.push_back({1, st.table, key, idx, Row{}});
      t.rows.push_back(std::move(r));
    }
    if (in_tx_) tx_sql_.push_back(raw);
    return (int64_t)newrows.size();
  }

  bool value_equal(const Table &t, const Value &a, const Value &b) {
    if (a.type != b.type) return false;
    if (a.type == VT_BIGINT) return a.i == b.i;
    if (a.type == VT_DOUBLE) return a.d == b.d;
    if (a.type == VT_VARCHAR) return a.s == b.s;
    if (a.type == VT_VECTOR) {
      size_t na = a.vdisk ? (size_t)a.vlen : a.v.size();
      size_t nb = b.vdisk ? (size_t)b.vlen : b.v.size();
      if (na != nb) return false;
      std::vector<float> sa, sb;
      const float *pa = t.vecs.get(a, sa);
      const float *pb = t.vecs.get(b, sb);
      for (size_t i = 0; i < na; ++i)
        if (pa[i] != pb[i]) return false;
      return true;
    }
    return false;
  }

  bool row_equal(const Table &t, const Row &a, const Row &b) {
    if (a.vals.size() != b.vals.size()) return false;
    for (size_t i = 0; i < a.vals.size(); ++i)
      if (!value_equal(t, a.vals[i], b.vals[i])) return false;
    return true;
  }

  void order_hits(Table &t, const Stmt &st, std::vector<size_t> &hits) {
    if (st.order.empty()) return;
    std::stable_sort(hits.begin(), hits.end(), [&](size_t a, size_t b) {
      const Row &ra = t.rows[a], &rb = t.rows[b];
      for (const auto &oi : st.order) {
        Value va = eval(oi.expr, t, ra);
        Value vb = eval(oi.expr, t, rb);
        int c = value_cmp(va, vb);
        if (c != 0) return oi.asc ? (c < 0) : (c > 0);
      }
      return ra.vals[(size_t)t.pk].i < rb.vals[(size_t)t.pk].i;
    });
  }

  int64_t update_rows(const Stmt &st, const std::string &raw, bool recovering) {
    Table &t = table(st.table);
    std::vector<std::pair<int, ExprP>> assigns;
    std::set<int> seen;
    for (const auto &p : st.sets) {
      int idx = col_index(t, p.first);
      if (idx == t.pk)
        throw SqlError(1105, "HY000", "Primary key update is not supported");
      if (seen.count(idx))
        throw SqlError(1110, "42000", "Column '" + p.first + "' specified twice");
      seen.insert(idx);
      assigns.emplace_back(idx, p.second);
    }
    if (assigns.empty()) throw syntax("SET requires at least one assignment");
    std::vector<size_t> hits;
    for (size_t i = 0; i < t.rows.size(); ++i) {
      if (t.rows[i].deleted) continue;
      if (where_matches(st.where, t, t.rows[i])) hits.push_back(i);
    }
    order_hits(t, st, hits);
    if (st.limit >= 0 && (uint64_t)st.limit < hits.size())
      hits.resize((size_t)st.limit);
    std::vector<Row> news;
    news.reserve(hits.size());
    for (size_t idx : hits) {
      Row nr = t.rows[idx];
      for (auto &a : assigns) {
        Value v = eval(a.second, t, t.rows[idx]);
        const Column &col = t.cols[(size_t)a.first];
        Value cv = coerce(v, col);
        if (col.type == VT_VECTOR && !cv.vdisk) cv = t.vecs.store(cv);
        nr.vals[(size_t)a.first] = std::move(cv);
      }
      news.push_back(std::move(nr));
    }
    int64_t changed = 0;
    for (size_t i = 0; i < hits.size(); ++i)
      if (!row_equal(t, t.rows[hits[i]], news[i])) ++changed;
    if (!in_tx_ && !recovering) wal_commit_one(raw);
    for (size_t i = 0; i < hits.size(); ++i) {
      size_t idx = hits[i];
      Row old = t.rows[idx];
      if (in_tx_) undo_.push_back({2, st.table, row_pk(t, old), idx, old});
      t.rows[idx] = std::move(news[i]);
    }
    if (in_tx_) tx_sql_.push_back(raw);
    return changed;
  }

  int64_t delete_rows(const Stmt &st, const std::string &raw, bool recovering) {
    Table &t = table(st.table);
    std::vector<size_t> hits;
    for (size_t i = 0; i < t.rows.size(); ++i) {
      if (t.rows[i].deleted) continue;
      if (where_matches(st.where, t, t.rows[i])) hits.push_back(i);
    }
    order_hits(t, st, hits);
    if (st.limit >= 0 && (uint64_t)st.limit < hits.size())
      hits.resize((size_t)st.limit);
    if (!in_tx_ && !recovering) wal_commit_one(raw);
    for (size_t idx : hits) {
      Row old = t.rows[idx];
      int64_t key = row_pk(t, old);
      if (in_tx_) undo_.push_back({3, st.table, key, idx, old});
      t.rows[idx].deleted = true;
      t.pkmap.erase(key);
    }
    if (in_tx_) tx_sql_.push_back(raw);
    return (int64_t)hits.size();
  }

  // ---- transactions --------------------------------------------------------
  void commit_tx() {
    if (!tx_sql_.empty()) wal_commit_tx();
    undo_.clear();
    tx_sql_.clear();
    in_tx_ = false;
  }

  void rollback_tx() {
    for (auto it = undo_.rbegin(); it != undo_.rend(); ++it) {
      auto tit = tables_.find(it->table);
      if (tit == tables_.end()) continue;
      Table &t = tit->second;
      if (it->op == 1) {
        // inserted row: remove it
        if (it->idx < t.rows.size() && !t.rows[it->idx].deleted) {
          t.rows[it->idx].deleted = true;
          t.pkmap.erase(it->pk);
        }
      } else if (it->op == 2 || it->op == 3) {
        if (it->idx < t.rows.size()) {
          t.rows[it->idx] = it->old;
          t.rows[it->idx].deleted = false;
          t.pkmap[it->pk] = it->idx;
        }
      }
    }
    undo_.clear();
    tx_sql_.clear();
    in_tx_ = false;
  }

  // ---- SELECT --------------------------------------------------------------
  Result select_rows(const Stmt &st) {
    Result res;
    res.query = true;
    if (st.table.empty()) {
      std::vector<std::string> headers;
      std::vector<Value> vals;
      for (const auto &it : st.items) {
        if (it.expr->kind == Expr::COL || it.expr->kind == Expr::L2)
          throw SqlError(1054, "42S22", "Unknown column in SELECT without FROM");
        vals.push_back(eval(it.expr, Table{}, Row{}));
        if (!it.alias.empty()) headers.push_back(it.alias);
        else if (it.expr->kind == Expr::LIT) headers.push_back(fmt_value(it.expr->lit));
        else headers.push_back("expr");
      }
      res.columns = headers;
      if (st.limit == 0) return res;
      res.rows.push_back(vals);
      return res;
    }
    Table &t = table(st.table);
    std::vector<size_t> matches;
    matches.reserve(t.rows.size());
    for (size_t i = 0; i < t.rows.size(); ++i) {
      if (t.rows[i].deleted) continue;
      if (where_matches(st.where, t, t.rows[i])) matches.push_back(i);
    }

    std::vector<OrderItem> orders = st.order;
    for (auto &oi : orders) {
      if (oi.expr && oi.expr->kind == Expr::COL) {
        for (const auto &it : st.items) {
          if (!it.alias.empty() && ieq(it.alias, oi.expr->name.c_str())) {
            oi.expr = it.expr;
            break;
          }
        }
      }
    }
    if (!orders.empty()) {
      if (orders.size() == 1 && orders[0].expr &&
          orders[0].expr->kind == Expr::L2) {
        // Hot path: exact L2 TopK.  Compute each distance once, then use a
        // bounded selection for LIMIT instead of sorting with vector copies.
        const Expr &oe = *orders[0].expr;
        int vcol = col_index(t, oe.name);
        struct Cand {
          double d;
          int64_t pk;
          size_t idx;
        };
        std::vector<Cand> cands;
        cands.reserve(matches.size());
        std::vector<float> scratch;
        for (size_t idx : matches) {
          const Row &r = t.rows[idx];
          const Value &v = r.vals[(size_t)vcol];
          if (v.type != VT_VECTOR)
            throw SqlError(1366, "22007", "L2_DISTANCE needs a VECTOR column");
          size_t n = v.vdisk ? (size_t)v.vlen : v.v.size();
          if (n != oe.vec.size())
            throw SqlError(1366, "22007", "Vector dimension mismatch");
          const float *p = t.vecs.get(v, scratch);
          double sum = 0;
          for (size_t i = 0; i < n; ++i) {
            double d = (double)p[i] - (double)oe.vec[i];
            sum += d * d;
          }
          cands.push_back({std::sqrt(sum), r.vals[(size_t)t.pk].i, idx});
        }
        auto cmp = [&](const Cand &a, const Cand &b) {
          if (a.d != b.d) return orders[0].asc ? (a.d < b.d) : (a.d > b.d);
          return a.pk < b.pk;
        };
        size_t keep = cands.size();
        if (st.limit >= 0) {
          uint64_t want = (uint64_t)st.offset + (uint64_t)st.limit;
          keep = (size_t)std::min<uint64_t>(want, cands.size());
        }
        if (keep < cands.size()) {
          std::nth_element(cands.begin(), cands.begin() + (std::ptrdiff_t)keep, cands.end(), cmp);
          cands.resize(keep);
        }
        std::sort(cands.begin(), cands.end(), cmp);
        matches.clear();
        matches.reserve(cands.size());
        for (const auto &c : cands) matches.push_back(c.idx);
      } else {
        std::stable_sort(matches.begin(), matches.end(), [&](size_t a, size_t b) {
          const Row &ra = t.rows[a], &rb = t.rows[b];
          for (const auto &oi : orders) {
            Value va = eval(oi.expr, t, ra);
            Value vb = eval(oi.expr, t, rb);
            int c = value_cmp(va, vb);
            if (c != 0) return oi.asc ? (c < 0) : (c > 0);
          }
          return ra.vals[(size_t)t.pk].i < rb.vals[(size_t)t.pk].i;
        });
      }
    }

    int64_t off = st.offset;
    int64_t lim = st.limit < 0 ? (int64_t)matches.size() : st.limit;
    std::vector<std::string> headers;
    std::vector<const ExprP *> exprs;  // null means a plain table column
    std::vector<int> colidx;
    if (st.star) {
      for (size_t i = 0; i < t.cols.size(); ++i) {
        headers.push_back(t.cols[i].name);
        exprs.push_back(nullptr);
        colidx.push_back((int)i);
      }
    } else {
      for (const auto &it : st.items) {
        if (it.expr->kind == Expr::COL) {
          int idx = col_index(t, it.expr->name);
          headers.push_back(it.alias.empty() ? t.cols[(size_t)idx].name : it.alias);
          exprs.push_back(nullptr);
          colidx.push_back(idx);
        } else {
          headers.push_back(it.alias.empty() ? "l2_distance" : it.alias);
          exprs.push_back(&it.expr);
          colidx.push_back(-1);
        }
      }
    }

    res.columns = headers;
    int64_t emitted = 0;
    for (size_t k = 0; k < matches.size(); ++k) {
      if ((int64_t)k < off) continue;
      if (emitted >= lim) break;
      const Row &r = t.rows[matches[k]];
      std::vector<Value> row;
      row.reserve(headers.size());
      for (size_t i = 0; i < headers.size(); ++i) {
        Value v = exprs[i] ? eval(*exprs[i], t, r) : r.vals[(size_t)colidx[i]];
        if (v.type == VT_VECTOR && v.vdisk) v = t.vecs.materialize(v);
        row.push_back(v);
      }
      res.rows.push_back(std::move(row));
      ++emitted;
    }
    return res;
  }

};

// ---------------------------------------------------------------------------
// stdin statement splitter
// ---------------------------------------------------------------------------
static void split_statements(std::string &buf, std::vector<std::string> &out) {
  size_t start = 0;
  int quote = 0;  // 1=' 2=" 3=`
  bool line_comment = false, block_comment = false;
  for (size_t i = 0; i < buf.size(); ++i) {
    char c = buf[i];
    if (line_comment) {
      if (c == '\n') line_comment = false;
      continue;
    }
    if (block_comment) {
      if (c == '*' && i + 1 < buf.size() && buf[i + 1] == '/') {
        block_comment = false;
        ++i;
      }
      continue;
    }
    if (quote) {
      if (c == '\\' && quote != 3) {
        ++i;
      } else if (c == (quote == 1 ? '\'' : quote == 2 ? '"' : '`')) {
        quote = 0;
      }
      continue;
    }
    if (c == '-' && i + 1 < buf.size() && buf[i + 1] == '-' &&
        (i + 2 >= buf.size() || std::isspace((unsigned char)buf[i + 2]))) {
      line_comment = true;
      ++i;
      continue;
    }
    if (c == '#') {
      line_comment = true;
      continue;
    }
    if (c == '/' && i + 1 < buf.size() && buf[i + 1] == '*') {
      block_comment = true;
      ++i;
      continue;
    }
    if (c == '\'') quote = 1;
    else if (c == '"') quote = 2;
    else if (c == '`') quote = 3;
    else if (c == ';') {
      out.push_back(buf.substr(start, i - start));
      start = i + 1;
    }
  }
  buf.erase(0, start);
}

static volatile std::sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

static int legacy_main(DB &db) {
  std::string buf, line;
  bool done = false;
  while (!g_stop && std::getline(std::cin, line)) {
    std::string t = trim(line);
    if (ieq(t, "exit") || ieq(t, "quit")) break;
    if (starts_with_ci(t, "DELIMITER")) continue;
    buf += line;
    buf.push_back('\n');
    std::vector<std::string> stmts;
    split_statements(buf, stmts);
    for (auto &s : stmts) {
      if (g_stop) break;
      std::string sql = trim(s);
      if (sql.empty()) continue;
      if (ieq(sql, "exit") || ieq(sql, "quit")) {
        buf.clear();
        done = true;
        break;
      }
      try {
        Result r = db.exec_sql(sql);
        std::string out = format_legacy(r);
        if (!out.empty()) {
          std::cout << out;
          if (out.back() != '\n') std::cout << '\n';
          std::cout.flush();
        }
      } catch (const SqlError &e) {
        std::cout << "ERROR " << e.code << " (" << e.state << "): " << e.what() << "\n";
        std::cout.flush();
      } catch (const std::exception &e) {
        std::cout << "ERROR 1105 (HY000): " << e.what() << "\n";
        std::cout.flush();
      }
    }
    if (done) break;
  }
  if (!done && !g_stop) {
    std::string sql = trim(buf);
    if (!sql.empty()) {
      try {
        Result r = db.exec_sql(sql);
        std::string out = format_legacy(r);
        if (!out.empty()) std::cout << out << (out.back() == '\n' ? "" : "\n");
        std::cout.flush();
      } catch (const SqlError &e) {
        std::cout << "ERROR " << e.code << " (" << e.state << "): " << e.what() << "\n";
      } catch (const std::exception &e) {
        std::cout << "ERROR 1105 (HY000): " << e.what() << "\n";
      }
    }
  }
  db.close();
  return 0;
}

static int protocol_main(DB &db) {
  std::cout << "{\"type\":\"ready\"}\n";
  std::cout.flush();
  std::string line;
  while (!g_stop && std::getline(std::cin, line)) {
    if (trim(line).empty()) continue;
    JRequest req;
    std::string perr;
    if (!parse_request(line, req, perr)) {
      std::cout << "{\"id\":null,\"ok\":false,\"error\":{\"code\":\"invalid_request\",\"message\":"
                << json_escape(perr) << "}}\n";
      std::cout.flush();
      continue;
    }
    std::string id = req.id_raw.empty() ? "null" : req.id_raw;
    if (req.op == "shutdown") {
      db.close();
      std::cout << "{\"id\":" << id << ",\"ok\":true}\n";
      std::cout.flush();
      return 0;
    }
    if (req.op != "execute") {
      std::cout << "{\"id\":" << id
                << ",\"ok\":false,\"error\":{\"code\":\"unsupported_op\",\"message\":\"unsupported operation\"}}\n";
      std::cout.flush();
      continue;
    }
    try {
      Result r = db.exec_sql(req.sql);
      std::cout << "{\"id\":" << id << ",\"ok\":true" << json_result_body(r) << "}\n";
      std::cout.flush();
    } catch (const SqlError &e) {
      std::cout << "{\"id\":" << id << ",\"ok\":false,\"error\":{\"code\":"
                << json_escape(protocol_error_code(e)) << ",\"message\":" << json_escape(e.what())
                << "}}\n";
      std::cout.flush();
    } catch (const std::exception &e) {
      std::cout << "{\"id\":" << id
                << ",\"ok\":false,\"error\":{\"code\":\"internal_error\",\"message\":"
                << json_escape(e.what()) << "}}\n";
      std::cout.flush();
    }
  }
  db.close();
  return 0;
}

int main(int argc, char **argv) {
  std::string dir = "/data";
  if (const char *e = std::getenv("VECTORDB_DATA_DIR")) dir = e;
  else if (const char *e = std::getenv("DATA_DIR")) dir = e;
  if (argc > 1) dir = argv[1];

  DB db;
  std::string err;
  if (!db.open(dir, err)) {
    std::cerr << "open failed: " << err << "\n";
    return 1;
  }

  struct sigaction sa;
  std::memset(&sa, 0, sizeof(sa));
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGINT, &sa, nullptr);

  const char *legacy = std::getenv("MVDB_LEGACY_STDIO");
  if (legacy && *legacy && std::string(legacy) != "0")
    return legacy_main(db);
  return protocol_main(db);
}

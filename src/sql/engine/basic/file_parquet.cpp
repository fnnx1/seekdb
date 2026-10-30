/* Copyright (c) 2025 OceanBase. Licensed under the Apache License, Version 2.0. */
#include "sql/engine/basic/file_parquet.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <limits>
#include <set>

namespace oceanbase {
namespace sql {
namespace {

// Limits apply before allocating from untrusted file metadata.
constexpr size_t MAX_PAGE_BYTES = 16 * 1024 * 1024;
constexpr size_t MAX_PAGE_VALUES = 1024 * 1024;
constexpr size_t MAX_METADATA_BYTES = 8 * 1024 * 1024;
constexpr size_t MAX_SCAN_MEMORY = 128 * 1024 * 1024;

enum ThriftType {
  TC_TRUE = 1, TC_FALSE = 2, TC_BYTE = 3, TC_I16 = 4, TC_I32 = 5, TC_I64 = 6,
  TC_DOUBLE = 7, TC_BINARY = 8, TC_LIST = 9, TC_SET = 10, TC_MAP = 11, TC_STRUCT = 12
};

enum ParquetType {
  PT_BOOLEAN = 0, PT_INT32 = 1, PT_INT64 = 2, PT_INT96 = 3, PT_FLOAT = 4,
  PT_DOUBLE = 5, PT_BYTE_ARRAY = 6, PT_FIXED = 7
};

struct TVal {
  int type = 0;
  int field_id = 0;
  int64_t i = 0;
  double d = 0;
  std::string s;
  std::vector<TVal> list;
  // vector supports an incomplete element type; pair<int, TVal> may instantiate
  // TVal's destructor before the pair is complete with Clang/libstdc++ in C++20.
  // Keep the Thrift field ID on the child so both recursive containers are TVal.
  std::vector<TVal> fields;

  const TVal *field(int id) const
  {
    for (const auto &item : fields) if (item.field_id == id) return &item;
    return nullptr;
  }
  int64_t get_int(int id, int64_t fallback = 0) const
  {
    const TVal *f = field(id);
    return f ? f->i : fallback;
  }
  const std::string *get_str(int id) const
  {
    const TVal *f = field(id);
    return f ? &f->s : nullptr;
  }
};

class Compact {
public:
  Compact(const uint8_t *data, size_t size, size_t pos = 0)
      : data_(data), size_(size), pos_(pos) {}
  bool ok() const { return !bad_; }
  size_t pos() const { return pos_; }

  bool byte(uint8_t &out)
  {
    if (pos_ >= size_) { bad_ = true; return false; }
    out = data_[pos_++];
    return true;
  }
  bool varint(uint64_t &out)
  {
    out = 0;
    int shift = 0;
    while (true) {
      uint8_t b = 0;
      if (!byte(b)) return false;
      if (shift == 63 && (b & 0x7e)) { bad_ = true; return false; }
      out |= static_cast<uint64_t>(b & 0x7f) << shift;
      if (0 == (b & 0x80)) return true;
      shift += 7;
      if (shift > 63) { bad_ = true; return false; }
    }
  }
  bool zigzag(int64_t &out)
  {
    uint64_t v = 0;
    if (!varint(v)) return false;
    out = static_cast<int64_t>(v >> 1) ^ -static_cast<int64_t>(v & 1);
    return true;
  }
  bool binary(std::string &out)
  {
    uint64_t n = 0;
    if (!varint(n) || n > size_ - pos_) { bad_ = true; return false; }
    out.assign(reinterpret_cast<const char *>(data_ + pos_), static_cast<size_t>(n));
    pos_ += static_cast<size_t>(n);
    return true;
  }
  bool parse_struct(TVal &out) { return value(TC_STRUCT, out); }
  bool value(int type, TVal &out, unsigned depth = 0);

private:
  const uint8_t *data_;
  size_t size_;
  size_t pos_;
  bool bad_ = false;
  size_t nodes_ = 0;
};

bool Compact::value(int type, TVal &out, unsigned depth)
{
  if (depth > 32 || ++nodes_ > 100000) { bad_ = true; return false; }
  out.type = type;
  switch (type) {
    case TC_TRUE:
      out.i = 1;
      return true;
    case TC_FALSE:
      out.i = 0;
      return true;
    case TC_BYTE: {
      uint8_t b = 0;
      if (!byte(b)) return false;
      out.i = b;
      return true;
    }
    case TC_I16:
    case TC_I32:
    case TC_I64:
      return zigzag(out.i);
    case TC_DOUBLE: {
      if (size_ - pos_ < 8) { bad_ = true; return false; }
      uint64_t bits = 0;
      for (int i = 0; i < 8; ++i) bits |= static_cast<uint64_t>(data_[pos_ + i]) << (8 * i);
      pos_ += 8;
      memcpy(&out.d, &bits, sizeof(out.d));
      return true;
    }
    case TC_BINARY:
      return binary(out.s);
    case TC_LIST:
    case TC_SET: {
      uint8_t header = 0;
      if (!byte(header)) return false;
      uint64_t n = header >> 4;
      const int elem = header & 0x0f;
      if (15 == n && !varint(n)) return false;
      if (n > 100000) { bad_ = true; return false; }
      for (uint64_t i = 0; i < n; ++i) {
        TVal item;
        if (!value(elem, item, depth + 1)) return false;
        out.list.push_back(std::move(item));
      }
      return true;
    }
    case TC_MAP: {
      uint64_t n = 0;
      if (!varint(n)) return false;
      if (0 == n) return true;
      uint8_t kv = 0;
      if (!byte(kv)) return false;
      if (n > 100000) { bad_ = true; return false; }
      for (uint64_t i = 0; i < n; ++i) {
        TVal key;
        TVal item;
        if (!value(kv >> 4, key, depth + 1) || !value(kv & 0x0f, item, depth + 1)) return false;
      }
      return true;
    }
    case TC_STRUCT: {
      out.type = TC_STRUCT;
      int64_t last = 0;
      while (true) {
        uint8_t header = 0;
        if (!byte(header)) return false;
        if (0 == header) return true;
        const int kind = header & 0x0f;
        const int delta = header >> 4;
        int64_t id = 0;
        if (delta) id = last + delta;
        else if (!zigzag(id)) return false;
        last = id;
        TVal item;
        if (!value(kind, item, depth + 1)) return false;
        item.field_id = static_cast<int>(id);
        out.fields.push_back(std::move(item));
      }
    }
    default:
      bad_ = true;
      return false;
  }
}

uint32_t read_le32(const uint8_t *data)
{
  return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8)
      | (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
}

uint64_t read_le64(const uint8_t *data)
{
  uint64_t value = 0;
  for (int i = 0; i < 8; ++i) value |= static_cast<uint64_t>(data[i]) << (8 * i);
  return value;
}

bool snappy_decompress(const uint8_t *data, size_t size, std::vector<uint8_t> &out)
{
  size_t pos = 0;
  uint64_t expected = 0;
  int shift = 0;
  while (pos < size) {
    const uint8_t b = data[pos++];
    expected |= static_cast<uint64_t>(b & 0x7f) << shift;
    if (0 == (b & 0x80)) break;
    shift += 7;
    if (shift > 35) return false;
  }
  out.clear();
  if (expected > MAX_PAGE_BYTES) return false;
  out.reserve(static_cast<size_t>(expected));
  while (pos < size) {
    const uint8_t tag = data[pos++];
    const int kind = tag & 3;
    if (0 == kind) {
      size_t len = tag >> 2;
      if (len < 60) {
        ++len;
      } else {
        const size_t extra = len - 59;
        if (pos + extra > size) return false;
        len = 0;
        for (size_t i = 0; i < extra; ++i) len |= static_cast<size_t>(data[pos + i]) << (8 * i);
        ++len;
        pos += extra;
      }
      if (len > size - pos || len > expected - out.size()) return false;
      out.insert(out.end(), data + pos, data + pos + len);
      pos += len;
    } else {
      size_t len = 0;
      size_t offset = 0;
      if (1 == kind) {
        len = ((tag >> 2) & 7) + 4;
        if (pos >= size) return false;
        offset = (static_cast<size_t>(tag >> 5) << 8) | data[pos++];
      } else if (2 == kind) {
        len = (tag >> 2) + 1;
        if (pos + 2 > size) return false;
        offset = static_cast<size_t>(data[pos]) | (static_cast<size_t>(data[pos + 1]) << 8);
        pos += 2;
      } else {
        len = (tag >> 2) + 1;
        if (pos + 4 > size) return false;
        offset = static_cast<size_t>(read_le32(data + pos));
        pos += 4;
      }
      if (0 == offset || offset > out.size() || len > expected - out.size()) return false;
      const size_t start = out.size() - offset;
      for (size_t i = 0; i < len; ++i) out.push_back(out[start + i]);
    }
  }
  return out.size() == static_cast<size_t>(expected);
}

bool decode_hybrid(const uint8_t *data, size_t size, int width, size_t count,
                   std::vector<uint32_t> &out, std::string &error)
{
  out.clear();
  if (width < 0 || width > 32 || count > MAX_PAGE_VALUES) { error = "invalid parquet RLE width or count"; return false; }
  if (width == 0) {
    out.assign(count, 0);
    return true;
  }
  size_t pos = 0;
  while (out.size() < count && pos < size) {
    uint64_t header = 0;
    int shift = 0;
    while (true) {
      if (pos >= size) { error = "truncated parquet level header"; return false; }
      const uint8_t b = data[pos++];
      if (shift > 63 || (shift == 63 && (b & 0x7e))) { error = "invalid parquet RLE varint"; return false; }
      header |= static_cast<uint64_t>(b & 0x7f) << shift;
      if (0 == (b & 0x80)) break;
      shift += 7;
    }
    if (header == 0) { error = "empty parquet RLE run"; return false; }
    if (header & 1) {
      const size_t groups = static_cast<size_t>(header >> 1);
      if (groups > (size - pos) / static_cast<size_t>(width)) { error = "truncated parquet bit run"; return false; }
      const size_t bytes = groups * static_cast<size_t>(width);
      if (pos + bytes > size) { error = "truncated bit-packed level run"; return false; }
      const uint8_t *packed = data + pos;
      const size_t total = groups * 8;
      for (size_t i = 0; i < total && out.size() < count; ++i) {
        uint32_t value = 0;
        for (int bit = 0; bit < width; ++bit) {
          const size_t index = i * static_cast<size_t>(width) + static_cast<size_t>(bit);
          value |= static_cast<uint32_t>((packed[index / 8] >> (index % 8)) & 1) << bit;
        }
        out.push_back(value);
      }
      pos += bytes;
    } else {
      const size_t run = static_cast<size_t>(header >> 1);
      const size_t bytes = static_cast<size_t>((width + 7) / 8);
      if (pos + bytes > size) { error = "truncated RLE level run"; return false; }
      uint32_t value = 0;
      for (size_t i = 0; i < bytes; ++i) value |= static_cast<uint32_t>(data[pos + i]) << (8 * i);
      pos += bytes;
      for (size_t i = 0; i < run && out.size() < count; ++i) out.push_back(value);
    }
  }
  if (out.size() != count) { error = "truncated parquet levels"; return false; }
  return true;
}

std::string format_double(double value)
{
  char buffer[64];
  snprintf(buffer, sizeof(buffer), "%.17g", value);
  return std::string(buffer);
}

std::string format_float(float value)
{
  char buffer[64];
  snprintf(buffer, sizeof(buffer), "%.9g", value);
  return std::string(buffer);
}

bool decode_plain(const uint8_t *data, size_t size, int type, int64_t count,
                  int64_t fixed_length, std::vector<std::string> &out, std::string &error)
{
  out.clear();
  if (count < 0 || count > static_cast<int64_t>(MAX_PAGE_VALUES)) { error = "invalid parquet value count"; return false; }
  out.reserve(static_cast<size_t>(count));
  size_t pos = 0;
  for (int64_t i = 0; i < count; ++i) {
    switch (type) {
      case PT_BOOLEAN: {
        if (0 == (i % 8)) { if (pos >= size) { error = "truncated parquet boolean"; return false; } }
        const uint8_t byte = data[i / 8];
        out.push_back(((byte >> (i % 8)) & 1) ? "1" : "0");
        pos = static_cast<size_t>(i / 8 + 1);
        break;
      }
      case PT_INT32: {
        if (pos + 4 > size) { error = "truncated parquet int32"; return false; }
        const int32_t value = static_cast<int32_t>(read_le32(data + pos));
        pos += 4;
        out.push_back(std::to_string(value));
        break;
      }
      case PT_INT64: {
        if (pos + 8 > size) { error = "truncated parquet int64"; return false; }
        const int64_t value = static_cast<int64_t>(read_le64(data + pos));
        pos += 8;
        out.push_back(std::to_string(value));
        break;
      }
      case PT_FLOAT: {
        if (pos + 4 > size) { error = "truncated parquet float"; return false; }
        const uint32_t bits = read_le32(data + pos);
        pos += 4;
        float value = 0;
        memcpy(&value, &bits, sizeof(value));
        out.push_back(format_float(value));
        break;
      }
      case PT_DOUBLE: {
        if (pos + 8 > size) { error = "truncated parquet double"; return false; }
        const uint64_t bits = read_le64(data + pos);
        pos += 8;
        double value = 0;
        memcpy(&value, &bits, sizeof(value));
        out.push_back(format_double(value));
        break;
      }
      case PT_BYTE_ARRAY: {
        if (pos + 4 > size) { error = "truncated parquet byte array"; return false; }
        const uint32_t length = read_le32(data + pos);
        pos += 4;
        if (length > FileScanReader::MAX_FIELD_BYTES || length > size - pos) { error = "truncated parquet byte array"; return false; }
        out.emplace_back(reinterpret_cast<const char *>(data + pos), static_cast<size_t>(length));
        pos += length;
        break;
      }
      case PT_FIXED: {
        if (fixed_length <= 0 || pos + static_cast<size_t>(fixed_length) > size) {
          error = "unsupported parquet fixed length array";
          return false;
        }
        out.emplace_back(reinterpret_cast<const char *>(data + pos), static_cast<size_t>(fixed_length));
        pos += static_cast<size_t>(fixed_length);
        break;
      }
      default:
        error = "unsupported parquet physical type";
        return false;
    }
  }
  return true;
}

bool decode_values(const uint8_t *data, size_t size, int type, int encoding, int64_t count,
                   int64_t fixed_length, const std::vector<std::string> &dictionary,
                   std::vector<std::string> &out, std::string &error)
{
  if (0 == encoding) {
    return decode_plain(data, size, type, count, fixed_length, out, error);
  }
  if (2 == encoding || 8 == encoding) {
    if (dictionary.empty() && count > 0) { error = "parquet dictionary missing"; return false; }
    if (count == 0) { out.clear(); return true; }
    if (size < 1) { error = "truncated parquet dictionary page"; return false; }
    const int width = data[0];
    std::vector<uint32_t> indices;
    if (!decode_hybrid(data + 1, size - 1, width, static_cast<size_t>(count), indices, error)) return false;
    out.clear();
    out.reserve(static_cast<size_t>(count));
    size_t decoded_bytes = 0;
    for (int64_t i = 0; i < count; ++i) {
      if (indices[i] >= dictionary.size()) { error = "parquet dictionary index out of range"; return false; }
      decoded_bytes += dictionary[indices[i]].size();
      if (decoded_bytes > MAX_PAGE_BYTES) { error = "decoded parquet page exceeds memory limit"; return false; }
      out.push_back(dictionary[indices[i]]);
    }
    return true;
  }
  if (3 == encoding && PT_BOOLEAN == type) {
    // Boolean RLE includes a four-byte length in both V1 and V2 pages.
    if (size < 4 || read_le32(data) != size - 4) {
      error = "invalid parquet boolean RLE length"; return false;
    }
    std::vector<uint32_t> values;
    if (!decode_hybrid(data + 4, size - 4, 1, static_cast<size_t>(count), values, error)) return false;
    out.clear();
    for (int64_t i = 0; i < count; ++i) out.push_back(values[i] ? "1" : "0");
    return true;
  }
  error = "unsupported parquet value encoding";
  return false;
}

FileColumnType map_type(int parquet_type)
{
  switch (parquet_type) {
    case PT_BOOLEAN: return FileColumnType::BOOLEAN;
    case PT_INT32: return FileColumnType::BIGINT;
    case PT_INT64: return FileColumnType::BIGINT;
    case PT_FLOAT: return FileColumnType::DOUBLE;
    case PT_DOUBLE: return FileColumnType::DOUBLE;
    default: return FileColumnType::VARCHAR;
  }
}

struct ColumnCursor {
  uint64_t offset = 0, end = 0;
  int type = 0, codec = 0;
  int64_t remaining = 0, max_def = 0;
  std::vector<std::string> dictionary;
  std::vector<FileCell> cells;
  size_t index = 0;
};

bool read_at(FILE *file, uint64_t offset, void *data, size_t size)
{
#ifdef _WIN32
  if (offset > INT64_MAX || _fseeki64(file, static_cast<int64_t>(offset), SEEK_SET)) return false;
#else
  if (offset > INT64_MAX || fseeko(file, static_cast<off_t>(offset), SEEK_SET)) return false;
#endif
  return fread(data, 1, size, file) == size;
}

bool load_page(FILE *file, ColumnCursor &cursor, std::string &error)
{
  auto &cells = cursor.cells;
  auto &dictionary = cursor.dictionary;
  cells.clear();
  cursor.index = 0;
  const int type = cursor.type, codec = cursor.codec;
  const int64_t max_def = cursor.max_def;
  while (cells.empty() && cursor.remaining > 0) {
    if (cursor.offset >= cursor.end) { error = "truncated parquet column chunk"; return false; }
    const size_t header_bytes = static_cast<size_t>(std::min<uint64_t>(65536, cursor.end - cursor.offset));
    std::vector<uint8_t> prefix(header_bytes);
    if (!read_at(file, cursor.offset, prefix.data(), prefix.size())) { error = "cannot read parquet page header"; return false; }
    Compact reader(prefix.data(), prefix.size());
    TVal header;
    if (!reader.parse_struct(header) || !reader.ok()) { error = "invalid parquet page header"; return false; }
    const uint64_t body_start = cursor.offset + reader.pos();
    const int64_t compressed = header.get_int(3, -1);
    const int64_t uncompressed_size = header.get_int(2, -1);
    const int64_t page_type = header.get_int(1, -1);
    if (compressed < 0 || compressed > static_cast<int64_t>(MAX_PAGE_BYTES) || uncompressed_size < 0
        || uncompressed_size > static_cast<int64_t>(MAX_PAGE_BYTES) || static_cast<uint64_t>(compressed) > cursor.end - body_start) {
      error = "invalid or oversized parquet page";
      return false;
    }
    std::vector<uint8_t> storage(static_cast<size_t>(compressed));
    if (!read_at(file, body_start, storage.data(), storage.size())) { error = "cannot read parquet page body"; return false; }
    const uint8_t *body = storage.data();
    const size_t body_size = storage.size();
    if (2 == page_type) {
      const TVal *dict_header = header.field(7);
      if (!dict_header) { error = "missing parquet dictionary header"; return false; }
      const int64_t count = dict_header->get_int(1, 0);
      if (dict_header->get_int(2, -1) != 0) { error = "unsupported dictionary encoding"; return false; }
      std::vector<uint8_t> plain;
      const uint8_t *payload = body;
      size_t payload_size = body_size;
      if (1 == codec) {
        if (!snappy_decompress(body, body_size, plain)) { error = "invalid parquet snappy page"; return false; }
        payload = plain.data();
        payload_size = plain.size();
      } else if (0 != codec) {
        error = "unsupported parquet compression codec";
        return false;
      }
      if (!decode_plain(payload, payload_size, type, count, 0, dictionary, error)) return false;
    } else if (0 == page_type || 3 == page_type) {
      const TVal *page = 0 == page_type ? header.field(5) : header.field(8);
      std::vector<uint8_t> uncompressed;
      const uint8_t *payload = body;
      size_t payload_size = body_size;
      int64_t count = 0;
      int encoding = 0;
      int32_t def_length = 0;
      int32_t rep_length = 0;
      if (0 == page_type) {
        if (!page) { error = "missing parquet data page header"; return false; }
        count = page->get_int(1, 0);
        encoding = static_cast<int>(page->get_int(2, 0));
        if (1 == codec) {
          if (!snappy_decompress(body, body_size, uncompressed)) { error = "invalid parquet snappy page"; return false; }
          payload = uncompressed.data();
          payload_size = uncompressed.size();
        } else if (0 != codec) {
          error = "unsupported parquet compression codec";
          return false;
        }
        if (max_def > 0) {
          if (page->get_int(3, -1) != 3) { error = "unsupported definition encoding"; return false; }
          if (payload_size < 4) { error = "truncated parquet definition levels"; return false; }
          def_length = static_cast<int32_t>(read_le32(payload));
          if (static_cast<size_t>(def_length) + 4 > payload_size) { error = "truncated parquet definition levels"; return false; }
          payload += 4;
          payload_size -= 4;
        }
      } else {
        (void)page;
        if (!header.field(8)) { error = "missing parquet data page v2 header"; return false; }
        const TVal *v2 = header.field(8);
        count = v2->get_int(1, 0);
        encoding = static_cast<int>(v2->get_int(4, 0));
        def_length = static_cast<int32_t>(v2->get_int(5, 0));
        rep_length = static_cast<int32_t>(v2->get_int(6, 0));
        if (static_cast<size_t>(def_length) + static_cast<size_t>(rep_length) > payload_size) {
          error = "truncated parquet page levels";
          return false;
        }
        const bool is_compressed = v2->field(7) ? v2->get_int(7) != 0 : true;
        const uint8_t *values = body + rep_length + def_length;
        const size_t values_size = body_size - static_cast<size_t>(rep_length) - static_cast<size_t>(def_length);
        if (is_compressed && 1 == codec) {
          if (!snappy_decompress(values, values_size, uncompressed)) { error = "invalid parquet snappy page"; return false; }
          payload = uncompressed.data();
          payload_size = uncompressed.size();
        } else if (0 != codec && is_compressed) {
          error = "unsupported parquet compression codec";
          return false;
        } else {
          payload = values;
          payload_size = values_size;
        }
      }
      if (count <= 0 || count > cursor.remaining || count > static_cast<int64_t>(MAX_PAGE_VALUES)
          || def_length < 0 || rep_length != 0) {
        error = "invalid parquet page counts or repetition levels";
        return false;
      }
      std::vector<uint32_t> defs;
      if (max_def > 0) {
        const uint8_t *levels = 0 == page_type ? payload : body + rep_length;
        const size_t levels_size = 0 == page_type ? static_cast<size_t>(def_length) : static_cast<size_t>(def_length);
        if (!decode_hybrid(levels, levels_size, 1, static_cast<size_t>(count), defs, error)) return false;

      }
      // V1 values follow the definition-level stream, whereas V2 already
      // points payload at the decompressed values section.
      if (page_type == 0 && max_def > 0) {
        payload += def_length;
        payload_size -= static_cast<size_t>(def_length);
      }
      int64_t present = count;
      if (max_def > 0) {
        present = 0;
        for (int64_t i = 0; i < count; ++i) if (defs[i] == static_cast<uint32_t>(max_def)) ++present;
        else if (defs[i] != 0) { error = "invalid parquet definition level"; return false; }
      }
      std::vector<std::string> values;
      if (!decode_values(payload, payload_size, type, encoding, present, 0, dictionary, values, error)) return false;
      int64_t used = 0;
      for (int64_t i = 0; i < count; ++i) {
        FileCell cell;
        if (max_def > 0 && defs[i] != static_cast<uint32_t>(max_def)) {
          cell.is_null = true;
        } else if (used < static_cast<int64_t>(values.size())) {
          cell.is_null = false;
          cell.text = std::move(values[used++]);
        } else {
          error = "parquet page has fewer values than levels";
          return false;
        }
        cells.push_back(std::move(cell));
      }
      cursor.remaining -= count;
    } else {
      error = "unsupported parquet page type";
      return false;
    }
    cursor.offset = body_start + body_size;
  }
  return true;
}

}  // namespace

struct ParquetReader::Impl {
  FILE *file = nullptr; // Borrowed from FileScanReader, which checks its identity.
  TVal metadata;
  uint64_t metadata_start = 0;
  int64_t total_rows = 0;
  size_t group = 0;
  int64_t group_row = 0, group_rows = 0;
  struct SchemaColumn { std::string name; int type; int64_t max_def; };
  std::vector<SchemaColumn> schema;
  std::vector<ColumnCursor> cursors;

  bool start_group(std::string &error)
  {
    const TVal &g = metadata.field(4)->list[group];
    const TVal *chunks = g.field(1);
    group_rows = g.get_int(3, -1);
    group_row = 0;
    cursors.clear();
    if (!chunks || chunks->list.size() != schema.size() || group_rows < 0) {
      error = "invalid parquet row group"; return false;
    }
    for (size_t i = 0; i < schema.size(); ++i) {
      const TVal &chunk = chunks->list[i];
      const TVal *m = chunk.field(3);
      if (!m || (chunk.get_str(1) && !chunk.get_str(1)->empty())) {
        error = "external parquet column chunks are unsupported"; return false;
      }
      const TVal *path = m->field(3);
      if (!path || path->list.size() != 1 || path->list[0].s != schema[i].name
          || m->get_int(1, -1) != schema[i].type || m->get_int(5, -1) != group_rows) {
        error = "parquet column metadata disagrees with schema"; return false;
      }
      // Empty row groups may use a zero data-page offset and have no pages.
      if (group_rows == 0) { cursors.emplace_back(); continue; }
      int64_t offset = m->get_int(9, -1);
      const int64_t dict = m->get_int(11, -1);
      if (dict >= 0) offset = std::min(offset, dict);
      const int64_t bytes = m->get_int(7, -1);
      if (offset < 4 || bytes < 0 || static_cast<uint64_t>(offset) > metadata_start
          || static_cast<uint64_t>(bytes) > metadata_start - offset) {
        error = "parquet column chunk outside file"; return false;
      }
      ColumnCursor cursor;
      cursor.offset = offset; cursor.end = offset + bytes;
      cursor.type = schema[i].type; cursor.max_def = schema[i].max_def;
      cursor.codec = static_cast<int>(m->get_int(4, -1)); cursor.remaining = group_rows;
      if (cursor.codec != 0 && cursor.codec != 1) { error = "unsupported parquet codec"; return false; }
      cursors.push_back(std::move(cursor));
    }
    return true;
  }

  bool check_memory(std::string &error) const
  {
    size_t bytes = 0;
    for (const auto &c : cursors) {
      bytes += c.cells.capacity() * sizeof(FileCell) + c.dictionary.capacity() * sizeof(std::string);
      for (const auto &cell : c.cells) bytes += cell.text.capacity();
      for (const auto &value : c.dictionary) bytes += value.capacity();
      if (bytes > MAX_SCAN_MEMORY) { error = "parquet scan memory limit exceeded"; return false; }
    }
    return true;
  }
};

ParquetReader::ParquetReader() : impl_(new Impl()) {}
ParquetReader::~ParquetReader() { delete impl_; }

bool ParquetReader::open(FILE *file, std::vector<FileColumn> &columns, std::string &error)
{
  auto &p = *impl_;
  p = Impl(); p.file = file;
#ifdef _WIN32
  if (_fseeki64(file, 0, SEEK_END)) { error = "cannot seek parquet file"; return false; }
  const int64_t size = _ftelli64(file);
#else
  if (fseeko(file, 0, SEEK_END)) { error = "cannot seek parquet file"; return false; }
  const int64_t size = ftello(file);
#endif
  uint8_t magic[4], footer[8];
  if (size < 12 || !read_at(file, 0, magic, 4) || memcmp(magic, "PAR1", 4)
      || !read_at(file, size - 8, footer, 8) || memcmp(footer + 4, "PAR1", 4)) {
    error = "not a parquet file"; return false;
  }
  const uint32_t length = read_le32(footer);
  if (length > size - 12 || length > MAX_METADATA_BYTES) { error = "invalid or oversized parquet footer"; return false; }
  p.metadata_start = size - 8 - length;
  std::vector<uint8_t> bytes(length);
  if (!read_at(file, p.metadata_start, bytes.data(), bytes.size())) { error = "cannot read parquet footer"; return false; }
  Compact reader(bytes.data(), bytes.size());
  if (!reader.parse_struct(p.metadata) || !reader.ok()) { error = "invalid parquet footer"; return false; }
  const TVal *schema = p.metadata.field(2), *groups = p.metadata.field(4);
  if (!schema || !groups || schema->list.size() < 2
      || schema->list.size() > FileScanReader::MAX_COLUMNS + 1
      || schema->list[0].get_int(5, -1) != static_cast<int64_t>(schema->list.size() - 1)) {
    error = "invalid or nested parquet schema"; return false;
  }
  columns.clear();
  std::set<std::string> names;
  for (size_t i = 1; i < schema->list.size(); ++i) {
    const TVal &s = schema->list[i];
    const auto *name = s.get_str(4);
    const int64_t type = s.get_int(1, -1), repetition = s.get_int(3, -1);
    if (!name || name->empty() || name->size() > 256 || !names.insert(*name).second
        || s.get_int(5) != 0 || repetition < 0 || repetition > 1
        || type < 0 || type > PT_BYTE_ARRAY || type == PT_INT96) {
      error = "unsupported parquet schema type or repetition"; return false;
    }
    // Never silently expose dates, decimals or unsigned integers as physical ints.
    const int64_t converted = s.get_int(6, -1);
    const TVal *logical = s.field(10);
    if ((converted != -1 && converted != 0 && (converted < 15 || converted > 18))
        || (logical && !logical->field(1) && !logical->field(10))) {
      error = "unsupported parquet logical type"; return false;
    }
    if (logical && logical->field(10) && logical->field(10)->get_int(2) != 1) {
      error = "unsigned parquet integers are unsupported"; return false;
    }
    p.schema.push_back({*name, static_cast<int>(type), repetition});
    columns.push_back({*name, map_type(static_cast<int>(type)), 0});
  }
  for (const auto &g : groups->list) {
    const int64_t count = g.get_int(3, -1);
    if (count < 0 || count > INT64_MAX - p.total_rows) { error = "invalid parquet row count"; return false; }
    p.total_rows += count;
  }
  if (p.metadata.get_int(3, -1) != p.total_rows) { error = "inconsistent parquet row count"; return false; }
  return rewind(error);
}

int64_t ParquetReader::row_count() const { return impl_->total_rows; }

bool ParquetReader::rewind(std::string &error)
{
  auto &p = *impl_;
  p.group = 0; p.group_row = 0; p.group_rows = 0; p.cursors.clear();
  return p.metadata.field(4)->list.empty() || p.start_group(error);
}

bool ParquetReader::next(std::vector<FileCell> &row, bool &eof, std::string &error)
{
  auto &p = *impl_;
  const auto &groups = p.metadata.field(4)->list;
  while (p.group < groups.size() && p.group_row == p.group_rows) {
    ++p.group;
    if (p.group < groups.size() && !p.start_group(error)) return false;
  }
  eof = p.group >= groups.size();
  row.clear();
  if (eof) return true;
  size_t record_bytes = 0;
  for (auto &cursor : p.cursors) {
    if (cursor.index == cursor.cells.size()) {
      if (!load_page(p.file, cursor, error) || !p.check_memory(error)) return false;
      if (cursor.cells.empty()) { error = "parquet column shorter than row group"; return false; }
    }
    record_bytes += cursor.cells[cursor.index].text.size();
    if (record_bytes > FileScanReader::MAX_RECORD_BYTES) { error = "parquet row too large"; return false; }
    row.push_back(std::move(cursor.cells[cursor.index++]));
  }
  ++p.group_row;
  return true;
}

}  // namespace sql
}  // namespace oceanbase

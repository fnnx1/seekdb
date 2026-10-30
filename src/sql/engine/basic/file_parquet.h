/* Copyright (c) 2025 OceanBase. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "sql/engine/basic/file_scan_reader.h"
namespace oceanbase { namespace sql {
// Holds a bounded footer and at most one decoded page plus dictionary per column.
// FileScanReader owns the file handle and validates its identity on each read.
class ParquetReader {
public:
  ParquetReader();
  ~ParquetReader();
  ParquetReader(const ParquetReader &) = delete;
  ParquetReader &operator=(const ParquetReader &) = delete;
  bool open(FILE *file, std::vector<FileColumn> &columns, std::string &error);
  bool rewind(std::string &error);
  bool next(std::vector<FileCell> &row, bool &eof, std::string &error);
  int64_t row_count() const;
private:
  struct Impl;
  Impl *impl_;
};
} }

/* Copyright (c) 2025 OceanBase. Licensed under the Apache License, Version 2.0. */
#pragma once

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace oceanbase { namespace sql {
class ParquetReader;

// Serialized in the table access spec's object parameters.  No file contents
// are stored in a plan; entries after SIGNATURE are the inferred column names.
enum FileScanParam { FILE_PATH, FILE_FORMAT, FILE_HEADER, FILE_DELIMITER,
                     FILE_QUOTE, FILE_ESCAPE, FILE_NULL_STRING, FILE_SIGNATURE,
                     FILE_COLUMN_NAMES };

enum class FileColumnType { NIL, BIGINT, DOUBLE, BOOLEAN, VARCHAR, DATE, DATETIME };
struct FileCell {
  std::string text;
  bool is_null = true;
};
struct FileColumn {
  std::string name;
  FileColumnType type = FileColumnType::NIL;
  int64_t non_null_count = 0;
  int16_t scale = 0;
};
struct FileReadOptions {
  bool header = true;
  char delimiter = ',';
  char quote = '"';
  char escape = '"';
  std::string null_string;
};

// A two-pass local file reader: infer the complete schema, then stream rows.
// Memory is bounded by the schema and one record, independently of file size.
class FileScanReader {
public:
  static constexpr size_t MAX_COLUMNS = 2048;
  static constexpr size_t MAX_FIELD_BYTES = 1024 * 1024;
  static constexpr size_t MAX_RECORD_BYTES = 16 * 1024 * 1024;
  FileScanReader() = default;
  ~FileScanReader() { close(); }
  FileScanReader(const FileScanReader &) = delete;
  FileScanReader &operator=(const FileScanReader &) = delete;

  bool open(const std::string &path, const std::string &format, const FileReadOptions &options);
  bool infer();
  bool rewind();
  bool next(std::vector<FileCell> &row, bool &eof);
  bool check_unchanged();
  void close();
  const std::vector<FileColumn> &columns() const { return columns_; }
  void set_columns(const std::vector<FileColumn> &columns) { columns_ = columns; }
  int64_t row_count() const { return row_count_; }
  const std::string &error() const { return error_; }
  const std::string &signature() const { return signature_; }
  const std::string &path() const { return path_; }
  static bool canonical_path(const std::string &input, std::string &output, std::string &error);
  static bool canonical_glob(const std::string &input, std::string &output, std::string &error);
  static FileColumnType infer_type(const std::string &text);
  static FileColumnType widen(FileColumnType left, FileColumnType right);

private:
  bool fail(const std::string &message);
  bool csv_record(std::vector<FileCell> &row, bool &eof);
  bool json_record(std::vector<std::string> &names, std::vector<FileCell> &row, bool &eof);
  bool read_record(std::vector<std::string> &names, std::vector<FileCell> &row, bool &eof);
  bool skip_header();
  bool fingerprint(std::string &value);
  bool expand_glob();
  FILE *file_ = nullptr;
  std::string path_;
  std::string format_;
  std::string error_;
  std::string signature_;
  FileReadOptions options_;
  std::vector<FileColumn> columns_;
  ParquetReader *parquet_ = nullptr;
  std::vector<std::string> glob_matches_;
  size_t glob_index_ = 0;
  int64_t line_ = 1;
  int64_t row_count_ = 0;
};

} }

/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OCEANBASE_SQL_RESOLVER_DML_OB_FILE_TABLE_READER_H_
#define OCEANBASE_SQL_RESOLVER_DML_OB_FILE_TABLE_READER_H_

#include <dirent.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "lib/allocator/ob_allocator.h"
#include "lib/container/ob_array.h"
#include "lib/container/ob_se_array.h"
#include "lib/oblog/ob_log_module.h"
#include "common/ob_accuracy.h"
#include "lib/string/ob_string.h"
#include "common/object/ob_object.h"
#include "common/timezone/ob_time_convert.h"

namespace oceanbase
{
namespace sql
{
namespace filetable
{

// A file read through read_csv()/read_json() behaves like a table, so the
// content has to be turned into (column name, column type, cell) triples that
// the ordinary resolver can attach to a TableItem.
enum ColumnKind
{
  COL_BIGINT = 0,
  COL_DOUBLE,
  COL_BOOLEAN,
  COL_DATE,
  COL_DATETIME,
  COL_VARCHAR,
};

const char *const COLUMN_KIND_NAME[] = {
  "BIGINT", "DOUBLE", "BOOLEAN", "DATE", "DATETIME", "VARCHAR"
};

struct ObFileColumn
{
  ObFileColumn() : name_(), kind_(COL_VARCHAR), nullable_(false) {}
  common::ObString name_;
  ColumnKind kind_;
  bool nullable_;
};

struct ObFileTableData
{
  ObFileTableData() : columns_(), cells_(), row_cnt_(0) {}
  common::ObArray<ObFileColumn> columns_;
  // row major: cells_.at(row * columns_.count() + col)
  common::ObArray<common::ObObjParam> cells_;
  int64_t row_cnt_;
};

static const int64_t FT_MAX_FILE_SIZE = 64LL * 1024 * 1024;
static const int64_t FT_MAX_ROWS = 100000;
static const int64_t FT_MAX_COLUMNS = 1024;

inline int read_whole_file(common::ObIAllocator &alloc,
                           const char *path,
                           char *&buf,
                           int64_t &len)
{
  int ret = OB_SUCCESS;
  int fd = -1;
  struct stat st;
  buf = NULL;
  len = 0;
  if (OB_ISNULL(path)) {
    ret = OB_INVALID_ARGUMENT;
  } else if ((fd = ::open(path, O_RDONLY)) < 0) {
    ret = OB_FILE_NOT_EXIST;
    LOG_WARN("failed to open file table source", K(path), K(errno), K(ret));
  } else if (0 != ::fstat(fd, &st)) {
    ret = OB_IO_ERROR;
    LOG_WARN("failed to stat file table source", K(path), K(errno), K(ret));
  } else if (st.st_size > FT_MAX_FILE_SIZE) {
    ret = OB_SIZE_OVERFLOW;
    LOG_WARN("file table source is too large", K(path), "size", st.st_size, K(ret));
  } else if (OB_ISNULL(buf = static_cast<char *>(alloc.alloc(st.st_size + 1)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    int64_t pos = 0;
    while (OB_SUCC(ret) && pos < st.st_size) {
      const ssize_t n = ::read(fd, buf + pos, st.st_size - pos);
      if (n < 0) {
        ret = OB_IO_ERROR;
        LOG_WARN("failed to read file table source", K(path), K(errno), K(ret));
      } else if (0 == n) {
        break;
      } else {
        pos += n;
      }
    }
    if (OB_SUCC(ret)) {
      len = pos;
      buf[len] = '\0';
    }
  }
  if (fd >= 0) {
    ::close(fd);
  }
  return ret;
}

// ---------------------------------------------------------------------------
// small text helpers (no std::string, everything points into the file buffer)
// ---------------------------------------------------------------------------

inline char ft_lower(const char c)
{
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

inline bool ft_equal_ci(const common::ObString &str, const char *literal)
{
  bool equal = false;
  const int64_t lit_len = static_cast<int64_t>(strlen(literal));
  if (str.length() == lit_len) {
    equal = true;
    for (int64_t i = 0; equal && i < lit_len; ++i) {
      if (ft_lower(str[i]) != ft_lower(literal[i])) {
        equal = false;
      }
    }
  }
  return equal;
}

inline bool ft_is_digit(const char c)
{
  return c >= '0' && c <= '9';
}

inline bool ft_is_integer(const common::ObString &str)
{
  bool result = false;
  int64_t i = 0;
  if (str.length() > 0 && (str[0] == '-' || str[0] == '+')) {
    i = 1;
  }
  if (i < str.length()) {
    result = true;
    for (; result && i < str.length(); ++i) {
      if (!ft_is_digit(str[i])) {
        result = false;
      }
    }
  }
  return result;
}

inline bool ft_is_number(const common::ObString &str)
{
  bool result = false;
  int64_t i = 0;
  int64_t digits = 0;
  int64_t dots = 0;
  if (str.length() > 0 && (str[0] == '-' || str[0] == '+')) {
    i = 1;
  }
  for (; i < str.length(); ++i) {
    if (ft_is_digit(str[i])) {
      ++digits;
    } else if ('.' == str[i]) {
      ++dots;
    } else if ('e' == ft_lower(str[i])) {
      // exponent: the remainder must be a signed integer
      common::ObString rest(str.length() - i - 1, str.ptr() + i + 1);
      if (digits > 0 && dots <= 1) {
        result = ft_is_integer(rest);
      }
      return result;
    } else {
      return false;
    }
  }
  return digits > 0 && dots <= 1;
}

inline bool ft_is_boolean(const common::ObString &str)
{
  return ft_equal_ci(str, "true") || ft_equal_ci(str, "false");
}

inline bool ft_is_date(const common::ObString &str)
{
  bool result = false;
  if (10 == str.length() && '-' == str[4] && '-' == str[7]) {
    result = true;
    for (int64_t i = 0; result && i < 10; ++i) {
      if (4 != i && 7 != i && !ft_is_digit(str[i])) {
        result = false;
      }
    }
  }
  return result;
}

inline bool ft_is_datetime(const common::ObString &str)
{
  bool result = false;
  if (str.length() >= 19 && (' ' == str[10] || 'T' == str[10] || 't' == str[10]) &&
      ft_is_date(common::ObString(10, str.ptr())) &&
      ':' == str[13] && ':' == str[16]) {
    result = true;
    for (int64_t i = 11; result && i < 19; ++i) {
      if (13 != i && 16 != i && !ft_is_digit(str[i])) {
        result = false;
      }
    }
  }
  return result;
}

inline ColumnKind ft_infer_kind(const bool all_null,
                                const bool all_int,
                                const bool all_num,
                                const bool all_bool,
                                const bool all_date,
                                const bool all_datetime)
{
  ColumnKind kind = COL_VARCHAR;
  if (all_null || all_bool) {
    kind = COL_BOOLEAN;
  } else if (all_date) {
    kind = COL_DATE;
  } else if (all_datetime) {
    kind = COL_DATETIME;
  } else if (all_int) {
    kind = COL_BIGINT;
  } else if (all_num) {
    kind = COL_DOUBLE;
  } else {
    kind = COL_VARCHAR;
  }
  return kind;
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

// Splits RFC4180 style records into fields. Quoted fields may contain commas,
// doubled quotes and line breaks; CRLF inside a quoted field is normalized to a
// single LF, which is what a text oriented reader is expected to hand back.
inline int ft_split_csv(common::ObIAllocator &alloc,
                        const char *buf,
                        const int64_t len,
                        common::ObArray<common::ObString> &fields,
                        int64_t &col_cnt)
{
  int ret = OB_SUCCESS;
  char *scratch = NULL;
  int64_t col_in_row = 0;
  col_cnt = -1;
  if (len < 0 || (len > 0 && OB_ISNULL(buf))) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_ISNULL(scratch = static_cast<char *>(alloc.alloc(len + 1)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    int64_t sp = 0;          // write position inside scratch
    int64_t field_begin = 0; // start of the current field inside scratch
    int64_t i = 0;
    bool in_quotes = false;
    while (OB_SUCC(ret) && i < len) {
      const char c = buf[i];
      bool field_done = false;
      bool row_done = false;
      if (in_quotes) {
        if ('"' == c) {
          if (i + 1 < len && '"' == buf[i + 1]) {
            scratch[sp++] = '"';
            i += 2;
          } else {
            in_quotes = false;
            ++i;
          }
        } else if ('\r' == c) {
          scratch[sp++] = '\n';
          ++i;
          if (i < len && '\n' == buf[i]) {
            ++i;
          }
        } else {
          scratch[sp++] = c;
          ++i;
        }
      } else if ('"' == c && sp == field_begin) {
        in_quotes = true;
        ++i;
      } else if (',' == c) {
        field_done = true;
        ++i;
      } else if ('\n' == c) {
        field_done = true;
        row_done = true;
        ++i;
      } else if ('\r' == c) {
        field_done = true;
        row_done = true;
        ++i;
        if (i < len && '\n' == buf[i]) {
          ++i;
        }
      } else {
        scratch[sp++] = c;
        ++i;
      }
      if (OB_SUCC(ret) && field_done) {
        common::ObString field(static_cast<int32_t>(sp - field_begin), scratch + field_begin);
        if (OB_FAIL(fields.push_back(field))) {
        } else {
          ++col_in_row;
          field_begin = sp;
        }
      }
      if (OB_SUCC(ret) && row_done) {
        if (col_cnt < 0) {
          col_cnt = col_in_row;
        } else if (col_cnt != col_in_row) {
          ret = OB_INVALID_DATA;
          LOG_WARN("file table row has an inconsistent number of fields",
                   K(col_cnt), K(col_in_row), K(ret));
        }
        col_in_row = 0;
      }
    }
    // close the trailing record when the file does not end with a line break
    if (OB_SUCC(ret) && (field_begin != sp || col_in_row > 0)) {
      common::ObString field(static_cast<int32_t>(sp - field_begin), scratch + field_begin);
      if (OB_FAIL(fields.push_back(field))) {
      } else if (col_cnt < 0) {
        col_cnt = col_in_row + 1;
      } else if (col_cnt != col_in_row + 1) {
        ret = OB_INVALID_DATA;
      }
    }
    if (OB_SUCC(ret) && (col_cnt <= 0 || col_cnt > FT_MAX_COLUMNS)) {
      ret = OB_INVALID_DATA;
      LOG_WARN("file table has an unsupported column count", K(col_cnt), K(ret));
    }
  }
  return ret;
}

// Builds the de-duplicated header names. A repeated name gets a "__N" suffix,
// counted from the second occurrence on: id, id -> id, id__2.
inline int ft_build_column_names(common::ObIAllocator &alloc,
                                 const common::ObIArray<common::ObString> &raw_names,
                                 const int64_t col_cnt,
                                 common::ObArray<ObFileColumn> &columns)
{
  int ret = OB_SUCCESS;
  for (int64_t i = 0; OB_SUCC(ret) && i < col_cnt; ++i) {
    ObFileColumn column;
    int64_t dup = 0;
    for (int64_t j = 0; j < i; ++j) {
      if (0 == raw_names.at(j).compare(raw_names.at(i))) {
        ++dup;
      }
    }
    if (0 == dup) {
      column.name_ = raw_names.at(i);
    } else {
      char suffix[16];
      int64_t suffix_len = snprintf(suffix, sizeof(suffix), "__%ld", dup + 1);
      if (suffix_len <= 0 || suffix_len >= static_cast<int64_t>(sizeof(suffix))) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        char *name_buf = static_cast<char *>(alloc.alloc(raw_names.at(i).length() + suffix_len));
        if (OB_ISNULL(name_buf)) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          MEMCPY(name_buf, raw_names.at(i).ptr(), raw_names.at(i).length());
          MEMCPY(name_buf + raw_names.at(i).length(), suffix, suffix_len);
          column.name_.assign_ptr(name_buf, raw_names.at(i).length() + static_cast<int32_t>(suffix_len));
        }
      }
    }
    if (OB_SUCC(ret) && OB_FAIL(columns.push_back(column))) {
    }
  }
  return ret;
}

inline int ft_infer_columns(const common::ObIArray<common::ObString> &fields,
                            const int64_t first_row,
                            const int64_t row_cnt,
                            const int64_t col_cnt,
                            common::ObArray<ObFileColumn> &columns)
{
  int ret = OB_SUCCESS;
  for (int64_t c = 0; OB_SUCC(ret) && c < col_cnt; ++c) {
    bool all_null = true;
    bool all_int = true;
    bool all_num = true;
    bool all_bool = true;
    bool all_date = true;
    bool all_datetime = true;
    bool nullable = false;
    for (int64_t r = first_row; r < row_cnt; ++r) {
      const common::ObString &value = fields.at(r * col_cnt + c);
      if (0 == value.length()) {
        nullable = true;
        continue;
      }
      all_null = false;
      if (!ft_is_integer(value)) {
        all_int = false;
      }
      if (!ft_is_number(value)) {
        all_num = false;
      }
      if (!ft_is_boolean(value)) {
        all_bool = false;
      }
      if (!ft_is_date(value)) {
        all_date = false;
      }
      if (!ft_is_datetime(value)) {
        all_datetime = false;
      }
    }
    columns.at(c).kind_ = ft_infer_kind(all_null, all_int, all_num, all_bool, all_date, all_datetime);
    columns.at(c).nullable_ = nullable;
  }
  return ret;
}

inline int64_t ft_parse_int64(const common::ObString &str)
{
  int64_t value = 0;
  int64_t i = 0;
  bool negative = false;
  if (str.length() > 0 && (str[0] == '-' || str[0] == '+')) {
    negative = ('-' == str[0]);
    i = 1;
  }
  for (; i < str.length(); ++i) {
    value = value * 10 + (str[i] - '0');
  }
  return negative ? -value : value;
}

inline double ft_parse_double(const common::ObString &str)
{
  double value = 0.0;
  char small[64];
  if (str.length() < static_cast<int32_t>(sizeof(small))) {
    MEMCPY(small, str.ptr(), str.length());
    small[str.length()] = '\0';
    value = strtod(small, NULL);
  }
  return value;
}

// Turns one text cell into an ObObjParam of the inferred column kind. Empty
// text means NULL regardless of the column kind.
inline int ft_cell_to_obj(common::ObIAllocator &alloc,
                          const ColumnKind kind,
                          const common::ObString &text,
                          const common::ObTimeConvertCtx &cvrt_ctx,
                          common::ObObjParam &obj)
{
  int ret = OB_SUCCESS;
  obj.reset();
  if (0 == text.length()) {
    obj.set_null();
    obj.set_param_meta();
  } else if (COL_VARCHAR == kind) {
    char *copy = static_cast<char *>(alloc.alloc(text.length()));
    if (OB_ISNULL(copy)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      MEMCPY(copy, text.ptr(), text.length());
      common::ObString value(text.length(), copy);
      obj.set_varchar(value);
      obj.set_collation_type(common::CS_TYPE_UTF8MB4_GENERAL_CI);
      obj.set_collation_level(common::CS_LEVEL_IMPLICIT);
      obj.set_param_meta();
    }
  } else if (COL_BIGINT == kind) {
    obj.set_int(ft_parse_int64(text));
    obj.set_param_meta();
  } else if (COL_DOUBLE == kind) {
    obj.set_double(ft_parse_double(text));
    obj.set_param_meta();
  } else if (COL_BOOLEAN == kind) {
    obj.set_tinyint(ft_equal_ci(text, "true") ? 1 : 0);
    obj.set_param_meta();
  } else if (COL_DATE == kind) {
    int32_t date_value = 0;
    if (OB_FAIL(common::ObTimeConverter::str_to_date(text, date_value))) {
      LOG_WARN("failed to convert date cell", K(text), K(ret));
    } else {
      obj.set_date(date_value);
      obj.set_param_meta();
    }
  } else if (COL_DATETIME == kind) {
    int64_t datetime_value = 0;
    // accept both "YYYY-MM-DD HH:MM:SS" and the ISO "YYYY-MM-DDTHH:MM:SS"
    char normalized[64];
    common::ObString datetime_text = text;
    if (text.length() < static_cast<int32_t>(sizeof(normalized)) && text.length() > 10 &&
        ('T' == text[10] || 't' == text[10])) {
      MEMCPY(normalized, text.ptr(), text.length());
      normalized[10] = ' ';
      datetime_text.assign_ptr(normalized, text.length());
    }
    if (OB_FAIL(common::ObTimeConverter::str_to_datetime(datetime_text, cvrt_ctx, datetime_value))) {
      LOG_WARN("failed to convert datetime cell", K(text), K(ret));
    } else {
      obj.set_datetime(datetime_value);
      obj.set_param_meta();
    }
  } else {
    ret = OB_ERR_UNEXPECTED;
  }
  return ret;
}

inline int ft_build_cells(common::ObIAllocator &alloc,
                          const common::ObIArray<common::ObString> &fields,
                          const int64_t first_row,
                          const int64_t row_cnt,
                          const int64_t col_cnt,
                          const common::ObTimeConvertCtx &cvrt_ctx,
                          ObFileTableData &table)
{
  int ret = OB_SUCCESS;
  for (int64_t r = first_row; OB_SUCC(ret) && r < row_cnt; ++r) {
    for (int64_t c = 0; OB_SUCC(ret) && c < col_cnt; ++c) {
      common::ObObjParam obj;
      if (OB_FAIL(ft_cell_to_obj(alloc, table.columns_.at(c).kind_,
                                 fields.at(r * col_cnt + c), cvrt_ctx, obj))) {
      } else if (OB_FAIL(table.cells_.push_back(obj))) {
      }
    }
  }
  if (OB_SUCC(ret)) {
    table.row_cnt_ = row_cnt - first_row;
  }
  return ret;
}

inline int load_csv(common::ObIAllocator &alloc,
                    const char *path,
                    const bool header,
                    const common::ObTimeConvertCtx &cvrt_ctx,
                    ObFileTableData &table)
{
  int ret = OB_SUCCESS;
  char *buf = NULL;
  int64_t len = 0;
  int64_t col_cnt = 0;
  common::ObArray<common::ObString> fields;
  common::ObArray<common::ObString> raw_names;
  if (OB_FAIL(read_whole_file(alloc, path, buf, len))) {
  } else if (OB_FAIL(ft_split_csv(alloc, buf, len, fields, col_cnt))) {
  } else {
    const int64_t total_rows = fields.count() / col_cnt;
    const int64_t first_row = header ? 1 : 0;
    if (total_rows - first_row > FT_MAX_ROWS) {
      ret = OB_SIZE_OVERFLOW;
      LOG_WARN("file table has too many rows", K(total_rows), K(ret));
    } else if (OB_UNLIKELY(total_rows < first_row)) {
      ret = OB_INVALID_DATA;
    } else {
      // column names: from the header row when present, otherwise positional
      for (int64_t i = 0; OB_SUCC(ret) && i < col_cnt; ++i) {
        if (header) {
          if (OB_FAIL(raw_names.push_back(fields.at(i)))) {
          }
        } else {
          char name_buf[32];
          const int64_t name_len = snprintf(name_buf, sizeof(name_buf), "column_%ld", i);
          char *copy = static_cast<char *>(alloc.alloc(name_len));
          if (OB_ISNULL(copy)) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else {
            MEMCPY(copy, name_buf, name_len);
            if (OB_FAIL(raw_names.push_back(common::ObString(static_cast<int32_t>(name_len), copy)))) {
            }
          }
        }
      }
      if (OB_SUCC(ret) && header && 0 == total_rows) {
        ret = OB_INVALID_DATA;
        LOG_WARN("csv file table has a header option but no header line", K(ret));
      }
      if (OB_FAIL(ret)) {
      } else if (OB_FAIL(ft_build_column_names(alloc, raw_names, col_cnt, table.columns_))) {
      } else if (OB_FAIL(ft_infer_columns(fields, first_row, total_rows, col_cnt, table.columns_))) {
      } else if (OB_FAIL(ft_build_cells(alloc, fields, first_row, total_rows, col_cnt, cvrt_ctx, table))) {
      }
    }
  }
  return ret;
}

// ---------------------------------------------------------------------------
// JSON lines
// ---------------------------------------------------------------------------

// A deliberately small JSON reader: it only has to understand the flat
// {"key": scalar, ...} objects that make up a newline delimited JSON file.
inline void ft_skip_spaces(const char *buf, const int64_t len, int64_t &pos)
{
  while (pos < len && (' ' == buf[pos] || '\t' == buf[pos] || '\r' == buf[pos] || '\n' == buf[pos])) {
    ++pos;
  }
}

inline int ft_parse_json_string(common::ObIAllocator &alloc,
                                const char *buf,
                                const int64_t len,
                                int64_t &pos,
                                common::ObString &value)
{
  int ret = OB_SUCCESS;
  char *out = static_cast<char *>(alloc.alloc(len - pos + 1));
  int64_t out_len = 0;
  if (OB_ISNULL(out)) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (pos >= len || '"' != buf[pos]) {
    ret = OB_INVALID_DATA;
  } else {
    ++pos;
    while (OB_SUCC(ret) && pos < len && '"' != buf[pos]) {
      if ('\\' == buf[pos] && pos + 1 < len) {
        ++pos;
        switch (buf[pos]) {
          case 'n': out[out_len++] = '\n'; break;
          case 't': out[out_len++] = '\t'; break;
          case 'r': out[out_len++] = '\r'; break;
          case 'b': out[out_len++] = '\b'; break;
          case 'f': out[out_len++] = '\f'; break;
          case '"': out[out_len++] = '"'; break;
          case '\\': out[out_len++] = '\\'; break;
          case '/': out[out_len++] = '/'; break;
          case 'u': {
            // keep the escape verbatim, the consumer only needs a text value
            out[out_len++] = '\\';
            out[out_len++] = 'u';
            break;
          }
          default: out[out_len++] = buf[pos]; break;
        }
        ++pos;
      } else {
        out[out_len++] = buf[pos++];
      }
    }
    if (OB_SUCC(ret) && pos >= len) {
      ret = OB_INVALID_DATA;
    } else if (OB_SUCC(ret)) {
      ++pos; // closing quote
      value.assign_ptr(out, static_cast<int32_t>(out_len));
    }
  }
  return ret;
}

inline int ft_parse_json_scalar(common::ObIAllocator &alloc,
                                const char *buf,
                                const int64_t len,
                                int64_t &pos,
                                common::ObString &value,
                                bool &is_string,
                                bool &is_null)
{
  int ret = OB_SUCCESS;
  is_string = false;
  is_null = false;
  ft_skip_spaces(buf, len, pos);
  if (OB_UNLIKELY(pos >= len)) {
    ret = OB_INVALID_DATA;
  } else if ('"' == buf[pos]) {
    is_string = true;
    ret = ft_parse_json_string(alloc, buf, len, pos, value);
  } else {
    const int64_t begin = pos;
    while (pos < len && ',' != buf[pos] && '}' != buf[pos] && ']' != buf[pos]) {
      ++pos;
    }
    int64_t end = pos;
    while (end > begin && (' ' == buf[end - 1] || '\t' == buf[end - 1])) {
      --end;
    }
    value.assign_ptr(buf + begin, static_cast<int32_t>(end - begin));
    if (ft_equal_ci(value, "null")) {
      is_null = true;
      value.reset();
    }
  }
  return ret;
}

inline int load_json(common::ObIAllocator &alloc,
                     const char *path,
                     const common::ObTimeConvertCtx &cvrt_ctx,
                     ObFileTableData &table)
{
  int ret = OB_SUCCESS;
  char *buf = NULL;
  int64_t len = 0;
  if (OB_FAIL(read_whole_file(alloc, path, buf, len))) {
  } else {
    // Pass 1: collect the union of the keys (first seen order) and, for every
    // line, the (key, value) pairs it actually carries. A line may omit keys,
    // so the pairs are kept with their names and laid out in pass 2. The pairs
    // of all lines live in one flat array; row_begin_ marks where each line
    // starts and carries one trailing sentinel.
    common::ObArray<common::ObString> keys;
    common::ObArray<common::ObString> all_pairs;
    common::ObArray<int64_t> row_begin;
    int64_t pos = 0;
    while (OB_SUCC(ret) && pos < len) {
      while (pos < len && ('\n' == buf[pos] || '\r' == buf[pos] || ' ' == buf[pos])) {
        ++pos;
      }
      if (pos >= len) {
        break;
      } else if ('{' != buf[pos]) {
        ret = OB_INVALID_DATA;
        LOG_WARN("json line does not contain an object", K(pos), K(ret));
      } else if (row_begin.count() >= FT_MAX_ROWS) {
        ret = OB_SIZE_OVERFLOW;
      } else if (OB_FAIL(row_begin.push_back(all_pairs.count()))) {
      } else {
        // pairs are stored interleaved: key0, value0, key1, value1, ...
        ++pos;
        while (OB_SUCC(ret)) {
          ft_skip_spaces(buf, len, pos);
          if (pos < len && '}' == buf[pos]) {
            ++pos;
            break;
          } else if (pos >= len) {
            ret = OB_INVALID_DATA;
          } else {
            common::ObString key;
            common::ObString value;
            bool is_string = false;
            bool is_null = false;
            if (OB_FAIL(ft_parse_json_string(alloc, buf, len, pos, key))) {
            } else if (FALSE_IT(ft_skip_spaces(buf, len, pos))) {
            } else if (pos >= len || ':' != buf[pos]) {
              ret = OB_INVALID_DATA;
            } else if (FALSE_IT(++pos)) {
            } else if (OB_FAIL(ft_parse_json_scalar(alloc, buf, len, pos, value, is_string, is_null))) {
            } else {
              bool known = false;
              for (int64_t k = 0; !known && k < keys.count(); ++k) {
                known = (keys.at(k).length() == key.length() && 0 == keys.at(k).compare(key));
              }
              if (!known && OB_FAIL(keys.push_back(key))) {
              } else if (OB_FAIL(all_pairs.push_back(key))) {
              } else if (OB_FAIL(all_pairs.push_back(value))) {
              }
              ft_skip_spaces(buf, len, pos);
              if (OB_SUCC(ret) && pos < len && ',' == buf[pos]) {
                ++pos;
              }
            }
          }
        }
      }
    }
    if (OB_SUCC(ret) && OB_FAIL(row_begin.push_back(all_pairs.count()))) {
    }
    if (OB_SUCC(ret) && (keys.count() <= 0 || keys.count() > FT_MAX_COLUMNS)) {
      ret = OB_INVALID_DATA;
      LOG_WARN("json file table has an unsupported number of keys", K(keys.count()), K(ret));
    }
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ft_build_column_names(alloc, keys, keys.count(), table.columns_))) {
    } else {
      // Pass 2: lay the pairs out into a dense row major array, missing keys
      // become NULL, then infer the types and materialize the cells.
      const int64_t col_cnt = keys.count();
      const int64_t row_cnt = row_begin.count() - 1;
      common::ObArray<common::ObString> flat;
      for (int64_t r = 0; OB_SUCC(ret) && r < row_cnt; ++r) {
        for (int64_t c = 0; OB_SUCC(ret) && c < col_cnt; ++c) {
          common::ObString value;
          for (int64_t p = row_begin.at(r); p + 1 < row_begin.at(r + 1); p += 2) {
            if (all_pairs.at(p).length() == keys.at(c).length() &&
                0 == all_pairs.at(p).compare(keys.at(c))) {
              value = all_pairs.at(p + 1);
              break;
            }
          }
          if (OB_FAIL(flat.push_back(value))) {
          }
        }
      }
      if (OB_SUCC(ret) && OB_FAIL(ft_infer_columns(flat, 0, row_cnt, col_cnt, table.columns_))) {
      } else if (OB_SUCC(ret) &&
                 OB_FAIL(ft_build_cells(alloc, flat, 0, row_cnt, col_cnt, cvrt_ctx, table))) {
      }
    }
  }
  return ret;
}

// ---------------------------------------------------------------------------
// glob
// ---------------------------------------------------------------------------

// Returns one row per matching file in a single "file" column, carrying the
// path exactly as the pattern spelled the directory part.
inline int load_glob(common::ObIAllocator &alloc, const char *pattern, ObFileTableData &table)
{
  int ret = OB_SUCCESS;
  DIR *dir = NULL;
  struct dirent *entry = NULL;
  const char *slash = (NULL == pattern) ? NULL : strrchr(pattern, '/');
  ObFileColumn column;
  if (OB_ISNULL(pattern)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_ISNULL(slash)) {
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("glob pattern needs a directory part", K(ret));
  } else {
    const int64_t dir_len = slash - pattern;
    const char *name_pattern = slash + 1;
    char *dir_path = static_cast<char *>(alloc.alloc(dir_len + 1));
    if (OB_ISNULL(dir_path)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      MEMCPY(dir_path, pattern, dir_len);
      dir_path[dir_len] = '\0';
      if (OB_ISNULL(dir = ::opendir(dir_len > 0 ? dir_path : "."))) {
        ret = OB_FILE_NOT_EXIST;
        LOG_WARN("cannot open glob directory", K(dir_path), K(errno), K(ret));
      }
      while (OB_SUCC(ret) && OB_NOT_NULL(entry = ::readdir(dir))) {
        const int64_t name_len = static_cast<int64_t>(strlen(entry->d_name));
        if (0 == strcmp(entry->d_name, ".") || 0 == strcmp(entry->d_name, "..")) {
          continue;
        } else if (0 != ::fnmatch(name_pattern, entry->d_name, 0)) {
          continue;
        } else if (table.row_cnt_ >= FT_MAX_ROWS) {
          ret = OB_SIZE_OVERFLOW;
        } else {
          const int64_t total = dir_len + 1 + name_len;
          char *copy = static_cast<char *>(alloc.alloc(total));
          common::ObObjParam cell;
          if (OB_ISNULL(copy)) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else {
            MEMCPY(copy, pattern, dir_len);
            copy[dir_len] = '/';
            MEMCPY(copy + dir_len + 1, entry->d_name, name_len);
            cell.set_varchar(common::ObString(static_cast<int32_t>(total), copy));
            cell.set_collation_type(common::CS_TYPE_UTF8MB4_GENERAL_CI);
            cell.set_collation_level(common::CS_LEVEL_IMPLICIT);
            cell.set_param_meta();
            if (OB_FAIL(table.cells_.push_back(cell))) {
            } else {
              ++table.row_cnt_;
            }
          }
        }
      }
      if (OB_NOT_NULL(dir)) {
        ::closedir(dir);
        dir = NULL;
      }
    }
  }
  if (OB_SUCC(ret)) {
    char *name_copy = static_cast<char *>(alloc.alloc(4));
    if (OB_ISNULL(name_copy)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      MEMCPY(name_copy, "file", 4);
      column.name_.assign_ptr(name_copy, 4);
      column.kind_ = COL_VARCHAR;
      column.nullable_ = false;
      if (OB_FAIL(table.columns_.push_back(column))) {
      }
    }
  }
  return ret;
}

// ---------------------------------------------------------------------------
// parquet
// ---------------------------------------------------------------------------

// Only what a flat, single row group file needs: primitive leaf columns,
// PLAIN encoding, and either uncompressed or snappy compressed pages.
enum FtThriftType
{
  FT_T_STOP = 0,
  FT_T_BOOL_TRUE = 1,
  FT_T_BOOL_FALSE = 2,
  FT_T_BYTE = 3,
  FT_T_I16 = 4,
  FT_T_I32 = 5,
  FT_T_I64 = 6,
  FT_T_DOUBLE = 7,
  FT_T_BINARY = 8,
  FT_T_LIST = 9,
  FT_T_SET = 10,
  FT_T_MAP = 11,
  FT_T_STRUCT = 12,
};

// Thrift compact protocol reader, the encoding parquet uses for its metadata
// and page headers.
struct ObFtThrift
{
  ObFtThrift() : buf_(NULL), len_(0), pos_(0) {}

  void assign(const uint8_t *buf, const int64_t len, const int64_t pos)
  {
    buf_ = buf;
    len_ = len;
    pos_ = pos;
  }

  int read_byte(uint8_t &value)
  {
    int ret = OB_SUCCESS;
    if (OB_ISNULL(buf_) || pos_ < 0 || pos_ >= len_) {
      ret = OB_INVALID_DATA;
    } else {
      value = buf_[pos_++];
    }
    return ret;
  }

  int read_varint(uint64_t &value)
  {
    int ret = OB_SUCCESS;
    uint64_t result = 0;
    int32_t shift = 0;
    bool done = false;
    uint8_t byte = 0;
    while (OB_SUCC(ret) && !done) {
      if (shift >= 70) {
        ret = OB_INVALID_DATA;
      } else if (OB_FAIL(read_byte(byte))) {
      } else {
        result |= (static_cast<uint64_t>(byte & 0x7f) << shift);
        shift += 7;
        done = (0 == (byte & 0x80));
      }
    }
    if (OB_SUCC(ret)) {
      value = result;
    }
    return ret;
  }

  int read_zigzag(int64_t &value)
  {
    int ret = OB_SUCCESS;
    uint64_t raw = 0;
    if (OB_FAIL(read_varint(raw))) {
    } else {
      value = static_cast<int64_t>(raw >> 1) ^ (-static_cast<int64_t>(raw & 1));
    }
    return ret;
  }

  int read_binary(const uint8_t *&ptr, int64_t &value_len)
  {
    int ret = OB_SUCCESS;
    uint64_t size = 0;
    ptr = NULL;
    value_len = 0;
    if (OB_FAIL(read_varint(size))) {
    } else if (pos_ < 0 || pos_ > len_ ||
               size > static_cast<uint64_t>(len_ - pos_)) {
      ret = OB_INVALID_DATA;
    } else {
      ptr = buf_ + pos_;
      value_len = static_cast<int64_t>(size);
      pos_ += value_len;
    }
    return ret;
  }

  int read_list_header(int64_t &count, int32_t &elem_type)
  {
    int ret = OB_SUCCESS;
    uint8_t byte = 0;
    count = 0;
    elem_type = 0;
    if (OB_FAIL(read_byte(byte))) {
    } else {
      elem_type = byte & 0x0f;
      count = (byte >> 4) & 0x0f;
      if (15 == count) {
        uint64_t size = 0;
        if (OB_FAIL(read_varint(size))) {
        } else {
          count = static_cast<int64_t>(size);
        }
      }
    }
    return ret;
  }

  int read_field(int64_t &last_id, int64_t &field_id, int32_t &type)
  {
    int ret = OB_SUCCESS;
    uint8_t byte = 0;
    field_id = 0;
    type = FT_T_STOP;
    if (OB_FAIL(read_byte(byte))) {
    } else {
      type = byte & 0x0f;
      const int32_t delta = (byte >> 4) & 0x0f;
      if (0 != delta) {
        field_id = last_id + delta;
      } else if (FT_T_STOP != type) {
        if (OB_FAIL(read_zigzag(field_id))) {
        }
      }
    }
    return ret;
  }

  int skip(const int32_t type)
  {
    int ret = OB_SUCCESS;
    if (FT_T_STOP == type || FT_T_BOOL_TRUE == type || FT_T_BOOL_FALSE == type) {
      // no payload
    } else if (FT_T_BYTE == type) {
      if (pos_ < 0 || pos_ >= len_) {
        ret = OB_INVALID_DATA;
      } else {
        ++pos_;
      }
    } else if (FT_T_I16 == type || FT_T_I32 == type || FT_T_I64 == type) {
      int64_t value = 0;
      ret = read_zigzag(value);
    } else if (FT_T_DOUBLE == type) {
      if (pos_ < 0 || pos_ + 8 > len_) {
        ret = OB_INVALID_DATA;
      } else {
        pos_ += 8;
      }
    } else if (FT_T_BINARY == type) {
      const uint8_t *ptr = NULL;
      int64_t value_len = 0;
      ret = read_binary(ptr, value_len);
    } else if (FT_T_LIST == type || FT_T_SET == type) {
      int64_t count = 0;
      int32_t elem_type = 0;
      if (OB_FAIL(read_list_header(count, elem_type))) {
      } else {
        for (int64_t i = 0; OB_SUCC(ret) && i < count; ++i) {
          ret = skip(elem_type);
        }
      }
    } else if (FT_T_MAP == type) {
      uint64_t count = 0;
      if (OB_FAIL(read_varint(count))) {
      } else if (count > 0) {
        uint8_t byte = 0;
        if (OB_FAIL(read_byte(byte))) {
        } else {
          const int32_t key_type = (byte >> 4) & 0x0f;
          const int32_t val_type = byte & 0x0f;
          for (uint64_t i = 0; OB_SUCC(ret) && i < count; ++i) {
            if (OB_FAIL(skip(key_type))) {
            } else {
              ret = skip(val_type);
            }
          }
        }
      }
    } else if (FT_T_STRUCT == type) {
      int64_t last_id = 0;
      int64_t field_id = 0;
      int32_t field_type = 0;
      while (OB_SUCC(ret)) {
        if (OB_FAIL(read_field(last_id, field_id, field_type))) {
        } else if (FT_T_STOP == field_type) {
          break;
        } else {
          last_id = field_id;
          ret = skip(field_type);
        }
      }
    } else {
      ret = OB_INVALID_DATA;
      LOG_WARN("unknown thrift compact type", K(type), K(ret));
    }
    return ret;
  }

  const uint8_t *buf_;
  int64_t len_;
  int64_t pos_;
};

// snappy block format: a varint with the uncompressed length, then a sequence
// of literal and copy operations.
inline int ft_snappy_decompress(const uint8_t *in,
                               const int64_t in_len,
                               uint8_t *out,
                               const int64_t out_cap,
                               int64_t &out_len)
{
  int ret = OB_SUCCESS;
  int64_t pos = 0;
  int64_t written = 0;
  // leading varint: uncompressed length
  uint64_t expect = 0;
  int32_t shift = 0;
  bool done = false;
  while (OB_SUCC(ret) && !done && pos < in_len) {
    const uint8_t byte = in[pos++];
    expect |= (static_cast<uint64_t>(byte & 0x7f) << shift);
    shift += 7;
    done = (0 == (byte & 0x80));
  }
  if (OB_UNLIKELY(!done)) {
    ret = OB_INVALID_DATA;
  } else if (expect > static_cast<uint64_t>(out_cap)) {
    ret = OB_SIZE_OVERFLOW;
  } else {
    while (OB_SUCC(ret) && pos < in_len) {
      const uint8_t tag = in[pos++];
      const int32_t kind = tag & 0x03;
      if (0 == kind) {
        int64_t len = tag >> 2;
        if (len < 60) {
          ++len;
        } else {
          const int32_t extra = static_cast<int32_t>(len) - 59;
          if (pos + extra > in_len) {
            ret = OB_INVALID_DATA;
          } else {
            len = 0;
            for (int32_t i = 0; i < extra; ++i) {
              len |= (static_cast<int64_t>(in[pos + i]) << (8 * i));
            }
            ++len;
            pos += extra;
          }
        }
        if (OB_FAIL(ret)) {
        } else if (pos + len > in_len || written + len > out_cap) {
          ret = OB_INVALID_DATA;
        } else {
          MEMCPY(out + written, in + pos, len);
          written += len;
          pos += len;
        }
      } else {
        int64_t copy_len = 0;
        int64_t offset = 0;
        if (1 == kind) {
          if (pos + 1 > in_len) {
            ret = OB_INVALID_DATA;
          } else {
            copy_len = ((tag >> 2) & 0x07) + 4;
            offset = (static_cast<int64_t>(tag >> 5) << 8) | in[pos];
            pos += 1;
          }
        } else if (2 == kind) {
          if (pos + 2 > in_len) {
            ret = OB_INVALID_DATA;
          } else {
            copy_len = (tag >> 2) + 1;
            offset = static_cast<int64_t>(in[pos]) | (static_cast<int64_t>(in[pos + 1]) << 8);
            pos += 2;
          }
        } else {
          if (pos + 4 > in_len) {
            ret = OB_INVALID_DATA;
          } else {
            copy_len = (tag >> 2) + 1;
            offset = static_cast<int64_t>(in[pos]) |
                     (static_cast<int64_t>(in[pos + 1]) << 8) |
                     (static_cast<int64_t>(in[pos + 2]) << 16) |
                     (static_cast<int64_t>(in[pos + 3]) << 24);
            pos += 4;
          }
        }
        if (OB_FAIL(ret)) {
        } else if (offset <= 0 || offset > written || written + copy_len > out_cap) {
          ret = OB_INVALID_DATA;
        } else {
          for (int64_t i = 0; i < copy_len; ++i) {
            out[written + i] = out[written - offset + i];
          }
          written += copy_len;
        }
      }
    }
  }
  if (OB_SUCC(ret) && static_cast<uint64_t>(written) != expect) {
    ret = OB_INVALID_DATA;
    LOG_WARN("snappy output size mismatch", K(written), K(expect), K(ret));
  }
  if (OB_SUCC(ret)) {
    out_len = written;
  }
  return ret;
}

inline int ft_bit_width(const int32_t max_level)
{
  int32_t width = 0;
  int32_t value = max_level;
  while (value > 0) {
    ++width;
    value >>= 1;
  }
  return width;
}

// RLE / bit packed hybrid, the level encoding used by parquet.
inline int ft_parquet_read_levels(const uint8_t *buf,
                                  const int64_t len,
                                  const int32_t bit_width,
                                  int32_t *levels,
                                  const int64_t level_cnt)
{
  int ret = OB_SUCCESS;
  int64_t pos = 0;
  int64_t produced = 0;
  if (0 == bit_width) {
    for (int64_t i = 0; i < level_cnt; ++i) {
      levels[i] = 0;
    }
    produced = level_cnt;
  }
  while (OB_SUCC(ret) && pos < len && produced < level_cnt) {
    // uleb128 run header
    uint64_t header = 0;
    int32_t shift = 0;
    bool done = false;
    while (OB_SUCC(ret) && !done && pos < len) {
      const uint8_t byte = buf[pos++];
      header |= (static_cast<uint64_t>(byte & 0x7f) << shift);
      shift += 7;
      done = (0 == (byte & 0x80));
    }
    if (OB_UNLIKELY(!done)) {
      ret = OB_INVALID_DATA;
    } else if (0 == bit_width) {
      // nothing to decode, filled above
    } else if (header & 0x01) {
      // bit packed run: (header >> 1) groups of 8 values
      const int64_t group_cnt = static_cast<int64_t>(header >> 1);
      const int64_t byte_cnt = group_cnt * bit_width;
      if (pos + byte_cnt > len) {
        ret = OB_INVALID_DATA;
      } else {
        const int64_t value_cnt = group_cnt * 8;
        for (int64_t i = 0; OB_SUCC(ret) && i < value_cnt && produced < level_cnt; ++i) {
          int32_t value = 0;
          for (int32_t b = 0; b < bit_width; ++b) {
            const int64_t bit = i * bit_width + b;
            const int64_t byte_idx = bit >> 3;
            if (byte_idx < byte_cnt && (buf[pos + byte_idx] & (1 << (bit & 7)))) {
              value |= (1 << b);
            }
          }
          levels[produced++] = value;
        }
        pos += byte_cnt;
      }
    } else {
      // repeated run: (header >> 1) copies of one value
      const int64_t repeat = static_cast<int64_t>(header >> 1);
      const int64_t byte_cnt = (bit_width + 7) / 8;
      if (pos + byte_cnt > len) {
        ret = OB_INVALID_DATA;
      } else {
        int32_t value = 0;
        for (int64_t i = 0; i < byte_cnt; ++i) {
          value |= (static_cast<int32_t>(buf[pos + i]) << (8 * i));
        }
        pos += byte_cnt;
        for (int64_t i = 0; i < repeat && produced < level_cnt; ++i) {
          levels[produced++] = value;
        }
      }
    }
  }
  if (OB_SUCC(ret) && produced != level_cnt) {
    ret = OB_INVALID_DATA;
    LOG_WARN("parquet level count mismatch", K(produced), K(level_cnt), K(ret));
  }
  return ret;
}

struct ObFtParquetColumn
{
  ObFtParquetColumn()
    : name_(), physical_type_(-1), repetition_(0), codec_(0), num_values_(0),
      data_page_offset_(0) {}
  common::ObString name_;
  int32_t physical_type_;
  int32_t repetition_;
  int32_t codec_;
  int64_t num_values_;
  int64_t data_page_offset_;
};

inline int ft_parquet_field_is_int(const int32_t type)
{
  return FT_T_I16 == type || FT_T_I32 == type || FT_T_I64 == type;
}

inline int ft_parquet_read_schema_element(ObFtThrift &reader, ObFtParquetColumn &column)
{
  int ret = OB_SUCCESS;
  int64_t last_id = 0;
  int64_t field_id = 0;
  int32_t type = 0;
  while (OB_SUCC(ret)) {
    if (OB_FAIL(reader.read_field(last_id, field_id, type))) {
    } else if (FT_T_STOP == type) {
      break;
    } else {
      last_id = field_id;
      if (1 == field_id && ft_parquet_field_is_int(type)) {
        int64_t value = 0;
        if (OB_FAIL(reader.read_zigzag(value))) {
        } else {
          column.physical_type_ = static_cast<int32_t>(value);
        }
      } else if (3 == field_id && ft_parquet_field_is_int(type)) {
        int64_t value = 0;
        if (OB_FAIL(reader.read_zigzag(value))) {
        } else {
          column.repetition_ = static_cast<int32_t>(value);
        }
      } else if (4 == field_id && FT_T_BINARY == type) {
        const uint8_t *ptr = NULL;
        int64_t value_len = 0;
        if (OB_FAIL(reader.read_binary(ptr, value_len))) {
        } else {
          column.name_.assign_ptr(reinterpret_cast<const char *>(ptr),
                                  static_cast<int32_t>(value_len));
        }
      } else {
        ret = reader.skip(type);
      }
    }
  }
  return ret;
}

inline int ft_parquet_read_column_meta(ObFtThrift &reader, ObFtParquetColumn &column)
{
  int ret = OB_SUCCESS;
  int64_t last_id = 0;
  int64_t field_id = 0;
  int32_t type = 0;
  while (OB_SUCC(ret)) {
    if (OB_FAIL(reader.read_field(last_id, field_id, type))) {
    } else if (FT_T_STOP == type) {
      break;
    } else {
      last_id = field_id;
      if (ft_parquet_field_is_int(type) && (4 == field_id || 5 == field_id || 9 == field_id)) {
        int64_t value = 0;
        if (OB_FAIL(reader.read_zigzag(value))) {
        } else if (4 == field_id) {
          column.codec_ = static_cast<int32_t>(value);
        } else if (5 == field_id) {
          column.num_values_ = value;
        } else {
          column.data_page_offset_ = value;
        }
      } else {
        ret = reader.skip(type);
      }
    }
  }
  return ret;
}

inline int ft_parquet_read_column_chunk(ObFtThrift &reader, ObFtParquetColumn &column)
{
  int ret = OB_SUCCESS;
  int64_t last_id = 0;
  int64_t field_id = 0;
  int32_t type = 0;
  while (OB_SUCC(ret)) {
    if (OB_FAIL(reader.read_field(last_id, field_id, type))) {
    } else if (FT_T_STOP == type) {
      break;
    } else {
      last_id = field_id;
      if (3 == field_id && FT_T_STRUCT == type) {
        ret = ft_parquet_read_column_meta(reader, column);
      } else {
        ret = reader.skip(type);
      }
    }
  }
  return ret;
}

inline int ft_parquet_read_row_group(ObFtThrift &reader,
                                     common::ObIArray<ObFtParquetColumn> &columns)
{
  int ret = OB_SUCCESS;
  int64_t last_id = 0;
  int64_t field_id = 0;
  int32_t type = 0;
  while (OB_SUCC(ret)) {
    if (OB_FAIL(reader.read_field(last_id, field_id, type))) {
    } else if (FT_T_STOP == type) {
      break;
    } else {
      last_id = field_id;
      if (1 == field_id && FT_T_LIST == type) {
        int64_t count = 0;
        int32_t elem_type = 0;
        if (OB_FAIL(reader.read_list_header(count, elem_type))) {
        }
        for (int64_t i = 0; OB_SUCC(ret) && i < count; ++i) {
          if (OB_UNLIKELY(i >= columns.count())) {
            ret = OB_NOT_SUPPORTED;
            LOG_WARN("parquet file has more chunks than leaf columns", K(count), K(ret));
          } else {
            ret = ft_parquet_read_column_chunk(reader, columns.at(i));
          }
        }
      } else {
        ret = reader.skip(type);
      }
    }
  }
  return ret;
}

// Reads FileMetaData from the footer and keeps only the leaf column
// descriptors of the first row group.
inline int ft_parquet_read_meta(const uint8_t *buf,
                                const int64_t len,
                                common::ObIArray<ObFtParquetColumn> &columns,
                                int64_t &row_cnt)
{
  int ret = OB_SUCCESS;
  ObFtThrift reader;
  int64_t last_id = 0;
  int64_t field_id = 0;
  int32_t type = 0;
  bool got_row_group = false;
  row_cnt = 0;
  if (len < 12) {
    ret = OB_INVALID_DATA;
  } else {
    const int64_t meta_len = static_cast<int64_t>(buf[len - 8]) |
                             (static_cast<int64_t>(buf[len - 7]) << 8) |
                             (static_cast<int64_t>(buf[len - 6]) << 16) |
                             (static_cast<int64_t>(buf[len - 5]) << 24);
    if (meta_len <= 0 || meta_len > len - 8) {
      ret = OB_INVALID_DATA;
      LOG_WARN("parquet footer length is invalid", K(meta_len), K(len), K(ret));
    } else {
      reader.assign(buf, len, len - 8 - meta_len);
    }
  }
  while (OB_SUCC(ret)) {
    if (OB_FAIL(reader.read_field(last_id, field_id, type))) {
    } else if (FT_T_STOP == type) {
      break;
    } else {
      last_id = field_id;
      if (2 == field_id && FT_T_LIST == type) {
        // schema: the first element is the root group, the rest are leaves
        int64_t count = 0;
        int32_t elem_type = 0;
        if (OB_FAIL(reader.read_list_header(count, elem_type))) {
        }
        for (int64_t i = 0; OB_SUCC(ret) && i < count; ++i) {
          ObFtParquetColumn column;
          if (OB_FAIL(ft_parquet_read_schema_element(reader, column))) {
          } else if (column.name_.length() > 0 && column.physical_type_ >= 0) {
            if (OB_FAIL(columns.push_back(column))) {
            }
          }
        }
      } else if (3 == field_id && ft_parquet_field_is_int(type)) {
        int64_t value = 0;
        if (OB_FAIL(reader.read_zigzag(value))) {
        } else {
          row_cnt = value;
        }
      } else if (4 == field_id && FT_T_LIST == type) {
        int64_t count = 0;
        int32_t elem_type = 0;
        if (OB_FAIL(reader.read_list_header(count, elem_type))) {
        } else if (count > 0) {
          if (OB_FAIL(ft_parquet_read_row_group(reader, columns))) {
          } else {
            got_row_group = true;
          }
          for (int64_t i = 1; OB_SUCC(ret) && i < count; ++i) {
            ret = reader.skip(elem_type);
          }
        }
      } else {
        ret = reader.skip(type);
      }
    }
  }
  if (OB_SUCC(ret) && (!got_row_group || columns.count() <= 0 || row_cnt <= 0)) {
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("parquet file is not a flat single row group file",
             K(got_row_group), K(columns.count()), K(row_cnt), K(ret));
  }
  return ret;
}

struct ObFtPageHeader
{
  ObFtPageHeader()
    : page_type_(0), uncompressed_size_(0), compressed_size_(0), num_values_(0),
      encoding_(0), def_level_len_(0), rep_level_len_(0), is_compressed_(false),
      is_v2_(false) {}
  int32_t page_type_;
  int32_t uncompressed_size_;
  int32_t compressed_size_;
  int32_t num_values_;
  int32_t encoding_;
  int32_t def_level_len_;
  int32_t rep_level_len_;
  bool is_compressed_;
  bool is_v2_;
};

inline int ft_parquet_read_data_page_header_v1(ObFtThrift &reader, ObFtPageHeader &header)
{
  int ret = OB_SUCCESS;
  int64_t last_id = 0;
  int64_t field_id = 0;
  int32_t type = 0;
  while (OB_SUCC(ret)) {
    if (OB_FAIL(reader.read_field(last_id, field_id, type))) {
    } else if (FT_T_STOP == type) {
      break;
    } else {
      last_id = field_id;
      if (ft_parquet_field_is_int(type) && (1 == field_id || 2 == field_id)) {
        int64_t value = 0;
        if (OB_FAIL(reader.read_zigzag(value))) {
        } else if (1 == field_id) {
          header.num_values_ = static_cast<int32_t>(value);
        } else {
          header.encoding_ = static_cast<int32_t>(value);
        }
      } else {
        ret = reader.skip(type);
      }
    }
  }
  return ret;
}

inline int ft_parquet_read_data_page_header_v2(ObFtThrift &reader, ObFtPageHeader &header)
{
  int ret = OB_SUCCESS;
  int64_t last_id = 0;
  int64_t field_id = 0;
  int32_t type = 0;
  while (OB_SUCC(ret)) {
    if (OB_FAIL(reader.read_field(last_id, field_id, type))) {
    } else if (FT_T_STOP == type) {
      break;
    } else {
      last_id = field_id;
      if (7 == field_id && (FT_T_BOOL_TRUE == type || FT_T_BOOL_FALSE == type)) {
        header.is_compressed_ = (FT_T_BOOL_TRUE == type);
      } else if (ft_parquet_field_is_int(type) && field_id >= 1 && field_id <= 6) {
        int64_t value = 0;
        if (OB_FAIL(reader.read_zigzag(value))) {
        } else if (1 == field_id) {
          header.num_values_ = static_cast<int32_t>(value);
        } else if (4 == field_id) {
          header.encoding_ = static_cast<int32_t>(value);
        } else if (5 == field_id) {
          header.def_level_len_ = static_cast<int32_t>(value);
        } else if (6 == field_id) {
          header.rep_level_len_ = static_cast<int32_t>(value);
        }
      } else {
        ret = reader.skip(type);
      }
    }
  }
  return ret;
}

inline int ft_parquet_read_page_header(ObFtThrift &reader, ObFtPageHeader &header)
{
  int ret = OB_SUCCESS;
  int64_t last_id = 0;
  int64_t field_id = 0;
  int32_t type = 0;
  while (OB_SUCC(ret)) {
    if (OB_FAIL(reader.read_field(last_id, field_id, type))) {
    } else if (FT_T_STOP == type) {
      break;
    } else {
      last_id = field_id;
      if (ft_parquet_field_is_int(type) && field_id >= 1 && field_id <= 4) {
        int64_t value = 0;
        if (OB_FAIL(reader.read_zigzag(value))) {
        } else if (1 == field_id) {
          header.page_type_ = static_cast<int32_t>(value);
        } else if (2 == field_id) {
          header.uncompressed_size_ = static_cast<int32_t>(value);
        } else if (3 == field_id) {
          header.compressed_size_ = static_cast<int32_t>(value);
        }
      } else if (5 == field_id && FT_T_STRUCT == type) {
        ret = ft_parquet_read_data_page_header_v1(reader, header);
      } else if (8 == field_id && FT_T_STRUCT == type) {
        header.is_v2_ = true;
        ret = ft_parquet_read_data_page_header_v2(reader, header);
      } else {
        ret = reader.skip(type);
      }
    }
  }
  return ret;
}

inline int ft_parquet_read_plain_value(common::ObIAllocator &alloc,
                                       const ColumnKind kind,
                                       const int32_t physical_type,
                                       const uint8_t *buf,
                                       const int64_t len,
                                       int64_t &pos,
                                       int64_t &bool_bit,
                                       common::ObObjParam &obj)
{
  int ret = OB_SUCCESS;
  obj.reset();
  if (1 == physical_type) {
    // INT32
    if (pos + 4 > len) {
      ret = OB_INVALID_DATA;
    } else {
      const int32_t value = static_cast<int32_t>(
          static_cast<uint32_t>(buf[pos]) |
          (static_cast<uint32_t>(buf[pos + 1]) << 8) |
          (static_cast<uint32_t>(buf[pos + 2]) << 16) |
          (static_cast<uint32_t>(buf[pos + 3]) << 24));
      pos += 4;
      obj.set_int(value);
      obj.set_param_meta();
    }
  } else if (2 == physical_type) {
    // INT64
    if (pos + 8 > len) {
      ret = OB_INVALID_DATA;
    } else {
      uint64_t raw = 0;
      for (int32_t i = 0; i < 8; ++i) {
        raw |= (static_cast<uint64_t>(buf[pos + i]) << (8 * i));
      }
      pos += 8;
      obj.set_int(static_cast<int64_t>(raw));
      obj.set_param_meta();
    }
  } else if (4 == physical_type || 5 == physical_type) {
    // FLOAT / DOUBLE
    const int32_t width = (4 == physical_type) ? 4 : 8;
    if (pos + width > len) {
      ret = OB_INVALID_DATA;
    } else {
      uint64_t raw = 0;
      for (int32_t i = 0; i < width; ++i) {
        raw |= (static_cast<uint64_t>(buf[pos + i]) << (8 * i));
      }
      pos += width;
      double value = 0.0;
      if (4 == width) {
        const uint32_t bits = static_cast<uint32_t>(raw);
        float as_float = 0.0f;
        MEMCPY(&as_float, &bits, 4);
        value = static_cast<double>(as_float);
      } else {
        MEMCPY(&value, &raw, 8);
      }
      obj.set_double(value);
      obj.set_param_meta();
    }
  } else if (6 == physical_type) {
    // BYTE_ARRAY: 4 byte length then the bytes
    if (pos + 4 > len) {
      ret = OB_INVALID_DATA;
    } else {
      const int64_t str_len = static_cast<int64_t>(
          static_cast<uint32_t>(buf[pos]) |
          (static_cast<uint32_t>(buf[pos + 1]) << 8) |
          (static_cast<uint32_t>(buf[pos + 2]) << 16) |
          (static_cast<uint32_t>(buf[pos + 3]) << 24));
      pos += 4;
      if (str_len < 0 || pos + str_len > len) {
        ret = OB_INVALID_DATA;
      } else {
        char *copy = static_cast<char *>(alloc.alloc(str_len + 1));
        if (OB_ISNULL(copy)) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          MEMCPY(copy, buf + pos, str_len);
          pos += str_len;
          obj.set_varchar(common::ObString(static_cast<int32_t>(str_len), copy));
          obj.set_collation_type(common::CS_TYPE_UTF8MB4_GENERAL_CI);
          obj.set_collation_level(common::CS_LEVEL_IMPLICIT);
          obj.set_param_meta();
        }
      }
    }
  } else if (0 == physical_type) {
    // BOOLEAN is bit packed, least significant bit first
    const int64_t bit_idx = bool_bit++;
    const int64_t byte_idx = bit_idx >> 3;
    if (byte_idx >= len) {
      ret = OB_INVALID_DATA;
    } else {
      obj.set_tinyint((buf[byte_idx] & (1 << (bit_idx & 7))) ? 1 : 0);
      obj.set_param_meta();
    }
  } else {
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("unsupported parquet physical type", K(physical_type), K(kind), K(ret));
  }
  return ret;
}

inline int ft_parquet_load_column(common::ObIAllocator &alloc,
                                  const uint8_t *buf,
                                  const int64_t len,
                                  const ObFtParquetColumn &column,
                                  const int64_t row_cnt,
                                  const int64_t col_idx,
                                  const int64_t col_total,
                                  ObFileTableData &table)
{
  int ret = OB_SUCCESS;
  const int32_t max_def_level = (1 == column.repetition_) ? 1 : 0;
  const int32_t bit_width = ft_bit_width(max_def_level);
  int64_t page_pos = column.data_page_offset_;
  int64_t rows_done = 0;
  while (OB_SUCC(ret) && rows_done < row_cnt) {
    ObFtPageHeader page;
    ObFtThrift reader;
    reader.assign(buf, len, page_pos);
    if (OB_FAIL(ft_parquet_read_page_header(reader, page))) {
      LOG_WARN("failed to read parquet page header", K(page_pos), K(ret));
    } else if (2 == page.page_type_) {
      ret = OB_NOT_SUPPORTED;
      LOG_WARN("parquet dictionary pages are not supported", K(ret));
    } else if (0 != page.page_type_ && 3 != page.page_type_) {
      ret = OB_NOT_SUPPORTED;
      LOG_WARN("unknown parquet page type", K(page.page_type_), K(ret));
    } else if (page.compressed_size_ <= 0 || page.uncompressed_size_ <= 0 ||
               page.num_values_ < 0 || reader.pos_ + page.compressed_size_ > len) {
      ret = OB_INVALID_DATA;
      LOG_WARN("parquet page sizes are invalid", K(page.compressed_size_),
               K(page.uncompressed_size_), K(reader.pos_), K(len), K(ret));
    } else {
      const uint8_t *payload = buf + reader.pos_;
      int64_t payload_len = page.compressed_size_;
      const uint8_t *level_ptr = NULL;
      int64_t level_len = 0;
      const uint8_t *value_ptr = NULL;
      int64_t value_len = 0;
      int64_t value_cap = 0;
      if (page.is_v2_) {
        level_len = page.def_level_len_ + page.rep_level_len_;
        if (level_len < 0 || level_len > payload_len ||
            level_len > page.uncompressed_size_) {
          ret = OB_INVALID_DATA;
        } else {
          level_ptr = payload;
          value_ptr = payload + level_len;
          value_len = payload_len - level_len;
          value_cap = page.uncompressed_size_ - level_len;
          if (page.is_compressed_ && 0 != column.codec_) {
            uint8_t *decompressed = NULL;
            if (1 != column.codec_) {
              ret = OB_NOT_SUPPORTED;
              LOG_WARN("unsupported parquet compression codec", K(column.codec_), K(ret));
            } else if (OB_ISNULL(decompressed = static_cast<uint8_t *>(
                                     alloc.alloc(value_cap > 0 ? value_cap : 1)))) {
              ret = OB_ALLOCATE_MEMORY_FAILED;
            } else {
              int64_t decoded = 0;
              if (OB_FAIL(ft_snappy_decompress(value_ptr, value_len, decompressed,
                                               value_cap, decoded))) {
                LOG_WARN("failed to decompress parquet page", K(ret));
              } else {
                value_ptr = decompressed;
                value_len = decoded;
              }
            }
          }
        }
      } else {
        // v1: the whole page is compressed as one block
        const uint8_t *page_data = payload;
        int64_t page_data_len = payload_len;
        if (0 != column.codec_) {
          uint8_t *decompressed = NULL;
          if (1 != column.codec_) {
            ret = OB_NOT_SUPPORTED;
            LOG_WARN("unsupported parquet compression codec", K(column.codec_), K(ret));
          } else if (OB_ISNULL(decompressed = static_cast<uint8_t *>(
                                   alloc.alloc(page.uncompressed_size_ > 0 ?
                                               page.uncompressed_size_ : 1)))) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else {
            int64_t decoded = 0;
            if (OB_FAIL(ft_snappy_decompress(payload, payload_len, decompressed,
                                             page.uncompressed_size_, decoded))) {
            } else {
              page_data = decompressed;
              page_data_len = decoded;
            }
          }
        }
        if (OB_SUCC(ret) && max_def_level > 0) {
          if (page_data_len < 4) {
            ret = OB_INVALID_DATA;
          } else {
            level_len = static_cast<int64_t>(page_data[0]) |
                        (static_cast<int64_t>(page_data[1]) << 8) |
                        (static_cast<int64_t>(page_data[2]) << 16) |
                        (static_cast<int64_t>(page_data[3]) << 24);
            if (level_len < 0 || 4 + level_len > page_data_len) {
              ret = OB_INVALID_DATA;
            } else {
              level_ptr = page_data + 4;
              value_ptr = page_data + 4 + level_len;
              value_len = page_data_len - 4 - level_len;
              value_cap = value_len;
            }
          }
        } else if (OB_SUCC(ret)) {
          level_len = 0;
          value_ptr = page_data;
          value_len = page_data_len;
          value_cap = value_len;
        }
      }
      int32_t *levels = NULL;
      if (OB_SUCC(ret) && page.num_values_ > 0) {
        if (max_def_level > 0) {
          if (OB_ISNULL(levels = static_cast<int32_t *>(
                            alloc.alloc(sizeof(int32_t) * page.num_values_)))) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else if (OB_FAIL(ft_parquet_read_levels(level_ptr, level_len, bit_width,
                                                    levels, page.num_values_))) {
            LOG_WARN("failed to decode parquet definition levels", K(ret));
          }
        } else {
          if (OB_ISNULL(levels = static_cast<int32_t *>(
                            alloc.alloc(sizeof(int32_t) * page.num_values_)))) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else {
            for (int32_t i = 0; i < page.num_values_; ++i) {
              levels[i] = 0;
            }
          }
        }
      }
      if (OB_SUCC(ret)) {
        int64_t value_pos = 0;
        int64_t bool_bit = 0;
        for (int32_t i = 0; OB_SUCC(ret) && i < page.num_values_; ++i) {
          const bool present = (max_def_level <= 0) || (levels[i] == max_def_level);
          ObObjParam obj;
          if (!present) {
            obj.set_null();
            obj.set_param_meta();
          } else if (OB_FAIL(ft_parquet_read_plain_value(alloc, table.columns_.at(col_idx).kind_,
                                                         column.physical_type_, value_ptr,
                                                         value_len, value_pos, bool_bit, obj))) {
            LOG_WARN("failed to decode parquet value", K(i), K(ret));
          }
          if (OB_SUCC(ret)) {
            if (OB_UNLIKELY(rows_done >= row_cnt)) {
              ret = OB_INVALID_DATA;
            } else {
              table.cells_.at(rows_done * col_total + col_idx) = obj;
              ++rows_done;
            }
          }
        }
      }
      if (OB_SUCC(ret)) {
        page_pos = reader.pos_ + page.compressed_size_;
      }
    }
  }
  return ret;
}

inline int load_parquet(common::ObIAllocator &alloc, const char *path, ObFileTableData &table)
{
  int ret = OB_SUCCESS;
  char *buf = NULL;
  int64_t len = 0;
  int64_t row_cnt = 0;
  common::ObArray<ObFtParquetColumn> columns;
  if (OB_FAIL(read_whole_file(alloc, path, buf, len))) {
  } else if (len < 12 || 0 != strncmp(buf, "PAR1", 4) ||
             0 != strncmp(buf + len - 4, "PAR1", 4)) {
    ret = OB_INVALID_DATA;
    LOG_WARN("file is not a parquet file", K(path), K(len), K(ret));
  } else if (OB_FAIL(ft_parquet_read_meta(reinterpret_cast<const uint8_t *>(buf), len,
                                          columns, row_cnt))) {
  } else if (row_cnt > FT_MAX_ROWS) {
    ret = OB_SIZE_OVERFLOW;
  } else {
    // the layout is row major, so reserve every slot first and let each
    // column fill its own stripe
    ObObjParam null_obj;
    null_obj.set_null();
    null_obj.set_param_meta();
    for (int64_t i = 0; OB_SUCC(ret) && i < columns.count(); ++i) {
      ObFileColumn column;
      column.name_ = columns.at(i).name_;
      column.nullable_ = (1 == columns.at(i).repetition_);
      switch (columns.at(i).physical_type_) {
        case 0: column.kind_ = COL_BOOLEAN; break;
        case 1: case 2: column.kind_ = COL_BIGINT; break;
        case 4: case 5: column.kind_ = COL_DOUBLE; break;
        case 6: column.kind_ = COL_VARCHAR; break;
        default:
          ret = OB_NOT_SUPPORTED;
          LOG_WARN("unsupported parquet column type", K(columns.at(i).physical_type_), K(ret));
          break;
      }
      if (OB_SUCC(ret) && OB_FAIL(table.columns_.push_back(column))) {
      }
    }
    const int64_t col_total = table.columns_.count();
    for (int64_t i = 0; OB_SUCC(ret) && i < row_cnt * col_total; ++i) {
      if (OB_FAIL(table.cells_.push_back(null_obj))) {
      }
    }
    for (int64_t c = 0; OB_SUCC(ret) && c < col_total; ++c) {
      if (OB_FAIL(ft_parquet_load_column(alloc, reinterpret_cast<const uint8_t *>(buf), len,
                                         columns.at(c), row_cnt, c, col_total, table))) {
        LOG_WARN("failed to load parquet column", K(c), K(ret));
      }
    }
    if (OB_SUCC(ret)) {
      table.row_cnt_ = row_cnt;
    }
  }
  return ret;
}

inline int ft_kind_to_obj_type(const ColumnKind kind, common::ObObjType &type)
{
  int ret = OB_SUCCESS;
  switch (kind) {
    case COL_BIGINT: type = common::ObIntType; break;
    case COL_DOUBLE: type = common::ObDoubleType; break;
    case COL_BOOLEAN: type = common::ObTinyIntType; break;
    case COL_DATE: type = common::ObDateType; break;
    case COL_DATETIME: type = common::ObDateTimeType; break;
    case COL_VARCHAR: type = common::ObVarcharType; break;
    default:
      ret = OB_ERR_UNEXPECTED;
      break;
  }
  return ret;
}

} // namespace filetable
} // namespace sql
} // namespace oceanbase

#endif // OCEANBASE_SQL_RESOLVER_DML_OB_FILE_TABLE_READER_H_

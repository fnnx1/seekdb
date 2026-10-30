/* Copyright (c) 2025 OceanBase. Licensed under the Apache License, Version 2.0. */
#define USING_LOG_PREFIX SQL_RESV
#include "sql/resolver/dml/ob_dml_resolver.h"
#include "sql/resolver/ob_resolver_utils.h"
#include "sql/engine/basic/file_scan_reader.h"
#include "share/system_variable/ob_sys_var_class_type.h"
#include <set>

namespace oceanbase { namespace sql {
using namespace common;

int ObDMLResolver::resolve_file_table_item(const ParseNode &node, TableItem *&table_item)
{
  int ret = OB_SUCCESS;
  ObObj enabled;
  ObString path, secure_path;
  FileReadOptions options;
  FileScanReader reader;
  std::string canonical, error, format;
  TableItem *item = nullptr;
  ObValuesTableDef *def = nullptr;
  if (OB_ISNULL(session_info_) || OB_ISNULL(allocator_) || OB_ISNULL(get_stmt())
      || node.num_child_ != 4 || !node.children_[0] || !node.children_[1]) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(session_info_->get_sys_variable(share::SYS_VAR_ENABLE_FILE_SQL, enabled))) {
  } else if (!enabled.get_bool()) {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "file SQL is disabled; SET SESSION enable_file_sql=ON first");
  } else if (node.children_[1]->type_ != T_VARCHAR && node.children_[1]->type_ != T_QUESTIONMARK) {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "file SQL requires a literal local path");
  } else if (OB_FAIL(resolve_str_const(*node.children_[1], path))) {
  } else {
    format.assign(node.children_[0]->str_value_, node.children_[0]->str_len_);
    if (format != "csv" && format != "jsonl" && format != "glob" && format != "parquet") {
      ret = OB_NOT_SUPPORTED;
      LOG_USER_ERROR(OB_NOT_SUPPORTED, "this file scan format");
    }
  }
  std::set<std::string> seen_options;
  const ParseNode *option_list = node.children_[2];
  for (int64_t i = 0; OB_SUCC(ret) && option_list && i < option_list->num_child_; ++i) {
    const ParseNode *option = option_list->children_[i];
    if (!option || option->num_child_ != 2 || !option->children_[0] || !option->children_[1]) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      const ParseNode *value = option->children_[1];
      std::string key(option->children_[0]->str_value_, option->children_[0]->str_len_);
      for (char &c : key) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
      if (!seen_options.insert(key).second) {
        ret = OB_INVALID_ARGUMENT;
      } else if (key == "sample_size") {
        if (!(value->type_ == T_OP_NEG && value->num_child_ == 1 && value->children_[0]
              && value->children_[0]->type_ == T_INT && value->children_[0]->value_ == 1)) {
          ret = OB_NOT_SUPPORTED;
          LOG_USER_ERROR(OB_NOT_SUPPORTED, "file SQL sample_size other than -1");
        }
      } else if (key == "header" && format == "csv") {
        if ((value->type_ != T_BOOL && value->type_ != T_INT) || value->value_ < 0 || value->value_ > 1) ret = OB_INVALID_ARGUMENT;
        else options.header = value->value_ != 0;
      } else if (value->type_ == T_VARCHAR || value->type_ == T_QUESTIONMARK) {
        ObString str;
        if (OB_FAIL(resolve_str_const(*value, str))) {
        } else if (key == "format" && format == "jsonl" && str.case_compare("newline_delimited") == 0) {
        } else if (key == "nullstr" && format == "csv") {
          options.null_string.assign(str.ptr(), str.length());
        } else if ((key == "delim" || key == "delimiter" || key == "quote" || key == "escape") && format == "csv" && str.length() == 1) {
          if (key == "quote") options.quote = str.ptr()[0];
          else if (key == "escape") options.escape = str.ptr()[0];
          else options.delimiter = str.ptr()[0];
        } else {
          ret = OB_NOT_SUPPORTED;
        }
      } else {
        ret = OB_NOT_SUPPORTED;
      }
    }
  }
  if (OB_SUCC(ret)) {
    const std::string raw_path(path.ptr() ? path.ptr() : "", path.length());
    const bool resolved = format == "glob"
        ? FileScanReader::canonical_glob(raw_path, canonical, error)
        : FileScanReader::canonical_path(raw_path, canonical, error);
    if (!resolved) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, error.c_str());
    } else if (OB_FAIL(session_info_->get_secure_file_priv(secure_path))) {
    } else if (OB_FAIL(ObResolverUtils::check_secure_path(secure_path, ObString(static_cast<int32_t>(canonical.size()), canonical.data())))) {
    } else if (!reader.open(canonical, format, options) || !reader.infer()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, reader.error().c_str());
    } else if (OB_ISNULL(item = get_stmt()->create_table_item(*allocator_))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("file_sql create item failed");
    } else {
      void *memory = allocator_->alloc(sizeof(ObValuesTableDef));
      if (!memory) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("file_sql def alloc failed");
      } else { def = new (memory) ObValuesTableDef(); item->values_table_def_ = def; }
    }
  }
  const auto add_metadata = [&](const std::string &text) -> int {
    ObString copied;
    int code = ob_write_string(*allocator_, ObString(static_cast<int32_t>(text.size()), text.data()), copied);
    if (code == OB_SUCCESS) {
      ObObjParam value;
      value.set_varchar(copied);
      value.set_collation_type(CS_TYPE_UTF8MB4_BIN);
      value.set_param_meta();
      code = def->access_objs_.push_back(value);
    }
    return code;
  };
  if (OB_SUCC(ret)) {
    def->access_type_ = ObValuesTableDef::ACCESS_FILE;
    def->column_cnt_ = reader.columns().size();
    def->row_cnt_ = reader.row_count();
    item->table_id_ = generate_table_id();
    item->type_ = TableItem::VALUES_TABLE;
    ObString alias;
    if (node.children_[3]) alias.assign_ptr(node.children_[3]->str_value_, static_cast<int32_t>(node.children_[3]->str_len_));
    else if (OB_FAIL(get_stmt()->generate_values_table_name(*allocator_, alias))) {}
    if (OB_SUCC(ret)) ret = ob_write_string(*allocator_, alias, item->table_name_);
    item->alias_name_ = item->table_name_;
    const std::string metadata[] = {canonical, format, options.header ? "1" : "0",
      std::string(1, options.delimiter), std::string(1, options.quote), std::string(1, options.escape),
      options.null_string, reader.signature()};
    for (const auto &value : metadata) if (OB_SUCC(ret)) ret = add_metadata(value);
    for (const auto &column : reader.columns()) if (OB_SUCC(ret)) ret = add_metadata(column.name);
    if (OB_SUCC(ret)) ret = get_stmt()->add_table_item(session_info_, item);
  }
  for (int64_t i = 0; OB_SUCC(ret) && i < def->column_cnt_; ++i) {
    const FileColumn &column = reader.columns()[i];
    ObRawExprResType type;
    switch (column.type) {
      case FileColumnType::BIGINT: type.set_type(ObIntType); break;
      case FileColumnType::DOUBLE: type.set_type(ObDoubleType); break;
      case FileColumnType::BOOLEAN: type.set_type(ObTinyIntType); break;
      case FileColumnType::DATE: type.set_type(ObDateType); break;
      case FileColumnType::DATETIME: type.set_type(ObDateTimeType); break;
      default: type.set_type(ObVarcharType); break;
    }
    type.set_accuracy(ObAccuracy::DDL_DEFAULT_ACCURACY2[0][type.get_type()]);
    if (column.type == FileColumnType::DATETIME) {
      // Preserve the inferred precision in the MySQL result metadata. The DDL
      // default is six, which adds fractional zeros even to second-only files.
      type.set_scale(column.scale);
    }
    if (type.is_string_type()) {
      type.set_length(FileScanReader::MAX_FIELD_BYTES);
      type.set_collation_type(CS_TYPE_UTF8MB4_BIN);
      type.set_collation_level(CS_LEVEL_IMPLICIT);
    }
    ObColumnRefRawExpr *expr = nullptr;
    ObString name;
    if (OB_FAIL(def->column_types_.push_back(type))) {
    } else if (OB_FAIL(def->column_ndvs_.push_back(column.non_null_count))) {
    } else if (OB_FAIL(def->column_nnvs_.push_back(def->row_cnt_ - column.non_null_count))) {
    } else if (OB_FAIL(ob_write_string(*allocator_, ObString(static_cast<int32_t>(column.name.size()), column.name.data()), name))) {
    } else if (OB_FAIL(params_.expr_factory_->create_raw_expr(T_REF_COLUMN, expr))) {
    } else {
      expr->set_result_type(type);
      if (def->row_cnt_ > 0 && column.non_null_count == def->row_cnt_) {
        expr->set_result_flag(expr->get_result_flag() | NOT_NULL_FLAG);
      }
      expr->set_ref_id(item->table_id_, i + OB_APP_MIN_COLUMN_ID);
      expr->set_column_attr(item->table_name_, name);
      if (OB_FAIL(expr->add_flag(IS_COLUMN))) {
      } else {
        ColumnItem entry;
        entry.expr_ = expr;
        entry.table_id_ = item->table_id_;
        entry.column_id_ = expr->get_column_id();
        entry.column_name_ = name;
        ret = get_stmt()->add_column_item(entry);
      }
    }
  }
  if (OB_SUCC(ret)) {
    ObGlobalHint &hint = const_cast<ObGlobalHint &>(get_stmt()->get_query_ctx()->get_query_hint_for_update().get_global_hint());
    hint.plan_cache_policy_ = OB_USE_PLAN_CACHE_NONE;
    hint.parallel_ = 1;
    table_item = item;
  }
  return ret;
}

} }




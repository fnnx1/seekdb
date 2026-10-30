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

#define USING_LOG_PREFIX SQL_RESV
#include "sql/resolver/ddl/ob_explain_resolver.h"
#include "sql/resolver/ob_resolver.h"
#include "sql/parser/ob_parser.h"
#include "sql/engine/basic/file_scan_reader.h"
#include "sql/resolver/dml/ob_dml_stmt.h"
#include "sql/resolver/ob_resolver_utils.h"
#include "share/system_variable/ob_sys_var_class_type.h"
#include <set>

using namespace oceanbase;
using namespace sql;
using namespace common;
const static int64_t EXPLAIN_FORMAT = 0;
const static int64_t EXPLAIN_DISPLAY_OPTION = 1;
const static int64_t EXPLAIN_INTO_TABLE = 2;
const static int64_t EXPLAIN_STMT_ID = 3;
const static int64_t EXPLAIN_CHILD_STMT = 4;
namespace {
const ParseNode *find_file_table(const ParseNode *node)
{
  if (!node) return nullptr;
  if (node->type_ == T_FILE_TABLE_EXPRESSION) return node;
  for (int64_t i = 0; i < node->num_child_; ++i) {
    const ParseNode *found = find_file_table(node->children_[i]);
    if (found) return found;
  }
  return nullptr;
}
}

int ObExplainResolver::resolve_file_describe(const ParseNode &select_node)
{
  int ret = OB_SUCCESS;
  ObSelectStmt *stmt = nullptr;
  TableItem *item = nullptr;
  ObValuesTableDef *def = nullptr;
  ObStmt *resolved = nullptr;
  ObResolver resolver(params_);
  struct MetaColumn { std::string name; std::string type; bool not_null; };
  std::vector<MetaColumn> columns;
  // Resolve the entire projection: aliases, joins, invalid column references
  // and all file options must obey exactly the same rules as an ordinary SELECT.
  if (OB_FAIL(resolver.resolve(ObResolver::IS_NOT_PREPARED_STMT, select_node, resolved))) {
  } else if (!resolved || !resolved->is_select_stmt()) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    const ObSelectStmt &query = *static_cast<ObSelectStmt *>(resolved);
    for (int64_t i = 0; i < query.get_select_item_size(); ++i) {
      const SelectItem &selected = query.get_select_item(i);
      const ObString name = selected.alias_name_.empty() ? selected.expr_name_ : selected.alias_name_;
      const char *type = "VARCHAR";
      switch (selected.expr_->get_result_type().get_type()) {
        case ObIntType: type = "BIGINT"; break;
        case ObDoubleType: type = "DOUBLE"; break;
        case ObTinyIntType: type = "BOOLEAN"; break;
        case ObDateType: type = "DATE"; break;
        case ObDateTimeType: type = "DATETIME"; break;
        default: break;
      }
      columns.push_back({std::string(name.ptr(), name.length()), type,
          (selected.expr_->get_result_flag() & NOT_NULL_FLAG) != 0});
    }
  }
  if (OB_SUCC(ret)) {
    if (OB_ISNULL(stmt = create_stmt<ObSelectStmt>())) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (OB_ISNULL(item = ObDMLStmt::create_table_item(*allocator_))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      void *memory = allocator_->alloc(sizeof(ObValuesTableDef));
      if (!memory) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        def = new (memory) ObValuesTableDef();
        item->values_table_def_ = def;
      }
    }
  }
  if (OB_SUCC(ret)) {
    const int64_t column_count = static_cast<int64_t>(columns.size());
    static const char *META_NAMES[] = {"column_name", "column_type", "null", "key", "default", "extra"};
    const int64_t meta_columns = 6;
    item->table_id_ = generate_table_id();
    item->type_ = TableItem::VALUES_TABLE;
    if (OB_FAIL(ob_write_string(*allocator_, ObString::make_string("file_describe"), item->table_name_))) {
    } else if (FALSE_IT(item->alias_name_ = item->table_name_)) {
    } else if (OB_FAIL(stmt->add_table_item(session_info_, item))) {
    } else if (OB_FAIL(stmt->add_from_item(item->table_id_))) {
    }
    def->access_type_ = ObValuesTableDef::ACCESS_OBJ;
    def->is_const_ = true;
    def->column_cnt_ = meta_columns;
    def->row_cnt_ = column_count;
    ObRawExprResType meta_type;
    meta_type.set_type(ObVarcharType);
    meta_type.set_accuracy(ObAccuracy::DDL_DEFAULT_ACCURACY2[0][ObVarcharType]);
    meta_type.set_length(FileScanReader::MAX_FIELD_BYTES);
    meta_type.set_collation_type(CS_TYPE_UTF8MB4_BIN);
    meta_type.set_collation_level(CS_LEVEL_IMPLICIT);
    for (int64_t i = 0; OB_SUCC(ret) && i < meta_columns; ++i) {
      if (OB_FAIL(def->column_types_.push_back(meta_type))) {
      } else if (OB_FAIL(def->column_ndvs_.push_back(0))) {
      } else if (OB_FAIL(def->column_nnvs_.push_back(i >= 3 ? column_count : 0))) {
      }
    }
    for (int64_t row = 0; OB_SUCC(ret) && row < column_count; ++row) {
      const MetaColumn &column = columns[row];
      const std::string cells[] = {column.name, column.type, column.not_null ? "NO" : "YES"};
      for (int64_t col = 0; OB_SUCC(ret) && col < meta_columns; ++col) {
        ObObjParam value;
        if (col < 3) {
          ObString copied;
          const std::string &text = cells[col];
          if (OB_FAIL(ob_write_string(*allocator_, ObString(static_cast<int32_t>(text.size()), text.data()), copied))) {
          } else {
            value.set_varchar(copied);
            value.set_collation_type(CS_TYPE_UTF8MB4_BIN);
          }
        } else {
          value.set_null();
        }
        if (OB_SUCC(ret)) {
          value.set_param_meta();
          ret = def->access_objs_.push_back(value);
        }
      }
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < meta_columns; ++i) {
      ObString name;
      ObColumnRefRawExpr *expr = nullptr;
      if (OB_FAIL(ob_write_string(*allocator_, ObString::make_string(META_NAMES[i]), name))) {
      } else if (OB_FAIL(params_.expr_factory_->create_raw_expr(T_REF_COLUMN, expr))) {
      } else {
        expr->set_result_type(meta_type);
        expr->set_ref_id(item->table_id_, i + OB_APP_MIN_COLUMN_ID);
        expr->set_column_attr(item->table_name_, name);
        if (OB_FAIL(expr->add_flag(IS_COLUMN))) {
        } else {
          ColumnItem column_item;
          column_item.expr_ = expr;
          column_item.table_id_ = item->table_id_;
          column_item.column_id_ = expr->get_column_id();
          column_item.column_name_ = name;
          if (OB_FAIL(stmt->add_column_item(column_item))) {
          } else {
            SelectItem select_item;
            select_item.expr_ = expr;
            select_item.alias_name_ = name;
            select_item.is_real_alias_ = true;
            ret = stmt->add_select_item(select_item);
          }
        }
      }
    }
  }
  if (OB_SUCC(ret)) {
    stmt->set_select_type(NOT_AFFECT_FOUND_ROWS);
    set_basic_stmt(stmt);
  }
  return ret;
}

int ObExplainResolver::resolve(const ParseNode &parse_tree)
{
  int ret = OB_SUCCESS;
  ObExplainStmt *explain_stmt = NULL;
  if (parse_tree.type_ == T_EXPLAIN && parse_tree.num_child_ == 5
      && parse_tree.children_[EXPLAIN_FORMAT]
      && parse_tree.children_[EXPLAIN_FORMAT]->type_ == T_TRADITIONAL
      && parse_tree.children_[EXPLAIN_FORMAT]->value_ == 1
      && parse_tree.children_[EXPLAIN_CHILD_STMT]
      && parse_tree.children_[EXPLAIN_CHILD_STMT]->type_ == T_SELECT
      && nullptr != find_file_table(parse_tree.children_[EXPLAIN_CHILD_STMT])) {
    return resolve_file_describe(*parse_tree.children_[EXPLAIN_CHILD_STMT]);
  }
  if (!(T_EXPLAIN == parse_tree.type_  && 5 == parse_tree.num_child_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("invalid EXPLAIN syntax",
             "type", parse_tree.type_,
             "# of child", parse_tree.num_child_);
  } else if (NULL == (explain_stmt = create_stmt<ObExplainStmt>())) {
    ret = OB_SQL_RESOLVER_NO_MEMORY;
    LOG_WARN("failed to create explain stmt");
  } else {
    ParseNode *child_node = parse_tree.children_[EXPLAIN_CHILD_STMT];
    ObStmt *child_stmt = NULL;
    ObExplainDisplayOpt opt;
    opt.with_tree_line_ = true;
    if (OB_ISNULL(child_node)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("empty parse tree for stmt");
    } else if (!(T_SELECT == child_node->type_
                 || T_DELETE == child_node->type_
                 || T_INSERT == child_node->type_
                 || T_UPDATE == child_node->type_
                 || T_REPLACE == child_node->type_)) {
      ret = OB_NOT_SUPPORTED;
      LOG_WARN("invalid stmt type for EXPLAIN", "type", child_node->type_);
      LOG_USER_ERROR(OB_NOT_SUPPORTED, "stmt type for EXPLAIN");
    } else {
      ObResolver resolver(params_);
      if (OB_FAIL(resolver.resolve(ObResolver::IS_NOT_PREPARED_STMT,
                                   *child_node,
                                   child_stmt))) {
      } else if (OB_ISNULL(child_stmt)) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("failed to generate stmt for child stmt");
      } else {
        explain_stmt->set_explain_query_stmt(static_cast<ObDMLStmt *>(child_stmt));
        ParseNode *format = parse_tree.children_[EXPLAIN_FORMAT];
        if (NULL != format) {
          switch (format->type_) {
          case T_OUTLINE:
          {
            explain_stmt->set_explain_format(EXPLAIN_OUTLINE);
            opt.with_tree_line_ = true;
          } break;
          case T_EXTENDED:
          {
            explain_stmt->set_explain_format(EXPLAIN_EXTENDED);
            opt.with_tree_line_ = true;
          } break;
          case T_PARTITIONS:
          {
            explain_stmt->set_explain_format(EXPLAIN_PARTITIONS);
            opt.with_tree_line_ = true;
          } break;
          case T_TRADITIONAL:
          {
            explain_stmt->set_explain_format(EXPLAIN_TRADITIONAL);
            opt.with_tree_line_ = true;
          } break;
          case T_FORMAT_JSON:
          {
            explain_stmt->set_explain_format(EXPLAIN_FORMAT_JSON);
            opt.with_tree_line_ = false;
          } break;
          case T_BASIC: {
            explain_stmt->set_explain_format(EXPLAIN_BASIC);
            opt.with_tree_line_ = true;
            break;
          }
          case T_EXTENDED_NOADDR: {
            explain_stmt->set_explain_format(EXPLAIN_EXTENDED_NOADDR);
            opt.with_tree_line_ = true;
            break;
          }
          case T_PLANREGRESS: {
            explain_stmt->set_explain_format(EXPLAIN_PLANREGRESS);
            opt.with_tree_line_ = true;
            break;
          }
          default:
            ret = OB_ERROR;
            LOG_WARN("unknown explain format", "format", format->type_);
            break;
          }
        }
      }
    }
    ParseNode *opt_node = parse_tree.children_[EXPLAIN_DISPLAY_OPTION];
    if (OB_SUCC(ret) && NULL != opt_node) {
      switch (opt_node->type_) {
      case T_PRETTY_COLOR:
        opt.with_color_ = true;
        // pass
      case T_PRETTY:
        opt.with_tree_line_ = true;
        break;
      default:
        ret = OB_ERR_UNEXPECTED;
      }
    }
    if (OB_SUCC(ret)) {
      explain_stmt->set_display_opt(opt);
    }
    ParseNode *into_table_node = parse_tree.children_[EXPLAIN_INTO_TABLE];
    if (OB_SUCC(ret) && NULL != into_table_node) {
      ObString into_table(into_table_node->str_len_, into_table_node->str_value_);
      explain_stmt->set_into_table(into_table);
    }
    ParseNode *statement_id_node = parse_tree.children_[EXPLAIN_STMT_ID];
    if (OB_SUCC(ret) && NULL != statement_id_node) {
      ObString statement_id(statement_id_node->str_len_, statement_id_node->str_value_);
      explain_stmt->set_statement_id(statement_id);
    }
  }

  return ret;
}

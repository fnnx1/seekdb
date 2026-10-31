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

#include "sql/engine/expr/ob_expr_tokenize.h"

#include "lib/alloc/alloc_struct.h"
#include "lib/allocator/ob_malloc.h"
#include "lib/allocator/page_arena.h"
#include "lib/charset/ob_charset.h"
#include "lib/lock/ob_mutex.h"
#include "lib/utility/alloc_assist.h"
#include "common/json_type/ob_json_base.h"
#include "common/json_type/ob_json_tree.h"
#include "lib/alloc/ob_malloc_allocator.h"
#include "lib/lock/ob_tc_rwlock.h"
#include "lib/ob_errno.h"
#include "lib/oblog/ob_log_module.h"
#include "lib/string/ob_string.h"
#include "lib/utility/ob_macro_utils.h"
#include "object/ob_object.h"
#include "sql/resolver/ddl/ob_fts_index_builder_util.h"
#include "share/ob_json_access_utils.h"
#include "data_plane/fts/ob_fts_parser_helper.h"
#include "data_plane/fts/dict/ob_gen_dic_loader.h"
#include "data_plane/fts/ob_fts_parser_property.h"

#define USING_LOG_PREFIX SQL_ENG
#include "sql/engine/expr/ob_expr_json_func_helper.h" // file not self-contained, there're logs inside.

namespace oceanbase
{
namespace sql
{
namespace
{

/// A tiny string to string cache.
/// The tokenize function is evaluated once per sql call, and every call rebuilds
/// the very same parser name and parser properties. Since only a handful of
/// distinct (parser name, parser properties) pairs exist, the results are
/// cached here to keep the tokenize hot path free of json building/printing.
class ObFTSStringCache final
{
public:
  static constexpr int64_t MAX_ENTRY_CNT = 16;

public:
  int get(const common::ObString &key1, const common::ObString &key2, common::ObString &value)
  {
    common::TCRWLock::RLockGuard guard(lock_);
    for (int64_t i = 0; i < cnt_; ++i) {
      if (match(i, key1, key2)) {
        value.assign_ptr(entries_[i].value_, entries_[i].value_len_);
        return OB_SUCCESS;
      }
    }
    return OB_ENTRY_NOT_EXIST;
  }

  int put(const common::ObString &key1, const common::ObString &key2, const common::ObString &value)
  {
    int ret = OB_SUCCESS;
    const int64_t key_len = key1.length() + key2.length();
    const int64_t total_len = key_len + value.length();
    lib::ObMemAttr attr("FtsTokCache");
    char *buf = nullptr;
    if (cnt_ >= MAX_ENTRY_CNT) {
      ret = OB_SIZE_OVERFLOW;
    } else if (OB_ISNULL(buf = static_cast<char *>(lib::ObMallocAllocator::get_instance()->alloc(total_len, attr)))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      MEMCPY(buf, key1.ptr(), key1.length());
      if (key2.length() > 0) {
        MEMCPY(buf + key1.length(), key2.ptr(), key2.length());
      }
      if (value.length() > 0) {
        MEMCPY(buf + key_len, value.ptr(), value.length());
      }
      common::TCRWLock::WLockGuard guard(lock_);
      if (cnt_ < MAX_ENTRY_CNT) {
        entries_[cnt_].key_len_ = key_len;
        entries_[cnt_].key1_len_ = key1.length();
        entries_[cnt_].value_ = buf + key_len;
        entries_[cnt_].value_len_ = value.length();
        entries_[cnt_].buf_ = buf;
        ++cnt_;
        buf = nullptr;
      } else {
        ret = OB_SIZE_OVERFLOW;
      }
    }
    if (OB_NOT_NULL(buf)) {
      lib::ObMallocAllocator::get_instance()->free(buf);
    }
    return ret;
  }

private:
  struct ObEntry final
  {
    char *buf_;
    char *value_;
    int64_t key_len_;
    int64_t key1_len_;
    int64_t value_len_;
  };

  bool match(int64_t idx, const common::ObString &key1, const common::ObString &key2) const
  {
    const ObEntry &entry = entries_[idx];
    return entry.key_len_ == key1.length() + key2.length() && entry.key1_len_ == key1.length()
           && 0 == MEMCMP(entry.buf_, key1.ptr(), key1.length())
           && (0 == key2.length() || 0 == MEMCMP(entry.buf_ + key1.length(), key2.ptr(), key2.length()));
  }

private:
  ObEntry entries_[MAX_ENTRY_CNT];
  int64_t cnt_ = 0;
  common::TCRWLock lock_;
};

/// The parser setups used by the tokenize function.
class ObTokenizeParserCache final
{
public:
  static ObTokenizeParserCache &instance()
  {
    static ObTokenizeParserCache cache;
    return cache;
  }

  // raw parser name, e.g. "ik", -> inner parser name, e.g. "ik.1"
  int get_inner_name(const common::ObString &raw_name, common::ObString &inner_name)
  {
    return name_cache_.get(raw_name, common::ObString(), inner_name);
  }

  int put_inner_name(const common::ObString &raw_name, const common::ObString &inner_name)
  {
    return name_cache_.put(raw_name, common::ObString(), inner_name);
  }

  // (inner parser name, raw parser properties) -> reformatted parser properties
  int get_props(const common::ObString &inner_name, const common::ObString &raw_props, common::ObString &props)
  {
    return props_cache_.get(inner_name, raw_props, props);
  }

  int put_props(const common::ObString &inner_name, const common::ObString &raw_props, const common::ObString &props)
  {
    return props_cache_.put(inner_name, raw_props, props);
  }

  // Whether the dictionary of the parser has already been loaded. The
  // dictionary loader keeps the loaded state forever, so there is no need to
  // look it up again and again.
  bool is_dic_loaded(const common::ObString &parser_name, common::ObCharsetType charset)
  {
    common::TCRWLock::RLockGuard guard(lock_);
    for (int64_t i = 0; i < dic_cnt_; ++i) {
      if (dic_entries_[i].charset_ == charset && dic_entries_[i].match(parser_name)) {
        return true;
      }
    }
    return false;
  }

  void mark_dic_loaded(const common::ObString &parser_name, common::ObCharsetType charset)
  {
    common::TCRWLock::WLockGuard guard(lock_);
    for (int64_t i = 0; i < dic_cnt_; ++i) {
      if (dic_entries_[i].charset_ == charset && dic_entries_[i].match(parser_name)) {
        return;
      }
    }
    if (dic_cnt_ < MAX_DIC_ENTRY_CNT) {
      dic_entries_[dic_cnt_].set(parser_name, charset);
      ++dic_cnt_;
    }
  }

private:
  static constexpr int64_t MAX_DIC_ENTRY_CNT = 8;

  struct ObDicEntry final
  {
    bool match(const common::ObString &parser_name) const
    {
      return parser_name.length() == parser_name_len_
             && 0 == MEMCMP(parser_name_, parser_name.ptr(), parser_name_len_);
    }

    void set(const common::ObString &parser_name, common::ObCharsetType charset)
    {
      parser_name_len_ = parser_name.length();
      if (parser_name_len_ > 0) {
        MEMCPY(parser_name_, parser_name.ptr(), parser_name_len_);
      }
      charset_ = charset;
    }

    char parser_name_[storage::OB_FT_PARSER_NAME_LENGTH];
    int64_t parser_name_len_ = 0;
    common::ObCharsetType charset_ = CHARSET_INVALID;
  };

private:
  ObFTSStringCache name_cache_;
  ObFTSStringCache props_cache_;
  ObDicEntry dic_entries_[MAX_DIC_ENTRY_CNT];
  int64_t dic_cnt_ = 0;
  common::TCRWLock lock_;
};

/// Copy a string into the given allocator, so that the lifetime of the result
/// is the same as the lifetime of the allocator.
int copy_to_allocator(common::ObIAllocator &allocator, const common::ObString &src, common::ObString &dst)
{
  int ret = OB_SUCCESS;
  if (src.empty()) {
    dst = common::ObString();
  } else {
    char *buf = static_cast<char *>(allocator.alloc(src.length()));
    if (OB_ISNULL(buf)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      MEMCPY(buf, src.ptr(), src.length());
      dst.assign_ptr(buf, src.length());
    }
  }
  return ret;
}

} // namespace
ObExprTokenize::ObExprTokenize(common::ObIAllocator &alloc)
    : ObStringExprOperator(alloc,
                           T_FUN_TOKENIZE,
                           N_TOKENIZE,
                           MORE_THAN_ZERO,
                           VALID_FOR_GENERATED_COL)
{
}

ObExprTokenize::~ObExprTokenize() {}

namespace {

// tokenize() is a pure function of the input text, the parser name, the parser
// properties and the text collation, so a repeated call with the same inputs does
// not have to re-parse the arguments, reload the dictionaries or re-segment the text.
// This table keeps the finished JSON payload of recent calls. Every candidate is
// compared byte for byte before it is used, so a hash collision can only cost a
// cache miss and never produce a wrong answer. The table is direct mapped; once
// the stored payloads exceed MAX_TOTAL_BYTES it drops everything, which bounds the
// footprint without any eviction bookkeeping.
class ObTokenizeResultCache
{
public:
  static const int64_t SLOT_CNT = 512;
  static const int64_t MAX_RESULT_BYTES = 64 * 1024;
  static const int64_t MAX_TOTAL_BYTES = 8 * 1024 * 1024;

  struct Entry
  {
    Entry() : buf_(NULL), hash_(0), collation_(0), flags_(0), text_len_(0),
              parser_len_(0), props_len_(0), result_len_(0) {}
    char *buf_;            // text | parser name | properties | result
    uint64_t hash_;
    int64_t collation_;
    int64_t flags_;        // per argument: 0 absent, 1 NULL, 2 present
    int64_t text_len_;
    int64_t parser_len_;
    int64_t props_len_;
    int64_t result_len_;
  };

  static ObTokenizeResultCache &instance()
  {
    static ObTokenizeResultCache cache;
    return cache;
  }

  // hit is set only when a verified entry was copied into allocator.
  int get(const int64_t collation,
          const int64_t flags,
          const ObString &text,
          const ObString &parser_name,
          const ObString &props,
          common::ObIAllocator &allocator,
          ObString &result,
          bool &hit)
  {
    int ret = OB_SUCCESS;
    hit = false;
    const uint64_t hash = hash_of(collation, flags, text, parser_name, props);
    lib::ObMutexGuard guard(mutex_);
    Entry &entry = slots_[hash % static_cast<uint64_t>(SLOT_CNT)];
    if (NULL == entry.buf_
        || entry.hash_ != hash
        || entry.collation_ != collation
        || entry.flags_ != flags
        || entry.text_len_ != text.length()
        || entry.parser_len_ != parser_name.length()
        || entry.props_len_ != props.length()
        || !keys_equal(entry, text, parser_name, props)) {
      // not cached yet
    } else {
      char *copy = static_cast<char *>(allocator.alloc(entry.result_len_));
      if (OB_ISNULL(copy)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("allocate tokenize cache result failed", K(ret), K(entry.result_len_));
      } else {
        MEMCPY(copy, entry.buf_ + entry.text_len_ + entry.parser_len_ + entry.props_len_,
               entry.result_len_);
        result.assign_ptr(copy, static_cast<int32_t>(entry.result_len_));
        hit = true;
      }
    }
    return ret;
  }

  void put(const int64_t collation,
           const int64_t flags,
           const ObString &text,
           const ObString &parser_name,
           const ObString &props,
           const ObString &result)
  {
    int ret = OB_SUCCESS;
    if (result.length() <= 0 || result.length() > MAX_RESULT_BYTES) {
      return;
    }
    const int64_t total = static_cast<int64_t>(text.length()) + parser_name.length()
                        + props.length() + result.length();
    lib::ObMemAttr attr;
    attr.label_ = common::ObModIds::OB_SQL_EXPR;
    char *buf = static_cast<char *>(common::ob_malloc(total, attr));
    if (NULL == buf) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("allocate tokenize cache entry failed", K(ret), K(total));
    } else {
      int64_t offset = 0;
      if (text.length() > 0) {
        MEMCPY(buf + offset, text.ptr(), text.length());
        offset += text.length();
      }
      if (parser_name.length() > 0) {
        MEMCPY(buf + offset, parser_name.ptr(), parser_name.length());
        offset += parser_name.length();
      }
      if (props.length() > 0) {
        MEMCPY(buf + offset, props.ptr(), props.length());
        offset += props.length();
      }
      MEMCPY(buf + offset, result.ptr(), result.length());

      const uint64_t hash = hash_of(collation, flags, text, parser_name, props);
      lib::ObMutexGuard guard(mutex_);
      if (total_bytes_ + total > MAX_TOTAL_BYTES) {
        clear_all();
      }
      Entry &entry = slots_[hash % static_cast<uint64_t>(SLOT_CNT)];
      if (NULL != entry.buf_) {
        total_bytes_ -= entry_size(entry);
        common::ob_free(entry.buf_);
      }
      entry.buf_ = buf;
      entry.hash_ = hash;
      entry.collation_ = collation;
      entry.flags_ = flags;
      entry.text_len_ = text.length();
      entry.parser_len_ = parser_name.length();
      entry.props_len_ = props.length();
      entry.result_len_ = result.length();
      total_bytes_ += total;
    }
  }

private:
  ObTokenizeResultCache() : mutex_(), slots_(), total_bytes_(0) {}
  DISALLOW_COPY_AND_ASSIGN(ObTokenizeResultCache);

  static int64_t entry_size(const Entry &entry)
  {
    return entry.text_len_ + entry.parser_len_ + entry.props_len_ + entry.result_len_;
  }

  static uint64_t fnv1a(uint64_t hash, const char *data, const int64_t len)
  {
    for (int64_t i = 0; i < len; ++i) {
      hash ^= static_cast<uint64_t>(static_cast<unsigned char>(data[i]));
      hash *= 1099511628211ULL;
    }
    return hash;
  }

  static uint64_t hash_of(const int64_t collation,
                          const int64_t flags,
                          const ObString &text,
                          const ObString &parser_name,
                          const ObString &props)
  {
    uint64_t hash = 14695981039346656037ULL;
    hash = fnv1a(hash, reinterpret_cast<const char *>(&collation), sizeof(collation));
    hash = fnv1a(hash, reinterpret_cast<const char *>(&flags), sizeof(flags));
    hash = fnv1a(hash, text.ptr(), text.length());
    hash = fnv1a(hash, parser_name.ptr(), parser_name.length());
    hash = fnv1a(hash, props.ptr(), props.length());
    return hash;
  }

  static bool segment_equal(const char *base, const ObString &value)
  {
    return value.length() <= 0 || 0 == MEMCMP(base, value.ptr(), value.length());
  }

  static bool keys_equal(const Entry &entry,
                         const ObString &text,
                         const ObString &parser_name,
                         const ObString &props)
  {
    const char *base = entry.buf_;
    bool equal = segment_equal(base, text);
    base += entry.text_len_;
    equal = equal && segment_equal(base, parser_name);
    base += entry.parser_len_;
    equal = equal && segment_equal(base, props);
    return equal;
  }

  void clear_all()
  {
    for (int64_t i = 0; i < SLOT_CNT; ++i) {
      if (NULL != slots_[i].buf_) {
        common::ob_free(slots_[i].buf_);
        slots_[i] = Entry();
      }
    }
    total_bytes_ = 0;
  }

private:
  lib::ObMutex mutex_;
  Entry slots_[SLOT_CNT];
  int64_t total_bytes_;
};

} // namespace

// Reads one tokenize() argument as raw text. A missing or NULL argument reads as an
// empty string, which is how the parser treats both cases.
static int read_tokenize_arg(const ObExpr &expr,
                             ObEvalCtx &ctx,
                             const int64_t idx,
                             ObString &value,
                             int64_t &kind)
{
  int ret = OB_SUCCESS;
  ObDatum *datum = nullptr;
  value = ObString();
  kind = 0;
  if (idx < expr.arg_cnt_) {
    if (OB_ISNULL(expr.args_[idx])) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(expr.args_[idx]->eval(ctx, datum))) {
    } else if (NULL == datum || datum->is_null()) {
      kind = 1;
    } else {
      kind = 2;
      value = datum->get_string();
    }
  }
  return ret;
}

int ObExprTokenize::eval_tokenize(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &expr_datum)
{
  int ret = OB_SUCCESS;

  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  common::ObArenaAllocator &temp_allocator = tmp_alloc_g.get_allocator();

  ObIJsonBase *json_result = nullptr;
  TokenizeParam param;
  ObString cached_result;
  ObString raw_text;
  ObString raw_parser;
  ObString raw_props;
  int64_t text_kind = 0;
  int64_t parser_kind = 0;
  int64_t props_kind = 0;
  bool cache_hit = false;

  // check param num, which is checked in ObExprOperator::calc_result_typeN.
  if (OB_UNLIKELY(expr.arg_cnt_ < 1 || expr.arg_cnt_ > 3)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(read_tokenize_arg(expr, ctx, 0, raw_text, text_kind))) {
  } else if (OB_FAIL(read_tokenize_arg(expr, ctx, 1, raw_parser, parser_kind))) {
  } else if (OB_FAIL(read_tokenize_arg(expr, ctx, 2, raw_props, props_kind))) {
  } else {
    // Everything tokenize() depends on follows from the raw argument text and the
    // text collation, so the lookup happens before parse_param() and a hit skips
    // argument parsing and the dictionary load along with the segmentation.
    const int64_t collation =
        static_cast<int64_t>(expr.args_[0]->obj_meta_.get_collation_type());
    // An absent argument and an explicit empty string are not the same input:/r/n    // tokenize('x') defaults to ik while tokenize('x', '') is rejected, so the
    // presence of every argument is part of the key.
    const int64_t flags = text_kind | (parser_kind << 4) | (props_kind << 8);
    if (OB_FAIL(ObTokenizeResultCache::instance().get(
                   collation, flags, raw_text, raw_parser, raw_props,
                   temp_allocator, cached_result, cache_hit))) {
      // A cache problem must never fail the statement; fall back to tokenizing.
      LOG_WARN("tokenize cache lookup failed, tokenizing anyway", K(ret));
      ret = OB_SUCCESS;
      cache_hit = false;
    }
  }

  if (OB_FAIL(ret)) {
  } else if (cache_hit) {
    // pack_json_str_res() takes the same raw JSON payload that pack_json_res()
    // would emit for a freshly built result, so the output is byte identical.
    if (OB_FAIL(ObJsonExprHelper::pack_json_str_res(expr, ctx, expr_datum, cached_result))) {
    }
  } else if (OB_FAIL(parse_param(expr, ctx, temp_allocator, param))) {
  } else if (OB_FAIL(tokenize_fulltext(param, param.output_mode_, temp_allocator, json_result))) {
  } else {
    if (NULL != json_result) {
      ObString raw_result;
      // Populating the cache is best effort: a failure here must not fail the query.
      const int cache_ret = share::ObJsonWrapper::get_raw_binary(json_result, raw_result, &temp_allocator);
      if (OB_SUCCESS != cache_ret) {
        LOG_WARN("read raw tokenize result failed, not caching it", K(cache_ret));
      } else {
        ObTokenizeResultCache::instance().put(
            static_cast<int64_t>(expr.args_[0]->obj_meta_.get_collation_type()),
            text_kind | (parser_kind << 4) | (props_kind << 8),
            raw_text, raw_parser, raw_props, raw_result);
      }
    }
    if (OB_FAIL(ObJsonExprHelper::pack_json_res(expr,
                                                ctx,
                                                temp_allocator,
                                                json_result,
                                                expr_datum))) {
    }
  }

  return ret;
}

int ObExprTokenize::tokenize_fulltext(const TokenizeParam &param,
                                      TokenizeParam::OUTPUT_MODE mode,
                                      ObIAllocator &allocator,
                                      ObIJsonBase *&result)
{
  int ret = OB_SUCCESS;
  storage::ObFTParseHelper tokenize_helper;
  const int64_t ft_word_bkt_cnt = MIN(MAX(param.fulltext_.length() / 2, 2), 997);
  int64_t doc_len = 0;
  storage::ObFTWordMap token_map;

  if (TokenizeParam::OUTPUT_MODE::DEFAULT != mode && TokenizeParam::OUTPUT_MODE::ALL != mode) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(tokenize_helper.init(&allocator, param.parser_name_, param.properties_))) {
  } else if (OB_FAIL(token_map.create(ft_word_bkt_cnt, common::ObMemAttr("FTWordMap")))) {
  } else if (
      (0 != param.fulltext_.length())
      && OB_FAIL(tokenize_helper.segment(
                     param.meta_,
                     param.fulltext_.ptr(),
                     param.fulltext_.length(),
                     doc_len,
                     token_map))) {
  } else {
    switch (param.output_mode_) {
    case TokenizeParam::OUTPUT_MODE::DEFAULT: {
      if (OB_FAIL(tokenize_helper.make_token_array_json(token_map, result))) {
      } else {
        // pass
      }
      break;
    }
    case TokenizeParam::OUTPUT_MODE::ALL: {
      if (OB_FAIL(tokenize_helper.make_detail_json(token_map, doc_len, result))) {
      } else {
        // pass
      }
      break;
    }
    default:
      ret = OB_INVALID_ARGUMENT;
    }
  }
  return ret;
}

ObExprTokenize::TokenizeParam ::TokenizeParam()
  : allocator_(ObMemAttr("TokenizeParam")),
    parser_name_(ObString(OB_DEFAULT_FULLTEXT_PARSER_NAME)),
    meta_(),
    fulltext_(),
    output_mode_(OUTPUT_MODE::DEFAULT)
{
}

int ObExprTokenize::TokenizeParam::parse_json_param(const ObIJsonBase *obj)
{
  int ret = OB_SUCCESS;
  ObString str;
  ObIJsonBase *val;

  if (OB_UNLIKELY(nullptr == obj)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (ObJsonNodeType::J_OBJECT != obj->json_type()) {
    ret = OB_INVALID_ARGUMENT;
  } else if (obj->element_count() == 0) {
    // no data
  } else if (OB_FAIL(obj->get_object_value(0, str, val))) {
  } else if (0 == str.case_compare(CASE_INDICATOR_STR)) {
    if (ObJsonNodeType::J_STRING != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("UPPER")) {
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("LOWER")) {
    } else {
      ret = OB_INVALID_ARGUMENT;
    }
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "case indentifier");
  } else if (0 == str.case_compare(OUTPUT_MODE_STR)) {
    if (ObJsonNodeType::J_STRING != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "output mode should be string default or all");
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("DEFAULT")) {
      output_mode_ = DEFAULT;
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("ALL")) {
      output_mode_ = ALL;
    } else {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "output mode should be string default or all");
    }
  } else if (0 == str.case_compare(STOPWORDS_LIST_STR)) {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "stopwords");
  } else if (0 == str.case_compare(ADDITIONAL_ARGS_STR)) {
    if (ObJsonNodeType::J_ARRAY != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser arguments");
    } else {
      ObString json_str;
      if (OB_FAIL(storage::ObFTParserJsonProps::tokenize_array_to_props_json(allocator_, val, json_str))) {
        ObSqlString message;
        message.append_fmt("format in %s form", ADDITIONAL_ARGS_STR);
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, message.ptr());
      } else {
        properties_ = json_str;
      }
    }
  } else {
    ret = OB_INVALID_ARGUMENT;
    ObSqlString message;
    message.append_fmt("config: %s", str.ptr());
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, message.ptr());
  }
  return ret;
}

int ObExprTokenize::parse_param(const ObExpr &expr,
                                ObEvalCtx &ctx,
                                common::ObArenaAllocator &allocator,
                                TokenizeParam &param)
{
  int ret = OB_SUCCESS;

  ObDatum *parser_params_datum;
  ObString raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);

  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  
  MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());

  if (OB_UNLIKELY(expr.arg_cnt_ < 1 || expr.arg_cnt_ > 3)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(parse_fulltext(expr, ctx, param))) {
  } else if (OB_FAIL(parse_parser_name(expr, ctx, param))) {
  } else if (OB_FAIL(parse_parser_properties(expr, ctx, temp_allocator, param))) {
  } else if (OB_FAIL(param.reform_parser_properties(param.properties_))) {
  } else if (OB_FAIL(param.try_load_dictionary_for_ik())) {
  }
  return ret;
}

int ObExprTokenize::construct_ft_parser_inner_name(const ObString &input_str, TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  storage::ObFTParser parser;

  // The inner parser name only depends on the user supplied parser name, so it
  // is cached to avoid rebuilding it on every tokenize call.
  ObString cached_inner_name;
  int tmp_ret = ObTokenizeParserCache::instance().get_inner_name(input_str, cached_inner_name);
  if (OB_ENTRY_NOT_EXIST == tmp_ret) {
    char *parser_name_buf = nullptr;
    if (OB_ISNULL(parser_name_buf
                  = static_cast<char *>(param.allocator_.alloc(storage::OB_FT_PARSER_NAME_LENGTH)))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (OB_FAIL(parser.init(input_str))) {
    } else if (OB_FAIL(parser.serialize_to_str(parser_name_buf, storage::OB_FT_PARSER_NAME_LENGTH))) {
    } else {
      cached_inner_name = ObString::make_string(parser_name_buf);
      tmp_ret = ObTokenizeParserCache::instance().put_inner_name(input_str, cached_inner_name);
      if (OB_UNLIKELY(OB_FAIL(tmp_ret))) {
        LOG_WARN("failed to cache the inner fulltext parser name", K(tmp_ret));
      }
    }
  } else if (OB_SUCC(tmp_ret)) {
    ret = OB_SUCCESS;
  } else {
    ret = tmp_ret;
  }

  if (OB_SUCC(ret)) {
    // The inner name is copied into the tokenize param allocator, so that its
    // lifetime is the same as before.
    if (OB_FAIL(copy_to_allocator(param.allocator_, cached_inner_name, param.parser_name_))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    }
  }
  return ret;
}

int ObExprTokenize::calc_result_typeN(ObExprResType &type,
                                      ObExprResType *types,
                                      int64_t param_num,
                                      common::ObExprTypeCtx &type_ctx) const
{
  int ret = OB_SUCCESS;

  if (OB_UNLIKELY(param_num < 1 || param_num > 3)) {
    ret = OB_ERR_PARAM_SIZE;
    ObString expr_name(N_TOKENIZE);
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, expr_name.length(), expr_name.ptr());
  } else {
    // just okay
  }

  ObLength length = ObAccuracy::DDL_DEFAULT_ACCURACY[ObJsonType].get_length();

  if (OB_SUCC(ret)) {
    // set res type
    type.set_json();
    type.set_length(length); // keep consistent with other json expr, maybe calc it later.

    // param type set, skip charset after first param
    for (int64_t i = 1; OB_SUCC(ret) && i < param_num; ++i) {
      if (ob_is_string_type(types[i].get_type())) {
        if (types[i].get_charset_type() != CHARSET_UTF8MB4) {
          types[i].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
        }
      }
    }

    // handle param
    if (param_num >= 2) {
      types[1].set_varchar();
    }

    if (param_num >= 3) {
      if (OB_FAIL(ObJsonExprHelper::is_valid_for_json(types, 2, N_TOKENIZE))) {
      }
    }
  }

  return ret;
}

int ObExprTokenize::cg_expr(ObExprCGCtx &op_cg_ctx,
                            const ObRawExpr &raw_expr,
                            ObExpr &rt_expr) const
{
  int ret = OB_SUCCESS;
  CK((rt_expr.arg_cnt_ >= 1 && rt_expr.arg_cnt_ <= 3));
  if (OB_SUCC(ret)) {
    // do register
    rt_expr.eval_func_ = eval_tokenize;
  }
  return ret;
}

int ObExprTokenize::parse_fulltext(const ObExpr &expr, ObEvalCtx &ctx, TokenizeParam &param)
{
  int ret = OB_SUCCESS;

  ObDatum *fulltext_datum;

  if (OB_FAIL(expr.args_[0]->eval(ctx, fulltext_datum))) {
  } else {
    if (fulltext_datum->is_null()) {
      // do nothing, return empty result
      param.fulltext_ = ObString::make_empty_string();
    } else {
      param.fulltext_ = fulltext_datum->get_string();
    }
    param.meta_.set_varchar(); // as we hardcoded in fts_index
    param.meta_.set_collation_type(expr.args_[0]->obj_meta_.get_collation_type());
  }
  return ret;
}

int ObExprTokenize::parse_parser_name(const ObExpr &expr, ObEvalCtx &ctx, TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  ObDatum *parser_datum = nullptr;
  ObString raw_parser_name;

  if (expr.arg_cnt_ < 2) {
    raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);
  } else if (OB_FAIL(expr.args_[1]->eval(ctx, parser_datum))) {
  } else {
    if (parser_datum->is_null()) {
      raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);
    } else {
      ObString name = parser_datum->get_string();
      raw_parser_name = name.trim();
    }
  }

  if (OB_FAIL(ret)) {
    // already logged
  } else if (OB_FAIL(construct_ft_parser_inner_name(raw_parser_name, param))) {
  }

  return ret;
}

int ObExprTokenize::parse_parser_properties(const ObExpr &expr,
                                            ObEvalCtx &ctx,
                                            MultimodeAlloctor &mm_alloc,
                                            TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  ObIJsonBase *base = nullptr;

  if (expr.arg_cnt_ < 3) {
    // do nothing
  } else {
    bool is_null = false;
    if (OB_FAIL(ObJsonExprHelper::get_json_doc(expr, ctx, mm_alloc, 2, base, is_null))) {
    } else {
      if (ObJsonNodeType::J_ARRAY != base->json_type()) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser args should be in an array.");
      } else {
        for (uint64_t i = 0; OB_SUCC(ret) && i < base->element_count(); ++i) {
          ObIJsonBase *node = nullptr;
          if (OB_FAIL(base->get_array_element(i, node))) {
          } else if (ObJsonNodeType::J_OBJECT != (node->json_type())) {
            ret = OB_INVALID_ARGUMENT;
          } else if (OB_FAIL(param.parse_json_param(node))) {
          }
        } // for
      }
    }
  }

  return ret;
}

int ObExprTokenize::TokenizeParam::reform_parser_properties(const ObString &properties)
{
  int ret = OB_SUCCESS;
  storage::ObFTParserJsonProps parser_properties;
  ObString formatted_props;
  bool is_built = false;
  common::ObArenaAllocator tmp_alloc(common::ObMemAttr("TokenizeProps"));

  // The reformatted parser properties only depend on the parser name and the
  // raw parser properties, so they are cached to avoid rebuilding the json on
  // every tokenize call.
  int tmp_ret = ObTokenizeParserCache::instance().get_props(parser_name_, properties, formatted_props);
  if (OB_ENTRY_NOT_EXIST == tmp_ret) {
    ObString tmp_props;
    if (OB_FAIL(parser_properties.init())) {
    } else if (OB_FAIL(parser_properties.parse_from_valid_str(properties))) {
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser properties invalid.");
    } else if (OB_FAIL(parser_properties.rebuild_props_for_ddl(parser_name_,
                                                               ObCollationType::CS_TYPE_UTF8MB4_BIN,
                                                               true))) {
    } else if (OB_FAIL(parser_properties.to_format_json(tmp_alloc, tmp_props))) {
    } else {
      is_built = true;
      int put_ret = ObTokenizeParserCache::instance().put_props(parser_name_, properties, tmp_props);
      if (OB_SUCC(put_ret)) {
        int get_ret = ObTokenizeParserCache::instance().get_props(parser_name_, properties, formatted_props);
        if (OB_FAIL(get_ret)) {
          formatted_props = tmp_props;
        }
      } else {
        // The cache is only an optimization, the reformatted properties are
        // still usable.
        LOG_WARN("failed to cache the reformatted parser properties", K(put_ret));
        formatted_props = tmp_props;
      }
    }
  } else if (OB_SUCC(tmp_ret)) {
    is_built = true;
  } else {
    // the cache lookup failed, keep the properties untouched
  }

  if (is_built) {
    // The reformatted properties are copied into the tokenize param allocator, so
    // that their lifetime is the same as before.
    if (OB_FAIL(copy_to_allocator(allocator_, formatted_props, properties_))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    }
  }

  return ret;
}

int ObExprTokenize::TokenizeParam::try_load_dictionary_for_ik()
{
  int ret = OB_SUCCESS;
  bool need_to_load_dic = false;
  storage::ObDicLoaderHandle dic_loader_handle;
  const common::ObCharsetType charset = ObCharset::charset_type_by_coll(meta_.get_collation_type());
  if (OB_FAIL(share::ObFtsIndexBuilderUtil::check_need_to_load_dic(
          parser_name_, need_to_load_dic))) {
  } else if (need_to_load_dic) {
    // The dictionary loader keeps the loaded state forever, so the loader is
    // only looked up once for a (parser, charset) pair.
    if (OB_UNLIKELY(ObTokenizeParserCache::instance().is_dic_loaded(parser_name_, charset))) {
      // already loaded
    } else if (OB_FAIL(storage::ObGenDicLoader::get_instance().get_dic_loader(
                    ObString::make_string(storage::ObFTSLiteral::PARSER_NAME_IK), // currently only ik, use parser_name_ without version suffix
                    charset,
                    dic_loader_handle))) {
    } else if (OB_UNLIKELY(!dic_loader_handle.is_valid())) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(dic_loader_handle.get_loader()->try_load_dictionary_in_trans())) {
    } else {
      ObTokenizeParserCache::instance().mark_dic_loaded(parser_name_, charset);
    }
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase

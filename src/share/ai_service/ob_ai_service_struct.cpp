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

#include "share/ai_service/ob_ai_service_struct.h"
#include "lib/json/ob_json.h"


#define USING_LOG_PREFIX SHARE

using namespace oceanbase::share;
using namespace oceanbase::common;
using namespace oceanbase::json;

namespace oceanbase
{
namespace share
{
const ObString ObAiModelEndpointInfo::DEFAULT_SCOPE = "ALL";

const char *VALID_PROVIDERS[] = {
  "ALIYUN-OPENAI",
  "ALIYUN-DASHSCOPE",
  "DEEPSEEK",
  "SILICONFLOW",
  "COHERE",
  "HUNYUAN-OPENAI",
  "OPENAI",
  "OLLAMA"
};

// Accepted range for the per-endpoint in-flight request cap.  The lower bound is
// 1 because a zero cap would make every request against the endpoint fail; the
// upper bound keeps one endpoint from draining the whole observer thread pool.
static const int64_t MIN_ENDPOINT_MAX_CONCURRENCY = 1;
static const int64_t MAX_ENDPOINT_MAX_CONCURRENCY = 1024;

#define EXTRACT_JSON_ELEM_STR(json_key, member) \
  EXTRACT_JSON_ELEM_STR_WITH_PROCESS(json_key, member, "void")

#define EXTRACT_JSON_ELEM_STR_WITH_PROCESS(json_key, member, post_process) \
      if (elem.first.case_compare(json_key) == 0) { \
        if (elem.second->json_type() != ObJsonNodeType::J_STRING) { \
          ret = OB_AI_FUNC_PARAM_VALUE_INVALID; \
          LOG_USER_ERROR(OB_AI_FUNC_PARAM_VALUE_INVALID, elem.first.length(), elem.first.ptr()); \
          LOG_WARN("invalid json type", K(ret), K(elem.first), K(elem.second->json_type())); \
        } else { \
          member = ObString(elem.second->get_data_length(), elem.second->get_data()); \
          post_process; \
        } \
      } else

#define EXTRACT_JSON_ELEM_END() \
  { \
    ret = OB_AI_FUNC_PARAM_INVALID; \
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_INVALID, elem.first.length(), elem.first.ptr()); \
    LOG_WARN("unknown json key param", K(ret), K(elem.first)); \
  }

// Integer valued option (e.g. max_concurrency).  A quoted number is rejected on
// purpose: the option is a server side cap, not a client side hint, so a value
// whose type has to be guessed is reported as an invalid value instead of being
// silently coerced.
#define EXTRACT_JSON_ELEM_INT_WITH_PROCESS(json_key, member, post_process) \
      if (elem.first.case_compare(json_key) == 0) { \
        if (elem.second->json_type() == ObJsonNodeType::J_INT) { \
          member = elem.second->get_int(); \
          post_process; \
        } else if (elem.second->json_type() == ObJsonNodeType::J_UINT) { \
          member = static_cast<int64_t>(elem.second->get_uint()); \
          post_process; \
        } else { \
          ret = OB_AI_FUNC_PARAM_VALUE_INVALID; \
          LOG_USER_ERROR(OB_AI_FUNC_PARAM_VALUE_INVALID, elem.first.length(), elem.first.ptr()); \
          LOG_WARN("invalid json type for int param", K(ret), K(elem.first), K(elem.second->json_type())); \
        } \
      } else

#define EXTRACT_JSON_ELEM_INT(json_key, member) \
  EXTRACT_JSON_ELEM_INT_WITH_PROCESS(json_key, member, "void")


int ObAiModelEndpointInfo::parse_from_json_base(common::ObArenaAllocator &allocator, const ObString &name, const ObIJsonBase &params_jbase)
{
  int ret = OB_SUCCESS;
  reset();
  name_ = name;
  if (OB_FAIL(merge_delta_endpoint(allocator, params_jbase))) {
  }
  LOG_INFO("parse from json base", K(ret), K(params_jbase), K(params_jbase.json_type()), K(params_jbase.element_count()));
  return ret;
}

int ObAiModelEndpointInfo::deep_copy(
    common::ObIAllocator &allocator,
    const ObAiModelEndpointInfo &other)
{
  int ret = OB_SUCCESS;
  reset();
  endpoint_id_ = other.endpoint_id_;
  if (OB_FAIL(ob_write_string(allocator, other.name_, name_))) {
  } else if (OB_FAIL(ob_write_string(allocator, other.scope_, scope_))) {
  } else if (OB_FAIL(ob_write_string(allocator, other.ai_model_name_, ai_model_name_))) {
  } else if (OB_FAIL(ob_write_string(allocator, other.url_, url_))) {
  } else if (OB_FAIL(ob_write_string(allocator, other.access_key_, access_key_))) {
  } else if (OB_FAIL(ob_write_string(allocator, other.provider_, provider_))) {
  } else if (OB_FAIL(ob_write_string(allocator, other.request_model_name_, request_model_name_))) {
  } else if (OB_FAIL(ob_write_string(allocator, other.parameters_, parameters_))) {
  } else if (OB_FAIL(ob_write_string(allocator, other.request_transform_fn_, request_transform_fn_))) {
  } else if (OB_FAIL(ob_write_string(allocator, other.response_transform_fn_, response_transform_fn_))) {
  }
  if (OB_SUCC(ret)) {
    max_concurrency_ = other.max_concurrency_;
  }
  return ret;
}

int ObAiModelEndpointInfo::check_valid() const
{
  int ret = OB_SUCCESS;
  if (name_.empty()) {
    ret = OB_AI_FUNC_PARAM_EMPTY;
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_EMPTY, strlen("name"), "name");
  } else if (scope_.empty()) {
    ret = OB_AI_FUNC_PARAM_EMPTY;
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_EMPTY, strlen("scope"), "scope");
  } else if (scope_.case_compare(DEFAULT_SCOPE) != 0) {
    ret = OB_AI_FUNC_PARAM_VALUE_INVALID;
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_VALUE_INVALID, strlen("scope"), "scope");
  } else if (ai_model_name_.empty()) {
    ret = OB_AI_FUNC_PARAM_EMPTY;
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_EMPTY, strlen("ai_model_name"), "ai_model_name");
  } else if (url_.empty()) {
    ret = OB_AI_FUNC_PARAM_EMPTY;
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_EMPTY, strlen("url"), "url");
  } else if (access_key_.empty() && provider_.case_compare(ObString::make_string("OLLAMA")) != 0) {
    // Ollama serves the model from a locally reachable endpoint and normally has no
    // API key, so an empty access_key is only rejected for providers that
    // authenticate against a remote service.
    ret = OB_AI_FUNC_PARAM_EMPTY;
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_EMPTY, strlen("access_key"), "access_key");
    LOG_WARN("access_key is empty", K(ret), K(*this));
  } else if (provider_.empty()) {
    ret = OB_AI_FUNC_PARAM_EMPTY;
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_EMPTY, strlen("provider"), "provider");
  } else if (!is_valid_provider(provider_)) {
    ret = OB_AI_FUNC_PARAM_VALUE_INVALID;
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_VALUE_INVALID, strlen("provider"), "provider");
  } else if (!parameters_.empty()) {
    ret = OB_AI_FUNC_PARAM_VALUE_INVALID;
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_VALUE_INVALID, strlen("parameters"), "parameters");
  } else if (!request_transform_fn_.empty()) {
    ret = OB_AI_FUNC_PARAM_VALUE_INVALID;
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_VALUE_INVALID, strlen("request_transform_fn"), "request_transform_fn");
  } else if (!response_transform_fn_.empty()) {
    ret = OB_AI_FUNC_PARAM_VALUE_INVALID;
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_VALUE_INVALID, strlen("response_transform_fn"), "response_transform_fn");
  } else if (max_concurrency_ < 0 || max_concurrency_ > MAX_ENDPOINT_MAX_CONCURRENCY) {
    ret = OB_AI_FUNC_PARAM_VALUE_INVALID;
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_VALUE_INVALID, strlen("max_concurrency"), "max_concurrency");
  }
  return ret;
}

bool ObAiModelEndpointInfo::is_valid_provider(const ObString &provider)
{
  bool is_valid = false;
  for (int i = 0; i < ARRAYSIZEOF(VALID_PROVIDERS); i++) {
    if (provider.case_compare(VALID_PROVIDERS[i]) == 0) {
      is_valid = true;
      break;
    }
  }
  return is_valid;
}

int ObAiModelEndpointInfo::merge_delta_endpoint(common::ObArenaAllocator &allocator, const ObIJsonBase &delta_jbase)
{
  int ret = OB_SUCCESS;
  JsonObjectIterator iter = delta_jbase.object_iterator();
  bool has_api_key = false;
  bool has_max_concurrency = false;
  while (!iter.end() && OB_SUCC(ret)) {
    ObJsonObjPair elem;
    if (OB_FAIL(iter.get_elem(elem))) {
    } else {
      EXTRACT_JSON_ELEM_STR("scope", scope_)
      EXTRACT_JSON_ELEM_STR("ai_model_name", ai_model_name_)
      EXTRACT_JSON_ELEM_STR("url", url_)
      EXTRACT_JSON_ELEM_STR_WITH_PROCESS("access_key", access_key_, has_api_key = true)
      EXTRACT_JSON_ELEM_STR("provider", provider_)
      EXTRACT_JSON_ELEM_STR("request_model_name", request_model_name_)
      EXTRACT_JSON_ELEM_STR("parameters", parameters_)
      EXTRACT_JSON_ELEM_STR("request_transform_fn", request_transform_fn_)
      EXTRACT_JSON_ELEM_STR("response_transform_fn", response_transform_fn_)
      EXTRACT_JSON_ELEM_INT_WITH_PROCESS("max_concurrency", max_concurrency_, has_max_concurrency = true)
      EXTRACT_JSON_ELEM_END()
    }
    iter.next();
  }

  if (OB_SUCC(ret)) {
    if (has_max_concurrency && (max_concurrency_ < MIN_ENDPOINT_MAX_CONCURRENCY
                                || max_concurrency_ > MAX_ENDPOINT_MAX_CONCURRENCY)) {
      ret = OB_AI_FUNC_PARAM_VALUE_INVALID;
      LOG_USER_ERROR(OB_AI_FUNC_PARAM_VALUE_INVALID, strlen("max_concurrency"), "max_concurrency");
      LOG_WARN("max_concurrency out of range", K(ret), K_(max_concurrency));
    } else if (has_api_key && !access_key_.empty() && OB_FAIL(encrypt_access_key_(allocator, access_key_, access_key_))) {
      LOG_WARN("failed to encrypt access key", K(ret));
    } else if (OB_FAIL(check_valid())) {
    }
  }

  LOG_INFO("merge delta endpoint", K(ret), K(delta_jbase), K(delta_jbase.json_type()), K(delta_jbase.element_count()));
  return ret;
}

int ObAiModelEndpointInfo::encrypt_access_key_(ObIAllocator &allocator, const ObString &access_key, ObString &encrypted_access_key)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(ob_write_string(allocator, access_key, encrypted_access_key))) {
  }
  return ret;
}

int ObAiModelEndpointInfo::decrypt_access_key_(ObIAllocator &allocator, const ObString &encrypted_access_key, ObString &unencrypted_access_key) const
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(ob_write_string(allocator, encrypted_access_key, unencrypted_access_key))) {
  }
  return ret;
}

const char *EndpointType::ENDPOINT_TYPE_STR[] = {
  "DENSE_EMBEDDING",
  "SPARSE_EMBEDDING",
  "COMPLETION",
  "RERANK",
};

EndpointType::TYPE EndpointType::str_to_endpoint_type(const ObString &type_str)
{
  STATIC_ASSERT(static_cast<int64_t>(EndpointType::MAX_TYPE) == ARRAYSIZEOF(ENDPOINT_TYPE_STR) + 1, "endpoint type str len is mismatch");
  EndpointType::TYPE endpoint_type = EndpointType::INVALID_TYPE;
  bool is_found = false;
  for (int i = 1; i < EndpointType::MAX_TYPE && !is_found; i++) {
    if (type_str.case_compare(ENDPOINT_TYPE_STR[i-1]) == 0) {
      endpoint_type = static_cast<EndpointType::TYPE>(i);
      is_found = true;
    }
  }
  return endpoint_type;
}

int ObAiModelEndpointInfo::get_unencrypted_access_key(common::ObIAllocator &allocator, ObString &unencrypted_access_key) const
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(decrypt_access_key_(allocator, access_key_, unencrypted_access_key))) {
  }
  return ret;
}

int ObAiServiceModelInfo::parse_from_json_base(const ObString &name, const common::ObIJsonBase &params_jbase)
{
  int ret = OB_SUCCESS;
  reset();
  name_ = name;
  JsonObjectIterator iter = params_jbase.object_iterator();
  ObString type_str;
  while (!iter.end() && OB_SUCC(ret)) {
    ObJsonObjPair elem;
    if (OB_FAIL(iter.get_elem(elem))) {
    } else {
      EXTRACT_JSON_ELEM_STR("model_name", model_name_)
      EXTRACT_JSON_ELEM_STR_WITH_PROCESS("type", type_str, type_ = EndpointType::str_to_endpoint_type(type_str))
      EXTRACT_JSON_ELEM_END()
    }
    iter.next();
  }

  if (OB_SUCC(ret)) {
    if (OB_FAIL(check_valid())) {
    }
  }

  LOG_TRACE("parse from json base", K(ret), K(params_jbase), K(params_jbase.json_type()), K(params_jbase.element_count()));
  return ret;
}

int ObAiServiceModelInfo::check_valid() const
{
  int ret = OB_SUCCESS;
  if (name_.empty()) {
    ret = OB_AI_FUNC_PARAM_EMPTY;
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_EMPTY, strlen("name"), "name");
  } else if (model_name_.empty()) {
    ret = OB_AI_FUNC_PARAM_EMPTY;
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_EMPTY, strlen("model_name"), "model_name");
  } else if (type_ == EndpointType::MAX_TYPE) {
    ret = OB_AI_FUNC_PARAM_EMPTY;
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_EMPTY, strlen("type"), "type");
  } else if (type_ == EndpointType::INVALID_TYPE) {
    ret = OB_AI_FUNC_PARAM_VALUE_INVALID;
    LOG_USER_ERROR(OB_AI_FUNC_PARAM_VALUE_INVALID, strlen("type"), "type");
  }
  return ret;
}

OB_SERIALIZE_MEMBER(ObAiServiceModelInfo, name_, type_, model_name_);

} // namespace share
} // namespace oceanbase

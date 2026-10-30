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

#define USING_LOG_PREFIX STORAGE

#include "ob_text_daat_iter.h"
#include "query/engine/expr/ob_expr.h"
#include "data_plane/access/ob_tablet_scan.h"

namespace oceanbase
{
namespace storage
{
int ObTextDaaTIter::init(const ObTextDaaTParam &param)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(param.base_param_) || OB_ISNULL(param.dim_iters_) || OB_ISNULL(param.allocator_)
      || OB_ISNULL(param.relevance_collector_)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObSRDaaTIterImpl::init(*param.base_param_, *param.dim_iters_,
                                            *param.allocator_, *param.relevance_collector_))) {
  } else if (nullptr != param.base_param_->relevance_expr_
      && OB_FAIL(bm25_param_estimator_.init(param.bm25_param_est_ctx_))) {
  } else {
    mode_flag_ = param.mode_flag_;
    function_lookup_mode_ = param.function_lookup_mode_;
  }
  return ret;
}

void ObTextDaaTIter::reset()
{
  bm25_param_estimator_.reset();
  // this class can know the type of dim_iters_, but ObSRDaaTIterImpl maybe not
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i))->reset();
    }
  }
  ObSRDaaTIterImpl::reset();
}

void ObTextDaaTIter::reuse(const bool switch_tablet)
{
  if ((OB_NOT_NULL(dim_iters_) && 0 == dim_iters_->count())) {
    // do nothing
  } else {
    bm25_param_estimator_.reuse(switch_tablet);
  }
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i))->reuse();
    }
  }
  ObSRDaaTIterImpl::reuse(switch_tablet);
}

int ObTextDaaTIter::pre_process()
{
  int ret = OB_SUCCESS;
  if (dim_iters_->count() == 0) {
    ret = OB_ITER_END;
  } else if (nullptr != iter_param_->relevance_expr_) {
    if (OB_FAIL(bm25_param_estimator_.do_estimation(*iter_param_->eval_ctx_))) {
    }
  }
  return ret;
}

int ObTextDaaTIter::get_next_rows(const int64_t capacity, int64_t &count)
{
  const ObDatumMeta *meta = is_inited_ ? &iter_param_->id_proj_expr_->datum_meta_ : nullptr;
  // Small natural-language unions can merge their heads directly. For scored
  // queries stay within the simple-merger limit so equal IDs are accumulated
  // in the same iterator-index order, preserving floating-point results.
  if (nullptr != meta
      && mode_flag_ == sql::ObMatchAgainstMode::NATURAL_LANGUAGE_MODE
      && !function_lookup_mode_ && dim_iters_->count() > 0 && dim_iters_->count() <= 8
      && (nullptr == iter_param_->relevance_expr_
          || dim_iters_->count() <= ObSRSimpleMerger::USE_SIMPLE_MERGER_MAX_TABLE_CNT)
      && (meta->type_ == common::ObUInt64Type
          || (meta->type_ == common::ObVarcharType && meta->cs_type_ == CS_TYPE_BINARY))) {
    return get_short_union_rows(capacity, count);
  }
  return ObSRDaaTIterImpl::get_next_rows(capacity, count);
}

int ObTextDaaTIter::get_short_union_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  count = 0;
  const int64_t dim_count = dim_iters_->count();
  const int64_t real_capacity = MIN(capacity, iter_param_->max_batch_size_);
  const common::ObLimitParam &limit = *iter_param_->limit_param_;
  const bool integer_id = iter_param_->id_proj_expr_->datum_meta_.type_ == common::ObUInt64Type;
  const bool scored = nullptr != iter_param_->relevance_expr_;
  ObTextRetrievalDaaTTokenIter *tokens[8];
  const ObDatum **ids = &iter_domain_ids_.at(0);
  int64_t *pending = &next_round_iter_idxes_.at(0);
  for (int64_t i = 0; i < dim_count; ++i) {
    tokens[i] = static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i));
  }
  const auto compare_id = [integer_id](const ObDatum &left, const ObDatum &right) {
    if (integer_id) {
      return left.get_uint64() < right.get_uint64() ? -1
          : (left.get_uint64() > right.get_uint64() ? 1 : 0);
    }
    const int64_t size = MIN(left.len_, right.len_);
    const int comparison = size > 0 ? MEMCMP(left.ptr_, right.ptr_, size) : 0;
    return comparison != 0 ? comparison
        : (left.len_ < right.len_ ? -1 : (left.len_ > right.len_ ? 1 : 0));
  };
  if (limit.is_valid() && output_row_cnt_ >= limit.limit_) {
    ret = OB_ITER_END;
  } else if (capacity > 0 && OB_FAIL(pre_process())) {
  }
  while (OB_SUCC(ret) && count < real_capacity) {
    for (int64_t i = 0; OB_SUCC(ret) && i < next_round_cnt_; ++i) {
      const int64_t idx = pending[i];
      if (OB_FAIL(tokens[idx]->get_next_row())) {
        if (OB_ITER_END == ret) {
          ids[idx] = nullptr;
          ret = OB_SUCCESS;
        }
      } else if (OB_FAIL(tokens[idx]->get_curr_id(ids[idx]))) {
      } else if (OB_ISNULL(ids[idx]) || OB_UNLIKELY(ids[idx]->is_null())) {
        ret = OB_ERR_UNEXPECTED;
      }
    }
    next_round_cnt_ = 0;
    const ObDatum *minimum = nullptr;
    for (int64_t i = 0; OB_SUCC(ret) && i < dim_count; ++i) {
      if (nullptr == ids[i]) {
        continue;
      }
      const int comparison = nullptr == minimum ? -1 : compare_id(*ids[i], *minimum);
      if (comparison < 0) {
        minimum = ids[i];
        next_round_cnt_ = 0;
      }
      if (comparison <= 0) {
        pending[next_round_cnt_++] = i;
      }
    }
    if (OB_FAIL(ret)) {
    } else if (nullptr == minimum) {
      ret = OB_ITER_END;
    } else {
      double relevance = 1.0;
      bool need_project = true;
      if (scored) {
        for (int64_t i = 0; OB_SUCC(ret) && i < next_round_cnt_; ++i) {
          const int64_t idx = pending[i];
          double score = 0.0;
          if (OB_FAIL(tokens[idx]->get_curr_score(score))) {
          } else {
            if (nullptr != iter_param_->dim_weights_) {
              score = score * iter_param_->field_boost_ * iter_param_->dim_weights_->at(idx);
            }
            ret = relevance_collector_->collect_one_dim(idx, score);
          }
        }
        if (OB_FAIL(ret)) {
        } else if (OB_FAIL(relevance_collector_->get_result(relevance, need_project))) {
        } else if (need_project && OB_FAIL(ObSRDaaTIterImpl::filter_on_demand(count, relevance, need_project))) {
        }
      }
      if (OB_SUCC(ret) && need_project
          && OB_FAIL(ObSRDaaTIterImpl::cache_result(count, *minimum, relevance))) {
      }
    }
  }
  // Copy before advancing any child: the next refill may reuse its datum array.
  if ((OB_SUCC(ret) || OB_ITER_END == ret) && count > 0) {
    ret = ObSRDaaTIterImpl::project_results(count);
  }
  return ret;
}

ObTextBMWIter::ObTextBMWIter()
  : ObSRBMWIterImpl(),
    bm25_param_estimator_() {}

void ObTextBMWIter::reuse(const bool switch_tablet)
{
  bm25_param_estimator_.reuse(switch_tablet);
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i))->reuse();
    }
  }
  ObSRBMWIterImpl::reuse(switch_tablet);
}

void ObTextBMWIter::reset()
{
  bm25_param_estimator_.reset();
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i))->reset();
    }
  }
  ObSRBMWIterImpl::reset();
}

int ObTextBMWIter::init(const ObTextDaaTParam &param)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(param.base_param_) || OB_ISNULL(param.dim_iters_) || OB_ISNULL(param.allocator_)
      || OB_ISNULL(param.relevance_collector_)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObSRBMWIterImpl::init(*param.base_param_, *param.dim_iters_,
                                           *param.allocator_, *param.relevance_collector_))) {
  } else if (OB_FAIL(bm25_param_estimator_.init(param.bm25_param_est_ctx_))) {
  }
  return ret;
}

int ObTextBMWIter::get_next_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
  } else if (dim_iters_->count() == 0) {
    ret = OB_ITER_END;
  } else if (OB_FAIL(bm25_param_estimator_.do_estimation(*iter_param_->eval_ctx_))) {
  }

  if (FAILEDx(ObSRBMWIterImpl::get_next_rows(capacity, count))) {
    if (OB_UNLIKELY(OB_ITER_END != ret)) {
    }
  }
  return ret;
}

int ObTextBMWIter::init_before_wand_process()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!bm25_param_estimator_.is_estimated())) {
    ret = OB_ERR_UNEXPECTED;
  }

  for (int64_t i = 0; OB_SUCC(ret) && i < dim_iters_->count(); ++i) {
    ObTextRetrievalBlockMaxIter *block_max_iter = static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i));
    if (OB_FAIL(block_max_iter->init_block_max_iter(
        bm25_param_estimator_.get_total_doc_cnt(), bm25_param_estimator_.get_avg_doc_token_cnt()))) {
    }
  }
  return ret;
}

} // namespace storage
} // namespace oceanbase

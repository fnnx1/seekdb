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
  for (int64_t i = 0; i < SHORT_UNION_MAX_DIMS; ++i) {
    presence_batches_[i].reset();
  }
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
  for (int64_t i = 0; i < SHORT_UNION_MAX_DIMS; ++i) {
    presence_batches_[i].reset();
  }
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
      && !function_lookup_mode_ && dim_iters_->count() > 0 && dim_iters_->count() <= SHORT_UNION_MAX_DIMS
      && (nullptr == iter_param_->relevance_expr_
          || dim_iters_->count() <= ObSRSimpleMerger::USE_SIMPLE_MERGER_MAX_TABLE_CNT)
      && (meta->type_ == common::ObUInt64Type
          || (meta->type_ == common::ObVarcharType && meta->cs_type_ == CS_TYPE_BINARY))) {
    if (nullptr == iter_param_->relevance_expr_ && iter_param_->eval_ctx_->is_vectorized()
        && iter_param_->id_proj_expr_->is_batch_result()) {
      return dim_iters_->count() == 2 ? get_fixed_presence_union_rows<2>(capacity, count)
          : dim_iters_->count() == 3 ? get_fixed_presence_union_rows<3>(capacity, count)
          : get_presence_union_rows(capacity, count);
    }
    return get_short_union_rows(capacity, count);
  }
  return ObSRDaaTIterImpl::get_next_rows(capacity, count);
}

template <int64_t DIMS>
int ObTextDaaTIter::get_fixed_presence_union_rows(const int64_t capacity, int64_t &count)
{
  static_assert(DIMS == 2 || DIMS == 3);
  int ret = OB_SUCCESS;
  count = 0;
  const int64_t real_capacity = MIN(capacity, iter_param_->max_batch_size_);
  const common::ObLimitParam &limit = *iter_param_->limit_param_;
  const bool integer_id = iter_param_->id_proj_expr_->datum_meta_.type_ == common::ObUInt64Type;
  ObDatum *output_ids = iter_param_->id_proj_expr_->locate_batch_datums(*iter_param_->eval_ctx_);
  bool scan_datums_restored = false;
  bool refill_needed = true;
  if (limit.is_valid() && output_row_cnt_ >= limit.limit_) {
    ret = OB_ITER_END;
  }
  while (OB_SUCC(ret) && count < real_capacity && refill_needed) {
    // Refills may write through the shared scan expression. Once an output ID
    // borrows a child buffer, return it before allowing any further refill.
    if (count > 0) {
      break;
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < DIMS; ++i) {
      PresenceIdBatch &batch = presence_batches_[i];
      if (!batch.ended_ && batch.pos_ == batch.count_) {
        if (!scan_datums_restored) {
          iter_param_->id_proj_expr_->locate_datums_for_update(
              *iter_param_->eval_ctx_, iter_param_->max_batch_size_);
          scan_datums_restored = true;
        }
        auto *token = static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i));
        if (OB_FAIL(token->get_next_doc_id_batch(batch.ids_, batch.count_))) {
          if (OB_ITER_END == ret) {
            batch.ended_ = true;
            ret = OB_SUCCESS;
          }
        } else if (OB_UNLIKELY(batch.count_ <= 0 || nullptr == batch.ids_)) {
          ret = OB_ERR_UNEXPECTED;
        }
        batch.pos_ = 0;
      }
    }
    refill_needed = false;
    // A fixed small fan-in needs only one comparison per remaining head.
    // Do not find a second minimum, allocate a winner list, or recheck refill
    // conditions for every emitted ID. Equal heads advance together.
    while (OB_SUCC(ret) && !refill_needed && count < real_capacity) {
      const ObDatum *minimum = nullptr;
      uint32_t equal_mask = 0;
      for (int64_t i = 0; OB_SUCC(ret) && i < DIMS; ++i) {
        const PresenceIdBatch &batch = presence_batches_[i];
        if (!batch.ended_) {
          const ObDatum &head = batch.ids_[batch.pos_].get_datum_fast();
          int comparison = -1;
          if (OB_UNLIKELY(head.is_null())) {
            ret = OB_ERR_UNEXPECTED;
          } else if (nullptr != minimum) {
            if (integer_id) {
              comparison = head.get_uint64() < minimum->get_uint64() ? -1
                  : (head.get_uint64() > minimum->get_uint64() ? 1 : 0);
            } else {
              ret = compare_binary_retrieval_id(head, *minimum, comparison, nullptr);
            }
          }
          if (comparison < 0) {
            minimum = &head;
            equal_mask = 1U << i;
          } else if (comparison == 0) {
            equal_mask |= 1U << i;
          }
        }
      }
      if (OB_FAIL(ret)) {
      } else if (nullptr == minimum) {
        ret = OB_ITER_END;
      } else {
        ++input_row_cnt_;
        if (!limit.is_valid() || input_row_cnt_ > limit.offset_) {
          output_ids[count++].set_datum(*minimum);
          ++output_row_cnt_;
          if (limit.is_valid() && output_row_cnt_ >= limit.limit_) {
            ret = OB_ITER_END;
          }
        }
        for (int64_t i = 0; i < DIMS; ++i) {
          if ((equal_mask & (1U << i)) != 0) {
            PresenceIdBatch &batch = presence_batches_[i];
            ++batch.pos_;
            refill_needed |= batch.pos_ == batch.count_;
          }
        }
      }
    }
  }
  if ((OB_SUCC(ret) || OB_ITER_END == ret) && count > 0) {
    project_presence_results(count);
    ret = OB_SUCCESS;
  }
  return ret;
}

int ObTextDaaTIter::get_presence_union_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  count = 0;
  const int64_t dim_count = dim_iters_->count();
  const int64_t real_capacity = MIN(capacity, iter_param_->max_batch_size_);
  const common::ObLimitParam &limit = *iter_param_->limit_param_;
  const bool integer_id = iter_param_->id_proj_expr_->datum_meta_.type_ == common::ObUInt64Type;
  int64_t equal_heads[SHORT_UNION_MAX_DIMS];
  ObDatum *output_ids = iter_param_->id_proj_expr_->locate_batch_datums(*iter_param_->eval_ctx_);
  bool batch_boundary = false;
  bool scan_datums_restored = false;
  const auto cache_id = [&](const ObDatum &id) __attribute__((always_inline)) {
    ++input_row_cnt_;
    if (limit.is_valid() && input_row_cnt_ <= limit.offset_) {
      // OFFSET consumes IDs but does not retain their buffer references.
    } else {
      OB_ASSERT(count < real_capacity);
      // Child caches own these bytes. Once any ID has been projected, the
      // refill guard below keeps all of those caches stable until we return.
      output_ids[count++].set_datum(id);
      ++output_row_cnt_;
      if (limit.is_valid() && output_row_cnt_ >= limit.limit_) {
        ret = OB_ITER_END;
      }
    }
  };
  const auto compare_id = [integer_id, &ret](const ObDatum &left, const ObDatum &right)
      __attribute__((always_inline)) {
    if (integer_id) {
      return left.get_uint64() < right.get_uint64() ? -1
          : (left.get_uint64() > right.get_uint64() ? 1 : 0);
    }
    int comparison = 0;
    // This path accepts only non-null stored IDs with binary semantics. The
    // helper also preserves extend-datum handling and variable-length ordering.
    const int cmp_ret = compare_binary_retrieval_id(left, right, comparison, nullptr);
    if (OB_UNLIKELY(OB_SUCCESS != cmp_ret)) {
      ret = cmp_ret;
    }
    return comparison;
  };
  if (limit.is_valid() && output_row_cnt_ >= limit.limit_) {
    ret = OB_ITER_END;
  }
  while (OB_SUCC(ret) && !batch_boundary && count < real_capacity) {
    const ObDatum *minimum = nullptr;
    const ObDatum *second = nullptr;
    int64_t equal_count = 0;
    for (int64_t i = 0; OB_SUCC(ret) && i < dim_count; ++i) {
      PresenceIdBatch &batch = presence_batches_[i];
      if (!batch.ended_ && batch.pos_ == batch.count_) {
        // Returned rows alias these child-owned buffers. Once this output
        // batch retains any ID, finish it before allowing another child read:
        // all borrowed IDs then stay valid until our next get_next_rows call.
        if (count > 0) {
          batch_boundary = true;
          break;
        }
        if (!scan_datums_restored) {
          // The projection and all inverted-index scans share this expression.
          // The previous output may have pointed its datums into one child's
          // ID cache. Detach every scan slot before a storage reader writes to
          // it, or that reader can overwrite another child's unconsumed head.
          // This call holds no aliases while count is zero; after its first
          // output, the guard above prohibits any further refill. Restoring
          // once before the first refill in this call is therefore sufficient.
          iter_param_->id_proj_expr_->locate_datums_for_update(
              *iter_param_->eval_ctx_, iter_param_->max_batch_size_);
          scan_datums_restored = true;
        }
        auto *token = static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i));
        if (OB_FAIL(token->get_next_doc_id_batch(batch.ids_, batch.count_))) {
          if (OB_ITER_END == ret) {
            batch.ended_ = true;
            ret = OB_SUCCESS;
          }
        }
        batch.pos_ = 0;
      }
      if (OB_FAIL(ret) || batch.ended_) {
      } else {
        const ObDatum &head = batch.ids_[batch.pos_].get_datum();
        if (OB_UNLIKELY(head.is_null())) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          const int comparison = nullptr == minimum ? -1 : compare_id(head, *minimum);
          if (comparison < 0) {
            second = minimum;
            minimum = &head;
            equal_count = 0;
          } else if (comparison > 0
              && (nullptr == second || compare_id(head, *second) < 0)) {
            second = &head;
          }
          if (comparison <= 0) {
            equal_heads[equal_count++] = i;
          }
        }
      }
    }
    if (OB_FAIL(ret) || batch_boundary) {
    } else if (nullptr == minimum) {
      ret = OB_ITER_END;
    } else if (equal_count > 1) {
      cache_id(*minimum);
      for (int64_t i = 0; i < equal_count; ++i) {
        ++presence_batches_[equal_heads[i]].pos_;
      }
    } else {
      PresenceIdBatch &batch = presence_batches_[equal_heads[0]];
      // The head was already checked against every other active dimension;
      // with a unique minimum it is strictly before second. Consume it once
      // here instead of repeating the same comparison at the start of the run.
      cache_id(*minimum);
      ++batch.pos_;
      // Drain a disjoint sorted run without repeating all head comparisons.
      // Stop before the second head so overlaps are deduplicated above. Neither
      // buffer can refill within this loop, including when an output batch ends.
      while (OB_SUCC(ret) && count < real_capacity && batch.pos_ < batch.count_) {
        const ObDatum &id = batch.ids_[batch.pos_].get_datum();
        if (OB_UNLIKELY(id.is_null())) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          const int comparison = nullptr == second ? -1 : compare_id(id, *second);
          if (OB_FAIL(ret)) {
          } else if (comparison >= 0) {
            break;
          } else {
            cache_id(id);
            ++batch.pos_;
          }
        }
      }
    }
  }
  if ((OB_SUCC(ret) || OB_ITER_END == ret) && count > 0) {
    project_presence_results(count);
    ret = OB_SUCCESS;
  }
  return ret;
}

void ObTextDaaTIter::project_presence_results(const int64_t count)
{
  ObEvalCtx &eval_ctx = *iter_param_->eval_ctx_;
  ObExpr &id_expr = *iter_param_->id_proj_expr_;
  id_expr.get_evaluated_flags(eval_ctx).set_all(count);
  id_expr.set_evaluated_projected(eval_ctx);
  if (iter_param_->need_project_relevance()) {
    ObExpr &presence_expr = *iter_param_->relevance_proj_expr_;
    const int64_t size = presence_expr.is_batch_result() ? count : 1;
    // set_double writes through datum.ptr_; locate_batch_datums alone does not
    // initialize it. Restore this expression's reserved buffers for either its
    // batch slots or its single scalar slot before writing the presence value.
    ObDatum *presence_datums = presence_expr.locate_datums_for_update(eval_ctx, size);
    for (int64_t i = 0; i < size; ++i) {
      presence_datums[i].set_double(1.0);
    }
    if (presence_expr.is_batch_result()) {
      presence_expr.get_evaluated_flags(eval_ctx).set_all(count);
    }
    presence_expr.set_evaluated_projected(eval_ctx);
  }
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

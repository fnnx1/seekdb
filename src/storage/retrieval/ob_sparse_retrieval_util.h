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

#ifndef OB_SPARSE_RETRIEVAL_UTIL_H_
#define OB_SPARSE_RETRIEVAL_UTIL_H_

#include "common/datum/ob_datum.h"
#include "share/datum/ob_datum_funcs.h"

namespace oceanbase
{
namespace storage
{

// Encoded document IDs use the binary collation: compare all bytes, including
// trailing spaces. No character-set lookup or Unicode decoding is required.
OB_INLINE int compare_binary_retrieval_id(const ObDatum &lhs, const ObDatum &rhs,
                                      int &cmp_ret, const common::ObDatumAccessContext *ctx)
{
  if (OB_UNLIKELY(lhs.is_ext() || rhs.is_ext())) {
    return ObDatumFuncs::get_basic_func(ObVarcharType, CS_TYPE_BINARY)
        ->null_first_cmp_(lhs, rhs, cmp_ret, ctx);
  } else if (OB_UNLIKELY(lhs.is_null() || rhs.is_null())) {
    cmp_ret = static_cast<int>(rhs.is_null()) - static_cast<int>(lhs.is_null());
  } else if (lhs.ptr_ == rhs.ptr_ && lhs.len_ == rhs.len_) {
    cmp_ret = 0;
  } else if (lhs.len_ == sizeof(uint64_t) && rhs.len_ == sizeof(uint64_t)) {
    // A constant-size comparison lets the compiler use unaligned word loads
    // for common fixed-width IDs, preserving unsigned byte order.
    cmp_ret = MEMCMP(lhs.ptr_, rhs.ptr_, sizeof(uint64_t));
  } else if (lhs.len_ == 2 * sizeof(uint64_t) && rhs.len_ == 2 * sizeof(uint64_t)) {
    // Generated IDs also use two words. A fixed-size byte comparison avoids
    // the variable-length libc path without assuming alignment or word order.
    cmp_ret = MEMCMP(lhs.ptr_, rhs.ptr_, 2 * sizeof(uint64_t));
  } else {
    const int64_t length = MIN(lhs.len_, rhs.len_);
    const int cmp = length == 0 ? 0 : MEMCMP(lhs.ptr_, rhs.ptr_, length);
    cmp_ret = cmp != 0 ? cmp : (lhs.len_ < rhs.len_ ? -1 : lhs.len_ > rhs.len_ ? 1 : 0);
  }
  return OB_SUCCESS;
}

inline common::ObDatumCmpFuncType retrieval_id_cmp(common::ObDatumBasicFuncs *basic_funcs)
{
  const common::ObDatumCmpFuncType cmp = nullptr == basic_funcs ? nullptr : basic_funcs->null_first_cmp_;
  // Encoded IDs can be CHAR/RAW as well as VARCHAR. Select by comparator
  // semantics instead of the SQL type name; LOB-aware comparators stay intact.
  return cmp == ObDatumFuncs::get_basic_func(ObVarcharType, CS_TYPE_BINARY)->null_first_cmp_
      ? compare_binary_retrieval_id : cmp;
}

class ObDomainIdCmp
{
public:
  ObDomainIdCmp() : cmp_func_(nullptr) {}
  ~ObDomainIdCmp() {}
  inline int init(const ObObjMeta &obj_meta);
  inline int compare(const ObDatum &lhs, const ObDatum &rhs, int &cmp_ret) const;
  inline void reset() { cmp_func_ = nullptr; }
private:
  common::ObDatumCmpFuncType cmp_func_;
  DISALLOW_COPY_AND_ASSIGN(ObDomainIdCmp);
};

int ObDomainIdCmp::init(const ObObjMeta &obj_meta)
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!obj_meta.is_valid())) {
    ret = OB_INVALID_ARGUMENT;
    STORAGE_LOG(WARN,"invalid argument", K(ret), K(obj_meta));
  } else {
    common::ObDatumBasicFuncs *id_basic_funcs = ObDatumFuncs::get_basic_func(
      obj_meta.get_type(), obj_meta.get_collation_type());
    if (OB_ISNULL(id_basic_funcs)) {
      ret = OB_ERR_UNEXPECTED;
      STORAGE_LOG(WARN,"failed to get basic functions", K(ret), K(obj_meta));
    } else {
      cmp_func_ = retrieval_id_cmp(id_basic_funcs);
    }
  }
  return ret;
}

int ObDomainIdCmp::compare(const ObDatum &lhs, const ObDatum &rhs, int &cmp_ret) const
{
  int ret = OB_SUCCESS;
  const bool lmax = lhs.is_max();
  const bool lmin = lhs.is_min();
  const bool rmax = rhs.is_max();
  const bool rmin = rhs.is_min();
  if (OB_UNLIKELY(nullptr == cmp_func_)) {
    ret = OB_NOT_INIT;
    STORAGE_LOG(WARN,"not init", K(ret));
  } else if (lhs.is_ext() || rhs.is_ext()) {
    if (OB_UNLIKELY(!(lmax || lmin || rmax || rmin))) {
      ret = OB_ERR_UNEXPECTED;
      STORAGE_LOG(WARN,"unexpected ext datum", K(ret), K(lhs), K(rhs));
    } else if (lmax || rmax) {
      cmp_ret = lmax - rmax;
    } else if (lmin || rmin) {
      cmp_ret = -(lmin - rmin);
    } else {
      ret = OB_ERR_UNEXPECTED;
    }
  } else {
    ret = cmp_func_(lhs, rhs, cmp_ret, nullptr);
  }
  return ret;
}

} // namespace storage
} // namespace oceanbase

#endif

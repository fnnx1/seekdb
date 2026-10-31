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

#include "ob_string_diff_decoder.h"

#include "ob_raw_decoder.h"

namespace oceanbase
{
namespace blocksstable
{
using namespace common;
const ObColumnHeader::Type ObStringDiffDecoder::type_;

ObStringDiffDecoder::ObStringDiffDecoder() : header_(NULL)
{
}

ObStringDiffDecoder::~ObStringDiffDecoder()
{
}

int ObStringDiffDecoder::decode(const ObColumnDecoderCtx &ctx, common::ObDatum &datum, const int64_t row_id,
    const ObBitStream &bs, const char *data, const int64_t len) const
{
  int ret = OB_SUCCESS;
  uint64_t val = STORED_NOT_EXT;
  ObBitStream fix_bs;
  const char *col_data = reinterpret_cast<const char *>(header_) + ctx.col_header_->length_;
  int64_t data_offset = 0;

  if (!is_inited()) {
    ret = OB_NOT_INIT;
  } else if (OB_UNLIKELY(nullptr == data || len < 0)) {
    ret = OB_INVALID_ARGUMENT;
  } else {

    // read extend value bit
    if (ctx.has_extend_value()) {
      if (ctx.is_fix_length()) {
        data_offset = ctx.micro_block_header_->row_count_ * ctx.micro_block_header_->extend_value_bit_;
        data_offset = (data_offset + CHAR_BIT - 1) / CHAR_BIT;
        if (OB_FAIL(ObBitStream::get(reinterpret_cast<const unsigned char *>(col_data),
            row_id * ctx.micro_block_header_->extend_value_bit_,
            ctx.micro_block_header_->extend_value_bit_, val))) {
        }
      } else {
        if (OB_FAIL(bs.get(ctx.col_header_->extend_value_index_,
                ctx.micro_block_header_->extend_value_bit_, val))) {
        }
      }
    }
  }

  if (OB_FAIL(ret)) {
  } else if (STORED_NOT_EXT != val) {
    set_stored_ext_value(datum, static_cast<ObStoredExtValue>(val));
  } else {
    const char *cell_data = NULL;
    int64_t cell_len = 0;
    // get cell data offset and length
    if (ctx.is_fix_length()) {
      cell_data = col_data + data_offset + row_id * header_->length_;
    } else {
      if (OB_FAIL(ObRawDecoder::locate_cell_data(cell_data, cell_len, data, len,
              *ctx.micro_block_header_, *ctx.col_header_, *header_))) {
      }
    }

    char *buf = NULL;
    if (OB_SUCC(ret)) {
      const static uint16_t min_buf_size = 128;
      const int64_t buf_size = std::max(header_->string_size_, min_buf_size);
      if (OB_ISNULL(buf = static_cast<char *>(ctx.allocator_->alloc(buf_size)))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        header_->copy_string(ObStringDiffHeader::LeftToRight(), std::logical_not<uint8_t>(),
            header_->common_data(), buf);
      }
    }

    if (OB_SUCC(ret)) {
      //int64_t cell_len = header_->length_;
      if (header_->is_hex_packing()) {
        ObHexStringUnpacker unpacker(header_->hex_char_array(),
            reinterpret_cast<const unsigned char *>(cell_data));
        header_->copy_hex_string(unpacker,
            static_cast<void (ObHexStringUnpacker::*)(unsigned char &)>(&ObHexStringUnpacker::unpack),
            reinterpret_cast<unsigned char *>(buf));
      } else {
        header_->copy_string(ObStringDiffHeader::LeftToRight(),
            ObStringDiffHeader::LogicTrue<uint8_t>(),
            cell_data, buf);
      }
      datum.pack_ = header_->string_size_;
      datum.ptr_ = buf;
    }
  }

  return ret;
}

int ObStringDiffDecoder::update_pointer(const char *old_block, const char *cur_block)
{
  int ret = OB_SUCCESS;
  if (!is_inited()) {
    ret = OB_NOT_INIT;
  } else if (OB_ISNULL(old_block) || OB_ISNULL(cur_block)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObIColumnDecoder::update_pointer(header_, old_block, cur_block);
  }
  return ret;
}

// Internal call, not check parameters for performance
int ObStringDiffDecoder::batch_decode(
    const ObColumnDecoderCtx &ctx,
    const ObIRowIndex* row_index,
    const int32_t *row_ids,
    const char **cell_datas,
    const int64_t row_cap,
    common::ObDatum *datums) const
{
  UNUSED(cell_datas);
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!is_inited())) {
    ret = OB_NOT_INIT;
  } else {
    const unsigned char *col_data = reinterpret_cast<const unsigned char *>(header_)
        + ctx.col_header_->length_;
    int64_t data_offset = 0;
    if (ctx.has_extend_value()) {
      if (ctx.is_fix_length()) {
        data_offset = ctx.micro_block_header_->row_count_ * ctx.micro_block_header_->extend_value_bit_;
        data_offset = (data_offset + CHAR_BIT - 1) / CHAR_BIT;
        if (OB_FAIL(set_null_datums_from_fixed_column(
            ctx, row_ids, row_cap, col_data, datums))) {
        }
      } else if (OB_FAIL(set_null_datums_from_var_column(
          ctx, row_index, row_ids, row_cap, datums))) {
      }
    }

    // Fill string data
    const uint16_t string_size = header_->string_size_;
    const static uint16_t min_buf_size = 128;
    const int64_t buf_size = std::max(string_size, min_buf_size);
    char *buf = nullptr;
    if (OB_FAIL(ret)) {
    } else if (OB_ISNULL(buf = static_cast<char *>(ctx.allocator_->alloc(buf_size * row_cap)))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (ctx.is_fix_length() && !header_->is_hex_packing()
        && string_size <= min_buf_size && header_->diff_desc_cnt_ <= min_buf_size && row_cap > 1) {
      // Short fixed-width cells share one segment layout. Decode that layout
      // once per batch, then initialize and patch each non-null row together.
      // Keep the original 128-byte allocation stride and datum ownership.
      char common_template[min_buf_size] = {};
      uint16_t dest_offsets[min_buf_size];
      uint16_t source_offsets[min_buf_size];
      uint8_t lengths[min_buf_size];
      int64_t patch_count = 0;
      uint16_t dest_offset = 0;
      uint16_t source_offset = 0;
      header_->copy_string(ObStringDiffHeader::LeftToRight(), std::logical_not<uint8_t>(),
          header_->common_data(), common_template);
      for (int64_t i = 0; i < header_->diff_desc_cnt_; ++i) {
        const auto &desc = header_->diff_descs_[i];
        if (desc.diff_) {
          dest_offsets[patch_count] = dest_offset;
          source_offsets[patch_count] = source_offset;
          lengths[patch_count++] = desc.count_;
          source_offset += desc.count_;
        }
        dest_offset += desc.count_;
      }
      const bool has_nulls = ctx.has_extend_value();
      for (int64_t i = 0; i < row_cap; ++i) {
        if (!has_nulls || !datums[i].is_null()) {
          char *dest = buf + i * buf_size;
          const unsigned char *source = col_data + data_offset + row_ids[i] * header_->length_;
          if (16 == string_size) {
            MEMCPY(dest, common_template, 16);
          } else {
            MEMCPY(dest, common_template, string_size);
          }
          for (int64_t patch = 0; patch < patch_count; ++patch) {
            char *to = dest + dest_offsets[patch];
            const unsigned char *from = source + source_offsets[patch];
            // Constant-size copies avoid a libc call for short changed runs.
            // memcpy also permits unaligned cell and segment boundaries.
            switch (lengths[patch]) {
              case 1: *to = *from; break;
              case 2: MEMCPY(to, from, 2); break;
              case 3: MEMCPY(to, from, 3); break;
              case 4: MEMCPY(to, from, 4); break;
              case 5: MEMCPY(to, from, 5); break;
              case 6: MEMCPY(to, from, 6); break;
              case 7: MEMCPY(to, from, 7); break;
              case 8: MEMCPY(to, from, 8); break;
              case 16: MEMCPY(to, from, 16); break;
              default: MEMCPY(to, from, lengths[patch]); break;
            }
          }
          datums[i].pack_ = string_size;
          datums[i].ptr_ = dest;
        }
      }
      return ret;
    } else if (string_size <= min_buf_size && row_cap > 1) {
      // Common segments are identical for the whole microblock. Assemble them
      // once instead of interpreting the segment descriptors for every row.
      // Initialize the differing positions too: they are overwritten below,
      // but the template copy must not read uninitialized bytes.
      char common_template[min_buf_size] = {};
      header_->copy_string(ObStringDiffHeader::LeftToRight(), std::logical_not<uint8_t>(),
          header_->common_data(), common_template);
      for (int64_t i = 0; i < row_cap; ++i) {
        if (16 == string_size) {
          MEMCPY(buf + i * buf_size, common_template, 16);
        } else {
          MEMCPY(buf + i * buf_size, common_template, string_size);
        }
      }
    } else {
      for (int64_t i = 0; i < row_cap; ++i) {
        header_->copy_string(ObStringDiffHeader::LeftToRight(), std::logical_not<uint8_t>(),
            header_->common_data(), buf + i * buf_size);
      }
    }

    if (OB_FAIL(ret)) {
    } else if (ctx.is_fix_length()) {
      int64_t row_id = 0;
      const unsigned char *cell_data = nullptr;
      char *str_ptr = nullptr;
      for (int64_t i = 0; i < row_cap; ++i) {
        if (ctx.has_extend_value() && datums[i].is_null()) {
          // Skip
        } else {
          row_id = row_ids[i];
          cell_data = col_data + data_offset + row_id * header_->length_;
          str_ptr = buf + i * buf_size;
          if (header_->is_hex_packing()) {
            ObHexStringUnpacker unpacker(header_->hex_char_array(),
                reinterpret_cast<const unsigned char *>(cell_data));
            header_->copy_hex_string(unpacker,
                static_cast<void (ObHexStringUnpacker::*)(unsigned char &)>(&ObHexStringUnpacker::unpack),
                reinterpret_cast<unsigned char *>(str_ptr));
          } else {
            header_->copy_string(
              ObStringDiffHeader::LeftToRight(),
              ObStringDiffHeader::LogicTrue<uint8_t>(),
              cell_data, str_ptr);
          }
          datums[i].pack_ = string_size;
          datums[i].ptr_ = str_ptr;
        }
      }
    } else {
      int64_t row_id = 0;
      const char *row_data = nullptr;
      int64_t row_len = 0;
      const char *cell_data = nullptr;
      int64_t cell_len = 0;
      char *str_ptr = nullptr;
      for (int64_t i = 0; OB_SUCC(ret) && i < row_cap; ++i) {
        row_id = row_ids[i];
        str_ptr = buf + i * buf_size;
        if (ctx.has_extend_value() && datums[i].is_null()) {
          // Do nothing
        } else if (OB_FAIL(locate_row_data(ctx, row_index, row_id, row_data, row_len))) {
        } else if (OB_FAIL(ObRawDecoder::locate_cell_data(cell_data, cell_len, row_data, row_len,
            *ctx.micro_block_header_, *ctx.col_header_, *header_))) {
        } else {
          if (header_->is_hex_packing()) {
            ObHexStringUnpacker unpacker(header_->hex_char_array(),
                reinterpret_cast<const unsigned char *>(cell_data));
            header_->copy_hex_string(unpacker,
                static_cast<void (ObHexStringUnpacker::*)(unsigned char &)>(&ObHexStringUnpacker::unpack),
                reinterpret_cast<unsigned char *>(str_ptr));
          } else {
            header_->copy_string(
              ObStringDiffHeader::LeftToRight(),
              ObStringDiffHeader::LogicTrue<uint8_t>(),
              cell_data, str_ptr);
          }
          datums[i].pack_ = string_size;
          datums[i].ptr_ = str_ptr;
        }
      }
    }
  }
  return ret;
}

} // end namespace blocksstable
} // end namespace oceanbase

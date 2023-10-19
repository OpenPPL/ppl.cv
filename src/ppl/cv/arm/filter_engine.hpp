#include "ppl/cv/types.h"
#include "ppl/common/sys.h"
#include "common.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstring>

#include <limits.h>
#include <algorithm>
#include <vector>
#include <memory>

namespace ppl {
namespace cv {
namespace arm {

// Taken from OpenCV
inline int borderInterpolate(int p, int len, BorderType border_type)
{
    if ((unsigned)p < (unsigned)len) { // if(p >= 0 && p < len)
        return p;
    }

    if (border_type == ppl::cv::BORDER_REPLICATE) {
        p = p < 0 ? 0 : len - 1;
    } else if (border_type == ppl::cv::BORDER_REFLECT || border_type == ppl::cv::BORDER_REFLECT_101) {
        int delta = border_type == ppl::cv::BORDER_REFLECT_101;
        if (len == 1) return 0;
        do {
            if (p < 0)
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        } while ((unsigned)p >= (unsigned)len); // while(p < 0 || p >= len);
    } else { // if( border_type == BORDER_CONSTANT )
        p = -1;
    }

    return p;
}

#define VEC_ALIGN 64

template <typename ST, typename MT, typename DT, typename RowFilterType, typename ColumnFilterType>
class SeparableFilterEngine {
public:
    SeparableFilterEngine(int _height,
                          int _width,
                          int _channels,
                          int _kHeight,
                          int _kWidth,
                          BorderType _borderType,
                          ST _borderValue,
                          RowFilterType _rowFilter,
                          ColumnFilterType _columnFilter)
        : height(_height)
        , width(_width)
        , channels(_channels)
        , kHeight(_kHeight)
        , kWidth(_kWidth)
        , borderType(_borderType)
        , borderValue(_borderValue)
        , rowFilter(_rowFilter)
        , columnFilter(_columnFilter)
    {
        init();
    };

    void process(const ST *src, int inWidthStride, DT *dst, int outWidthStride);

private:
    void init();

    // initialized on object construction
    int height;
    int width;
    int channels;
    int kHeight;
    int kWidth;
    BorderType borderType;
    ST borderValue;
    // initialized in init function
    std::vector<int> borderTab;
    std::vector<ST> srcRowBuf;
    std::vector<MT> constBorderRow;
    std::vector<uint8_t> ringBuf;
    std::vector<uint8_t *> bufRowsPtrs;
    uint8_t *alignedRingBufPtr;
    int bufStep;
    int fill_x_left;
    int fill_x_right;
    int rowCount;
    int startY;

    RowFilterType rowFilter;
    ColumnFilterType columnFilter;
};

template <typename ST, typename MT, typename DT, typename RowFilterType, typename ColumnFilterType>
void SeparableFilterEngine<ST, MT, DT, RowFilterType, ColumnFilterType>::init()
{
    // init border info
    int borderMaxFill = std::max(kWidth - 1, 1);
    borderTab.resize(borderMaxFill);

    // init buffer for intermediate result and column filter
    int maxBufRows = kHeight + 3;
    bufRowsPtrs.resize(maxBufRows);

    // init buffer for row filter
    int paddedInputWidth = width + kWidth - 1;
    srcRowBuf.resize(channels * paddedInputWidth);
    if (borderType == ppl::cv::BORDER_CONSTANT) {
        // set up border
        constBorderRow.resize(channels * width);
        std::fill(srcRowBuf.begin(), srcRowBuf.end(), borderValue);
        rowFilter.operator()(srcRowBuf.data(), constBorderRow.data(), width, channels);
    }

    bufStep = sizeof(MT) * width * channels;
    bufStep = bufStep - (bufStep & (VEC_ALIGN - 1)) + VEC_ALIGN;
    size_t ringbuf_real_size = bufStep * maxBufRows + VEC_ALIGN;
    ringBuf.resize(ringbuf_real_size);
    void *alignedPtr = ringBuf.data();
    std::align(VEC_ALIGN, bufStep * maxBufRows, alignedPtr, ringbuf_real_size);
    alignedRingBufPtr = reinterpret_cast<uint8_t *>(alignedPtr);

    // set up fill info
    int anchor_x = kWidth / 2;
    fill_x_left = anchor_x;
    fill_x_right = kWidth - anchor_x - 1;

    // compute border tables
    int borderLength = std::max(kWidth - 1, 1);
    borderTab.resize(borderLength * channels);
    if (fill_x_left > 0 || fill_x_right > 0) {
        if (borderType == ppl::cv::BORDER_CONSTANT) {
            ; // pass
        } else {
            int btab_esz = channels;
            int *btab = borderTab.data();
            for (int i = 0; i < fill_x_left; i++) {
                int p0 = (borderInterpolate(-fill_x_left + i, width, borderType)) * btab_esz;
                for (int j = 0; j < btab_esz; j++) {
                    btab[i * btab_esz + j] = p0 + j;
                }
            }

            for (int i = 0; i < fill_x_right; i++) {
                int p0 = (borderInterpolate(width + i, width, borderType)) * btab_esz;
                for (int j = 0; j < btab_esz; j++) {
                    btab[(i + fill_x_left) * btab_esz + j] = p0 + j;
                }
            }
        }
    }

    // setup varibles
    rowCount = 0;
    startY = 0;
}

template <typename ST, typename MT, typename DT, typename RowFilterType, typename ColumnFilterType>
void SeparableFilterEngine<ST, MT, DT, RowFilterType, ColumnFilterType>::process(const ST *src,
                                                                                 int inWidthStride,
                                                                                 DT *dst,
                                                                                 int outWidthStride)
{
    int count = height;
    int anchor_y = kHeight / 2;
    const int *btab = borderTab.data();
    int esz = channels * sizeof(ST);
    int rowFilterInputSize = width + kWidth - 1;
    int bufRows = bufRowsPtrs.size();
    bool makeBorder = (fill_x_left > 0 || fill_x_right > 0) && borderType != ppl::cv::BORDER_CONSTANT;

    int dy = 0;
    int i = 0;
    for (;; dst += outWidthStride * i, dy += i) {
        int dcount = bufRows - anchor_y - startY - rowCount;
        dcount = dcount > 0 ? dcount : bufRows - kHeight + 1;
        dcount = std::min(dcount, count);
        count -= dcount;
        for (; dcount-- > 0; src += inWidthStride) {
            int bufRowIdx = (startY + rowCount) % bufRows;
            uint8_t *brow = alignedRingBufPtr + bufRowIdx * bufStep;
            ST *row = srcRowBuf.data();

            // rowCount: Rows in bufRows
            // startY: where bufRows Start
            if (++rowCount > bufRows) {
                --rowCount;
                ++startY;
            }

            memcpy(reinterpret_cast<void *>(row + fill_x_left * channels),
                   src,
                   (rowFilterInputSize - fill_x_left - fill_x_right) * esz);

            if (makeBorder) {
                for (int i = 0; i < fill_x_left * channels; i++) {
                    row[i] = src[btab[i]];
                }
                for (int i = 0; i < fill_x_right * channels; i++) {
                    row[i + (rowFilterInputSize - fill_x_right) * channels] = src[btab[i + fill_x_left * channels]];
                }
            }

            rowFilter.operator()(row, (MT *)brow, width, channels);
        }

        // setup bufRowsPtrs
        MT **bufRowsPtr_data = reinterpret_cast<MT **>(bufRowsPtrs.data());
        int max_i = std::min(bufRows, height - dy + (kHeight - 1));
        for (i = 0; i < max_i; i++) {
            int srcY = borderInterpolate(dy + i - anchor_y, height, borderType);
            if (srcY < 0) {
                // can happen only with constant border type
                bufRowsPtr_data[i] = constBorderRow.data();
            } else {
                // CV_Assert(srcY >= this_.startY);
                if (srcY >= startY + rowCount) { break; }
                int bi = srcY % bufRows;
                bufRowsPtr_data[i] = reinterpret_cast<MT *>(alignedRingBufPtr + bi * bufStep);
            }
        }

        if (i < kHeight) { break; }
        i -= kHeight - 1;

        columnFilter.operator()(bufRowsPtr_data, dst, outWidthStride, i, width * channels);
    }

    return;
}

template <typename ST, typename DT, typename FilterType>
class FilterEngine {
public:
    FilterEngine(int _height,
                 int _width,
                 int _channels,
                 int _ksize,
                 BorderType _borderType,
                 ST _borderValue,
                 FilterType _filter)
        : height(_height)
        , width(_width)
        , channels(_channels)
        , kHeight(_ksize)
        , kWidth(_ksize)
        , borderType(_borderType)
        , borderValue(_borderValue)
        , filter(_filter)
    {
        init();
    };

    void process(const ST *src, int inWidthStride, DT *dst, int outWidthStride);

private:
    void init();

    // initialized on object construction
    int height;
    int width;
    int channels;
    int kHeight;
    int kWidth;
    BorderType borderType;
    ST borderValue;
    // initialized in init function
    std::vector<int> borderTab;
    std::vector<ST> srcRowBuf;
    std::vector<ST> constBorderRow;
    std::vector<uint8_t> ringBuf;
    std::vector<uint8_t *> bufRowsPtrs;
    uint8_t *alignedRingBufPtr;
    int bufStep;
    int fill_x_left;
    int fill_x_right;
    int rowCount;
    int startY;

    FilterType filter;
};

template <typename ST, typename DT, typename FilterType>
void FilterEngine<ST, DT, FilterType>::init()
{
    // init border info
    int borderMaxFill = std::max(kWidth - 1, 1);
    borderTab.resize(borderMaxFill);

    // init buffer for intermediate result and column filter
    int maxBufRows = kHeight + 3;
    bufRowsPtrs.resize(maxBufRows);

    // init buffer for row filter
    int paddedInputWidth = width + kWidth - 1;
    srcRowBuf.resize(channels * paddedInputWidth);
    if (borderType == ppl::cv::BORDER_CONSTANT) {
        // set up border for non-separable
        constBorderRow.resize(channels * width);
        std::fill(constBorderRow.begin(), constBorderRow.end(), borderValue);
    }

    bufStep = sizeof(DT) * (width + kWidth - 1) * channels;
    bufStep = bufStep - (bufStep & (VEC_ALIGN - 1)) + VEC_ALIGN;
    size_t ringbuf_real_size = bufStep * maxBufRows + VEC_ALIGN;
    ringBuf.resize(ringbuf_real_size);
    void *alignedPtr = ringBuf.data();
    std::align(VEC_ALIGN, bufStep * maxBufRows, alignedPtr, ringbuf_real_size);
    alignedRingBufPtr = reinterpret_cast<uint8_t *>(alignedPtr);

    // set up fill info
    int anchor_x = kWidth / 2;
    fill_x_left = anchor_x;
    fill_x_right = kWidth - anchor_x - 1;

    // compute border tables
    int borderLength = std::max(kWidth - 1, 1);
    borderTab.resize(borderLength * channels);
    if (fill_x_left > 0 || fill_x_right > 0) {
        if (borderType == ppl::cv::BORDER_CONSTANT) {
            ; // not implemented!
        } else {
            int btab_esz = channels;
            int *btab = borderTab.data();
            for (int i = 0; i < fill_x_left; i++) {
                int p0 = (borderInterpolate(-fill_x_left + i, width, borderType)) * btab_esz;
                for (int j = 0; j < btab_esz; j++) {
                    btab[i * btab_esz + j] = p0 + j;
                }
            }

            for (int i = 0; i < fill_x_right; i++) {
                int p0 = (borderInterpolate(width + i, width, borderType)) * btab_esz;
                for (int j = 0; j < btab_esz; j++) {
                    btab[(i + fill_x_left) * btab_esz + j] = p0 + j;
                }
            }
        }
    }

    // setup varibles
    rowCount = 0;
    startY = 0;
}

template <typename ST, typename DT, typename FilterType>
void FilterEngine<ST, DT, FilterType>::process(const ST *src, int inWidthStride, DT *dst, int outWidthStride)
{
    int count = height;
    int anchor_y = kHeight / 2;
    const int *btab = borderTab.data();
    int esz = channels * sizeof(ST);
    int rowFilterInputSize = width + kWidth - 1;
    int bufRows = bufRowsPtrs.size();
    bool makeBorder = (fill_x_left > 0 || fill_x_right > 0) && borderType != ppl::cv::BORDER_CONSTANT;

    int dy = 0;
    int i = 0;
    for (;; dst += outWidthStride * i, dy += i) {
        int dcount = bufRows - anchor_y - startY - rowCount;
        dcount = dcount > 0 ? dcount : bufRows - kHeight + 1;
        dcount = std::min(dcount, count);
        count -= dcount;
        for (; dcount-- > 0; src += inWidthStride) {
            int bufRowIdx = (startY + rowCount) % bufRows;
            uint8_t *brow = alignedRingBufPtr + bufRowIdx * bufStep;
            ST *row = reinterpret_cast<ST *>(brow);

            // rowCount: Rows in bufRows
            // startY: where bufRows Start
            if (++rowCount > bufRows) {
                --rowCount;
                ++startY;
            }

            memcpy(reinterpret_cast<void *>(row + fill_x_left * channels),
                   src,
                   (rowFilterInputSize - fill_x_left - fill_x_right) * esz);

            if (makeBorder) {
                for (int i = 0; i < fill_x_left * channels; i++) {
                    row[i] = src[btab[i]];
                }
                for (int i = 0; i < fill_x_right * channels; i++) {
                    row[i + (rowFilterInputSize - fill_x_right) * channels] = src[btab[i + fill_x_left * channels]];
                }
            }
        }

        // setup bufRowsPtrs
        ST **bufRowsPtr_data = reinterpret_cast<ST **>(bufRowsPtrs.data());
        int max_i = std::min(bufRows, height - dy + (kHeight - 1));
        for (i = 0; i < max_i; i++) {
            int srcY = borderInterpolate(dy + i - anchor_y, height, borderType);
            if (srcY < 0) {
                // can happen only with constant border type
                bufRowsPtr_data[i] = constBorderRow.data();
            } else {
                // CV_Assert(srcY >= this_.startY);
                if (srcY >= startY + rowCount) { break; }
                int bi = srcY % bufRows;
                bufRowsPtr_data[i] = reinterpret_cast<ST *>(alignedRingBufPtr + bi * bufStep);
            }
        }

        if (i < kHeight) { break; }
        i -= kHeight - 1;

        filter.operator()(bufRowsPtr_data, dst, outWidthStride, i, width, channels);
    }

    return;
}

}
}
} // namespace ppl::cv::arm

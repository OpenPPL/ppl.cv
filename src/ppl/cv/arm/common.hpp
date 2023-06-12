#ifndef __ST_HPC_PPL_CV_AARCH64_COMMON_H_
#define __ST_HPC_PPL_CV_AARCH64_COMMON_H_

namespace ppl {
namespace cv {
namespace arm {

inline void prefetch(const void *ptr, size_t offset = 512)
{
#if defined __GNUC__
    __builtin_prefetch(reinterpret_cast<const char *>(ptr) + offset);
#elif defined _MSC_VER && defined CAROTENE_NEON
    __prefetch(reinterpret_cast<const char *>(ptr) + offset);
#else
    (void)ptr;
    (void)offset;
#endif
}

static inline void prefetch_range(const void *addr, size_t len)
{
#ifdef ARCH_HAS_PREFETCH
    char *cp;
    char *end = addr + len;

    for (cp = addr; cp < end; cp += PREFETCH_STRIDE)
        __builtin_prefetch(cp);
#endif
}

}}} // namespace ppl::cv::arm
#endif
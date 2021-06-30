#include "benchmark/benchmark.h"
#include "ppl/cv/debug.h"

int main(int argc, char** argv) {
#ifdef PPLCV_BENCHMARK_OPENCV
    cv::setNumThreads(0);
#endif // PPLCV_BENCHMARK_OPENCV
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}

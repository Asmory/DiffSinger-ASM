#include "dsasm_kernels.h"
#include "dsasm_threadpool.h"
#include "dsasm_vocoder.h"
#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <unordered_map>
#include <vector>

using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

static float random_value(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    uint64_t value = state * UINT64_C(2685821657736338717);
    return static_cast<float>((value >> 40) * (1.0 / 16777216.0) - 0.5);
}

struct Stats {
    double median;
    double p90;
    double worst;
};

static Stats stats(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    auto percentile = [&](double fraction) {
        double position = (values.size() - 1) * fraction;
        size_t lo = static_cast<size_t>(position);
        size_t hi = lo + (position > lo);
        return values[lo] + (values[hi] - values[lo]) * (position - lo);
    };
    return {percentile(0.5), percentile(0.9), values.back()};
}

static void print_quality(const std::vector<float> &actual,
        const std::vector<float> &reference) {
    double error2 = 0.0, actual2 = 0.0, reference2 = 0.0, dot = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < actual.size(); i++) {
        double a = actual[i], r = reference[i], error = a - r;
        max_abs = std::max(max_abs, std::abs(error));
        error2 += error * error;
        actual2 += a * a;
        reference2 += r * r;
        dot += a * r;
    }
    double cosine = dot / (std::sqrt(actual2 * reference2) + 1e-300);
    double snr = 10.0 * std::log10((reference2 + 1e-300) / (error2 + 1e-300));
    std::printf("ONEDNN_QUALITY max_abs=%.9g cosine=%.9f SNR=%.2f pass=%s\n",
        max_abs, cosine, snr,
        cosine >= 0.999 && snr >= 25.0 ? "PASS" : "FAIL");
}

int main(int argc, char **argv) try {
    size_t workers = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 6;
    size_t warmup = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 10;
    size_t rounds = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 25;
    size_t dilation = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 1;
    if (!workers || !rounds || !dilation || argc > 5) return 2;

    constexpr int64_t cin = 128, cout = 128, kernel = 7, time = 2048;
    int64_t pad = static_cast<int64_t>(dilation) * (kernel - 1) / 2;
    constexpr size_t input_count = cin * time;
    constexpr size_t weight_count = cout * cin * kernel;
    constexpr size_t output_count = cout * time;
    std::vector<float> input(input_count), weights(weight_count), bias(cout);
    std::vector<float> packed(weight_count), leaky(input_count);
    std::vector<float> asm_output(output_count), dnnl_output(output_count);
    std::vector<float> workspace(cin * (time + 2 * pad));
    uint64_t state = UINT64_C(0x123456789abcdef);
    for (float &value : input) value = random_value(state);
    for (float &value : weights) value = 0.05f * random_value(state);
    for (float &value : bias) value = 0.01f * random_value(state);
    for (int64_t oc = 0; oc < cout; oc++)
        for (int64_t ic = 0; ic < cin; ic++)
            for (int64_t k = 0; k < kernel; k++)
                packed[(((oc / 8) * cin + ic) * kernel + k) * 8 + oc % 8] =
                    weights[(oc * cin + ic) * kernel + k];

    DSAsmThreadPool *pool = ds_threadpool_create(workers);
    if (!pool) return 3;

    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream stream(engine);
    auto source_desc = dnnl::memory::desc(
        {1, cin, time}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ncw);
    auto user_weights_desc = dnnl::memory::desc(
        {cout, cin, kernel}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oiw);
    auto any_weights_desc = dnnl::memory::desc(
        {cout, cin, kernel}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto bias_desc = dnnl::memory::desc(
        {cout}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
    auto destination_desc = dnnl::memory::desc(
        {1, cout, time}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ncw);
    dnnl::primitive_attr attributes;
    attributes.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto descriptor = dnnl::convolution_forward::primitive_desc(
        engine, dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_direct, source_desc, any_weights_desc,
        bias_desc, destination_desc, {1},
        {static_cast<int64_t>(dilation - 1)}, {pad}, {pad}, attributes);
    dnnl::convolution_forward convolution(descriptor);
    dnnl::memory source(source_desc, engine, leaky.data());
    dnnl::memory user_weights(user_weights_desc, engine, weights.data());
    dnnl::memory internal_weights(descriptor.weights_desc(), engine);
    dnnl::reorder(user_weights, internal_weights)
        .execute(stream, user_weights, internal_weights);
    dnnl::memory bias_memory(bias_desc, engine, bias.data());
    dnnl::memory destination(destination_desc, engine, dnnl_output.data());
    dnnl::memory scratchpad(descriptor.scratchpad_desc(), engine);
    stream.wait();
    std::unordered_map<int, dnnl::memory> arguments = {
        {DNNL_ARG_SRC, source},
        {DNNL_ARG_WEIGHTS, internal_weights},
        {DNNL_ARG_BIAS, bias_memory},
        {DNNL_ARG_DST, destination},
        {DNNL_ARG_SCRATCHPAD, scratchpad},
    };

    auto run_asm = [&]() {
        auto begin = Clock::now();
        int rc = ds_vocoder_conv1d_ex_f32_avx2(
            input.data(), packed.data(), bias.data(), asm_output.data(),
            cin, cout, kernel, time, pad, dilation, 8, 1, 0.1f,
            workspace.data(), pool);
        if (rc) std::abort();
        return elapsed_ms(begin);
    };
    auto run_dnnl = [&]() {
        auto begin = Clock::now();
        ds_leaky_relu_f32_avx2(input.data(), leaky.data(), input.size(), 0.1f);
        convolution.execute(stream, arguments);
        stream.wait();
        return elapsed_ms(begin);
    };

    std::vector<double> asm_times, dnnl_times;
    asm_times.reserve(rounds);
    dnnl_times.reserve(rounds);
    for (size_t i = 0; i < warmup + rounds; i++) {
        double a, d;
        if ((i & 1u) == 0) { a = run_asm(); d = run_dnnl(); }
        else { d = run_dnnl(); a = run_asm(); }
        if (i >= warmup) {
            asm_times.push_back(a);
            dnnl_times.push_back(d);
            std::printf("ONEDNN_PAIR index=%zu asm_ms=%.3f dnnl_ms=%.3f\n",
                i - warmup, a, d);
        }
    }
    Stats a = stats(asm_times), d = stats(dnnl_times);
    double gain = (a.median / d.median - 1.0) * 100.0;
    std::printf("ONEDNN_FINAL workers=%zu shape=128x128_K7_T2048_d%zu "
        "asm_median_ms=%.3f asm_p90_ms=%.3f asm_worst_ms=%.3f "
        "dnnl_median_ms=%.3f dnnl_p90_ms=%.3f dnnl_worst_ms=%.3f "
        "median_gain=%.2f%%\n", workers, dilation, a.median, a.p90, a.worst,
        d.median, d.p90, d.worst, gain);
    print_quality(dnnl_output, asm_output);
    std::printf("ASM_CPUS=");
    for (size_t i = 0; i < workers; i++)
        std::printf("%s%d", i ? "," : "", ds_threadpool_cpu_at(pool, i));
    std::putchar('\n');
    ds_threadpool_destroy(pool);
    return 0;
} catch (const dnnl::error &error) {
    std::fprintf(stderr, "oneDNN error: %s (status=%d)\n", error.what(),
        static_cast<int>(error.status));
    return 4;
}

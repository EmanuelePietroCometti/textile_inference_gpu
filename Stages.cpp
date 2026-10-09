#include "Stages.h"
#include "RealTimeConfig.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

    double MsSince(const std::chrono::steady_clock::time_point& t0)
    {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    }

    /**
     * @brief Releases the slot back to the pool regardless of how the scope exits.
     * @details This includes normal returns, loop breaks, and thrown exceptions.
     * It is the only safeguard against silent pool exhaustion.
     */
    class SlotGuard {
    public:
        SlotGuard(RingBuffer<PipelineSlot*>& pool, PipelineSlot* slot)
            : pool_(pool), slot_(slot) {
        }

        ~SlotGuard()
        {
            if (!slot_) return;
            if (!pool_.try_push(slot_)) {
                // Impossible by design: the slot comes from this pool, so there        
                // is enough space. If this happens, we are either shutting down        
                // (pool stopped) or there is a sizing error, and a lost slot cannot be recovered.        
                Log::Error("slot {} not returned to the pool: capacity error or shutting down", slot_->seq);
            }
            slot_ = nullptr;
        }

        SlotGuard(const SlotGuard&) = delete;
        SlotGuard& operator=(const SlotGuard&) = delete;

    private:
        RingBuffer<PipelineSlot*>& pool_;
        PipelineSlot* slot_;
    };

    /**
     * @brief Fused scale and offset per channel: out = u8 * scale + offset.
     */
    struct ChannelTransform {
        float scale[3];
        float offset[3];
        bool swapRB;
    };

    ChannelTransform MakeTransform(const ContractMetadata& contract)
    {
        ChannelTransform t{};
        t.swapRB = contract.ConvertBgrToRgb();

        if (contract.NormalizationInGraph()) {
            // The graph applies mean/std normalization internally. 
            // The host only scales from [0, 255] to [0.0, 1.0].
            for (int c = 0; c < 3; ++c) {
                t.scale[c] = 1.0f / 255.0f;
                t.offset[c] = 0.0f;
            }
        }
        else {
            // Host fallback, ImageNet standard: (u8/255 - mean) / std.
            // Fused algebraically into a single multiply-add operation per channel.
            static constexpr float mean[3] = { 0.485f, 0.456f, 0.406f };
            static constexpr float stdv[3] = { 0.229f, 0.224f, 0.225f };
            for (int c = 0; c < 3; ++c) {
                t.scale[c] = 1.0f / (255.0f * stdv[c]);
                t.offset[c] = -mean[c] / stdv[c];
            }
        }
        return t;
    }

    /**
     * @brief One row of W interleaved BGR u8 pixels -> the same row of the 3 float planes.
     *
     * @details out = u8 * scale + offset, channel order of the model (with swapRB the source
     * R lands in plane 0). `dst` points at the row inside plane 0; `plane` is the plane size
     * in floats. AVX2: 16 pixels per iteration, de-interleaved with pshufb, one FMA per
     * 8 floats; scalar tail for W % 16 != 0.
     */
    void PackRow(const std::uint8_t* row, int W, float* dst, std::size_t plane,
        const ChannelTransform& t)
    {
        const int cB = t.swapRB ? 2 : 0;   // output plane that receives the source B
        const int cR = t.swapRB ? 0 : 2;   // output plane that receives the source R
        float* pB = dst + cB * plane;
        float* pG = dst + plane;
        float* pR = dst + cR * plane;
        const float sB = t.scale[cB], oB = t.offset[cB];
        const float sG = t.scale[1], oG = t.offset[1];
        const float sR = t.scale[cR], oR = t.offset[cR];

        int x = 0;
#if defined(__AVX2__)
        const __m128i shB0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i shB1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
        const __m128i shB2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
        const __m128i shG0 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i shG1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
        const __m128i shG2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
        const __m128i shR0 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i shR1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
        const __m128i shR2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);
        const __m256 vsB = _mm256_set1_ps(sB), voB = _mm256_set1_ps(oB);
        const __m256 vsG = _mm256_set1_ps(sG), voG = _mm256_set1_ps(oG);
        const __m256 vsR = _mm256_set1_ps(sR), voR = _mm256_set1_ps(oR);

        auto store16 = [](float* d, __m128i v, __m256 s, __m256 o) {
            const __m256 lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v));
            const __m256 hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(v, 8)));
            _mm256_storeu_ps(d, _mm256_fmadd_ps(lo, s, o));
            _mm256_storeu_ps(d + 8, _mm256_fmadd_ps(hi, s, o));
        };

        for (; x + 16 <= W; x += 16) {
            const std::uint8_t* p = row + 3 * x;
            const __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
            const __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16));
            const __m128i a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 32));
            const __m128i b = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a0, shB0),
                _mm_shuffle_epi8(a1, shB1)), _mm_shuffle_epi8(a2, shB2));
            const __m128i g = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a0, shG0),
                _mm_shuffle_epi8(a1, shG1)), _mm_shuffle_epi8(a2, shG2));
            const __m128i r = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a0, shR0),
                _mm_shuffle_epi8(a1, shR1)), _mm_shuffle_epi8(a2, shR2));
            store16(pB + x, b, vsB, voB);
            store16(pG + x, g, vsG, voG);
            store16(pR + x, r, vsR, voR);
        }
#endif
        for (; x < W; ++x) {
            const std::uint8_t* p = row + 3 * x;
            pB[x] = p[0] * sB + oB;
            pG[x] = p[1] * sG + oG;
            pR[x] = p[2] * sR + oR;
        }
    }

    /**
     * @brief Native-size path: u8 BGR (HWC) patch -> f32 planar (CHW), one pass, no allocation.
     */
    void PackNchw(const std::uint8_t* src, std::size_t srcStride, int H, int W,
        float* dst, const ChannelTransform& t)
    {
        const std::size_t plane = static_cast<std::size_t>(H) * W;
        for (int y = 0; y < H; ++y)
            PackRow(src + static_cast<std::size_t>(y) * srcStride, W,
                dst + static_cast<std::size_t>(y) * W, plane, t);
    }

    constexpr int kWeightShift = 14;                 // Q14 fixed-point weights
    constexpr int kWeightOne = 1 << kWeightShift;

    inline std::uint8_t RoundShiftClipU8(int acc)
    {
        const int r = (acc + (1 << (kWeightShift - 1))) >> kWeightShift;
        return static_cast<std::uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
    }

    /**
     * @brief Triangle-filter (antialiased bilinear) coefficients in Q14, PIL-style.
     * Same algorithm and integer weights as the removed AntialiasResizer.
     */
    void TriangleCoeffs(int inSize, int outSize, std::vector<int>& bounds,
        std::vector<std::int16_t>& weights, int& ksize)
    {
        const double scale = static_cast<double>(inSize) / outSize;
        const double filterscale = scale >= 1.0 ? scale : 1.0;
        const double support = filterscale;
        const double ss = 1.0 / filterscale;

        ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
        bounds.assign(outSize, 0);
        weights.assign(static_cast<std::size_t>(outSize) * ksize, 0);
        std::vector<double> wd(static_cast<std::size_t>(ksize), 0.0);

        for (int o = 0; o < outSize; ++o) {
            const double center = (o + 0.5) * scale;
            int xmin = static_cast<int>(center - support + 0.5);
            if (xmin < 0) xmin = 0;
            int xmax = static_cast<int>(center + support + 0.5);
            if (xmax > inSize) xmax = inSize;
            int n = xmax - xmin;
            if (n > ksize) n = ksize;

            double total = 0.0;
            for (int t = 0; t < n; ++t) {
                double w = 1.0 - std::abs((xmin + t - center + 0.5) * ss);
                if (w < 0.0) w = 0.0;
                wd[t] = w;
                total += w;
            }
            if (total > 0.0)
                for (int t = 0; t < n; ++t) wd[t] /= total;

            // Q14 with sum correction on the dominant weight (sum must be exactly 1.0).
            std::int16_t* row = &weights[static_cast<std::size_t>(o) * ksize];
            int isum = 0, imax = 0;
            for (int t = 0; t < n; ++t) {
                const int wi = static_cast<int>(std::lround(wd[t] * kWeightOne));
                row[t] = static_cast<std::int16_t>(wi);
                isum += wi;
                if (wd[t] > wd[imax]) imax = t;
            }
            if (n > 0)
                row[imax] = static_cast<std::int16_t>(row[imax] + (kWeightOne - isum));
            bounds[o] = xmin;
        }
    }

    /**
     * @brief Exact 2x downscale of a u8 BGR patch fused with the planar packing.
     *
     * @details One instance per prep thread: all scratch is allocated in the constructor,
     * Run() never allocates.
     * - antialias = true: separable triangle filter (PIL / torchvision antialias=True),
     *   horizontal pass then vertical pass with u8 rounding after each. Bit-exact with the
     *   removed AntialiasResizer.
     * - antialias = false: 2x2 average (a+b+c+d+2)>>2, i.e. what cv::resize(INTER_LINEAR)
     *   does at exactly 2x (OpenCV switches to INTER_AREA): the behaviour of f2545aa.
     */
    class Downscale2x {
    public:
        Downscale2x(int inW, int inH, bool antialias)
            : inW_(inW), inH_(inH), outW_(inW / 2), outH_(inH / 2), antialias_(antialias),
            row_(static_cast<std::size_t>(outW_) * 3)
        {
            if (antialias_) {
                TriangleCoeffs(inW_, outW_, hBounds_, hWeights_, hK_);
                TriangleCoeffs(inH_, outH_, vBounds_, vWeights_, vK_);
                hpass_.assign(static_cast<std::size_t>(inH_) * outW_ * 3, 0);
                acc_.assign(static_cast<std::size_t>(outW_) * 3, 0);
                hFast_ = FastMask(hWeights_, hK_, outW_);
                vFast_ = FastMask(vWeights_, vK_, outH_);
            }
        }

        /// src: inH x inW BGR u8 with row stride in bytes; dst: [3][inH/2][inW/2] floats.
        void Run(const std::uint8_t* src, std::size_t stride, float* dst, const ChannelTransform& t)
        {
            const std::size_t plane = static_cast<std::size_t>(outH_) * outW_;
            if (antialias_) RunAntialias(src, stride, dst, plane, t);
            else            RunBox(src, stride, dst, plane, t);
        }

    private:
        /// Outputs whose taps are exactly [2048, 6144, 6144, 2048] (= [1,3,3,1]/8 in Q14),
        /// i.e. every output except the borders at exactly 2x. For those, sum*w + 8192 >> 14
        /// equals (p0 + 3*p1 + 3*p2 + p3 + 4) >> 3 bit for bit, with no multiplications.
        static std::vector<std::uint8_t> FastMask(const std::vector<std::int16_t>& w, int k, int n)
        {
            std::vector<std::uint8_t> m(static_cast<std::size_t>(n), 0);
            for (int o = 0; o < n; ++o) {
                const std::int16_t* r = &w[static_cast<std::size_t>(o) * k];
                bool fast = k >= 4 && r[0] == 2048 && r[1] == 6144 && r[2] == 6144 && r[3] == 2048;
                for (int j = 4; fast && j < k; ++j) fast = (r[j] == 0);
                m[o] = fast ? 1 : 0;
            }
            return m;
        }

        void RunBox(const std::uint8_t* src, std::size_t stride, float* dst,
            std::size_t plane, const ChannelTransform& t)
        {
            for (int y = 0; y < outH_; ++y) {
                const std::uint8_t* r0 = src + static_cast<std::size_t>(2 * y) * stride;
                const std::uint8_t* r1 = r0 + stride;
                std::uint8_t* o = row_.data();
                for (int x = 0; x < outW_; ++x, r0 += 6, r1 += 6, o += 3) {
                    o[0] = static_cast<std::uint8_t>((r0[0] + r0[3] + r1[0] + r1[3] + 2) >> 2);
                    o[1] = static_cast<std::uint8_t>((r0[1] + r0[4] + r1[1] + r1[4] + 2) >> 2);
                    o[2] = static_cast<std::uint8_t>((r0[2] + r0[5] + r1[2] + r1[5] + 2) >> 2);
                }
                PackRow(row_.data(), outW_, dst + static_cast<std::size_t>(y) * outW_, plane, t);
            }
        }

        void RunAntialias(const std::uint8_t* src, std::size_t stride, float* dst,
            std::size_t plane, const ChannelTransform& t)
        {
            const std::size_t hStride = static_cast<std::size_t>(outW_) * 3;

            // Horizontal pass: [inH x inW] -> [inH x outW], rounded to u8.
            for (int y = 0; y < inH_; ++y) {
                const std::uint8_t* srow = src + static_cast<std::size_t>(y) * stride;
                std::uint8_t* hrow = hpass_.data() + static_cast<std::size_t>(y) * hStride;
                for (int o = 0; o < outW_; ++o) {
                    const int s = hBounds_[o];
                    if (hFast_[o]) {
                        const std::uint8_t* p = srow + static_cast<std::size_t>(s) * 3;
                        std::uint8_t* op = hrow + static_cast<std::size_t>(o) * 3;
                        op[0] = static_cast<std::uint8_t>((p[0] + p[9] + 3 * (p[3] + p[6]) + 4) >> 3);
                        op[1] = static_cast<std::uint8_t>((p[1] + p[10] + 3 * (p[4] + p[7]) + 4) >> 3);
                        op[2] = static_cast<std::uint8_t>((p[2] + p[11] + 3 * (p[5] + p[8]) + 4) >> 3);
                        continue;
                    }
                    const int avail = (std::min)(hK_, inW_ - s);
                    const std::int16_t* w = &hWeights_[static_cast<std::size_t>(o) * hK_];
                    int a0 = 0, a1 = 0, a2 = 0;
                    const std::uint8_t* px = srow + static_cast<std::size_t>(s) * 3;
                    for (int k = 0; k < avail; ++k, px += 3) {
                        const int wt = w[k];
                        a0 += wt * px[0]; a1 += wt * px[1]; a2 += wt * px[2];
                    }
                    std::uint8_t* op = hrow + static_cast<std::size_t>(o) * 3;
                    op[0] = RoundShiftClipU8(a0);
                    op[1] = RoundShiftClipU8(a1);
                    op[2] = RoundShiftClipU8(a2);
                }
            }

            // Vertical pass, one output row at a time, packed straight into the planes.
            const int n3 = outW_ * 3;
            int* acc = acc_.data();
            for (int o = 0; o < outH_; ++o) {
                const int s = vBounds_[o];
                const int avail = (std::min)(vK_, inH_ - s);
                const std::int16_t* w = &vWeights_[static_cast<std::size_t>(o) * vK_];

                const std::uint8_t* first = hpass_.data() + static_cast<std::size_t>(s) * hStride;
                if (vFast_[o]) {
                    const std::uint8_t* r1 = first + hStride;
                    const std::uint8_t* r2 = r1 + hStride;
                    const std::uint8_t* r3 = r2 + hStride;
                    std::uint8_t* out = row_.data();
                    for (int i = 0; i < n3; ++i)   // contiguous: auto-vectorized
                        out[i] = static_cast<std::uint8_t>((first[i] + r3[i] + 3 * (r1[i] + r2[i]) + 4) >> 3);
                    PackRow(row_.data(), outW_, dst + static_cast<std::size_t>(o) * outW_, plane, t);
                    continue;
                }
                const int w0 = w[0];
                for (int i = 0; i < n3; ++i) acc[i] = w0 * first[i];
                for (int k = 1; k < avail; ++k) {
                    const int wt = w[k];
                    if (wt == 0) continue;
                    const std::uint8_t* r = hpass_.data() + static_cast<std::size_t>(s + k) * hStride;
                    for (int i = 0; i < n3; ++i) acc[i] += wt * r[i];
                }
                for (int i = 0; i < n3; ++i) row_[i] = RoundShiftClipU8(acc[i]);

                PackRow(row_.data(), outW_, dst + static_cast<std::size_t>(o) * outW_, plane, t);
            }
        }

        int inW_, inH_, outW_, outH_;
        bool antialias_;
        int hK_ = 0, vK_ = 0;
        std::vector<int> hBounds_, vBounds_;
        std::vector<std::int16_t> hWeights_, vWeights_;
        std::vector<std::uint8_t> hFast_, vFast_;   ///< per-output flag: taps are [1,3,3,1]/8
        std::vector<std::uint8_t> hpass_;   ///< [inH][outW*3] horizontal-pass scratch (antialias)
        std::vector<int> acc_;              ///< [outW*3] vertical accumulator (antialias)
        std::vector<std::uint8_t> row_;     ///< [outW*3] one output row, u8 BGR
    };

} // namespace


void PrepStage(const AppConfig& cfg,
    const OrtSessionConfig& session,
    RingBuffer<RawFrame*>& qRaw,
    RingBuffer<RawFrame*>& rawPool,
    RingBuffer<PipelineSlot*>& slotPool,
    RingBuffer<PipelineSlot*>& qPrep,
    PerformanceMetrics& metrics)
{
    if (const DWORD err = RT::ConfigureThread(cfg.PrepPriority()); err != 0)
        Log::Warning("prep: thread priority {} not applied (GetLastError={})", cfg.PrepPriority(), err);

    const ChannelTransform transform = MakeTransform(session.Contract());

    // main() guarantees model == patch (native) or model == patch / 2 (fused 2x downscale).
    const int H = session.ModelHeight();
    const int W = session.ModelWidth();
    const std::size_t imgElems = static_cast<std::size_t>(3) * H * W;
    const int batch = session.BatchSize();

    // Per-thread downscaler, all scratch allocated here, once.
    std::optional<Downscale2x> down;
    if (H != static_cast<int>(cfg.Geometry().patchHeight))
        down.emplace(static_cast<int>(cfg.Geometry().stripWidth),
            static_cast<int>(cfg.Geometry().patchHeight), cfg.ResizeAntialias());

    for (;;) {
        RawFrame* raw = nullptr;
        if (!qRaw.pop(raw)) break; // Queue stopped

        PipelineSlot* slot = nullptr;
        if (!slotPool.pop(slot)) {
            (void)rawPool.try_push(raw);
            break;
        }

        const auto t0 = std::chrono::steady_clock::now();
        const int n = (std::min)(batch, static_cast<int>(raw->patches.size()));

        // Straight into the pinned slot: no temporaries, no allocation.
        for (int i = 0; i < n; ++i) {
            const cv::Mat& p = raw->patches[i];
            float* out = slot->input + static_cast<std::size_t>(i) * imgElems;
            if (down) down->Run(p.data, p.step[0], out, transform);
            else      PackNchw(p.data, p.step[0], H, W, out, transform);
        }

        slot->seq = raw->seq;
        slot->acquiredQPC = raw->acquiredQPC;
        slot->prepMs = MsSince(t0);

        (void)rawPool.try_push(raw);
        metrics.addPreprocessingTime(slot->prepMs);

        if (!qPrep.push(slot)) {
            (void)slotPool.try_push(slot);
            break;
        }
    }
}


void InferStage(OrtSessionConfig& session,
    RingBuffer<PipelineSlot*>& qPrep,
    RingBuffer<PipelineSlot*>& qInf,
    RingBuffer<PipelineSlot*>& slotPool,
    PerformanceMetrics& metrics)
{
    for (;;) {
        PipelineSlot* slot = nullptr;
        if (!qPrep.pop(slot)) break;

        OrtSessionConfig::Timings t{};
        try {
            session.RunBatch(slot->input, slot->scores, slot->rawMap, t);
        }
        catch (const std::exception& e) {
            Log::Error("Inference failed on batch {}: {}", slot->seq, e.what());
            (void)slotPool.try_push(slot); // Prevent slot leak on failure
            continue;
        }

        slot->h2dMs = t.h2dMs;
        slot->runMs = t.runMs;
        slot->d2hMs = t.d2hMs;

        metrics.addH2DTime(t.h2dMs);
        metrics.addRunTime(t.runMs);
        metrics.addD2HTime(t.d2hMs);
        metrics.addGpuTime(t.h2dMs + t.runMs + t.d2hMs);

        if (!qInf.push(slot)) { // blocking, same reasoning as in PrepStage
            (void)slotPool.try_push(slot);
            break;
        }
    }
}


void PostStage(const AppConfig& cfg,
    const OrtSessionConfig& session,
    RingBuffer<PipelineSlot*>& qInf,
    RingBuffer<PipelineSlot*>& slotPool,
    IResultSink& sink,
    PerformanceMetrics& metrics)
{
    if (const DWORD err = RT::ConfigureThread(cfg.PostPriority()); err != 0)
        Log::Warning("post: thread priority {} not applied (GetLastError={})", cfg.PostPriority(), err);

    const ContractMetadata& contract = session.Contract();
    const int batch = session.BatchSize();
    const int mapH = session.MapHeight();
    const int mapW = session.MapWidth();
    const std::size_t rawElems = static_cast<std::size_t>(mapH) * mapW;

    const int outH = cfg.MapAtModelResolution() ? mapH : static_cast<int>(cfg.Geometry().patchHeight);
    const int outW = cfg.MapAtModelResolution() ? mapW : static_cast<int>(cfg.Geometry().stripWidth);
    const std::size_t outElems = static_cast<std::size_t>(outH) * outW;

    // Precalculated Min-Max normalization bounds based on model contract.
    // Maps the raw logit outputs from [mapMin, mapMax] to [0, 255] uint8_t:
    // u8 = raw * alpha + beta = (raw - mapMin) * alpha.
    const float span = contract.MapMax() - contract.MapMin();
    const double alpha = 255.0 / span;
    const double beta = -255.0 * contract.MapMin() / span;

    // Loop invariants, hoisted.
    const bool atModelRes = cfg.MapAtModelResolution();
    const float scoreThr = contract.ScoreThreshold();
    const double pixThrRaw = contract.PixelThreshold();
    // Pixel threshold in the normalized 8-bit domain: (thr - mapMin) * alpha.
    // (The previous "(thr - mapMin) * alpha + beta" subtracted mapMin twice.)
    const double pixThr8 = (pixThrRaw - contract.MapMin()) * alpha;

    // Scratch only for the resize path: at model resolution convertTo writes straight
    // into the slot. Allocated once, before the loop.
    cv::Mat scaled;
    if (!atModelRes) scaled.create(mapH, mapW, CV_8UC1);

    for (;;) {
        PipelineSlot* slot = nullptr;
        if (!qInf.pop(slot)) break;

        SlotGuard guard(slotPool, slot); // Guaranteed release back to slotPool
        const auto t0 = std::chrono::steady_clock::now();

        int anomalies = 0;
        for (int b = 0; b < batch; ++b) {
            if (slot->scores[b] >= scoreThr) {
                ++anomalies;
            }

            // Zero-copy wraps: raw logits in the D2H pinned buffer, output in the slot.
            const cv::Mat raw(mapH, mapW, CV_32FC1, slot->rawMap + b * rawElems);
            cv::Mat dst(outH, outW, CV_8UC1, slot->outMap + b * outElems);

            if (atModelRes) {
                // dst already has the target size and type: create() is a no-op, no copy.
                raw.convertTo(dst, CV_8UC1, alpha, beta);
            }
            else {
                raw.convertTo(scaled, CV_8UC1, alpha, beta);
                cv::resize(scaled, dst, dst.size(), 0, 0, cv::INTER_LINEAR);
            }

            if (slot->outMask) {
                cv::Mat mask(outH, outW, CV_8UC1, slot->outMask + b * outElems);
                if (atModelRes)
                    cv::compare(raw, pixThrRaw, mask, cv::CMP_GE); // exact, on the raw logits
                else
                    cv::compare(dst, pixThr8, mask, cv::CMP_GE);   // on the resized 8-bit map
            }
        }

        BatchResult result;
        result.seq = slot->seq;
        result.acquiredQPC = slot->acquiredQPC;
        result.batch = batch;
        result.scores = slot->scores;
        result.map = slot->outMap;
        result.mask = slot->outMask;
        result.mapElemsPerImage = outElems;
        result.anomalies = anomalies;

        sink.Publish(result); // Raw pointers inside `result` are only valid during this call

        slot->postMs = MsSince(t0);
		metrics.printRollingAverage(metrics.addPostprocessingTime(slot->postMs));
    }
}
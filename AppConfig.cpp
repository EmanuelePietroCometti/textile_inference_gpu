#include "AppConfig.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include <stdexcept>
#include <thread>

#include "SharedFrameContract.h"

namespace {

    std::uint32_t U32(const IniConfig& ini, const wchar_t* section, const wchar_t* key,
        long fallback, long minValue, long maxValue, const char* label)
    {
        const long v = ini.GetInt(section, key, fallback);
        if (v < minValue || v > maxValue) {
            throw std::runtime_error(std::string(label) + " = " + std::to_string(v)
                + ": expected between " + std::to_string(minValue) + " and " + std::to_string(maxValue));
        }
        return static_cast<std::uint32_t>(v);
    }

    bool Flag(const IniConfig& ini, const wchar_t* section, const wchar_t* key, long fallback)
    {
        return ini.GetInt(section, key, fallback) != 0;
    }

} // namespace

AppConfig::AppConfig(PipelineConfig model, PatchGeometry geometry)
    : model_(std::move(model)), geometry_(geometry)
{
}

std::uint32_t AppConfig::MinimumSlots() const
{
    return prepThreads_ + qPrepCap_ + inferThreads_ + qInfCap_ + postThreads_;
}

AppConfig AppConfig::LoadFromIni(const std::wstring& iniPath)
{
    IniConfig ini;
    if (!ini.Load(iniPath))
        throw std::runtime_error("Failed to load INI configuration file.");

    PatchGeometry g;
    g.stripWidth = U32(ini, L"Source", L"StripWidth", 512, 1, 65535, "[Source] StripWidth");
    g.stripHeight = U32(ini, L"Source", L"StripHeight", 8192, 1, 1 << 20, "[Source] StripHeight");
    g.channels = U32(ini, L"Source", L"Channels", 3, 1, 3, "[Source] Channels");
    g.patchHeight = U32(ini, L"Patches", L"PatchHeight", 512, 1, 65535, "[Patches] PatchHeight");
    g.overlap = U32(ini, L"Patches", L"Overlap", 32, 0, 65534, "[Patches] Overlap");
    g.count = U32(ini, L"Patches", L"Count", 17, 1, 4096, "[Patches] Count");

    AppConfig cfg(PipelineConfig::LoadFromIni(ini), g);

    cfg.rawFrames_ = U32(ini, L"Pipeline", L"RawFrames", 3, 1, 256, "[Pipeline] RawFrames");
    cfg.qRawCap_ = U32(ini, L"Pipeline", L"QueueRawCapacity", 4, 1, 1024, "[Pipeline] QueueRawCapacity");
    cfg.qPrepCap_ = U32(ini, L"Pipeline", L"QueuePrepCapacity", 4, 1, 1024, "[Pipeline] QueuePrepCapacity");
    cfg.qInfCap_ = U32(ini, L"Pipeline", L"QueueInfCapacity", 4, 1, 1024, "[Pipeline] QueueInfCapacity");
    cfg.prepThreads_ = U32(ini, L"Pipeline", L"PrepThreads", 1, 1, 64, "[Pipeline] PrepThreads");
    cfg.inferThreads_ = U32(ini, L"Pipeline", L"InferenceThreads", 1, 1, 4, "[Pipeline] InferenceThreads");
    cfg.postThreads_ = U32(ini, L"Pipeline", L"PostThreads", 1, 1, 64, "[Pipeline] PostThreads");
    cfg.openCvThreads_ = U32(ini, L"Pipeline", L"OpenCvThreads", 1, 1, 64, "[Pipeline] OpenCvThreads");
    cfg.readTimeoutMs_ = U32(ini, L"Pipeline", L"ReadTimeoutMs", 100, 1, 60000, "[Pipeline] ReadTimeoutMs");

    // 0 = computed from minimum: this is the recommended default.
    // The slot constraint depends on thread counts and queue capacities. 
    // Setting it manually incorrectly would silently serialize the pipeline without throwing visible errors.
    const std::uint32_t slots = U32(ini, L"Pipeline", L"Slots", 0, 0, 1024, "[Pipeline] Slots");
    cfg.slots_ = (slots == 0) ? cfg.MinimumSlots() : slots;

    cfg.mapAtModelRes_ = Flag(ini, L"Output", L"MapAtModelResolution", 1);
    cfg.drawMask_ = Flag(ini, L"Output", L"DrawMask", 0);

    cfg.elevateProcess_ = Flag(ini, L"RealTime", L"ElevateProcess", 1);
    cfg.prepRealtime_ = Flag(ini, L"RealTime", L"PrepRealtime", 1);
    cfg.inferRealtime_ = Flag(ini, L"RealTime", L"InferenceRealtime", 1);
    cfg.postRealtime_ = Flag(ini, L"RealTime", L"PostRealtime", 0);

    cfg.logMaxChars_ = U32(ini, L"Logger", L"maxMessageChars", 480, 32, 8192, "[Logger] maxMessageChars");
    cfg.logQueueCap_ = U32(ini, L"Logger", L"queueCapacity", 4096, 8, 1 << 20, "[Logger] queueCapacity");
    cfg.logFlushMs_ = U32(ini, L"Logger", L"flushIntervalMs", 200, 0, 60000, "[Logger] flushIntervalMs");
    cfg.logNotify_ = U32(ini, L"Logger", L"notifyThreshold", 1024, 0, 1 << 20, "[Logger] notifyThreshold");

    cfg.metricsWindow_ = U32(ini, L"Metrics", L"WindowDimension", 100, 1, 100000, "[Metrics] WindowDimension");
    cfg.metricsPrintMs_ = U32(ini, L"Metrics", L"PrintEveryMs", 2000, 0, 600000, "[Metrics] PrintEveryMs");

    cfg.ValidateCrossConstraints();
    return cfg;
}

void AppConfig::ValidateCrossConstraints() const
{
    if (geometry_.channels != 3) {
        throw std::runtime_error("[Source] Channels = " + std::to_string(geometry_.channels)
            + ": AntialiasResizer and preprocessing stages are strictly hardcoded for 3 channels.");
    }
    // PatchLayout validates its own internal consistency during construction.
    // What remains here are cross-module constraints that no single module can evaluate alone.
    const PatchLayout layout(geometry_);

    if (layout.PayloadBytes() != sfc::FULL_PAYLOAD) {
        throw std::runtime_error(
            "INI geometry is incompatible with the compiled contract: INI dictates "
            + std::to_string(layout.PayloadBytes()) + " bytes, but SharedFrameContract.h expects "
            + std::to_string(sfc::FULL_PAYLOAD) + " bytes.");
    }

    if (qRawCap_ < rawFrames_) {
        throw std::runtime_error("[Pipeline] QueueRawCapacity ("
            + std::to_string(qRawCap_) + ") must be >= RawFrames ("
            + std::to_string(rawFrames_) + "): with this capacity, the reader's push "
            "cannot fail, leaving the slot pool as the sole backpressure bottleneck.");
    }

    if (rawFrames_ < prepThreads_ + 1) {
        throw std::runtime_error("[Pipeline] RawFrames (" + std::to_string(rawFrames_)
            + ") must be >= PrepThreads + 1 to prevent starvation.");
    }

    if (slots_ < MinimumSlots()) {
        throw std::runtime_error("[Pipeline] Slots (" + std::to_string(slots_)
            + ") is below the theoretical minimum of " + std::to_string(MinimumSlots())
            + " (PrepThreads + QueuePrepCapacity + InferenceThreads + QueueInfCapacity + PostThreads). "
            "Below this threshold, the pipeline will silently serialize and stall without visible errors.");
    }

    // Every prep and post thread executes a cv::parallel_for_ internally.
    // Therefore, the required thread counts MULTIPLY, they don't just add up.
    // This is the most common cause of OS context-switch jitter in this architecture.
    const unsigned logical = std::thread::hardware_concurrency();
    const std::uint32_t demanded = (prepThreads_ + postThreads_) * openCvThreads_ + inferThreads_ + 1;

    if (logical > 0 && demanded > logical) {
        throw std::runtime_error("(PrepThreads + PostThreads) * OpenCvThreads + InferenceThreads + Ingest = "
            + std::to_string(demanded) + " workers demanded on only " + std::to_string(logical)
            + " logical processors: reduce OpenCvThreads or the stage threads to avoid CPU thrashing.");
    }
}

std::string AppConfig::Describe() const
{
    return "Config | " + model_.Describe() + "\n"
        + "Config | strip " + std::to_string(geometry_.stripWidth) + "x"
        + std::to_string(geometry_.stripHeight) + "x" + std::to_string(geometry_.channels)
        + " | " + std::to_string(geometry_.count) + " patches of "
        + std::to_string(geometry_.patchHeight) + " stride "
        + std::to_string(geometry_.patchHeight - geometry_.overlap) + "\n"
        + "Config | rawFrames=" + std::to_string(rawFrames_)
        + " slots=" + std::to_string(slots_) + " (min " + std::to_string(MinimumSlots()) + ")"
        + " queues=" + std::to_string(qRawCap_) + "/" + std::to_string(qPrepCap_) + "/" + std::to_string(qInfCap_)
        + " threads=" + std::to_string(prepThreads_) + "/" + std::to_string(inferThreads_)
        + "/" + std::to_string(postThreads_) + " opencv=" + std::to_string(openCvThreads_) + "\n"
        + "Config | map=" + (mapAtModelRes_ ? "model" : "patch")
        + " mask=" + (drawMask_ ? "1" : "0")
        + " | realtime proc=" + (elevateProcess_ ? "1" : "0")
        + " prep=" + (prepRealtime_ ? "1" : "0")
        + " inf=" + (inferRealtime_ ? "1" : "0")
        + " post=" + (postRealtime_ ? "1" : "0");
}
#include "PipelineConfig.h"
#include "AsyncLogger.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <stdexcept>

namespace {

    std::string Narrow(const std::wstring& w)
    {
        if (w.empty()) return {};
        const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(),
            nullptr, 0, nullptr, nullptr);
        std::string s(n, '\0');
        WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
        return s;
    }

    const char* ProviderName(ExecutionProvider p)
    {
        switch (p) {
        case ExecutionProvider::CPU:      return "CPU";
        case ExecutionProvider::CUDA:     return "CUDA";
        case ExecutionProvider::TensorRT: return "TensorRT";
        }
        return "<unknown>";
    }

} // namespace

PipelineConfig::PipelineConfig(std::wstring modelPath, int batchSize, bool loopBatch1,
    ExecutionProvider provider, int deviceId, bool enableFp16,
    std::size_t trtWorkspaceSize, std::string engineCachePath,
    std::string timingCachePath, int builderOptLevel, int warmupRuns)
    : modelPath_(std::move(modelPath)),
    batchSize_(batchSize),
    loopBatch1_(loopBatch1),
    provider_(provider),
    deviceId_(deviceId),
    enableFp16_(enableFp16),
    trtWorkspaceSize_(trtWorkspaceSize),
    engineCachePath_(std::move(engineCachePath)),
    timingCachePath_(std::move(timingCachePath)),
    builderOptLevel_(builderOptLevel),
    warmupRuns_(warmupRuns)
{
}

PipelineConfig PipelineConfig::LoadFromIni(const IniConfig& ini)
{
    // Extract [Model] values
    std::wstring path = ini.GetString(L"Model", L"Path", L"");
    // GetString DOES NOT remove quotes (your test documents this): a
    // Path="C:\m.onnx" arrives with quotes inside and ORT fails with a
    // message that doesn't mention them.
    if (path.size() >= 2 && path.front() == L'"' && path.back() == L'"')
        path = path.substr(1, path.size() - 2);
    int batch = static_cast<int>(ini.GetInt(L"Model", L"BatchSize", 1));
    bool loop1 = (ini.GetInt(L"Model", L"LoopBatch1", 0) != 0);

    // Extract [Hardware] values
    const long provRaw = ini.GetInt(L"Hardware", L"Provider", 1);
    if (provRaw < 0 || provRaw > 2)
        throw std::runtime_error("[Hardware] Provider = " + std::to_string(provRaw)
            + ": expected 0 (CPU), 1 (CUDA) or 2 (TensorRT)");
    const auto prov = static_cast<ExecutionProvider>(provRaw);

    int devId = static_cast<int>(ini.GetInt(L"Hardware", L"DeviceId", 0));
    bool fp16 = (ini.GetInt(L"Hardware", L"EnableFP16", 1) != 0);

    long trtWorkspaceMB = ini.GetInt(L"Hardware", L"TrtWorkspaceMB", 2048);
    if (trtWorkspaceMB < 0)
        throw std::runtime_error("[Hardware] TrtWorkspaceMB = " + std::to_string(trtWorkspaceMB) + ": must be >= 0");
    std::size_t trtWorkspaceBytes = static_cast<std::size_t>(trtWorkspaceMB) * 1024ULL * 1024ULL;

    // Instantiate and return via the private constructor
    // TRT options require const char*: conversion here, once.
    auto narrow = [](const std::wstring& w) {
        if (w.empty()) return std::string{};
        const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
        std::string s(n, '\0');
        WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
        return s;
        };

    PipelineConfig cfg(std::move(path), batch, loop1, prov, devId, fp16, trtWorkspaceBytes,
        narrow(ini.GetString(L"Hardware", L"TrtEngineCachePath", L"trt_engine_cache")),
        narrow(ini.GetString(L"Hardware", L"TrtTimingCachePath", L"trt_timing_cache")),
        static_cast<int>(ini.GetInt(L"Hardware", L"TrtBuilderOptLevel", 3)),
        static_cast<int>(ini.GetInt(L"Model", L"WarmupRuns", 10)));

    std::string error;
    if (!cfg.Validate(error))
        throw std::runtime_error("invalid configuration: " + error);

    Log::Info("{}", cfg.Describe());
    return cfg;
}

bool PipelineConfig::Validate(std::string& error) const
{
    error.clear();
    if (modelPath_.empty()) { error = "[Model] Path is empty"; return false; }
    if (batchSize_ <= 0) { error = "[Model] BatchSize = " + std::to_string(batchSize_); return false; }
    if (deviceId_ < 0) { error = "[Hardware] DeviceId = " + std::to_string(deviceId_); return false; }
    if (builderOptLevel_ < 0 || builderOptLevel_ > 5)
    {
        error = "[Hardware] TrtBuilderOptLevel = " + std::to_string(builderOptLevel_) + ", expected 0-5"; return false;
    }
    if (provider_ == ExecutionProvider::TensorRT && trtWorkspaceSize_ == 0)
    {
        error = "[Hardware] TrtWorkspaceMB = 0 with Provider = TensorRT"; return false;
    }
    return true;
}

std::string PipelineConfig::Describe() const
{
    return "Config | model=" + Narrow(modelPath_)
        + " | batch=" + std::to_string(batchSize_)
        + " | loopBatch1=" + (loopBatch1_ ? "1" : "0")
        + " | provider=" + ProviderName(provider_)
        + " | device=" + std::to_string(deviceId_)
        + " | fp16=" + (enableFp16_ ? "1" : "0")
        + " | trtWorkspace=" + std::to_string(trtWorkspaceSize_ / (1024 * 1024)) + " MiB"
        + " | trtOptLevel=" + std::to_string(builderOptLevel_)
        + " | engineCache=" + engineCachePath_
        + " | timingCache=" + timingCachePath_
        + " | warmup=" + std::to_string(warmupRuns_);
}

const std::wstring& PipelineConfig::ModelPath() const { return modelPath_; }
int PipelineConfig::BatchSize() const { return batchSize_; }
bool PipelineConfig::LoopBatch1() const { return loopBatch1_; }
ExecutionProvider PipelineConfig::Provider() const { return provider_; }
int PipelineConfig::DeviceId() const { return deviceId_; }
bool PipelineConfig::EnableFp16() const { return enableFp16_; }
std::size_t PipelineConfig::TrtWorkspaceSize() const { return trtWorkspaceSize_; }
const std::string& PipelineConfig::TrtEngineCachePath() const { return engineCachePath_; }
const std::string& PipelineConfig::TrtTimingCachePath() const { return timingCachePath_; }
int PipelineConfig::TrtBuilderOptLevel() const { return builderOptLevel_; }
int PipelineConfig::WarmupRuns() const { return warmupRuns_; }
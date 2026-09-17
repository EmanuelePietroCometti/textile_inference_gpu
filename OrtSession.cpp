#include "OrtSession.h"
#include "AsyncLogger.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>

namespace {

	void CudaCheck(cudaError_t err, const char* what)
	{
		if (err != cudaSuccess)
			throw std::runtime_error(std::string(what) + " failed: "
				+ cudaGetErrorString(err));
	}

	/// Destructor for TensorRT V2 options: they are allocated by ORT and must
	/// be released even if AppendExecutionProvider throws.
	struct TrtOptionsDeleter {
		void operator()(OrtTensorRTProviderOptionsV2* p) const {
			if (p) Ort::GetApi().ReleaseTensorRTProviderOptions(p);
		}
	};
	using TrtOptions = std::unique_ptr<OrtTensorRTProviderOptionsV2, TrtOptionsDeleter>;

	void AppendTensorRt(Ort::SessionOptions& so, const PipelineConfig& cfg, cudaStream_t stream)
	{
		OrtTensorRTProviderOptionsV2* raw = nullptr;
		Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&raw));
		TrtOptions options(raw);

		const std::string deviceId = std::to_string(cfg.DeviceId());
		const std::string workspace = std::to_string(cfg.TrtWorkspaceSize());
		const std::string optLevel = std::to_string(cfg.TrtBuilderOptLevel());

		// Values must survive the Update call: keeping them in local variables
		// and passing their .c_str() is the only safe way.
		const char* keys[] = {
			"device_id",
			"trt_max_workspace_size",
			"trt_fp16_enable",
			"trt_engine_cache_enable",
			"trt_engine_cache_path",
			"trt_timing_cache_enable",
			"trt_timing_cache_path",
			"trt_builder_optimization_level",
			"has_user_compute_stream",
		};
		const char* values[] = {
			deviceId.c_str(),
			workspace.c_str(),
			cfg.EnableFp16() ? "1" : "0",
			"1",
			cfg.TrtEngineCachePath().c_str(),
			"1",
			cfg.TrtTimingCachePath().c_str(),
			optLevel.c_str(),
			"1",
		};
		constexpr std::size_t count = sizeof(keys) / sizeof(keys[0]);
		static_assert(count == sizeof(values) / sizeof(values[0]), "keys and values mismatch");

		std::error_code ec;
		std::filesystem::create_directories(cfg.TrtEngineCachePath(), ec);
		std::filesystem::create_directories(cfg.TrtTimingCachePath(), ec);

		if (ec) Log::Warning("TensorRT cache cannot be created: {}", ec.message());

		Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(
			options.get(), keys, values, count));

		// The stream is a pointer, not a string: the dedicated API is required.
		Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptionsWithValue(
			options.get(), "user_compute_stream", stream));

		so.AppendExecutionProvider_TensorRT_V2(*options);
	}

	void AppendCuda(Ort::SessionOptions& so, const PipelineConfig& cfg, cudaStream_t stream)
	{
		OrtCUDAProviderOptions cuda{};
		cuda.device_id = cfg.DeviceId();
		cuda.has_user_compute_stream = 1;
		cuda.user_compute_stream = stream;
		cuda.do_copy_in_default_stream = 0;
		so.AppendExecutionProvider_CUDA(cuda);
	}

	void ConfigureSessionOptions(Ort::SessionOptions& so, const PipelineConfig& cfg,
		cudaStream_t stream)
	{
		so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		// On the GPU path, CPU work is marginal: more intra-op threads
		// only add contention with the prep and post stages.
		so.SetIntraOpNumThreads(1);
		so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

		if (cfg.Provider() == ExecutionProvider::TensorRT)
			AppendTensorRt(so, cfg, stream);

		// CUDA must be appended below TensorRT anyway: it absorbs the nodes that TRT
		// doesn't take, and without it those nodes would fallback to the CPU with a
		// data transfer back and forth at each batch.
		AppendCuda(so, cfg, stream);
	}

	Ort::Session MakeSession(Ort::Env& env, const PipelineConfig& cfg, cudaStream_t stream)
	{
		Ort::SessionOptions so;
		ConfigureSessionOptions(so, cfg, stream);
		return Ort::Session(env, cfg.ModelPath().c_str(), so);
	}

} // namespace


OrtSessionConfig::OrtSessionConfig(Ort::Env& env, const PipelineConfig& cfg)
	: session_(nullptr),
	binding_(nullptr),
	deviceMemInfo_("Cuda", OrtDeviceAllocator, cfg.DeviceId(), OrtMemTypeDefault),
	loopBatch1_(cfg.LoopBatch1())
{
	// This class allocates device buffers and binds tensors to CUDA memory:
	// it does not have a CPU path. Better to state it immediately than fail later
	// on a cudaMalloc without context.
	if (cfg.Provider() == ExecutionProvider::CPU)
		throw std::runtime_error("OrtSessionConfig requires CUDA or TensorRT: "
			"[Hardware] Provider = 0 is not supported by this class.");

	try {
		// The stream is created BEFORE the session: the provider options
		// must receive it as user_compute_stream, otherwise ORT creates its own
		// and our async memcpys end up on a different stream than
		// Run. The result would be correct only by chance.
		CudaCheck(cudaSetDevice(cfg.DeviceId()), "cudaSetDevice");
		CudaCheck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");

		session_ = MakeSession(env, cfg, stream_);
		binding_ = Ort::IoBinding(session_);

		// First: if the calibration metadata is incomplete, it is better
		// to catch it before allocating VRAM.
		contract_.Load(session_);

		ReadModelGeometry(cfg);
		CreateEvents();
		AllocateDeviceBuffers();
		BuildTensors();

		Log::Info("ORT Session ready | batch {} | input {}x{}x{} | map {}x{} | loopBatch1={} | VRAM ~{:.1f} MiB",
			batch_, modelC_, modelH_, modelW_, mapH_, mapW_, loopBatch1_,
			(InputElems() + ScoreElems() + MapElems()) * sizeof(float) / (1024.0 * 1024.0));
	}
	catch (...) {
		// The destructor is NOT called on a constructor that throws:
		// without this, already created streams, events, and buffers remain dangling.
		// The structural alternative is to wrap each CUDA resource in its own RAII
		// type, which would make this catch block disappear.
		Cleanup();
		throw;
	}
}

OrtSessionConfig::~OrtSessionConfig()
{
	Cleanup();
}

void OrtSessionConfig::Cleanup()
{
	// Unbind BEFORE destroying: the binding holds references to the
	// OrtValues, and Cleanup also runs from the constructor's catch block, where binding_
	// survives the call.
	if (binding_) {
		binding_.ClearBoundInputs();
		binding_.ClearBoundOutputs();
	}
	// Tensors do not own the device memory, but they reference it: they must
	// be destroyed before cudaFree.
	inputTensors_.clear();
	scoreTensors_.clear();
	mapTensors_.clear();

	if (dInput_) { cudaFree(dInput_); dInput_ = nullptr; }
	if (dScore_) { cudaFree(dScore_); dScore_ = nullptr; }
	if (dMap_) { cudaFree(dMap_);   dMap_ = nullptr; }

	if (evStart_) { cudaEventDestroy(evStart_); evStart_ = nullptr; }
	if (evH2D_) { cudaEventDestroy(evH2D_);   evH2D_ = nullptr; }
	if (evRun_) { cudaEventDestroy(evRun_);   evRun_ = nullptr; }
	if (evD2H_) { cudaEventDestroy(evD2H_);   evD2H_ = nullptr; }

	if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

void OrtSessionConfig::ReadModelGeometry(const PipelineConfig& cfg)
{
	Ort::AllocatorWithDefaultOptions allocator;

	if (session_.GetInputCount() != 1)
		throw std::runtime_error("expected a single input, the model has "
			+ std::to_string(session_.GetInputCount()));

	inputName_ = session_.GetInputNameAllocated(0, allocator).get();

	const auto inShape = session_.GetInputTypeInfo(0)
		.GetTensorTypeAndShapeInfo().GetShape();
	if (inShape.size() != 4)
		throw std::runtime_error("input of rank " + std::to_string(inShape.size())
			+ ", expected NCHW");

	// The batch is almost always dynamic (-1) and comes from the config; C, H, and W
	// must be fixed, or the buffers cannot be sized.
	batch_ = inShape[0] > 0 ? static_cast<int>(inShape[0]) : cfg.BatchSize();
	modelC_ = static_cast<int>(inShape[1]);
	modelH_ = static_cast<int>(inShape[2]);
	modelW_ = static_cast<int>(inShape[3]);

	if (batch_ <= 0 || modelC_ <= 0 || modelH_ <= 0 || modelW_ <= 0)
		throw std::runtime_error("unresolvable input shape: "
			+ std::to_string(inShape[0]) + "x" + std::to_string(inShape[1]) + "x"
			+ std::to_string(inShape[2]) + "x" + std::to_string(inShape[3])
			+ ". Only the batch size can be dynamic.");

	// Outputs are recognized by RANK, not by index: the score has
	// rank 1 or 2, the map rank 3 or 4. Relying on the export order
	// is the easiest way to swap them unnoticed, and the
	// symptom would be a "map" that is actually a vector of 17 values.
	const std::size_t outCount = session_.GetOutputCount();
	if (outCount == 0 || outCount > 2)
		throw std::runtime_error("expected one or two outputs, the model has "
			+ std::to_string(outCount));

	bool haveScore = false, haveMap = false;
	for (std::size_t i = 0; i < outCount; ++i) {
		const std::string name = session_.GetOutputNameAllocated(i, allocator).get();
		const auto shape = session_.GetOutputTypeInfo(i)
			.GetTensorTypeAndShapeInfo().GetShape();

		if (shape.size() <= 2) {
			scoreName_ = name;
			haveScore = true;
		}
		else {
			mapName_ = name;
			mapH_ = static_cast<int>(shape[shape.size() - 2]);
			mapW_ = static_cast<int>(shape[shape.size() - 1]);
			if (mapH_ <= 0 || mapW_ <= 0)
				throw std::runtime_error("map with dynamic dimensions: "
					+ std::to_string(mapH_) + "x" + std::to_string(mapW_));
			haveMap = true;
		}
	}
	if (!haveScore || !haveMap)
		throw std::runtime_error("the model must expose a score (rank <= 2) "
			"and a map (rank >= 3); found " + std::to_string(outCount) + " outputs");
}

void OrtSessionConfig::CreateEvents()
{
	CudaCheck(cudaEventCreate(&evStart_), "cudaEventCreate(start)");
	CudaCheck(cudaEventCreate(&evH2D_), "cudaEventCreate(h2d)");
	CudaCheck(cudaEventCreate(&evRun_), "cudaEventCreate(run)");
	CudaCheck(cudaEventCreate(&evD2H_), "cudaEventCreate(d2h)");
}

void OrtSessionConfig::AllocateDeviceBuffers()
{
	CudaCheck(cudaMalloc(&dInput_, InputElems() * sizeof(float)), "cudaMalloc(input)");
	CudaCheck(cudaMalloc(&dScore_, ScoreElems() * sizeof(float)), "cudaMalloc(score)");
	CudaCheck(cudaMalloc(&dMap_, MapElems() * sizeof(float)), "cudaMalloc(map)");
}

void OrtSessionConfig::BuildTensors()
{
	// Device pointers and shapes are FIXED for the entire life of the session:
	// tensors are built only once. With loopBatch1 there are 3*batch_
	// objects that would otherwise be rebuilt at each batch inside the
	// hot loop, with the corresponding allocation.
	const std::size_t imgElems = static_cast<std::size_t>(modelC_) * modelH_ * modelW_;
	const std::size_t mapElems1 = static_cast<std::size_t>(mapH_) * mapW_;

	if (loopBatch1_) {
		const int64_t inShape[4] = { 1, modelC_, modelH_, modelW_ };
		const int64_t scShape[1] = { 1 };
		const int64_t mpShape[4] = { 1, 1, mapH_, mapW_ };

		inputTensors_.reserve(batch_);
		scoreTensors_.reserve(batch_);
		mapTensors_.reserve(batch_);

		for (int i = 0; i < batch_; ++i) {
			inputTensors_.push_back(Ort::Value::CreateTensor<float>(
				deviceMemInfo_, dInput_ + i * imgElems, imgElems, inShape, 4));
			scoreTensors_.push_back(Ort::Value::CreateTensor<float>(
				deviceMemInfo_, dScore_ + i, 1, scShape, 1));
			mapTensors_.push_back(Ort::Value::CreateTensor<float>(
				deviceMemInfo_, dMap_ + i * mapElems1, mapElems1, mpShape, 4));
		}
	}
	else {
		const int64_t inShape[4] = { batch_, modelC_, modelH_, modelW_ };
		const int64_t scShape[1] = { batch_ };
		const int64_t mpShape[4] = { batch_, 1, mapH_, mapW_ };

		inputTensors_.push_back(Ort::Value::CreateTensor<float>(
			deviceMemInfo_, dInput_, InputElems(), inShape, 4));
		scoreTensors_.push_back(Ort::Value::CreateTensor<float>(
			deviceMemInfo_, dScore_, ScoreElems(), scShape, 1));
		mapTensors_.push_back(Ort::Value::CreateTensor<float>(
			deviceMemInfo_, dMap_, MapElems(), mpShape, 4));

		// Full batch: the binding never changes, it is done once and
		// at steady state only Run is left.
		binding_.BindInput(inputName_.c_str(), inputTensors_[0]);
		binding_.BindOutput(scoreName_.c_str(), scoreTensors_[0]);
		binding_.BindOutput(mapName_.c_str(), mapTensors_[0]);
	}
}

void OrtSessionConfig::RunAllBound()
{
	const Ort::RunOptions runOptions{ nullptr };

	if (loopBatch1_) {
		// PatchCore and similar: the memory bank does not fit in VRAM with a full batch,
		// hence batch_ executions of size 1. Tensors are already prepared: in the loop
		// only the bind and Run remain.
		for (int i = 0; i < batch_; ++i) {
			binding_.ClearBoundInputs();
			binding_.ClearBoundOutputs();
			binding_.BindInput(inputName_.c_str(), inputTensors_[i]);
			binding_.BindOutput(scoreName_.c_str(), scoreTensors_[i]);
			binding_.BindOutput(mapName_.c_str(), mapTensors_[i]);
			session_.Run(runOptions, binding_);
		}
	}
	else {
		session_.Run(runOptions, binding_);
	}
}

void OrtSessionConfig::RunBatch(const float* input, float* scores, float* map, Timings& t)
{
	CudaCheck(cudaEventRecord(evStart_, stream_), "eventRecord(start)");

	CudaCheck(cudaMemcpyAsync(dInput_, input, InputElems() * sizeof(float),
		cudaMemcpyHostToDevice, stream_), "memcpyAsync H2D");

	CudaCheck(cudaEventRecord(evH2D_, stream_), "eventRecord(h2d)");

	RunAllBound();

	CudaCheck(cudaEventRecord(evRun_, stream_), "eventRecord(run)");

	CudaCheck(cudaMemcpyAsync(scores, dScore_, ScoreElems() * sizeof(float),
		cudaMemcpyDeviceToHost, stream_), "memcpyAsync D2H score");
	CudaCheck(cudaMemcpyAsync(map, dMap_, MapElems() * sizeof(float),
		cudaMemcpyDeviceToHost, stream_), "memcpyAsync D2H map");

	CudaCheck(cudaEventRecord(evD2H_, stream_), "eventRecord(d2h)");

	// Upon return, the caller's buffers must be ready. It is also the point
	// where the GPU stalls while the host prepares the next batch:
	// when we need to overlap it, the modification will be entirely within this file.
	CudaCheck(cudaStreamSynchronize(stream_), "streamSynchronize");

	float ms = 0.0f;
	cudaEventElapsedTime(&ms, evStart_, evH2D_); t.h2dMs = ms;
	cudaEventElapsedTime(&ms, evH2D_, evRun_);   t.runMs = ms;
	cudaEventElapsedTime(&ms, evRun_, evD2H_);   t.d2hMs = ms;
}

void OrtSessionConfig::Warmup(int runs)
{
	if (runs <= 0) return;

	// Zeros on the device: no host buffer to allocate. With TensorRT, the first
	// execution compiles or loads the engine, which is the real purpose of all
	// this; the subsequent ones warm up the clocks.
	CudaCheck(cudaMemsetAsync(dInput_, 0, InputElems() * sizeof(float), stream_),
		"memsetAsync(warmup)");

	for (int r = 0; r < runs; ++r) RunAllBound();

	CudaCheck(cudaStreamSynchronize(stream_), "streamSynchronize(warmup)");
	Log::Info("Warmup completed: {} runs", runs);
}

// accessors

int OrtSessionConfig::BatchSize()     const { return batch_; }
int OrtSessionConfig::ModelChannels() const { return modelC_; }
int OrtSessionConfig::ModelHeight()   const { return modelH_; }
int OrtSessionConfig::ModelWidth()    const { return modelW_; }
int OrtSessionConfig::MapHeight()     const { return mapH_; }
int OrtSessionConfig::MapWidth()      const { return mapW_; }

std::size_t OrtSessionConfig::InputElems() const
{
	return static_cast<std::size_t>(batch_) * modelC_ * modelH_ * modelW_;
}

std::size_t OrtSessionConfig::ScoreElems() const
{
	return static_cast<std::size_t>(batch_);
}

std::size_t OrtSessionConfig::MapElems() const
{
	return static_cast<std::size_t>(batch_) * mapH_ * mapW_;
}

const ContractMetadata& OrtSessionConfig::Contract() const { return contract_; }
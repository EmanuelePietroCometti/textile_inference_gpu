#include "ContractMetadata.h"
#include "AsyncLogger.h"
#include <stdexcept>
ContractMetadata::ContractMetadata(ContractKeys keys) :
	normalizationInGraph(false),
	convertBgrToRgb(true),
	hasPixelThreshold(false),
	hasMin(false),
	hasMax(false),
	hasThreshold(false),
	mapMin(0.0f),
	mapMax(1.0f),
	scoreThreshold(0.5f),
	pixelThreshold(0.5f),
	keys_(std::move(keys))
{
}


void ContractMetadata::Load(Ort::Session& session)
{
	Ort::AllocatorWithDefaultOptions allocator;
	Ort::ModelMetadata metadata = session.GetModelMetadata();

	auto lookupString = [&](const char* key) -> std::string {
		auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
		return value ? std::string(value.get()) : std::string();
	};

	auto lookupFloat = [&](const char* key, float& target) -> bool {
		auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
		if (!value) return false;
		try {
			target = std::stof(value.get());
		}
		catch (const std::exception&) {
			throw std::runtime_error(std::string("Malformed metadata value for key '") + key + "': '" + value.get() + "'");
		}
		Log::Info("Metadata '{}' = {}", key, target);
		return true;
	};

	contractVersion = lookupString(keys_.contractVersion.c_str());
	if (contractVersion.empty())
	{
		Log::Warning("Metadata '{}' missing. Proceeding assuming backwards compatibility.", keys_.contractVersion);
	}
	else
	{
		Log::Info("ONNX Contract Version validated: {}", contractVersion);
	}

	const std::string normalization = lookupString(keys_.normalizationInsideGraph.c_str());
	normalizationInGraph = (normalization == "true");
	if (!normalizationInGraph) {
		Log::Warning("### WARNING: '{}'='{}' (expected 'true'). Host will apply ImageNet normalization as fallback.", keys_.normalizationInsideGraph ,normalization);
	}

	const std::string colorConversion = lookupString(keys_.preprocColorConversion.c_str());
	convertBgrToRgb = colorConversion.empty() || colorConversion == "bgr2rgb";
	Log::Info("Metadata '{}' = '{}' -> BGR2RGB swap {}.", keys_.preprocColorConversion ,colorConversion.empty() ? "(missing, default)" : colorConversion, convertBgrToRgb ? "ENABLED" : "disabled");

	const std::string calibratedStatus = lookupString(keys_.calibrationStatus.c_str());
	if (calibratedStatus != "true") 
	{
		Log::Warning("Model '{}' flag is not true. Using uncalibrated raw tensor bounds.", keys_.calibrationStatus);
	}

	// Assuming lookupFloat and keys_ (ContractKeys) are available in the class scope
	hasMin = lookupFloat(keys_.minScore.c_str(), mapMin);
	hasMax = lookupFloat(keys_.maxScore.c_str(), mapMax);
	hasThreshold = lookupFloat(keys_.threshold.c_str(), scoreThreshold);
	hasPixelThreshold = lookupFloat(keys_.pixelThreshold.c_str(), pixelThreshold);

	// check completeness: if a key is missing, the variables retain their default 
	// or uninitialized values, and the range guard below would misdiagnose the problem.
	if (!hasMin || !hasMax || !hasThreshold || !hasPixelThreshold)
	{
		std::string missing;
		// Lambda to append missing keys dynamically
		auto note = [&missing](bool present, const std::string& key) {
			if (present) return;
			if (!missing.empty()) missing += ", ";
			missing += key;
			};

		note(hasMin, keys_.minScore);
		note(hasMax, keys_.maxScore);
		note(hasThreshold, keys_.threshold);
		note(hasPixelThreshold, keys_.pixelThreshold);

		throw std::runtime_error(
			"Anomaly model rejected: calibration metadata is incomplete. Missing keys: " + missing
		);
	}

	// THEN check the range, now that we are certain both values actually came from the model.
	if (mapMax - mapMin <= 0.0f)
	{
		throw std::runtime_error(fmt::format(
			"Invalid calibration metadata: '{}' ({}) must be greater than '{}' ({}).",
			keys_.maxScore, mapMax, keys_.minScore, mapMin
		));
	}

	const std::string normFormula = lookupString(keys_.normalizationFormula.c_str());
	if (!normFormula.empty() && normFormula != "minmax")
	{
		Log::Warning("Model exported with '{}'='{}'. C++ engine uses 'minmax' natively.", keys_.normalizationFormula,normFormula);
	}

	Log::Info("Contract limits verified. mapMin={}, mapMax={}, scoreThreshold={}, pixelThreshold={} (present={}), normalizationInGraph={}.",
		mapMin, mapMax, scoreThreshold, pixelThreshold, hasPixelThreshold, normalizationInGraph);
}

bool ContractMetadata::NormalizationInGraph() const
{
	return normalizationInGraph;
}

bool ContractMetadata::ConvertBgrToRgb() const
{
	return convertBgrToRgb;
}

bool ContractMetadata::HasPixelThreshold() const
{
	return hasPixelThreshold;
}

bool ContractMetadata::HasMin() const
{
	return hasMin;
}

bool ContractMetadata::HasMax() const
{
	return hasMax;
}

bool ContractMetadata::HasThreshold() const
{
	return hasThreshold;
}

float ContractMetadata::MapMin() const
{
	return mapMin;
}

float ContractMetadata::MapMax() const
{
	return mapMax;
}

float ContractMetadata::ScoreThreshold() const
{
	return scoreThreshold;
}

float ContractMetadata::PixelThreshold() const
{
	return pixelThreshold;
}

const std::string& ContractMetadata::ContractVersion() const
{
	return contractVersion;
}
#include "ContractMetadata.h"
#include "AsyncLogger.h"
#include <stdexcept>

Metadata::Metadata(
	std::string contractVersionFieldName, 
	std::string preprocColorConversionFieldName, 
	std::string minScoreFieldName, 
	std::string maxScoreFieldName, 
	std::string thresholdFieldName, 
	std::string pixelthresholdFieldName, 
	std::string calibationStatusFieldName,
	std::string normalizationFormulaFieldName,
	std::string normalizationInsideGraph
) :
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
	contractVersionFieldName_(contractVersionFieldName),
	preprocColorConversionFieldName_(preprocColorConversionFieldName),
	minScoreFieldName_(minScoreFieldName),
	maxScoreFieldName_(maxScoreFieldName),
	thresholdFieldName_(thresholdFieldName),
	pixelthresholdFieldName_(pixelthresholdFieldName),
	calibationStatusFieldName_(calibationStatusFieldName),
	normalizationFormulaFieldName_(normalizationFormulaFieldName),
	normalizationInsideGraphFieldName_(normalizationInsideGraph)
{
	
}

Metadata::~Metadata()
{
}

void Metadata::Load(Ort::Session& session)
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

	contractVersion = lookupString(contractVersionFieldName_.c_str());
	if (contractVersion.empty())
	{
		Log::Warning("Metadata '{}' missing. Proceeding assuming backwards compatibility.", contractVersionFieldName_);
	}
	else
	{
		Log::Info("ONNX Contract Version validated: {}", contractVersion);
	}

	const std::string normalization = lookupString(normalizationInsideGraphFieldName_.c_str());
	normalizationInGraph = (normalization == "true");
	if (!normalizationInGraph) {
		Log::Warning("### WARNING: '{}'='{}' (expected 'true'). Host will apply ImageNet normalization as fallback.", normalizationFormulaFieldName_ ,normalization);
	}

	const std::string colorConversion = lookupString(preprocColorConversionFieldName_.c_str());
	convertBgrToRgb = colorConversion.empty() || colorConversion == "bgr2rgb";
	Log::Info("Metadata '{}' = '{}' -> BGR2RGB swap {}.", preprocColorConversionFieldName_ ,colorConversion.empty() ? "(missing, default)" : colorConversion, convertBgrToRgb ? "ENABLED" : "disabled");

	const std::string calibratedStatus = lookupString(calibationStatusFieldName_.c_str());
	if (calibratedStatus != "true") 
	{
		Log::Warning("Model '{}' flag is not true. Using uncalibrated raw tensor bounds.", calibationStatusFieldName_);
	}

	hasMin = lookupFloat(minScoreFieldName_.c_str(), mapMin);
	hasMax = lookupFloat(maxScoreFieldName_.c_str(), mapMax);
	if (mapMax - mapMin <= 0.0f)
	{
		throw std::runtime_error("Invalid calibration metadata: map_max_raw must be greater than map_min_raw.");
	}


	hasThreshold = lookupFloat(thresholdFieldName_.c_str(), scoreThreshold);
	hasPixelThreshold = lookupFloat(pixelthresholdFieldName_.c_str(), pixelThreshold);

	if (!hasMin || !hasMax || !hasThreshold || !hasPixelThreshold)
	{
		throw std::runtime_error(
			"Anomaly model rejected: calibration metadata is incomplete "
		);
	}

	const std::string normFormula = lookupString(normalizationFormulaFieldName_.c_str());
	if (!normFormula.empty() && normFormula != "minmax")
	{
		Log::Warning("Model exported with '{}'='{}'. C++ engine uses 'minmax' natively.", normalizationFormulaFieldName_,normFormula);
	}

	Log::Info("Contract limits verified. mapMin={}, mapMax={}, scoreThreshold={}, pixelThreshold={} (present={}), normalizationInGraph={}.",
		mapMin, mapMax, scoreThreshold, pixelThreshold, hasPixelThreshold, normalizationInGraph);
}

bool Metadata::NormalizationInGraph() const
{
	return normalizationInGraph;
}

bool Metadata::ConvertBgrToRgb() const
{
	return convertBgrToRgb;
}

bool Metadata::HasPixelThreshold() const
{
	return hasPixelThreshold;
}

bool Metadata::HasMin() const
{
	return hasMin;
}

bool Metadata::HasMax() const
{
	return hasMax;
}

bool Metadata::HasThreshold() const
{
	return hasThreshold;
}

float Metadata::MapMin() const
{
	return mapMin;
}

float Metadata::MapMax() const
{
	return mapMax;
}

float Metadata::ScoreThreshold() const
{
	return scoreThreshold;
}

float Metadata::PixelThreshold() const
{
	return pixelThreshold;
}

const std::string& Metadata::ContractVersion() const
{
	return contractVersion;
}
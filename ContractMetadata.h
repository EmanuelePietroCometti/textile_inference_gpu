#pragma once

#include <string>
#include <onnxruntime_cxx_api.h>

/**
 * @brief Extractor and container for custom metadata embedded within an ONNX model.
 *
 * @details This class is responsible for querying an `Ort::Session` to extract specific
 * custom metadata properties injected into the ONNX model during export (e.g., from PyTorch).
 * It stores critical inference configurations such as color space requirements, normalization
 * strategies, and post-processing thresholds (e.g., for anomaly detection or segmentation masks).
 */
class Metadata {
public:
    /**
     * @brief Constructs the Metadata object by defining the keys used to query the ONNX model.
     *
     * @param contractVersionFieldName Key for the model's contract or API version.
     * @param preprocColorConversionFieldName Key for color space conversion rules (e.g., BGR to RGB).
     * @param minScoreFieldName Key for the minimum score mapping value.
     * @param maxScoreFieldName Key for the maximum score mapping value.
     * @param thresholdFieldName Key for the anomaly/classification score threshold.
     * @param pixelthresholdFieldName Key for the pixel-level segmentation mask threshold.
     * @param calibationStatusFieldName Key indicating if the model requires or has undergone calibration.
     * @param normalizationFormulaFieldName Key specifying the formula used for input normalization.
     * @param normalizationInsideGraph Key indicating whether normalization is embedded as a graph node.
     */
    Metadata(std::string contractVersionFieldName, std::string preprocColorConversionFieldName,
        std::string minScoreFieldName, std::string maxScoreFieldName,
        std::string thresholdFieldName, std::string pixelthresholdFieldName,
        std::string calibationStatusFieldName, std::string normalizationFormulaFieldName,
        std::string normalizationInsideGraph);

    /**
     * @brief Destructor.
     */
    ~Metadata();

    /**
     * @brief Queries the loaded ONNX session and populates internal variables with the retrieved metadata.
     *
     * @param session Reference to an active ONNX Runtime session initialized with the target model.
     */
    void Load(Ort::Session& session);

    /**
     * @brief Checks if input normalization (mean/std subtraction) is handled internally by the ONNX graph.
     * @return `true` if normalization is in-graph, `false` if it must be done explicitly during preprocessing.
     */
    bool NormalizationInGraph() const;

    /**
     * @brief Checks if the model expects RGB input instead of the OpenCV default BGR.
     * @return `true` if BGR to RGB conversion is required before inference.
     */
    bool ConvertBgrToRgb() const;

    /**
     * @brief Checks if the model provides a specific threshold for pixel-level mask binarization.
     * @return `true` if a pixel threshold is defined.
     */
    bool HasPixelThreshold() const;

    /**
     * @brief Checks if a minimum mapping bound is defined for score normalization.
     * @return `true` if the min bound exists.
     */
    bool HasMin() const;

    /**
     * @brief Checks if a maximum mapping bound is defined for score normalization.
     * @return `true` if the max bound exists.
     */
    bool HasMax() const;

    /**
     * @brief Checks if a global image-level score threshold is defined.
     * @return `true` if the score threshold exists.
     */
    bool HasThreshold() const;

    /**
     * @brief Retrieves the minimum value used for min-max score mapping.
     * @return The minimum bound value.
     */
    float MapMin() const;

    /**
     * @brief Retrieves the maximum value used for min-max score mapping.
     * @return The maximum bound value.
     */
    float MapMax() const;

    /**
     * @brief Retrieves the global anomaly or classification score threshold.
     * @return The classification threshold.
     */
    float ScoreThreshold() const;

    /**
     * @brief Retrieves the threshold used for binarizing pixel-level predictions.
     * @return The pixel-level threshold.
     */
    float PixelThreshold() const;

    /**
     * @brief Retrieves the model's contract or API version string.
     * @return A const reference to the version string.
     */
    const std::string& ContractVersion() const;

private:
    bool normalizationInGraph, convertBgrToRgb, hasPixelThreshold, hasMin, hasMax, hasThreshold;
    float mapMin, mapMax, scoreThreshold, pixelThreshold;
    std::string contractVersion, configIniPath;

    // Stored keys used for ONNX metadata lookup
    std::string contractVersionFieldName_;
    std::string preprocColorConversionFieldName_;
    std::string minScoreFieldName_;
    std::string maxScoreFieldName_;
    std::string thresholdFieldName_;
    std::string pixelthresholdFieldName_;
    std::string calibationStatusFieldName_;
    std::string normalizationFormulaFieldName_;
    std::string normalizationInsideGraphFieldName_;
};
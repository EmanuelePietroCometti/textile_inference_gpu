#pragma once

#include "SimpleIni.h"
#include <string>

/**
 * @brief Lightweight wrapper class for parsing and retrieving INI configurations.
 *
 * This class encapsulates the `CSimpleIniW` library to provide a simplified and
 * robust interface for managing configuration settings. It supports loading data
 * both from memory payloads and physical files.
 *
 * @details
 * - **Initialization & Loading**: You can instantiate an empty config, parse an
 *   in-memory string immediately via the parameterized constructor, or load a
 *   configuration later from the filesystem using `Load()`.
 * - **Safe Retrieval**: The getter methods (`GetString`, `GetInt`) are designed
 *   to be fail-safe. They require a default fallback value (`def`) that is automatically
 *   returned if the requested `section` or `key` is missing or malformed, preventing
 *   runtime crashes due to missing configuration entries.
 */
class IniConfig {
public:
    /**
     * @brief Default constructor. Initializes an empty configuration object.
     */
    IniConfig();

    /**
     * @brief Constructs and parses the configuration directly from an INI-formatted string.
     *
     * @param iniText A raw string containing the INI configuration data payload.
     */
    explicit IniConfig(const std::string& iniText);

    /**
     * @brief Loads and parses an INI configuration from a file path.
     *
     * @param iniPath The wide-string filesystem path pointing to the `.ini` file.
     * @return `true` if the file was successfully found, loaded, and parsed; `false` otherwise.
     */
    bool Load(const std::wstring& iniPath);

    /**
     * @brief Retrieves a string value from the specified INI section and key.
     *
     * @param section The name of the INI section (e.g., L"Settings").
     * @param key The specific key within the section to look up.
     * @param def The default fallback value to return if the key/section is not found.
     * @return The retrieved wide-string value, or `def` if the entry does not exist.
     */
    std::wstring GetString(const std::wstring& section, const std::wstring& key, std::wstring def) const;

    /**
     * @brief Retrieves an integer value from the specified INI section and key.
     *
     * @param section The name of the INI section.
     * @param key The specific key within the section to look up.
     * @param def The default fallback value to return if the key/section is not found or is invalid.
     * @return The retrieved integer value, or `def` if the entry does not exist/fails conversion.
     */
    long GetInt(const std::wstring& section, const std::wstring& key, long def) const;

private:
    CSimpleIniW ini; ///< The underlying SimpleIni instance handling the parsing logic.
};
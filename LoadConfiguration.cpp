#include "LoadConfiguration.h"

IniConfig::IniConfig()
{
	ini.SetUnicode(true);
}

IniConfig::IniConfig(const std::string& iniText)
{
	ini.SetUnicode(true);
	ini.LoadData(iniText);
}

bool IniConfig::Load(const std::wstring& iniPath)
{
	return ini.LoadFile(iniPath.c_str()) == SI_OK;
}

std::wstring IniConfig::GetString(const std::wstring& section, const std::wstring& key, std::wstring def) const
{
	return ini.GetValue(section.c_str(), key.c_str(), def.c_str());
}

long IniConfig::GetInt(const std::wstring& section, const std::wstring& key, long def) const
{
	return ini.GetLongValue(section.c_str(), key.c_str(), def);
}
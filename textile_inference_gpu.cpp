// textile_inference_gpu.cpp : Questo file contiene la funzione 'main', in cui inizia e termina l'esecuzione del programma.
//

#include <iostream>
#include "AsyncLogger.h"
#include "IniConfig.h"
#include "fmt/core.h"
#include "fmt/color.h"
#include <windows.h>


// Configuration parameters
const std::wstring iniPath = L"iniConfigFile_onnxInference.ini";

namespace ConfigKeys {
    namespace Section {
        constexpr const wchar_t* Logger = L"Logger";
    }

    namespace Field {
        constexpr const wchar_t* MaxMessageChars = L"maxMessageChars";
        constexpr const wchar_t* QueueCapacity = L"queueCapacity";
        constexpr const wchar_t* FlushIntervalMs = L"flushIntervalMs";
        constexpr const wchar_t* NotifyThreshold = L"notifyThreshold";
    }
}

bool verifyColorSupport()
{
	bool s_ansiColorSupported = true;
	for (DWORD stdHandle : { STD_OUTPUT_HANDLE, STD_ERROR_HANDLE }) {
		HANDLE h = GetStdHandle(stdHandle);
		DWORD mode = 0;
		if (h == NULL || h == INVALID_HANDLE_VALUE || !GetConsoleMode(h, &mode)) {
			s_ansiColorSupported = false;
			continue;
		}
		if (!SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
			s_ansiColorSupported = false;
		}
	}
	return s_ansiColorSupported;
}


void preprocessingWorker()
{

}

void inferenceWorker()
{

}

void postprocessingWorker()
{

}


int main()
{
    IniConfig cfg;
    bool cfgFound = cfg.Load(iniPath);
    if (cfgFound)
    {
        fmt::print(verifyColorSupport() ? fg(fmt::color::yellow) : fmt::text_style{},
            "[ERROR] Ini configuration file did not found! \n");
        return -1;
    }

    long maxMessageChars = cfg.GetInt(ConfigKeys::Section::Logger, ConfigKeys::Field::MaxMessageChars, 480L);
    long queueCapacity = cfg.GetInt(ConfigKeys::Section::Logger, ConfigKeys::Field::QueueCapacity, 4096L);
    long flushIntervalMs = cfg.GetInt(ConfigKeys::Section::Logger, ConfigKeys::Field::FlushIntervalMs, 200L);
    long notifyThreshold = cfg.GetInt(ConfigKeys::Section::Logger, ConfigKeys::Field::NotifyThreshold, 1024L);

    AsyncLogger logger(maxMessageChars, queueCapacity, flushIntervalMs, notifyThreshold);

    Log::Info("ONNX inference program started!");



}

// Per eseguire il programma: CTRL+F5 oppure Debug > Avvia senza eseguire debug
// Per eseguire il debug del programma: F5 oppure Debug > Avvia debug

// Suggerimenti per iniziare: 
//   1. Usare la finestra Esplora soluzioni per aggiungere/gestire i file
//   2. Usare la finestra Team Explorer per connettersi al controllo del codice sorgente
//   3. Usare la finestra di output per visualizzare l'output di compilazione e altri messaggi
//   4. Usare la finestra Elenco errori per visualizzare gli errori
//   5. Passare a Progetto > Aggiungi nuovo elemento per creare nuovi file di codice oppure a Progetto > Aggiungi elemento esistente per aggiungere file di codice esistenti al progetto
//   6. Per aprire di nuovo questo progetto in futuro, passare a File > Apri > Progetto e selezionare il file con estensione sln

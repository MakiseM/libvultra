#include "vultra/core/profiling/renderdoc_api.hpp"
#include "vultra/core/base/common_context.hpp"

#include <vbase/core/exe_path.hpp>

#include <renderdoc_app.h>

#include <filesystem>
#include <cstdlib>
#include <string>

#ifdef WIN32
#include <Windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#endif

namespace vultra
{
    namespace
    {
        std::filesystem::path getCaptureRoot() { return vbase::executable_dir(); }

        std::filesystem::path getRenderDocDllPath()
        {
            if (const char* env = std::getenv("VULTRA_RENDERDOC_DLL"); env && env[0] != '\0')
                return std::filesystem::path {env};
            if (std::filesystem::exists("D:/tools/RenderDoc/renderdoc.dll"))
                return "D:/tools/RenderDoc/renderdoc.dll";
            return "C:/Program Files/RenderDoc/renderdoc.dll";
        }

        std::filesystem::path getRenderDocCaptureTemplate()
        {
            if (const char* env = std::getenv("VULTRA_RENDERDOC_CAPTURE_TEMPLATE");
                env && env[0] != '\0')
                return std::filesystem::path {env};
            return getCaptureRoot() / "captures" / "myframe";
        }
    } // namespace

    RenderDocAPI::RenderDocAPI(bool enable)
    {
#ifdef VULTRA_ENABLE_RENDERDOC
        VULTRA_CORE_TRACE("[Profiling] Initializing RenderDoc API...");
        if (const char* disableEnv = std::getenv("VULTRA_RENDERDOC_DISABLE");
            disableEnv && disableEnv[0] != '\0' && disableEnv[0] != '0')
        {
            VULTRA_CORE_TRACE("[Profiling] RenderDoc API is disabled by VULTRA_RENDERDOC_DISABLE.");
            return;
        }
        if (!enable)
        {
            VULTRA_CORE_TRACE("[Profiling] RenderDoc API is disabled by user.");
            return;
        }

        // Load the RenderDoc API
        m_IsAvailable = load();
        if (m_IsAvailable)
        {
            m_RenderDocAPI = getRenderDocAPI();
        }
#endif
    }

    RenderDocAPI::~RenderDocAPI()
    {
        if (m_RenderDocAPI)
        {
            m_RenderDocAPI->Shutdown();
            m_RenderDocAPI = nullptr;
        }
        unload();
    }

    bool RenderDocAPI::load()
    {
        if (m_IsAvailable)
        {
            VULTRA_CORE_TRACE("[Profiling] RenderDoc API is already loaded.");
            return true;
        }

        return loadDLL();
    }

    void RenderDocAPI::unload()
    {
        if (!m_IsAvailable)
        {
            return;
        }

        unloadDLL();

        m_IsAvailable    = false;
        m_RenderDocAPI   = nullptr;
        RENDERDOC_GetAPI = nullptr;

        VULTRA_CORE_TRACE("[Profiling] RenderDoc API unloaded successfully.");
    }

    // const RENDERDOC_API_1_6_0* RenderDocAPI::getAPI() const
    // {
    //     if (!m_IsAvailable)
    //     {
    //         VULTRA_CORE_ERROR("[Profiling] RenderDoc API is not available.");
    //         return nullptr;
    //     }
    //     return m_RenderDocAPI;
    // }

    bool RenderDocAPI::isAvailable() const { return m_IsAvailable; }

    bool RenderDocAPI::isFrameCapturing() const
    {
        if (m_RenderDocAPI)
        {
            return m_RenderDocAPI->IsFrameCapturing() != 0;
        }
        return false;
    }

    void RenderDocAPI::startFrameCapture() const
    {
        if (m_RenderDocAPI)
        {
            m_RenderDocAPI->StartFrameCapture(nullptr, nullptr);
        }
    }

    bool RenderDocAPI::endFrameCapture() const
    {
        if (m_RenderDocAPI)
        {
            return m_RenderDocAPI->EndFrameCapture(nullptr, nullptr) != 0;
        }
        return false;
    }

    void RenderDocAPI::setCaptureFilePathTemplate(const std::string_view path) const
    {
        if (m_RenderDocAPI)
        {
            m_RenderDocAPI->SetCaptureFilePathTemplate(path.data());
        }
    }

    void RenderDocAPI::setCaptureTitle(const std::string_view title) const
    {
        if (m_RenderDocAPI)
        {
            m_RenderDocAPI->SetCaptureTitle(title.data());
        }
    }

    void RenderDocAPI::setCaptureFileComments(const std::string_view path, const std::string_view comments) const
    {
        if (m_RenderDocAPI)
        {
            m_RenderDocAPI->SetCaptureFileComments(path.data(), comments.data());
        }
    }

    bool RenderDocAPI::isTargetControlConnected() const
    {
        if (m_RenderDocAPI)
        {
            return m_RenderDocAPI->IsTargetControlConnected() != 0;
        }
        return false;
    }

    void RenderDocAPI::launchReplayUI(uint32_t connectTargetControl, const std::string_view cmdline) const
    {
        if (m_RenderDocAPI)
        {
            m_RenderDocAPI->LaunchReplayUI(connectTargetControl, cmdline.data());
        }
    }

    void RenderDocAPI::showReplayUI() const
    {
        if (m_RenderDocAPI)
        {
            m_RenderDocAPI->ShowReplayUI();
        }
    }

    uint32_t RenderDocAPI::getCaptureCount() const
    {
        if (m_RenderDocAPI)
        {
            return m_RenderDocAPI->GetNumCaptures();
        }
        return 0;
    }

    std::string RenderDocAPI::getCapturePath(uint32_t index) const
    {
        if (!m_RenderDocAPI)
        {
            return {};
        }

        uint32_t pathLength = 0;
        uint64_t timestamp  = 0;
        if (m_RenderDocAPI->GetCapture(index, nullptr, &pathLength, &timestamp) == 0u || pathLength == 0u)
        {
            return {};
        }

        std::string path(pathLength, '\0');
        if (m_RenderDocAPI->GetCapture(index, path.data(), &pathLength, &timestamp) == 0u)
        {
            return {};
        }

        if (!path.empty() && path.back() == '\0')
        {
            path.pop_back();
        }
        return path;
    }

    bool RenderDocAPI::loadDLL()
    {
        if (m_IsAvailable)
        {
            return true;
        }

#ifdef _WIN32
        const std::filesystem::path dllPath = getRenderDocDllPath();
        if (m_Module = LoadLibraryW(dllPath.wstring().c_str()); m_Module)
        {
            RENDERDOC_GetAPI = reinterpret_cast<pRENDERDOC_GetAPI>(
                GetProcAddress(reinterpret_cast<HMODULE>(m_Module), "RENDERDOC_GetAPI"));
        }
        else
        {
            // Handle warning
            VULTRA_CORE_WARN("[Profiling] Failed to load RenderDoc DLL at {}", dllPath.string());
            return false;
        }
#elif defined(__linux__)
        if (m_Module = dlopen("/usr/lib/librenderdoc.so", RTLD_NOW | RTLD_LOCAL); m_Module)
        {
            RENDERDOC_GetAPI = reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(m_Module, "RENDERDOC_GetAPI"));
        }
        else
        {
            // Handle warning
            VULTRA_CORE_WARN("[Profiling] Failed to load RenderDoc shared library. Ensure it is installed and the "
                             "path is correct.");
            return false;
        }
#else
#warning "Unsupported platform for RenderDoc API loading"
        return false;
#endif

        return true;
    }

    void RenderDocAPI::unloadDLL()
    {
        if (!m_IsAvailable)
        {
            return;
        }

#ifdef _WIN32
        FreeLibrary(reinterpret_cast<HMODULE>(m_Module));
#elif defined(__linux__) || defined(__android__)
        dlclose(m_Module);
#else
#warning "Unsupported platform for RenderDoc API loading"
#endif

        m_Module = nullptr;
    }

    RENDERDOC_API_1_6_0* RenderDocAPI::getRenderDocAPI()
    {
        if (!m_RenderDocAPI)
        {
            if (RENDERDOC_GetAPI)
            {
                int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, reinterpret_cast<void**>(&m_RenderDocAPI));
                assert(ret == 1);

                // Print the RenderDoc API version
                int major, minor, patch;
                m_RenderDocAPI->GetAPIVersion(&major, &minor, &patch);
                VULTRA_CORE_TRACE("[Profiling] RenderDoc API version: {}.{}.{}", major, minor, patch);

                // Set options for RenderDoc
#if _DEBUG
                m_RenderDocAPI->SetCaptureOptionU32(eRENDERDOC_Option_DebugOutputMute, 0);
                m_RenderDocAPI->SetCaptureOptionU32(eRENDERDOC_Option_APIValidation, 1);
#endif
                const std::filesystem::path captureTemplate = getRenderDocCaptureTemplate();
                if (const auto parent = captureTemplate.parent_path(); !parent.empty())
                    std::filesystem::create_directories(parent);
                const std::string captureTemplateString = captureTemplate.generic_string();
                m_RenderDocAPI->SetCaptureFilePathTemplate(captureTemplateString.c_str());
                m_RenderDocAPI->MaskOverlayBits(eRENDERDOC_Overlay_None, eRENDERDOC_Overlay_None);
            }
        }
        return m_RenderDocAPI;
    }
} // namespace vultra

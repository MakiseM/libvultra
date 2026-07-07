#include "vultra/function/rendering/shader_system.hpp"
#include "vultra/core/base/common_context.hpp"
#include "vultra/core/rhi/structs/render_backend_api.hpp"

#if defined(__ANDROID__) || defined(__EMSCRIPTEN__)
#include <builtin_shaders.hpp>
#endif

#include <filesystem>

namespace vultra
{
    namespace
    {
        bool loadBuiltinShaderLibraryFromFile(rhi::ShaderLibraryRuntime& shaderLibrary, const char* relativePath)
        {
            const std::filesystem::path requestedPath(relativePath);
            if (shaderLibrary.loadFromFile(requestedPath.string()))
                return true;

            std::error_code ec;
            auto            currentPath = std::filesystem::current_path(ec);
            if (ec)
                return false;

            for (;;)
            {
                const auto candidatePath = currentPath / requestedPath;
                if (std::filesystem::exists(candidatePath, ec) && shaderLibrary.loadFromFile(candidatePath.string()))
                    return true;

                const auto parentPath = currentPath.parent_path();
                if (parentPath.empty() || parentPath == currentPath)
                    break;

                currentPath = parentPath;
            }

            return false;
        }
    } // namespace

    bool ShaderSystem::onInit()
    {
        VULTRA_CORE_INFO("[ShaderSystem] Initializing...");

        const auto backendApi           = ctx().config.render.backendApi;
        const auto builtinShaderLibrary = ctx().config.render.builtinShaderLibrary;
        const bool useWebGpuLibrary     = backendApi == rhi::RenderBackendApi::eWebGPU;
        const bool useCompatibilityLibrary =
            builtinShaderLibrary == EngineContext::Config::RenderConfig::BuiltinShaderLibrary::eCompatibility;

        const uint8_t* shaderLibData = nullptr;
        size_t         shaderLibSize = 0;
        const char*    shaderLibPath = nullptr;

        if (useWebGpuLibrary)
        {
#if defined(VULTRA_ENABLE_WEBGPU) && VULTRA_ENABLE_WEBGPU
#if defined(__EMSCRIPTEN__)
            shaderLibData = builtin_shaders_compatibility_web_vshweblib;
            shaderLibSize = builtin_shaders_compatibility_web_vshweblib_size;
#else
            shaderLibPath = "builtin/shader_lib/builtin_compatibility.vshweblib";
#endif
#else
            VULTRA_CORE_ERROR("[ShaderSystem] WebGPU shader library requested but WebGPU is disabled for this build");
            return false;
#endif
        }
        else
        {
#if defined(__ANDROID__)
            shaderLibData = builtin_shaders_compatibility_vshlib;
            shaderLibSize = builtin_shaders_compatibility_vshlib_size;
#else
            if (useCompatibilityLibrary)
            {
                shaderLibPath = "builtin/shader_lib/builtin_compatibility.vshlib";
            }
            else
            {
                shaderLibPath = "builtin/shader_lib/builtin_highend.vshlib";
            }
#endif
        }

        if (shaderLibPath != nullptr)
        {
            if (!loadBuiltinShaderLibraryFromFile(m_BuiltinShaderLibrary, shaderLibPath))
            {
                VULTRA_CORE_ERROR("[ShaderSystem] Failed to load builtin shader library: {}", shaderLibPath);
                return false;
            }
        }
        else if (!m_BuiltinShaderLibrary.loadFromMemory(shaderLibData, shaderLibSize))
        {
            VULTRA_CORE_ERROR("[ShaderSystem] Failed to load builtin shader library");
            return false;
        }

        VULTRA_CORE_TRACE("[ShaderSystem] Providing IShaderService");
        ctx().services.provide<IShaderService>(this);

        return true;
    }

    void ShaderSystem::onShutdown() { VULTRA_CORE_INFO("[ShaderSystem] Shutting down"); }
} // namespace vultra

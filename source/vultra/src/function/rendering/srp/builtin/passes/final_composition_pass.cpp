#include "vultra/function/rendering/srp/builtin/passes/final_composition_pass.hpp"
#include "vultra/core/rhi/texture.hpp"
#include "vultra/function/framegraph/framegraph_resource_access.hpp"
#include "vultra/function/framegraph/framegraph_import.hpp"
#include "vultra/function/rendering/srp/builtin/resource_keys.hpp"
#include "vultra/function/resource/gpu_scene_view.hpp"

#include <fg/FrameGraph.hpp>

#include <algorithm>
#include <cmath>
#include <vector>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

namespace vultra
{
    FinalCompositionPass::FinalCompositionPass() { setShaderProfile(rhi::ShaderProfile::eGeneral); }

    namespace
    {
        [[nodiscard]] constexpr bool isSrgbColorFormat(const rhi::PixelFormat format)
        {
            return format == rhi::PixelFormat::eRGBA8_sRGB || format == rhi::PixelFormat::eBGRA8_sRGB;
        }

        struct FinalCompositionPushConstants
        {
            glm::vec4 gazeMarker {0.5f, 0.5f, 0.0f, 10.0f};
            glm::vec4 targetSize {1.0f, 1.0f, 0.0f, 0.0f};
            glm::vec4 temporalGazeAndRings {0.5f, 0.5f, 32.0f, 56.0f};
            glm::vec4 temporalParams {0.0f, 0.0f, 1.0f, 0.18f};
            glm::vec4 temporalParams2 {0.20f, 0.85f, 1e-4f, 0.0f};
        };

        [[nodiscard]] glm::vec2 tanHalfFov(const RenderView& view)
        {
            if (!view.camera)
                return glm::vec2 {1.0f};
            const glm::mat4& projection = view.camera->projection;
            return glm::vec2 {1.0f / std::max(std::abs(projection[0][0]), 1e-5f),
                              1.0f / std::max(std::abs(projection[1][1]), 1e-5f)};
        }

        [[nodiscard]] float projectionYSign(const RenderView& view, const rhi::RenderBackendApi backendApi)
        {
            if (!view.camera)
                return 1.0f;

            float sign = view.camera->projection[1][1] < 0.0f ? -1.0f : 1.0f;
            if (backendApi == rhi::RenderBackendApi::eVulkan)
                sign *= -1.0f;
            return sign;
        }

        [[nodiscard]] bool temporalFilterEnabled(const RenderView& view, const bool useMultiview)
        {
            (void)view;
            (void)useMultiview;
            // M3 is applied to the Gaussian scene-color source before final
            // composition so benchmark capture and final output see the same
            // filtered image. Keep this pass as a plain presentation pass.
            return false;
        }

        [[nodiscard]] FinalCompositionPushConstants makePushConstants(const RenderView&        view,
                                                                      const rhi::RenderBackendApi backendApi,
                                                                      const bool              temporalEnabled,
                                                                      const bool              temporalHistoryValid)
        {
            FinalCompositionPushConstants pc {};
            pc.targetSize = glm::vec4 {
                static_cast<float>(std::max(view.extent.width, 1u)),
                static_cast<float>(std::max(view.extent.height, 1u)),
                0.0f,
                0.0f,
            };

            if (view.xrGazeValid)
            {
                pc.gazeMarker.x = std::clamp(view.xrGazeUv.x, 0.0f, 1.0f);
                pc.gazeMarker.y = 1.0f - std::clamp(view.xrGazeUv.y, 0.0f, 1.0f);
                pc.gazeMarker.z = 1.0f;
            }
            if (temporalEnabled && view.gpuSceneView)
            {
                const auto& gpuSceneView = *view.gpuSceneView;
                const glm::vec2 tanFov = tanHalfFov(view);
                const float midDegrees =
                    std::max(gpuSceneView.generalGaussianSplatFoveatedRingDegrees.y,
                             gpuSceneView.generalGaussianSplatFoveatedRingDegrees.x + 1.0f);
                const float outerDegrees =
                    std::max(gpuSceneView.generalGaussianSplatPeripheralTemporalFilterOuterDegrees,
                             midDegrees + 1.0f);
                pc.temporalGazeAndRings =
                    glm::vec4 {std::clamp(gpuSceneView.generalGaussianSplatFoveatedGaze.x, 0.0f, 1.0f),
                               std::clamp(gpuSceneView.generalGaussianSplatFoveatedGaze.y, 0.0f, 1.0f),
                               midDegrees,
                               outerDegrees};
                pc.temporalParams =
                    glm::vec4 {tanFov.x,
                               tanFov.y,
                               projectionYSign(view, backendApi),
                               std::max(gpuSceneView.generalGaussianSplatPeripheralTemporalFilterRejectionThreshold,
                                        1e-4f)};
                pc.temporalParams2 =
                    glm::vec4 {std::max(gpuSceneView.generalGaussianSplatPeripheralTemporalFilterClampRadius, 0.0f),
                               std::clamp(gpuSceneView.generalGaussianSplatPeripheralTemporalFilterLambdaScale,
                                          0.0f,
                                          1.0f),
                               temporalHistoryValid ? 1.0f : 0.0f,
                               0.0f};
            }
            return pc;
        }

    } // namespace

    constexpr auto PASS_NAME = "FinalCompositionPass";

    FrameGraphResource FinalCompositionPass::compose(FrameGraphBuildContext& ctx, FrameGraphResource target)
    {
        auto       source       = ctx.data.get(kResKey_FinalCompositionSource);
        const bool useMultiview = ctx.view().enableMultiview && ctx.view().multiviewCameraCount == 2u;
        const bool temporalEnabled = temporalFilterEnabled(ctx.view(), useMultiview);
        if (temporalEnabled)
            ensureTemporalHistory(ctx.rd, ctx.view().extent);
        else
            resetTemporalHistory();

        FrameGraphResource temporalHistoryRead {};
        FrameGraphResource temporalHistoryWrite {};
        bool temporalHistoryValid = false;
        uint32_t temporalWriteIndex = 0u;
        if (temporalEnabled && m_TemporalHistory[0] && m_TemporalHistory[1])
        {
            temporalHistoryValid = m_TemporalHistoryValid;
            temporalWriteIndex = 1u - m_TemporalHistoryReadIndex;
            temporalHistoryRead =
                framegraph::importTexture(ctx.fg,
                                          "M3TemporalHistoryRead",
                                          m_TemporalHistory[m_TemporalHistoryReadIndex].get());
            temporalHistoryWrite =
                framegraph::importTexture(ctx.fg,
                                          "M3TemporalHistoryWrite",
                                          m_TemporalHistory[temporalWriteIndex].get());
        }
        const auto pushConstants =
            makePushConstants(ctx.view(), ctx.rd.getBackendApi(), temporalEnabled, temporalHistoryValid);

        ctx.fg.addCallbackPass(
            PASS_NAME,
            [source, &target, temporalEnabled, temporalHistoryRead, temporalHistoryWrite](
                FrameGraph::Builder& builder, auto&) {
                PASS_SETUP_ZONE;

                builder.read(source,
                             framegraph::TextureRead {
                                 .binding =
                                     {
                                         .location      = {.set = 3, .binding = 0},
                                         .pipelineStage = framegraph::PipelineStage::eFragmentShader,
                                     },
                                 .type        = framegraph::TextureRead::Type::eCombinedImageSampler,
                                 .imageAspect = rhi::ImageAspect::eColor,
                             });
                if (temporalEnabled && temporalHistoryRead)
                {
                    builder.read(temporalHistoryRead,
                                 framegraph::TextureRead {
                                     .binding =
                                         {
                                             .location      = {.set = 3, .binding = 1},
                                             .pipelineStage = framegraph::PipelineStage::eFragmentShader,
                                         },
                                     .type        = framegraph::TextureRead::Type::eCombinedImageSampler,
                                     .imageAspect = rhi::ImageAspect::eColor,
                                 });
                }

                target = builder.write(target,
                                       framegraph::Attachment {
                                           .index       = 0,
                                           .imageAspect = rhi::ImageAspect::eColor,
                                           .clearValue  = framegraph::ClearValue::eOpaqueBlack,
                                       });
                if (temporalEnabled && temporalHistoryWrite)
                {
                    const auto writtenHistory =
                        builder.write(temporalHistoryWrite,
                                      framegraph::Attachment {
                                          .index       = 1,
                                          .imageAspect = rhi::ImageAspect::eColor,
                                      });
                    (void)writtenHistory;
                }
            },
            [this, target, useMultiview, temporalEnabled, pushConstants](
                const auto&, FrameGraphPassResources&, void* ctxPtr) {
                VULTRA_SCOPED_FRAMEGRAPH_EXEC_CONTEXT(rc, ctxPtr);
                setRenderDevice(rc.rd);
                if (!rc.ext.builtinShaderLib)
                {
                    return;
                }
                setShaderLib(*rc.ext.builtinShaderLib);

                RHI_GPU_ZONE(rc.cb, PASS_NAME);

                assert(rc.framebufferInfo().has_value());
                const auto* pipeline =
                    getPipeline(rhi::getColorFormat(rc.framebufferInfo().value(), 0), useMultiview, temporalEnabled);
                if (!pipeline)
                {
                    return;
                }

                rc.overrideSampler(rc.resourceSet[3][0], rc.ext.samplers["nearest"]);
                if (temporalEnabled)
                    rc.overrideSampler(rc.resourceSet[3][1], rc.ext.samplers["nearest"]);
                rc.cb.bindPipeline(*pipeline);
                rc.bindDescriptorSets(*pipeline);
                auto framebufferInfo = rc.framebufferInfo().value();
                if (useMultiview)
                {
                    framebufferInfo.layers   = 2u;
                    framebufferInfo.viewMask = 0x3u;
                }
                rc.cb.pushConstants(rhi::ShaderStages::eFragment, 0, &pushConstants);
                rc.cb.beginRendering(framebufferInfo).drawFullScreenTriangle().endRendering();
            });

        if (temporalEnabled && temporalHistoryWrite)
        {
            m_TemporalHistoryReadIndex = temporalWriteIndex;
            m_TemporalHistoryValid = true;
        }

        return target;
    }

    rhi::GraphicsPipeline FinalCompositionPass::createPipeline(const rhi::PixelFormat colorFormat,
                                                               const bool             useMultiview,
                                                               const bool             temporalFilter) const
    {
        auto vertexShader = loadGeneralShader("fullscreen_triangle.vert", vshadersystem::ShaderStage::eVert);
        if (!vertexShader)
        {
            VULTRA_CORE_ERROR("[FinalCompositionPass] Failed to load vertex shader variant");
            return {};
        }

        const bool manualSrgbEncode = !isSrgbColorFormat(colorFormat);
        rhi::ShaderLibraryRuntime::KeywordValues fragmentKeywords {
            {"MANUAL_SRGB_ENCODE", manualSrgbEncode ? 1u : 0u},
            {"USE_MULTIVIEW", useMultiview ? 1u : 0u},
            {"USE_TEMPORAL_FILTER", temporalFilter ? 1u : 0u},
        };

        auto fragmentShader = loadGeneralShader("final_composition.frag", vshadersystem::ShaderStage::eFrag, fragmentKeywords);
        if (!fragmentShader)
        {
            VULTRA_CORE_ERROR("[FinalCompositionPass] Failed to load fragment shader variant");
            return {};
        }

        auto builder = rhi::GraphicsPipeline::Builder {};
        builder.setViewMask(useMultiview ? 0x3u : 0u)
            .setColorFormats(temporalFilter ? std::vector<rhi::PixelFormat> {colorFormat, rhi::PixelFormat::eRGBA16F} :
                                               std::vector<rhi::PixelFormat> {colorFormat})
            .setInputAssembly({})
            .setDepthStencil({
                .depthTest  = false,
                .depthWrite = false,
            })
            .setRasterizer({
                .polygonMode = rhi::PolygonMode::eFill,
                .cullMode    = rhi::CullMode::eNone,
            })
            .setBlending(0, {.enabled = false});

        const bool webgpu = getRenderDevice().getBackendApi() == rhi::RenderBackendApi::eWebGPU;
        if (webgpu)
        {
            builder
                .addShader(rhi::ShaderType::eVertex,
                           {
                               .code           = vertexShader->wgsl,
                               .entryPointName = "main",
                               .defines        = {},
                               .reflection     = vertexShader->reflection,
                           })
                .addShader(rhi::ShaderType::eFragment,
                           {
                               .code           = fragmentShader->wgsl,
                               .entryPointName = "main",
                               .defines        = {},
                               .reflection     = fragmentShader->reflection,
                           });
        }
        else
        {
            builder.addBuiltinShader(rhi::ShaderType::eVertex, vertexShader->spirv)
                .addBuiltinShader(rhi::ShaderType::eFragment, fragmentShader->spirv);
        }

        return builder.build(getRenderDevice());
    }

    void FinalCompositionPass::ensureTemporalHistory(rhi::RenderDevice& rd, const rhi::Extent2D extent)
    {
        const bool needsHistory =
            !m_TemporalHistory[0] || !m_TemporalHistory[1] ||
            m_TemporalHistoryExtent.width != extent.width ||
            m_TemporalHistoryExtent.height != extent.height;
        if (!needsHistory)
            return;

        m_TemporalHistoryExtent = extent;
        for (auto& history : m_TemporalHistory)
        {
            history = createRef<rhi::Texture>(
                rd.createTexture2D(extent,
                                   rhi::PixelFormat::eRGBA16F,
                                   1u,
                                   1u,
                                   rhi::ImageUsage::eRenderTarget | rhi::ImageUsage::eSampled));
        }
        m_TemporalHistoryReadIndex = 0u;
        m_TemporalHistoryValid = false;
    }

    void FinalCompositionPass::resetTemporalHistory()
    {
        m_TemporalHistoryValid = false;
    }
} // namespace vultra

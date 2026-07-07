#include "vultra/function/rendering/srp/builtin/passes/general_gaussian_splat_temporal_filter_pass.hpp"

#include "vultra/core/rhi/command_buffer.hpp"
#include "vultra/core/rhi/texture.hpp"
#include "vultra/function/framegraph/framegraph_import.hpp"
#include "vultra/function/framegraph/framegraph_resource_access.hpp"
#include "vultra/function/framegraph/framegraph_texture.hpp"
#include "vultra/function/resource/gpu_scene_view.hpp"

#include <algorithm>
#include <cmath>
#include <fg/FrameGraph.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <vector>

namespace vultra
{
    namespace
    {
        constexpr auto PASS_NAME = "GeneralGaussianSplatTemporalFilterPass";

        struct TemporalFilterPushConstants
        {
            glm::vec4 targetSize {1.0f, 1.0f, 0.0f, 0.0f};
            glm::vec4 temporalGazeAndRings {0.5f, 0.5f, 32.0f, 56.0f};
            glm::vec4 temporalParams {1.0f, 1.0f, 1.0f, 0.18f};
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
            if (useMultiview || !view.gpuSceneView)
                return false;
            return view.gpuSceneView->generalGaussianSplatPeripheralTemporalFilterEnabled;
        }

        [[nodiscard]] TemporalFilterPushConstants makePushConstants(const RenderView&           view,
                                                                    const rhi::RenderBackendApi backendApi,
                                                                    const bool                  historyValid)
        {
            const auto* gpuSceneView = view.gpuSceneView;
            const glm::vec2 tanHalf = tanHalfFov(view);
            const float ySign = projectionYSign(view, backendApi);
            const float midDegrees = gpuSceneView ?
                                         std::max(gpuSceneView->generalGaussianSplatFoveatedRingDegrees.y,
                                                  gpuSceneView->generalGaussianSplatFoveatedRingDegrees.x + 1.0f) :
                                         32.0f;
            const float outerDegrees = gpuSceneView ?
                                           std::max(gpuSceneView->generalGaussianSplatPeripheralTemporalFilterOuterDegrees,
                                                    midDegrees + 1.0f) :
                                           56.0f;

            TemporalFilterPushConstants pc {};
            pc.targetSize = glm::vec4 {static_cast<float>(std::max(view.extent.width, 1u)),
                                       static_cast<float>(std::max(view.extent.height, 1u)),
                                       0.0f,
                                       0.0f};
            if (gpuSceneView)
            {
                pc.temporalGazeAndRings =
                    glm::vec4 {gpuSceneView->generalGaussianSplatFoveatedGaze.x,
                               gpuSceneView->generalGaussianSplatFoveatedGaze.y,
                               midDegrees,
                               outerDegrees};
                pc.temporalParams =
                    glm::vec4 {tanHalf.x,
                               tanHalf.y,
                               ySign,
                               std::max(gpuSceneView->generalGaussianSplatPeripheralTemporalFilterRejectionThreshold,
                                        1e-4f)};
                pc.temporalParams2 =
                    glm::vec4 {std::max(gpuSceneView->generalGaussianSplatPeripheralTemporalFilterClampRadius, 0.0f),
                               std::clamp(gpuSceneView->generalGaussianSplatPeripheralTemporalFilterLambdaScale,
                                          0.0f,
                                          1.0f),
                               historyValid ? 1.0f : 0.0f,
                               0.0f};
            }
            return pc;
        }
    } // namespace

    GeneralGaussianSplatTemporalFilterPass::GeneralGaussianSplatTemporalFilterPass()
    {
        setShaderProfile(rhi::ShaderProfile::eGeneral);
    }

    FrameGraphResource GeneralGaussianSplatTemporalFilterPass::filter(FrameGraphBuildContext& ctx,
                                                                      FrameGraphResource      source)
    {
        if (!source)
            return source;

        const bool useMultiview = ctx.view().enableMultiview && ctx.view().multiviewCameraCount == 2u;
        const bool temporalEnabled = temporalFilterEnabled(ctx.view(), useMultiview);
        if (!temporalEnabled)
        {
            resetTemporalHistory();
            return source;
        }

        ensureTemporalHistory(ctx.rd, ctx.view().extent);
        if (!m_TemporalHistory[0] || !m_TemporalHistory[1])
            return source;

        const bool historyValid = m_TemporalHistoryValid;
        const uint32_t temporalWriteIndex = 1u - m_TemporalHistoryReadIndex;
        const auto temporalHistoryRead =
            framegraph::importTexture(ctx.fg, "M3GaussianTemporalHistoryRead", m_TemporalHistory[m_TemporalHistoryReadIndex].get());
        const auto temporalHistoryWrite =
            framegraph::importTexture(ctx.fg, "M3GaussianTemporalHistoryWrite", m_TemporalHistory[temporalWriteIndex].get());
        const auto pushConstants = makePushConstants(ctx.view(), ctx.rd.getBackendApi(), historyValid);
        const auto resolution = ctx.view().extent;

        struct PassData
        {
            FrameGraphResource source;
            FrameGraphResource historyRead;
            FrameGraphResource historyWrite;
            FrameGraphResource color;
        };

        auto data = ctx.fg.addCallbackPass<PassData>(
            PASS_NAME,
            [source, temporalHistoryRead, temporalHistoryWrite, resolution, useMultiview](
                FrameGraph::Builder& builder, PassData& data) {
                PASS_SETUP_ZONE;

                data.source =
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
                data.historyRead =
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
                data.color = builder.create<framegraph::FrameGraphTexture>(
                    "General Gaussian Peripheral Temporal Color",
                    {
                        .extent     = resolution,
                        .format     = rhi::PixelFormat::eRGBA8_UNorm,
                        .layers     = useMultiview ? 2u : 0u,
                        .usageFlags = rhi::ImageUsage::eRenderTarget | rhi::ImageUsage::eSampled,
                    });
                data.color = builder.write(data.color,
                                           framegraph::Attachment {
                                               .index       = 0,
                                               .imageAspect = rhi::ImageAspect::eColor,
                                               .clearValue  = framegraph::ClearValue::eOpaqueBlack,
                                           });
                data.historyWrite =
                    builder.write(temporalHistoryWrite,
                                  framegraph::Attachment {
                                      .index       = 1,
                                      .imageAspect = rhi::ImageAspect::eColor,
                                  });
            },
            [this, useMultiview, pushConstants](const PassData&, FrameGraphPassResources&, void* ctxPtr) {
                VULTRA_SCOPED_FRAMEGRAPH_EXEC_CONTEXT(rc, ctxPtr);
                setRenderDevice(rc.rd);
                if (!rc.ext.builtinShaderLib)
                    return;
                setShaderLib(*rc.ext.builtinShaderLib);

                RHI_GPU_ZONE(rc.cb, PASS_NAME);
                if (!rc.framebufferInfo().has_value())
                    return;

                const auto* pipeline = getPipeline(rhi::getColorFormat(rc.framebufferInfo().value(), 0), useMultiview);
                if (!pipeline)
                    return;

                rc.overrideSampler(rc.resourceSet[3][0], rc.ext.samplers["nearest"]);
                rc.overrideSampler(rc.resourceSet[3][1], rc.ext.samplers["nearest"]);

                auto framebufferInfo = rc.framebufferInfo().value();
                if (useMultiview)
                {
                    framebufferInfo.layers   = 2u;
                    framebufferInfo.viewMask = 0x3u;
                }

                rc.cb.bindPipeline(*pipeline);
                rc.bindDescriptorSets(*pipeline);
                rc.cb.pushConstants(rhi::ShaderStages::eFragment, 0, &pushConstants);
                rc.cb.beginRendering(framebufferInfo).drawFullScreenTriangle().endRendering();
            });

        m_TemporalHistoryReadIndex = temporalWriteIndex;
        m_TemporalHistoryValid = true;
        return data.color;
    }

    rhi::GraphicsPipeline GeneralGaussianSplatTemporalFilterPass::createPipeline(const rhi::PixelFormat colorFormat,
                                                                                 const bool             useMultiview) const
    {
        auto vertexShader = loadGeneralShader("fullscreen_triangle.vert", vshadersystem::ShaderStage::eVert);
        if (!vertexShader)
            return {};

        rhi::ShaderLibraryRuntime::KeywordValues fragmentKeywords {
            {"USE_MULTIVIEW", useMultiview ? 1u : 0u},
        };
        auto fragmentShader = loadGeneralShader("gaussian_splat_peripheral_temporal_filter.frag",
                                                vshadersystem::ShaderStage::eFrag,
                                                fragmentKeywords);
        if (!fragmentShader)
            return {};

        auto builder = rhi::GraphicsPipeline::Builder {};
        builder.setViewMask(useMultiview ? 0x3u : 0u)
            .setColorFormats(std::vector<rhi::PixelFormat> {colorFormat, rhi::PixelFormat::eRGBA16F})
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

    void GeneralGaussianSplatTemporalFilterPass::ensureTemporalHistory(rhi::RenderDevice& rd,
                                                                       const rhi::Extent2D extent)
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

    void GeneralGaussianSplatTemporalFilterPass::resetTemporalHistory()
    {
        m_TemporalHistoryValid = false;
    }
} // namespace vultra

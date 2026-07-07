#include "vultra/function/rendering/srp/builtin/passes/general_gaussian_splat_preprocess_pass.hpp"
#include "vultra/core/rhi/command_buffer.hpp"
#include "vultra/function/framegraph/framegraph_buffer.hpp"
#include "vultra/function/framegraph/framegraph_import.hpp"
#include "vultra/function/framegraph/framegraph_resource_access.hpp"
#include "vultra/function/rendering/srp/builtin/resource_keys.hpp"

#include <glm/vec4.hpp>

#include <fg/FrameGraph.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <initializer_list>
#include <vector>

namespace vultra
{
    namespace
    {
        constexpr auto PASS_NAME = "GeneralGaussianSplatPreprocessPass";
        constexpr auto kStereoCameraBinding = 23u;
        constexpr auto kFoveatedLayerCount = resource::kGeneralGaussianSplatFoveatedLayerCount;
        constexpr auto kFoveatedClodEnabledFlag = 1u;
        constexpr auto kFoveatedClodCoverageCompensationFlag = 8u;
        constexpr auto kFoveatedClodTemporalHysteresisFlag = 16u;
        constexpr auto kFoveatedClodBoundarySmoothingFlag = 32u;
        constexpr auto kFoveatedClodScoreSelectedFlag = 64u;
        constexpr auto kFoveatedShSmoothSuppressionFlag = 2u;
        constexpr auto kCoverageStableReleaseMode = 8u;
        constexpr auto kCoverageReleaseFlagCoverageFloor = 1u;
        constexpr auto kCoverageReleaseFlagDiagnostics = 8u;
        constexpr auto kCoverageReleaseFlagTextureFloor = 32u;
        constexpr auto kCoverageReleaseFlagTextureUpdateThisFrame = 64u;
        constexpr auto kCoverageReleaseFlagGuideModifier = 128u;
        constexpr auto kCoverageReleaseFlagGuideHistoryValid = 256u;
        constexpr auto kCoverageReleasePhaseResolve = 0x40000000u;
        constexpr auto kCoverageReleasePhaseAccumulate = 0x80000000u;
        constexpr auto kCoverageReleaseTileCounterOffset = 256u;
        constexpr auto kCoverageReleaseMaxTileCount = 512u;
        constexpr auto kReleaseCostSummaryOffset =
            kCoverageReleaseTileCounterOffset + kCoverageReleaseMaxTileCount * 5u;
        constexpr auto kCoverageReleaseTileCounterClearCount =
            kReleaseCostSummaryOffset + 16u;
        constexpr auto kEcsptCounterStorageCount = 4096u;
        constexpr auto kShBandL1Binding = 47u;
        constexpr auto kShBandL2Binding = 48u;
        constexpr auto kShBandL3Binding = 49u;
        constexpr std::array<uint32_t, kFoveatedLayerCount> kFoveatedVisibleSplatBindings {31u, 32u, 33u};
        constexpr std::array<uint32_t, kFoveatedLayerCount> kFoveatedSortKeyBindings {34u, 35u, 36u};
        constexpr std::array<uint32_t, kFoveatedLayerCount> kFoveatedSortIndexBindings {37u, 38u, 39u};
        constexpr std::array<uint32_t, kFoveatedLayerCount> kFoveatedVisibleCountBindings {40u, 41u, 42u};
        constexpr std::array<uint32_t, kFoveatedLayerCount> kFoveatedIndirectBindings {43u, 44u, 45u};

        struct GeneralGaussianSplatPreprocessPushConstants
        {
            uint32_t pointCount {0};
            uint32_t maxVisibleSplats {0};
            uint32_t rankTotalCount {0};
            uint32_t foveatedClodEnabled {0};
            glm::vec4 foveatedGazeAndRings {0.5f, 0.5f, 12.0f, 32.0f};
            glm::vec4 foveatedLevelsAndTransition {1.0f, 0.40f, 0.15f, 8.0f};
            glm::uvec4 foveatedLayerParams {0u, 0u, 0u, 0u};
            glm::vec4 foveatedCoverageGuardParams {0.05f, 0.75f, 0.45f, 0.15f};
            glm::vec4 foveatedTemporalParams {0.25f, 0.18f, 6.0f, 0.0f};
            glm::vec4 foveatedContinuousParams {24.0f, 2.0f, 0.12f, 0.0f};
            glm::uvec4 shLodParams {0u, 3u, 3u, 3u};
            glm::vec4 shGuardParams {0.05f, 0.20f, 0.0f, 0.0f};
            glm::uvec4 shaderAntiPopParams {0u, 0u, 0u, 0u};
            glm::vec4 shaderAntiPopFloats {0.05f, 0.0f, 0.0f, 0.0f};
            glm::uvec4 gazeAnchorParams {0u, 4u, 2u, 0u};
            glm::vec4 gazeAnchorGazes {0.5f, 0.5f, 0.5f, 0.5f};
            glm::vec4 gazeAnchorFloats {1.0f, -1.0f, 0.0f, 0.0f};
            glm::uvec4 coverageReleaseParams {0u, 16u, 8u, 0u};
            glm::vec4 coverageReleaseFloats {0.7743f, 4.0f, 0.65f, 1e-4f};
            glm::vec4 coverageTextureFloats {0.25f, 0.80f, 1.0f, 0.0f};
            glm::uvec4 coverageReleaseCostParams {0u, 0u, 0u, 0u};
            glm::vec4 coverageReleaseCostFloats {0.12f, 0.35f, 0.55f, 48.0f};
        };

        struct GeneralGaussianSplatFoveatedClodPushParams
        {
            uint32_t  enabled {0};
            uint32_t  distribution {2};
            uint32_t  coverageGuardMode {1u};
            float     coverageProtectionDegrees {0.0f};
            float     coverageGuardBudgetRatio {0.05f};
            glm::vec3 coverageGuardMinLevels {0.75f, 0.45f, 0.15f};
            uint32_t  coverageGuardRiskSectors {0x7u};
            float     continuousTheta0Degrees {24.0f};
            float     continuousAlpha {2.0f};
            float     continuousMinLevel {0.12f};
            float     temporalPeripheralScale {0.0f};
            float     temporalHysteresisRatio {0.25f};
            float     boundarySmoothingRatio {0.18f};
            uint32_t  temporalResidencyFrames {6u};
            glm::vec4 gazeAndRings {0.5f, 0.5f, 12.0f, 32.0f};
            glm::vec4 levelsAndTransition {1.0f, 0.40f, 0.15f, 8.0f};
            glm::uvec4 shLodParams {0u, 3u, 3u, 3u};
            glm::vec4 shGuardParams {0.05f, 0.20f, 0.0f, 0.0f};
            glm::uvec4 shaderAntiPopParams {0u, 0u, 0u, 0u};
            glm::vec4 shaderAntiPopFloats {0.05f, 0.0f, 0.0f, 0.0f};
            glm::uvec4 gazeAnchorParams {0u, 4u, 2u, 0u};
            glm::vec4 gazeAnchorGazes {0.5f, 0.5f, 0.5f, 0.5f};
            glm::vec4 gazeAnchorFloats {1.0f, -1.0f, 0.0f, 0.0f};
            glm::uvec4 coverageReleaseParams {0u, 16u, 8u, 0u};
            glm::vec4 coverageReleaseFloats {0.7743f, 4.0f, 0.65f, 1e-4f};
            glm::vec4 coverageTextureFloats {0.25f, 0.80f, 1.0f, 0.0f};
            glm::uvec4 coverageReleaseCostParams {0u, 0u, 0u, 0u};
            glm::vec4 coverageReleaseCostFloats {0.12f, 0.35f, 0.55f, 48.0f};
        };

        GeneralGaussianSplatFoveatedClodPushParams makeFoveatedClodPushParams(
            const resource::GpuSceneView& gpuSceneView)
        {
            GeneralGaussianSplatFoveatedClodPushParams params {};
            params.enabled = gpuSceneView.generalGaussianSplatFoveatedClodEnabled ? kFoveatedClodEnabledFlag : 0u;
            params.distribution = gpuSceneView.generalGaussianSplatFoveatedDistribution;
            params.coverageGuardMode =
                std::min(gpuSceneView.generalGaussianSplatFoveatedCoverageGuardMode, 3u);
            params.coverageProtectionDegrees =
                std::max(gpuSceneView.generalGaussianSplatFoveatedCoverageProtectionDegrees, 0.0f);
            params.coverageGuardBudgetRatio =
                std::clamp(gpuSceneView.generalGaussianSplatFoveatedCoverageGuardBudgetRatio, 0.0f, 0.25f);
            params.coverageGuardMinLevels =
                glm::clamp(gpuSceneView.generalGaussianSplatFoveatedCoverageGuardMinLevels,
                           glm::vec3 {0.0f},
                           glm::vec3 {1.0f});
            params.coverageGuardRiskSectors =
                gpuSceneView.generalGaussianSplatFoveatedCoverageGuardRiskSectors & 0x7u;
            params.continuousTheta0Degrees =
                std::max(gpuSceneView.generalGaussianSplatFoveatedContinuousTheta0Degrees, 1e-4f);
            params.continuousAlpha =
                std::max(gpuSceneView.generalGaussianSplatFoveatedContinuousAlpha, 0.0f);
            params.continuousMinLevel =
                std::clamp(gpuSceneView.generalGaussianSplatFoveatedContinuousMinLevel, 0.0f, 1.0f);
            params.temporalPeripheralScale =
                std::max(gpuSceneView.generalGaussianSplatFoveatedTemporalPeripheralScale, 0.0f);
            if (gpuSceneView.generalGaussianSplatFoveatedClodEnabled &&
                gpuSceneView.generalGaussianSplatFoveatedCoverageCompensationEnabled)
                params.enabled |= kFoveatedClodCoverageCompensationFlag;
            if (gpuSceneView.generalGaussianSplatFoveatedClodEnabled &&
                gpuSceneView.generalGaussianSplatFoveatedTemporalHysteresisEnabled)
                params.enabled |= kFoveatedClodTemporalHysteresisFlag;
            if (gpuSceneView.generalGaussianSplatFoveatedClodEnabled &&
                gpuSceneView.generalGaussianSplatFoveatedBoundarySmoothingEnabled)
                params.enabled |= kFoveatedClodBoundarySmoothingFlag;
            if (gpuSceneView.generalGaussianSplatFoveatedClodEnabled &&
                gpuSceneView.generalGaussianSplatFoveatedScoreSelected)
                params.enabled |= kFoveatedClodScoreSelectedFlag;
            params.temporalHysteresisRatio =
                std::clamp(gpuSceneView.generalGaussianSplatFoveatedTemporalHysteresisRatio, 0.0f, 1.0f);
            params.boundarySmoothingRatio =
                std::clamp(gpuSceneView.generalGaussianSplatFoveatedBoundarySmoothingRatio, 0.0f, 1.0f);
            params.temporalResidencyFrames =
                std::clamp(gpuSceneView.generalGaussianSplatFoveatedTemporalResidencyFrames, 0u, 255u);
            params.gazeAndRings = glm::vec4 {gpuSceneView.generalGaussianSplatFoveatedGaze.x,
                                             gpuSceneView.generalGaussianSplatFoveatedGaze.y,
                                             gpuSceneView.generalGaussianSplatFoveatedRingDegrees.x,
                                             gpuSceneView.generalGaussianSplatFoveatedRingDegrees.y};
            params.levelsAndTransition =
                glm::vec4 {gpuSceneView.generalGaussianSplatFoveatedRingLevels.x,
                           gpuSceneView.generalGaussianSplatFoveatedRingLevels.y,
                           gpuSceneView.generalGaussianSplatFoveatedRingLevels.z,
                           std::max(gpuSceneView.generalGaussianSplatFoveatedTransitionDegrees, 0.0f)};
            const auto shDegrees =
                glm::clamp(gpuSceneView.generalGaussianSplatFoveatedShLodDegrees,
                           glm::uvec3 {0u},
                           glm::uvec3 {3u});
            params.shLodParams =
                glm::uvec4 {gpuSceneView.generalGaussianSplatFoveatedShLodEnabled ? 1u : 0u,
                            shDegrees.x,
                            shDegrees.y,
                            shDegrees.z};
            if (gpuSceneView.generalGaussianSplatFoveatedShSmoothSuppressionEnabled)
                params.shLodParams.x |= kFoveatedShSmoothSuppressionFlag;
            const uint32_t shGuardMode =
                gpuSceneView.generalGaussianSplatFoveatedShLodEnabled ?
                    std::min(gpuSceneView.generalGaussianSplatFoveatedShLodGuardMode, 3u) :
                    0u;
            params.shGuardParams =
                glm::vec4 {std::max(gpuSceneView.generalGaussianSplatFoveatedShLodGuardThresholdMid, 0.0f),
                           std::max(gpuSceneView.generalGaussianSplatFoveatedShLodGuardThresholdHigh,
                                    gpuSceneView.generalGaussianSplatFoveatedShLodGuardThresholdMid),
                           static_cast<float>(shGuardMode),
                           0.0f};
            params.shaderAntiPopParams =
                glm::uvec4 {gpuSceneView.generalGaussianSplatShaderAntiPopMode,
                            gpuSceneView.generalGaussianSplatShaderAntiPopHashSeed,
                            gpuSceneView.generalGaussianSplatShaderAntiPopPKeepCurve,
                            gpuSceneView.generalGaussianSplatShaderAntiPopNormalizeMode};
            params.shaderAntiPopFloats =
                glm::vec4 {std::clamp(gpuSceneView.generalGaussianSplatShaderAntiPopRampWidth, 0.0f, 1.0f),
                           std::clamp(gpuSceneView.generalGaussianSplatShaderAntiPopGuardThreshold, 0.0f, 1.0f),
                           std::clamp(gpuSceneView.generalGaussianSplatShaderAntiPopGuardFloor, 0.0f, 1.0f),
                           std::clamp(gpuSceneView.generalGaussianSplatShaderAntiPopNormalizeFactor,
                                      gpuSceneView.generalGaussianSplatShaderAntiPopNormalizeClampMin,
                                      gpuSceneView.generalGaussianSplatShaderAntiPopNormalizeClampMax)};
            params.gazeAnchorParams =
                glm::uvec4 {gpuSceneView.generalGaussianSplatGazeAnchorFlags,
                            gpuSceneView.generalGaussianSplatGazeAnchorFadeFrames,
                            gpuSceneView.generalGaussianSplatGazeAnchorMinUpdateFrames,
                            0u};
            params.gazeAnchorGazes =
                glm::vec4 {gpuSceneView.generalGaussianSplatGazeAnchorOldGaze.x,
                           gpuSceneView.generalGaussianSplatGazeAnchorOldGaze.y,
                           gpuSceneView.generalGaussianSplatGazeAnchorNewGaze.x,
                           gpuSceneView.generalGaussianSplatGazeAnchorNewGaze.y};
            params.gazeAnchorFloats =
                glm::vec4 {std::clamp(gpuSceneView.generalGaussianSplatGazeAnchorFadePhase, 0.0f, 1.0f),
                           gpuSceneView.generalGaussianSplatGazeAnchorTransitionBudgetRatio,
                           gpuSceneView.generalGaussianSplatGazeAnchorAux0,
                           gpuSceneView.generalGaussianSplatGazeAnchorAux1};
            params.coverageReleaseParams =
                glm::uvec4 {gpuSceneView.generalGaussianSplatCoverageStableReleaseFlags,
                            gpuSceneView.generalGaussianSplatCoverageStableReleaseTileGridX,
                            gpuSceneView.generalGaussianSplatCoverageStableReleaseTileGridY,
                            gpuSceneView.generalGaussianSplatCoverageStableReleaseFrameIndex};
            params.coverageReleaseFloats =
                glm::vec4 {gpuSceneView.generalGaussianSplatCoverageStableReleaseLambda,
                           gpuSceneView.generalGaussianSplatCoverageStableReleaseDMin,
                           gpuSceneView.generalGaussianSplatCoverageStableReleaseSigmaMax,
                           gpuSceneView.generalGaussianSplatCoverageStableReleaseReleaseEpsilon};
            params.coverageTextureFloats =
                glm::vec4 {gpuSceneView.generalGaussianSplatCoverageTextureStrength,
                           gpuSceneView.generalGaussianSplatCoverageTextureHistoryBeta,
                           static_cast<float>(
                               std::max(gpuSceneView.generalGaussianSplatCoverageTextureUpdateInterval, 1u)),
                           gpuSceneView.generalGaussianSplatShaderAntiPopPrefixRatio > 0.0f ?
                               gpuSceneView.generalGaussianSplatShaderAntiPopPrefixRatio :
                               1.0f};
            params.coverageReleaseCostParams =
                glm::uvec4 {gpuSceneView.generalGaussianSplatDelayedGuideTemporalReleasePolicy, 0u, 0u, 0u};
            params.coverageReleaseCostFloats =
                glm::vec4 {gpuSceneView.generalGaussianSplatDelayedGuideTemporalReleaseCapRatio,
                           gpuSceneView.generalGaussianSplatDelayedGuideTemporalRiskThreshold,
                           gpuSceneView.generalGaussianSplatDelayedGuideTemporalFootprintDecayScale,
                           gpuSceneView.generalGaussianSplatDelayedGuideTemporalLargeFootprintPx};
            return params;
        }

    } // namespace

    GeneralGaussianSplatPreprocessPass::GeneralGaussianSplatPreprocessPass() { setShaderProfile(rhi::ShaderProfile::eGeneral); }

    void GeneralGaussianSplatPreprocessPass::addPass(FrameGraphBuildContext& ctx)
    {
        auto* gpuSceneView = ctx.view().gpuSceneView;
        if (!gpuSceneView)
            return;
        if (!gpuSceneView->hasGeneralGaussianSplats())
            return;

        const auto cameraBlock       = ctx.bb.get<CameraData>().cameraBlock.fgResource;
        const auto stereoCameraBlock = ctx.bb.get<CameraData>().stereoCameraBlock.fgResource;
        const bool useMultiview =
            ctx.view().enableMultiview && ctx.view().multiviewCameraCount >= 2u && static_cast<bool>(stereoCameraBlock);
        const bool useDirectPrefix = gpuSceneView->generalGaussianSplatDirectPrefix;
        const bool useFoveatedLayerOutput =
            gpuSceneView->generalGaussianSplatFoveatedLayeredCompositeEnabled;

        if (gpuSceneView->generalGaussianSplatDrawBuffer)
        {
            ctx.data.set(kResKey_GeneralGaussianSplatDrawBuffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatDrawBuffer",
                                                  gpuSceneView->generalGaussianSplatDrawBuffer.get(),
                                                  framegraph::BufferType::eStorageBuffer,
                                                  sizeof(resource::GpuGeneralGaussianSplatDrawRecord)));
        }

        if (gpuSceneView->generalGaussianSplatPackedSourceBuffer)
        {
            ctx.data.set(kResKey_GeneralGaussianSplatPackedSourceBuffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatPackedSourceBuffer",
                                                  gpuSceneView->generalGaussianSplatPackedSourceBuffer.get(),
                                                  framegraph::BufferType::eStorageBuffer,
                                                  sizeof(resource::GpuGeneralGaussianSplatPackedSource)));
        }

        if (!useDirectPrefix && gpuSceneView->generalGaussianSplatSelectedSourceBuffer)
        {
            ctx.data.set(kResKey_GeneralGaussianSplatSelectedSourceBuffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatSelectedSourceBuffer",
                                                  gpuSceneView->generalGaussianSplatSelectedSourceBuffer.get(),
                                                  framegraph::BufferType::eStorageBuffer,
                                                  sizeof(resource::GpuGeneralGaussianSplatSelectedSource)));
        }

        if (gpuSceneView->generalGaussianSplatVisibleSplatBuffer)
        {
            ctx.data.set(kResKey_GeneralGaussianSplatVisibleSplatBuffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatVisibleSplatBuffer",
                                                  gpuSceneView->generalGaussianSplatVisibleSplatBuffer.get(),
                                                  framegraph::BufferType::eStorageBuffer,
                                                  sizeof(resource::GpuGeneralGaussianSplatVisibleSplat)));
        }

        if (gpuSceneView->generalGaussianSplatSortKeyBuffer)
        {
            ctx.data.set(kResKey_GeneralGaussianSplatSortKeyBuffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatSortKeyBuffer",
                                                  gpuSceneView->generalGaussianSplatSortKeyBuffer.get(),
                                                  framegraph::BufferType::eStorageBuffer,
                                                  sizeof(uint32_t)));
        }

        if (gpuSceneView->generalGaussianSplatSortIndexBuffer)
        {
            ctx.data.set(kResKey_GeneralGaussianSplatSortIndexBuffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatSortIndexBuffer",
                                                  gpuSceneView->generalGaussianSplatSortIndexBuffer.get(),
                                                  framegraph::BufferType::eStorageBuffer,
                                                  sizeof(uint32_t)));
        }

        if (gpuSceneView->generalGaussianSplatVisibleCountBuffer)
        {
            ctx.data.set(kResKey_GeneralGaussianSplatVisibleCountBuffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatVisibleCountBuffer",
                                                  gpuSceneView->generalGaussianSplatVisibleCountBuffer.get(),
                                                  framegraph::BufferType::eStorageBuffer,
                                                  sizeof(uint32_t)));
        }

        if (gpuSceneView->generalGaussianSplatEcsptCounterBuffer)
        {
            ctx.data.set(kResKey_GeneralGaussianSplatEcsptCounterBuffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatEcsptCounterBuffer",
                                                  gpuSceneView->generalGaussianSplatEcsptCounterBuffer.get(),
                                                  framegraph::BufferType::eStorageBuffer,
                                                  sizeof(uint32_t)));
        }

        if (gpuSceneView->generalGaussianSplatIndirectBuffer.has_value())
        {
            ctx.data.set(kResKey_GeneralGaussianSplatIndirectBuffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatIndirectBuffer",
                                                  &gpuSceneView->generalGaussianSplatIndirectBuffer.value(),
                                                  framegraph::BufferType::eDrawIndirectBuffer,
                                                  sizeof(rhi::DrawIndirectCommand)));
        }

        if (useFoveatedLayerOutput)
        {
            for (uint32_t layer = 0u; layer < kFoveatedLayerCount; ++layer)
            {
                if (gpuSceneView->generalGaussianSplatFoveatedVisibleSplatBuffers[layer])
                {
                    ctx.data.set(kResKey_GeneralGaussianSplatFoveatedVisibleSplatBuffers[layer],
                                 framegraph::importBuffer(
                                     ctx.fg,
                                     "GeneralGaussianSplatFoveatedVisibleSplatBuffer",
                                     gpuSceneView->generalGaussianSplatFoveatedVisibleSplatBuffers[layer].get(),
                                     framegraph::BufferType::eStorageBuffer,
                                     sizeof(resource::GpuGeneralGaussianSplatVisibleSplat)));
                }
                if (gpuSceneView->generalGaussianSplatFoveatedSortKeyBuffers[layer])
                {
                    ctx.data.set(kResKey_GeneralGaussianSplatFoveatedSortKeyBuffers[layer],
                                 framegraph::importBuffer(
                                     ctx.fg,
                                     "GeneralGaussianSplatFoveatedSortKeyBuffer",
                                     gpuSceneView->generalGaussianSplatFoveatedSortKeyBuffers[layer].get(),
                                     framegraph::BufferType::eStorageBuffer,
                                     sizeof(uint32_t)));
                }
                if (gpuSceneView->generalGaussianSplatFoveatedSortIndexBuffers[layer])
                {
                    ctx.data.set(kResKey_GeneralGaussianSplatFoveatedSortIndexBuffers[layer],
                                 framegraph::importBuffer(
                                     ctx.fg,
                                     "GeneralGaussianSplatFoveatedSortIndexBuffer",
                                     gpuSceneView->generalGaussianSplatFoveatedSortIndexBuffers[layer].get(),
                                     framegraph::BufferType::eStorageBuffer,
                                     sizeof(uint32_t)));
                }
                if (gpuSceneView->generalGaussianSplatFoveatedVisibleCountBuffers[layer])
                {
                    ctx.data.set(kResKey_GeneralGaussianSplatFoveatedVisibleCountBuffers[layer],
                                 framegraph::importBuffer(
                                     ctx.fg,
                                     "GeneralGaussianSplatFoveatedVisibleCountBuffer",
                                     gpuSceneView->generalGaussianSplatFoveatedVisibleCountBuffers[layer].get(),
                                     framegraph::BufferType::eStorageBuffer,
                                     sizeof(uint32_t)));
                }
                if (gpuSceneView->generalGaussianSplatFoveatedIndirectBuffers[layer].has_value())
                {
                    ctx.data.set(kResKey_GeneralGaussianSplatFoveatedIndirectBuffers[layer],
                                 framegraph::importBuffer(
                                     ctx.fg,
                                     "GeneralGaussianSplatFoveatedIndirectBuffer",
                                     &gpuSceneView->generalGaussianSplatFoveatedIndirectBuffers[layer].value(),
                                     framegraph::BufferType::eDrawIndirectBuffer,
                                     sizeof(rhi::DrawIndirectCommand)));
                }
            }
        }

        if (gpuSceneView->generalGaussianSplatSortStorageBuffer)
        {
            ctx.data.set(kResKey_GeneralGaussianSplatSortStorageBuffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatSortStorageBuffer",
                                                  gpuSceneView->generalGaussianSplatSortStorageBuffer.get(),
                                                  framegraph::BufferType::eStorageBuffer,
                                                  sizeof(uint32_t)));
        }

        if (gpuSceneView->generalGaussianSplatShBuffer)
        {
            ctx.data.set(kResKey_GeneralGaussianSplatShBuffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatShBuffer",
                                                  gpuSceneView->generalGaussianSplatShBuffer.get(),
                                                  framegraph::BufferType::eStorageBuffer,
                                                  sizeof(glm::uvec2)));
        }

        if (gpuSceneView->generalGaussianSplatShL1Buffer)
        {
            ctx.data.set(kResKey_GeneralGaussianSplatShL1Buffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatShL1Buffer",
                                                  gpuSceneView->generalGaussianSplatShL1Buffer.get(),
                                                  framegraph::BufferType::eStorageBuffer,
                                                  sizeof(glm::uvec2)));
        }

        if (gpuSceneView->generalGaussianSplatShL2Buffer)
        {
            ctx.data.set(kResKey_GeneralGaussianSplatShL2Buffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatShL2Buffer",
                                                  gpuSceneView->generalGaussianSplatShL2Buffer.get(),
                                                  framegraph::BufferType::eStorageBuffer,
                                                  sizeof(glm::uvec2)));
        }

        if (gpuSceneView->generalGaussianSplatShL3Buffer)
        {
            ctx.data.set(kResKey_GeneralGaussianSplatShL3Buffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatShL3Buffer",
                                                  gpuSceneView->generalGaussianSplatShL3Buffer.get(),
                                                  framegraph::BufferType::eStorageBuffer,
                                                  sizeof(glm::uvec2)));
        }

        if (gpuSceneView->generalGaussianSplatShEnergyMetadataBuffer)
        {
            ctx.data.set(kResKey_GeneralGaussianSplatShEnergyMetadataBuffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatShEnergyMetadataBuffer",
                                                  gpuSceneView->generalGaussianSplatShEnergyMetadataBuffer.get(),
                                                  framegraph::BufferType::eStorageBuffer,
                                                  sizeof(glm::vec4)));
        }

        if (gpuSceneView->generalGaussianSplatTemporalStateBuffer)
        {
            ctx.data.set(kResKey_GeneralGaussianSplatTemporalStateBuffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatTemporalStateBuffer",
                                                  gpuSceneView->generalGaussianSplatTemporalStateBuffer.get(),
                                                  framegraph::BufferType::eStorageBuffer,
                                                  sizeof(uint32_t)));
        }

        if (gpuSceneView->generalGaussianSplatCoverageTextureBuffer)
        {
            ctx.data.set(kResKey_GeneralGaussianSplatCoverageTextureBuffer,
                         framegraph::importBuffer(ctx.fg,
                                                  "GeneralGaussianSplatCoverageTextureBuffer",
                                                  gpuSceneView->generalGaussianSplatCoverageTextureBuffer.get(),
                                                  framegraph::BufferType::eStorageBuffer,
                                                  sizeof(uint32_t)));
        }

        auto packedSourceBuffer = ctx.data.tryGet(kResKey_GeneralGaussianSplatPackedSourceBuffer);
        auto selectedSourceBuffer = ctx.data.tryGet(kResKey_GeneralGaussianSplatSelectedSourceBuffer);
        auto drawBuffer         = ctx.data.tryGet(kResKey_GeneralGaussianSplatDrawBuffer);
        auto visibleSplatBuffer = ctx.data.tryGet(kResKey_GeneralGaussianSplatVisibleSplatBuffer);
        auto sortKeyBuffer      = ctx.data.tryGet(kResKey_GeneralGaussianSplatSortKeyBuffer);
        auto sortIndexBuffer    = ctx.data.tryGet(kResKey_GeneralGaussianSplatSortIndexBuffer);
        auto visibleCountBuffer = ctx.data.tryGet(kResKey_GeneralGaussianSplatVisibleCountBuffer);
        auto ecsptCounterBuffer = ctx.data.tryGet(kResKey_GeneralGaussianSplatEcsptCounterBuffer);
        auto indirectBuffer     = ctx.data.tryGet(kResKey_GeneralGaussianSplatIndirectBuffer);
        auto sortStorageBuffer  = ctx.data.tryGet(kResKey_GeneralGaussianSplatSortStorageBuffer);
        auto shBuffer           = ctx.data.tryGet(kResKey_GeneralGaussianSplatShBuffer);
        auto shL1Buffer         = ctx.data.tryGet(kResKey_GeneralGaussianSplatShL1Buffer);
        auto shL2Buffer         = ctx.data.tryGet(kResKey_GeneralGaussianSplatShL2Buffer);
        auto shL3Buffer         = ctx.data.tryGet(kResKey_GeneralGaussianSplatShL3Buffer);
        auto shEnergyMetadataBuffer =
            ctx.data.tryGet(kResKey_GeneralGaussianSplatShEnergyMetadataBuffer);
        auto temporalStateBuffer = ctx.data.tryGet(kResKey_GeneralGaussianSplatTemporalStateBuffer);
        auto coverageTextureBuffer = ctx.data.tryGet(kResKey_GeneralGaussianSplatCoverageTextureBuffer);
        std::array<FrameGraphResource, kFoveatedLayerCount> foveatedVisibleSplatBuffers {};
        std::array<FrameGraphResource, kFoveatedLayerCount> foveatedSortKeyBuffers {};
        std::array<FrameGraphResource, kFoveatedLayerCount> foveatedSortIndexBuffers {};
        std::array<FrameGraphResource, kFoveatedLayerCount> foveatedVisibleCountBuffers {};
        std::array<FrameGraphResource, kFoveatedLayerCount> foveatedIndirectBuffers {};
        for (uint32_t layer = 0u; layer < kFoveatedLayerCount; ++layer)
        {
            foveatedVisibleSplatBuffers[layer] =
                ctx.data.tryGet(kResKey_GeneralGaussianSplatFoveatedVisibleSplatBuffers[layer]);
            foveatedSortKeyBuffers[layer] =
                ctx.data.tryGet(kResKey_GeneralGaussianSplatFoveatedSortKeyBuffers[layer]);
            foveatedSortIndexBuffers[layer] =
                ctx.data.tryGet(kResKey_GeneralGaussianSplatFoveatedSortIndexBuffers[layer]);
            foveatedVisibleCountBuffers[layer] =
                ctx.data.tryGet(kResKey_GeneralGaussianSplatFoveatedVisibleCountBuffers[layer]);
            foveatedIndirectBuffers[layer] =
                ctx.data.tryGet(kResKey_GeneralGaussianSplatFoveatedIndirectBuffers[layer]);
        }

        const uint32_t pointCount = gpuSceneView->activeGeneralGaussianSplatPoints;
        const uint32_t maxVisible = gpuSceneView->maxGeneralGaussianSplatVisibleSplats;
        const uint32_t rankTotalCount =
            useDirectPrefix ? gpuSceneView->maxGeneralGaussianSplatSourceCount :
                              gpuSceneView->maxGeneralGaussianSplatPoints;
        auto foveatedClodParams = makeFoveatedClodPushParams(*gpuSceneView);
        const uint32_t shGuardMode =
            static_cast<uint32_t>(std::round(std::clamp(foveatedClodParams.shGuardParams.z, 0.0f, 3.0f)));
        const bool shGuardUsesEnergy = shGuardMode == 1u || shGuardMode == 3u;
        const bool useShSplitBands =
            gpuSceneView->generalGaussianSplatShStorageLayout ==
                resource::GpuGaussianSplatShStorageLayout::eSplitBands &&
            static_cast<bool>(shL1Buffer && shL2Buffer && shL3Buffer);
        const bool hasShStorage = useShSplitBands || static_cast<bool>(shBuffer);
        const bool useShLodGuard =
            shGuardMode != 0u && hasShStorage &&
            (!shGuardUsesEnergy || static_cast<bool>(shEnergyMetadataBuffer));
        const bool useShLodEnergyGuard = useShLodGuard && shGuardUsesEnergy;
        if (!useShLodGuard)
            foveatedClodParams.shGuardParams.z = 0.0f;

        auto hasAllFoveatedLayerResources = [&]() {
            for (uint32_t layer = 0u; layer < kFoveatedLayerCount; ++layer)
            {
                if (!foveatedVisibleSplatBuffers[layer] || !foveatedSortKeyBuffers[layer] ||
                    !foveatedSortIndexBuffers[layer] || !foveatedVisibleCountBuffers[layer] ||
                    !foveatedIndirectBuffers[layer])
                {
                    return false;
                }
            }
            return true;
        };

        const bool hasOutputResources =
            useFoveatedLayerOutput ?
                hasAllFoveatedLayerResources() :
                static_cast<bool>(visibleSplatBuffer && sortKeyBuffer && sortIndexBuffer && visibleCountBuffer &&
                                  indirectBuffer);
        if (!cameraBlock || !drawBuffer || !packedSourceBuffer || !hasOutputResources || !ecsptCounterBuffer ||
            !sortStorageBuffer || !hasShStorage || !temporalStateBuffer || (!useDirectPrefix && !selectedSourceBuffer) ||
            !coverageTextureBuffer || (useShLodEnergyGuard && !shEnergyMetadataBuffer))
        {
            return;
        }

        struct PassData
        {
            FrameGraphResource camera;
            FrameGraphResource stereoCamera;
            FrameGraphResource drawBuffer;
            FrameGraphResource packedSourceBuffer;
            FrameGraphResource selectedSourceBuffer;
            FrameGraphResource visibleSplatBuffer;
            FrameGraphResource sortKeyBuffer;
            FrameGraphResource sortIndexBuffer;
            FrameGraphResource visibleCountBuffer;
            FrameGraphResource ecsptCounterBuffer;
            FrameGraphResource indirectBuffer;
            FrameGraphResource sortStorageBuffer;
            FrameGraphResource shBuffer;
            FrameGraphResource shL1Buffer;
            FrameGraphResource shL2Buffer;
            FrameGraphResource shL3Buffer;
            FrameGraphResource shEnergyMetadataBuffer;
            FrameGraphResource temporalStateBuffer;
            FrameGraphResource coverageTextureBuffer;
            std::array<FrameGraphResource, kFoveatedLayerCount> foveatedVisibleSplatBuffers;
            std::array<FrameGraphResource, kFoveatedLayerCount> foveatedSortKeyBuffers;
            std::array<FrameGraphResource, kFoveatedLayerCount> foveatedSortIndexBuffers;
            std::array<FrameGraphResource, kFoveatedLayerCount> foveatedVisibleCountBuffers;
            std::array<FrameGraphResource, kFoveatedLayerCount> foveatedIndirectBuffers;
        };

        ctx.fg.addCallbackPass<PassData>(
            PASS_NAME,
            [cameraBlock,
             stereoCameraBlock,
             useMultiview,
             drawBuffer,
             packedSourceBuffer,
             visibleSplatBuffer,
             sortKeyBuffer,
             sortIndexBuffer,
             visibleCountBuffer,
             ecsptCounterBuffer,
             indirectBuffer,
             sortStorageBuffer,
             shBuffer,
             shL1Buffer,
             shL2Buffer,
             shL3Buffer,
             shEnergyMetadataBuffer,
             temporalStateBuffer,
             coverageTextureBuffer,
             foveatedVisibleSplatBuffers,
             foveatedSortKeyBuffers,
             foveatedSortIndexBuffers,
             foveatedVisibleCountBuffers,
             foveatedIndirectBuffers,
             selectedSourceBuffer,
             useDirectPrefix,
             useFoveatedLayerOutput,
             useShSplitBands,
             useShLodEnergyGuard](FrameGraph::Builder& builder, PassData& data) {
                PASS_SETUP_ZONE;

                data.camera             = builder.read(cameraBlock,
                                           framegraph::BindingInfo {
                                                           .location      = {.set = 0, .binding = 0},
                                                           .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                           });
                if (useMultiview)
                {
                    data.stereoCamera = builder.read(stereoCameraBlock,
                                                     framegraph::BindingInfo {
                                                         .location      = {.set = 0, .binding = kStereoCameraBinding},
                                                         .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                                     });
                }
                data.drawBuffer         = builder.read(drawBuffer,
                                               framegraph::BindingInfo {
                                                           .location      = {.set = 0, .binding = 13},
                                                           .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                               });
                data.packedSourceBuffer = builder.read(packedSourceBuffer,
                                                       framegraph::BindingInfo {
                                                           .location      = {.set = 0, .binding = 14},
                                                           .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                                       });
                if (!useDirectPrefix)
                {
                    data.selectedSourceBuffer =
                        builder.read(selectedSourceBuffer,
                                     framegraph::BindingInfo {
                                         .location      = {.set = 0, .binding = 27},
                                         .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                     });
                }
                if (useFoveatedLayerOutput)
                {
                    for (uint32_t layer = 0u; layer < kFoveatedLayerCount; ++layer)
                    {
                        data.foveatedVisibleSplatBuffers[layer] =
                            builder.write(foveatedVisibleSplatBuffers[layer],
                                          framegraph::BindingInfo {
                                              .location      = {.set = 0, .binding = kFoveatedVisibleSplatBindings[layer]},
                                              .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                          });
                        data.foveatedSortKeyBuffers[layer] =
                            builder.write(foveatedSortKeyBuffers[layer],
                                          framegraph::BindingInfo {
                                              .location      = {.set = 0, .binding = kFoveatedSortKeyBindings[layer]},
                                              .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                          });
                        data.foveatedSortIndexBuffers[layer] =
                            builder.write(foveatedSortIndexBuffers[layer],
                                          framegraph::BindingInfo {
                                              .location      = {.set = 0, .binding = kFoveatedSortIndexBindings[layer]},
                                              .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                          });
                        data.foveatedVisibleCountBuffers[layer] =
                            builder.write(foveatedVisibleCountBuffers[layer],
                                          framegraph::BindingInfo {
                                              .location      = {.set = 0, .binding = kFoveatedVisibleCountBindings[layer]},
                                              .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                          });
                        data.foveatedIndirectBuffers[layer] =
                            builder.write(foveatedIndirectBuffers[layer],
                                          framegraph::BindingInfo {
                                              .location      = {.set = 0, .binding = kFoveatedIndirectBindings[layer]},
                                              .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                          });
                    }
                }
                else
                {
                    data.visibleSplatBuffer =
                        builder.write(visibleSplatBuffer,
                                      framegraph::BindingInfo {
                                          .location      = {.set = 0, .binding = 15},
                                          .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                      });
                    data.sortKeyBuffer =
                        builder.write(sortKeyBuffer,
                                      framegraph::BindingInfo {
                                          .location      = {.set = 0, .binding = 16},
                                          .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                      });
                    data.sortIndexBuffer =
                        builder.write(sortIndexBuffer,
                                      framegraph::BindingInfo {
                                          .location      = {.set = 0, .binding = 17},
                                          .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                      });
                    data.visibleCountBuffer =
                        builder.write(visibleCountBuffer,
                                      framegraph::BindingInfo {
                                          .location      = {.set = 0, .binding = 18},
                                          .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                      });
                    data.indirectBuffer =
                        builder.write(indirectBuffer,
                                      framegraph::BindingInfo {
                                          .location      = {.set = 0, .binding = 20},
                                          .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                      });
                }
                data.ecsptCounterBuffer =
                    builder.write(ecsptCounterBuffer,
                                  framegraph::BindingInfo {
                                      .location      = {.set = 0, .binding = 50},
                                      .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                  });
                data.sortStorageBuffer = builder.write(sortStorageBuffer,
                                                       framegraph::BindingInfo {
                                                           .location      = {.set = 0, .binding = 21},
                                                           .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                                       });
                if (useShSplitBands)
                {
                    data.shL1Buffer = builder.read(shL1Buffer,
                                                   framegraph::BindingInfo {
                                                       .location      = {.set = 0, .binding = kShBandL1Binding},
                                                       .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                                   });
                    data.shL2Buffer = builder.read(shL2Buffer,
                                                   framegraph::BindingInfo {
                                                       .location      = {.set = 0, .binding = kShBandL2Binding},
                                                       .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                                   });
                    data.shL3Buffer = builder.read(shL3Buffer,
                                                   framegraph::BindingInfo {
                                                       .location      = {.set = 0, .binding = kShBandL3Binding},
                                                       .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                                   });
                }
                else
                {
                    data.shBuffer = builder.read(shBuffer,
                                                 framegraph::BindingInfo {
                                                     .location      = {.set = 0, .binding = 22},
                                                     .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                                 });
                }
                if (useShLodEnergyGuard)
                {
                    data.shEnergyMetadataBuffer =
                        builder.read(shEnergyMetadataBuffer,
                                     framegraph::BindingInfo {
                                         .location      = {.set = 0, .binding = 28},
                                         .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                     });
                }
                data.temporalStateBuffer =
                    builder.write(temporalStateBuffer,
                                  framegraph::BindingInfo {
                                      .location      = {.set = 0, .binding = 46},
                                      .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                  });
                data.coverageTextureBuffer =
                    builder.write(coverageTextureBuffer,
                                  framegraph::BindingInfo {
                                      .location      = {.set = 0, .binding = 51},
                                      .pipelineStage = framegraph::PipelineStage::eComputeShader,
                                  });
            },
            [this,
             pointCount,
             maxVisible,
             rankTotalCount,
             foveatedClodParams,
             useMultiview,
             useDirectPrefix,
             useFoveatedLayerOutput,
             useShLodGuard,
             useShLodEnergyGuard,
             useShSplitBands](
                const PassData& data, FrameGraphPassResources& resources, void* ctxPtr) {
                VULTRA_SCOPED_FRAMEGRAPH_EXEC_CONTEXT(rc, ctxPtr);
                setRenderDevice(rc.rd);
                if (!rc.ext.builtinShaderLib)
                {
                    return;
                }
                setShaderLib(*rc.ext.builtinShaderLib);

                auto bindSubset = [&rc](const rhi::BasePipeline& pipeline, const auto& bindings) {
                    auto saved = rc.resourceSet;
                    rc.resourceSet.clear();

                    if (const auto setIt = saved.find(0); setIt != saved.end())
                    {
                        for (const auto binding : bindings)
                        {
                            if (const auto bindingIt = setIt->second.find(binding); bindingIt != setIt->second.end())
                            {
                                rc.resourceSet[0][binding] = bindingIt->second;
                            }
                        }
                    }

                    rc.bindDescriptorSets(pipeline);
                    rc.resourceSet = std::move(saved);
                };

                RHI_GPU_ZONE(rc.cb, PASS_NAME);

                auto* gpuSceneView = rc.view().gpuSceneView;
                if (!gpuSceneView || pointCount == 0u || maxVisible == 0u)
                {
                    return;
                }

                auto* visibleCountBuf = useFoveatedLayerOutput ?
                                            nullptr :
                                            resources.get<framegraph::FrameGraphBuffer>(data.visibleCountBuffer).buffer;
                auto* ecsptCounterBuf =
                    resources.get<framegraph::FrameGraphBuffer>(data.ecsptCounterBuffer).buffer;
                auto* coverageTextureBuf =
                    resources.get<framegraph::FrameGraphBuffer>(data.coverageTextureBuffer).buffer;
                const bool coverageStableRelease =
                    foveatedClodParams.shaderAntiPopParams.x == kCoverageStableReleaseMode;
                const bool coverageTextureFloorEnabled =
                    coverageStableRelease &&
                    ((foveatedClodParams.coverageReleaseParams.x & kCoverageReleaseFlagTextureFloor) != 0u);
                const bool coverageFloorEnabled =
                    coverageStableRelease &&
                    ((foveatedClodParams.coverageReleaseParams.x & kCoverageReleaseFlagCoverageFloor) != 0u);
                const bool sameFrameCoverageFloorEnabled =
                    coverageFloorEnabled && !coverageTextureFloorEnabled;
                const bool coverageReleaseDiagnosticsEnabled =
                    coverageStableRelease &&
                    ((foveatedClodParams.coverageReleaseParams.x & kCoverageReleaseFlagDiagnostics) != 0u);
                const bool fullCounterClearNeeded =
                    !coverageStableRelease || coverageReleaseDiagnosticsEnabled;
                const uint32_t coverageTextureUpdateInterval =
                    std::max(1u, static_cast<uint32_t>(std::max(foveatedClodParams.coverageTextureFloats.z, 1.0f)));
                const uint32_t coverageTextureFrameIndex =
                    foveatedClodParams.coverageReleaseParams.w & 0xffffu;
                const bool coverageTextureResetNeeded =
                    coverageTextureFloorEnabled && gpuSceneView->generalGaussianSplatCoverageTextureNeedsReset;
                const bool delayedGuideHistoryValid =
                    coverageTextureFloorEnabled &&
                    ((foveatedClodParams.coverageReleaseParams.x & kCoverageReleaseFlagGuideModifier) != 0u) &&
                    !coverageTextureResetNeeded;
                const bool coverageTextureUpdateThisFrame =
                    coverageTextureFloorEnabled &&
                    (coverageTextureResetNeeded ||
                     (coverageTextureFrameIndex % coverageTextureUpdateInterval) == 0u);
                std::array<rhi::Buffer*, kFoveatedLayerCount> foveatedVisibleCountBufs {};
                if (useFoveatedLayerOutput)
                {
                    for (uint32_t layer = 0u; layer < kFoveatedLayerCount; ++layer)
                    {
                        foveatedVisibleCountBufs[layer] =
                            resources.get<framegraph::FrameGraphBuffer>(data.foveatedVisibleCountBuffers[layer]).buffer;
                    }
                }
                {
                    RHI_GPU_ZONE(rc.cb, "GeneralGaussianSplatPreprocess::ProjectCull");

                    if (useFoveatedLayerOutput)
                    {
                        const uint32_t zero = 0u;
                        for (auto* countBuf : foveatedVisibleCountBufs)
                            rc.cb.update(*countBuf, 0u, sizeof(uint32_t), &zero);
                    }
                    else
                    {
                        const uint32_t zero = 0u;
                        rc.cb.update(*visibleCountBuf, 0u, sizeof(uint32_t), &zero);
                    }
                    if (!useFoveatedLayerOutput && (fullCounterClearNeeded || sameFrameCoverageFloorEnabled))
                    {
                        std::array<uint32_t, kEcsptCounterStorageCount> zeros {};
                        if (fullCounterClearNeeded)
                        {
                            rc.cb.update(*ecsptCounterBuf, 0u, sizeof(zeros), zeros.data());
                        }
                        else
                        {
                            rc.cb.update(*ecsptCounterBuf,
                                         0u,
                                         kCoverageReleaseTileCounterClearCount * sizeof(uint32_t),
                                         zeros.data());
                        }
                    }
                    if (!useFoveatedLayerOutput &&
                        coverageTextureFloorEnabled &&
                        coverageTextureUpdateThisFrame &&
                        coverageTextureBuf)
                    {
                        const uint32_t textureWidth =
                            std::clamp(foveatedClodParams.coverageReleaseParams.y, 1u, 1024u);
                        const uint32_t textureHeight =
                            std::clamp(foveatedClodParams.coverageReleaseParams.z, 1u, 1024u);
                        const uint32_t textureTexels = textureWidth * textureHeight;
                        static thread_local std::vector<uint32_t> coverageTextureClearScratch;
                        const size_t partialClearCount = static_cast<size_t>(textureTexels) * 2u;
                        const size_t fullClearCount = static_cast<size_t>(textureTexels) * 4u;
                        const size_t requiredClearCount =
                            coverageTextureResetNeeded ?
                                fullClearCount :
                                partialClearCount;
                        if (coverageTextureClearScratch.size() < requiredClearCount)
                            coverageTextureClearScratch.resize(requiredClearCount, 0u);
                        if (coverageTextureResetNeeded)
                        {
                            rc.cb.update(*coverageTextureBuf,
                                         0u,
                                         static_cast<uint64_t>(fullClearCount) * sizeof(uint32_t),
                                         coverageTextureClearScratch.data());
                            gpuSceneView->generalGaussianSplatCoverageTextureNeedsReset = false;
                        }
                        else
                        {
                            rc.cb.update(*coverageTextureBuf,
                                         static_cast<uint64_t>(textureTexels) * 2ull * sizeof(uint32_t),
                                         static_cast<uint64_t>(partialClearCount) * sizeof(uint32_t),
                                         coverageTextureClearScratch.data());
                        }
                    }
                    rc.cb.getBarrierBuilder().memoryBarrier(
                        {
                            .srcStage  = rhi::PipelineStages::eTransfer,
                            .srcAccess = rhi::Access::eTransferWrite,
                        },
                        {
                            .dstStage  = rhi::PipelineStages::eComputeShader,
                            .dstAccess = rhi::Access::eShaderRead | rhi::Access::eShaderWrite,
                        });

                    const bool deterministicSourceOrderSort =
                        gpuSceneView->generalGaussianSplatDeterministicSourceOrderSort;
                    const auto* preprocessPipeline =
                        getPipeline(useMultiview,
                                    useDirectPrefix,
                                    useFoveatedLayerOutput,
                                    useShLodGuard,
                                    useShLodEnergyGuard,
                                    useShSplitBands,
                                    deterministicSourceOrderSort);
                    if (!preprocessPipeline)
                    {
                        VULTRA_CORE_ERROR("[GeneralGaussianSplatPreprocessPass] Missing project/cull pipeline "
                                          "(multiview={}, directPrefix={}, layeredOutput={})",
                                          useMultiview ? 1 : 0,
                                          useDirectPrefix ? 1 : 0,
                                          useFoveatedLayerOutput ? 1 : 0);
                        return;
                    }
                    rc.cb.bindPipeline(*preprocessPipeline);
                    std::vector<uint32_t> preprocessBindings {0u, 13u, 14u, 46u};
                    if (useShSplitBands)
                    {
                        preprocessBindings.insert(preprocessBindings.end(),
                                                  {kShBandL1Binding, kShBandL2Binding, kShBandL3Binding});
                    }
                    else
                    {
                        preprocessBindings.push_back(22u);
                    }
                    if (useFoveatedLayerOutput)
                    {
                        preprocessBindings.insert(preprocessBindings.end(),
                                                  {31u, 32u, 33u, 34u, 35u, 36u, 37u,
                                                   38u, 39u, 40u, 41u, 42u, 43u, 44u, 45u});
                    }
                    else
                    {
                        preprocessBindings.insert(preprocessBindings.end(), {15u, 16u, 17u, 18u});
                    }
                    if (!useDirectPrefix)
                    {
                        preprocessBindings.push_back(27u);
                    }
                    preprocessBindings.push_back(50u);
                    preprocessBindings.push_back(51u);
                    if (useShLodEnergyGuard)
                    {
                        preprocessBindings.push_back(28u);
                    }
                    if (useMultiview)
                    {
                        preprocessBindings.push_back(kStereoCameraBinding);
                    }
                    bindSubset(*preprocessPipeline, preprocessBindings);

                    auto dispatchPreprocess = [&](const uint32_t dispatchPointCount,
                                                  const uint32_t foveatedOutputLayer,
                                                  const uint32_t coverageReleasePhase) {
                        if (dispatchPointCount == 0u)
                            return;

                        GeneralGaussianSplatPreprocessPushConstants pc {};
                        pc.pointCount            = dispatchPointCount;
                        pc.maxVisibleSplats      = maxVisible;
                        pc.rankTotalCount        = rankTotalCount;
                        pc.foveatedClodEnabled   = foveatedClodParams.enabled;
                        pc.foveatedGazeAndRings  = foveatedClodParams.gazeAndRings;
                        pc.foveatedLevelsAndTransition =
                            foveatedClodParams.levelsAndTransition;
                        pc.foveatedLayerParams =
                            glm::uvec4 {foveatedOutputLayer,
                                        foveatedClodParams.distribution,
                                        std::bit_cast<uint32_t>(foveatedClodParams.coverageProtectionDegrees),
                                        foveatedClodParams.coverageGuardMode};
                        pc.foveatedCoverageGuardParams =
                            glm::vec4 {foveatedClodParams.coverageGuardBudgetRatio,
                                       foveatedClodParams.coverageGuardMinLevels.x,
                                       foveatedClodParams.coverageGuardMinLevels.y,
                                       foveatedClodParams.coverageGuardMinLevels.z};
                        pc.foveatedTemporalParams =
                            glm::vec4 {foveatedClodParams.temporalHysteresisRatio,
                                       foveatedClodParams.boundarySmoothingRatio,
                                       static_cast<float>(foveatedClodParams.temporalResidencyFrames),
                                       std::bit_cast<float>(foveatedClodParams.coverageGuardRiskSectors)};
                        pc.foveatedContinuousParams =
                            glm::vec4 {foveatedClodParams.continuousTheta0Degrees,
                                       foveatedClodParams.continuousAlpha,
                                       foveatedClodParams.continuousMinLevel,
                                       foveatedClodParams.temporalPeripheralScale};
                        pc.shLodParams = foveatedClodParams.shLodParams;
                        pc.shGuardParams = foveatedClodParams.shGuardParams;
                        pc.shaderAntiPopParams = foveatedClodParams.shaderAntiPopParams;
                        pc.shaderAntiPopFloats = foveatedClodParams.shaderAntiPopFloats;
                        pc.gazeAnchorParams    = foveatedClodParams.gazeAnchorParams;
                        pc.gazeAnchorGazes     = foveatedClodParams.gazeAnchorGazes;
                        pc.gazeAnchorFloats    = foveatedClodParams.gazeAnchorFloats;
                        pc.coverageReleaseParams = foveatedClodParams.coverageReleaseParams;
                        if (coverageTextureUpdateThisFrame)
                            pc.coverageReleaseParams.x |= kCoverageReleaseFlagTextureUpdateThisFrame;
                        if (delayedGuideHistoryValid)
                            pc.coverageReleaseParams.x |= kCoverageReleaseFlagGuideHistoryValid;
                        pc.coverageReleaseParams.w =
                            foveatedClodParams.coverageReleaseParams.w | coverageReleasePhase;
                        pc.coverageReleaseFloats = foveatedClodParams.coverageReleaseFloats;
                        pc.coverageTextureFloats = foveatedClodParams.coverageTextureFloats;
                        pc.coverageReleaseCostParams = foveatedClodParams.coverageReleaseCostParams;
                        pc.coverageReleaseCostFloats = foveatedClodParams.coverageReleaseCostFloats;

                        rc.cb.pushConstants(rhi::ShaderStages::eCompute, 0, &pc);
                        const uint32_t dispatchX = (dispatchPointCount + 255u) / 256u;
                        rc.cb.dispatch({dispatchX, 1u, 1u});
                    };

                    if (!useFoveatedLayerOutput && sameFrameCoverageFloorEnabled)
                    {
                        RHI_GPU_ZONE(rc.cb, "GeneralGaussianSplatPreprocess::GuideAccum");
                        dispatchPreprocess(pointCount, 0u, kCoverageReleasePhaseAccumulate);
                        rc.cb.insertComputeUavBarrier();
                    }

                    if (useFoveatedLayerOutput)
                    {
                        RHI_GPU_ZONE(rc.cb, "GeneralGaussianSplatPreprocess::Selection");
                        const auto layerPointCount = [&](const float level) {
                            const float clampedLevel = std::clamp(level, 0.0f, 1.0f);
                            if (clampedLevel <= 0.0f)
                                return 0u;
                            const auto budget =
                                static_cast<uint32_t>(
                                    std::ceil(static_cast<float>(rankTotalCount) * clampedLevel));
                            return std::min(pointCount, std::max(1u, budget));
                        };

                        dispatchPreprocess(layerPointCount(foveatedClodParams.levelsAndTransition.x), 1u, 0u);
                        dispatchPreprocess(layerPointCount(foveatedClodParams.levelsAndTransition.y), 2u, 0u);
                        dispatchPreprocess(layerPointCount(foveatedClodParams.levelsAndTransition.z), 3u, 0u);
                    }
                    else
                    {
                        RHI_GPU_ZONE(rc.cb, "GeneralGaussianSplatPreprocess::Selection");
                        dispatchPreprocess(pointCount, 0u, 0u);
                    }
                    if (!useFoveatedLayerOutput && coverageTextureFloorEnabled && coverageTextureUpdateThisFrame)
                    {
                        const uint32_t textureWidth =
                            std::clamp(foveatedClodParams.coverageReleaseParams.y, 1u, 1024u);
                        const uint32_t textureHeight =
                            std::clamp(foveatedClodParams.coverageReleaseParams.z, 1u, 1024u);
                        const uint32_t textureTexels = textureWidth * textureHeight;
                        rc.cb.insertComputeUavBarrier();
                        RHI_GPU_ZONE(rc.cb, "GeneralGaussianSplatPreprocess::GuideResolve");
                        dispatchPreprocess(textureTexels, 0u, kCoverageReleasePhaseResolve);
                    }
                    rc.cb.insertComputeUavBarrier();
                }

                auto* sortStorageBuf = resources.get<framegraph::FrameGraphBuffer>(data.sortStorageBuffer).buffer;

                if (useFoveatedLayerOutput && gpuSceneView->generalGaussianSplatSorter.has_value() &&
                    static_cast<bool>(*gpuSceneView->generalGaussianSplatSorter))
                {
                    RHI_GPU_ZONE(rc.cb, "GeneralGaussianSplatPreprocess::Sort");

                    for (uint32_t layer = 0u; layer < kFoveatedLayerCount; ++layer)
                    {
                        auto* countBuf     = foveatedVisibleCountBufs[layer];
                        auto* sortKeyBuf   =
                            resources.get<framegraph::FrameGraphBuffer>(data.foveatedSortKeyBuffers[layer]).buffer;
                        auto* sortIndexBuf =
                            resources.get<framegraph::FrameGraphBuffer>(data.foveatedSortIndexBuffers[layer]).buffer;

                        gpuSceneView->generalGaussianSplatSorter->sortKeyValuesIndirect(rc.cb,
                                                                                        maxVisible,
                                                                                        *countBuf,
                                                                                        0u,
                                                                                        *sortKeyBuf,
                                                                                        0u,
                                                                                        *sortIndexBuf,
                                                                                        0u,
                                                                                        *sortStorageBuf,
                                                                                        0u);
                        rc.cb.insertComputeUavBarrier();
                    }
                }
                else if (gpuSceneView->generalGaussianSplatSorter.has_value() &&
                         static_cast<bool>(*gpuSceneView->generalGaussianSplatSorter))
                {
                    auto* sortKeyBuf   = resources.get<framegraph::FrameGraphBuffer>(data.sortKeyBuffer).buffer;
                    auto* sortIndexBuf = resources.get<framegraph::FrameGraphBuffer>(data.sortIndexBuffer).buffer;
                    RHI_GPU_ZONE(rc.cb, "GeneralGaussianSplatPreprocess::Sort");

                    gpuSceneView->generalGaussianSplatSorter->sortKeyValuesIndirect(rc.cb,
                                                                                    maxVisible,
                                                                                    *visibleCountBuf,
                                                                                    0u,
                                                                                    *sortKeyBuf,
                                                                                    0u,
                                                                                    *sortIndexBuf,
                                                                                    0u,
                                                                                    *sortStorageBuf,
                                                                                    0u);
                    rc.cb.insertComputeUavBarrier();
                }

                {
                    RHI_GPU_ZONE(rc.cb, "GeneralGaussianSplatPreprocess::WriteIndirect");

                    rhi::ShaderLibraryRuntime::KeywordValues writeIndirectKeywords {
                        {"USE_FOVEATED_LAYER_OUTPUT", useFoveatedLayerOutput ? 1u : 0u},
                    };
                    auto writeIndirectVariantHash =
                        computeShaderVariantHash("gaussian_splat_write_indirect.comp",
                                                 vshadersystem::ShaderStage::eComp,
                                                 writeIndirectKeywords);
                    const auto* writeIndirectPipeline = getPipeline(writeIndirectVariantHash);
                    if (!writeIndirectPipeline)
                    {
                        VULTRA_CORE_ERROR("[GeneralGaussianSplatPreprocessPass] Missing write-indirect pipeline "
                                          "(variantHash={}, layeredOutput={})",
                                          writeIndirectVariantHash,
                                          useFoveatedLayerOutput ? 1 : 0);
                        return;
                    }

                    rc.cb.bindPipeline(*writeIndirectPipeline);
                    if (useFoveatedLayerOutput)
                    {
                        bindSubset(*writeIndirectPipeline,
                                   std::initializer_list<uint32_t> {31u, 32u, 33u, 34u, 35u, 36u, 37u,
                                                                    38u, 39u, 40u, 41u, 42u, 43u, 44u, 45u});
                    }
                    else
                    {
                        bindSubset(*writeIndirectPipeline, std::initializer_list<uint32_t> {18u, 20u});
                    }
                    rc.cb.dispatch({1u, 1u, 1u});
                    rc.cb.insertComputeUavBarrier();
                }
            });
    }

    rhi::ComputePipeline GeneralGaussianSplatPreprocessPass::createPipeline(const bool useMultiview,
                                                                            const bool useDirectPrefix,
                                                                            const bool useFoveatedLayerOutput,
                                                                            const bool useShLodGuard,
                                                                            const bool useShLodEnergyGuard,
                                                                            const bool useShSplitBands,
                                                                            const bool useDeterministicSourceOrderSort) const
    {
        rhi::ShaderLibraryRuntime::KeywordValues keywords {
            {"USE_MULTIVIEW", useMultiview ? 1u : 0u},
            {"USE_DIRECT_PREFIX", useDirectPrefix ? 1u : 0u},
            {"USE_FOVEATED_LAYER_OUTPUT", useFoveatedLayerOutput ? 1u : 0u},
            {"USE_SH_LOD_GUARD", useShLodGuard ? 1u : 0u},
            {"USE_SH_LOD_ENERGY_GUARD", useShLodEnergyGuard ? 1u : 0u},
            {"USE_SH_SPLIT_BANDS", useShSplitBands ? 1u : 0u},
            {"USE_DETERMINISTIC_SOURCE_ORDER_SORT", useDeterministicSourceOrderSort ? 1u : 0u},
        };
        auto shader = loadGeneralShader("gaussian_splat_preprocess.comp", vshadersystem::ShaderStage::eComp, keywords);
        if (!shader)
            return {};

        if (getRenderDevice().getBackendApi() == rhi::RenderBackendApi::eWebGPU)
        {
            return getRenderDevice().createComputePipeline(rhi::ShaderStageInfo {
                .code       = shader->wgsl,
                .reflection = shader->reflection,
            });
        }

        return getRenderDevice().createComputePipelineBuiltin(shader->spirv);
    }

    rhi::ComputePipeline GeneralGaussianSplatPreprocessPass::createPipeline(const uint64_t variantHash) const
    {
        auto shader = loadGeneralShaderVariant(variantHash, vshadersystem::ShaderStage::eComp);
        if (!shader)
            return {};

        if (getRenderDevice().getBackendApi() == rhi::RenderBackendApi::eWebGPU)
        {
            return getRenderDevice().createComputePipeline(rhi::ShaderStageInfo {
                .code       = shader->wgsl,
                .reflection = shader->reflection,
            });
        }

        return getRenderDevice().createComputePipelineBuiltin(shader->spirv);
    }
} // namespace vultra

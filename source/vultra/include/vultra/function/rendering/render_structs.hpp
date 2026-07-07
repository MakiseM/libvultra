#pragma once

#include "vultra/core/base/uuid.hpp"
#include "vultra/function/resource/gpu_scene_database.hpp"
#include "vultra/function/resource/gpu_scene_view.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace vultra
{
    using GaussianSplatShStorageLayout = resource::GpuGaussianSplatShStorageLayout;

    inline constexpr uint32_t kGaussianSplatEcsptScalarCounterCount = 64u;
    inline constexpr uint32_t kGaussianSplatEcsptEventBinCount = 5u;
    inline constexpr uint32_t kGaussianSplatEcsptEventTypeCount = 6u;
    inline constexpr uint32_t kGaussianSplatEcsptEventMetricCount = 7u;
    inline constexpr uint32_t kGaussianSplatEcsptEventBinCounterCount =
        kGaussianSplatEcsptEventBinCount *
        kGaussianSplatEcsptEventTypeCount *
        kGaussianSplatEcsptEventMetricCount;
    inline constexpr uint32_t kGaussianSplatReleaseCostHotspotCount = 8u;
    inline constexpr uint32_t kGaussianSplatEcsptCounterCount =
        kGaussianSplatEcsptScalarCounterCount + kGaussianSplatEcsptEventBinCounterCount;

    namespace rhi
    {
        class Texture;
    }

    // A cooked camera used by the renderer (SRP-style).
    // ECS CameraComponent should be cooked into this struct by CameraSystem.
    struct RenderCamera
    {
        CoreUUID uuid;

        // Debug / editor name (optional)
        std::string name;

        // Sorting
        int priority {0};

        // Multi-view metadata (mono: viewCount=1, stereo: viewCount=2)
        uint32_t viewIndex {0};
        uint32_t viewCount {1};
        bool     isXRView {false};
        bool     isXRPrimaryView {true};

        // Matrices
        glm::mat4 view {1.0f};
        glm::mat4 projection {1.0f};
        glm::mat4 viewProjection {1.0f};
        glm::mat4 inverseView {1.0f};
        glm::mat4 inverseProjection {1.0f};
        glm::mat4 inverseViewProjection {1.0f};

        float zNear {0.1f};
        float zFar {1000.0f};
        float fovY {glm::radians(60.0f)};

        std::array<glm::vec4, 6> frustumPlanes {glm::vec4(0.0f),
                                                glm::vec4(0.0f),
                                                glm::vec4(0.0f),
                                                glm::vec4(0.0f),
                                                glm::vec4(0.0f),
                                                glm::vec4(0.0f)};

        // Render target (nullptr => backbuffer or XR-provided target)
        rhi::Texture* target {nullptr};
        glm::vec4     clearValue {0, 0, 0, 1};
        bool          renderImGui {true};

        // SRP binding (string key, resolved to a Renderer instance by RenderSystem)
        // Example: "universal", "hd"
        std::string rendererKey {"universal"};
    };

    // Cooked render instance extracted from World.
    // Renderer consumes RenderWorld only.
    struct RenderInstance
    {
        CoreUUID  entity;
        uint32_t  meshIndex {0};
        uint32_t  materialIndex {0};
        glm::mat4 worldMatrix {1.0f};
    };

    struct RenderGaussianSplatInstance
    {
        CoreUUID  entity;
        uint32_t  splatIndex {0};
        glm::mat4 worldMatrix {1.0f};
    };

    // Gaussian splat rendering has one LOD path: imported/trained assets are
    // physically sorted by importance, and Ordered CLOD renders a prefix of that
    // packed order. There is no hierarchy/proxy LOD or runtime importance sort.
    enum class GaussianSplatBaselineMode : uint8_t
    {
        eBaseline = 0,
        eOrderedClod,
    };

    enum class GaussianSplatLodBudgetMode : uint8_t
    {
        eCount = 0,
        eProjectedTileCost,
        eFoveatedScore,
        eCoverageBinScore,
    };

    enum class GaussianSplatCachedSelectionMembershipMode : uint8_t
    {
        eStaticRandom = 0,
        eDynamicDeterministic,
        eDynamicFrozenHash,
        eMatchedRateRandomNull,
    };

    enum class GaussianSplatShaderAntiPopMode : uint8_t
    {
        eOff = 0,
        eSoftRamp,
        eHashRamp,
        eHashRampGuarded,
        eLinearDefault,
        eGuardedGazeAnchorCrossfade,
        eEccentricityStochasticTransition,
        eStableOpticalDepthThinning,
        eCoverageStableLogpolarRelease,
    };

    enum class GaussianSplatShaderAntiPopPKeepCurve : uint8_t
    {
        eCurrent = 0,
        eLinear,
        eSmoothWide,
        eLogisticSoft,
        eLogisticSteep,
    };

    enum class GaussianSplatShaderAntiPopNormalizeMode : uint8_t
    {
        eOff = 0,
        eGlobalLuma,
        eAlphaMass,
    };

    enum class GaussianSplatDelayedGuideTemporalReleasePolicy : uint8_t
    {
        eCurrent = 0,
        eRiskOnly,
        eCapped,
        eFootprintDecay,
        eTileLocalCapped,
    };

    struct GaussianSplatCachedSelectionOracleDebugSample
    {
        uint32_t sourceId {0u};
        float    eccentricityDegrees {0.0f};
        float    keepProbability {0.0f};
        float    opacityProxy {0.0f};
        float    hashThreshold {0.0f};
        double   projectedAreaPx {0.0};
        uint64_t tileCost {0u};
        bool     visibleAtCache {false};
        bool     selected {false};
    };

    enum class GaussianProjectedCostBuildMode : uint8_t
    {
        eCpu = 0,
        eGpuSync,
    };

    enum class GaussianSplatFoveatedRenderMode : uint8_t
    {
        eSinglePass = 0,
        eLayeredComposite,
    };

    enum class GaussianSplatFoveatedAdaptationMode : uint8_t
    {
        eFixed = 0,
        eDynamicBudget,
        eStabilityAwareBudget,
        eDynamicRange,
        eProgressiveCenterOut,
        eProgressiveGreedy,
    };

    enum class GaussianSplatFoveatedDistribution : uint8_t
    {
        eHardRing = 0,
        eSmoothstep,
        eGaussian,
        eExponential,
        eInversePower,
        eLogPolar,
        eContinuousScheduler,
        eFoveaProtectedContinuous,
        eCortical,
        eConeDensityFitted,
    };

    enum class GaussianSplatFoveatedCoverageGuardMode : uint8_t
    {
        eOff = 0,
        eGlobal,
        eLocalBounded,
        eRiskTriggered,
    };

    enum class GaussianSplatShLodGuardMode : uint8_t
    {
        eOff = 0,
        eEnergy,
        eProjectedCost,
        eEnergyProjectedCost,
    };

    inline constexpr uint32_t kGaussianSplatFoveatedLayerCount =
        resource::kGeneralGaussianSplatFoveatedLayerCount;

    struct GaussianSplatFoveatedLayerConfig
    {
        float eccentricityDegrees {0.0f};
        float resolutionScale {1.0f};
        float lodLevel {1.0f};
    };

    struct GaussianSplatRenderSettings
    {
        GaussianSplatBaselineMode           baselineMode {GaussianSplatBaselineMode::eBaseline};
        uint32_t                            lodBudget {0}; // 0 means derive the selected count from clodLevel.
        float                               clodLevel {1.0f}; // Fraction of the ordered list to keep when lodBudget is automatic.
        GaussianSplatLodBudgetMode          lodBudgetMode {GaussianSplatLodBudgetMode::eCount};
        uint64_t                            projectedCostBudget {0}; // 0 means derive from projectedCostBudgetRatio.
        float                               projectedCostBudgetRatio {0.0f}; // 0 means derive from effective CLOD level.
        uint32_t                            projectedCostChunkSize {1024u};
        uint32_t                            foveatedScoreSelectionStride {1u};
        uint32_t                            foveatedCoverageBinGridX {12u};
        uint32_t                            foveatedCoverageBinGridY {8u};
        float                               foveatedCoverageBinQuotaRatio {0.35f};
        float                               foveatedCoverageBinPrefixRatio {0.35f};
        GaussianProjectedCostBuildMode      projectedCostBuildMode {GaussianProjectedCostBuildMode::eCpu};
        bool                                projectedCostSemanticDiffEnabled {false};
        bool                                cleanTimingMode {false};
        bool                                foveatedClodEnabled {false};
        bool                                foveatedManualGazeControlEnabled {false};
        bool                                foveatedXrGazeEnabled {false};
        bool                                foveatedCoverageCompensationEnabled {true};
        GaussianSplatFoveatedRenderMode     foveatedRenderMode {GaussianSplatFoveatedRenderMode::eSinglePass};
        glm::vec2                           foveatedGaze {0.5f, 0.5f}; // Viewport UV, top-left origin.
        glm::vec2                           foveatedRingDegrees {12.0f, 32.0f};
        glm::vec3                           foveatedRingLevels {1.0f, 0.40f, 0.15f};
        glm::vec3                           foveatedResolutionScales {1.0f, 0.75f, 0.50f};
        float                               foveatedTransitionDegrees {8.0f};
        GaussianSplatFoveatedDistribution   foveatedDistribution {GaussianSplatFoveatedDistribution::eGaussian};
        float                               foveatedContinuousTheta0Degrees {24.0f};
        float                               foveatedContinuousAlpha {2.0f};
        float                               foveatedContinuousMinLevel {0.12f};
        GaussianSplatFoveatedAdaptationMode foveatedAdaptationMode {GaussianSplatFoveatedAdaptationMode::eFixed};
        float                               foveatedTargetFrameMs {11.1f};
        float                               foveatedBudgetAdjustRate {0.05f};
        bool                                schedulerShrinkEvent {false};
        bool                                schedulerGrowEvent {false};
        float                               schedulerBudgetBefore {-1.0f};
        float                               schedulerBudgetAfter {-1.0f};
        double                              schedulerP95Ema {-1.0};
        double                              schedulerP95Window {-1.0};
        float                               schedulerTargetMs {-1.0f};
        float                               schedulerMarginLow {-1.0f};
        float                               schedulerMarginHigh {-1.0f};
        GaussianSplatFoveatedCoverageGuardMode foveatedCoverageGuardMode {
            GaussianSplatFoveatedCoverageGuardMode::eGlobal};
        float                               foveatedCoverageProtectionDegrees {0.0f};
        float                               foveatedCoverageGuardBudgetRatio {0.05f};
        glm::vec3                           foveatedCoverageGuardMinLevels {0.75f, 0.45f, 0.15f};
        uint32_t                            foveatedCoverageGuardMaxAdds {0u}; // 0 means only the cost cap limits repair.
        bool                                foveatedTemporalHysteresisEnabled {true};
        bool                                foveatedBoundarySmoothingEnabled {true};
        uint32_t                            foveatedTemporalResidencyFrames {6u};
        float                               foveatedTemporalHysteresisRatio {0.25f};
        float                               foveatedBoundarySmoothingRatio {0.18f};
        float                               foveatedTemporalPeripheralScale {0.0f};
        bool                                foveatedShLodEnabled {false};
        bool                                foveatedShSmoothSuppressionEnabled {false};
        bool                                peripheralTemporalFilterEnabled {false};
        float                               peripheralTemporalFilterOuterDegrees {56.0f};
        float                               peripheralTemporalFilterLambdaScale {0.85f};
        float                               peripheralTemporalFilterRejectionThreshold {0.18f};
        float                               peripheralTemporalFilterClampRadius {0.20f};
        glm::uvec3                          foveatedShLodDegrees {3u, 3u, 3u};
        GaussianSplatShLodGuardMode         foveatedShLodGuardMode {GaussianSplatShLodGuardMode::eOff};
        float                               foveatedShLodGuardThresholdMid {0.05f};
        float                               foveatedShLodGuardThresholdHigh {0.20f};
        bool                                shPopLogEnabled {false};
        bool                                shDegreeHysteresisEnabled {false};
        uint32_t                            shDegreeDowngradeDelay {3u};
        float                               shDegreeGuardBandDegrees {0.0f};
        float                               gazeAnchorDeadbandDegrees {2.0f};
        uint32_t                            gazeAnchorFadeFrames {4u};
        uint32_t                            gazeAnchorMinUpdateFrames {2u};
        float                               gazeAnchorTransitionBudgetRatio {-1.0f};
        bool                                gazeAnchorImmediateFoveaFill {true};
        bool                                gazeAnchorContributionGuard {false};
        bool                                gazeAnchorDebugLogEnabled {false};
        uint32_t                            eccStochasticFadeFrames {4u};
        float                               eccStochasticProtectOldFoveaDegrees {-1.0f};
        float                               eccStochasticProtectNewFoveaDegrees {-1.0f};
        float                               eccStochasticBoundaryBandDegrees {4.0f};
        float                               eccStochasticMinPDelta {0.02f};
        bool                                eccStochasticContributionGuard {true};
        bool                                eccStochasticDebugLogEnabled {false};
        GaussianSplatShaderAntiPopMode      shaderAntiPopMode {GaussianSplatShaderAntiPopMode::eOff};
        uint32_t                            shaderAntiPopHashSeed {0u};
        float                               shaderAntiPopRampWidth {0.05f};
        float                               shaderAntiPopGuardThreshold {0.0f};
        float                               shaderAntiPopGuardFloor {0.0f};
        GaussianSplatShaderAntiPopPKeepCurve shaderAntiPopPKeepCurve {
            GaussianSplatShaderAntiPopPKeepCurve::eCurrent};
        float                               shaderAntiPopPrefixRatio {-1.0f};
        GaussianSplatShaderAntiPopNormalizeMode shaderAntiPopNormalizeMode {
            GaussianSplatShaderAntiPopNormalizeMode::eOff};
        float                               shaderAntiPopNormalizeStrength {1.0f};
        float                               shaderAntiPopNormalizeClampMin {0.75f};
        float                               shaderAntiPopNormalizeClampMax {1.15f};
        float                               shaderAntiPopNormalizeFactor {1.0f};
        bool                                coverageStableReleaseCoverageFloorEnabled {true};
        bool                                coverageStableReleaseStableHashEnabled {true};
        bool                                coverageStableReleaseStaggeredReleaseEnabled {true};
        bool                                coverageStableReleaseSaturatedFloorEnabled {false};
        float                               coverageStableReleaseLambda {0.7743f};
        float                               coverageStableReleaseDMin {4.0f};
        float                               coverageStableReleaseSigmaMax {0.65f};
        float                               coverageStableReleaseFloorEma {0.0f};
        float                               coverageStableReleaseReleaseEpsilon {1e-4f};
        uint32_t                            coverageStableReleaseTileGridX {16u};
        uint32_t                            coverageStableReleaseTileGridY {8u};
        bool                                coverageTextureFloorEnabled {false};
        uint32_t                            coverageTextureFloorWidth {160u};
        uint32_t                            coverageTextureFloorHeight {90u};
        float                               coverageTextureFloorDMin {4.0f};
        float                               coverageTextureFloorSigmaMax {0.65f};
        float                               coverageTextureFloorStrength {0.25f};
        float                               coverageTextureFloorHistoryBeta {0.80f};
        uint32_t                            coverageTextureFloorUpdateInterval {1u};
        bool                                coverageTextureFloorDebugEnabled {false};
        bool                                guideBeforeDiscardEnabled {false};
        bool                                delayedGuideLogPolarEnabled {false};
        bool                                delayedGuideTemporalLogPolarEnabled {false};
        GaussianSplatDelayedGuideTemporalReleasePolicy delayedGuideTemporalReleasePolicy {
            GaussianSplatDelayedGuideTemporalReleasePolicy::eCurrent};
        float                               delayedGuideTemporalReleaseCapRatio {0.12f};
        float                               delayedGuideTemporalRiskThreshold {0.35f};
        float                               delayedGuideTemporalFootprintDecayScale {0.55f};
        float                               delayedGuideTemporalLargeFootprintPx {48.0f};
        bool                                shaderAntiPopOrderFree {false};
        bool                                shaderAntiPopDebugLogEnabled {false};
        bool                                selectedIdDebugLoggingEnabled {false};
        bool                                cachedSelectionOracleEnabled {false};
        GaussianSplatCachedSelectionMembershipMode cachedSelectionMembershipMode {
            GaussianSplatCachedSelectionMembershipMode::eDynamicDeterministic};
        uint32_t                            cachedSelectionOracleSeed {0u};
        uint32_t                            cachedSelectionOracleStaticTargetCount {UINT32_MAX};
        uint32_t                            cachedSelectionOracleMatchedNullFrameIndex {UINT32_MAX};
        uint32_t                            cachedSelectionOracleMatchedNullTargetAddCount {0u};
        uint32_t                            cachedSelectionOracleMatchedNullTargetRemoveCount {0u};
        uint32_t                            cachedSelectionOracleMatchedNullTargetSelectedCount {0u};
        uint32_t                            cachedSelectionOracleMatchedNullTargetSymmetricDiff {0u};
        bool                                cachedSelectionOracleDeterministicSourceOrderSort {false};
        bool                                cachedSelectionOracleForcedSourceIdsEnabled {false};
        std::vector<uint32_t>               cachedSelectionOracleForcedSourceIds;
        GaussianSplatShStorageLayout        shStorageLayout {GaussianSplatShStorageLayout::eMonolithic};
        [[nodiscard]] std::array<GaussianSplatFoveatedLayerConfig, kGaussianSplatFoveatedLayerCount> foveatedLayers() const
        {
            return {
                GaussianSplatFoveatedLayerConfig {
                    .eccentricityDegrees = foveatedRingDegrees.x,
                    .resolutionScale     = foveatedResolutionScales.x,
                    .lodLevel            = foveatedRingLevels.x,
                },
                GaussianSplatFoveatedLayerConfig {
                    .eccentricityDegrees = foveatedRingDegrees.y,
                    .resolutionScale     = foveatedResolutionScales.y,
                    .lodLevel            = foveatedRingLevels.y,
                },
                GaussianSplatFoveatedLayerConfig {
                    .eccentricityDegrees = 180.0f,
                    .resolutionScale     = foveatedResolutionScales.z,
                    .lodLevel            = foveatedRingLevels.z,
                },
            };
        }

        [[nodiscard]] bool orderedClodEnabled() const
        {
            return baselineMode == GaussianSplatBaselineMode::eOrderedClod;
        }

        [[nodiscard]] bool lodBudgetEnabled() const
        {
            return orderedClodEnabled();
        }

        [[nodiscard]] bool projectedCostBudgetEnabled() const
        {
            return lodBudgetEnabled() && lodBudgetMode == GaussianSplatLodBudgetMode::eProjectedTileCost;
        }

        [[nodiscard]] bool foveatedScoreBudgetEnabled() const
        {
            return lodBudgetEnabled() &&
                   (lodBudgetMode == GaussianSplatLodBudgetMode::eFoveatedScore ||
                    lodBudgetMode == GaussianSplatLodBudgetMode::eCoverageBinScore);
        }

        [[nodiscard]] bool coverageBinScoreBudgetEnabled() const
        {
            return lodBudgetEnabled() && lodBudgetMode == GaussianSplatLodBudgetMode::eCoverageBinScore;
        }

        [[nodiscard]] bool foveatedClodActive() const
        {
            return orderedClodEnabled() && foveatedClodEnabled;
        }

        [[nodiscard]] bool foveatedLayeredCompositeActive() const
        {
            return foveatedClodActive() && foveatedRenderMode == GaussianSplatFoveatedRenderMode::eLayeredComposite;
        }

        [[nodiscard]] bool foveatedShLodActive() const
        {
            return orderedClodEnabled() && foveatedShLodEnabled;
        }

        [[nodiscard]] bool foveatedShSmoothSuppressionActive() const
        {
            return foveatedShSmoothSuppressionEnabled;
        }

        [[nodiscard]] bool shaderAntiPopActive() const
        {
            return (orderedClodEnabled() || shaderAntiPopOrderFree) && foveatedClodEnabled &&
                   shaderAntiPopMode != GaussianSplatShaderAntiPopMode::eOff;
        }
    };

    inline void applyGaussianSplatFovGsStyleBaseline(GaussianSplatRenderSettings& settings)
    {
        settings.baselineMode                     = GaussianSplatBaselineMode::eOrderedClod;
        settings.lodBudget                        = 0u;
        settings.clodLevel                        = 1.0f;
        settings.lodBudgetMode                    = GaussianSplatLodBudgetMode::eCount;
        settings.projectedCostBudget              = 0u;
        settings.projectedCostBudgetRatio         = 0.0f;
        settings.projectedCostChunkSize           = 1024u;
        settings.projectedCostBuildMode           = GaussianProjectedCostBuildMode::eCpu;
        settings.projectedCostSemanticDiffEnabled = false;
        settings.foveatedClodEnabled              = true;
        settings.foveatedManualGazeControlEnabled = false;
        settings.foveatedXrGazeEnabled            = false;
        settings.foveatedCoverageCompensationEnabled = false;
        settings.foveatedRenderMode               = GaussianSplatFoveatedRenderMode::eSinglePass;
        settings.foveatedRingDegrees              = glm::vec2 {10.0f, 42.0f};
        settings.foveatedRingLevels               = glm::vec3 {1.0f, 0.25f, 0.125f};
        settings.foveatedResolutionScales         = glm::vec3 {1.0f, 1.0f, 1.0f};
        settings.foveatedTransitionDegrees        = 4.0f;
        settings.foveatedDistribution             = GaussianSplatFoveatedDistribution::eHardRing;
        settings.foveatedContinuousTheta0Degrees  = 24.0f;
        settings.foveatedContinuousAlpha          = 2.0f;
        settings.foveatedContinuousMinLevel       = 0.12f;
        settings.foveatedAdaptationMode           = GaussianSplatFoveatedAdaptationMode::eFixed;
        settings.foveatedCoverageGuardMode         = GaussianSplatFoveatedCoverageGuardMode::eOff;
        settings.foveatedCoverageProtectionDegrees = 0.0f;
        settings.foveatedCoverageGuardBudgetRatio  = 0.0f;
        settings.foveatedCoverageGuardMaxAdds       = 0u;
        settings.foveatedTemporalHysteresisEnabled = false;
        settings.foveatedBoundarySmoothingEnabled  = false;
        settings.foveatedTemporalPeripheralScale   = 0.0f;
        settings.foveatedShLodEnabled              = false;
        settings.foveatedShSmoothSuppressionEnabled = false;
        settings.peripheralTemporalFilterEnabled   = false;
        settings.foveatedShLodDegrees              = glm::uvec3 {3u, 3u, 3u};
        settings.foveatedShLodGuardMode            = GaussianSplatShLodGuardMode::eOff;
        settings.foveatedShLodGuardThresholdMid    = 0.05f;
        settings.foveatedShLodGuardThresholdHigh   = 0.20f;
        settings.shPopLogEnabled                   = false;
        settings.shDegreeHysteresisEnabled         = false;
        settings.shDegreeDowngradeDelay            = 3u;
        settings.shDegreeGuardBandDegrees          = 0.0f;
        settings.shStorageLayout                   = GaussianSplatShStorageLayout::eMonolithic;
    }

    struct GaussianSplatFrameStats
    {
        uint64_t                            frameIndex {0};
        GaussianSplatBaselineMode           baselineMode {GaussianSplatBaselineMode::eBaseline};
        GaussianSplatFoveatedRenderMode     foveatedRenderMode {GaussianSplatFoveatedRenderMode::eSinglePass};
        GaussianSplatLodBudgetMode          lodBudgetMode {GaussianSplatLodBudgetMode::eCount};
        bool                                lodBudgetEnabled {false};
        bool                                foveatedClodEnabled {false};
        bool                                foveatedLayeredCompositeEnabled {false};
        bool                                foveatedXrGazeEnabled {false};
        bool                                foveatedXrGazeActive {false};
        bool                                foveatedCoverageCompensationEnabled {false};
        GaussianSplatFoveatedAdaptationMode foveatedAdaptationMode {GaussianSplatFoveatedAdaptationMode::eFixed};
        GaussianSplatFoveatedDistribution   foveatedDistribution {GaussianSplatFoveatedDistribution::eGaussian};
        bool                                directPrefix {false};
        uint32_t                            lodBudget {0};
        uint64_t                            costBudget {0};
        uint64_t                            actualProjectedCost {0};
        uint64_t                            costOvershoot {0};
        float                               projectedCostBudgetRatio {0.0f};
        uint32_t                            projectedCostChunkSize {1024u};
        uint32_t                            foveatedCoverageBinGridX {12u};
        uint32_t                            foveatedCoverageBinGridY {8u};
        float                               foveatedCoverageBinQuotaRatio {0.35f};
        float                               foveatedCoverageBinPrefixRatio {0.35f};
        GaussianProjectedCostBuildMode      projectedCostBuildMode {GaussianProjectedCostBuildMode::eCpu};
        bool                                projectedCostSemanticDiffEnabled {false};
        uint32_t                            selectedCostChunks {0};
        double                              a4ProjectedCostSelectionCpuMs {-1.0};
        double                              a4CostEstimationCpuMs {-1.0};
        double                              a4ChunkSelectionCpuMs {-1.0};
        double                              a4ChunkStopCpuMs {-1.0};
        double                              a4GuardAnalysisCpuMs {-1.0};
        double                              a4TotalSelectionCpuMs {-1.0};
        uint32_t                            a4NumChunksScanned {0u};
        uint32_t                            a4NumGaussiansOrChunksConsidered {0u};
        uint64_t                            a4ProjectedSampleCacheHit {0u};
        uint64_t                            a4ProjectedSampleCacheMiss {0u};
        bool                                a4ProjectedSampleReused {false};
        uint64_t                            a4NumProjectedSamplesBuilt {0u};
        uint64_t                            a4NumProjectedSamplesReused {0u};
        uint64_t                            a4CacheEntryCount {0u};
        uint64_t                            a4CacheLookupCount {0u};
        uint64_t                            a4CacheHitCount {0u};
        uint64_t                            a4CacheMissCount {0u};
        uint64_t                            a4CacheFullMissCount {0u};
        uint64_t                            a4CacheBuildCount {0u};
        uint64_t                            a4CacheEvictionCount {0u};
        uint64_t                            a4SampleCacheKeyHash {0u};
        uint64_t                            a4ChunkCacheHitCount {0u};
        uint64_t                            a4ChunkCacheMissCount {0u};
        uint64_t                            a4ProjectedSamplesBuilt {0u};
        uint64_t                            a4ProjectedSamplesReused {0u};
        double                              a4ChunkAggregationCpuMs {-1.0};
        double                              a4CacheLookupCpuMs {-1.0};
        double                              a4CacheHitRate {-1.0};
        bool                                a4GpuCostBuildEnabled {false};
        double                              a4GpuCostBuildGpuMs {-1.0};
        double                              a4GpuCostReadbackCpuMs {-1.0};
        uint32_t                            a4GpuCostChunkCount {0u};
        bool                                a4GpuCostValid {false};
        bool                                a4CpuFallbackUsed {false};
        double                              a4GpuCpuCostL1Error {-1.0};
        double                              a4GpuCpuCostMaxError {-1.0};
        double                              a4GpuCpuSelectedRatioDelta {-1.0};
        double                              a4GpuCpuActualCostDelta {-1.0};
        double                              a4GpuCpuOvershootDelta {-1.0};
        glm::vec3                           foveatedRingLevels {1.0f, 0.40f, 0.15f};
        glm::vec3                           foveatedResolutionScales {1.0f, 0.75f, 0.50f};
        glm::vec2                           foveatedGaze {0.5f, 0.5f};
        glm::vec2                           foveatedRingDegrees {12.0f, 32.0f};
        float                               foveatedTargetFrameMs {11.1f};
        float                               foveatedContinuousTheta0Degrees {24.0f};
        float                               foveatedContinuousAlpha {2.0f};
        float                               foveatedContinuousMinLevel {0.12f};
        GaussianSplatFoveatedCoverageGuardMode foveatedCoverageGuardMode {
            GaussianSplatFoveatedCoverageGuardMode::eGlobal};
        float                               foveatedCoverageProtectionDegrees {0.0f};
        float                               foveatedCoverageGuardBudgetRatio {0.05f};
        glm::vec3                           foveatedCoverageGuardMinLevels {0.75f, 0.45f, 0.15f};
        uint32_t                            foveatedCoverageGuardMaxAdds {0u};
        uint32_t                            foveatedCoverageGuardMaxAddsEffective {0u};
        bool                                foveatedCoverageGuardModeEnabled {false};
        uint32_t                            foveatedCoverageGuardRiskSectors {0u};
        bool                                foveatedCoverageGuardRiskDetected {false};
        bool                                foveatedCoverageGuardActive {false};
        bool                                foveatedCoverageGuardRepairActive {false};
        bool                                foveatedCoverageGuardAddedAny {false};
        uint32_t                            foveatedCoverageGuardAddedCount {0u};
        uint64_t                            foveatedCoverageGuardAddedCost {0u};
        uint64_t                            foveatedCoverageGuardBudgetCap {0u};
        uint32_t                            foveatedCoverageRiskSectorCountTotal {0u};
        uint32_t                            foveatedCoverageRiskSectorCountCenter {0u};
        uint32_t                            foveatedCoverageRiskSectorCountMid {0u};
        uint32_t                            foveatedCoverageRiskSectorCountOuter {0u};
        bool                                foveatedCoverageRiskSectorActiveCenter {false};
        bool                                foveatedCoverageRiskSectorActiveMid {false};
        bool                                foveatedCoverageRiskSectorActiveOuter {false};
        float                               foveatedCoverageFailureBeforeGuard {0.0f};
        float                               foveatedCoverageFailureAfterGuard {0.0f};
        float                               foveatedCoverageFailureBeforeCenter {-1.0f};
        float                               foveatedCoverageFailureBeforeMid {-1.0f};
        float                               foveatedCoverageFailureBeforeOuter {-1.0f};
        float                               foveatedCoverageFailureAfterCenter {-1.0f};
        float                               foveatedCoverageFailureAfterMid {-1.0f};
        float                               foveatedCoverageFailureAfterOuter {-1.0f};
        float                               foveatedCoverageScoreBeforeGuard {1.0f};
        float                               foveatedCoverageScoreAfterGuard {1.0f};
        uint32_t                            foveatedCoverageGuardAddedCountCenter {0u};
        uint32_t                            foveatedCoverageGuardAddedCountMid {0u};
        uint32_t                            foveatedCoverageGuardAddedCountOuter {0u};
        uint64_t                            foveatedCoverageGuardAddedCostCenter {0u};
        uint64_t                            foveatedCoverageGuardAddedCostMid {0u};
        uint64_t                            foveatedCoverageGuardAddedCostOuter {0u};
        bool                                foveatedCoverageGuardRepairCapHit {false};
        bool                                foveatedCoverageGuardRepairCappedByMaxAdds {false};
        bool                                foveatedCoverageGuardRepairCappedByCost {false};
        bool                                foveatedCoverageGuardRepairNoCandidate {false};
        uint32_t                            foveatedCoverageGuardCandidateCount {0u};
        uint32_t                            foveatedCoverageGuardCandidateCountCenter {0u};
        uint32_t                            foveatedCoverageGuardCandidateCountMid {0u};
        uint32_t                            foveatedCoverageGuardCandidateCountOuter {0u};
        uint32_t                            foveatedCoverageGuardTrueCandidateCount {0u};
        bool                                foveatedCoverageGuardCandidateCountTruncated {false};
        uint32_t                            foveatedCoverageGuardRequiredExtraCountEstimate {UINT32_MAX};
        double                              foveatedCoverageGuardRequiredExtraAreaPx {-1.0};
        double                              foveatedCoverageGuardDeficitAreaBefore {-1.0};
        double                              foveatedCoverageGuardDeficitAreaAfter {-1.0};
        uint32_t                            foveatedCoverageGuardAddedCountToFailingSector {0u};
        uint64_t                            foveatedCoverageGuardAddedCostToFailingSector {0u};
        uint32_t                            foveatedCoverageGuardFailingSectorId {UINT32_MAX};
        double                              foveatedCoverageGuardRepairSectorMatchRate {-1.0};
        double                              foveatedCoverageGuardAnalysisCpuMs {-1.0};
        double                              foveatedCoverageGuardProjectedSampleBuildCpuMs {-1.0};
        double                              foveatedCoverageGuardCoverageAreaCpuMs {-1.0};
        double                              foveatedCoverageGuardCandidateScanCpuMs {-1.0};
        double                              foveatedCoverageGuardRegionRebuildCpuMs {-1.0};
        uint64_t                            foveatedCoverageGuardCacheFullMissCount {0u};
        uint64_t                            foveatedCoverageGuardCacheHitCount {0u};
        uint64_t                            foveatedCoverageGuardCacheMissCount {0u};
        uint64_t                            foveatedCoverageGuardGeometryCacheHitCount {0u};
        uint64_t                            foveatedCoverageGuardGeometryCacheMissCount {0u};
        uint64_t                            foveatedCoverageGuardGeometryCacheFullMissCount {0u};
        uint64_t                            foveatedCoverageGuardGeometrySamplesBuilt {0u};
        uint64_t                            foveatedCoverageGuardGeometrySamplesReused {0u};
        bool                                schedulerShrinkEvent {false};
        bool                                schedulerGrowEvent {false};
        float                               schedulerBudgetBefore {-1.0f};
        float                               schedulerBudgetAfter {-1.0f};
        double                              schedulerP95Ema {-1.0};
        double                              schedulerP95Window {-1.0};
        float                               schedulerTargetMs {-1.0f};
        float                               schedulerMarginLow {-1.0f};
        float                               schedulerMarginHigh {-1.0f};
        float                               projectedCostBudgetRatioBeforeGuard {-1.0f};
        float                               projectedCostBudgetRatioAfterScheduler {-1.0f};
        float                               selectedRatioBeforeGuard {-1.0f};
        float                               selectedRatioAfterGuard {-1.0f};
        double                              foveatedCoverageGuardAddedCostRatioToBudget {-1.0};
        double                              foveatedCoverageGuardAddedCountRatioToSelected {-1.0};
        bool                                foveatedTemporalHysteresisEnabled {true};
        bool                                foveatedBoundarySmoothingEnabled {true};
        uint32_t                            foveatedTemporalResidencyFrames {6u};
        float                               foveatedTemporalHysteresisRatio {0.25f};
        float                               foveatedBoundarySmoothingRatio {0.18f};
        float                               foveatedTemporalPeripheralScale {0.0f};
        bool                                foveatedShLodEnabled {false};
        bool                                foveatedShSmoothSuppressionEnabled {false};
        bool                                peripheralTemporalFilterEnabled {false};
        float                               peripheralTemporalFilterMidDegrees {32.0f};
        float                               peripheralTemporalFilterOuterDegrees {56.0f};
        float                               peripheralTemporalFilterLambdaScale {0.85f};
        float                               peripheralTemporalFilterRejectionThreshold {0.18f};
        float                               peripheralTemporalFilterClampRadius {0.20f};
        uint32_t                            shDegreeCenter {3u};
        uint32_t                            shDegreeMid {3u};
        uint32_t                            shDegreeOuter {3u};
        float                               shSmoothL1StartDegrees {32.0f};
        float                               shSmoothL1EndDegrees {40.0f};
        float                               shSmoothL2StartDegrees {22.0f};
        float                               shSmoothL2EndDegrees {32.0f};
        float                               shSmoothL3StartDegrees {12.0f};
        float                               shSmoothL3EndDegrees {22.0f};
        uint64_t                            estimatedShAcCoeffReads {0u};
        double                              estimatedShAcReadReductionVsDegree3 {0.0};
        GaussianSplatShStorageLayout        shStorageLayout {GaussianSplatShStorageLayout::eMonolithic};
        uint64_t                            shBandL1ReadsEst {0u};
        uint64_t                            shBandL2ReadsEst {0u};
        uint64_t                            shBandL3ReadsEst {0u};
        uint64_t                            shBandBytesEst {0u};
        double                              shBandBytesReductionVsMonolithic {0.0};
        uint64_t                            shStorageMetadataBytesEst {0u};
        bool                                shEnergyMetadataAvailable {false};
        double                              shEnergyMeanAfter0 {0.0};
        double                              shEnergyMeanAfter1 {0.0};
        double                              shEnergyMeanAfter2 {0.0};
        double                              shEnergyP95After0 {0.0};
        double                              shEnergyP95After1 {0.0};
        double                              shEnergyP95After2 {0.0};
        GaussianSplatShLodGuardMode         shGuardMode {GaussianSplatShLodGuardMode::eOff};
        uint32_t                            shGuardRaisedCount {0u};
        double                              shGuardRaisedRatio {0.0};
        uint64_t                            shGuardRecoveredAcReads {0u};
        uint64_t                            shGuardSavedAcReadsAfterGuard {0u};
        float                               shGuardThresholdMid {0.05f};
        float                               shGuardThresholdHigh {0.20f};
        bool                                shPopLogEnabled {false};
        bool                                shDegreeHysteresisEnabled {false};
        uint32_t                            shDegreeDowngradeDelay {3u};
        float                               shDegreeGuardBandDegrees {0.0f};
        uint32_t                            shDegreeChangedCount {0u};
        double                              shDegreeChangedRatio {0.0};
        double                              shPopEnergyProxy {0.0};
        double                              shPopEnergyFovea {0.0};
        double                              shPopEnergyMid {0.0};
        double                              shPopEnergyPeriphery {0.0};
        uint32_t                            shPopGuardRaiseCount {0u};
        uint32_t                            shPopDelayedDowngradeCount {0u};
        bool                                gazeAnchorCrossfadeEnabled {false};
        bool                                gazeAnchorGpuCrossfadeEnabled {false};
        float                               gazeAnchorDeadbandDegrees {2.0f};
        uint32_t                            gazeAnchorFadeFrames {4u};
        uint32_t                            gazeAnchorMinUpdateFrames {2u};
        float                               gazeAnchorTransitionBudgetRatio {-1.0f};
        bool                                gazeAnchorImmediateFoveaFill {true};
        bool                                gazeAnchorContributionGuard {false};
        bool                                gazeAnchorDebugLogEnabled {false};
        float                               gazeAnchorAnchorX {0.5f};
        float                               gazeAnchorAnchorY {0.5f};
        float                               gazeAnchorNewAnchorX {0.5f};
        float                               gazeAnchorNewAnchorY {0.5f};
        float                               gazeAnchorDistanceDegrees {0.0f};
        uint32_t                            gazeAnchorFramesSinceUpdate {0u};
        float                               gazeAnchorFadePhase {1.0f};
        bool                                gazeAnchorUpdateEvent {false};
        uint32_t                            gazeAnchorUpdateEventCount {0u};
        uint32_t                            gazeAnchorDeadbandViolationCount {0u};
        uint32_t                            gazeAnchorUnionCount {0u};
        float                               gazeAnchorUnionCountRatio {0.0f};
        uint32_t                            gazeAnchorSharedCount {0u};
        float                               gazeAnchorSharedRatio {0.0f};
        uint32_t                            gazeAnchorOldOnlyCount {0u};
        uint32_t                            gazeAnchorNewOnlyCount {0u};
        uint32_t                            gazeAnchorFadeActiveCount {0u};
        uint32_t                            gazeAnchorImmediateFoveaFillCount {0u};
        uint32_t                            gazeAnchorTransitionBudgetTargetCount {0u};
        uint32_t                            gazeAnchorTransitionBudgetCappedCount {0u};
        uint32_t                            gazeAnchorRenderedOldOnlyCount {0u};
        uint32_t                            gazeAnchorRenderedNewOnlyCount {0u};
        uint32_t                            gazeAnchorDroppedOldOnlyCount {0u};
        uint32_t                            gazeAnchorDroppedNewOnlyCount {0u};
        uint32_t                            gazeAnchorHighContributionProtectedCount {0u};
        float                               gazeAnchorWeightedTvChurn {0.0f};
        bool                                eccStochasticTransitionEnabled {false};
        uint32_t                            eccStochasticFadeFrames {4u};
        float                               eccStochasticProtectOldFoveaDegrees {-1.0f};
        float                               eccStochasticProtectNewFoveaDegrees {-1.0f};
        float                               eccStochasticBoundaryBandDegrees {4.0f};
        float                               eccStochasticMinPDelta {0.02f};
        bool                                eccStochasticContributionGuard {true};
        bool                                eccStochasticDebugLogEnabled {false};
        float                               eccStochasticOldGazeX {0.5f};
        float                               eccStochasticOldGazeY {0.5f};
        float                               eccStochasticNewGazeX {0.5f};
        float                               eccStochasticNewGazeY {0.5f};
        float                               eccStochasticFadePhase {1.0f};
        uint32_t                            eccStochasticFramesSinceUpdate {0u};
        bool                                eccStochasticUpdateEvent {false};
        uint32_t                            eccStochasticUpdateEventCount {0u};
        bool                                eccStochasticCounterReadbackValid {false};
        double                              eccStochasticCounterReadbackCpuMs {-1.0};
        uint32_t                            eccStochasticTotalCandidatesSeen {0u};
        uint32_t                            eccStochasticBaseNewSelectedCount {0u};
        uint32_t                            eccStochasticEffectiveVisibleAfterEcsptCount {0u};
        uint32_t                            eccStochasticSharedCount {0u};
        uint32_t                            eccStochasticUpgradeCount {0u};
        uint32_t                            eccStochasticDowngradeCount {0u};
        uint32_t                            eccStochasticProtectedDowngradeCount {0u};
        uint32_t                            eccStochasticDroppedDowngradeCount {0u};
        uint32_t                            eccStochasticOldOnlyFadeVisibleCount {0u};
        uint32_t                            eccStochasticImmediateNewFoveaCount {0u};
        uint32_t                            eccStochasticBoundaryProtectedCount {0u};
        uint32_t                            eccStochasticContributionGuardProtectedCount {0u};
        uint32_t                            eccStochasticZeroWeightDiscardCount {0u};
        uint32_t                            eccStochasticMinPDeltaDiscardCount {0u};
        uint32_t                            eccStochasticFarPeripheryHardDropCount {0u};
        uint32_t                            stableOpticalDepthAlphaClampedCount {0u};
        uint32_t                            coverageStableReleaseFloorActiveCount {0u};
        uint32_t                            coverageStableReleaseFloorRaisedCount {0u};
        uint32_t                            coverageStableReleaseHeldCount {0u};
        uint32_t                            coverageStableReleaseDroppedCount {0u};
        uint32_t                            coverageStableReleaseOldOnlyCount {0u};
        uint32_t                            coverageStableReleaseFloorSampleCount {0u};
        float                               coverageStableReleasePFloorMean {0.0f};
        float                               coverageStableReleasePFloorMax {0.0f};
        float                               coverageStableReleaseDAlphaMean {0.0f};
        float                               coverageStableReleaseNEffMean {0.0f};
        uint32_t                            coverageStableReleaseTileTotalCount {0u};
        uint32_t                            coverageStableReleaseTileNonEmptyCount {0u};
        uint32_t                            coverageStableReleaseTileEmptyCount {0u};
        float                               coverageStableReleaseDAlphaMeanAll {0.0f};
        float                               coverageStableReleaseDAlphaMeanNonEmpty {0.0f};
        float                               coverageStableReleaseDAlphaP50NonEmpty {0.0f};
        float                               coverageStableReleaseDAlphaP90NonEmpty {0.0f};
        float                               coverageStableReleaseDAlphaP95NonEmpty {0.0f};
        float                               coverageStableReleaseNEffMeanNonEmpty {0.0f};
        float                               coverageStableReleaseNEffP50NonEmpty {0.0f};
        float                               coverageStableReleaseNEffP90NonEmpty {0.0f};
        float                               coverageStableReleaseNEffP95NonEmpty {0.0f};
        float                               coverageStableReleasePFloorMeanAll {0.0f};
        float                               coverageStableReleasePFloorMeanNonEmpty {0.0f};
        float                               coverageStableReleasePFloorP90NonEmpty {0.0f};
        float                               coverageStableReleasePFloorP95NonEmpty {0.0f};
        float                               coverageStableReleasePLpMean {0.0f};
        float                               coverageStableReleasePStaticMean {0.0f};
        float                               coverageStableReleaseFloorActiveFractionVsCandidates {0.0f};
        float                               coverageStableReleaseFloorRaisedFractionVsEffectiveVisible {0.0f};
        float                               coverageStableReleaseDeltaSum {0.0f};
        float                               coverageStableReleaseDeltaMeanOverCandidates {0.0f};
        float                               coverageStableReleaseDeltaMeanOverHeld {0.0f};
        float                               coverageStableReleaseHeldEffectiveRatio {0.0f};
        uint32_t                            coverageStableReleaseHeldEffectiveCount {0u};
        uint32_t                            coverageStableReleaseHistoryResetCount {0u};
        uint32_t                            coverageStableReleaseReentryResetCount {0u};
        uint32_t                            coverageStableReleaseEpsilonCutoffCount {0u};
        bool                                coverageStableReleaseSaturatedFloorEnabled {false};
        uint32_t                            coverageStableReleaseFloorCellsSafeSaturated {0u};
        uint32_t                            coverageStableReleaseFloorCellsUnsafe {0u};
        uint32_t                            coverageStableReleaseSkippedFloorUpdateCount {0u};
        float                               coverageStableReleaseDAlphaSafeThreshold {0.0f};
        float                               coverageStableReleaseNEffSafeThreshold {0.0f};
        float                               coverageStableReleaseFloorSaturationRatio {0.0f};
        float                               coverageStableReleaseCandidateAlphaMass {0.0f};
        float                               coverageStableReleaseRetainedAlphaMass {0.0f};
        float                               coverageTextureCurrentAlphaMass {0.0f};
        float                               coverageTextureCurrentAlpha2Mass {0.0f};
        uint32_t                            coverageTextureNonzeroTexelCount {0u};
        uint32_t                            coverageTextureWidth {0u};
        uint32_t                            coverageTextureHeight {0u};
        float                               coverageTextureHistoryBeta {0.0f};
        float                               coverageTextureStrength {0.0f};
        float                               coverageTexturePFinalMean {0.0f};
        uint32_t                            coverageTextureStableHashKeptCount {0u};
        bool                                guideBeforeDiscardEnabled {false};
        uint32_t                            guideBeforeDiscardTexelCount {0u};
        float                               guideBeforeDiscardCoverageMean {0.0f};
        float                               guideBeforeDiscardCoverageMin {0.0f};
        float                               guideBeforeDiscardCoverageMax {0.0f};
        uint32_t                            guideBeforeDiscardLowCoverageBoostedSplats {0u};
        uint32_t                            guideBeforeDiscardHighCoverageReducedSplats {0u};
        uint32_t                            guideBeforeDiscardSelectedBeforeTileExpansion {0u};
        uint32_t                            guideBeforeDiscardTileDuplicateCount {UINT32_MAX};
        bool                                delayedGuideLogPolarEnabled {false};
        bool                                delayedGuideTemporalLogPolarEnabled {false};
        uint32_t                            delayedGuideReleaseActiveCount {0u};
        uint32_t                            delayedGuidePHistoryNonzeroCount {0u};
        uint32_t                            delayedGuidePTargetLessThanHistoryCount {0u};
        uint64_t                            delayedGuidePHistoryBytes {0u};
        uint32_t                            delayedGuideReleasedEffectiveVisibleCount {0u};
        float                               delayedGuideReleasedAlphaProxySum {0.0f};
        uint32_t                            delayedGuideReleaseCapHitCount {0u};
        uint32_t                            delayedGuideRiskProtectedCount {0u};
        uint32_t                            delayedGuideFootprintFastDecayCount {0u};
        uint32_t                            delayedGuideBaseTileDuplicateProxyCount {0u};
        uint32_t                            delayedGuideReleasedTileDuplicateProxyCount {0u};
        float                               delayedGuideReleasedDuplicateProxyRatio {0.0f};
        uint32_t                            delayedGuideLargeFootprintReleasedCount {0u};
        uint32_t                            delayedGuideLargeFootprintReleasedTileDuplicateProxyCount {0u};
        float                               delayedGuideReleasedRadiusMean {0.0f};
        float                               delayedGuideReleasedRadiusP95 {0.0f};
        float                               delayedGuideReleasedRadiusMax {0.0f};
        std::array<uint32_t, kGaussianSplatReleaseCostHotspotCount> delayedGuideHotspotTileIds {};
        std::array<uint32_t, kGaussianSplatReleaseCostHotspotCount> delayedGuideHotspotBaseDuplicates {};
        std::array<uint32_t, kGaussianSplatReleaseCostHotspotCount> delayedGuideHotspotReleasedDuplicates {};
        std::array<uint32_t, kGaussianSplatEcsptEventBinCounterCount> eccStochasticEventBinCounters {};
        GaussianSplatShaderAntiPopMode      shaderAntiPopMode {GaussianSplatShaderAntiPopMode::eOff};
        bool                                shaderAntiPopEnabled {false};
        bool                                shaderAntiPopAlphaMultiplierBased {false};
        bool                                shaderAntiPopAvoidsCpuSelectedSourceRebuild {false};
        bool                                shaderAntiPopStableCandidateSet {false};
        bool                                shaderAntiPopDirectPrefixStable {false};
        bool                                shaderAntiPopDebugLogEnabled {false};
        uint32_t                            shaderAntiPopHashSeed {0u};
        float                               shaderAntiPopRampWidth {0.05f};
        float                               shaderAntiPopGuardThreshold {0.0f};
        float                               shaderAntiPopGuardFloor {0.0f};
        GaussianSplatShaderAntiPopPKeepCurve shaderAntiPopPKeepCurve {
            GaussianSplatShaderAntiPopPKeepCurve::eCurrent};
        float                               shaderAntiPopPrefixRatio {-1.0f};
        GaussianSplatShaderAntiPopNormalizeMode shaderAntiPopNormalizeMode {
            GaussianSplatShaderAntiPopNormalizeMode::eOff};
        float                               shaderAntiPopNormalizeStrength {1.0f};
        float                               shaderAntiPopNormalizeClampMin {0.75f};
        float                               shaderAntiPopNormalizeClampMax {1.15f};
        float                               shaderAntiPopNormalizeFactor {1.0f};
        uint32_t                            shaderAntiPopCandidateCount {0u};
        int32_t                             shaderAntiPopEffectiveNonzeroEstimate {-1};
        float                               shaderAntiPopWeightMinEstimate {-1.0f};
        float                               shaderAntiPopWeightMeanEstimate {-1.0f};
        float                               shaderAntiPopWeightMaxEstimate {-1.0f};
        double                              shaderAntiPopAlphaMassEstimate {-1.0};
        uint32_t                            shaderAntiPopGuardProxyMode {0u}; // 0 none, 1 ordered rank/importance proxy.
        uint32_t splatAssets {0};
        uint32_t drawRecords {0};
        uint32_t totalSplats {0};
        uint32_t preparedSplats {0};
        uint32_t foveaSplatBudget {0};
        uint32_t midSplatBudget {0};
        uint32_t outerSplatBudget {0};
        uint32_t maxVisibleSplatCap {0};
        uint32_t lodSelectedRawSplats {0};
        bool     selectedIdDebugLoggingEnabled {false};
        std::vector<uint32_t> selectedSourceIds;
        bool     cachedSelectionOracleEnabled {false};
        uint32_t cachedSelectionOracleCandidateCount {0};
        uint32_t cachedSelectionOracleCacheFrameIndex {UINT32_MAX};
        bool     cachedSelectionOracleMembershipRecomputedThisFrame {false};
        bool     cachedSelectionOracleRenderBufferMatchesLoggedSet {false};
        bool     cachedSelectionOracleShaderFoveationDisabled {false};
        float    cachedSelectionOracleSelectedFraction {0.0f};
        float    cachedSelectionOracleThresholdFoveaDegrees {0.0f};
        float    cachedSelectionOracleThresholdMidDegrees {0.0f};
        float    cachedSelectionOracleFalloffTransitionDegrees {0.0f};
        float    cachedSelectionOracleLevelCenter {1.0f};
        float    cachedSelectionOracleLevelMid {1.0f};
        float    cachedSelectionOracleLevelOuter {1.0f};
        float    cachedSelectionOracleEccentricityMin {0.0f};
        float    cachedSelectionOracleEccentricityMean {0.0f};
        float    cachedSelectionOracleEccentricityMax {0.0f};
        float    cachedSelectionOracleKeepProbabilityMin {0.0f};
        float    cachedSelectionOracleKeepProbabilityMean {0.0f};
        float    cachedSelectionOracleKeepProbabilityMax {0.0f};
        uint32_t cachedSelectionOracleMatchedNullTargetAddCount {0u};
        uint32_t cachedSelectionOracleMatchedNullTargetRemoveCount {0u};
        uint32_t cachedSelectionOracleMatchedNullTargetSelectedCount {0u};
        uint32_t cachedSelectionOracleMatchedNullTargetSymmetricDiff {0u};
        uint32_t cachedSelectionOracleMatchedNullRealizedAddCount {0u};
        uint32_t cachedSelectionOracleMatchedNullRealizedRemoveCount {0u};
        uint32_t cachedSelectionOracleMatchedNullRealizedSelectedCount {0u};
        uint32_t cachedSelectionOracleMatchedNullRealizedSymmetricDiff {0u};
        bool     cachedSelectionOracleMatchedNullScheduleMatched {true};
        bool     cachedSelectionOracleForcedSourceIdsEnabled {false};
        uint32_t cachedSelectionOracleForcedSourceIdsRequestedCount {0u};
        uint32_t cachedSelectionOracleForcedSourceIdsMatchedCount {0u};
        bool     cachedSelectionOracleForcedSourceIdsAllFound {true};
        std::vector<GaussianSplatCachedSelectionOracleDebugSample>
            cachedSelectionOracleCandidateDebugSamples;
        uint64_t numTileIntersections {0};
        double   sumProjectedAreaPx {0.0};

        bool     cleanTimingMode {false};
        uint32_t visibleSplats {UINT32_MAX};
        uint32_t drawnSplats {UINT32_MAX};
        uint32_t visibleInstant {UINT32_MAX};
        uint32_t tileInstances {UINT32_MAX};
        uint32_t coveredPixelCount {UINT32_MAX};
    };

    // Double-buffered cooked scene for rendering.
    struct RenderWorld
    {
        uint64_t                    frameIndex {0};
        std::vector<RenderCamera>   cameras;
        std::vector<RenderInstance> instances;
        std::vector<RenderGaussianSplatInstance> gaussianSplats;

        resource::GpuSceneDatabase* gpuSceneDatabase {nullptr};
        resource::GpuSceneView*     gpuSceneView {nullptr};

        void clear()
        {
            cameras.clear();
            instances.clear();
            gaussianSplats.clear();
        }
    };
} // namespace vultra

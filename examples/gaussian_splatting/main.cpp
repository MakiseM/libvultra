#include <vultra/core/app/demo_app_host.hpp>
#include <vultra/core/base/common_context.hpp>
#include <vultra/core/rhi/texture.hpp>
#include <vultra/core/services/input_service.hpp>
#include <vultra/core/services/window_service.hpp>
#include <vultra/function/services/camera_service.hpp>
#include <vultra/function/services/frame_debugger_service.hpp>
#include <vultra/function/rendering/render_structs.hpp>
#include <vultra/function/rendering/runtime_profiler.hpp>
#include <vultra/function/services/asset_service.hpp>
#include <vultra/function/services/render_backend_service.hpp>
#include <vultra/function/services/render_service.hpp>
#include <vultra/function/services/scene_service.hpp>
#include <vultra/function/services/world_service.hpp>
#include <vultra/function/world/components/gaussian_splat_component.hpp>
#include <vultra/function/world/components/mesh_component.hpp>
#include <vultra/function/world/components/name_component.hpp>
#include <vultra/function/world/components/transform_component.hpp>

#include "gaussian_splatting_benchmark.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <complex>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>

using namespace vultra;
using namespace vultra::gaussian_splatting_example;

namespace
{
    enum class BenchmarkCameraPath : uint8_t
    {
        eStatic,
        eDataset,
    };

    enum class BenchmarkGazePath : uint8_t
    {
        eStatic,
        eGrid,
        eGridStep3x3,
        eSlowHorizontalSweep,
        eMediumHorizontalSweep,
        eFastHorizontalSweep,
        eSlowVerticalSweep,
        eSaccadeJumpCenterToEdge,
        eSaccadeJumpEdgeToCenter,
        eStepJumpCenterCornerCenter,
        eRepeatedStepJump,
    };

    struct BenchmarkGazeFrame
    {
        glm::vec2 gaze {0.5f, 0.5f};
        uint32_t  scriptedGazeIndex {0};
        uint32_t  isMoving {0};
        uint32_t  jumpEvent {0};
        int32_t   transitionWindowId {-1};
    };

    struct CachedSelectionOracleLogRow
    {
        uint32_t    sampleIndex {0};
        uint64_t    frameIndex {0};
        std::string membershipMode;
        glm::vec2   gaze {0.5f, 0.5f};
        uint32_t    candidateCount {0};
        uint32_t    selectedCount {0};
        double      selectedFraction {0.0};
        uint64_t    selectedSetHash {0u};
        std::string selectedSourceIdRanges;
        uint32_t    symmetricDifference {0};
        double      jaccard {1.0};
        uint32_t    seed {0u};
        std::string cacheModeStatus;
        uint32_t    cacheFrameIndexUsed {UINT32_MAX};
        uint32_t    membershipRecomputedThisFrame {0u};
        uint32_t    renderBufferMatchesLoggedSet {0u};
        uint32_t    shaderSideFoveationDisabled {0u};
        float       thresholdFoveaDegrees {0.0f};
        float       thresholdMidDegrees {0.0f};
        float       falloffTransitionDegrees {0.0f};
        float       levelCenter {1.0f};
        float       levelMid {1.0f};
        float       levelOuter {1.0f};
        float       eccentricityMin {0.0f};
        float       eccentricityMax {0.0f};
        float       keepProbabilityMin {0.0f};
        float       keepProbabilityMax {0.0f};
        uint32_t    pass {1u};
    };

    enum class FormalProtocolCondition : uint8_t
    {
        eNone = 0,
        eC2DynamicDeterministic,
        eC1StaticRandom,
        eC1StaticMeanCount,
        eC3DynamicFrozenHash,
        eC2NullMatchedRandomSwap,
    };

    struct FormalProtocolFrameRow
    {
        uint32_t sampleIndex {0};
        uint32_t localFrameIndex {0};
        uint64_t frameIndex {0};
        FormalProtocolCondition condition {FormalProtocolCondition::eNone};
        GaussianSplatCachedSelectionMembershipMode membershipMode {
            GaussianSplatCachedSelectionMembershipMode::eDynamicDeterministic};
        uint32_t seed {0};
        glm::vec2 gaze {0.5f, 0.5f};
        glm::vec2 gazeVelocity {0.0f, 0.0f};
        double gazeSpeed {0.0};
        uint32_t candidateCount {0};
        uint32_t selectedCount {0};
        double selectedFraction {0.0};
        uint64_t selectedSetHash {0u};
        std::string selectedSourceIdRanges;
        uint32_t addedCount {0};
        uint32_t removedCount {0};
        uint32_t symmetricDifference {0};
        double normalizedChurn {0.0};
        double jaccard {1.0};
        uint32_t cacheFrameIndexUsed {UINT32_MAX};
        uint32_t membershipRecomputedThisFrame {0u};
        uint32_t renderBufferMatchesLoggedSet {0u};
        uint32_t shaderSideFoveationDisabled {0u};
        float thresholdFoveaDegrees {0.0f};
        float thresholdMidDegrees {0.0f};
        float falloffTransitionDegrees {0.0f};
        float levelCenter {1.0f};
        float levelMid {1.0f};
        float levelOuter {1.0f};
        float eccentricityMin {0.0f};
        float eccentricityMean {0.0f};
        float eccentricityMax {0.0f};
        float keepProbabilityMin {0.0f};
        float keepProbabilityMean {0.0f};
        float keepProbabilityMax {0.0f};
        uint32_t c2TargetAddCount {0};
        uint32_t c2TargetRemoveCount {0};
        uint32_t c2TargetSelectedCount {0};
        uint32_t c2TargetSymmetricDiff {0};
        uint32_t c2RealizedAddCount {0};
        uint32_t c2RealizedRemoveCount {0};
        uint32_t c2RealizedSelectedCount {0};
        uint32_t c2RealizedSymmetricDiff {0};
        uint32_t c2ChangedIdOverlap {0};
        uint32_t scheduleMatchingSucceeded {1u};
        std::string runLabel;
        std::string cacheModeStatus;
        std::string notes;
    };

    struct FormalProtocolScheduleRow
    {
        uint32_t localFrameIndex {0};
        uint64_t frameIndex {0};
        uint32_t addCount {0};
        uint32_t removeCount {0};
        uint32_t selectedCount {0};
        uint32_t symmetricDifference {0};
        double normalizedChurn {0.0};
        double jaccard {1.0};
        uint64_t selectedSetHash {0u};
        std::vector<uint32_t> changedIds;
    };

    struct FormalProtocolConditionState
    {
        bool previousValid {false};
        glm::vec2 previousGaze {0.5f, 0.5f};
        std::vector<uint32_t> previousIds;
    };

    struct FormalChurnEventRow
    {
        uint32_t localFrameIndex {0};
        uint64_t frameIndex {0};
        FormalProtocolCondition condition {FormalProtocolCondition::eNone};
        GaussianSplatCachedSelectionMembershipMode membershipMode {
            GaussianSplatCachedSelectionMembershipMode::eDynamicDeterministic};
        uint32_t sourceId {0};
        bool enter {false};
        glm::vec2 gaze {0.5f, 0.5f};
        float eccentricity {0.0f};
        float alphaProxy {0.0f};
        float postWeightAlphaProxy {0.0f};
        double projectedRadiusProxy {0.0};
        double footprintAreaProxy {0.0};
        uint64_t tileCost {0u};
        double transmittance {0.0};
        double alphaT {0.0};
        double contributionProxy {0.0};
        bool exactAlphaTAvailable {false};
        bool exactTransmittanceAvailable {false};
    };

    struct ImageFlickerFrameRow
    {
        uint32_t sampleIndex {0};
        uint32_t localFrameIndex {0};
        uint64_t frameIndex {0};
        std::string condition;
        GaussianSplatCachedSelectionMembershipMode membershipMode {
            GaussianSplatCachedSelectionMembershipMode::eDynamicDeterministic};
        uint32_t seed {0};
        glm::vec2 gaze {0.5f, 0.5f};
        glm::vec2 gazeVelocity {0.0f, 0.0f};
        double gazeSpeed {0.0};
        uint32_t selectedCount {0};
        uint32_t symmetricDifference {0};
        double normalizedChurn {0.0};
        double jaccard {1.0};
        uint32_t width {0};
        uint32_t height {0};
        float fovealRadiusDegrees {12.0f};
        float verticalFovDegrees {60.0f};
        std::string bandLabel;
        float bandMinDegrees {0.0f};
        float bandMaxDegrees {0.0f};
        uint64_t pixelCount {0};
        uint32_t hasPreviousFrame {0};
        double rawDeltaEnergy {0.0};
        double rawDeltaMean {0.0};
        double hpDeltaEnergy {0.0};
        double hpDeltaMean {0.0};
        uint32_t hpAvailable {0};
        double weightedLowFrequencyFraction {0.0};
        double weightedMidFrequencyFraction {0.0};
        double weightedHighFrequencyFraction {0.0};
        double weightedSpectrumTotalPower {0.0};
        uint32_t weightedSpectrumAvailable {0};
        std::string colorSpace;
        std::string captureTargetType;
        std::string imageFormat;
        uint32_t sampleCount {1};
        uint64_t frameImageHash {0u};
        uint32_t preUiCapture {0};
        uint32_t postTonemapCapture {0};
        uint32_t temporalEffectsDisabled {0};
        uint32_t overlaysDisabled {0};
        uint32_t readbackSynchronized {0};
        uint64_t selectedSetHash {0u};
        uint32_t renderLogSetMatch {0};
        std::string runLabel;
    };

    struct ImageFlickerPreviousFrame
    {
        bool valid {false};
        uint32_t width {0};
        uint32_t height {0};
        glm::vec2 gaze {0.5f, 0.5f};
        std::vector<float> luminance;
    };

    struct ImageFlickerReadback
    {
        bool valid {false};
        uint32_t width {0};
        uint32_t height {0};
        uint64_t frameImageHash {0u};
        std::string imageFormat;
        std::vector<float> luminance;
    };

    struct ImageFlickerBandSpec
    {
        std::string label;
        float minDegrees {0.0f};
        float maxDegrees {0.0f};
    };

    struct E3HonestPendingCounterfactual
    {
        bool valid {false};
        uint32_t sampleIndex {0};
        uint32_t localFrameIndex {0};
        uint64_t actualFrameIndex {0};
        FormalProtocolCondition condition {FormalProtocolCondition::eNone};
        GaussianSplatCachedSelectionMembershipMode membershipMode {
            GaussianSplatCachedSelectionMembershipMode::eDynamicDeterministic};
        glm::vec2 gaze {0.5f, 0.5f};
        glm::vec2 gazeVelocity {0.0f, 0.0f};
        double gazeSpeed {0.0};
        uint32_t selectedCount {0};
        uint32_t previousSelectedCount {0};
        uint32_t symmetricDifference {0};
        double normalizedChurn {0.0};
        double jaccard {1.0};
        uint64_t selectedSetHash {0u};
        uint64_t previousSelectedSetHash {0u};
        std::vector<uint32_t> previousSourceIds;
        std::vector<uint32_t> currentSourceIds;
        std::vector<uint32_t> changedSourceIds;
        ImageFlickerReadback actualReadback;
    };

    struct E3PixelDeltaFrameRow
    {
        uint32_t sampleIndex {0};
        uint32_t localFrameIndex {0};
        uint64_t actualFrameIndex {0};
        uint64_t forcedFrameIndex {0};
        FormalProtocolCondition condition {FormalProtocolCondition::eNone};
        GaussianSplatCachedSelectionMembershipMode membershipMode {
            GaussianSplatCachedSelectionMembershipMode::eDynamicDeterministic};
        uint32_t seed {0u};
        glm::vec2 gaze {0.5f, 0.5f};
        glm::vec2 gazeVelocity {0.0f, 0.0f};
        double gazeSpeed {0.0};
        uint32_t selectedCount {0u};
        uint32_t previousSelectedCount {0u};
        uint32_t forcedSelectedCount {0u};
        uint32_t symmetricDifference {0u};
        double normalizedChurn {0.0};
        double jaccard {1.0};
        uint64_t selectedSetHash {0u};
        uint64_t previousSelectedSetHash {0u};
        uint64_t forcedSelectedSetHash {0u};
        uint32_t width {0u};
        uint32_t height {0u};
        std::string bandLabel;
        float bandMinDegrees {0.0f};
        float bandMaxDegrees {0.0f};
        uint64_t pixelCount {0u};
        uint64_t nonzeroPixelCount {0u};
        double rawPixelDeltaEnergy {0.0};
        double rawPixelDeltaMean {0.0};
        double hpPixelDeltaEnergy {0.0};
        double hpPixelDeltaMean {0.0};
        uint32_t hpAvailable {1u};
        uint64_t actualFrameImageHash {0u};
        uint64_t forcedFrameImageHash {0u};
        uint32_t forcedIdsRequestedCount {0u};
        uint32_t forcedIdsMatchedCount {0u};
        uint32_t forcedIdsAllFound {0u};
        uint32_t renderLogSetMatch {0u};
        uint32_t shaderSideFoveationDisabled {0u};
        uint32_t deltaNonzero {0u};
        std::string colorSpace;
        std::string captureTargetType;
        std::string imageFormat;
        std::string runLabel;
        std::string notes;
    };

    struct E6SpectrumBinRow
    {
        uint32_t sampleIndex {0};
        uint32_t localFrameIndex {0};
        uint64_t actualFrameIndex {0};
        uint64_t forcedFrameIndex {0};
        FormalProtocolCondition condition {FormalProtocolCondition::eNone};
        std::string fieldType;
        uint32_t radiusBin {0u};
        double normalizedRadius {0.0};
        double power {0.0};
        uint32_t binSampleCount {0u};
        std::string runLabel;
        std::string notes;
    };

    struct E6FrequencyBandRow
    {
        uint32_t sampleIndex {0};
        uint32_t localFrameIndex {0};
        uint64_t actualFrameIndex {0};
        uint64_t forcedFrameIndex {0};
        FormalProtocolCondition condition {FormalProtocolCondition::eNone};
        std::string fieldType;
        uint32_t selectedCount {0u};
        uint32_t previousSelectedCount {0u};
        uint32_t symmetricDifference {0u};
        double normalizedChurn {0.0};
        double lowFrequencyFraction {0.0};
        double midFrequencyFraction {0.0};
        double highFrequencyFraction {0.0};
        double totalPower {0.0};
        std::string runLabel;
        std::string notes;
    };

    struct GaussianDemoOptions
    {
        std::optional<GaussianSplatBaselineMode>       mode;
        std::optional<float>                           clodLevel;
        std::optional<uint32_t>                        lodBudget;
        std::optional<GaussianSplatLodBudgetMode>      lodBudgetMode;
        std::optional<uint64_t>                        projectedCostBudget;
        std::optional<float>                           projectedCostBudgetRatio;
        std::optional<uint32_t>                        projectedCostChunkSize;
        std::optional<uint32_t>                        foveatedScoreSelectionStride;
        std::optional<uint32_t>                        foveatedCoverageBinGridX;
        std::optional<uint32_t>                        foveatedCoverageBinGridY;
        std::optional<float>                           foveatedCoverageBinQuotaRatio;
        std::optional<float>                           foveatedCoverageBinPrefixRatio;
        std::optional<GaussianProjectedCostBuildMode>  projectedCostBuildMode;
        std::optional<bool>                            projectedCostSemanticDiffEnabled;
        std::optional<std::string>                     splatUri;
        std::optional<bool>                            foveatedClodEnabled;
        std::optional<bool>                            foveatedCoverageCompensationEnabled;
        std::optional<GaussianSplatFoveatedRenderMode> foveatedRenderMode;
        std::optional<float>                           gazeX;
        std::optional<float>                           gazeY;
        std::optional<float>                           foveaDegrees;
        std::optional<float>                           midDegrees;
        std::optional<float>                           foveaLod;
        std::optional<float>                           midLod;
        std::optional<float>                           outerLod;
        std::optional<float>                           foveaResolutionScale;
        std::optional<float>                           midResolutionScale;
        std::optional<float>                           outerResolutionScale;
        std::optional<float>                           transitionDegrees;
        std::optional<GaussianSplatFoveatedDistribution> foveatedDistribution;
        bool                                             mouseGazeControl {false};
        std::optional<float>                           continuousTheta0Degrees;
        std::optional<float>                           continuousAlpha;
        std::optional<float>                           continuousMinLevel;
        std::optional<GaussianSplatFoveatedAdaptationMode> foveatedAdaptationMode;
        std::optional<float>                           targetFrameMs;
        std::optional<float>                           budgetAdjustRate;
        std::optional<GaussianSplatFoveatedCoverageGuardMode> coverageGuardMode;
        std::optional<float>                           coverageProtectionDegrees;
        std::optional<float>                           coverageGuardBudgetRatio;
        std::optional<float>                           coverageGuardCenterMin;
        std::optional<float>                           coverageGuardTransitionMin;
        std::optional<float>                           coverageGuardPeripheryMin;
        std::optional<uint32_t>                        coverageGuardMaxAdds;
        std::optional<bool>                            foveatedTemporalHysteresisEnabled;
        std::optional<bool>                            foveatedBoundarySmoothingEnabled;
        std::optional<uint32_t>                        foveatedTemporalResidencyFrames;
        std::optional<float>                           foveatedTemporalHysteresisRatio;
        std::optional<float>                           foveatedBoundarySmoothingRatio;
        std::optional<float>                           foveatedTemporalPeripheralScale;
        std::optional<bool>                            foveatedShLodEnabled;
        std::optional<bool>                            foveatedShSmoothSuppressionEnabled;
        std::optional<bool>                            peripheralTemporalFilterEnabled;
        std::optional<uint32_t>                        shDegreeCenter;
        std::optional<uint32_t>                        shDegreeMid;
        std::optional<uint32_t>                        shDegreeOuter;
        std::optional<GaussianSplatShLodGuardMode>     shLodGuardMode;
        std::optional<float>                           shLodGuardThresholdMid;
        std::optional<float>                           shLodGuardThresholdHigh;
        std::optional<GaussianSplatShStorageLayout>    shStorageLayout;
        std::optional<bool>                            shPopLogEnabled;
        std::optional<bool>                            shDegreeHysteresisEnabled;
        std::optional<uint32_t>                        shDegreeDowngradeDelay;
        std::optional<float>                           shDegreeGuardBandDegrees;
        std::optional<GaussianSplatShaderAntiPopMode>    shaderAntiPopMode;
        std::optional<uint32_t>                          shaderAntiPopHashSeed;
        std::optional<float>                             shaderAntiPopRampWidth;
        std::optional<float>                             shaderAntiPopGuardThreshold;
        std::optional<float>                             shaderAntiPopGuardFloor;
        std::optional<GaussianSplatShaderAntiPopPKeepCurve> shaderAntiPopPKeepCurve;
        std::optional<float>                             shaderAntiPopPrefixRatio;
        std::optional<GaussianSplatShaderAntiPopNormalizeMode> shaderAntiPopNormalizeMode;
        std::optional<float>                             shaderAntiPopNormalizeStrength;
        std::optional<float>                             shaderAntiPopNormalizeClampMin;
        std::optional<float>                             shaderAntiPopNormalizeClampMax;
        std::optional<float>                             shaderAntiPopNormalizeFactor;
        std::optional<bool>                              coverageStableReleaseCoverageFloorEnabled;
        std::optional<bool>                              coverageStableReleaseStableHashEnabled;
        std::optional<bool>                              coverageStableReleaseStaggeredReleaseEnabled;
        std::optional<bool>                              coverageStableReleaseSaturatedFloorEnabled;
        std::optional<float>                             coverageStableReleaseLambda;
        std::optional<uint32_t>                          coverageStableReleaseDurationFrames;
        std::optional<float>                             coverageStableReleaseDMin;
        std::optional<float>                             coverageStableReleaseSigmaMax;
        std::optional<float>                             coverageStableReleaseFloorEma;
        std::optional<float>                             coverageStableReleaseReleaseEpsilon;
        std::optional<uint32_t>                          coverageStableReleaseTileGridX;
        std::optional<uint32_t>                          coverageStableReleaseTileGridY;
        std::optional<bool>                              coverageTextureFloorEnabled;
        std::optional<uint32_t>                          coverageTextureFloorWidth;
        std::optional<uint32_t>                          coverageTextureFloorHeight;
        std::optional<float>                             coverageTextureFloorDMin;
        std::optional<float>                             coverageTextureFloorSigmaMax;
        std::optional<float>                             coverageTextureFloorStrength;
        std::optional<float>                             coverageTextureFloorHistoryBeta;
        std::optional<uint32_t>                          coverageTextureFloorUpdateInterval;
        std::optional<bool>                              coverageTextureFloorDebugEnabled;
        std::optional<bool>                              guideBeforeDiscardEnabled;
        std::optional<bool>                              delayedGuideLogPolarEnabled;
        std::optional<bool>                              delayedGuideTemporalLogPolarEnabled;
        std::optional<GaussianSplatDelayedGuideTemporalReleasePolicy> delayedGuideTemporalReleasePolicy;
        std::optional<float>                             delayedGuideTemporalReleaseCapRatio;
        std::optional<float>                             delayedGuideTemporalRiskThreshold;
        std::optional<float>                             delayedGuideTemporalFootprintDecayScale;
        std::optional<float>                             delayedGuideTemporalLargeFootprintPx;
        bool                                             shaderAntiPopDebugLog {false};
        std::optional<std::filesystem::path>             shaderAntiPopOutputDir;
        bool                                             gpuAntiPopCliNaming {false};
        bool                                             runtimeAntiPopOrderFreeHashRamp {false};

        bool                  benchmarkEnabled {false};
        std::string           sceneUri {"res://scenes/3dgs_example.vmanifest"};
        uint32_t              benchmarkFrames {300};
        uint32_t              warmupFrames {60};
        uint32_t              benchmarkSettleFrames {4};
        BenchmarkCameraPath   benchmarkCameraPath {BenchmarkCameraPath::eStatic};
        BenchmarkGazePath     benchmarkGazePath {BenchmarkGazePath::eGrid};
        uint32_t              benchmarkViewFrames {4};
        bool                  benchmarkStereoSequential {false};
        bool                  cleanTiming {false};
        std::optional<uint32_t> renderDocCaptureSample;
        float                 benchmarkIpdMeters {0.064f};
        bool                  selectedIdLogEnabled {false};
        uint32_t              selectedIdLogStride {1};
        std::optional<std::filesystem::path> benchmarkCameraFile;
        std::optional<std::filesystem::path> benchmarkScreenshotOutput;
        bool                  captureFrameSequence {false};
        std::filesystem::path captureFrameDir {"build/gaussian_splat_frame_capture"};
        std::string           captureFramePrefix {"frame"};
        uint32_t              captureFrameLimit {0};
        bool                  captureAlpha {false};
        std::filesystem::path captureAlphaDir {"build/gaussian_splat_alpha_capture"};
        uint32_t              captureAlphaLimit {0};
        std::filesystem::path outputPath {"build/gaussian_splat_benchmark.csv"};
        bool                  cachedSelectionOracle {false};
        GaussianSplatCachedSelectionMembershipMode membershipMode {
            GaussianSplatCachedSelectionMembershipMode::eDynamicDeterministic};
        std::optional<glm::vec2> fixedGaze;
        bool                  logChurn {false};
        uint32_t              seed {0u};
        std::filesystem::path outputDir {"build/gaussian_splat_e0_oracle"};
        bool                  formalE1E2E3 {false};
        bool                  formalExploratoryThresholdUncommitted {true};
        bool                  logFlicker {false};
        float                 flickerFovealRadiusDegrees {12.0f};
        float                 flickerVerticalFovDegrees {60.0f};
        bool                  e3HonestE6 {false};
        uint32_t              e6SpectrumGridSize {64u};
    };

    std::string normalizeToken(const std::string_view value)
    {
        std::string normalized;
        normalized.reserve(value.size());
        for (const char ch : value)
        {
            if (ch == '_')
            {
                normalized.push_back('-');
                continue;
            }
            normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
        return normalized;
    }

    std::optional<std::string_view> takeOptionValue(std::span<const std::string> args,
                                                    size_t&                      index,
                                                    const std::string_view       option)
    {
        const std::string_view arg = args[index];
        if (arg == option)
        {
            if (index + 1u >= args.size())
                return std::string_view {};
            return args[++index];
        }

        if (arg.size() > option.size() && arg.starts_with(option) && arg[option.size()] == '=')
            return arg.substr(option.size() + 1u);

        return std::nullopt;
    }

    std::optional<uint32_t> parseU32(const std::string_view value)
    {
        try
        {
            const std::string text {value};
            size_t            parsedChars = 0;
            const auto        parsed      = std::stoull(text, &parsedChars, 10);
            if (parsedChars != text.size() || parsed > std::numeric_limits<uint32_t>::max())
                return std::nullopt;
            return static_cast<uint32_t>(parsed);
        }
        catch (...)
        {
            return std::nullopt;
        }
    }

    std::optional<uint64_t> parseU64(const std::string_view value)
    {
        try
        {
            const std::string text {value};
            size_t            parsedChars = 0;
            const auto        parsed      = std::stoull(text, &parsedChars, 10);
            if (parsedChars != text.size())
                return std::nullopt;
            return parsed;
        }
        catch (...)
        {
            return std::nullopt;
        }
    }

    std::optional<float> parseFloat(const std::string_view value)
    {
        try
        {
            const std::string text {value};
            size_t            parsedChars = 0;
            const float       parsed      = std::stof(text, &parsedChars);
            if (parsedChars != text.size())
                return std::nullopt;
            return parsed;
        }
        catch (...)
        {
            return std::nullopt;
        }
    }

    std::optional<bool> parseBoolFlag(const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on" ||
            normalized == "enabled")
            return true;
        if (normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off" ||
            normalized == "disabled")
            return false;
        return std::nullopt;
    }

    std::optional<glm::vec2> parseGazeUv(const std::string_view value)
    {
        const std::string text {value};
        const size_t comma = text.find(',');
        if (comma == std::string::npos || comma == 0u || comma + 1u >= text.size())
            return std::nullopt;

        const auto x = parseFloat(std::string_view {text}.substr(0u, comma));
        const auto y = parseFloat(std::string_view {text}.substr(comma + 1u));
        if (!x || !y)
            return std::nullopt;
        return glm::clamp(glm::vec2 {*x, *y}, glm::vec2 {0.0f}, glm::vec2 {1.0f});
    }

    std::optional<GaussianSplatBaselineMode> parseGaussianMode(const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "baseline" || normalized == "default")
            return GaussianSplatBaselineMode::eBaseline;
        if (normalized == "ordered" || normalized == "ordered-clod" || normalized == "clod")
            return GaussianSplatBaselineMode::eOrderedClod;
        return std::nullopt;
    }

    std::optional<GaussianSplatFoveatedRenderMode> parseFoveatedRenderMode(const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "single-pass" || normalized == "single")
            return GaussianSplatFoveatedRenderMode::eSinglePass;
        if (normalized == "layered" || normalized == "layered-composite")
            return GaussianSplatFoveatedRenderMode::eLayeredComposite;
        return std::nullopt;
    }

    std::optional<GaussianSplatLodBudgetMode> parseLodBudgetMode(const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "count" || normalized == "count-prefix")
            return GaussianSplatLodBudgetMode::eCount;
        if (normalized == "projected-cost" || normalized == "projected-tile-cost" ||
            normalized == "tile-cost" || normalized == "cost")
            return GaussianSplatLodBudgetMode::eProjectedTileCost;
        if (normalized == "foveated-score" || normalized == "score" ||
            normalized == "per-primitive-score" || normalized == "gaze-score")
            return GaussianSplatLodBudgetMode::eFoveatedScore;
        if (normalized == "coverage-bin-score" || normalized == "coverage-bin" ||
            normalized == "spatial-bin-score" || normalized == "bin-score")
            return GaussianSplatLodBudgetMode::eCoverageBinScore;
        return std::nullopt;
    }

    std::optional<GaussianSplatCachedSelectionMembershipMode> parseMembershipMode(const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "static-random")
            return GaussianSplatCachedSelectionMembershipMode::eStaticRandom;
        if (normalized == "dynamic-deterministic")
            return GaussianSplatCachedSelectionMembershipMode::eDynamicDeterministic;
        if (normalized == "dynamic-frozen-hash")
            return GaussianSplatCachedSelectionMembershipMode::eDynamicFrozenHash;
        if (normalized == "matched-rate-random-null" || normalized == "c2-null" ||
            normalized == "c2-null-matched-random-swap")
            return GaussianSplatCachedSelectionMembershipMode::eMatchedRateRandomNull;
        return std::nullopt;
    }

    std::optional<GaussianSplatShaderAntiPopMode> parseShaderAntiPopMode(const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "off" || normalized == "none")
            return GaussianSplatShaderAntiPopMode::eOff;
        if (normalized == "coverage-stable-logpolar-release" ||
            normalized == "coverage_stable_logpolar_release" ||
            normalized == "coverage-stable-release" ||
            normalized == "coverage-release" ||
            normalized == "coverage-texture-stable-release" ||
            normalized == "coverage-texture-floor" ||
            normalized == "guide-before-discard-logpolar" ||
            normalized == "cslr" ||
            normalized == "gpu-cslr")
            return GaussianSplatShaderAntiPopMode::eCoverageStableLogpolarRelease;
        return std::nullopt;
    }

    std::optional<GaussianSplatDelayedGuideTemporalReleasePolicy> parseDelayedGuideTemporalReleasePolicy(
        const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "current" || normalized == "baseline")
            return GaussianSplatDelayedGuideTemporalReleasePolicy::eCurrent;
        if (normalized == "risk-only" || normalized == "riskonly")
            return GaussianSplatDelayedGuideTemporalReleasePolicy::eRiskOnly;
        if (normalized == "capped" || normalized == "cap")
            return GaussianSplatDelayedGuideTemporalReleasePolicy::eCapped;
        if (normalized == "footprint-decay" || normalized == "footprintdecay")
            return GaussianSplatDelayedGuideTemporalReleasePolicy::eFootprintDecay;
        if (normalized == "tile-local-capped" || normalized == "tilelocalcapped" ||
            normalized == "tile-cap" || normalized == "tilecap")
            return GaussianSplatDelayedGuideTemporalReleasePolicy::eTileLocalCapped;
        return std::nullopt;
    }

    bool parseRuntimeAntiPopMethod(const std::string_view value, GaussianDemoOptions& options)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "full-render" || normalized == "full-render-reference" ||
            normalized == "full-render-all" || normalized == "full-renderer")
        {
            options.runtimeAntiPopOrderFreeHashRamp = false;
            options.shaderAntiPopMode = GaussianSplatShaderAntiPopMode::eOff;
            options.mode = GaussianSplatBaselineMode::eOrderedClod;
            options.lodBudget = 0u;
            options.clodLevel = 1.0f;
            options.foveatedClodEnabled = false;
            options.foveatedCoverageCompensationEnabled = false;
            options.foveatedTemporalHysteresisEnabled = false;
            options.foveatedBoundarySmoothingEnabled = false;
            options.foveatedShLodEnabled = false;
            options.foveatedShSmoothSuppressionEnabled = false;
            options.peripheralTemporalFilterEnabled = false;
            options.benchmarkEnabled = true;
            return true;
        }
        if (normalized == "b0" || normalized == "b0-logpolar" || normalized == "hard-logpolar")
        {
            options.runtimeAntiPopOrderFreeHashRamp = false;
            options.shaderAntiPopMode = GaussianSplatShaderAntiPopMode::eOff;
            options.mode = GaussianSplatBaselineMode::eOrderedClod;
            options.foveatedClodEnabled = true;
            options.foveatedDistribution = GaussianSplatFoveatedDistribution::eLogPolar;
            options.foveatedCoverageCompensationEnabled = false;
            options.coverageGuardMode = GaussianSplatFoveatedCoverageGuardMode::eOff;
            options.foveatedTemporalHysteresisEnabled = false;
            options.foveatedBoundarySmoothingEnabled = false;
            options.benchmarkEnabled = true;
            return true;
        }
        if (normalized == "off" || normalized == "none")
        {
            options.runtimeAntiPopOrderFreeHashRamp = false;
            options.shaderAntiPopMode = GaussianSplatShaderAntiPopMode::eOff;
            return true;
        }
        if (normalized == "coverage-texture-floor" || normalized == "coverage-texture-stable-release")
        {
            options.runtimeAntiPopOrderFreeHashRamp = true;
            options.shaderAntiPopMode = GaussianSplatShaderAntiPopMode::eCoverageStableLogpolarRelease;
            options.shaderAntiPopPKeepCurve = GaussianSplatShaderAntiPopPKeepCurve::eLinear;
            options.mode = GaussianSplatBaselineMode::eBaseline;
            options.foveatedClodEnabled = true;
            options.foveatedDistribution = GaussianSplatFoveatedDistribution::eLogPolar;
            options.coverageGuardMode = GaussianSplatFoveatedCoverageGuardMode::eOff;
            options.foveatedCoverageCompensationEnabled = false;
            options.foveatedTemporalHysteresisEnabled = false;
            options.foveatedBoundarySmoothingEnabled = false;
            options.coverageStableReleaseCoverageFloorEnabled = false;
            options.coverageTextureFloorEnabled = true;
            options.coverageTextureFloorDebugEnabled = false;
            options.benchmarkEnabled = true;
            return true;
        }
        if (normalized == "delayed-guide-logpolar" || normalized == "delayedguidelogpolar")
        {
            options.runtimeAntiPopOrderFreeHashRamp = true;
            options.shaderAntiPopMode = GaussianSplatShaderAntiPopMode::eCoverageStableLogpolarRelease;
            options.shaderAntiPopPKeepCurve = GaussianSplatShaderAntiPopPKeepCurve::eLinear;
            options.mode = GaussianSplatBaselineMode::eBaseline;
            options.foveatedClodEnabled = true;
            options.foveatedDistribution = GaussianSplatFoveatedDistribution::eLogPolar;
            options.coverageGuardMode = GaussianSplatFoveatedCoverageGuardMode::eOff;
            options.foveatedCoverageCompensationEnabled = false;
            options.foveatedTemporalHysteresisEnabled = false;
            options.foveatedBoundarySmoothingEnabled = false;
            options.coverageStableReleaseCoverageFloorEnabled = false;
            options.coverageStableReleaseStaggeredReleaseEnabled = false;
            options.coverageTextureFloorEnabled = true;
            options.coverageTextureFloorDebugEnabled = false;
            options.guideBeforeDiscardEnabled = false;
            options.delayedGuideLogPolarEnabled = true;
            options.delayedGuideTemporalLogPolarEnabled = false;
            options.benchmarkEnabled = true;
            return true;
        }
        if (normalized == "delayed-guide-temporal-logpolar" ||
            normalized == "delayedguidetemporallogpolar" ||
            normalized == "delayed-guide-temporal-logpolar-current" ||
            normalized == "delayedguidetemporallogpolarcurrent" ||
            normalized == "delayed-guide-temporal-logpolar-risk-only" ||
            normalized == "delayed-guide-temporal-logpolar-riskonly" ||
            normalized == "delayedguidetemporallogpolarriskonly" ||
            normalized == "delayed-guide-temporal-logpolar-capped" ||
            normalized == "delayedguidetemporallogpolarcapped" ||
            normalized == "delayed-guide-temporal-logpolar-footprint-decay" ||
            normalized == "delayedguidetemporallogpolarfootprintdecay" ||
            normalized == "delayed-guide-temporal-logpolar-tile-local-capped" ||
            normalized == "delayedguidetemporallogpolartilelocalcapped" ||
            normalized == "delayed-guide-temporal-logpolar-tile-cap" ||
            normalized == "delayedguidetemporallogpolartilecap")
        {
            options.runtimeAntiPopOrderFreeHashRamp = true;
            options.shaderAntiPopMode = GaussianSplatShaderAntiPopMode::eCoverageStableLogpolarRelease;
            options.shaderAntiPopPKeepCurve = GaussianSplatShaderAntiPopPKeepCurve::eLinear;
            options.mode = GaussianSplatBaselineMode::eBaseline;
            options.foveatedClodEnabled = true;
            options.foveatedDistribution = GaussianSplatFoveatedDistribution::eLogPolar;
            options.coverageGuardMode = GaussianSplatFoveatedCoverageGuardMode::eOff;
            options.foveatedCoverageCompensationEnabled = false;
            options.foveatedTemporalHysteresisEnabled = false;
            options.foveatedBoundarySmoothingEnabled = false;
            options.coverageStableReleaseCoverageFloorEnabled = false;
            options.coverageStableReleaseStaggeredReleaseEnabled = true;
            options.coverageTextureFloorEnabled = true;
            options.coverageTextureFloorDebugEnabled = false;
            options.guideBeforeDiscardEnabled = false;
            options.delayedGuideLogPolarEnabled = true;
            options.delayedGuideTemporalLogPolarEnabled = true;
            if (normalized.find("risk") != std::string::npos)
                options.delayedGuideTemporalReleasePolicy =
                    GaussianSplatDelayedGuideTemporalReleasePolicy::eRiskOnly;
            else if (normalized.find("tile") != std::string::npos)
                options.delayedGuideTemporalReleasePolicy =
                    GaussianSplatDelayedGuideTemporalReleasePolicy::eTileLocalCapped;
            else if (normalized.find("capped") != std::string::npos)
                options.delayedGuideTemporalReleasePolicy =
                    GaussianSplatDelayedGuideTemporalReleasePolicy::eCapped;
            else if (normalized.find("footprint") != std::string::npos)
                options.delayedGuideTemporalReleasePolicy =
                    GaussianSplatDelayedGuideTemporalReleasePolicy::eFootprintDecay;
            else
                options.delayedGuideTemporalReleasePolicy =
                    GaussianSplatDelayedGuideTemporalReleasePolicy::eCurrent;
            options.benchmarkEnabled = true;
            return true;
        }
        return false;
    }

    bool parseRuntimeAntiPopShLod(const std::string_view value, GaussianDemoOptions& options)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "off" || normalized == "none")
        {
            options.foveatedShSmoothSuppressionEnabled = false;
            return true;
        }
        if (normalized == "peripheral-smooth" || normalized == "smooth-peripheral" ||
            normalized == "m2" || normalized == "m2-peripheral-smooth")
        {
            options.foveatedShSmoothSuppressionEnabled = true;
            options.foveatedShLodEnabled = false;
            options.foveatedClodEnabled = true;
            options.benchmarkEnabled = true;
            return true;
        }
        return false;
    }

    bool parseRuntimeAntiPopTemporal(const std::string_view value, GaussianDemoOptions& options)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "off" || normalized == "none")
        {
            options.peripheralTemporalFilterEnabled = false;
            return true;
        }
        if (normalized == "peripheral-normalized" || normalized == "normalized-peripheral" ||
            normalized == "m3" || normalized == "m3-peripheral-temporal")
        {
            options.peripheralTemporalFilterEnabled = true;
            options.foveatedClodEnabled = true;
            options.benchmarkEnabled = true;
            return true;
        }
        return false;
    }

    std::optional<GaussianSplatShaderAntiPopNormalizeMode> parseShaderAntiPopNormalizeMode(
        const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "off" || normalized == "none" || normalized == "disabled")
            return GaussianSplatShaderAntiPopNormalizeMode::eOff;
        if (normalized == "global-luma" || normalized == "global-luminance" || normalized == "luma")
            return GaussianSplatShaderAntiPopNormalizeMode::eGlobalLuma;
        if (normalized == "alpha-mass" || normalized == "alphamass")
            return GaussianSplatShaderAntiPopNormalizeMode::eAlphaMass;
        return std::nullopt;
    }

    std::optional<GaussianSplatShaderAntiPopPKeepCurve> parseShaderAntiPopPKeepCurve(const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "current" || normalized == "default")
            return GaussianSplatShaderAntiPopPKeepCurve::eCurrent;
        if (normalized == "linear")
            return GaussianSplatShaderAntiPopPKeepCurve::eLinear;
        if (normalized == "smooth-wide" || normalized == "smooth_wide")
            return GaussianSplatShaderAntiPopPKeepCurve::eSmoothWide;
        if (normalized == "logistic-soft" || normalized == "logistic_soft")
            return GaussianSplatShaderAntiPopPKeepCurve::eLogisticSoft;
        if (normalized == "logistic-steep" || normalized == "logistic_steep")
            return GaussianSplatShaderAntiPopPKeepCurve::eLogisticSteep;
        return std::nullopt;
    }

    std::string_view membershipModeLabel(const GaussianSplatCachedSelectionMembershipMode mode)
    {
        switch (mode)
        {
            case GaussianSplatCachedSelectionMembershipMode::eStaticRandom:
                return "static_random";
            case GaussianSplatCachedSelectionMembershipMode::eDynamicDeterministic:
                return "dynamic_deterministic";
            case GaussianSplatCachedSelectionMembershipMode::eDynamicFrozenHash:
                return "dynamic_frozen_hash";
            case GaussianSplatCachedSelectionMembershipMode::eMatchedRateRandomNull:
                return "matched_rate_random_null";
        }
        return "unknown";
    }

    std::string_view formalConditionLabel(const FormalProtocolCondition condition)
    {
        switch (condition)
        {
            case FormalProtocolCondition::eC2DynamicDeterministic:
                return "C2_dynamic_deterministic";
            case FormalProtocolCondition::eC1StaticRandom:
                return "C1_static_random";
            case FormalProtocolCondition::eC1StaticMeanCount:
                return "C1_static_mean_count";
            case FormalProtocolCondition::eC3DynamicFrozenHash:
                return "C3_dynamic_frozen_hash";
            case FormalProtocolCondition::eC2NullMatchedRandomSwap:
                return "C2_null_matched_random_swap";
            case FormalProtocolCondition::eNone:
                break;
        }
        return "unknown";
    }

    bool formalConditionIsE1(const FormalProtocolCondition condition)
    {
        return condition == FormalProtocolCondition::eC1StaticRandom ||
               condition == FormalProtocolCondition::eC1StaticMeanCount ||
               condition == FormalProtocolCondition::eC2DynamicDeterministic ||
               condition == FormalProtocolCondition::eC3DynamicFrozenHash;
    }

    GaussianSplatCachedSelectionMembershipMode formalConditionMembershipMode(
        const FormalProtocolCondition condition)
    {
        switch (condition)
        {
            case FormalProtocolCondition::eC1StaticRandom:
            case FormalProtocolCondition::eC1StaticMeanCount:
                return GaussianSplatCachedSelectionMembershipMode::eStaticRandom;
            case FormalProtocolCondition::eC2DynamicDeterministic:
                return GaussianSplatCachedSelectionMembershipMode::eDynamicDeterministic;
            case FormalProtocolCondition::eC3DynamicFrozenHash:
                return GaussianSplatCachedSelectionMembershipMode::eDynamicFrozenHash;
            case FormalProtocolCondition::eC2NullMatchedRandomSwap:
                return GaussianSplatCachedSelectionMembershipMode::eMatchedRateRandomNull;
            case FormalProtocolCondition::eNone:
                break;
        }
        return GaussianSplatCachedSelectionMembershipMode::eDynamicDeterministic;
    }

    std::optional<GaussianProjectedCostBuildMode> parseProjectedCostBuildMode(const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "cpu")
            return GaussianProjectedCostBuildMode::eCpu;
        if (normalized == "gpu-sync" || normalized == "gpusync" || normalized == "sync-gpu")
            return GaussianProjectedCostBuildMode::eGpuSync;
        return std::nullopt;
    }

    std::optional<GaussianSplatFoveatedAdaptationMode> parseFoveatedAdaptationMode(const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "fixed")
            return GaussianSplatFoveatedAdaptationMode::eFixed;
        if (normalized == "dynamic-budget")
            return GaussianSplatFoveatedAdaptationMode::eDynamicBudget;
        if (normalized == "stability-aware" || normalized == "stable-budget" ||
            normalized == "stability-aware-budget" || normalized == "temporal-stability-aware")
            return GaussianSplatFoveatedAdaptationMode::eStabilityAwareBudget;
        if (normalized == "dynamic-range")
            return GaussianSplatFoveatedAdaptationMode::eDynamicRange;
        if (normalized == "progressive" || normalized == "center-out" ||
            normalized == "progressive-center-out")
            return GaussianSplatFoveatedAdaptationMode::eProgressiveCenterOut;
        if (normalized == "progressive-greedy" || normalized == "greedy-progressive" ||
            normalized == "old-progressive")
            return GaussianSplatFoveatedAdaptationMode::eProgressiveGreedy;
        return std::nullopt;
    }

    std::optional<GaussianSplatFoveatedDistribution> parseFoveatedDistribution(const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "hard" || normalized == "hard-ring" || normalized == "ring")
            return GaussianSplatFoveatedDistribution::eHardRing;
        if (normalized == "smooth" || normalized == "smoothstep")
            return GaussianSplatFoveatedDistribution::eSmoothstep;
        if (normalized == "gaussian" || normalized == "normal")
            return GaussianSplatFoveatedDistribution::eGaussian;
        if (normalized == "exponential" || normalized == "exp")
            return GaussianSplatFoveatedDistribution::eExponential;
        if (normalized == "inverse-power" || normalized == "hyperbolic" || normalized == "inverse")
            return GaussianSplatFoveatedDistribution::eInversePower;
        if (normalized == "log-polar" || normalized == "logpolar")
            return GaussianSplatFoveatedDistribution::eLogPolar;
        if (normalized == "cortical" || normalized == "cortical-style" ||
            normalized == "cortical-magnification")
            return GaussianSplatFoveatedDistribution::eCortical;
        if (normalized == "cone-density" || normalized == "cone-density-fitted" ||
            normalized == "cone" || normalized == "photoreceptor")
            return GaussianSplatFoveatedDistribution::eConeDensityFitted;
        if (normalized == "continuous" || normalized == "continuous-scheduler" ||
            normalized == "eccentricity-continuous" || normalized == "eccentricity")
        {
            return GaussianSplatFoveatedDistribution::eContinuousScheduler;
        }
        if (normalized == "fovea-protected" || normalized == "fovea-protected-continuous" ||
            normalized == "protected-continuous" || normalized == "fovea-protected-cct")
        {
            return GaussianSplatFoveatedDistribution::eFoveaProtectedContinuous;
        }
        return std::nullopt;
    }

    std::optional<GaussianSplatFoveatedCoverageGuardMode> parseCoverageGuardMode(const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "off" || normalized == "none" || normalized == "disabled")
            return GaussianSplatFoveatedCoverageGuardMode::eOff;
        if (normalized == "global" || normalized == "legacy")
            return GaussianSplatFoveatedCoverageGuardMode::eGlobal;
        if (normalized == "local" || normalized == "local-bounded" || normalized == "bounded")
            return GaussianSplatFoveatedCoverageGuardMode::eLocalBounded;
        if (normalized == "risk" || normalized == "risk-triggered" || normalized == "triggered" ||
            normalized == "risk-local" || normalized == "local-risk")
            return GaussianSplatFoveatedCoverageGuardMode::eRiskTriggered;
        return std::nullopt;
    }

    std::optional<GaussianSplatShLodGuardMode> parseShLodGuardMode(const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "off" || normalized == "none" || normalized == "disabled")
            return GaussianSplatShLodGuardMode::eOff;
        if (normalized == "energy")
            return GaussianSplatShLodGuardMode::eEnergy;
        if (normalized == "projected-cost" || normalized == "projected" || normalized == "cost")
            return GaussianSplatShLodGuardMode::eProjectedCost;
        if (normalized == "energy-projected-cost" || normalized == "energy-cost" ||
            normalized == "projected-energy" || normalized == "combined")
        {
            return GaussianSplatShLodGuardMode::eEnergyProjectedCost;
        }
        return std::nullopt;
    }

    std::optional<GaussianSplatShStorageLayout> parseShStorageLayout(const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "monolithic" || normalized == "default")
            return GaussianSplatShStorageLayout::eMonolithic;
        if (normalized == "split-bands" || normalized == "split" || normalized == "bands")
            return GaussianSplatShStorageLayout::eSplitBands;
        return std::nullopt;
    }

    std::optional<BenchmarkCameraPath> parseBenchmarkCameraPath(const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "static" || normalized == "fixed")
            return BenchmarkCameraPath::eStatic;
        if (normalized == "dataset" || normalized == "file" || normalized == "cameras")
            return BenchmarkCameraPath::eDataset;
        return std::nullopt;
    }

    std::optional<BenchmarkGazePath> parseBenchmarkGazePath(const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "static" || normalized == "fixed" || normalized == "center")
            return BenchmarkGazePath::eStatic;
        if (normalized == "grid" || normalized == "sweep")
            return BenchmarkGazePath::eGrid;
        if (normalized == "3x3-grid-step" || normalized == "grid-step")
            return BenchmarkGazePath::eGridStep3x3;
        if (normalized == "slow-horizontal-sweep" || normalized == "horizontal-sweep")
            return BenchmarkGazePath::eSlowHorizontalSweep;
        if (normalized == "medium-horizontal-sweep" || normalized == "medium-sweep")
            return BenchmarkGazePath::eMediumHorizontalSweep;
        if (normalized == "fast-horizontal-sweep" || normalized == "fast-sweep")
            return BenchmarkGazePath::eFastHorizontalSweep;
        if (normalized == "slow-vertical-sweep" || normalized == "vertical-sweep")
            return BenchmarkGazePath::eSlowVerticalSweep;
        if (normalized == "saccade-jump-center-to-edge" || normalized == "center-to-edge")
            return BenchmarkGazePath::eSaccadeJumpCenterToEdge;
        if (normalized == "saccade-jump-edge-to-center" || normalized == "edge-to-center")
            return BenchmarkGazePath::eSaccadeJumpEdgeToCenter;
        if (normalized == "step-jump-center-corner-center" ||
            normalized == "center-corner-center" ||
            normalized == "step-jump")
        {
            return BenchmarkGazePath::eStepJumpCenterCornerCenter;
        }
        if (normalized == "repeated-step-jump" ||
            normalized == "center-corner-center-repeat" ||
            normalized == "saccade-storm")
        {
            return BenchmarkGazePath::eRepeatedStepJump;
        }
        return std::nullopt;
    }

    std::string_view benchmarkCameraPathLabel(const BenchmarkCameraPath path)
    {
        switch (path)
        {
            case BenchmarkCameraPath::eStatic:
                return "static";
            case BenchmarkCameraPath::eDataset:
                return "dataset";
        }
        return "unknown";
    }

    std::string_view benchmarkGazePathLabel(const BenchmarkGazePath path)
    {
        switch (path)
        {
            case BenchmarkGazePath::eStatic:
                return "static";
            case BenchmarkGazePath::eGrid:
                return "grid";
            case BenchmarkGazePath::eGridStep3x3:
                return "3x3-grid-step";
            case BenchmarkGazePath::eSlowHorizontalSweep:
                return "slow-horizontal-sweep";
            case BenchmarkGazePath::eMediumHorizontalSweep:
                return "medium-horizontal-sweep";
            case BenchmarkGazePath::eFastHorizontalSweep:
                return "fast-horizontal-sweep";
            case BenchmarkGazePath::eSlowVerticalSweep:
                return "slow-vertical-sweep";
            case BenchmarkGazePath::eSaccadeJumpCenterToEdge:
                return "saccade-jump-center-to-edge";
            case BenchmarkGazePath::eSaccadeJumpEdgeToCenter:
                return "saccade-jump-edge-to-center";
            case BenchmarkGazePath::eStepJumpCenterCornerCenter:
                return "step-jump-center-corner-center";
            case BenchmarkGazePath::eRepeatedStepJump:
                return "repeated-step-jump";
        }
        return "unknown";
    }

    uint64_t hashSelectedSourceIds(const std::vector<uint32_t>& ids)
    {
        uint64_t hash = 1469598103934665603ull;
        for (const uint32_t id : ids)
        {
            uint32_t value = id;
            for (uint32_t byte = 0u; byte < 4u; ++byte)
            {
                hash ^= static_cast<uint8_t>(value & 0xffu);
                hash *= 1099511628211ull;
                value >>= 8u;
            }
        }
        return hash;
    }

    uint64_t hashStablePrefixCandidateSet(const uint32_t count)
    {
        uint64_t hash = 1469598103934665603ull;
        auto mix = [&hash](uint64_t value) {
            for (uint32_t byteIndex = 0; byteIndex < 8; ++byteIndex)
            {
                hash ^= static_cast<uint8_t>(value & 0xffu);
                hash *= 1099511628211ull;
                value >>= 8u;
            }
        };
        mix(0x4750555f50524658ull); // "GPU_PRFX" marker: stable direct-prefix candidate set.
        mix(count);
        return hash;
    }

    uint64_t hashBytes(const std::span<const uint8_t> bytes)
    {
        uint64_t hash = 1469598103934665603ull;
        for (const uint8_t byte : bytes)
        {
            hash ^= byte;
            hash *= 1099511628211ull;
        }
        return hash;
    }

    uint32_t selectedSetIntersectionCount(const std::vector<uint32_t>& lhs,
                                          const std::vector<uint32_t>& rhs)
    {
        size_t i = 0u;
        size_t j = 0u;
        uint32_t intersection = 0u;
        while (i < lhs.size() && j < rhs.size())
        {
            if (lhs[i] == rhs[j])
            {
                ++intersection;
                ++i;
                ++j;
            }
            else if (lhs[i] < rhs[j])
            {
                ++i;
            }
            else
            {
                ++j;
            }
        }
        return intersection;
    }

    std::string selectedSourceIdRanges(const std::vector<uint32_t>& ids)
    {
        if (ids.empty())
            return {};

        std::ostringstream out;
        uint32_t rangeStart = ids.front();
        uint32_t previous = ids.front();
        auto flushRange = [&]() {
            if (out.tellp() > std::streampos {0})
                out << ';';
            if (rangeStart == previous)
                out << rangeStart;
            else
                out << rangeStart << '-' << previous;
        };

        for (size_t i = 1u; i < ids.size(); ++i)
        {
            if (ids[i] == previous + 1u)
            {
                previous = ids[i];
                continue;
            }
            flushRange();
            rangeStart = previous = ids[i];
        }
        flushRange();
        return out.str();
    }

    std::string csvEscape(const std::string_view value)
    {
        std::string escaped;
        escaped.reserve(value.size() + 2u);
        escaped.push_back('"');
        for (const char ch : value)
        {
            if (ch == '"')
                escaped.push_back('"');
            escaped.push_back(ch);
        }
        escaped.push_back('"');
        return escaped;
    }

    std::string jsonEscape(const std::string_view value)
    {
        std::string escaped;
        escaped.reserve(value.size() + 2u);
        escaped.push_back('"');
        for (const char ch : value)
        {
            switch (ch)
            {
                case '"':
                    escaped += "\\\"";
                    break;
                case '\\':
                    escaped += "\\\\";
                    break;
                case '\b':
                    escaped += "\\b";
                    break;
                case '\f':
                    escaped += "\\f";
                    break;
                case '\n':
                    escaped += "\\n";
                    break;
                case '\r':
                    escaped += "\\r";
                    break;
                case '\t':
                    escaped += "\\t";
                    break;
                default:
                    escaped.push_back(ch);
                    break;
            }
        }
        escaped.push_back('"');
        return escaped;
    }

    std::string hexHashText(const uint64_t hash)
    {
        std::ostringstream out;
        out << "0x" << std::hex << std::setw(16) << std::setfill('0') << hash;
        return out.str();
    }

    struct SelectedSetDiff
    {
        uint32_t intersection {0};
        std::vector<uint32_t> added;
        std::vector<uint32_t> removed;
    };

    SelectedSetDiff diffSelectedSourceIds(const std::vector<uint32_t>& previous,
                                          const std::vector<uint32_t>& current)
    {
        SelectedSetDiff diff {};
        size_t i = 0u;
        size_t j = 0u;
        while (i < previous.size() || j < current.size())
        {
            if (i >= previous.size())
            {
                diff.added.push_back(current[j++]);
            }
            else if (j >= current.size())
            {
                diff.removed.push_back(previous[i++]);
            }
            else if (previous[i] == current[j])
            {
                ++diff.intersection;
                ++i;
                ++j;
            }
            else if (previous[i] < current[j])
            {
                diff.removed.push_back(previous[i++]);
            }
            else
            {
                diff.added.push_back(current[j++]);
            }
        }
        return diff;
    }

    bool commandLineEnablesBenchmark(std::span<const std::string> args)
    {
        for (const std::string& arg : args)
        {
            if (arg == "--mouse-gaze")
                return false;
        }

        for (const std::string& arg : args)
        {
            if (arg == "--benchmark" || arg.starts_with("--benchmark=") ||
                arg == "--benchmark-frames" || arg.starts_with("--benchmark-frames=") ||
                arg == "--benchmark-settle-frames" || arg.starts_with("--benchmark-settle-frames=") ||
                arg == "--benchmark-output" || arg.starts_with("--benchmark-output=") ||
                arg == "--capture-frame-sequence" || arg.starts_with("--capture-frame-dir") ||
                arg.starts_with("--capture-frame-prefix") || arg.starts_with("--capture-frame-limit") ||
                arg == "--capture-alpha" || arg.starts_with("--capture-alpha-dir") ||
                arg.starts_with("--capture-alpha-limit") ||
                arg == "--selected-id-log" || arg.starts_with("--selected-id-log-stride") ||
                arg == "--cached-selection-oracle" || arg.starts_with("--membership-mode") ||
                arg.starts_with("--gaze-trajectory") ||
                arg.starts_with("--fixed-gaze") || arg == "--log-churn" ||
                arg == "--clean-timing" ||
                arg.starts_with("--renderdoc-capture-sample") ||
                arg == "--log-flicker" || arg.starts_with("--flicker-foveal-radius-deg") ||
                arg == "--e3-honest-e6" || arg.starts_with("--e6-spectrum-grid") ||
                arg.starts_with("--seed") || arg.starts_with("--output-dir") ||
                arg.starts_with("--shader-antipop-mode") ||
                arg.starts_with("--shader-antipop-hash-seed") ||
                arg.starts_with("--shader-antipop-ramp-width") ||
                arg.starts_with("--shader-antipop-guard-threshold") ||
                arg.starts_with("--shader-antipop-guard-floor") ||
                arg.starts_with("--shader-antipop-pkeep-curve") ||
                arg.starts_with("--shader-antipop-prefix-ratio") ||
                arg.starts_with("--shader-antipop-normalize-alpha") ||
                arg.starts_with("--shader-antipop-normalize-strength") ||
                arg.starts_with("--shader-antipop-normalize-clamp-min") ||
                arg.starts_with("--shader-antipop-normalize-clamp-max") ||
                arg.starts_with("--shader-antipop-normalize-factor") ||
                arg.starts_with("--coverage-release-") ||
                arg.starts_with("--coverage-texture-") ||
                arg.starts_with("--guide-before-discard") ||
                arg == "--shader-antipop-debug-log" ||
                arg.starts_with("--shader-antipop-output-dir") ||
                arg.starts_with("--scene") || arg.starts_with("--scene-path") ||
                arg.starts_with("--scene-manifest") ||
                arg == "--formal-e1e2e3")
            {
                return true;
            }
        }
        return false;
    }

    std::string splatUriFromCliValue(const std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "lizard" || normalized == "hornedlizard")
            return "res://models/3dgs/hornedlizard.spz";
        if (normalized == "racoonfamily")
            return "res://models/3dgs/racoonfamily.spz";
        constexpr std::array<std::string_view, 4> kResearchScenes {"train", "truck", "drjohnson", "playroom"};
        for (const auto scene : kResearchScenes)
        {
            const std::string sceneName {scene};
            if (normalized == sceneName)
                return "res://models/3dgs/" + sceneName + "/" + sceneName + "_clod.ply";
        }

        const std::string text {value};
        if (text.find("://") != std::string::npos)
            return text;
        if (text.starts_with("models/") || text.starts_with("imported/"))
            return "res://" + text;
        return text;
    }

    std::string sceneUriFromCliValue(const std::string_view value)
    {
        const std::string text {value};
        if (text.find("://") != std::string::npos)
            return text;
        if (text.starts_with("scenes/") || text.starts_with("resources/scenes/"))
        {
            if (text.starts_with("resources/"))
                return "res://" + text.substr(std::string_view {"resources/"}.size());
            return "res://" + text;
        }
        return text;
    }

    bool parseFloatOption(std::span<const std::string> args,
                          size_t&                      index,
                          const std::string_view       option,
                          std::optional<float>&        outValue)
    {
        if (const auto value = takeOptionValue(args, index, option))
        {
            outValue = parseFloat(*value);
            if (!outValue)
                VULTRA_CLIENT_WARN("Ignoring invalid {} value: {}", option, *value);
            return true;
        }
        return false;
    }

    GaussianDemoOptions parseDemoOptions(std::span<const std::string> args)
    {
        GaussianDemoOptions options {};

        for (size_t i = 0; i < args.size(); ++i)
        {
            const std::string_view arg = args[i];

            if (arg == "--benchmark")
            {
                options.benchmarkEnabled = true;
                continue;
            }

            if (arg == "--clean-timing")
            {
                options.cleanTiming = true;
                options.benchmarkEnabled = true;
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--renderdoc-capture-sample"))
            {
                options.benchmarkEnabled = true;
                if (const auto sample = parseU32(*value); sample && *sample > 0u)
                    options.renderDocCaptureSample = *sample;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --renderdoc-capture-sample value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--benchmark"))
            {
                options.benchmarkEnabled = true;
                if (const auto frames = parseU32(*value); frames && *frames > 0u)
                    options.benchmarkFrames = *frames;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --benchmark value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--benchmark-frames"))
            {
                options.benchmarkEnabled = true;
                if (const auto frames = parseU32(*value); frames && *frames > 0u)
                    options.benchmarkFrames = *frames;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --benchmark-frames value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--benchmark-warmup"))
            {
                if (const auto frames = parseU32(*value))
                    options.warmupFrames = *frames;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --benchmark-warmup value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--benchmark-settle-frames"))
            {
                if (const auto frames = parseU32(*value))
                    options.benchmarkSettleFrames = *frames;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --benchmark-settle-frames value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--benchmark-output"))
            {
                options.benchmarkEnabled = true;
                options.outputPath       = std::filesystem::path {std::string {*value}};
                continue;
            }

            if (arg == "--cached-selection-oracle")
            {
                options.cachedSelectionOracle = true;
                options.benchmarkEnabled = true;
                options.selectedIdLogEnabled = true;
                options.selectedIdLogStride = 1u;
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--membership-mode"))
            {
                if (const auto mode = parseMembershipMode(*value))
                    options.membershipMode = *mode;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --membership-mode value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--fixed-gaze"))
            {
                if (const auto gaze = parseGazeUv(*value))
                    options.fixedGaze = *gaze;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --fixed-gaze value: {}", *value);
                continue;
            }

            if (arg == "--log-churn")
            {
                options.logChurn = true;
                options.benchmarkEnabled = true;
                options.selectedIdLogEnabled = true;
                options.selectedIdLogStride = 1u;
                continue;
            }

            if (arg == "--log-flicker")
            {
                options.logFlicker = true;
                options.benchmarkEnabled = true;
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--flicker-foveal-radius-deg"))
            {
                if (const auto radius = parseFloat(*value); radius && *radius >= 0.0f)
                    options.flickerFovealRadiusDegrees = *radius;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --flicker-foveal-radius-deg value: {}", *value);
                continue;
            }

            if (arg == "--formal-e1e2e3")
            {
                options.formalE1E2E3 = true;
                options.cachedSelectionOracle = true;
                options.logChurn = true;
                options.logFlicker = true;
                options.benchmarkEnabled = true;
                options.selectedIdLogEnabled = true;
                options.selectedIdLogStride = 1u;
                options.benchmarkCameraPath = BenchmarkCameraPath::eStatic;
                continue;
            }

            if (arg == "--e3-honest-e6")
            {
                options.e3HonestE6 = true;
                options.formalE1E2E3 = true;
                options.cachedSelectionOracle = true;
                options.logChurn = true;
                options.logFlicker = true;
                options.benchmarkEnabled = true;
                options.selectedIdLogEnabled = true;
                options.selectedIdLogStride = 1u;
                options.benchmarkCameraPath = BenchmarkCameraPath::eStatic;
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--e6-spectrum-grid"))
            {
                if (const auto grid = parseU32(*value); grid && (*grid == 32u || *grid == 64u || *grid == 128u))
                    options.e6SpectrumGridSize = *grid;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --e6-spectrum-grid value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--seed"))
            {
                if (const auto seed = parseU32(*value))
                    options.seed = *seed;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --seed value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--output-dir"))
            {
                options.benchmarkEnabled = true;
                options.outputDir = std::filesystem::path {std::string {*value}};
                options.outputPath = options.outputDir / "benchmark.csv";
                continue;
            }
            if (const auto value = takeOptionValue(args, i, "--shader-antipop-mode"))
            {
                if (parseRuntimeAntiPopMethod(*value, options))
                {
                    options.benchmarkEnabled = true;
                }
                else if (const auto mode = parseShaderAntiPopMode(*value))
                {
                    options.shaderAntiPopMode = *mode;
                    if (*mode == GaussianSplatShaderAntiPopMode::eCoverageStableLogpolarRelease)
                    {
                        options.runtimeAntiPopOrderFreeHashRamp = true;
                        options.mode = GaussianSplatBaselineMode::eBaseline;
                        options.foveatedClodEnabled = true;
                    }
                    options.benchmarkEnabled = true;
                }
                else
                {
                    VULTRA_CLIENT_WARN("Ignoring invalid --shader-antipop-mode value: {}", *value);
                }
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--shader-antipop-hash-seed"))
            {
                if (const auto seed = parseU32(*value))
                    options.shaderAntiPopHashSeed = *seed;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --shader-antipop-hash-seed value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--shader-antipop-ramp-width"))
            {
                if (const auto width = parseFloat(*value); width && *width >= 0.0f)
                    options.shaderAntiPopRampWidth = *width;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --shader-antipop-ramp-width value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--shader-antipop-guard-threshold"))
            {
                if (const auto threshold = parseFloat(*value); threshold && *threshold >= 0.0f)
                    options.shaderAntiPopGuardThreshold = *threshold;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --shader-antipop-guard-threshold value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--shader-antipop-guard-floor"))
            {
                if (const auto floor = parseFloat(*value); floor && *floor >= 0.0f)
                    options.shaderAntiPopGuardFloor = *floor;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --shader-antipop-guard-floor value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--shader-antipop-pkeep-curve"))
            {
                if (const auto curve = parseShaderAntiPopPKeepCurve(*value))
                    options.shaderAntiPopPKeepCurve = *curve;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --shader-antipop-pkeep-curve value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--shader-antipop-prefix-ratio"))
            {
                if (const auto ratio = parseFloat(*value); ratio && *ratio > 0.0f)
                    options.shaderAntiPopPrefixRatio = *ratio;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --shader-antipop-prefix-ratio value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--shader-antipop-normalize-alpha"))
            {
                if (const auto mode = parseShaderAntiPopNormalizeMode(*value))
                    options.shaderAntiPopNormalizeMode = *mode;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --shader-antipop-normalize-alpha value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--shader-antipop-normalize-strength"))
            {
                if (const auto strength = parseFloat(*value); strength && *strength >= 0.0f && *strength <= 1.0f)
                    options.shaderAntiPopNormalizeStrength = *strength;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --shader-antipop-normalize-strength value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--shader-antipop-normalize-clamp-min"))
            {
                if (const auto clampMin = parseFloat(*value); clampMin && *clampMin >= 0.0f)
                    options.shaderAntiPopNormalizeClampMin = *clampMin;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --shader-antipop-normalize-clamp-min value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--shader-antipop-normalize-clamp-max"))
            {
                if (const auto clampMax = parseFloat(*value); clampMax && *clampMax >= 0.0f)
                    options.shaderAntiPopNormalizeClampMax = *clampMax;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --shader-antipop-normalize-clamp-max value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--shader-antipop-normalize-factor"))
            {
                if (const auto factor = parseFloat(*value); factor && *factor >= 0.0f)
                    options.shaderAntiPopNormalizeFactor = *factor;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --shader-antipop-normalize-factor value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-release-coverage-floor"))
            {
                if (const auto enabled = parseBoolFlag(*value))
                    options.coverageStableReleaseCoverageFloorEnabled = *enabled;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-release-coverage-floor value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-release-stable-hash"))
            {
                if (const auto enabled = parseBoolFlag(*value))
                    options.coverageStableReleaseStableHashEnabled = *enabled;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-release-stable-hash value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-release-staggered-release"))
            {
                if (const auto enabled = parseBoolFlag(*value))
                    options.coverageStableReleaseStaggeredReleaseEnabled = *enabled;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-release-staggered-release value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-release-saturated-floor"))
            {
                if (const auto enabled = parseBoolFlag(*value))
                    options.coverageStableReleaseSaturatedFloorEnabled = *enabled;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-release-saturated-floor value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-release-lambda"))
            {
                if (const auto lambda = parseFloat(*value); lambda && *lambda >= 0.0f && *lambda <= 1.0f)
                    options.coverageStableReleaseLambda = *lambda;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-release-lambda value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-release-duration-frames"))
            {
                if (const auto frames = parseU32(*value); frames && *frames > 0u)
                {
                    options.coverageStableReleaseDurationFrames = *frames;
                    options.coverageStableReleaseLambda =
                        static_cast<float>(std::exp(-std::log(10.0) / static_cast<double>(*frames)));
                }
                else
                {
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-release-duration-frames value: {}", *value);
                }
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-release-d-min"))
            {
                if (const auto dMin = parseFloat(*value); dMin && *dMin >= 0.0f)
                    options.coverageStableReleaseDMin = *dMin;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-release-d-min value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-release-sigma-max"))
            {
                if (const auto sigmaMax = parseFloat(*value); sigmaMax && *sigmaMax >= 0.0f)
                    options.coverageStableReleaseSigmaMax = *sigmaMax;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-release-sigma-max value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-release-floor-ema"))
            {
                if (const auto ema = parseFloat(*value); ema && *ema >= 0.0f && *ema <= 1.0f)
                    options.coverageStableReleaseFloorEma = *ema;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-release-floor-ema value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-release-release-epsilon"))
            {
                if (const auto epsilon = parseFloat(*value); epsilon && *epsilon >= 0.0f)
                    options.coverageStableReleaseReleaseEpsilon = *epsilon;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-release-release-epsilon value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--delayed-guide-temporal-release-policy"))
            {
                if (const auto policy = parseDelayedGuideTemporalReleasePolicy(*value))
                    options.delayedGuideTemporalReleasePolicy = *policy;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --delayed-guide-temporal-release-policy value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--delayed-guide-temporal-release-cap-ratio"))
            {
                if (const auto cap = parseFloat(*value); cap && *cap >= 0.0f && *cap <= 1.0f)
                    options.delayedGuideTemporalReleaseCapRatio = *cap;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --delayed-guide-temporal-release-cap-ratio value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--delayed-guide-temporal-risk-threshold"))
            {
                if (const auto threshold = parseFloat(*value); threshold && *threshold >= 0.0f && *threshold <= 1.0f)
                    options.delayedGuideTemporalRiskThreshold = *threshold;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --delayed-guide-temporal-risk-threshold value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--delayed-guide-temporal-footprint-decay-scale"))
            {
                if (const auto scale = parseFloat(*value); scale && *scale >= 0.0f && *scale <= 1.0f)
                    options.delayedGuideTemporalFootprintDecayScale = *scale;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --delayed-guide-temporal-footprint-decay-scale value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--delayed-guide-temporal-large-footprint-px"))
            {
                if (const auto threshold = parseFloat(*value); threshold && *threshold >= 1.0f)
                    options.delayedGuideTemporalLargeFootprintPx = *threshold;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --delayed-guide-temporal-large-footprint-px value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-release-tile-grid"))
            {
                const auto comma = value->find(',');
                if (comma != std::string_view::npos)
                {
                    const auto x = parseU32(value->substr(0, comma));
                    const auto y = parseU32(value->substr(comma + 1));
                    if (x && y && *x > 0u && *y > 0u)
                    {
                        options.coverageStableReleaseTileGridX = *x;
                        options.coverageStableReleaseTileGridY = *y;
                    }
                    else
                    {
                        VULTRA_CLIENT_WARN("Ignoring invalid --coverage-release-tile-grid value: {}", *value);
                    }
                }
                else
                {
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-release-tile-grid value: {}", *value);
                }
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-texture-floor"))
            {
                if (const auto enabled = parseBoolFlag(*value))
                    options.coverageTextureFloorEnabled = *enabled;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-texture-floor value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-texture-size"))
            {
                const auto comma = value->find(',');
                if (comma != std::string_view::npos)
                {
                    const auto x = parseU32(value->substr(0, comma));
                    const auto y = parseU32(value->substr(comma + 1));
                    if (x && y && *x > 0u && *y > 0u)
                    {
                        options.coverageTextureFloorWidth = *x;
                        options.coverageTextureFloorHeight = *y;
                    }
                    else
                    {
                        VULTRA_CLIENT_WARN("Ignoring invalid --coverage-texture-size value: {}", *value);
                    }
                }
                else
                {
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-texture-size value: {}", *value);
                }
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-texture-d-min"))
            {
                if (const auto dMin = parseFloat(*value); dMin && *dMin >= 0.0f)
                    options.coverageTextureFloorDMin = *dMin;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-texture-d-min value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-texture-sigma-max"))
            {
                if (const auto sigmaMax = parseFloat(*value); sigmaMax && *sigmaMax >= 0.0f)
                    options.coverageTextureFloorSigmaMax = *sigmaMax;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-texture-sigma-max value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-texture-strength"))
            {
                if (const auto strength = parseFloat(*value); strength && *strength >= 0.0f && *strength <= 1.0f)
                    options.coverageTextureFloorStrength = *strength;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-texture-strength value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-texture-floor-strength"))
            {
                if (const auto strength = parseFloat(*value); strength && *strength >= 0.0f && *strength <= 1.0f)
                    options.coverageTextureFloorStrength = *strength;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-texture-floor-strength value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-texture-history-beta"))
            {
                if (const auto beta = parseFloat(*value); beta && *beta >= 0.0f && *beta <= 1.0f)
                    options.coverageTextureFloorHistoryBeta = *beta;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-texture-history-beta value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-texture-update-interval"))
            {
                if (const auto interval = parseU32(*value); interval && *interval > 0u)
                    options.coverageTextureFloorUpdateInterval = *interval;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-texture-update-interval value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-texture-debug"))
            {
                if (const auto enabled = parseBoolFlag(*value))
                    options.coverageTextureFloorDebugEnabled = *enabled;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-texture-debug value: {}", *value);
                continue;
            }

            if (arg == "--guide-before-discard")
            {
                options.guideBeforeDiscardEnabled = true;
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--guide-before-discard"))
            {
                if (const auto enabled = parseBoolFlag(*value))
                    options.guideBeforeDiscardEnabled = *enabled;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --guide-before-discard value: {}", *value);
                continue;
            }

            if (arg == "--shader-antipop-debug-log")
            {
                options.shaderAntiPopDebugLog = true;
                options.benchmarkEnabled = true;
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--shader-antipop-output-dir"))
            {
                options.shaderAntiPopOutputDir = std::filesystem::path {std::string {*value}};
                options.outputDir = *options.shaderAntiPopOutputDir;
                options.outputPath = options.outputDir / "shader_antipop_vertical_slice_per_frame.csv";
                options.benchmarkEnabled = true;
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--scene"))
            {
                options.sceneUri = sceneUriFromCliValue(*value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--scene-path"))
            {
                options.sceneUri = sceneUriFromCliValue(*value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--scene-manifest"))
            {
                options.sceneUri = sceneUriFromCliValue(*value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--benchmark-screenshot-output"))
            {
                options.benchmarkEnabled = true;
                options.benchmarkScreenshotOutput = std::filesystem::path {std::string {*value}};
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--benchmark-screenshot"))
            {
                options.benchmarkEnabled = true;
                options.benchmarkScreenshotOutput = std::filesystem::path {std::string {*value}};
                continue;
            }

            if (arg == "--capture-frame-sequence")
            {
                options.benchmarkEnabled     = true;
                options.captureFrameSequence = true;
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--capture-frame-dir"))
            {
                options.benchmarkEnabled     = true;
                options.captureFrameSequence = true;
                options.captureFrameDir      = std::filesystem::path {std::string {*value}};
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--capture-frame-prefix"))
            {
                options.benchmarkEnabled     = true;
                options.captureFrameSequence = true;
                options.captureFramePrefix   = std::string {*value};
                if (options.captureFramePrefix.empty())
                    options.captureFramePrefix = "frame";
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--capture-frame-limit"))
            {
                options.benchmarkEnabled     = true;
                options.captureFrameSequence = true;
                if (const auto limit = parseU32(*value); limit && *limit > 0u)
                    options.captureFrameLimit = *limit;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --capture-frame-limit value: {}", *value);
                continue;
            }

            if (arg == "--capture-alpha")
            {
                options.benchmarkEnabled = true;
                options.captureAlpha     = true;
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--capture-alpha-dir"))
            {
                options.benchmarkEnabled = true;
                options.captureAlpha     = true;
                options.captureAlphaDir  = std::filesystem::path {std::string {*value}};
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--capture-alpha-limit"))
            {
                options.benchmarkEnabled = true;
                options.captureAlpha     = true;
                if (const auto limit = parseU32(*value); limit && *limit > 0u)
                    options.captureAlphaLimit = *limit;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --capture-alpha-limit value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--benchmark-camera-path"))
            {
                if (const auto path = parseBenchmarkCameraPath(*value))
                    options.benchmarkCameraPath = *path;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --benchmark-camera-path value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--benchmark-camera-file"))
            {
                options.benchmarkCameraFile = std::filesystem::path {std::string {*value}};
                if (options.benchmarkCameraPath == BenchmarkCameraPath::eStatic)
                    options.benchmarkCameraPath = BenchmarkCameraPath::eDataset;
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--benchmark-gaze-path"))
            {
                if (const auto path = parseBenchmarkGazePath(*value))
                    options.benchmarkGazePath = *path;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --benchmark-gaze-path value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--gaze-trajectory"))
            {
                if (const auto path = parseBenchmarkGazePath(*value))
                    options.benchmarkGazePath = *path;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --gaze-trajectory value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--benchmark-view-frames"))
            {
                if (const auto frames = parseU32(*value); frames && *frames > 0u)
                    options.benchmarkViewFrames = *frames;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --benchmark-view-frames value: {}", *value);
                continue;
            }

            if (arg == "--benchmark-stereo-sequential")
            {
                options.benchmarkStereoSequential = true;
                continue;
            }

            if (arg == "--no-benchmark-stereo-sequential")
            {
                options.benchmarkStereoSequential = false;
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--benchmark-ipd"))
            {
                if (const auto ipd = parseFloat(*value); ipd && *ipd >= 0.0f)
                    options.benchmarkIpdMeters = *ipd;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --benchmark-ipd value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--gaussian-mode"))
            {
                options.mode = parseGaussianMode(*value);
                if (!options.mode)
                    VULTRA_CLIENT_WARN("Ignoring invalid --gaussian-mode value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--lod-budget"))
            {
                options.lodBudget = parseU32(*value);
                if (!options.lodBudget)
                    VULTRA_CLIENT_WARN("Ignoring invalid --lod-budget value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--lod-budget-mode"))
            {
                options.lodBudgetMode = parseLodBudgetMode(*value);
                if (!options.lodBudgetMode)
                    VULTRA_CLIENT_WARN("Ignoring invalid --lod-budget-mode value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--projected-cost-build-mode"))
            {
                options.projectedCostBuildMode = parseProjectedCostBuildMode(*value);
                if (!options.projectedCostBuildMode)
                    VULTRA_CLIENT_WARN("Ignoring invalid --projected-cost-build-mode value: {}", *value);
                continue;
            }

            if (arg == "--projected-cost-semantic-diff")
            {
                options.projectedCostSemanticDiffEnabled = true;
                continue;
            }

            if (arg == "--no-projected-cost-semantic-diff")
            {
                options.projectedCostSemanticDiffEnabled = false;
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--budget-mode"))
            {
                options.lodBudgetMode = parseLodBudgetMode(*value);
                if (!options.lodBudgetMode)
                    VULTRA_CLIENT_WARN("Ignoring invalid --budget-mode value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--projected-cost-budget"))
            {
                options.projectedCostBudget = parseU64(*value);
                if (!options.projectedCostBudget)
                    VULTRA_CLIENT_WARN("Ignoring invalid --projected-cost-budget value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--cost-budget"))
            {
                options.projectedCostBudget = parseU64(*value);
                if (!options.projectedCostBudget)
                    VULTRA_CLIENT_WARN("Ignoring invalid --cost-budget value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--projected-cost-chunk-size"))
            {
                options.projectedCostChunkSize = parseU32(*value);
                if (!options.projectedCostChunkSize || *options.projectedCostChunkSize == 0u)
                    VULTRA_CLIENT_WARN("Ignoring invalid --projected-cost-chunk-size value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--foveated-score-selection-stride"))
            {
                options.foveatedScoreSelectionStride = parseU32(*value);
                if (!options.foveatedScoreSelectionStride || *options.foveatedScoreSelectionStride == 0u)
                    VULTRA_CLIENT_WARN("Ignoring invalid --foveated-score-selection-stride value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--score-selection-stride"))
            {
                options.foveatedScoreSelectionStride = parseU32(*value);
                if (!options.foveatedScoreSelectionStride || *options.foveatedScoreSelectionStride == 0u)
                    VULTRA_CLIENT_WARN("Ignoring invalid --score-selection-stride value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-bin-grid-x"))
            {
                options.foveatedCoverageBinGridX = parseU32(*value);
                if (!options.foveatedCoverageBinGridX || *options.foveatedCoverageBinGridX == 0u)
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-bin-grid-x value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-bin-grid-y"))
            {
                options.foveatedCoverageBinGridY = parseU32(*value);
                if (!options.foveatedCoverageBinGridY || *options.foveatedCoverageBinGridY == 0u)
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-bin-grid-y value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-bin-quota-ratio"))
            {
                options.foveatedCoverageBinQuotaRatio = parseFloat(*value);
                if (!options.foveatedCoverageBinQuotaRatio)
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-bin-quota-ratio value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-bin-prefix-ratio"))
            {
                options.foveatedCoverageBinPrefixRatio = parseFloat(*value);
                if (!options.foveatedCoverageBinPrefixRatio)
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-bin-prefix-ratio value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--splat"))
            {
                options.splatUri = splatUriFromCliValue(*value);
                continue;
            }

            if (arg == "--gaze-rendering")
            {
                options.foveatedClodEnabled = true;
                continue;
            }

            if (arg == "--mouse-gaze")
            {
                options.mouseGazeControl = true;
                options.foveatedClodEnabled = true;
                continue;
            }

            if (arg == "--no-gaze-rendering")
            {
                options.foveatedClodEnabled = false;
                continue;
            }

            if (arg == "--coverage-compensation")
            {
                options.foveatedCoverageCompensationEnabled = true;
                continue;
            }

            if (arg == "--no-coverage-compensation")
            {
                options.foveatedCoverageCompensationEnabled = false;
                continue;
            }

            if (arg == "--foveated-temporal-hysteresis" || arg == "--temporal-hysteresis")
            {
                options.foveatedTemporalHysteresisEnabled = true;
                continue;
            }

            if (arg == "--no-foveated-temporal-hysteresis" || arg == "--no-temporal-hysteresis")
            {
                options.foveatedTemporalHysteresisEnabled = false;
                continue;
            }

            if (arg == "--continuous-scheduler")
            {
                options.foveatedDistribution = GaussianSplatFoveatedDistribution::eContinuousScheduler;
                continue;
            }

            if (arg == "--fovea-protected-continuous-scheduler" ||
                arg == "--fovea-protected-scheduler")
            {
                options.foveatedDistribution = GaussianSplatFoveatedDistribution::eFoveaProtectedContinuous;
                continue;
            }

            if (arg == "--selected-id-log")
            {
                options.selectedIdLogEnabled = true;
                continue;
            }

            if (arg == "--no-selected-id-log")
            {
                options.selectedIdLogEnabled = false;
                continue;
            }

            if (arg == "--foveated-boundary-smoothing" || arg == "--boundary-smoothing")
            {
                options.foveatedBoundarySmoothingEnabled = true;
                continue;
            }

            if (arg == "--no-foveated-boundary-smoothing" || arg == "--no-boundary-smoothing")
            {
                options.foveatedBoundarySmoothingEnabled = false;
                continue;
            }

            if (arg == "--sh-lod" || arg == "--foveated-sh-lod")
            {
                options.foveatedShLodEnabled = true;
                continue;
            }

            if (arg == "--no-sh-lod" || arg == "--no-foveated-sh-lod")
            {
                options.foveatedShLodEnabled = false;
                continue;
            }

            if (arg == "--sh-pop-log")
            {
                options.shPopLogEnabled = true;
                continue;
            }

            if (arg == "--no-sh-pop-log")
            {
                options.shPopLogEnabled = false;
                continue;
            }

            if (arg == "--sh-degree-hysteresis")
            {
                options.shDegreeHysteresisEnabled = true;
                continue;
            }

            if (arg == "--no-sh-degree-hysteresis")
            {
                options.shDegreeHysteresisEnabled = false;
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--sh-lod-guard"))
            {
                options.shLodGuardMode = parseShLodGuardMode(*value);
                if (!options.shLodGuardMode)
                    VULTRA_CLIENT_WARN("Ignoring invalid --sh-lod-guard value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--sh-storage-layout"))
            {
                options.shStorageLayout = parseShStorageLayout(*value);
                if (!options.shStorageLayout)
                    VULTRA_CLIENT_WARN("Ignoring invalid --sh-storage-layout value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--gaze-render-mode"))
            {
                options.foveatedRenderMode = parseFoveatedRenderMode(*value);
                if (!options.foveatedRenderMode)
                    VULTRA_CLIENT_WARN("Ignoring invalid --gaze-render-mode value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--gaze-adaptation"))
            {
                options.foveatedAdaptationMode = parseFoveatedAdaptationMode(*value);
                if (!options.foveatedAdaptationMode)
                    VULTRA_CLIENT_WARN("Ignoring invalid --gaze-adaptation value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--gaze-distribution"))
            {
                options.foveatedDistribution = parseFoveatedDistribution(*value);
                if (!options.foveatedDistribution)
                    VULTRA_CLIENT_WARN("Ignoring invalid --gaze-distribution value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--foveated-distribution"))
            {
                options.foveatedDistribution = parseFoveatedDistribution(*value);
                if (!options.foveatedDistribution)
                    VULTRA_CLIENT_WARN("Ignoring invalid --foveated-distribution value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-guard"))
            {
                options.coverageGuardMode = parseCoverageGuardMode(*value);
                if (!options.coverageGuardMode)
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-guard value: {}", *value);
                continue;
            }

            if (parseFloatOption(args, i, "--clod-level", options.clodLevel) ||
                parseFloatOption(args, i, "--projected-cost-budget-ratio", options.projectedCostBudgetRatio) ||
                parseFloatOption(args, i, "--cost-budget-ratio", options.projectedCostBudgetRatio) ||
                parseFloatOption(args, i, "--gaze-x", options.gazeX) ||
                parseFloatOption(args, i, "--gaze-y", options.gazeY) ||
                parseFloatOption(args, i, "--fovea-degrees", options.foveaDegrees) ||
                parseFloatOption(args, i, "--mid-degrees", options.midDegrees) ||
                parseFloatOption(args, i, "--fovea-lod", options.foveaLod) ||
                parseFloatOption(args, i, "--mid-lod", options.midLod) ||
                parseFloatOption(args, i, "--outer-lod", options.outerLod) ||
                parseFloatOption(args, i, "--fovea-res-scale", options.foveaResolutionScale) ||
                parseFloatOption(args, i, "--mid-res-scale", options.midResolutionScale) ||
                parseFloatOption(args, i, "--outer-res-scale", options.outerResolutionScale) ||
                parseFloatOption(args, i, "--transition-degrees", options.transitionDegrees) ||
                parseFloatOption(args, i, "--continuous-theta0-degrees", options.continuousTheta0Degrees) ||
                parseFloatOption(args, i, "--continuous-theta0", options.continuousTheta0Degrees) ||
                parseFloatOption(args, i, "--continuous-alpha", options.continuousAlpha) ||
                parseFloatOption(args, i, "--continuous-min-lod", options.continuousMinLevel) ||
                parseFloatOption(args, i, "--target-frame-ms", options.targetFrameMs) ||
                parseFloatOption(args, i, "--budget-adjust-rate", options.budgetAdjustRate) ||
                parseFloatOption(args, i, "--coverage-protection-degrees", options.coverageProtectionDegrees) ||
                parseFloatOption(args, i, "--coverage-guard-budget-ratio", options.coverageGuardBudgetRatio) ||
                parseFloatOption(args, i, "--coverage-guard-center-min", options.coverageGuardCenterMin) ||
                parseFloatOption(args, i, "--coverage-guard-transition-min", options.coverageGuardTransitionMin) ||
                parseFloatOption(args, i, "--coverage-guard-periphery-min", options.coverageGuardPeripheryMin) ||
                parseFloatOption(args, i, "--temporal-hysteresis-ratio", options.foveatedTemporalHysteresisRatio) ||
                parseFloatOption(args, i, "--boundary-smoothing-ratio", options.foveatedBoundarySmoothingRatio) ||
                parseFloatOption(args, i, "--temporal-peripheral-scale", options.foveatedTemporalPeripheralScale) ||
                parseFloatOption(args,
                                 i,
                                 "--sh-lod-guard-threshold-mid",
                                 options.shLodGuardThresholdMid) ||
                parseFloatOption(args, i, "--sh-guard-threshold-mid", options.shLodGuardThresholdMid) ||
                parseFloatOption(args,
                                 i,
                                 "--sh-lod-guard-threshold-high",
                                 options.shLodGuardThresholdHigh) ||
                parseFloatOption(args, i, "--sh-guard-threshold-high", options.shLodGuardThresholdHigh) ||
                parseFloatOption(args,
                                 i,
                                 "--sh-degree-guard-band-degrees",
                                 options.shDegreeGuardBandDegrees))
            {
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--temporal-residency-frames"))
            {
                if (const auto frames = parseU32(*value))
                    options.foveatedTemporalResidencyFrames = *frames;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --temporal-residency-frames value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--selected-id-log-stride"))
            {
                if (const auto stride = parseU32(*value))
                    options.selectedIdLogStride = std::max(1u, *stride);
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --selected-id-log-stride value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--coverage-guard-max-adds"))
            {
                if (const auto maxAdds = parseU32(*value))
                    options.coverageGuardMaxAdds = *maxAdds;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-guard-max-adds value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--sh-degree-downgrade-delay"))
            {
                if (const auto delay = parseU32(*value))
                    options.shDegreeDowngradeDelay = *delay;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --sh-degree-downgrade-delay value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--sh-degree-center"))
            {
                options.shDegreeCenter = parseU32(*value);
                if (!options.shDegreeCenter)
                    VULTRA_CLIENT_WARN("Ignoring invalid --sh-degree-center value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--sh-degree-mid"))
            {
                options.shDegreeMid = parseU32(*value);
                if (!options.shDegreeMid)
                    VULTRA_CLIENT_WARN("Ignoring invalid --sh-degree-mid value: {}", *value);
                continue;
            }

            if (const auto value = takeOptionValue(args, i, "--sh-degree-outer"))
            {
                options.shDegreeOuter = parseU32(*value);
                if (!options.shDegreeOuter)
                    VULTRA_CLIENT_WARN("Ignoring invalid --sh-degree-outer value: {}", *value);
                continue;
            }

        }
        if (options.formalE1E2E3)
        {
            options.cachedSelectionOracle = true;
            options.logChurn = true;
            options.benchmarkEnabled = true;
            options.selectedIdLogEnabled = true;
            options.selectedIdLogStride = 1u;
            options.benchmarkCameraPath = BenchmarkCameraPath::eStatic;
        }

        if (options.cachedSelectionOracle)
        {
            options.benchmarkEnabled = true;
            options.selectedIdLogEnabled = true;
            options.selectedIdLogStride = 1u;
            options.benchmarkCameraPath = BenchmarkCameraPath::eStatic;
            if (options.fixedGaze && !options.formalE1E2E3)
                options.benchmarkGazePath = BenchmarkGazePath::eStatic;
            if (options.outputPath == std::filesystem::path {"build/gaussian_splat_benchmark.csv"})
                options.outputPath = options.outputDir / "benchmark.csv";
        }
        if (options.shaderAntiPopMode && *options.shaderAntiPopMode != GaussianSplatShaderAntiPopMode::eOff)
        {
            options.benchmarkEnabled = true;
            if (!options.mode)
                options.mode = GaussianSplatBaselineMode::eOrderedClod;
            if (!options.foveatedClodEnabled)
                options.foveatedClodEnabled = true;
            if (options.outputPath == std::filesystem::path {"build/gaussian_splat_benchmark.csv"})
                options.outputPath =
                    options.shaderAntiPopOutputDir.value_or(
                        options.gpuAntiPopCliNaming ? std::filesystem::path {"build/gpu_antipop_vertical_slice"} :
                                                       std::filesystem::path {"build/shader_antipop_vertical_slice"}) /
                    (options.gpuAntiPopCliNaming ? "gpu_antipop_vertical_slice_per_frame.csv" :
                                                   "shader_antipop_vertical_slice_per_frame.csv");
        }
        if (options.cleanTiming)
        {
            options.logFlicker = false;
            options.captureFrameSequence = false;
            options.captureAlpha = false;
            options.selectedIdLogEnabled = false;
            options.benchmarkScreenshotOutput.reset();
            options.e3HonestE6 = false;
        }
        if (options.mouseGazeControl)
            options.benchmarkEnabled = false;

        return options;
    }

    void applyGaussianSplatOverride(World& world, IAssetService& assets, const std::string& uri)
    {
        auto handle = assets.loadGaussianSplatSync(uri);
        if (!handle || !handle.uuid().valid() || handle.state() == AssetState::eFailed)
        {
            VULTRA_CLIENT_WARN("Ignoring --splat override; failed to resolve {}", uri);
            return;
        }

        uint32_t patched = 0;
        auto     view    = world.registry().view<GaussianSplatComponent>();
        for (const auto entity : view)
        {
            view.get<GaussianSplatComponent>(entity).gaussianSplat = handle.uuid();
            ++patched;
        }

        if (patched == 0)
            VULTRA_CLIENT_WARN("Resolved --splat {}, but the scene has no GaussianSplatComponent", uri);
    }

    bool hasGazeRenderingParameterOverride(const GaussianDemoOptions& options)
    {
        return options.foveatedRenderMode.has_value() ||
               options.gazeX.has_value() || options.gazeY.has_value() ||
               options.foveaDegrees.has_value() || options.midDegrees.has_value() ||
               options.foveaLod.has_value() || options.midLod.has_value() ||
               options.outerLod.has_value() || options.foveaResolutionScale.has_value() ||
               options.midResolutionScale.has_value() || options.outerResolutionScale.has_value() ||
               options.transitionDegrees.has_value() || options.foveatedAdaptationMode.has_value() ||
               options.foveatedDistribution.has_value() ||
               options.continuousTheta0Degrees.has_value() ||
               options.continuousAlpha.has_value() ||
               options.continuousMinLevel.has_value() ||
               options.coverageGuardMode.has_value() ||
               options.foveatedCoverageCompensationEnabled.has_value() ||
               options.targetFrameMs.has_value() || options.budgetAdjustRate.has_value() ||
               options.coverageProtectionDegrees.has_value() ||
               options.coverageGuardBudgetRatio.has_value() ||
               options.coverageGuardCenterMin.has_value() ||
               options.coverageGuardTransitionMin.has_value() ||
               options.coverageGuardPeripheryMin.has_value() ||
               options.coverageGuardMaxAdds.has_value() ||
               options.foveatedTemporalHysteresisEnabled.has_value() ||
               options.foveatedBoundarySmoothingEnabled.has_value() ||
               options.foveatedTemporalResidencyFrames.has_value() ||
               options.foveatedTemporalHysteresisRatio.has_value() ||
               options.foveatedBoundarySmoothingRatio.has_value() ||
               options.foveatedTemporalPeripheralScale.has_value() ||
               options.foveatedShLodEnabled.has_value() ||
               options.foveatedShSmoothSuppressionEnabled.has_value() ||
               options.peripheralTemporalFilterEnabled.has_value() ||
               options.shDegreeCenter.has_value() ||
               options.shDegreeMid.has_value() ||
               options.shDegreeOuter.has_value() ||
               options.shLodGuardMode.has_value() ||
               options.shLodGuardThresholdMid.has_value() ||
               options.shLodGuardThresholdHigh.has_value() ||
               options.shaderAntiPopMode.has_value();
    }

    bool hasGazeRenderingOverride(const GaussianDemoOptions& options)
    {
        if (options.foveatedClodEnabled.has_value())
            return *options.foveatedClodEnabled;
        return hasGazeRenderingParameterOverride(options);
    }

    bool hasOrderedClodOverride(const GaussianDemoOptions& options)
    {
        return options.clodLevel.has_value() || options.lodBudget.has_value() ||
               options.lodBudgetMode.has_value() || options.projectedCostBudget.has_value() ||
               options.projectedCostBudgetRatio.has_value() || options.projectedCostChunkSize.has_value() ||
               options.foveatedScoreSelectionStride.has_value() ||
               options.foveatedCoverageBinGridX.has_value() ||
               options.foveatedCoverageBinGridY.has_value() ||
               options.foveatedCoverageBinQuotaRatio.has_value() ||
               options.foveatedCoverageBinPrefixRatio.has_value() ||
               options.projectedCostBuildMode.has_value() ||
               options.projectedCostSemanticDiffEnabled.has_value() ||
               options.shaderAntiPopMode.has_value() ||
               hasGazeRenderingOverride(options);
    }

    void applyFoveatedExampleDefaults(GaussianSplatRenderSettings& settings)
    {
        settings.baselineMode                      = GaussianSplatBaselineMode::eOrderedClod;
        settings.clodLevel                         = 1.0f;
        settings.foveatedClodEnabled               = true;
        settings.foveatedManualGazeControlEnabled  = false;
        settings.foveatedCoverageCompensationEnabled = true;
        settings.foveatedRenderMode               = GaussianSplatFoveatedRenderMode::eSinglePass;
        settings.foveatedGaze                     = glm::vec2 {0.5f, 0.5f};
        settings.foveatedRingDegrees              = glm::vec2 {12.0f, 32.0f};
        settings.foveatedRingLevels               = glm::vec3 {1.0f, 0.40f, 0.15f};
        settings.foveatedResolutionScales         = glm::vec3 {1.0f, 0.75f, 0.50f};
        settings.foveatedTransitionDegrees        = 8.0f;
        settings.foveatedDistribution             = GaussianSplatFoveatedDistribution::eGaussian;
        settings.foveatedContinuousTheta0Degrees  = 24.0f;
        settings.foveatedContinuousAlpha          = 2.0f;
        settings.foveatedContinuousMinLevel       = 0.12f;
        settings.foveatedAdaptationMode           = GaussianSplatFoveatedAdaptationMode::eFixed;
        settings.foveatedTargetFrameMs            = 11.1f;
        settings.foveatedBudgetAdjustRate         = 0.05f;
        settings.foveatedCoverageGuardMode         = GaussianSplatFoveatedCoverageGuardMode::eGlobal;
        settings.foveatedCoverageProtectionDegrees = 0.0f;
        settings.foveatedCoverageGuardBudgetRatio  = 0.05f;
        settings.foveatedCoverageGuardMinLevels    = glm::vec3 {0.75f, 0.45f, 0.15f};
        settings.foveatedCoverageGuardMaxAdds      = 0u;
        settings.foveatedTemporalHysteresisEnabled = true;
        settings.foveatedBoundarySmoothingEnabled  = true;
        settings.foveatedTemporalResidencyFrames   = 6u;
        settings.foveatedTemporalHysteresisRatio   = 0.25f;
        settings.foveatedBoundarySmoothingRatio    = 0.18f;
        settings.foveatedTemporalPeripheralScale   = 0.0f;
    }

    void applyProgressiveCenterOutDefaults(GaussianSplatRenderSettings& settings)
    {
        settings.foveatedRenderMode        = GaussianSplatFoveatedRenderMode::eSinglePass;
        settings.foveatedRingDegrees       = glm::vec2 {8.0f, 24.0f};
        settings.foveatedRingLevels        = glm::vec3 {0.35f, 0.18f, 0.08f};
        settings.foveatedResolutionScales  = glm::vec3 {1.0f, 0.75f, 0.50f};
        settings.foveatedTransitionDegrees = 8.0f;
        settings.foveatedDistribution      = GaussianSplatFoveatedDistribution::eGaussian;
        settings.foveatedContinuousTheta0Degrees = 24.0f;
        settings.foveatedContinuousAlpha = 2.0f;
        settings.foveatedContinuousMinLevel = settings.foveatedRingLevels.z;
        settings.foveatedTemporalPeripheralScale = 0.0f;
        settings.foveatedCoverageCompensationEnabled = true;
        settings.foveatedCoverageGuardMode = GaussianSplatFoveatedCoverageGuardMode::eGlobal;
        settings.foveatedCoverageProtectionDegrees = 3.0f;
        settings.foveatedCoverageGuardBudgetRatio = 0.05f;
        settings.foveatedCoverageGuardMinLevels = glm::vec3 {0.75f, 0.45f, 0.15f};
        settings.foveatedCoverageGuardMaxAdds = 0u;
        settings.foveatedTargetFrameMs = 8.333f;
        settings.foveatedBudgetAdjustRate = 0.02f;
    }

    void applyGaussianDemoOptions(const GaussianDemoOptions& options, GaussianSplatRenderSettings& settings)
    {
        if (options.mode)
            settings.baselineMode = *options.mode;
        if (options.clodLevel)
            settings.clodLevel = std::clamp(*options.clodLevel, 0.0f, 1.0f);
        if (options.lodBudget)
            settings.lodBudget = *options.lodBudget;
        if (options.lodBudgetMode)
            settings.lodBudgetMode = *options.lodBudgetMode;
        if (options.projectedCostBudget)
            settings.projectedCostBudget = *options.projectedCostBudget;
        if (options.projectedCostBudgetRatio)
            settings.projectedCostBudgetRatio = std::clamp(*options.projectedCostBudgetRatio, 0.0f, 1.0f);
        if (options.projectedCostChunkSize && *options.projectedCostChunkSize > 0u)
            settings.projectedCostChunkSize = *options.projectedCostChunkSize;
        if (options.foveatedScoreSelectionStride && *options.foveatedScoreSelectionStride > 0u)
            settings.foveatedScoreSelectionStride = *options.foveatedScoreSelectionStride;
        if (options.foveatedCoverageBinGridX && *options.foveatedCoverageBinGridX > 0u)
            settings.foveatedCoverageBinGridX = *options.foveatedCoverageBinGridX;
        if (options.foveatedCoverageBinGridY && *options.foveatedCoverageBinGridY > 0u)
            settings.foveatedCoverageBinGridY = *options.foveatedCoverageBinGridY;
        if (options.foveatedCoverageBinQuotaRatio)
            settings.foveatedCoverageBinQuotaRatio =
                std::clamp(*options.foveatedCoverageBinQuotaRatio, 0.0f, 0.80f);
        if (options.foveatedCoverageBinPrefixRatio)
            settings.foveatedCoverageBinPrefixRatio =
                std::clamp(*options.foveatedCoverageBinPrefixRatio, 0.0f, 0.90f);
        if (options.projectedCostBuildMode)
            settings.projectedCostBuildMode = *options.projectedCostBuildMode;
        if (options.projectedCostSemanticDiffEnabled.has_value())
            settings.projectedCostSemanticDiffEnabled = *options.projectedCostSemanticDiffEnabled;
        settings.cleanTimingMode = options.cleanTiming;
        settings.cachedSelectionOracleDeterministicSourceOrderSort =
            options.logFlicker &&
            (options.cachedSelectionOracle ||
             (options.shaderAntiPopMode && *options.shaderAntiPopMode != GaussianSplatShaderAntiPopMode::eOff));

        if (hasGazeRenderingParameterOverride(options) && !options.foveatedClodEnabled.has_value())
            settings.foveatedClodEnabled = true;
        if (options.foveatedClodEnabled.has_value())
            settings.foveatedClodEnabled = *options.foveatedClodEnabled;
        if (options.foveatedCoverageCompensationEnabled.has_value())
            settings.foveatedCoverageCompensationEnabled = *options.foveatedCoverageCompensationEnabled;
        if (options.foveatedRenderMode)
            settings.foveatedRenderMode = *options.foveatedRenderMode;
        if (options.gazeX)
            settings.foveatedGaze.x = std::clamp(*options.gazeX, 0.0f, 1.0f);
        if (options.gazeY)
            settings.foveatedGaze.y = std::clamp(*options.gazeY, 0.0f, 1.0f);
        if (options.foveaDegrees)
            settings.foveatedRingDegrees.x = std::max(*options.foveaDegrees, 0.0f);
        if (options.midDegrees)
            settings.foveatedRingDegrees.y = std::max(*options.midDegrees, 0.0f);
        settings.foveatedRingDegrees.y =
            std::max(settings.foveatedRingDegrees.y, settings.foveatedRingDegrees.x);
        if (options.foveaLod)
            settings.foveatedRingLevels.x = std::clamp(*options.foveaLod, 0.0f, 1.0f);
        if (options.midLod)
            settings.foveatedRingLevels.y = std::clamp(*options.midLod, 0.0f, 1.0f);
        if (options.outerLod)
            settings.foveatedRingLevels.z = std::clamp(*options.outerLod, 0.0f, 1.0f);
        if (options.foveaResolutionScale)
            settings.foveatedResolutionScales.x = std::clamp(*options.foveaResolutionScale, 0.05f, 1.0f);
        if (options.midResolutionScale)
            settings.foveatedResolutionScales.y = std::clamp(*options.midResolutionScale, 0.05f, 1.0f);
        if (options.outerResolutionScale)
            settings.foveatedResolutionScales.z = std::clamp(*options.outerResolutionScale, 0.05f, 1.0f);
        if (options.transitionDegrees)
            settings.foveatedTransitionDegrees = std::max(*options.transitionDegrees, 0.0f);
        if (options.foveatedAdaptationMode)
            settings.foveatedAdaptationMode = *options.foveatedAdaptationMode;
        if (options.foveatedDistribution)
            settings.foveatedDistribution = *options.foveatedDistribution;
        if (options.mouseGazeControl)
        {
            settings.foveatedClodEnabled = true;
            settings.foveatedManualGazeControlEnabled = true;
            settings.foveatedRenderMode = GaussianSplatFoveatedRenderMode::eSinglePass;
        }
        if (options.continuousTheta0Degrees)
            settings.foveatedContinuousTheta0Degrees = std::max(*options.continuousTheta0Degrees, 1e-4f);
        if (options.continuousAlpha)
            settings.foveatedContinuousAlpha = std::max(*options.continuousAlpha, 0.0f);
        if (options.continuousMinLevel)
            settings.foveatedContinuousMinLevel = std::clamp(*options.continuousMinLevel, 0.0f, 1.0f);
        else if (settings.foveatedDistribution == GaussianSplatFoveatedDistribution::eContinuousScheduler)
            settings.foveatedContinuousMinLevel = std::clamp(settings.foveatedRingLevels.z, 0.0f, 1.0f);
        if (options.targetFrameMs)
            settings.foveatedTargetFrameMs = std::max(*options.targetFrameMs, 0.1f);
        if (options.budgetAdjustRate)
            settings.foveatedBudgetAdjustRate = std::clamp(*options.budgetAdjustRate, 0.001f, 0.25f);
        if (options.coverageGuardMode)
            settings.foveatedCoverageGuardMode = *options.coverageGuardMode;
        if (options.coverageProtectionDegrees)
            settings.foveatedCoverageProtectionDegrees = std::clamp(*options.coverageProtectionDegrees, 0.0f, 12.0f);
        if (options.coverageGuardBudgetRatio)
            settings.foveatedCoverageGuardBudgetRatio = std::clamp(*options.coverageGuardBudgetRatio, 0.0f, 0.25f);
        if (options.coverageGuardCenterMin)
            settings.foveatedCoverageGuardMinLevels.x = std::clamp(*options.coverageGuardCenterMin, 0.0f, 1.0f);
        if (options.coverageGuardTransitionMin)
            settings.foveatedCoverageGuardMinLevels.y = std::clamp(*options.coverageGuardTransitionMin, 0.0f, 1.0f);
        if (options.coverageGuardPeripheryMin)
            settings.foveatedCoverageGuardMinLevels.z = std::clamp(*options.coverageGuardPeripheryMin, 0.0f, 1.0f);
        if (options.coverageGuardMaxAdds)
            settings.foveatedCoverageGuardMaxAdds = *options.coverageGuardMaxAdds;
        if (options.foveatedTemporalHysteresisEnabled)
            settings.foveatedTemporalHysteresisEnabled = *options.foveatedTemporalHysteresisEnabled;
        if (options.foveatedBoundarySmoothingEnabled)
            settings.foveatedBoundarySmoothingEnabled = *options.foveatedBoundarySmoothingEnabled;
        if (options.foveatedTemporalResidencyFrames)
            settings.foveatedTemporalResidencyFrames = std::clamp(*options.foveatedTemporalResidencyFrames, 0u, 255u);
        if (options.foveatedTemporalHysteresisRatio)
            settings.foveatedTemporalHysteresisRatio =
                std::clamp(*options.foveatedTemporalHysteresisRatio, 0.0f, 1.0f);
        if (options.foveatedBoundarySmoothingRatio)
            settings.foveatedBoundarySmoothingRatio =
                std::clamp(*options.foveatedBoundarySmoothingRatio, 0.0f, 1.0f);
        if (options.foveatedTemporalPeripheralScale)
            settings.foveatedTemporalPeripheralScale =
                std::clamp(*options.foveatedTemporalPeripheralScale, 0.0f, 4.0f);
        if (options.foveatedShLodEnabled.has_value())
            settings.foveatedShLodEnabled = *options.foveatedShLodEnabled;
        if (options.foveatedShSmoothSuppressionEnabled.has_value())
            settings.foveatedShSmoothSuppressionEnabled = *options.foveatedShSmoothSuppressionEnabled;
        if (options.peripheralTemporalFilterEnabled.has_value())
            settings.peripheralTemporalFilterEnabled = *options.peripheralTemporalFilterEnabled;
        if (options.shDegreeCenter)
            settings.foveatedShLodDegrees.x = std::min(*options.shDegreeCenter, 3u);
        if (options.shDegreeMid)
            settings.foveatedShLodDegrees.y = std::min(*options.shDegreeMid, 3u);
        if (options.shDegreeOuter)
            settings.foveatedShLodDegrees.z = std::min(*options.shDegreeOuter, 3u);
        if (options.shLodGuardMode)
            settings.foveatedShLodGuardMode = *options.shLodGuardMode;
        if (options.shLodGuardThresholdMid)
            settings.foveatedShLodGuardThresholdMid = std::max(*options.shLodGuardThresholdMid, 0.0f);
        if (options.shLodGuardThresholdHigh)
            settings.foveatedShLodGuardThresholdHigh =
                std::max(*options.shLodGuardThresholdHigh, settings.foveatedShLodGuardThresholdMid);
        if (options.shStorageLayout)
            settings.shStorageLayout = *options.shStorageLayout;
        if (options.shPopLogEnabled.has_value())
            settings.shPopLogEnabled = *options.shPopLogEnabled;
        if (options.shDegreeHysteresisEnabled.has_value())
            settings.shDegreeHysteresisEnabled = *options.shDegreeHysteresisEnabled;
        if (options.shDegreeDowngradeDelay)
            settings.shDegreeDowngradeDelay = *options.shDegreeDowngradeDelay;
        if (options.shDegreeGuardBandDegrees)
            settings.shDegreeGuardBandDegrees = std::max(*options.shDegreeGuardBandDegrees, 0.0f);        if (options.shaderAntiPopMode)
            settings.shaderAntiPopMode = *options.shaderAntiPopMode;
        if (options.shaderAntiPopHashSeed)
            settings.shaderAntiPopHashSeed = *options.shaderAntiPopHashSeed;
        else if (options.seed != 0u && settings.shaderAntiPopMode != GaussianSplatShaderAntiPopMode::eOff)
            settings.shaderAntiPopHashSeed = options.seed;
        if (options.shaderAntiPopRampWidth)
            settings.shaderAntiPopRampWidth = std::clamp(*options.shaderAntiPopRampWidth, 0.0f, 1.0f);
        if (options.shaderAntiPopGuardThreshold)
            settings.shaderAntiPopGuardThreshold =
                std::clamp(*options.shaderAntiPopGuardThreshold, 0.0f, 1.0f);
        if (options.shaderAntiPopGuardFloor)
            settings.shaderAntiPopGuardFloor = std::clamp(*options.shaderAntiPopGuardFloor, 0.0f, 1.0f);
        if (options.shaderAntiPopPKeepCurve)
            settings.shaderAntiPopPKeepCurve = *options.shaderAntiPopPKeepCurve;
        if (options.shaderAntiPopPrefixRatio)
            settings.shaderAntiPopPrefixRatio = *options.shaderAntiPopPrefixRatio;
        if (options.shaderAntiPopNormalizeMode)
            settings.shaderAntiPopNormalizeMode = *options.shaderAntiPopNormalizeMode;
        if (options.shaderAntiPopNormalizeStrength)
            settings.shaderAntiPopNormalizeStrength =
                std::clamp(*options.shaderAntiPopNormalizeStrength, 0.0f, 1.0f);
        if (options.shaderAntiPopNormalizeClampMin)
            settings.shaderAntiPopNormalizeClampMin = std::max(*options.shaderAntiPopNormalizeClampMin, 0.0f);
        if (options.shaderAntiPopNormalizeClampMax)
            settings.shaderAntiPopNormalizeClampMax =
                std::max(*options.shaderAntiPopNormalizeClampMax, settings.shaderAntiPopNormalizeClampMin);
        if (options.shaderAntiPopNormalizeFactor)
            settings.shaderAntiPopNormalizeFactor =
                std::clamp(*options.shaderAntiPopNormalizeFactor,
                           settings.shaderAntiPopNormalizeClampMin,
                           settings.shaderAntiPopNormalizeClampMax);
        if (options.coverageStableReleaseCoverageFloorEnabled.has_value())
            settings.coverageStableReleaseCoverageFloorEnabled =
                *options.coverageStableReleaseCoverageFloorEnabled;
        if (options.coverageStableReleaseStableHashEnabled.has_value())
            settings.coverageStableReleaseStableHashEnabled =
                *options.coverageStableReleaseStableHashEnabled;
        if (options.coverageStableReleaseStaggeredReleaseEnabled.has_value())
            settings.coverageStableReleaseStaggeredReleaseEnabled =
                *options.coverageStableReleaseStaggeredReleaseEnabled;
        if (options.coverageStableReleaseSaturatedFloorEnabled.has_value())
            settings.coverageStableReleaseSaturatedFloorEnabled =
                *options.coverageStableReleaseSaturatedFloorEnabled;
        if (options.coverageStableReleaseLambda)
            settings.coverageStableReleaseLambda =
                std::clamp(*options.coverageStableReleaseLambda, 0.0f, 1.0f);
        if (options.coverageStableReleaseDMin)
            settings.coverageStableReleaseDMin = std::max(*options.coverageStableReleaseDMin, 0.0f);
        if (options.coverageStableReleaseSigmaMax)
            settings.coverageStableReleaseSigmaMax = std::max(*options.coverageStableReleaseSigmaMax, 0.0f);
        if (options.coverageStableReleaseFloorEma)
            settings.coverageStableReleaseFloorEma =
                std::clamp(*options.coverageStableReleaseFloorEma, 0.0f, 1.0f);
        if (options.coverageStableReleaseReleaseEpsilon)
            settings.coverageStableReleaseReleaseEpsilon =
                std::max(*options.coverageStableReleaseReleaseEpsilon, 0.0f);
        if (options.coverageStableReleaseTileGridX)
            settings.coverageStableReleaseTileGridX =
                std::clamp(*options.coverageStableReleaseTileGridX, 1u, 32u);
        if (options.coverageStableReleaseTileGridY)
            settings.coverageStableReleaseTileGridY =
                std::clamp(*options.coverageStableReleaseTileGridY, 1u, 16u);
        if (options.coverageTextureFloorEnabled.has_value())
            settings.coverageTextureFloorEnabled =
                *options.coverageTextureFloorEnabled;
        if (options.coverageTextureFloorWidth)
            settings.coverageTextureFloorWidth =
                std::clamp(*options.coverageTextureFloorWidth, 1u, 1024u);
        if (options.coverageTextureFloorHeight)
            settings.coverageTextureFloorHeight =
                std::clamp(*options.coverageTextureFloorHeight, 1u, 1024u);
        if (options.coverageTextureFloorDMin)
            settings.coverageTextureFloorDMin =
                std::max(*options.coverageTextureFloorDMin, 0.0f);
        if (options.coverageTextureFloorSigmaMax)
            settings.coverageTextureFloorSigmaMax =
                std::max(*options.coverageTextureFloorSigmaMax, 0.0f);
        if (options.coverageTextureFloorStrength)
            settings.coverageTextureFloorStrength =
                std::clamp(*options.coverageTextureFloorStrength, 0.0f, 1.0f);
        if (options.coverageTextureFloorHistoryBeta)
            settings.coverageTextureFloorHistoryBeta =
                std::clamp(*options.coverageTextureFloorHistoryBeta, 0.0f, 1.0f);
        if (options.coverageTextureFloorUpdateInterval)
            settings.coverageTextureFloorUpdateInterval =
                std::max(*options.coverageTextureFloorUpdateInterval, 1u);
        if (options.coverageTextureFloorDebugEnabled.has_value())
            settings.coverageTextureFloorDebugEnabled =
                *options.coverageTextureFloorDebugEnabled;
        if (options.guideBeforeDiscardEnabled.has_value())
            settings.guideBeforeDiscardEnabled =
                *options.guideBeforeDiscardEnabled;
        if (options.delayedGuideLogPolarEnabled.has_value())
            settings.delayedGuideLogPolarEnabled =
                *options.delayedGuideLogPolarEnabled;
        if (options.delayedGuideTemporalLogPolarEnabled.has_value())
            settings.delayedGuideTemporalLogPolarEnabled =
                *options.delayedGuideTemporalLogPolarEnabled;
        if (options.delayedGuideTemporalReleasePolicy)
            settings.delayedGuideTemporalReleasePolicy =
                *options.delayedGuideTemporalReleasePolicy;
        if (options.delayedGuideTemporalReleaseCapRatio)
            settings.delayedGuideTemporalReleaseCapRatio =
                std::clamp(*options.delayedGuideTemporalReleaseCapRatio, 0.0f, 1.0f);
        if (options.delayedGuideTemporalRiskThreshold)
            settings.delayedGuideTemporalRiskThreshold =
                std::clamp(*options.delayedGuideTemporalRiskThreshold, 0.0f, 1.0f);
        if (options.delayedGuideTemporalFootprintDecayScale)
            settings.delayedGuideTemporalFootprintDecayScale =
                std::clamp(*options.delayedGuideTemporalFootprintDecayScale, 0.0f, 1.0f);
        if (options.delayedGuideTemporalLargeFootprintPx)
            settings.delayedGuideTemporalLargeFootprintPx =
                std::max(*options.delayedGuideTemporalLargeFootprintPx, 1.0f);
        if (settings.shaderAntiPopMode == GaussianSplatShaderAntiPopMode::eLinearDefault)
            settings.shaderAntiPopPKeepCurve = GaussianSplatShaderAntiPopPKeepCurve::eLinear;
        if (settings.shaderAntiPopMode == GaussianSplatShaderAntiPopMode::eCoverageStableLogpolarRelease)
        {
            settings.shaderAntiPopPKeepCurve = GaussianSplatShaderAntiPopPKeepCurve::eLinear;
            settings.shaderAntiPopOrderFree = true;
            if (!options.foveatedDistribution.has_value())
                settings.foveatedDistribution = GaussianSplatFoveatedDistribution::eLogPolar;
            settings.foveatedCoverageCompensationEnabled = false;
            settings.foveatedCoverageGuardMode = GaussianSplatFoveatedCoverageGuardMode::eOff;
            settings.foveatedTemporalHysteresisEnabled = false;
            settings.foveatedBoundarySmoothingEnabled = false;
        }
        settings.shaderAntiPopOrderFree = options.runtimeAntiPopOrderFreeHashRamp;
        if (settings.shaderAntiPopMode == GaussianSplatShaderAntiPopMode::eCoverageStableLogpolarRelease)
            settings.shaderAntiPopOrderFree = true;
        if (settings.guideBeforeDiscardEnabled)
        {
            settings.shaderAntiPopMode = GaussianSplatShaderAntiPopMode::eCoverageStableLogpolarRelease;
            settings.shaderAntiPopPKeepCurve = GaussianSplatShaderAntiPopPKeepCurve::eLinear;
            settings.shaderAntiPopOrderFree = true;
            settings.baselineMode = GaussianSplatBaselineMode::eBaseline;
            settings.foveatedClodEnabled = true;
            if (!options.foveatedDistribution.has_value())
                settings.foveatedDistribution = GaussianSplatFoveatedDistribution::eLogPolar;
            settings.coverageStableReleaseCoverageFloorEnabled = true;
            settings.coverageTextureFloorEnabled = false;
            settings.coverageStableReleaseStableHashEnabled = true;
            settings.coverageStableReleaseStaggeredReleaseEnabled = false;
            settings.foveatedCoverageCompensationEnabled = false;
            settings.foveatedCoverageGuardMode = GaussianSplatFoveatedCoverageGuardMode::eOff;
            settings.foveatedTemporalHysteresisEnabled = false;
            settings.foveatedBoundarySmoothingEnabled = false;
            settings.delayedGuideLogPolarEnabled = false;
            settings.delayedGuideTemporalLogPolarEnabled = false;
        }
        if (settings.delayedGuideLogPolarEnabled)
        {
            settings.shaderAntiPopMode = GaussianSplatShaderAntiPopMode::eCoverageStableLogpolarRelease;
            settings.shaderAntiPopPKeepCurve = GaussianSplatShaderAntiPopPKeepCurve::eLinear;
            settings.shaderAntiPopOrderFree = true;
            settings.baselineMode = GaussianSplatBaselineMode::eBaseline;
            settings.foveatedClodEnabled = true;
            if (!options.foveatedDistribution.has_value())
                settings.foveatedDistribution = GaussianSplatFoveatedDistribution::eLogPolar;
            settings.coverageStableReleaseCoverageFloorEnabled = false;
            settings.coverageTextureFloorEnabled = true;
            if (!options.coverageTextureFloorDebugEnabled.has_value())
                settings.coverageTextureFloorDebugEnabled = false;
            settings.foveatedCoverageCompensationEnabled = false;
            settings.foveatedCoverageGuardMode = GaussianSplatFoveatedCoverageGuardMode::eOff;
            settings.foveatedTemporalHysteresisEnabled = false;
            settings.foveatedBoundarySmoothingEnabled = false;
            settings.coverageStableReleaseStaggeredReleaseEnabled =
                settings.delayedGuideTemporalLogPolarEnabled;
        }
        settings.shaderAntiPopDebugLogEnabled = options.shaderAntiPopDebugLog;
        if (settings.shaderAntiPopMode != GaussianSplatShaderAntiPopMode::eOff)
        {
            if (!settings.shaderAntiPopOrderFree)
                settings.baselineMode = GaussianSplatBaselineMode::eOrderedClod;
            settings.foveatedClodEnabled = true;
            settings.foveatedManualGazeControlEnabled = true;
            settings.foveatedRenderMode = GaussianSplatFoveatedRenderMode::eSinglePass;
        }
        if (settings.shaderAntiPopDebugLogEnabled && settings.shaderAntiPopMode != GaussianSplatShaderAntiPopMode::eOff)
        {
            VULTRA_CLIENT_INFO(
                "GPU anti-pop CLI/settings: cli_naming={}, mode={}, hash_seed={}, ramp_width={}, "
                "guard_threshold={}, guard_floor={}, pkeep_curve={}, prefix_ratio={}, "
                "normalize_mode={}, normalize_factor={}",
                options.gpuAntiPopCliNaming ? 1 : 0,
                static_cast<uint32_t>(settings.shaderAntiPopMode),
                settings.shaderAntiPopHashSeed,
                settings.shaderAntiPopRampWidth,
                settings.shaderAntiPopGuardThreshold,
                settings.shaderAntiPopGuardFloor,
                static_cast<uint32_t>(settings.shaderAntiPopPKeepCurve),
                settings.shaderAntiPopPrefixRatio,
                static_cast<uint32_t>(settings.shaderAntiPopNormalizeMode),
                settings.shaderAntiPopNormalizeFactor);
        }
        settings.selectedIdDebugLoggingEnabled = options.selectedIdLogEnabled;
        if (options.cachedSelectionOracle)
        {
            settings.cachedSelectionOracleEnabled = true;
            settings.cachedSelectionMembershipMode = options.membershipMode;
            settings.cachedSelectionOracleSeed = options.seed;
            settings.foveatedClodEnabled = false;
            settings.foveatedCoverageCompensationEnabled = false;
            settings.foveatedTemporalHysteresisEnabled = false;
            settings.foveatedBoundarySmoothingEnabled = false;
            settings.foveatedRenderMode = GaussianSplatFoveatedRenderMode::eSinglePass;
            settings.foveatedShLodEnabled = false;
            settings.foveatedShSmoothSuppressionEnabled = false;
            settings.peripheralTemporalFilterEnabled = false;
            if (options.fixedGaze)
                settings.foveatedGaze = *options.fixedGaze;
            settings.selectedIdDebugLoggingEnabled = true;
        }

        const bool userSpecifiedProgressiveProfile =
            options.foveaDegrees.has_value() || options.midDegrees.has_value() ||
            options.foveaLod.has_value() || options.midLod.has_value() ||
            options.outerLod.has_value() || options.foveaResolutionScale.has_value() ||
            options.midResolutionScale.has_value() || options.outerResolutionScale.has_value() ||
            options.transitionDegrees.has_value() || options.foveatedDistribution.has_value() ||
            options.continuousTheta0Degrees.has_value() || options.continuousAlpha.has_value() ||
            options.continuousMinLevel.has_value() || options.foveatedTemporalPeripheralScale.has_value() ||
            options.coverageProtectionDegrees.has_value() || options.coverageGuardMode.has_value() ||
            options.coverageGuardBudgetRatio.has_value() || options.coverageGuardCenterMin.has_value() ||
            options.coverageGuardTransitionMin.has_value() || options.coverageGuardPeripheryMin.has_value() ||
            options.coverageGuardMaxAdds.has_value();
        if ((settings.foveatedAdaptationMode == GaussianSplatFoveatedAdaptationMode::eProgressiveCenterOut ||
             settings.foveatedAdaptationMode == GaussianSplatFoveatedAdaptationMode::eProgressiveGreedy ||
             settings.foveatedAdaptationMode == GaussianSplatFoveatedAdaptationMode::eStabilityAwareBudget) &&
            !userSpecifiedProgressiveProfile)
        {
            const auto explicitRenderMode = options.foveatedRenderMode;
            applyProgressiveCenterOutDefaults(settings);
            if (explicitRenderMode)
                settings.foveatedRenderMode = *explicitRenderMode;
        }

        if (options.cachedSelectionOracle)
        {
            settings.foveatedClodEnabled = false;
            settings.foveatedCoverageCompensationEnabled = false;
            settings.foveatedTemporalHysteresisEnabled = false;
            settings.foveatedBoundarySmoothingEnabled = false;
            settings.foveatedRenderMode = GaussianSplatFoveatedRenderMode::eSinglePass;
            settings.foveatedShLodEnabled = false;
            settings.selectedIdDebugLoggingEnabled = true;
        }

    }

    struct BenchmarkCameraFrame
    {
        std::string name;
        glm::vec3   position {0.0f};
        glm::vec3   target {0.0f, 0.0f, -1.0f};
        glm::vec3   up {0.0f, 1.0f, 0.0f};
        float       fovYDegrees {60.0f};
        float       zNear {0.05f};
        float       zFar {1000.0f};
    };

    std::vector<std::string> splitCsvLine(const std::string& line)
    {
        std::vector<std::string> cells;
        std::stringstream stream {line};
        std::string cell;
        while (std::getline(stream, cell, ','))
            cells.push_back(cell);
        return cells;
    }

    float csvFloatOr(const std::vector<std::string>& cells, const size_t index, const float fallback)
    {
        if (index >= cells.size() || cells[index].empty())
            return fallback;
        try
        {
            return std::stof(cells[index]);
        }
        catch (...)
        {
            return fallback;
        }
    }

    std::vector<BenchmarkCameraFrame> loadBenchmarkCameraFrames(const std::filesystem::path& path)
    {
        std::ifstream in {path};
        if (!in)
        {
            VULTRA_CLIENT_WARN("Could not open benchmark camera file: {}", path.string());
            return {};
        }

        std::vector<BenchmarkCameraFrame> frames;
        std::string line;
        bool firstLine = true;
        while (std::getline(in, line))
        {
            if (line.empty())
                continue;
            if (firstLine)
            {
                firstLine = false;
                if (line.find("name,") == 0u)
                    continue;
            }

            const auto cells = splitCsvLine(line);
            if (cells.size() < 10u)
                continue;

            BenchmarkCameraFrame frame {};
            frame.name     = cells[0];
            frame.position = {csvFloatOr(cells, 1, 0.0f), csvFloatOr(cells, 2, 0.0f), csvFloatOr(cells, 3, 0.0f)};
            frame.target   = {csvFloatOr(cells, 4, 0.0f), csvFloatOr(cells, 5, 0.0f), csvFloatOr(cells, 6, -1.0f)};
            frame.up       = {csvFloatOr(cells, 7, 0.0f), csvFloatOr(cells, 8, 1.0f), csvFloatOr(cells, 9, 0.0f)};
            frame.fovYDegrees = csvFloatOr(cells, 10, 60.0f);
            frame.zNear       = std::max(csvFloatOr(cells, 11, 0.05f), 0.001f);
            frame.zFar        = std::max(csvFloatOr(cells, 12, 1000.0f), frame.zNear + 1.0f);
            frames.push_back(frame);
        }
        return frames;
    }

    bool isProxyBenchmarkGazePath(const BenchmarkGazePath path)
    {
        switch (path)
        {
            case BenchmarkGazePath::eGridStep3x3:
            case BenchmarkGazePath::eSlowHorizontalSweep:
            case BenchmarkGazePath::eMediumHorizontalSweep:
            case BenchmarkGazePath::eFastHorizontalSweep:
            case BenchmarkGazePath::eSlowVerticalSweep:
            case BenchmarkGazePath::eSaccadeJumpCenterToEdge:
            case BenchmarkGazePath::eSaccadeJumpEdgeToCenter:
            case BenchmarkGazePath::eStepJumpCenterCornerCenter:
            case BenchmarkGazePath::eRepeatedStepJump:
                return true;
            case BenchmarkGazePath::eStatic:
            case BenchmarkGazePath::eGrid:
                return false;
        }
        return false;
    }

    BenchmarkGazeFrame legacyBenchmarkGazeFrame(const BenchmarkGazePath path, const uint32_t index)
    {
        BenchmarkGazeFrame frame {};
        if (path == BenchmarkGazePath::eStatic)
            return frame;

        static constexpr std::array<glm::vec2, 9> kGazeGrid {
            glm::vec2 {0.50f, 0.50f},
            glm::vec2 {0.35f, 0.50f},
            glm::vec2 {0.65f, 0.50f},
            glm::vec2 {0.50f, 0.35f},
            glm::vec2 {0.50f, 0.65f},
            glm::vec2 {0.35f, 0.35f},
            glm::vec2 {0.65f, 0.35f},
            glm::vec2 {0.35f, 0.65f},
            glm::vec2 {0.65f, 0.65f},
        };
        frame.scriptedGazeIndex = index % static_cast<uint32_t>(kGazeGrid.size());
        frame.gaze              = kGazeGrid[frame.scriptedGazeIndex];
        return frame;
    }

    BenchmarkGazeFrame proxyBenchmarkGazePosition(const BenchmarkGazePath path,
                                                  const uint32_t          frameIndex,
                                                  const uint32_t          frameCount)
    {
        const uint32_t totalFrames = std::max(frameCount, 1u);
        const uint32_t frame       = std::min(frameIndex, totalFrames - 1u);
        BenchmarkGazeFrame result {};

        const auto makeHorizontalSweep = [&](const float oneWaySweeps) {
            if (totalFrames == 1u)
                return result;
            const float t = static_cast<float>(frame) / static_cast<float>(totalFrames - 1u);
            const float phase = std::max(oneWaySweeps, 1.0f) * t;
            const float segmentFloat = std::floor(phase);
            const uint32_t segment = static_cast<uint32_t>(segmentFloat);
            float u = phase - segmentFloat;
            if ((segment % 2u) == 1u)
                u = 1.0f - u;
            result.scriptedGazeIndex = frame;
            result.gaze = {0.2f + 0.6f * std::clamp(u, 0.0f, 1.0f), 0.5f};
            return result;
        };

        switch (path)
        {
            case BenchmarkGazePath::eGridStep3x3:
            {
                static constexpr std::array<glm::vec2, 9> kGrid {
                    glm::vec2 {0.2f, 0.2f},
                    glm::vec2 {0.5f, 0.2f},
                    glm::vec2 {0.8f, 0.2f},
                    glm::vec2 {0.2f, 0.5f},
                    glm::vec2 {0.5f, 0.5f},
                    glm::vec2 {0.8f, 0.5f},
                    glm::vec2 {0.2f, 0.8f},
                    glm::vec2 {0.5f, 0.8f},
                    glm::vec2 {0.8f, 0.8f},
                };
                const uint32_t hold = std::max(1u, (totalFrames + static_cast<uint32_t>(kGrid.size()) - 1u) /
                                                       static_cast<uint32_t>(kGrid.size()));
                result.scriptedGazeIndex = std::min(frame / hold, static_cast<uint32_t>(kGrid.size() - 1u));
                result.gaze              = kGrid[result.scriptedGazeIndex];
                return result;
            }
            case BenchmarkGazePath::eSlowHorizontalSweep:
            {
                return makeHorizontalSweep(1.0f);
            }
            case BenchmarkGazePath::eMediumHorizontalSweep:
            {
                return makeHorizontalSweep(2.0f);
            }
            case BenchmarkGazePath::eFastHorizontalSweep:
            {
                return makeHorizontalSweep(4.0f);
            }
            case BenchmarkGazePath::eSlowVerticalSweep:
            {
                if (totalFrames == 1u)
                    return result;
                const float t = static_cast<float>(frame) / static_cast<float>(totalFrames - 1u);
                result.scriptedGazeIndex = frame;
                result.gaze              = {0.5f, 0.2f + 0.6f * t};
                return result;
            }
            case BenchmarkGazePath::eSaccadeJumpCenterToEdge:
            {
                const uint32_t jumpFrame = std::max(1u, totalFrames / 2u);
                result.scriptedGazeIndex = frame < jumpFrame ? 0u : 1u;
                result.gaze              = frame < jumpFrame ? glm::vec2 {0.5f, 0.5f} : glm::vec2 {0.9f, 0.5f};
                return result;
            }
            case BenchmarkGazePath::eSaccadeJumpEdgeToCenter:
            {
                const uint32_t jumpFrame = std::max(1u, totalFrames / 2u);
                result.scriptedGazeIndex = frame < jumpFrame ? 0u : 1u;
                result.gaze              = frame < jumpFrame ? glm::vec2 {0.9f, 0.5f} : glm::vec2 {0.5f, 0.5f};
                return result;
            }
            case BenchmarkGazePath::eStepJumpCenterCornerCenter:
            {
                const uint32_t firstJumpFrame = std::max(1u, totalFrames / 3u);
                const uint32_t secondJumpFrame = std::max(firstJumpFrame + 1u, (2u * totalFrames) / 3u);
                if (frame < firstJumpFrame)
                {
                    result.scriptedGazeIndex = 0u;
                    result.gaze = {0.5f, 0.5f};
                }
                else if (frame < secondJumpFrame)
                {
                    result.scriptedGazeIndex = 1u;
                    result.gaze = {0.85f, 0.85f};
                    result.jumpEvent = frame == firstJumpFrame ? 1u : 0u;
                }
                else
                {
                    result.scriptedGazeIndex = 2u;
                    result.gaze = {0.5f, 0.5f};
                    result.jumpEvent = frame == secondJumpFrame ? 1u : 0u;
                }
                return result;
            }
            case BenchmarkGazePath::eRepeatedStepJump:
            {
                static constexpr std::array<glm::vec2, 4> kPattern {
                    glm::vec2 {0.5f, 0.5f},
                    glm::vec2 {0.85f, 0.85f},
                    glm::vec2 {0.5f, 0.5f},
                    glm::vec2 {0.15f, 0.85f},
                };
                const uint32_t segmentLength = std::max(1u, totalFrames / 6u);
                const uint32_t segment = frame / segmentLength;
                const uint32_t patternIndex = segment % static_cast<uint32_t>(kPattern.size());
                result.scriptedGazeIndex = patternIndex;
                result.gaze = kPattern[patternIndex];
                result.jumpEvent = (frame % segmentLength) == 0u && frame > 0u ? 1u : 0u;
                return result;
            }
            case BenchmarkGazePath::eStatic:
            case BenchmarkGazePath::eGrid:
                return legacyBenchmarkGazeFrame(path, frame);
        }

        return result;
    }

    BenchmarkGazeFrame proxyBenchmarkGazeFrame(const BenchmarkGazePath path,
                                               const uint32_t          frameIndex,
                                               const uint32_t          frameCount)
    {
        static constexpr float    kMovingVelocityThreshold = 0.005f;
        static constexpr float    kJumpVelocityThreshold   = 0.20f;
        static constexpr uint32_t kPostJumpWindowRadius    = 3u;

        BenchmarkGazeFrame result = proxyBenchmarkGazePosition(path, frameIndex, frameCount);
        const uint32_t     totalFrames = std::max(frameCount, 1u);
        const uint32_t     frame       = std::min(frameIndex, totalFrames - 1u);
        const auto         previous =
            frame > 0u ? proxyBenchmarkGazePosition(path, frame - 1u, totalFrames) : result;
        const float velocity = frame > 0u ? glm::length(result.gaze - previous.gaze) : 0.0f;

        result.isMoving  = velocity >= kMovingVelocityThreshold ? 1u : 0u;
        result.jumpEvent = velocity >= kJumpVelocityThreshold ? 1u : 0u;

        uint32_t nextTransitionId = 0u;
        for (uint32_t jumpFrame = 1u; jumpFrame <= frame; ++jumpFrame)
        {
            const auto currentJump = proxyBenchmarkGazePosition(path, jumpFrame, totalFrames);
            const auto previousJump = proxyBenchmarkGazePosition(path, jumpFrame - 1u, totalFrames);
            const bool isJump = glm::length(currentJump.gaze - previousJump.gaze) >= kJumpVelocityThreshold;
            if (!isJump)
                continue;

            if (jumpFrame <= frame && frame <= jumpFrame + kPostJumpWindowRadius)
            {
                result.transitionWindowId = static_cast<int32_t>(nextTransitionId);
                break;
            }
            ++nextTransitionId;
        }

        if (result.isMoving && result.transitionWindowId < 0)
            result.transitionWindowId = 0;

        return result;
    }

    BenchmarkGazeFrame benchmarkScriptedGazeFrame(const BenchmarkGazePath path,
                                                  const uint32_t          gazeFrameIndex,
                                                  const uint32_t          frameCount)
    {
        if (isProxyBenchmarkGazePath(path))
            return proxyBenchmarkGazeFrame(path, gazeFrameIndex, frameCount);
        return legacyBenchmarkGazeFrame(path, gazeFrameIndex);
    }

    RenderCamera makeBenchmarkDatasetCamera(const BenchmarkCameraFrame& frame, const float aspect)
    {
        glm::vec3 target = frame.target;
        if (glm::dot(target - frame.position, target - frame.position) < 1e-8f)
            target = frame.position + glm::vec3 {0.0f, 0.0f, -1.0f};

        glm::vec3 up = frame.up;
        if (glm::dot(up, up) < 1e-8f)
            up = glm::vec3 {0.0f, 1.0f, 0.0f};

        RenderCamera camera {};
        camera.name        = frame.name.empty() ? "BenchmarkDatasetCamera" : frame.name;
        camera.rendererKey = "universal";
        camera.view        = glm::lookAt(frame.position, target, glm::normalize(up));
        camera.fovY        = glm::radians(std::clamp(frame.fovYDegrees, 1.0f, 170.0f));
        camera.zNear       = frame.zNear;
        camera.zFar        = frame.zFar;
        camera.projection  = glm::perspectiveRH_ZO(camera.fovY, std::max(aspect, 0.001f), camera.zNear, camera.zFar);
        camera.renderImGui = false;
        return camera;
    }

    BenchmarkCameraFrame shiftedBenchmarkCameraFrame(const BenchmarkCameraFrame& frame, const float lateralOffset)
    {
        auto shifted = frame;
        if (std::abs(lateralOffset) <= 1e-8f)
            return shifted;

        glm::vec3 forward = frame.target - frame.position;
        if (glm::dot(forward, forward) < 1e-8f)
            forward = glm::vec3 {0.0f, 0.0f, -1.0f};
        else
            forward = glm::normalize(forward);
        glm::vec3       up      = frame.up;
        if (glm::dot(up, up) < 1e-8f)
            up = glm::vec3 {0.0f, 1.0f, 0.0f};
        else
            up = glm::normalize(up);

        glm::vec3 right = glm::cross(forward, up);
        if (glm::dot(right, right) < 1e-8f)
            right = glm::vec3 {1.0f, 0.0f, 0.0f};
        else
            right = glm::normalize(right);

        const glm::vec3 offset = right * lateralOffset;
        shifted.position += offset;
        shifted.target += offset;
        return shifted;
    }

} // namespace

class GaussianSplattingDemoApp final : public DemoAppHost
{
protected:
    std::string_view demoWindowTitle() const override
    {
#if defined(VULTRA_GAUSSIAN_FOVEATED_EXAMPLE)
        return "Gaussian Splatting Foveated CLOD Demo";
#else
        return "Gaussian Splatting Demo";
#endif
    }

    bool demoEnableExperimentalWebGPUContent() const override { return true; }

    FPSCameraController makeFPSCameraController() const override
    {
        auto controller = DemoAppHost::makeFPSCameraController();
        if (commandLineEnablesBenchmark(commandLineArgs()))
        {
            controller.enabled = false;
            controller.captureMouse = false;
        }
        return controller;
    }

    void onPostConfigureDemo(Engine& engine) override
    {
        m_Options = parseDemoOptions(commandLineArgs());

        auto& sceneService = engine.ctx().services.require<ISceneService>();
        auto& worldService = engine.ctx().services.require<IWorldService>();
        auto& renderService = engine.ctx().services.require<IRenderService>();
        m_RenderService     = &renderService;

        auto& world = worldService.world();
        sceneService.instantiateScene(world, m_Options.sceneUri);
        if (m_Options.splatUri)
            applyGaussianSplatOverride(world, engine.ctx().services.require<IAssetService>(), *m_Options.splatUri);

        auto& settings = renderService.gaussianSplatSettings();
#if defined(VULTRA_GAUSSIAN_FOVEATED_EXAMPLE)
        applyFoveatedExampleDefaults(settings);
#endif
        if (!m_Options.mode && hasOrderedClodOverride(m_Options))
        {
            m_Options.mode = GaussianSplatBaselineMode::eOrderedClod;
            VULTRA_CLIENT_INFO("CLOD option detected; using gaussian mode: ordered-clod");
        }
        applyGaussianDemoOptions(m_Options, settings);
        if (m_Options.benchmarkEnabled && m_Options.benchmarkCameraFile)
        {
            m_BenchmarkCameraFrames = loadBenchmarkCameraFrames(*m_Options.benchmarkCameraFile);
            VULTRA_CLIENT_INFO("Loaded {} benchmark dataset cameras from {}", m_BenchmarkCameraFrames.size(),
                               m_Options.benchmarkCameraFile->string());
            if (m_BenchmarkCameraFrames.empty() && m_Options.benchmarkCameraPath == BenchmarkCameraPath::eDataset)
                VULTRA_CLIENT_WARN("Dataset benchmark camera path requested, but no valid cameras were loaded");
        }
        logProtocolThresholdGuardrails();
        if (m_Options.benchmarkEnabled)
            applyScriptedBenchmarkState(0u);
        if (m_Options.benchmarkEnabled && m_Options.logFlicker &&
            m_Options.benchmarkCameraPath == BenchmarkCameraPath::eStatic)
        {
            installStaticBenchmarkCameraWithoutImGui();
        }

        if (m_Options.benchmarkEnabled)
        {
            const uint32_t effectiveWarmupFrames = effectiveBenchmarkWarmupFrames();
            if (auto* profiler = renderService.runtimeProfiler())
            {
                profiler->setEnabled(true);
                m_Profiler = profiler;
                VULTRA_CLIENT_INFO(
                    "Gaussian benchmark enabled: mode={}, gaze_render_mode={}, frames={}, warmup={} (requested={}), settle={}, output={}",
                    gaussianModeLabel(settings.baselineMode), foveatedRenderModeLabel(settings.foveatedRenderMode),
                    m_Options.benchmarkFrames, effectiveWarmupFrames, m_Options.warmupFrames,
                    m_Options.benchmarkSettleFrames,
                    m_Options.outputPath.string());
                VULTRA_CLIENT_INFO("Gaussian benchmark scripted sweep: camera_path={}, gaze_path={}, view_frames={}",
                                   benchmarkCameraPathLabel(m_Options.benchmarkCameraPath),
                                   benchmarkGazePathLabel(m_Options.benchmarkGazePath),
                                   m_Options.benchmarkViewFrames);
                VULTRA_CLIENT_INFO("Gaussian benchmark stereo_sequential={}, ipd={}",
                                   m_Options.benchmarkStereoSequential ? "yes" : "no",
                                   m_Options.benchmarkIpdMeters);
                if (m_Options.cleanTiming)
                {
                    VULTRA_CLIENT_INFO(
                        "Gaussian benchmark clean timing enabled: flicker readback, frame capture, alpha capture, screenshot capture, and coverage counter readback disabled");
                }
                if (m_Options.captureFrameSequence)
                {
                    VULTRA_CLIENT_INFO("Gaussian benchmark frame capture enabled: dir={}, prefix={}, limit={}",
                                       m_Options.captureFrameDir.string(),
                                       m_Options.captureFramePrefix,
                                       m_Options.captureFrameLimit);
                }
                if (m_Options.captureAlpha)
                {
                    VULTRA_CLIENT_INFO("Gaussian benchmark alpha capture enabled: dir={}, limit={}",
                                       m_Options.captureAlphaDir.string(),
                                       m_Options.captureAlphaLimit);
                }
                if (m_Options.logFlicker)
                {
                    VULTRA_CLIENT_INFO(
                        "Regime-A image flicker logging enabled: foveal_radius={} deg, vertical_fov={} deg, capture=offscreen_scene_color, color_space=offscreen_srgb_proxy, sort_order=deterministic_source_id_sort_proxy, high_pass=3x3_delta_minus_box_blur",
                        m_Options.flickerFovealRadiusDegrees,
                        m_Options.flickerVerticalFovDegrees);
                }
                if (m_Options.cachedSelectionOracle)
                {
                    std::ostringstream fixedGazeText;
                    if (m_Options.fixedGaze)
                        fixedGazeText << m_Options.fixedGaze->x << ',' << m_Options.fixedGaze->y;
                    else
                        fixedGazeText << "<live trajectory>";
                    if (m_Options.formalE1E2E3)
                    {
                        VULTRA_CLIENT_INFO(
                            "Formal E1/E2/E3 cached oracle runner enabled: per_condition_frames={}, total_samples={}, gaze_trajectory={}, seed={}, output_dir={}",
                            m_Options.benchmarkFrames,
                            benchmarkSampleTargetCount(),
                            benchmarkGazePathLabel(m_Options.benchmarkGazePath),
                            m_Options.seed,
                            m_Options.outputDir.string());
                    }
                    VULTRA_CLIENT_INFO(
                        "E0.5 cached selection oracle enabled: membership_mode={}, fixed_gaze={}, seed={}, output_dir={}",
                        membershipModeLabel(m_Options.membershipMode),
                        fixedGazeText.str(),
                        m_Options.seed,
                        m_Options.outputDir.string());
                    VULTRA_CLIENT_INFO(
                        "E0.5 oracle mode caches frame-0 candidates and recomputes CPU membership from live gaze; shader-side foveated membership is disabled");
                }
            }
            else
            {
                VULTRA_CLIENT_WARN("Gaussian benchmark requested, but RuntimeProfiler is unavailable");
            }
        }

        VULTRA_CLIENT_INFO("Loaded world from scene manifest: \"{}\"", m_Options.sceneUri);
        if (m_Options.mouseGazeControl)
            VULTRA_CLIENT_INFO("Mouse gaze control enabled: move the cursor inside the window to drive foveated gaze");
    }

    bool onShouldClose() const override
    {
        return m_BenchmarkExitRequested || DemoAppHost::onShouldClose();
    }

    void onBeforeEngineTick(fsec /*dt*/) override
    {
        updateMouseGaze();
    }

    void onAfterEngineTick(fsec dt) override
    {
        if (!m_Options.benchmarkEnabled || !m_RenderService || !m_Profiler)
            return;

        ++m_BenchmarkTicks;
        const uint32_t effectiveWarmupFrames = effectiveBenchmarkWarmupFrames();
        const uint64_t effectiveMeasuredStart =
            static_cast<uint64_t>(effectiveWarmupFrames) + m_Options.benchmarkSettleFrames;
        if (m_BenchmarkTicks <= effectiveMeasuredStart)
        {
            applyScriptedBenchmarkState(static_cast<uint32_t>(m_BenchmarkTicks));
            return;
        }

        const auto* frame = m_Profiler->selectedFrame();
        if (!frame)
        {
            applyScriptedBenchmarkState(static_cast<uint32_t>(m_BenchmarkTicks));
            return;
        }
        if (frame->frameIndex < effectiveMeasuredStart)
        {
            applyScriptedBenchmarkState(static_cast<uint32_t>(m_BenchmarkTicks));
            return;
        }

        if (m_E3CounterfactualPending.valid)
        {
            processE3HonestCounterfactualFrame(frame->frameIndex);

            const uint32_t sampleTarget = benchmarkSampleTargetCount();
            const bool collectedEnough = m_Samples.size() >= sampleTarget;
            const bool timedOut =
                m_BenchmarkTicks > effectiveMeasuredStart + sampleTarget * 3u + 240u;
            if (m_E0OracleFailed || collectedEnough || timedOut)
            {
                if (timedOut && !collectedEnough)
                {
                    VULTRA_CLIENT_WARN("Gaussian E3-honest benchmark stopped early: collected {}/{} samples",
                                       m_Samples.size(),
                                       sampleTarget);
                }
                finishBenchmark();
                m_BenchmarkExitRequested = true;
                engineCtx().services.require<IWindowService>().window().close();
                return;
            }

            applyScriptedBenchmarkState(static_cast<uint32_t>(m_BenchmarkTicks));
            return;
        }

        if (frame->frameIndex != m_LastCollectedFrame)
        {
            m_LastCollectedFrame = frame->frameIndex;
            m_Samples.push_back(makeBenchmarkSample(static_cast<uint32_t>(m_Samples.size()), dt, *frame,
                                                    m_RenderService->gaussianSplatFrameStats(),
                                                    m_CurrentScriptedViewIndex,
                                                    m_CurrentScriptedGazeIndex,
                                                    m_CurrentIsGazeMoving,
                                                    m_CurrentGazeJumpEvent,
                                                    m_CurrentTransitionWindowId));
            updateSelectedIdLogging(m_Samples.back());
            if (m_Options.formalE1E2E3)
                recordFormalProtocolFrame(m_Samples.back());
            else
                recordCachedSelectionOracleChurn(m_Samples.back());
            recordImageFlickerFrame(m_Samples.back());
            saveBenchmarkFrameCapture(m_Samples.back());
            saveBenchmarkAlphaCapture(m_Samples.back());
            maybeScheduleE3HonestCounterfactual(m_Samples.back());
            if (m_Options.renderDocCaptureSample &&
                !m_RenderDocCaptureRequested &&
                m_Samples.size() >= *m_Options.renderDocCaptureSample)
            {
                if (auto* frameDebuggerService =
                        engineCtx().services.tryGet<IFrameDebuggerService>())
                {
                    frameDebuggerService->captureSingleFrame();
                    m_RenderDocCaptureRequested = true;
                    VULTRA_CLIENT_INFO("Gaussian benchmark requested RenderDoc capture after sample {}",
                                       m_Samples.size());
                }
                else
                {
                    VULTRA_CLIENT_WARN("RenderDoc capture requested, but frame debugger service is unavailable");
                    m_RenderDocCaptureRequested = true;
                }
            }
        }

        if (m_E3CounterfactualPending.valid)
            return;

        const uint32_t sampleTarget = benchmarkSampleTargetCount();
        const bool collectedEnough = m_Samples.size() >= sampleTarget;
        const bool timedOut =
            m_BenchmarkTicks > effectiveMeasuredStart + sampleTarget * (m_Options.e3HonestE6 ? 3u : 1u) + 240u;

        if (m_E0OracleFailed || collectedEnough || timedOut)
        {
            if (timedOut && !collectedEnough)
            {
                VULTRA_CLIENT_WARN("Gaussian benchmark stopped early: collected {}/{} samples", m_Samples.size(),
                                   sampleTarget);
            }
            if (m_E0OracleFailed)
                VULTRA_CLIENT_ERROR(
                    "Gaussian E0.5 cached selection oracle stopped after fixed-gaze churn or render/log mismatch");
            finishBenchmark();
            m_BenchmarkExitRequested = true;
            engineCtx().services.require<IWindowService>().window().close();
            return;
        }

        applyScriptedBenchmarkState(static_cast<uint32_t>(m_BenchmarkTicks));
    }

    void onBeforeShutdown(Engine& /*engine*/) override
    {
        finishBenchmark();
    }

private:
    void updateMouseGaze()
    {
        if (!m_Options.mouseGazeControl || !m_RenderService || m_Options.benchmarkEnabled)
            return;

        auto& window = engineCtx().services.require<IWindowService>().window();
        if (window.getMouseRelativeMode())
            return;

        const auto extent = window.getExtent();
        if (extent.x <= 1 || extent.y <= 1)
            return;

        const glm::vec2 mouse = engineCtx().services.require<IInputService>().getMousePosition();
        auto&           settings = m_RenderService->gaussianSplatSettings();
        settings.foveatedGaze = glm::clamp(mouse / glm::vec2 {extent}, glm::vec2 {0.0f}, glm::vec2 {1.0f});
    }

    uint32_t formalProtocolPhaseCount() const
    {
        return m_Options.e3HonestE6 ? 2u : 5u;
    }

    bool loadProtocolThresholdCommitmentStatus() const
    {
        std::ifstream in {"REGISTERED_PROTOCOL.md"};
        if (!in)
            return true;

        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return text.find("____") != std::string::npos;
    }

    void logProtocolThresholdGuardrails()
    {
        if (!m_Options.formalE1E2E3)
            return;

        const bool thresholdUncommitted = loadProtocolThresholdCommitmentStatus();
        m_Options.formalExploratoryThresholdUncommitted = thresholdUncommitted;
        const char* value = thresholdUncommitted ? "blank/uncommitted" : "see REGISTERED_PROTOCOL.md";
        VULTRA_CLIENT_INFO("Pre-registered parameter Delta_F_exist: {}", value);
        VULTRA_CLIENT_INFO("Pre-registered parameter Delta_F_null: {}", value);
        VULTRA_CLIENT_INFO("Pre-registered parameter epsilon_F: {}", value);
        VULTRA_CLIENT_INFO("Pre-registered parameter epsilon_count: {}", value);
        VULTRA_CLIENT_INFO("Pre-registered parameter alpha_diffuse: {}", value);
        VULTRA_CLIENT_INFO("Pre-registered parameter alpha_tail: {}", value);
        if (thresholdUncommitted)
        {
            VULTRA_CLIENT_WARN(
                "REGISTERED_PROTOCOL.md still contains blank threshold placeholders; E1/E2/E3 logs are exploratory_threshold_uncommitted and no pass/fail gate will be claimed");
        }
    }

    uint32_t benchmarkSampleTargetCount() const
    {
        if (m_Options.formalE1E2E3)
            return m_Options.benchmarkFrames * formalProtocolPhaseCount();
        return m_Options.benchmarkFrames;
    }

    FormalProtocolCondition formalConditionForPhase(const uint32_t phase) const
    {
        if (m_Options.e3HonestE6)
        {
            switch (phase)
            {
                case 0u:
                    return FormalProtocolCondition::eC2DynamicDeterministic;
                case 1u:
                    return FormalProtocolCondition::eC3DynamicFrozenHash;
                default:
                    break;
            }
            return FormalProtocolCondition::eNone;
        }

        switch (phase)
        {
            case 0u:
                return FormalProtocolCondition::eC2DynamicDeterministic;
            case 1u:
                return FormalProtocolCondition::eC1StaticRandom;
            case 2u:
                return FormalProtocolCondition::eC1StaticMeanCount;
            case 3u:
                return FormalProtocolCondition::eC3DynamicFrozenHash;
            case 4u:
                return FormalProtocolCondition::eC2NullMatchedRandomSwap;
            default:
                break;
        }
        return FormalProtocolCondition::eNone;
    }

    FormalProtocolCondition formalConditionForSampleIndex(const uint32_t sampleIndex) const
    {
        const uint32_t phase = std::min(sampleIndex / std::max(m_Options.benchmarkFrames, 1u),
                                        formalProtocolPhaseCount() - 1u);
        return formalConditionForPhase(phase);
    }

    uint32_t formalLocalFrameIndexForSampleIndex(const uint32_t sampleIndex) const
    {
        return sampleIndex % std::max(m_Options.benchmarkFrames, 1u);
    }

    void updateC1StaticMeanCountTarget()
    {
        if (m_C1StaticMeanCountTarget != UINT32_MAX ||
            m_C2SelectedCountSampleCount < m_Options.benchmarkFrames ||
            m_C2SelectedCountSampleCount == 0u)
            return;
        const double mean =
            static_cast<double>(m_C2SelectedCountSum) /
            static_cast<double>(m_C2SelectedCountSampleCount);
        m_C1StaticMeanCountTarget = static_cast<uint32_t>(
            std::min<double>(std::numeric_limits<uint32_t>::max(), std::llround(mean)));
    }

    void applyScriptedBenchmarkState(const uint32_t tick)
    {
        if (!m_RenderService)
            return;

        const uint32_t viewFrames = std::max(m_Options.benchmarkViewFrames, 1u);
        const uint32_t bucket = tick / viewFrames;
        const uint32_t viewCount = benchmarkScriptedViewCount();
        const uint32_t nextSampleIndex = static_cast<uint32_t>(
            std::min<uint64_t>(m_Samples.size(), std::max<uint32_t>(benchmarkSampleTargetCount(), 1u) - 1u));
        if (m_Options.formalE1E2E3)
        {
            m_CurrentFormalCondition = formalConditionForSampleIndex(nextSampleIndex);
            m_CurrentFormalLocalFrameIndex = formalLocalFrameIndexForSampleIndex(nextSampleIndex);
        }
        else
        {
            m_CurrentFormalCondition = FormalProtocolCondition::eNone;
            m_CurrentFormalLocalFrameIndex = nextSampleIndex;
        }

        const uint32_t viewIndex =
            m_Options.benchmarkCameraPath == BenchmarkCameraPath::eStatic ? 0u : bucket % viewCount;
        const uint32_t legacyGazeIndex =
            m_Options.benchmarkGazePath == BenchmarkGazePath::eGrid ? (bucket / viewCount) % 9u : 0u;
        const uint32_t gazeFrameIndex = isProxyBenchmarkGazePath(m_Options.benchmarkGazePath) ?
                                            static_cast<uint32_t>(std::min<uint64_t>(
                                                m_Options.formalE1E2E3 ? m_CurrentFormalLocalFrameIndex :
                                                    m_Samples.size(),
                                                std::numeric_limits<uint32_t>::max())) :
                                            legacyGazeIndex;
        const BenchmarkGazeFrame gazeFrame =
            benchmarkScriptedGazeFrame(m_Options.benchmarkGazePath, gazeFrameIndex, m_Options.benchmarkFrames);
        const glm::vec2 gaze =
            m_Options.fixedGaze.value_or(gazeFrame.gaze);

        m_CurrentScriptedViewIndex = viewIndex;
        m_CurrentScriptedGazeIndex = gazeFrame.scriptedGazeIndex;
        m_CurrentIsGazeMoving      = gazeFrame.isMoving;
        m_CurrentGazeJumpEvent     = gazeFrame.jumpEvent;
        m_CurrentTransitionWindowId = gazeFrame.transitionWindowId;

        auto& settings = m_RenderService->gaussianSplatSettings();
        settings.foveatedGaze = gaze;
        if (m_Options.formalE1E2E3)
        {
            updateC1StaticMeanCountTarget();
            settings.cachedSelectionMembershipMode =
                formalConditionMembershipMode(m_CurrentFormalCondition);
            settings.cachedSelectionOracleStaticTargetCount = UINT32_MAX;
            settings.cachedSelectionOracleMatchedNullFrameIndex = UINT32_MAX;
            settings.cachedSelectionOracleMatchedNullTargetAddCount = 0u;
            settings.cachedSelectionOracleMatchedNullTargetRemoveCount = 0u;
            settings.cachedSelectionOracleMatchedNullTargetSelectedCount = 0u;
            settings.cachedSelectionOracleMatchedNullTargetSymmetricDiff = 0u;

            if (m_CurrentFormalCondition == FormalProtocolCondition::eC1StaticMeanCount &&
                m_C1StaticMeanCountTarget != UINT32_MAX)
            {
                settings.cachedSelectionOracleStaticTargetCount = m_C1StaticMeanCountTarget;
            }
            else if (m_CurrentFormalCondition == FormalProtocolCondition::eC2NullMatchedRandomSwap)
            {
                const uint32_t localFrame = m_CurrentFormalLocalFrameIndex;
                if (localFrame < m_C2ScheduleRows.size())
                {
                    const auto& schedule = m_C2ScheduleRows[localFrame];
                    settings.cachedSelectionOracleMatchedNullFrameIndex = localFrame;
                    settings.cachedSelectionOracleMatchedNullTargetAddCount = schedule.addCount;
                    settings.cachedSelectionOracleMatchedNullTargetRemoveCount = schedule.removeCount;
                    settings.cachedSelectionOracleMatchedNullTargetSelectedCount = schedule.selectedCount;
                    settings.cachedSelectionOracleMatchedNullTargetSymmetricDiff =
                        schedule.symmetricDifference;
                }
            }
        }

        if (m_Options.benchmarkCameraPath == BenchmarkCameraPath::eStatic)
            return;

        auto& window = engineCtx().services.require<IWindowService>().window();
        const auto extent = window.getExtent();
        const float aspect = static_cast<float>(std::max(extent.x, 1)) /
                             static_cast<float>(std::max(extent.y, 1));
        auto& cameraService = engineCtx().services.require<ICameraService>();
        cameraService.setCameraControlInputSuppressed(true);
        cameraService.clearManualCameras();
        if (m_Options.benchmarkCameraPath == BenchmarkCameraPath::eDataset && !m_BenchmarkCameraFrames.empty())
        {
            const auto& frame =
                m_BenchmarkCameraFrames[viewIndex % static_cast<uint32_t>(m_BenchmarkCameraFrames.size())];
            if (m_Options.benchmarkStereoSequential)
            {
                const float halfIpd = std::max(m_Options.benchmarkIpdMeters, 0.0f) * 0.5f;

                auto leftEye = makeBenchmarkDatasetCamera(shiftedBenchmarkCameraFrame(frame, -halfIpd), aspect);
                leftEye.name = frame.name + "_left";
                cameraService.addManualCamera(leftEye);

                auto rightEye = makeBenchmarkDatasetCamera(shiftedBenchmarkCameraFrame(frame, halfIpd), aspect);
                rightEye.name = frame.name + "_right";
                rightEye.renderImGui = false;
                cameraService.addManualCamera(rightEye);
            }
            else
            {
                auto camera = makeBenchmarkDatasetCamera(frame, aspect);
                if (singleDatasetBenchmarkCameraForImageFlicker())
                {
                    if (ensureImageFlickerOffscreenCaptureTarget())
                    {
                        camera.name = camera.name.empty() ? "DatasetBenchmarkFlickerCapture" :
                                                            camera.name + "_FlickerCapture";
                        camera.target = &m_ImageFlickerOffscreenTarget;
                        camera.clearValue = {0.0f, 0.0f, 0.0f, 1.0f};
                        camera.renderImGui = false;
                    }
                }
                cameraService.addManualCamera(camera);
            }
        }
    }

    bool singleDatasetBenchmarkCameraForImageFlicker() const
    {
        return m_Options.logFlicker &&
               m_Options.benchmarkCameraPath == BenchmarkCameraPath::eDataset &&
               m_BenchmarkCameraFrames.size() == 1u &&
               m_Options.benchmarkViewFrames == 1u &&
               !m_Options.benchmarkStereoSequential;
    }

    bool ensureImageFlickerOffscreenCaptureTarget()
    {
        auto& backendService = engineCtx().services.require<IRenderBackendService>();
        auto& rd = backendService.renderDevice();
        const auto extent = backendService.backbuffer().getExtent();
        constexpr rhi::PixelFormat kFlickerCaptureFormat = rhi::PixelFormat::eRGBA8_UNorm;
        const bool needsTarget =
            !m_ImageFlickerOffscreenTarget ||
            m_ImageFlickerOffscreenTarget.getExtent().width != extent.width ||
            m_ImageFlickerOffscreenTarget.getExtent().height != extent.height ||
            m_ImageFlickerOffscreenTarget.getPixelFormat() != kFlickerCaptureFormat;
        if (needsTarget)
        {
            m_ImageFlickerOffscreenTarget =
                rd.createTexture2D(extent,
                                   kFlickerCaptureFormat,
                                   1u,
                                   1u,
                                   rhi::ImageUsage::eRenderTarget |
                                       rhi::ImageUsage::eTransferSrc |
                                       rhi::ImageUsage::eSampled);
            if (!m_ImageFlickerOffscreenTarget)
            {
                VULTRA_CLIENT_WARN("Regime-A image flicker logging failed to create offscreen capture target");
                return false;
            }
            VULTRA_CLIENT_INFO(
                "Regime-A image flicker logging installed offscreen capture target: {}x{}, format={}, sample_count=1, pre_ui=true",
                extent.width,
                extent.height,
                rhi::toString(kFlickerCaptureFormat));
        }
        return true;
    }

    void installStaticBenchmarkCameraWithoutImGui()
    {
        auto& cameraService = engineCtx().services.require<ICameraService>();
        const auto cameras = cameraService.cameras();
        if (cameras.empty())
        {
            VULTRA_CLIENT_WARN("Regime-A image flicker logging could not find a static benchmark camera");
            return;
        }

        if (!ensureImageFlickerOffscreenCaptureTarget())
            return;

        RenderCamera camera = cameras.front();
        camera.name = camera.name.empty() ? "StaticBenchmarkFlickerCapture" : camera.name + "_FlickerCapture";
        camera.priority = std::numeric_limits<int>::max();
        camera.target = &m_ImageFlickerOffscreenTarget;
        camera.clearValue = {0.0f, 0.0f, 0.0f, 1.0f};
        camera.renderImGui = false;
        cameraService.clearManualCameras();
        cameraService.addManualCamera(camera);
        cameraService.setCameraControlInputSuppressed(true);
    }

    void releaseImageFlickerOffscreenCaptureTarget()
    {
        if (!m_ImageFlickerOffscreenTarget)
            return;

        auto& backendService = engineCtx().services.require<IRenderBackendService>();
        backendService.renderDevice().waitIdle();
        engineCtx().services.require<ICameraService>().clearManualCameras();
        m_ImageFlickerOffscreenTarget = {};
    }

    uint32_t benchmarkScriptedViewCount() const
    {
        if (m_Options.benchmarkCameraPath == BenchmarkCameraPath::eDataset && !m_BenchmarkCameraFrames.empty())
            return static_cast<uint32_t>(m_BenchmarkCameraFrames.size());
        return 1u;
    }

    uint32_t effectiveBenchmarkWarmupFrames() const
    {
        const uint64_t requestedWarmup = m_Options.warmupFrames;
        const uint64_t viewCycleFrames =
            static_cast<uint64_t>(benchmarkScriptedViewCount()) *
            static_cast<uint64_t>(std::max(m_Options.benchmarkViewFrames, 1u));
        const uint64_t minWarmup = viewCycleFrames + 2u;
        return static_cast<uint32_t>(std::min<uint64_t>(
            std::numeric_limits<uint32_t>::max(),
            std::max(requestedWarmup, minWarmup)));
    }

    void updateSelectedIdLogging(GaussianBenchmarkSample& sample)
    {
        const auto start = std::chrono::steady_clock::now();
        sample.selectedIdLogEnabled = m_Options.selectedIdLogEnabled;
        sample.selectedIdLogStride  = std::max(1u, m_Options.selectedIdLogStride);

        if (!sample.selectedIdLogEnabled ||
            (sample.sampleIndex % sample.selectedIdLogStride) != 0u)
        {
            sample.selectedSourceIds.clear();
            sample.a4SelectedIdLoggingCpuMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            return;
        }

        if (!sample.selectedSourceIds.empty())
        {
            auto& currentIds = sample.selectedSourceIds;
            std::sort(currentIds.begin(), currentIds.end());
            currentIds.erase(std::unique(currentIds.begin(), currentIds.end()), currentIds.end());
            sample.selectedIdExactAvailable = true;
            sample.selectedIdCount          = static_cast<uint32_t>(currentIds.size());

            if (m_PreviousSelectedIdLogValid && !m_PreviousSelectedSourceIds.empty())
            {
                size_t i = 0u;
                size_t j = 0u;
                uint32_t intersection = 0u;
                while (i < m_PreviousSelectedSourceIds.size() && j < currentIds.size())
                {
                    if (m_PreviousSelectedSourceIds[i] == currentIds[j])
                    {
                        ++intersection;
                        ++i;
                        ++j;
                    }
                    else if (m_PreviousSelectedSourceIds[i] < currentIds[j])
                    {
                        ++i;
                    }
                    else
                    {
                        ++j;
                    }
                }

                const uint32_t previousCount = static_cast<uint32_t>(m_PreviousSelectedSourceIds.size());
                const uint32_t currentCount  = static_cast<uint32_t>(currentIds.size());
                const uint32_t unionCount    = previousCount + currentCount - intersection;
                sample.selectedIdAddedCount  = currentCount - intersection;
                sample.selectedIdRemovedCount = previousCount - intersection;
                sample.trueSelectedIdIou     = unionCount > 0u ?
                                                   static_cast<double>(intersection) /
                                                       static_cast<double>(unionCount) :
                                                   1.0;
                sample.trueSelectedIdChurn   = 1.0 - sample.trueSelectedIdIou;
                sample.selectedIdIou         = sample.trueSelectedIdIou;
                sample.selectedIdChurn       = sample.trueSelectedIdChurn;
            }

            m_PreviousSelectedSourceIds = currentIds;
        }
        else
        {
            sample.selectedIdCount = sample.lodSelectedRawSplats;
            if (m_PreviousSelectedIdLogValid)
            {
                const uint32_t denominator = std::max(m_PreviousSelectedIdCount, sample.selectedIdCount);
                sample.selectedIdIou = denominator > 0u ?
                                           static_cast<double>(std::min(m_PreviousSelectedIdCount,
                                                                        sample.selectedIdCount)) /
                                               static_cast<double>(denominator) :
                                           1.0;
                sample.selectedIdChurn = 1.0 - sample.selectedIdIou;
            }
        }

        m_PreviousSelectedIdLogValid = true;
        m_PreviousSelectedIdCount = sample.selectedIdCount;
        sample.a4SelectedIdLoggingCpuMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    void recordFormalChurnEvents(const FormalProtocolFrameRow& row,
                                  const std::vector<uint32_t>& addedIds,
                                  const std::vector<uint32_t>& removedIds,
                                  const GaussianSplatFrameStats& oracleStats)
    {
        if (row.condition != FormalProtocolCondition::eC2DynamicDeterministic &&
            row.condition != FormalProtocolCondition::eC3DynamicFrozenHash)
        {
            return;
        }

        if (addedIds.empty() && removedIds.empty())
            return;

        std::unordered_map<uint32_t, const GaussianSplatCachedSelectionOracleDebugSample*> sampleBySourceId;
        sampleBySourceId.reserve(oracleStats.cachedSelectionOracleCandidateDebugSamples.size());
        for (const auto& sample : oracleStats.cachedSelectionOracleCandidateDebugSamples)
            sampleBySourceId.emplace(sample.sourceId, &sample);

        auto appendEvent = [&](const uint32_t sourceId, const bool enter) {
            const auto it = sampleBySourceId.find(sourceId);
            if (it == sampleBySourceId.end())
                return;

            const auto& proxy = *it->second;
            const double footprintArea =
                proxy.projectedAreaPx > 0.0 ? proxy.projectedAreaPx :
                static_cast<double>(proxy.tileCost);
            FormalChurnEventRow event {};
            event.localFrameIndex = row.localFrameIndex;
            event.frameIndex = row.frameIndex;
            event.condition = row.condition;
            event.membershipMode = row.membershipMode;
            event.sourceId = sourceId;
            event.enter = enter;
            event.gaze = row.gaze;
            event.eccentricity = proxy.eccentricityDegrees;
            event.alphaProxy = proxy.opacityProxy;
            event.postWeightAlphaProxy = proxy.opacityProxy * proxy.keepProbability;
            event.projectedRadiusProxy =
                footprintArea > 0.0 ? std::sqrt(footprintArea / 3.14159265358979323846) : 0.0;
            event.footprintAreaProxy = footprintArea;
            event.tileCost = proxy.tileCost;
            event.transmittance = 0.0;
            event.alphaT = 0.0;
            event.contributionProxy = static_cast<double>(proxy.opacityProxy) * footprintArea;
            event.exactAlphaTAvailable = false;
            event.exactTransmittanceAvailable = false;
            m_E3ChurnEvents.push_back(event);
        };

        for (const uint32_t id : addedIds)
            appendEvent(id, true);
        for (const uint32_t id : removedIds)
            appendEvent(id, false);
    }

    void recordFormalProtocolFrame(const GaussianBenchmarkSample& sample)
    {
        if (!m_Options.formalE1E2E3 || !m_Options.cachedSelectionOracle)
            return;

        const auto& oracleStats = m_RenderService->gaussianSplatFrameStats();
        std::vector<uint32_t> currentIds = sample.selectedSourceIds;
        std::sort(currentIds.begin(), currentIds.end());
        currentIds.erase(std::unique(currentIds.begin(), currentIds.end()), currentIds.end());
        m_LastFormalFrameHasPrevious = false;
        m_LastFormalPreviousSourceIds.clear();
        m_LastFormalCurrentSourceIds.clear();
        m_LastFormalChangedSourceIds.clear();

        const FormalProtocolCondition condition = m_CurrentFormalCondition;
        const size_t conditionIndex = static_cast<size_t>(condition);
        auto& state = m_FormalConditionStates[conditionIndex];

        SelectedSetDiff diff {};
        uint32_t symmetricDifference = 0u;
        double jaccard = 1.0;
        double normalizedChurn = 0.0;
        glm::vec2 gazeVelocity {0.0f};
        if (state.previousValid)
        {
            diff = diffSelectedSourceIds(state.previousIds, currentIds);
            const uint32_t previousCount = static_cast<uint32_t>(state.previousIds.size());
            const uint32_t currentCount = static_cast<uint32_t>(currentIds.size());
            const uint32_t unionCount = previousCount + currentCount - diff.intersection;
            symmetricDifference =
                static_cast<uint32_t>(diff.added.size() + diff.removed.size());
            jaccard = unionCount > 0u ?
                          static_cast<double>(diff.intersection) / static_cast<double>(unionCount) :
                          1.0;
            normalizedChurn = currentCount > 0u ?
                                  static_cast<double>(symmetricDifference) /
                                      static_cast<double>(currentCount) :
                                  0.0;
            gazeVelocity = glm::vec2 {sample.gazeX, sample.gazeY} - state.previousGaze;
        }

        FormalProtocolFrameRow row {};
        row.sampleIndex = sample.sampleIndex;
        row.localFrameIndex = m_CurrentFormalLocalFrameIndex;
        row.frameIndex = sample.frameIndex;
        row.condition = condition;
        row.membershipMode = formalConditionMembershipMode(condition);
        row.seed = m_Options.seed;
        row.gaze = {sample.gazeX, sample.gazeY};
        row.gazeVelocity = gazeVelocity;
        row.gazeSpeed = glm::length(gazeVelocity);
        row.candidateCount = oracleStats.cachedSelectionOracleCandidateCount;
        row.selectedCount = static_cast<uint32_t>(currentIds.size());
        row.selectedFraction = oracleStats.cachedSelectionOracleSelectedFraction;
        row.selectedSetHash = hashSelectedSourceIds(currentIds);
        row.selectedSourceIdRanges = selectedSourceIdRanges(currentIds);
        row.addedCount = static_cast<uint32_t>(diff.added.size());
        row.removedCount = static_cast<uint32_t>(diff.removed.size());
        row.symmetricDifference = symmetricDifference;
        row.normalizedChurn = normalizedChurn;
        row.jaccard = jaccard;
        row.cacheFrameIndexUsed = oracleStats.cachedSelectionOracleCacheFrameIndex;
        row.membershipRecomputedThisFrame =
            oracleStats.cachedSelectionOracleMembershipRecomputedThisFrame ? 1u : 0u;
        row.renderBufferMatchesLoggedSet =
            oracleStats.cachedSelectionOracleRenderBufferMatchesLoggedSet ? 1u : 0u;
        row.shaderSideFoveationDisabled =
            oracleStats.cachedSelectionOracleShaderFoveationDisabled ? 1u : 0u;
        row.thresholdFoveaDegrees = oracleStats.cachedSelectionOracleThresholdFoveaDegrees;
        row.thresholdMidDegrees = oracleStats.cachedSelectionOracleThresholdMidDegrees;
        row.falloffTransitionDegrees = oracleStats.cachedSelectionOracleFalloffTransitionDegrees;
        row.levelCenter = oracleStats.cachedSelectionOracleLevelCenter;
        row.levelMid = oracleStats.cachedSelectionOracleLevelMid;
        row.levelOuter = oracleStats.cachedSelectionOracleLevelOuter;
        row.eccentricityMin = oracleStats.cachedSelectionOracleEccentricityMin;
        row.eccentricityMean = oracleStats.cachedSelectionOracleEccentricityMean;
        row.eccentricityMax = oracleStats.cachedSelectionOracleEccentricityMax;
        row.keepProbabilityMin = oracleStats.cachedSelectionOracleKeepProbabilityMin;
        row.keepProbabilityMean = oracleStats.cachedSelectionOracleKeepProbabilityMean;
        row.keepProbabilityMax = oracleStats.cachedSelectionOracleKeepProbabilityMax;
        row.runLabel = m_Options.formalExploratoryThresholdUncommitted ?
                           "exploratory_threshold_uncommitted" :
                           "formal_threshold_committed";
        row.cacheModeStatus = sample.sampleIndex == 0u ?
                                  "cached_frame0_candidates" :
                                  "live_membership_from_cache";
        row.notes = "selection_churn_only_flicker_gate_incomplete";

        if (condition == FormalProtocolCondition::eC2DynamicDeterministic)
        {
            m_C2SelectedCountSum += row.selectedCount;
            ++m_C2SelectedCountSampleCount;

            FormalProtocolScheduleRow schedule {};
            schedule.localFrameIndex = row.localFrameIndex;
            schedule.frameIndex = row.frameIndex;
            schedule.addCount = row.addedCount;
            schedule.removeCount = row.removedCount;
            schedule.selectedCount = row.selectedCount;
            schedule.symmetricDifference = row.symmetricDifference;
            schedule.normalizedChurn = row.normalizedChurn;
            schedule.jaccard = row.jaccard;
            schedule.selectedSetHash = row.selectedSetHash;
            schedule.changedIds.reserve(diff.added.size() + diff.removed.size());
            schedule.changedIds.insert(schedule.changedIds.end(), diff.added.begin(), diff.added.end());
            schedule.changedIds.insert(schedule.changedIds.end(), diff.removed.begin(), diff.removed.end());
            std::sort(schedule.changedIds.begin(), schedule.changedIds.end());
            schedule.changedIds.erase(std::unique(schedule.changedIds.begin(), schedule.changedIds.end()),
                                      schedule.changedIds.end());
            if (m_C2ScheduleRows.size() <= row.localFrameIndex)
                m_C2ScheduleRows.resize(row.localFrameIndex + 1u);
            m_C2ScheduleRows[row.localFrameIndex] = std::move(schedule);
        }
        else if (condition == FormalProtocolCondition::eC2NullMatchedRandomSwap)
        {
            if (row.localFrameIndex < m_C2ScheduleRows.size())
            {
                const auto& target = m_C2ScheduleRows[row.localFrameIndex];
                row.c2TargetAddCount = target.addCount;
                row.c2TargetRemoveCount = target.removeCount;
                row.c2TargetSelectedCount = target.selectedCount;
                row.c2TargetSymmetricDiff = target.symmetricDifference;
                row.c2RealizedAddCount = row.addedCount;
                row.c2RealizedRemoveCount = row.removedCount;
                row.c2RealizedSelectedCount = row.selectedCount;
                row.c2RealizedSymmetricDiff = row.symmetricDifference;

                std::vector<uint32_t> nullChangedIds;
                nullChangedIds.reserve(diff.added.size() + diff.removed.size());
                nullChangedIds.insert(nullChangedIds.end(), diff.added.begin(), diff.added.end());
                nullChangedIds.insert(nullChangedIds.end(), diff.removed.begin(), diff.removed.end());
                std::sort(nullChangedIds.begin(), nullChangedIds.end());
                nullChangedIds.erase(std::unique(nullChangedIds.begin(), nullChangedIds.end()),
                                     nullChangedIds.end());
                row.c2ChangedIdOverlap =
                    selectedSetIntersectionCount(target.changedIds, nullChangedIds);
                row.scheduleMatchingSucceeded =
                    row.c2TargetAddCount == row.c2RealizedAddCount &&
                            row.c2TargetRemoveCount == row.c2RealizedRemoveCount &&
                            row.c2TargetSelectedCount == row.c2RealizedSelectedCount &&
                            row.c2TargetSymmetricDiff == row.c2RealizedSymmetricDiff &&
                            oracleStats.cachedSelectionOracleMatchedNullScheduleMatched ?
                        1u :
                        0u;
            }
            else
            {
                row.scheduleMatchingSucceeded = 0u;
                row.notes = "missing_c2_schedule";
            }
        }

        recordFormalChurnEvents(row, diff.added, diff.removed, oracleStats);
        m_FormalRows.push_back(row);

        if (m_Options.e3HonestE6 && state.previousValid &&
            (condition == FormalProtocolCondition::eC2DynamicDeterministic ||
             condition == FormalProtocolCondition::eC3DynamicFrozenHash))
        {
            m_LastFormalFrameHasPrevious = true;
            m_LastFormalCondition = condition;
            m_LastFormalLocalFrameIndex = row.localFrameIndex;
            m_LastFormalGazeVelocity = row.gazeVelocity;
            m_LastFormalJaccard = row.jaccard;
            m_LastFormalPreviousSourceIds = state.previousIds;
            m_LastFormalCurrentSourceIds = currentIds;
            m_LastFormalChangedSourceIds.reserve(diff.added.size() + diff.removed.size());
            m_LastFormalChangedSourceIds.insert(m_LastFormalChangedSourceIds.end(),
                                                diff.added.begin(),
                                                diff.added.end());
            m_LastFormalChangedSourceIds.insert(m_LastFormalChangedSourceIds.end(),
                                                diff.removed.begin(),
                                                diff.removed.end());
            std::sort(m_LastFormalChangedSourceIds.begin(), m_LastFormalChangedSourceIds.end());
            m_LastFormalChangedSourceIds.erase(
                std::unique(m_LastFormalChangedSourceIds.begin(), m_LastFormalChangedSourceIds.end()),
                m_LastFormalChangedSourceIds.end());
        }

        state.previousIds = std::move(currentIds);
        state.previousGaze = row.gaze;
        state.previousValid = true;
    }

    void recordCachedSelectionOracleChurn(const GaussianBenchmarkSample& sample)
    {
        if (!m_Options.cachedSelectionOracle || !m_Options.logChurn)
            return;

        const auto& oracleStats = m_RenderService->gaussianSplatFrameStats();
        std::vector<uint32_t> currentIds = sample.selectedSourceIds;
        std::sort(currentIds.begin(), currentIds.end());
        currentIds.erase(std::unique(currentIds.begin(), currentIds.end()), currentIds.end());

        uint32_t symmetricDifference = 0u;
        double jaccard = 1.0;
        bool countStable = true;
        if (m_E0OraclePreviousValid)
        {
            const uint32_t intersection =
                selectedSetIntersectionCount(m_E0OraclePreviousSourceIds, currentIds);
            const uint32_t previousCount =
                static_cast<uint32_t>(m_E0OraclePreviousSourceIds.size());
            const uint32_t currentCount = static_cast<uint32_t>(currentIds.size());
            const uint32_t unionCount = previousCount + currentCount - intersection;
            symmetricDifference = previousCount + currentCount - 2u * intersection;
            jaccard = unionCount > 0u ?
                          static_cast<double>(intersection) / static_cast<double>(unionCount) :
                          1.0;
            countStable = previousCount == currentCount;
        }

        CachedSelectionOracleLogRow row {};
        row.sampleIndex = sample.sampleIndex;
        row.frameIndex = sample.frameIndex;
        row.membershipMode = std::string {membershipModeLabel(m_Options.membershipMode)};
        row.gaze = {sample.gazeX, sample.gazeY};
        row.candidateCount = oracleStats.cachedSelectionOracleCandidateCount;
        row.selectedCount = static_cast<uint32_t>(currentIds.size());
        row.selectedFraction = oracleStats.cachedSelectionOracleSelectedFraction;
        row.selectedSetHash = hashSelectedSourceIds(currentIds);
        row.selectedSourceIdRanges = selectedSourceIdRanges(currentIds);
        row.symmetricDifference = symmetricDifference;
        row.jaccard = jaccard;
        row.seed = m_Options.seed;
        row.cacheModeStatus = sample.sampleIndex == 0u ? "cached_frame0_candidates" : "live_membership_from_cache";
        row.cacheFrameIndexUsed = oracleStats.cachedSelectionOracleCacheFrameIndex;
        row.membershipRecomputedThisFrame =
            oracleStats.cachedSelectionOracleMembershipRecomputedThisFrame ? 1u : 0u;
        row.renderBufferMatchesLoggedSet =
            oracleStats.cachedSelectionOracleRenderBufferMatchesLoggedSet ? 1u : 0u;
        row.shaderSideFoveationDisabled =
            oracleStats.cachedSelectionOracleShaderFoveationDisabled ? 1u : 0u;
        row.thresholdFoveaDegrees = oracleStats.cachedSelectionOracleThresholdFoveaDegrees;
        row.thresholdMidDegrees = oracleStats.cachedSelectionOracleThresholdMidDegrees;
        row.falloffTransitionDegrees = oracleStats.cachedSelectionOracleFalloffTransitionDegrees;
        row.levelCenter = oracleStats.cachedSelectionOracleLevelCenter;
        row.levelMid = oracleStats.cachedSelectionOracleLevelMid;
        row.levelOuter = oracleStats.cachedSelectionOracleLevelOuter;
        row.eccentricityMin = oracleStats.cachedSelectionOracleEccentricityMin;
        row.eccentricityMax = oracleStats.cachedSelectionOracleEccentricityMax;
        row.keepProbabilityMin = oracleStats.cachedSelectionOracleKeepProbabilityMin;
        row.keepProbabilityMax = oracleStats.cachedSelectionOracleKeepProbabilityMax;
        const bool fixedGazeFailure =
            m_Options.fixedGaze && m_E0OraclePreviousValid &&
            (symmetricDifference != 0u || jaccard != 1.0 || !countStable);
        const bool renderMismatch = row.renderBufferMatchesLoggedSet == 0u ||
                                    row.membershipRecomputedThisFrame == 0u ||
                                    row.shaderSideFoveationDisabled == 0u;
        row.pass = (!fixedGazeFailure && !renderMismatch) ? 1u : 0u;
        m_E0OracleRows.push_back(std::move(row));

        if (m_E0OracleRows.back().pass == 0u)
        {
            m_E0OracleFailed = true;
            VULTRA_CLIENT_ERROR(
                "E0.5 cached selection oracle failed: mode={}, sample={}, frame={}, symmetric_diff={}, jaccard={}, selected_count={}, render_match={}, shader_foveation_disabled={}",
                membershipModeLabel(m_Options.membershipMode),
                sample.sampleIndex,
                sample.frameIndex,
                symmetricDifference,
                jaccard,
                currentIds.size(),
                m_E0OracleRows.back().renderBufferMatchesLoggedSet,
                m_E0OracleRows.back().shaderSideFoveationDisabled);
        }

        m_E0OraclePreviousSourceIds = std::move(currentIds);
        m_E0OraclePreviousValid = true;
    }

    std::vector<ImageFlickerBandSpec> imageFlickerBands() const
    {
        const float foveal = std::max(m_Options.flickerFovealRadiusDegrees, 0.0f);
        const float inf = std::numeric_limits<float>::infinity();
        std::vector<ImageFlickerBandSpec> bands;
        bands.push_back(ImageFlickerBandSpec {
            .label = "peripheral_all",
            .minDegrees = foveal,
            .maxDegrees = inf,
        });
        if (foveal < 20.0f)
        {
            bands.push_back(ImageFlickerBandSpec {
                .label = "ecc_foveal_to_20",
                .minDegrees = foveal,
                .maxDegrees = 20.0f,
            });
            bands.push_back(ImageFlickerBandSpec {
                .label = "ecc_20_to_40",
                .minDegrees = 20.0f,
                .maxDegrees = 40.0f,
            });
        }
        else if (foveal < 40.0f)
        {
            bands.push_back(ImageFlickerBandSpec {
                .label = "ecc_foveal_to_40",
                .minDegrees = foveal,
                .maxDegrees = 40.0f,
            });
        }
        bands.push_back(ImageFlickerBandSpec {
            .label = "ecc_40_plus",
            .minDegrees = std::max(foveal, 40.0f),
            .maxDegrees = inf,
        });
        return bands;
    }

    float pixelEccentricityDegrees(const uint32_t x,
                                   const uint32_t y,
                                   const uint32_t width,
                                   const uint32_t height,
                                   const glm::vec2 gaze) const
    {
        const float safeWidth = static_cast<float>(std::max(width, 1u));
        const float safeHeight = static_cast<float>(std::max(height, 1u));
        const float aspect = safeWidth / safeHeight;
        const float tanHalfY =
            std::tan(glm::radians(std::clamp(m_Options.flickerVerticalFovDegrees, 1.0f, 170.0f)) * 0.5f);
        const float tanHalfX = tanHalfY * aspect;
        const float uvx = (static_cast<float>(x) + 0.5f) / safeWidth;
        const float uvy = (static_cast<float>(y) + 0.5f) / safeHeight;
        const float sx = 2.0f * (uvx - gaze.x);
        const float sy = 2.0f * (uvy - gaze.y);
        const float ax = std::atan(sx * tanHalfX);
        const float ay = std::atan(sy * tanHalfY);
        return glm::degrees(std::sqrt(ax * ax + ay * ay));
    }

    static bool eccentricityInBand(const float eccentricity, const ImageFlickerBandSpec& band)
    {
        if (eccentricity <= band.minDegrees)
            return false;
        if (std::isfinite(band.maxDegrees) && eccentricity > band.maxDegrees)
            return false;
        return true;
    }

    std::optional<ImageFlickerReadback> readImageFlickerReadback()
    {
        if (!m_ImageFlickerOffscreenTarget)
        {
            if (!m_ImageFlickerReadbackFailedWarned)
            {
                VULTRA_CLIENT_WARN(
                    "Image flicker offscreen capture target is unavailable; flicker outputs will be incomplete");
                m_ImageFlickerReadbackFailedWarned = true;
            }
            return std::nullopt;
        }

        auto& backendService = engineCtx().services.require<IRenderBackendService>();
        auto& rd             = backendService.renderDevice();

        std::vector<uint8_t> rgba;
        uint32_t width = 0u;
        uint32_t height = 0u;
        rd.waitIdle();
        if (!rd.readTextureToRgba8(m_ImageFlickerOffscreenTarget, rgba, width, height))
        {
            if (!m_ImageFlickerReadbackFailedWarned)
            {
                VULTRA_CLIENT_WARN("Image flicker offscreen readback failed; flicker outputs will be incomplete");
                m_ImageFlickerReadbackFailedWarned = true;
            }
            return std::nullopt;
        }

        const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
        if (rgba.size() < pixelCount * 4u)
            return std::nullopt;

        ImageFlickerReadback readback {};
        readback.valid = true;
        readback.width = width;
        readback.height = height;
        readback.frameImageHash = hashBytes(rgba);
        readback.imageFormat = rhi::toString(m_ImageFlickerOffscreenTarget.getPixelFormat());
        readback.luminance.resize(pixelCount);
        for (size_t i = 0u; i < pixelCount; ++i)
        {
            const float r = static_cast<float>(rgba[i * 4u + 0u]) / 255.0f;
            const float g = static_cast<float>(rgba[i * 4u + 1u]) / 255.0f;
            const float b = static_cast<float>(rgba[i * 4u + 2u]) / 255.0f;
            readback.luminance[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        }

        return readback;
    }

    void recordImageFlickerFrame(const GaussianBenchmarkSample& sample)
    {
        m_LastImageFlickerReadback = {};
        if (!m_Options.logFlicker)
            return;
        const bool runtimeShaderAntiPop = sample.shaderAntiPopEnabled;
        const bool benchmarkFullRender =
            !m_Options.cachedSelectionOracle && !runtimeShaderAntiPop;
        if (!m_Options.cachedSelectionOracle && !runtimeShaderAntiPop && !benchmarkFullRender)
            return;
        const bool singleDatasetBenchmarkCamera = singleDatasetBenchmarkCameraForImageFlicker();
        if (m_Options.benchmarkCameraPath != BenchmarkCameraPath::eStatic && !singleDatasetBenchmarkCamera)
        {
            if (!m_ImageFlickerNonRegimeAWarned)
            {
                VULTRA_CLIENT_WARN(
                    "Image flicker logging requires a static camera or one dataset camera with view_frames=1; skipping moving camera run");
                m_ImageFlickerNonRegimeAWarned = true;
            }
            return;
        }

        std::string condition;
        GaussianSplatCachedSelectionMembershipMode membershipMode = m_Options.membershipMode;
        uint32_t localFrameIndex = sample.sampleIndex;
        glm::vec2 gaze {sample.gazeX, sample.gazeY};
        glm::vec2 gazeVelocity {0.0f, 0.0f};
        double gazeSpeed = 0.0;
        uint32_t selectedCount = sample.selectedIdCount;
        uint32_t symmetricDifference = 0u;
        double normalizedChurn = 0.0;
        double jaccard = 1.0;
        uint64_t selectedSetHash = 0u;
        uint32_t renderLogSetMatch = 0u;

        if (m_Options.formalE1E2E3)
        {
            if (m_FormalRows.empty() || m_FormalRows.back().sampleIndex != sample.sampleIndex)
                return;
            const auto& row = m_FormalRows.back();
            condition = std::string {formalConditionLabel(row.condition)};
            membershipMode = row.membershipMode;
            localFrameIndex = row.localFrameIndex;
            gaze = row.gaze;
            gazeVelocity = row.gazeVelocity;
            gazeSpeed = row.gazeSpeed;
            selectedCount = row.selectedCount;
            symmetricDifference = row.symmetricDifference;
            normalizedChurn = row.normalizedChurn;
            jaccard = row.jaccard;
            selectedSetHash = row.selectedSetHash;
            renderLogSetMatch = row.renderBufferMatchesLoggedSet;
        }
        else if (runtimeShaderAntiPop)
        {
            condition = std::string {m_Options.gpuAntiPopCliNaming ? "gpu_antipop_" : "shader_antipop_"} +
                        std::string {shaderAntiPopModeLabel(sample.shaderAntiPopMode)};
            membershipMode = GaussianSplatCachedSelectionMembershipMode::eDynamicFrozenHash;
            gaze = {sample.gazeX, sample.gazeY};
            selectedCount = sample.shaderAntiPopCandidateCount;
            symmetricDifference = 0u;
            normalizedChurn = 0.0;
            jaccard = 1.0;
            selectedSetHash = hashStablePrefixCandidateSet(sample.shaderAntiPopCandidateCount);
            renderLogSetMatch =
                (sample.shaderAntiPopAlphaMultiplierBased &&
                 sample.shaderAntiPopStableCandidateSet) ? 1u : 0u;
            auto& previous = m_ImageFlickerPreviousByCondition[condition];
            if (previous.valid)
                gazeVelocity = gaze - previous.gaze;
            gazeSpeed = glm::length(gazeVelocity);
        }
        else if (benchmarkFullRender)
        {
            condition = "full_render";
            membershipMode = GaussianSplatCachedSelectionMembershipMode::eDynamicDeterministic;
            gaze = {sample.gazeX, sample.gazeY};
            selectedCount =
                sample.drawnSplats != UINT32_MAX ? sample.drawnSplats :
                (sample.visibleSplats != UINT32_MAX ? sample.visibleSplats : sample.selectedIdCount);
            symmetricDifference = 0u;
            normalizedChurn = 0.0;
            jaccard = 1.0;
            selectedSetHash = 0u;
            renderLogSetMatch = 1u;
            auto& previous = m_ImageFlickerPreviousByCondition[condition];
            if (previous.valid)
                gazeVelocity = gaze - previous.gaze;
            gazeSpeed = glm::length(gazeVelocity);
        }
        else
        {
            if (m_E0OracleRows.empty() || m_E0OracleRows.back().sampleIndex != sample.sampleIndex)
                return;
            const auto& row = m_E0OracleRows.back();
            condition = row.membershipMode;
            membershipMode = m_Options.membershipMode;
            gaze = row.gaze;
            selectedCount = row.selectedCount;
            symmetricDifference = row.symmetricDifference;
            normalizedChurn = row.selectedCount > 0u ?
                                  static_cast<double>(row.symmetricDifference) /
                                      static_cast<double>(row.selectedCount) :
                                  0.0;
            jaccard = row.jaccard;
            selectedSetHash = row.selectedSetHash;
            renderLogSetMatch = row.renderBufferMatchesLoggedSet;
            auto& previous = m_ImageFlickerPreviousByCondition[condition];
            if (previous.valid)
                gazeVelocity = gaze - previous.gaze;
            gazeSpeed = glm::length(gazeVelocity);
        }

        const auto readback = readImageFlickerReadback();
        if (!readback)
            return;

        m_LastImageFlickerReadback = *readback;
        const uint32_t width = readback->width;
        const uint32_t height = readback->height;
        const size_t pixelCount = readback->luminance.size();
        const auto& luminance = readback->luminance;
        const uint64_t frameImageHash = readback->frameImageHash;
        const std::string captureImageFormat = readback->imageFormat;

        auto& previous = m_ImageFlickerPreviousByCondition[condition];
        const bool hasPrevious =
            previous.valid && previous.width == width && previous.height == height &&
            previous.luminance.size() == luminance.size();

        std::vector<float> delta;
        std::vector<float> hpDelta;
        RuntimeWeightedSpectrumSummary weightedSpectrum {};
        if (hasPrevious)
        {
            delta.resize(pixelCount);
            hpDelta.resize(pixelCount);
            for (size_t i = 0u; i < pixelCount; ++i)
                delta[i] = luminance[i] - previous.luminance[i];

            for (uint32_t y = 0u; y < height; ++y)
            {
                for (uint32_t x = 0u; x < width; ++x)
                {
                    double sum = 0.0;
                    uint32_t count = 0u;
                    const int32_t ix = static_cast<int32_t>(x);
                    const int32_t iy = static_cast<int32_t>(y);
                    for (int32_t oy = -1; oy <= 1; ++oy)
                    {
                        const int32_t ny = iy + oy;
                        if (ny < 0 || ny >= static_cast<int32_t>(height))
                            continue;
                        for (int32_t ox = -1; ox <= 1; ++ox)
                        {
                            const int32_t nx = ix + ox;
                            if (nx < 0 || nx >= static_cast<int32_t>(width))
                                continue;
                            sum += delta[static_cast<size_t>(ny) * width + static_cast<size_t>(nx)];
                            ++count;
                        }
                    }
                    const size_t index = static_cast<size_t>(y) * width + x;
                    const float blur = count > 0u ? static_cast<float>(sum / static_cast<double>(count)) : 0.0f;
                    hpDelta[index] = delta[index] - blur;
                }
            }

            std::vector<float> absDelta(pixelCount, 0.0f);
            for (size_t i = 0u; i < pixelCount; ++i)
                absDelta[i] = std::abs(delta[i]);
            weightedSpectrum = runtimeWeightedSpectrumSummary(absDelta, width, height);
        }

        for (const auto& band : imageFlickerBands())
        {
            uint64_t bandPixelCount = 0u;
            double rawEnergy = 0.0;
            double hpEnergy = 0.0;
            for (uint32_t y = 0u; y < height; ++y)
            {
                for (uint32_t x = 0u; x < width; ++x)
                {
                    const float eccentricity = pixelEccentricityDegrees(x, y, width, height, gaze);
                    if (!eccentricityInBand(eccentricity, band))
                        continue;
                    ++bandPixelCount;
                    if (!hasPrevious)
                        continue;
                    const size_t index = static_cast<size_t>(y) * width + x;
                    rawEnergy += static_cast<double>(delta[index]) * static_cast<double>(delta[index]);
                    hpEnergy += static_cast<double>(hpDelta[index]) * static_cast<double>(hpDelta[index]);
                }
            }

            ImageFlickerFrameRow flicker {};
            flicker.sampleIndex = sample.sampleIndex;
            flicker.localFrameIndex = localFrameIndex;
            flicker.frameIndex = sample.frameIndex;
            flicker.condition = condition;
            flicker.membershipMode = membershipMode;
            flicker.seed = m_Options.seed;
            flicker.gaze = gaze;
            flicker.gazeVelocity = gazeVelocity;
            flicker.gazeSpeed = gazeSpeed;
            flicker.selectedCount = selectedCount;
            flicker.symmetricDifference = symmetricDifference;
            flicker.normalizedChurn = normalizedChurn;
            flicker.jaccard = jaccard;
            flicker.width = width;
            flicker.height = height;
            flicker.fovealRadiusDegrees = m_Options.flickerFovealRadiusDegrees;
            flicker.verticalFovDegrees = m_Options.flickerVerticalFovDegrees;
            flicker.bandLabel = band.label;
            flicker.bandMinDegrees = band.minDegrees;
            flicker.bandMaxDegrees = std::isfinite(band.maxDegrees) ? band.maxDegrees : -1.0f;
            flicker.pixelCount = bandPixelCount;
            flicker.hasPreviousFrame = hasPrevious ? 1u : 0u;
            flicker.rawDeltaEnergy = rawEnergy;
            flicker.rawDeltaMean =
                bandPixelCount > 0u ? rawEnergy / static_cast<double>(bandPixelCount) : 0.0;
            flicker.hpDeltaEnergy = hpEnergy;
            flicker.hpDeltaMean =
                bandPixelCount > 0u ? hpEnergy / static_cast<double>(bandPixelCount) : 0.0;
            flicker.hpAvailable = hasPrevious ? 1u : 0u;
            flicker.weightedLowFrequencyFraction = weightedSpectrum.lowFrequencyFraction;
            flicker.weightedMidFrequencyFraction = weightedSpectrum.midFrequencyFraction;
            flicker.weightedHighFrequencyFraction = weightedSpectrum.highFrequencyFraction;
            flicker.weightedSpectrumTotalPower = weightedSpectrum.totalPower;
            flicker.weightedSpectrumAvailable = weightedSpectrum.available ? 1u : 0u;
            flicker.colorSpace = "offscreen_srgb_proxy";
            flicker.captureTargetType = "offscreen_scene_color";
            flicker.imageFormat = captureImageFormat;
            flicker.sampleCount = 1u;
            flicker.frameImageHash = frameImageHash;
            flicker.preUiCapture = 1u;
            flicker.postTonemapCapture = 0u;
            flicker.temporalEffectsDisabled = 1u;
            flicker.overlaysDisabled = 1u;
            flicker.readbackSynchronized = 1u;
            flicker.selectedSetHash = selectedSetHash;
            flicker.renderLogSetMatch = renderLogSetMatch;
            flicker.runLabel = m_Options.formalExploratoryThresholdUncommitted ?
                                   "exploratory_threshold_uncommitted" :
                                   "formal_threshold_committed";
            m_ImageFlickerRows.push_back(std::move(flicker));
        }

        previous.valid = true;
        previous.width = width;
        previous.height = height;
        previous.gaze = gaze;
        previous.luminance = luminance;
    }

    std::vector<float> highPassFromDelta(const std::vector<float>& delta,
                                         const uint32_t width,
                                         const uint32_t height) const
    {
        std::vector<float> hpDelta(delta.size(), 0.0f);
        if (delta.empty())
            return hpDelta;

        for (uint32_t y = 0u; y < height; ++y)
        {
            for (uint32_t x = 0u; x < width; ++x)
            {
                double sum = 0.0;
                uint32_t count = 0u;
                const int32_t ix = static_cast<int32_t>(x);
                const int32_t iy = static_cast<int32_t>(y);
                for (int32_t oy = -1; oy <= 1; ++oy)
                {
                    const int32_t ny = iy + oy;
                    if (ny < 0 || ny >= static_cast<int32_t>(height))
                        continue;
                    for (int32_t ox = -1; ox <= 1; ++ox)
                    {
                        const int32_t nx = ix + ox;
                        if (nx < 0 || nx >= static_cast<int32_t>(width))
                            continue;
                        sum += delta[static_cast<size_t>(ny) * width + static_cast<size_t>(nx)];
                        ++count;
                    }
                }
                const size_t index = static_cast<size_t>(y) * width + x;
                const float blur = count > 0u ? static_cast<float>(sum / static_cast<double>(count)) : 0.0f;
                hpDelta[index] = delta[index] - blur;
            }
        }
        return hpDelta;
    }

    std::vector<ImageFlickerBandSpec> e3PixelDeltaBands() const
    {
        auto bands = imageFlickerBands();
        bands.insert(bands.begin(),
                     ImageFlickerBandSpec {
                         .label = "full_frame",
                         .minDegrees = -1.0f,
                         .maxDegrees = std::numeric_limits<float>::infinity(),
                     });
        return bands;
    }

    static bool e3BandContainsPixel(const float eccentricity, const ImageFlickerBandSpec& band)
    {
        if (band.label == "full_frame")
            return true;
        return eccentricityInBand(eccentricity, band);
    }

    std::string e3CalibrationRunLabel() const
    {
        return "exploratory_calibration_seen|not_formal_validation|thresholds_uncommitted";
    }

    static bool isPowerOfTwo(const uint32_t value)
    {
        return value != 0u && (value & (value - 1u)) == 0u;
    }

    static void fft1D(std::vector<std::complex<double>>& values, const bool inverse)
    {
        const size_t n = values.size();
        for (size_t i = 1u, j = 0u; i < n; ++i)
        {
            size_t bit = n >> 1u;
            for (; (j & bit) != 0u; bit >>= 1u)
                j ^= bit;
            j ^= bit;
            if (i < j)
                std::swap(values[i], values[j]);
        }

        for (size_t len = 2u; len <= n; len <<= 1u)
        {
            const double angle =
                2.0 * 3.14159265358979323846 / static_cast<double>(len) * (inverse ? -1.0 : 1.0);
            const std::complex<double> wLen {std::cos(angle), std::sin(angle)};
            for (size_t i = 0u; i < n; i += len)
            {
                std::complex<double> w {1.0, 0.0};
                for (size_t j = 0u; j < len / 2u; ++j)
                {
                    const auto u = values[i + j];
                    const auto v = values[i + j + len / 2u] * w;
                    values[i + j] = u + v;
                    values[i + j + len / 2u] = u - v;
                    w *= wLen;
                }
            }
        }

        if (inverse && n > 0u)
        {
            for (auto& value : values)
                value /= static_cast<double>(n);
        }
    }

    static std::vector<double> fft2DPowerSpectrum(std::vector<double> field,
                                                  const uint32_t width,
                                                  const uint32_t height)
    {
        if (width == 0u || height == 0u || !isPowerOfTwo(width) || !isPowerOfTwo(height))
            return {};

        std::vector<std::complex<double>> data(field.size());
        for (size_t i = 0u; i < field.size(); ++i)
            data[i] = {field[i], 0.0};

        std::vector<std::complex<double>> scratch(std::max(width, height));
        for (uint32_t y = 0u; y < height; ++y)
        {
            for (uint32_t x = 0u; x < width; ++x)
                scratch[x] = data[static_cast<size_t>(y) * width + x];
            scratch.resize(width);
            fft1D(scratch, false);
            for (uint32_t x = 0u; x < width; ++x)
                data[static_cast<size_t>(y) * width + x] = scratch[x];
            scratch.resize(std::max(width, height));
        }

        for (uint32_t x = 0u; x < width; ++x)
        {
            for (uint32_t y = 0u; y < height; ++y)
                scratch[y] = data[static_cast<size_t>(y) * width + x];
            scratch.resize(height);
            fft1D(scratch, false);
            for (uint32_t y = 0u; y < height; ++y)
                data[static_cast<size_t>(y) * width + x] = scratch[y];
            scratch.resize(std::max(width, height));
        }

        std::vector<double> power(data.size(), 0.0);
        const double scale = static_cast<double>(std::max<size_t>(data.size(), 1u));
        for (size_t i = 0u; i < data.size(); ++i)
            power[i] = std::norm(data[i]) / scale;
        return power;
    }

    static std::vector<double> downsampleFieldAverage(const std::vector<float>& values,
                                                      const uint32_t srcWidth,
                                                      const uint32_t srcHeight,
                                                      const uint32_t gridSize,
                                                      const bool binary)
    {
        std::vector<double> out(static_cast<size_t>(gridSize) * gridSize, 0.0);
        std::vector<uint32_t> counts(out.size(), 0u);
        if (values.empty() || srcWidth == 0u || srcHeight == 0u || gridSize == 0u)
            return out;

        constexpr float kBinaryDeltaEpsilon = 1.0e-6f;
        for (uint32_t y = 0u; y < srcHeight; ++y)
        {
            const uint32_t gy = std::min<uint32_t>(
                gridSize - 1u,
                static_cast<uint32_t>((static_cast<uint64_t>(y) * gridSize) / srcHeight));
            for (uint32_t x = 0u; x < srcWidth; ++x)
            {
                const uint32_t gx = std::min<uint32_t>(
                    gridSize - 1u,
                    static_cast<uint32_t>((static_cast<uint64_t>(x) * gridSize) / srcWidth));
                const size_t srcIndex = static_cast<size_t>(y) * srcWidth + x;
                const size_t dstIndex = static_cast<size_t>(gy) * gridSize + gx;
                if (binary)
                    out[dstIndex] = std::max(out[dstIndex], values[srcIndex] > kBinaryDeltaEpsilon ? 1.0 : 0.0);
                else
                    out[dstIndex] += static_cast<double>(values[srcIndex]);
                ++counts[dstIndex];
            }
        }

        if (!binary)
        {
            for (size_t i = 0u; i < out.size(); ++i)
            {
                if (counts[i] > 0u)
                    out[i] /= static_cast<double>(counts[i]);
            }
        }
        return out;
    }

    struct RuntimeWeightedSpectrumSummary
    {
        bool available {false};
        double lowFrequencyFraction {0.0};
        double midFrequencyFraction {0.0};
        double highFrequencyFraction {0.0};
        double totalPower {0.0};
    };

    RuntimeWeightedSpectrumSummary runtimeWeightedSpectrumSummary(const std::vector<float>& field,
                                                                  const uint32_t width,
                                                                  const uint32_t height) const
    {
        RuntimeWeightedSpectrumSummary summary {};
        const uint32_t grid = m_Options.e6SpectrumGridSize;
        if (!isPowerOfTwo(grid) || grid == 0u || field.empty() || width == 0u || height == 0u)
            return summary;

        const auto downsampled = downsampleFieldAverage(field, width, height, grid, false);
        const auto power = fft2DPowerSpectrum(downsampled, grid, grid);
        if (power.empty())
            return summary;

        double low = 0.0;
        double mid = 0.0;
        double high = 0.0;
        double total = 0.0;
        const double maxRadius = std::sqrt(2.0) * static_cast<double>(grid) * 0.5;
        for (uint32_t y = 0u; y < grid; ++y)
        {
            const int32_t fy = y <= grid / 2u ? static_cast<int32_t>(y) :
                                                  static_cast<int32_t>(y) - static_cast<int32_t>(grid);
            for (uint32_t x = 0u; x < grid; ++x)
            {
                const int32_t fx = x <= grid / 2u ? static_cast<int32_t>(x) :
                                                      static_cast<int32_t>(x) - static_cast<int32_t>(grid);
                const double radius = std::sqrt(static_cast<double>(fx * fx + fy * fy));
                const double normalizedRadius = maxRadius > 0.0 ? radius / maxRadius : 0.0;
                const double p = power[static_cast<size_t>(y) * grid + x];
                total += p;
                if (normalizedRadius <= 0.15)
                    low += p;
                else if (normalizedRadius <= 0.35)
                    mid += p;
                else
                    high += p;
            }
        }

        summary.available = true;
        summary.totalPower = total;
        if (total > 0.0)
        {
            summary.lowFrequencyFraction = low / total;
            summary.midFrequencyFraction = mid / total;
            summary.highFrequencyFraction = high / total;
        }
        return summary;
    }

    void appendE6SpectrumRows(const E3HonestPendingCounterfactual& pending,
                              const uint64_t forcedFrameIndex,
                              const std::string_view fieldType,
                              const std::vector<float>& field,
                              const uint32_t width,
                              const uint32_t height,
                              const bool binary)
    {
        const uint32_t grid = m_Options.e6SpectrumGridSize;
        if (!isPowerOfTwo(grid) || grid == 0u)
            return;

        const auto downsampled = downsampleFieldAverage(field, width, height, grid, binary);
        const auto power = fft2DPowerSpectrum(downsampled, grid, grid);
        if (power.empty())
            return;

        const uint32_t radiusBinCount = grid / 2u + 1u;
        std::vector<double> radialPower(radiusBinCount, 0.0);
        std::vector<uint32_t> radialCount(radiusBinCount, 0u);
        double low = 0.0;
        double mid = 0.0;
        double high = 0.0;
        double total = 0.0;
        const double maxRadius = std::sqrt(2.0) * static_cast<double>(grid) * 0.5;

        for (uint32_t y = 0u; y < grid; ++y)
        {
            const int32_t fy = y <= grid / 2u ? static_cast<int32_t>(y) :
                                                  static_cast<int32_t>(y) - static_cast<int32_t>(grid);
            for (uint32_t x = 0u; x < grid; ++x)
            {
                const int32_t fx = x <= grid / 2u ? static_cast<int32_t>(x) :
                                                      static_cast<int32_t>(x) - static_cast<int32_t>(grid);
                const double radius = std::sqrt(static_cast<double>(fx * fx + fy * fy));
                const double normalizedRadius = maxRadius > 0.0 ? radius / maxRadius : 0.0;
                const uint32_t bin = std::min<uint32_t>(
                    radiusBinCount - 1u,
                    static_cast<uint32_t>(std::floor(normalizedRadius * static_cast<double>(radiusBinCount - 1u))));
                const double p = power[static_cast<size_t>(y) * grid + x];
                radialPower[bin] += p;
                ++radialCount[bin];
                total += p;
                if (normalizedRadius <= 0.15)
                    low += p;
                else if (normalizedRadius <= 0.35)
                    mid += p;
                else
                    high += p;
            }
        }

        for (uint32_t bin = 0u; bin < radiusBinCount; ++bin)
        {
            E6SpectrumBinRow row {};
            row.sampleIndex = pending.sampleIndex;
            row.localFrameIndex = pending.localFrameIndex;
            row.actualFrameIndex = pending.actualFrameIndex;
            row.forcedFrameIndex = forcedFrameIndex;
            row.condition = pending.condition;
            row.fieldType = std::string {fieldType};
            row.radiusBin = bin;
            row.normalizedRadius = radiusBinCount > 1u ?
                                       static_cast<double>(bin) / static_cast<double>(radiusBinCount - 1u) :
                                       0.0;
            row.power = radialPower[bin];
            row.binSampleCount = radialCount[bin];
            row.runLabel = e3CalibrationRunLabel();
            row.notes = binary ? "binary_flip_field_delta_nonzero_proxy_exact_gaussian_touch_unavailable" :
                                 "weighted_flip_field_abs_freeze_membership_delta_luminance";
            if (binary)
                m_E6BinarySpectrumRows.push_back(std::move(row));
            else
                m_E6WeightedSpectrumRows.push_back(std::move(row));
        }

        E6FrequencyBandRow band {};
        band.sampleIndex = pending.sampleIndex;
        band.localFrameIndex = pending.localFrameIndex;
        band.actualFrameIndex = pending.actualFrameIndex;
        band.forcedFrameIndex = forcedFrameIndex;
        band.condition = pending.condition;
        band.fieldType = std::string {fieldType};
        band.selectedCount = pending.selectedCount;
        band.previousSelectedCount = pending.previousSelectedCount;
        band.symmetricDifference = pending.symmetricDifference;
        band.normalizedChurn = pending.normalizedChurn;
        band.lowFrequencyFraction = total > 0.0 ? low / total : 0.0;
        band.midFrequencyFraction = total > 0.0 ? mid / total : 0.0;
        band.highFrequencyFraction = total > 0.0 ? high / total : 0.0;
        band.totalPower = total;
        band.runLabel = e3CalibrationRunLabel();
        band.notes = binary ? "binary_flip_field_delta_nonzero_proxy_exact_gaussian_touch_unavailable" :
                              "weighted_flip_field_abs_freeze_membership_delta_luminance";
        m_E6FrequencyBandRows.push_back(std::move(band));
    }

    void maybeScheduleE3HonestCounterfactual(const GaussianBenchmarkSample& sample)
    {
        if (!m_Options.e3HonestE6 || !m_RenderService || !m_LastFormalFrameHasPrevious ||
            !m_LastImageFlickerReadback.valid)
        {
            return;
        }
        if (m_LastFormalCondition != FormalProtocolCondition::eC2DynamicDeterministic &&
            m_LastFormalCondition != FormalProtocolCondition::eC3DynamicFrozenHash)
        {
            return;
        }
        if (m_LastFormalPreviousSourceIds.empty())
            return;

        E3HonestPendingCounterfactual pending {};
        pending.valid = true;
        pending.sampleIndex = sample.sampleIndex;
        pending.localFrameIndex = m_LastFormalLocalFrameIndex;
        pending.actualFrameIndex = sample.frameIndex;
        pending.condition = m_LastFormalCondition;
        pending.membershipMode = formalConditionMembershipMode(m_LastFormalCondition);
        pending.gaze = {sample.gazeX, sample.gazeY};
        pending.gazeVelocity = m_LastFormalGazeVelocity;
        pending.gazeSpeed = glm::length(m_LastFormalGazeVelocity);
        pending.selectedCount = static_cast<uint32_t>(m_LastFormalCurrentSourceIds.size());
        pending.previousSelectedCount = static_cast<uint32_t>(m_LastFormalPreviousSourceIds.size());
        pending.symmetricDifference = static_cast<uint32_t>(m_LastFormalChangedSourceIds.size());
        pending.normalizedChurn = pending.selectedCount > 0u ?
                                      static_cast<double>(pending.symmetricDifference) /
                                          static_cast<double>(pending.selectedCount) :
                                      0.0;
        pending.jaccard = m_LastFormalJaccard;
        pending.selectedSetHash = hashSelectedSourceIds(m_LastFormalCurrentSourceIds);
        pending.previousSelectedSetHash = hashSelectedSourceIds(m_LastFormalPreviousSourceIds);
        pending.previousSourceIds = m_LastFormalPreviousSourceIds;
        pending.currentSourceIds = m_LastFormalCurrentSourceIds;
        pending.changedSourceIds = m_LastFormalChangedSourceIds;
        pending.actualReadback = m_LastImageFlickerReadback;

        auto& settings = m_RenderService->gaussianSplatSettings();
        settings.cachedSelectionOracleForcedSourceIdsEnabled = true;
        settings.cachedSelectionOracleForcedSourceIds = pending.previousSourceIds;
        settings.cachedSelectionMembershipMode = pending.membershipMode;
        settings.foveatedGaze = pending.gaze;
        m_E3CounterfactualPending = std::move(pending);
    }

    void clearE3HonestForcedSelection()
    {
        if (!m_RenderService)
            return;
        auto& settings = m_RenderService->gaussianSplatSettings();
        settings.cachedSelectionOracleForcedSourceIdsEnabled = false;
        settings.cachedSelectionOracleForcedSourceIds.clear();
    }

    void processE3HonestCounterfactualFrame(const uint64_t forcedFrameIndex)
    {
        if (!m_E3CounterfactualPending.valid)
            return;

        auto forcedReadback = readImageFlickerReadback();
        const auto pending = std::move(m_E3CounterfactualPending);
        const auto oracleStats = m_RenderService->gaussianSplatFrameStats();
        m_E3CounterfactualPending = {};
        clearE3HonestForcedSelection();

        if (!forcedReadback || !forcedReadback->valid || !pending.actualReadback.valid)
            return;
        if (forcedReadback->width != pending.actualReadback.width ||
            forcedReadback->height != pending.actualReadback.height ||
            forcedReadback->luminance.size() != pending.actualReadback.luminance.size())
        {
            VULTRA_CLIENT_WARN("E3-honest counterfactual readback dimensions changed; skipping frame {}",
                               pending.localFrameIndex);
            return;
        }

        std::vector<uint32_t> forcedIds = oracleStats.selectedSourceIds;
        std::sort(forcedIds.begin(), forcedIds.end());
        forcedIds.erase(std::unique(forcedIds.begin(), forcedIds.end()), forcedIds.end());

        const uint32_t width = forcedReadback->width;
        const uint32_t height = forcedReadback->height;
        const size_t pixelCount = forcedReadback->luminance.size();
        std::vector<float> signedDelta(pixelCount, 0.0f);
        std::vector<float> absDelta(pixelCount, 0.0f);
        uint64_t totalNonzeroPixels = 0u;
        constexpr float kDeltaEpsilon = 1.0e-6f;
        for (size_t i = 0u; i < pixelCount; ++i)
        {
            const float d = pending.actualReadback.luminance[i] - forcedReadback->luminance[i];
            signedDelta[i] = d;
            absDelta[i] = std::abs(d);
            totalNonzeroPixels += absDelta[i] > kDeltaEpsilon ? 1u : 0u;
        }
        const auto hpDelta = highPassFromDelta(signedDelta, width, height);

        for (const auto& band : e3PixelDeltaBands())
        {
            uint64_t bandPixelCount = 0u;
            uint64_t bandNonzeroPixels = 0u;
            double rawEnergy = 0.0;
            double hpEnergy = 0.0;
            for (uint32_t y = 0u; y < height; ++y)
            {
                for (uint32_t x = 0u; x < width; ++x)
                {
                    const float eccentricity = pixelEccentricityDegrees(x, y, width, height, pending.gaze);
                    if (!e3BandContainsPixel(eccentricity, band))
                        continue;
                    const size_t index = static_cast<size_t>(y) * width + x;
                    ++bandPixelCount;
                    bandNonzeroPixels += absDelta[index] > kDeltaEpsilon ? 1u : 0u;
                    rawEnergy += static_cast<double>(signedDelta[index]) *
                                 static_cast<double>(signedDelta[index]);
                    hpEnergy += static_cast<double>(hpDelta[index]) *
                                static_cast<double>(hpDelta[index]);
                }
            }

            E3PixelDeltaFrameRow row {};
            row.sampleIndex = pending.sampleIndex;
            row.localFrameIndex = pending.localFrameIndex;
            row.actualFrameIndex = pending.actualFrameIndex;
            row.forcedFrameIndex = forcedFrameIndex;
            row.condition = pending.condition;
            row.membershipMode = pending.membershipMode;
            row.seed = m_Options.seed;
            row.gaze = pending.gaze;
            row.gazeVelocity = pending.gazeVelocity;
            row.gazeSpeed = pending.gazeSpeed;
            row.selectedCount = pending.selectedCount;
            row.previousSelectedCount = pending.previousSelectedCount;
            row.forcedSelectedCount = static_cast<uint32_t>(forcedIds.size());
            row.symmetricDifference = pending.symmetricDifference;
            row.normalizedChurn = pending.normalizedChurn;
            row.jaccard = pending.jaccard;
            row.selectedSetHash = pending.selectedSetHash;
            row.previousSelectedSetHash = pending.previousSelectedSetHash;
            row.forcedSelectedSetHash = hashSelectedSourceIds(forcedIds);
            row.width = width;
            row.height = height;
            row.bandLabel = band.label;
            row.bandMinDegrees = band.minDegrees;
            row.bandMaxDegrees = std::isfinite(band.maxDegrees) ? band.maxDegrees : -1.0f;
            row.pixelCount = bandPixelCount;
            row.nonzeroPixelCount = bandNonzeroPixels;
            row.rawPixelDeltaEnergy = rawEnergy;
            row.rawPixelDeltaMean =
                bandPixelCount > 0u ? rawEnergy / static_cast<double>(bandPixelCount) : 0.0;
            row.hpPixelDeltaEnergy = hpEnergy;
            row.hpPixelDeltaMean =
                bandPixelCount > 0u ? hpEnergy / static_cast<double>(bandPixelCount) : 0.0;
            row.actualFrameImageHash = pending.actualReadback.frameImageHash;
            row.forcedFrameImageHash = forcedReadback->frameImageHash;
            row.forcedIdsRequestedCount = oracleStats.cachedSelectionOracleForcedSourceIdsRequestedCount;
            row.forcedIdsMatchedCount = oracleStats.cachedSelectionOracleForcedSourceIdsMatchedCount;
            row.forcedIdsAllFound =
                oracleStats.cachedSelectionOracleForcedSourceIdsAllFound ? 1u : 0u;
            row.renderLogSetMatch =
                oracleStats.cachedSelectionOracleRenderBufferMatchesLoggedSet ? 1u : 0u;
            row.shaderSideFoveationDisabled =
                oracleStats.cachedSelectionOracleShaderFoveationDisabled ? 1u : 0u;
            row.deltaNonzero = totalNonzeroPixels > 0u ? 1u : 0u;
            row.colorSpace = "offscreen_srgb_proxy";
            row.captureTargetType = "offscreen_scene_color";
            row.imageFormat = forcedReadback->imageFormat;
            row.runLabel = e3CalibrationRunLabel();
            row.notes = "freeze_membership_pixel_delta;event_level_pixel_attribution_unavailable";
            m_E3PixelDeltaRows.push_back(std::move(row));
        }

        appendE6SpectrumRows(pending,
                             forcedFrameIndex,
                             "binary_delta_nonzero_proxy",
                             absDelta,
                             width,
                             height,
                             true);
        appendE6SpectrumRows(pending,
                             forcedFrameIndex,
                             "weighted_abs_delta_luminance",
                             absDelta,
                             width,
                             height,
                             false);
    }

    std::filesystem::path cachedSelectionOracleCsvPath() const
    {
        const std::string filename =
            "e05_live_membership_cached_oracle_" +
            std::string {membershipModeLabel(m_Options.membershipMode)} + ".csv";
        return m_Options.outputDir / filename;
    }

    void writeCachedSelectionOracleCsv()
    {
        if (!m_Options.cachedSelectionOracle || !m_Options.logChurn || m_E0OracleRows.empty())
            return;

        const auto path = cachedSelectionOracleCsvPath();
        if (path.has_parent_path())
        {
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
            if (ec)
            {
                VULTRA_CLIENT_WARN("Failed to create E0 oracle directory {}: {}",
                                   path.parent_path().string(),
                                   ec.message());
                return;
            }
        }

        std::ofstream out {path};
        if (!out)
        {
            VULTRA_CLIENT_WARN("Failed to write E0 oracle CSV: {}", path.string());
            return;
        }

        out << "sample_index,frame_index,membership_mode,gaze_x,gaze_y,total_candidate_count,"
               "selected_count,selected_fraction,selected_set_hash,selected_source_id_ranges,"
               "symmetric_difference,jaccard,seed,cache_mode_status,cache_frame_index_used,"
               "membership_recomputed_this_frame,render_buffer_matches_logged_set,"
               "shader_side_foveation_disabled,threshold_fovea_degrees,threshold_mid_degrees,"
               "falloff_transition_degrees,level_center,level_mid,level_outer,"
               "eccentricity_min,eccentricity_max,keep_probability_min,keep_probability_max,pass\n";
        out << std::setprecision(10);
        for (const auto& row : m_E0OracleRows)
        {
            std::ostringstream hashText;
            hashText << "0x" << std::hex << std::setw(16) << std::setfill('0') << row.selectedSetHash;
            out << row.sampleIndex
                << ',' << row.frameIndex
                << ',' << row.membershipMode
                << ',' << row.gaze.x
                << ',' << row.gaze.y
                << ',' << row.candidateCount
                << ',' << row.selectedCount
                << ',' << row.selectedFraction
                << ',' << hashText.str()
                << ",\"" << row.selectedSourceIdRanges << '"'
                << ',' << row.symmetricDifference
                << ',' << row.jaccard
                << ',' << row.seed
                << ',' << row.cacheModeStatus
                << ',' << row.cacheFrameIndexUsed
                << ',' << row.membershipRecomputedThisFrame
                << ',' << row.renderBufferMatchesLoggedSet
                << ',' << row.shaderSideFoveationDisabled
                << ',' << row.thresholdFoveaDegrees
                << ',' << row.thresholdMidDegrees
                << ',' << row.falloffTransitionDegrees
                << ',' << row.levelCenter
                << ',' << row.levelMid
                << ',' << row.levelOuter
                << ',' << row.eccentricityMin
                << ',' << row.eccentricityMax
                << ',' << row.keepProbabilityMin
                << ',' << row.keepProbabilityMax
                << ',' << row.pass
                << '\n';
        }

        VULTRA_CLIENT_INFO("E0.5 cached selection oracle wrote {} rows to {}",
                           m_E0OracleRows.size(),
                           path.string());
    }

    bool ensureOutputParentDirectory(const std::filesystem::path& path, const std::string_view label) const
    {
        if (!path.has_parent_path())
            return true;

        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec)
        {
            VULTRA_CLIENT_WARN("Failed to create {} directory {}: {}",
                               label,
                               path.parent_path().string(),
                               ec.message());
            return false;
        }
        return true;
    }

    void writeImageFlickerPerFrameCsv()
    {
        if (m_ImageFlickerRows.empty())
            return;

        const auto path = m_Options.outputDir / "e1_flicker_per_frame.csv";
        if (!ensureOutputParentDirectory(path, "image flicker"))
            return;
        std::ofstream out {path};
        if (!out)
        {
            VULTRA_CLIENT_WARN("Failed to write image flicker per-frame CSV: {}", path.string());
            return;
        }
        out << std::setprecision(10);
        out << "sample_index,local_frame_index,frame_index,condition,membership_mode,seed,"
               "gaze_x,gaze_y,gaze_velocity_x,gaze_velocity_y,gaze_speed,selected_count,"
               "symmetric_difference,normalized_churn,jaccard,width,height,"
               "flicker_foveal_radius_deg,vertical_fov_deg,band_label,band_min_deg,"
               "band_max_deg,pixel_count,has_previous_frame,raw_delta_energy,raw_delta_mean,"
               "hp_delta_energy,hp_delta_mean,hp_available,weighted_low_frequency_fraction,"
               "weighted_mid_frequency_fraction,weighted_high_frequency_fraction,"
               "weighted_spectrum_total_power,weighted_spectrum_available,color_space,run_label,"
               "capture_target_type,image_format,sample_count,frame_image_hash,pre_ui_capture,"
               "post_tonemap_capture,temporal_effects_disabled,overlays_disabled,"
               "readback_synchronized,selected_set_hash,render_log_set_match\n";
        for (const auto& row : m_ImageFlickerRows)
        {
            out << row.sampleIndex
                << ',' << row.localFrameIndex
                << ',' << row.frameIndex
                << ',' << row.condition
                << ',' << membershipModeLabel(row.membershipMode)
                << ',' << row.seed
                << ',' << row.gaze.x
                << ',' << row.gaze.y
                << ',' << row.gazeVelocity.x
                << ',' << row.gazeVelocity.y
                << ',' << row.gazeSpeed
                << ',' << row.selectedCount
                << ',' << row.symmetricDifference
                << ',' << row.normalizedChurn
                << ',' << row.jaccard
                << ',' << row.width
                << ',' << row.height
                << ',' << row.fovealRadiusDegrees
                << ',' << row.verticalFovDegrees
                << ',' << row.bandLabel
                << ',' << row.bandMinDegrees
                << ',' << row.bandMaxDegrees
                << ',' << row.pixelCount
                << ',' << row.hasPreviousFrame
                << ',' << row.rawDeltaEnergy
                << ',' << row.rawDeltaMean
                << ',' << row.hpDeltaEnergy
                << ',' << row.hpDeltaMean
                << ',' << row.hpAvailable
                << ',' << row.weightedLowFrequencyFraction
                << ',' << row.weightedMidFrequencyFraction
                << ',' << row.weightedHighFrequencyFraction
                << ',' << row.weightedSpectrumTotalPower
                << ',' << row.weightedSpectrumAvailable
                << ',' << row.colorSpace
                << ',' << row.runLabel
                << ',' << row.captureTargetType
                << ',' << row.imageFormat
                << ',' << row.sampleCount
                << ',' << hexHashText(row.frameImageHash)
                << ',' << row.preUiCapture
                << ',' << row.postTonemapCapture
                << ',' << row.temporalEffectsDisabled
                << ',' << row.overlaysDisabled
                << ',' << row.readbackSynchronized
                << ',' << hexHashText(row.selectedSetHash)
                << ',' << row.renderLogSetMatch
                << '\n';
        }
        VULTRA_CLIENT_INFO("Image flicker wrote {}", path.string());
    }

    struct ImageFlickerAggregate
    {
        uint32_t rows {0};
        double rawSum {0.0};
        double rawMax {0.0};
        double hpSum {0.0};
        double hpMax {0.0};
        double selectedSum {0.0};
        double churnSum {0.0};
        double normalizedChurnSum {0.0};
        double gazeSpeedMax {0.0};
        bool hpAvailable {true};

        void add(const ImageFlickerFrameRow& row)
        {
            if (row.bandLabel != "peripheral_all" || row.hasPreviousFrame == 0u)
                return;
            ++rows;
            rawSum += row.rawDeltaEnergy;
            rawMax = std::max(rawMax, row.rawDeltaEnergy);
            hpSum += row.hpDeltaEnergy;
            hpMax = std::max(hpMax, row.hpDeltaEnergy);
            selectedSum += row.selectedCount;
            churnSum += row.symmetricDifference;
            normalizedChurnSum += row.normalizedChurn;
            gazeSpeedMax = std::max(gazeSpeedMax, row.gazeSpeed);
            hpAvailable = hpAvailable && row.hpAvailable != 0u;
        }

        [[nodiscard]] double rawMean() const { return rows > 0u ? rawSum / static_cast<double>(rows) : 0.0; }
        [[nodiscard]] double hpMean() const { return rows > 0u ? hpSum / static_cast<double>(rows) : 0.0; }
        [[nodiscard]] double selectedMean() const
        {
            return rows > 0u ? selectedSum / static_cast<double>(rows) : 0.0;
        }
        [[nodiscard]] double churnMean() const { return rows > 0u ? churnSum / static_cast<double>(rows) : 0.0; }
        [[nodiscard]] double normalizedChurnMean() const
        {
            return rows > 0u ? normalizedChurnSum / static_cast<double>(rows) : 0.0;
        }
    };

    std::map<std::string, ImageFlickerAggregate> imageFlickerAggregates() const
    {
        std::map<std::string, ImageFlickerAggregate> aggregates;
        for (const auto& row : m_ImageFlickerRows)
            aggregates[row.condition].add(row);
        return aggregates;
    }

    void writeImageFlickerConditionSummaryJson()
    {
        if (m_ImageFlickerRows.empty())
            return;

        const auto path = m_Options.outputDir / "e1_flicker_condition_summary.json";
        if (!ensureOutputParentDirectory(path, "image flicker summary"))
            return;
        std::ofstream out {path};
        if (!out)
        {
            VULTRA_CLIENT_WARN("Failed to write image flicker condition summary: {}", path.string());
            return;
        }

        const auto aggregates = imageFlickerAggregates();
        const auto& firstRow = m_ImageFlickerRows.front();
        out << std::setprecision(10);
        out << "{\n";
        out << "  \"scene\": " << jsonEscape(m_Options.sceneUri) << ",\n";
        out << "  \"run_label\": \"" << (m_Options.formalExploratoryThresholdUncommitted ?
                     "exploratory_threshold_uncommitted" : "formal_threshold_committed") << "\",\n";
        out << "  \"capture_target_type\": " << jsonEscape(firstRow.captureTargetType) << ",\n";
        out << "  \"color_space\": " << jsonEscape(firstRow.colorSpace) << ",\n";
        out << "  \"image_format\": " << jsonEscape(firstRow.imageFormat) << ",\n";
        out << "  \"resolution\": \"" << firstRow.width << "x" << firstRow.height << "\",\n";
        out << "  \"sample_count\": " << firstRow.sampleCount << ",\n";
        out << "  \"sort_order\": \"deterministic_source_id_sort_proxy\",\n";
        out << "  \"pre_ui_capture\": " << (firstRow.preUiCapture != 0u ? "true" : "false") << ",\n";
        out << "  \"post_tonemap_capture\": " << (firstRow.postTonemapCapture != 0u ? "true" : "false") << ",\n";
        out << "  \"temporal_effects_disabled\": "
            << (firstRow.temporalEffectsDisabled != 0u ? "true" : "false") << ",\n";
        out << "  \"overlays_disabled\": " << (firstRow.overlaysDisabled != 0u ? "true" : "false") << ",\n";
        out << "  \"readback_synchronized\": " << (firstRow.readbackSynchronized != 0u ? "true" : "false") << ",\n";
        out << "  \"raw_delta_energy\": \"sum_squared_luminance_delta_over_eccentricity_mask\",\n";
        out << "  \"high_pass\": \"3x3 spatial delta minus box blur\",\n";
        out << "  \"high_pass_available\": true,\n";
        out << "  \"flicker_foveal_radius_deg\": " << m_Options.flickerFovealRadiusDegrees << ",\n";
        out << "  \"vertical_fov_deg\": " << m_Options.flickerVerticalFovDegrees << ",\n";
        out << "  \"conditions\": [\n";
        bool first = true;
        for (const auto& [condition, aggregate] : aggregates)
        {
            if (!first)
                out << ",\n";
            first = false;
            out << "    {\n";
            out << "      \"condition\": " << jsonEscape(condition) << ",\n";
            out << "      \"rows\": " << aggregate.rows << ",\n";
            out << "      \"selected_count_mean\": " << aggregate.selectedMean() << ",\n";
            out << "      \"mean_symmetric_difference\": " << aggregate.churnMean() << ",\n";
            out << "      \"mean_normalized_churn\": " << aggregate.normalizedChurnMean() << ",\n";
            out << "      \"max_gaze_speed\": " << aggregate.gazeSpeedMax << ",\n";
            out << "      \"raw_delta_energy_sum\": " << aggregate.rawSum << ",\n";
            out << "      \"raw_delta_energy_mean\": " << aggregate.rawMean() << ",\n";
            out << "      \"raw_delta_energy_max\": " << aggregate.rawMax << ",\n";
            out << "      \"hp_delta_energy_sum\": " << aggregate.hpSum << ",\n";
            out << "      \"hp_delta_energy_mean\": " << aggregate.hpMean() << ",\n";
            out << "      \"hp_delta_energy_max\": " << aggregate.hpMax << ",\n";
            out << "      \"hp_available\": " << (aggregate.hpAvailable ? "true" : "false") << "\n";
            out << "    }";
        }
        out << "\n  ]\n";
        out << "}\n";
    }

    void writeImageFlickerDerivedCsv()
    {
        if (m_ImageFlickerRows.empty())
            return;

        const auto path = m_Options.outputDir / "e1_flicker_vs_velocity.csv";
        if (!ensureOutputParentDirectory(path, "image flicker derived"))
            return;
        std::ofstream out {path};
        if (!out)
        {
            VULTRA_CLIENT_WARN("Failed to write image flicker derived CSV: {}", path.string());
            return;
        }
        out << std::setprecision(10);
        out << "condition,local_frame_index,gaze_speed,normalized_churn,symmetric_difference,"
               "raw_delta_energy,hp_delta_energy,band_label\n";
        for (const auto& row : m_ImageFlickerRows)
        {
            if (row.bandLabel != "peripheral_all")
                continue;
            out << row.condition
                << ',' << row.localFrameIndex
                << ',' << row.gazeSpeed
                << ',' << row.normalizedChurn
                << ',' << row.symmetricDifference
                << ',' << row.rawDeltaEnergy
                << ',' << row.hpDeltaEnergy
                << ',' << row.bandLabel
                << '\n';
        }
    }

    void writeImageFlickerE2SummaryJson()
    {
        if (m_ImageFlickerRows.empty())
            return;

        const auto aggregates = imageFlickerAggregates();
        const auto c2It = aggregates.find("C2_dynamic_deterministic");
        const auto nullIt = aggregates.find("C2_null_matched_random_swap");
        if (c2It == aggregates.end() || nullIt == aggregates.end())
            return;

        const auto path = m_Options.outputDir / "e2_c2_vs_null_flicker_summary.json";
        if (!ensureOutputParentDirectory(path, "E2 image flicker summary"))
            return;
        std::ofstream out {path};
        if (!out)
        {
            VULTRA_CLIENT_WARN("Failed to write E2 image flicker summary: {}", path.string());
            return;
        }

        const auto& c2 = c2It->second;
        const auto& null = nullIt->second;
        const auto& firstRow = m_ImageFlickerRows.front();
        const auto ratio = [](const double numerator, const double denominator) {
            return denominator != 0.0 ? numerator / denominator : 0.0;
        };
        out << std::setprecision(10);
        out << "{\n";
        out << "  \"scene\": " << jsonEscape(m_Options.sceneUri) << ",\n";
        out << "  \"run_label\": \"" << (m_Options.formalExploratoryThresholdUncommitted ?
                     "exploratory_threshold_uncommitted" : "formal_threshold_committed") << "\",\n";
        out << "  \"capture_target_type\": " << jsonEscape(firstRow.captureTargetType) << ",\n";
        out << "  \"color_space\": " << jsonEscape(firstRow.colorSpace) << ",\n";
        out << "  \"image_format\": " << jsonEscape(firstRow.imageFormat) << ",\n";
        out << "  \"sample_count\": " << firstRow.sampleCount << ",\n";
        out << "  \"sort_order\": \"deterministic_source_id_sort_proxy\",\n";
        out << "  \"high_pass_available\": true,\n";
        out << "  \"c2_rows\": " << c2.rows << ",\n";
        out << "  \"c2_null_rows\": " << null.rows << ",\n";
        out << "  \"c2_raw_delta_energy_mean\": " << c2.rawMean() << ",\n";
        out << "  \"c2_null_raw_delta_energy_mean\": " << null.rawMean() << ",\n";
        out << "  \"c2_minus_null_raw_delta_energy_mean\": " << (c2.rawMean() - null.rawMean()) << ",\n";
        out << "  \"c2_over_null_raw_delta_energy_mean_ratio\": " << ratio(c2.rawMean(), null.rawMean()) << ",\n";
        out << "  \"c2_hp_delta_energy_mean\": " << c2.hpMean() << ",\n";
        out << "  \"c2_null_hp_delta_energy_mean\": " << null.hpMean() << ",\n";
        out << "  \"c2_minus_null_hp_delta_energy_mean\": " << (c2.hpMean() - null.hpMean()) << ",\n";
        out << "  \"c2_over_null_hp_delta_energy_mean_ratio\": " << ratio(c2.hpMean(), null.hpMean()) << "\n";
        out << "}\n";
    }

    void writeImageFlickerOutputs()
    {
        if (m_ImageFlickerRows.empty())
            return;
        writeImageFlickerPerFrameCsv();
        writeImageFlickerConditionSummaryJson();
        writeImageFlickerDerivedCsv();
        writeImageFlickerE2SummaryJson();
    }

    void writeFormalFrameCsvRow(std::ofstream& out, const FormalProtocolFrameRow& row) const
    {
        out << row.sampleIndex
            << ',' << row.localFrameIndex
            << ',' << row.frameIndex
            << ',' << formalConditionLabel(row.condition)
            << ',' << membershipModeLabel(row.membershipMode)
            << ',' << row.seed
            << ',' << row.gaze.x
            << ',' << row.gaze.y
            << ',' << row.gazeVelocity.x
            << ',' << row.gazeVelocity.y
            << ',' << row.gazeSpeed
            << ',' << row.candidateCount
            << ',' << row.selectedCount
            << ',' << row.selectedFraction
            << ',' << hexHashText(row.selectedSetHash)
            << ',' << csvEscape(row.selectedSourceIdRanges)
            << ',' << row.addedCount
            << ',' << row.removedCount
            << ',' << row.symmetricDifference
            << ',' << row.normalizedChurn
            << ',' << row.jaccard
            << ',' << row.cacheFrameIndexUsed
            << ',' << row.membershipRecomputedThisFrame
            << ',' << row.renderBufferMatchesLoggedSet
            << ',' << row.shaderSideFoveationDisabled
            << ',' << row.thresholdFoveaDegrees
            << ',' << row.thresholdMidDegrees
            << ',' << row.falloffTransitionDegrees
            << ',' << row.levelCenter
            << ',' << row.levelMid
            << ',' << row.levelOuter
            << ',' << row.eccentricityMin
            << ',' << row.eccentricityMean
            << ',' << row.eccentricityMax
            << ',' << row.keepProbabilityMin
            << ',' << row.keepProbabilityMean
            << ',' << row.keepProbabilityMax
            << ',' << row.c2TargetAddCount
            << ',' << row.c2TargetRemoveCount
            << ',' << row.c2TargetSelectedCount
            << ',' << row.c2TargetSymmetricDiff
            << ',' << row.c2RealizedAddCount
            << ',' << row.c2RealizedRemoveCount
            << ',' << row.c2RealizedSelectedCount
            << ',' << row.c2RealizedSymmetricDiff
            << ',' << row.c2ChangedIdOverlap
            << ',' << row.scheduleMatchingSucceeded
            << ',' << row.cacheModeStatus
            << ',' << row.runLabel
            << ',' << csvEscape(row.notes)
            << '\n';
    }

    void writeFormalE1PerFrameCsv()
    {
        const auto path = m_Options.outputDir / "e1_per_frame.csv";
        if (!ensureOutputParentDirectory(path, "E1"))
            return;
        std::ofstream out {path};
        if (!out)
        {
            VULTRA_CLIENT_WARN("Failed to write E1 per-frame CSV: {}", path.string());
            return;
        }
        out << std::setprecision(10);
        out << "sample_index,local_frame_index,frame_index,condition,membership_mode,seed,"
               "gaze_x,gaze_y,gaze_velocity_x,gaze_velocity_y,gaze_speed,candidate_count,"
               "selected_count,selected_fraction,selected_set_hash,selected_source_id_ranges,"
               "added_count,removed_count,symmetric_difference,normalized_churn,jaccard,"
               "cache_frame_index,membership_recomputed_this_frame,render_buffer_matches_logged_set,"
               "shader_side_foveation_disabled,threshold_fovea_degrees,threshold_mid_degrees,"
               "falloff_transition_degrees,level_center,level_mid,level_outer,"
               "eccentricity_min,eccentricity_mean,eccentricity_max,"
               "keep_probability_min,keep_probability_mean,keep_probability_max,"
               "c2_target_add_count,c2_target_remove_count,c2_target_selected_count,"
               "c2_target_symmetric_diff,c2_realized_add_count,c2_realized_remove_count,"
               "c2_realized_selected_count,c2_realized_symmetric_diff,c2_changed_id_overlap,"
               "schedule_matching_succeeded,cache_mode_status,run_label,notes\n";
        for (const auto& row : m_FormalRows)
        {
            if (formalConditionIsE1(row.condition))
                writeFormalFrameCsvRow(out, row);
        }
        VULTRA_CLIENT_INFO("Formal E1 wrote {}", path.string());
    }

    void writeFormalE1DerivedCsvs()
    {
        const auto churnPath = m_Options.outputDir / "e1_churn_vs_velocity.csv";
        const auto countPath = m_Options.outputDir / "e1_selected_count_drift.csv";
        const auto jaccardPath = m_Options.outputDir / "e1_jaccard_vs_velocity.csv";
        ensureOutputParentDirectory(churnPath, "E1");

        std::ofstream churn {churnPath};
        std::ofstream count {countPath};
        std::ofstream jaccard {jaccardPath};
        if (!churn || !count || !jaccard)
        {
            VULTRA_CLIENT_WARN("Failed to write one or more formal E1 derived CSVs");
            return;
        }
        churn << std::setprecision(10);
        count << std::setprecision(10);
        jaccard << std::setprecision(10);
        churn << "condition,local_frame_index,gaze_speed,symmetric_difference,normalized_churn\n";
        count << "condition,local_frame_index,gaze_x,gaze_y,selected_count,selected_fraction,"
                 "selected_count_delta_from_first\n";
        jaccard << "condition,local_frame_index,gaze_speed,jaccard\n";

        std::map<FormalProtocolCondition, uint32_t> firstSelectedCount;
        for (const auto& row : m_FormalRows)
        {
            if (!formalConditionIsE1(row.condition))
                continue;
            if (!firstSelectedCount.contains(row.condition))
                firstSelectedCount.emplace(row.condition, row.selectedCount);
            const int64_t delta =
                static_cast<int64_t>(row.selectedCount) -
                static_cast<int64_t>(firstSelectedCount[row.condition]);
            churn << formalConditionLabel(row.condition)
                  << ',' << row.localFrameIndex
                  << ',' << row.gazeSpeed
                  << ',' << row.symmetricDifference
                  << ',' << row.normalizedChurn
                  << '\n';
            count << formalConditionLabel(row.condition)
                  << ',' << row.localFrameIndex
                  << ',' << row.gaze.x
                  << ',' << row.gaze.y
                  << ',' << row.selectedCount
                  << ',' << row.selectedFraction
                  << ',' << delta
                  << '\n';
            jaccard << formalConditionLabel(row.condition)
                    << ',' << row.localFrameIndex
                    << ',' << row.gazeSpeed
                    << ',' << row.jaccard
                    << '\n';
        }
    }

    void writeFormalE1SummaryJson()
    {
        const auto path = m_Options.outputDir / "e1_condition_summary.json";
        if (!ensureOutputParentDirectory(path, "E1"))
            return;
        std::ofstream out {path};
        if (!out)
        {
            VULTRA_CLIENT_WARN("Failed to write E1 summary JSON: {}", path.string());
            return;
        }

        out << std::setprecision(10);
        out << "{\n";
        out << "  \"scene\": " << jsonEscape(m_Options.sceneUri) << ",\n";
        out << "  \"run_label\": \"" << (m_Options.formalExploratoryThresholdUncommitted ?
                     "exploratory_threshold_uncommitted" : "formal_threshold_committed") << "\",\n";
        out << "  \"image_flicker_energy\": "
            << jsonEscape(m_ImageFlickerRows.empty() ? "missing_selection_churn_only" :
                          "available_regime_a_offscreen_srgb_proxy_deterministic_source_id_sort")
            << ",\n";
        out << "  \"c1_static_mean_count_target\": "
            << (m_C1StaticMeanCountTarget == UINT32_MAX ? 0u : m_C1StaticMeanCountTarget) << ",\n";
        out << "  \"conditions\": [\n";

        bool firstCondition = true;
        for (const FormalProtocolCondition condition : {
                 FormalProtocolCondition::eC1StaticRandom,
                 FormalProtocolCondition::eC1StaticMeanCount,
                 FormalProtocolCondition::eC2DynamicDeterministic,
                 FormalProtocolCondition::eC3DynamicFrozenHash})
        {
            uint32_t rows = 0u;
            uint32_t minSelected = UINT32_MAX;
            uint32_t maxSelected = 0u;
            uint32_t maxSymDiff = 0u;
            double minJaccard = 1.0;
            double selectedSum = 0.0;
            double churnSum = 0.0;
            double gazeSpeedMax = 0.0;
            std::vector<uint64_t> hashes;
            bool renderMatch = true;
            bool recomputed = true;
            bool shaderDisabled = true;
            for (const auto& row : m_FormalRows)
            {
                if (row.condition != condition)
                    continue;
                ++rows;
                minSelected = std::min(minSelected, row.selectedCount);
                maxSelected = std::max(maxSelected, row.selectedCount);
                maxSymDiff = std::max(maxSymDiff, row.symmetricDifference);
                minJaccard = std::min(minJaccard, row.jaccard);
                selectedSum += row.selectedCount;
                churnSum += row.symmetricDifference;
                gazeSpeedMax = std::max(gazeSpeedMax, row.gazeSpeed);
                hashes.push_back(row.selectedSetHash);
                renderMatch = renderMatch && row.renderBufferMatchesLoggedSet != 0u;
                recomputed = recomputed && row.membershipRecomputedThisFrame != 0u;
                shaderDisabled = shaderDisabled && row.shaderSideFoveationDisabled != 0u;
            }
            std::sort(hashes.begin(), hashes.end());
            hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
            if (!firstCondition)
                out << ",\n";
            firstCondition = false;
            out << "    {\n";
            out << "      \"condition\": \"" << formalConditionLabel(condition) << "\",\n";
            out << "      \"rows\": " << rows << ",\n";
            out << "      \"selected_count_min\": " << (rows > 0u ? minSelected : 0u) << ",\n";
            out << "      \"selected_count_max\": " << maxSelected << ",\n";
            out << "      \"selected_count_mean\": " << (rows > 0u ? selectedSum / rows : 0.0) << ",\n";
            out << "      \"max_symmetric_difference\": " << maxSymDiff << ",\n";
            out << "      \"mean_symmetric_difference\": " << (rows > 0u ? churnSum / rows : 0.0) << ",\n";
            out << "      \"min_jaccard\": " << minJaccard << ",\n";
            out << "      \"unique_selected_hashes\": " << hashes.size() << ",\n";
            out << "      \"max_gaze_speed\": " << gazeSpeedMax << ",\n";
            out << "      \"render_buffer_matched_logged_set\": " << (renderMatch ? "true" : "false") << ",\n";
            out << "      \"membership_recomputed\": " << (recomputed ? "true" : "false") << ",\n";
            out << "      \"shader_side_foveation_disabled\": " << (shaderDisabled ? "true" : "false") << "\n";
            out << "    }";
        }

        out << "\n  ]\n";
        out << "}\n";
    }

    void writeFormalE2Outputs()
    {
        const auto schedulePath = m_Options.outputDir / "e2_c2_schedule.csv";
        const auto nullPath = m_Options.outputDir / "e2_c2_null_per_frame.csv";
        const auto summaryPath = m_Options.outputDir / "e2_c2_vs_null_summary.json";
        ensureOutputParentDirectory(schedulePath, "E2");

        std::ofstream schedule {schedulePath};
        if (schedule)
        {
            schedule << std::setprecision(10);
            schedule << "local_frame_index,frame_index,add_count,remove_count,selected_count,"
                        "symmetric_difference,normalized_churn,jaccard,selected_set_hash,changed_id_count\n";
            for (const auto& row : m_C2ScheduleRows)
            {
                schedule << row.localFrameIndex
                         << ',' << row.frameIndex
                         << ',' << row.addCount
                         << ',' << row.removeCount
                         << ',' << row.selectedCount
                         << ',' << row.symmetricDifference
                         << ',' << row.normalizedChurn
                         << ',' << row.jaccard
                         << ',' << hexHashText(row.selectedSetHash)
                         << ',' << row.changedIds.size()
                         << '\n';
            }
        }

        std::ofstream nullCsv {nullPath};
        if (nullCsv)
        {
            nullCsv << std::setprecision(10);
            nullCsv << "sample_index,local_frame_index,frame_index,condition,membership_mode,seed,"
                       "gaze_x,gaze_y,gaze_speed,selected_count,selected_fraction,selected_set_hash,"
                       "target_add_count,realized_add_count,target_remove_count,realized_remove_count,"
                       "target_selected_count,realized_selected_count,target_symmetric_difference,"
                       "realized_symmetric_difference,jaccard,c2_changed_id_overlap,"
                       "schedule_matching_succeeded,render_buffer_matches_logged_set\n";
            for (const auto& row : m_FormalRows)
            {
                if (row.condition != FormalProtocolCondition::eC2NullMatchedRandomSwap)
                    continue;
                nullCsv << row.sampleIndex
                        << ',' << row.localFrameIndex
                        << ',' << row.frameIndex
                        << ',' << formalConditionLabel(row.condition)
                        << ',' << membershipModeLabel(row.membershipMode)
                        << ',' << row.seed
                        << ',' << row.gaze.x
                        << ',' << row.gaze.y
                        << ',' << row.gazeSpeed
                        << ',' << row.selectedCount
                        << ',' << row.selectedFraction
                        << ',' << hexHashText(row.selectedSetHash)
                        << ',' << row.c2TargetAddCount
                        << ',' << row.c2RealizedAddCount
                        << ',' << row.c2TargetRemoveCount
                        << ',' << row.c2RealizedRemoveCount
                        << ',' << row.c2TargetSelectedCount
                        << ',' << row.c2RealizedSelectedCount
                        << ',' << row.c2TargetSymmetricDiff
                        << ',' << row.c2RealizedSymmetricDiff
                        << ',' << row.jaccard
                        << ',' << row.c2ChangedIdOverlap
                        << ',' << row.scheduleMatchingSucceeded
                        << ',' << row.renderBufferMatchesLoggedSet
                        << '\n';
            }
        }

        uint32_t nullRows = 0u;
        uint32_t matchedRows = 0u;
        uint32_t maxSelectedDelta = 0u;
        uint32_t maxAddDelta = 0u;
        uint32_t maxRemoveDelta = 0u;
        uint32_t maxSymDelta = 0u;
        uint64_t overlapSum = 0u;
        uint64_t targetChangedSum = 0u;
        for (const auto& row : m_FormalRows)
        {
            if (row.condition != FormalProtocolCondition::eC2NullMatchedRandomSwap)
                continue;
            ++nullRows;
            matchedRows += row.scheduleMatchingSucceeded ? 1u : 0u;
            maxSelectedDelta = std::max<uint32_t>(
                maxSelectedDelta,
                static_cast<uint32_t>(std::abs(static_cast<int64_t>(row.c2TargetSelectedCount) -
                                               static_cast<int64_t>(row.c2RealizedSelectedCount))));
            maxAddDelta = std::max<uint32_t>(
                maxAddDelta,
                static_cast<uint32_t>(std::abs(static_cast<int64_t>(row.c2TargetAddCount) -
                                               static_cast<int64_t>(row.c2RealizedAddCount))));
            maxRemoveDelta = std::max<uint32_t>(
                maxRemoveDelta,
                static_cast<uint32_t>(std::abs(static_cast<int64_t>(row.c2TargetRemoveCount) -
                                               static_cast<int64_t>(row.c2RealizedRemoveCount))));
            maxSymDelta = std::max<uint32_t>(
                maxSymDelta,
                static_cast<uint32_t>(std::abs(static_cast<int64_t>(row.c2TargetSymmetricDiff) -
                                               static_cast<int64_t>(row.c2RealizedSymmetricDiff))));
            overlapSum += row.c2ChangedIdOverlap;
            targetChangedSum += row.c2TargetSymmetricDiff;
        }

        std::ofstream summary {summaryPath};
        if (summary)
        {
            summary << std::setprecision(10);
            summary << "{\n";
            summary << "  \"scene\": " << jsonEscape(m_Options.sceneUri) << ",\n";
            summary << "  \"condition\": \"C2_null_matched_random_swap\",\n";
            summary << "  \"rows\": " << nullRows << ",\n";
            summary << "  \"matched_rows\": " << matchedRows << ",\n";
            summary << "  \"schedule_matching_succeeded\": "
                    << (nullRows > 0u && matchedRows == nullRows ? "true" : "false") << ",\n";
            summary << "  \"max_selected_count_delta\": " << maxSelectedDelta << ",\n";
            summary << "  \"max_add_count_delta\": " << maxAddDelta << ",\n";
            summary << "  \"max_remove_count_delta\": " << maxRemoveDelta << ",\n";
            summary << "  \"max_symmetric_difference_delta\": " << maxSymDelta << ",\n";
            summary << "  \"changed_id_overlap_fraction\": "
                    << (targetChangedSum > 0u ?
                            static_cast<double>(overlapSum) / static_cast<double>(targetChangedSum) :
                            0.0)
                    << "\n";
            summary << "}\n";
        }
    }

    void writeFormalE3Outputs()
    {
        const auto eventsPath = m_Options.outputDir / "e3_churn_events.csv";
        const auto concentrationPath = m_Options.outputDir / "e3_energy_concentration.json";
        const auto countCdfPath = m_Options.outputDir / "e3_event_count_cdf.csv";
        const auto energyCdfPath = m_Options.outputDir / "e3_energy_weighted_cdf.csv";
        ensureOutputParentDirectory(eventsPath, "E3");

        std::ofstream events {eventsPath};
        if (events)
        {
            events << std::setprecision(10);
            events << "frame_index,local_frame_index,condition,membership_mode,source_id,event_type,"
                      "gaze_x,gaze_y,eccentricity,base_opacity_alpha_proxy,post_weight_alpha_proxy,"
                      "projected_radius_proxy,footprint_area_proxy,tile_cost,transmittance,alphaT,"
                      "fallback_contribution_proxy,exact_alphaT_available,exact_transmittance_available\n";
            for (const auto& event : m_E3ChurnEvents)
            {
                events << event.frameIndex
                       << ',' << event.localFrameIndex
                       << ',' << formalConditionLabel(event.condition)
                       << ',' << membershipModeLabel(event.membershipMode)
                       << ',' << event.sourceId
                       << ',' << (event.enter ? "enter" : "leave")
                       << ',' << event.gaze.x
                       << ',' << event.gaze.y
                       << ',' << event.eccentricity
                       << ',' << event.alphaProxy
                       << ',' << event.postWeightAlphaProxy
                       << ',' << event.projectedRadiusProxy
                       << ',' << event.footprintAreaProxy
                       << ',' << event.tileCost
                       << ',' << event.transmittance
                       << ',' << event.alphaT
                       << ',' << event.contributionProxy
                       << ',' << (event.exactAlphaTAvailable ? "true" : "false")
                       << ',' << (event.exactTransmittanceAvailable ? "true" : "false")
                       << '\n';
            }
        }

        struct EnergyGroup
        {
            FormalProtocolCondition condition {FormalProtocolCondition::eNone};
            std::string eventType;
            std::vector<double> values;
        };
        std::map<std::pair<FormalProtocolCondition, std::string>, EnergyGroup> groups;
        for (const auto& event : m_E3ChurnEvents)
        {
            const std::string eventType = event.enter ? "enter" : "leave";
            auto& group = groups[{event.condition, eventType}];
            group.condition = event.condition;
            group.eventType = eventType;
            group.values.push_back(std::max(event.contributionProxy, 0.0));
        }

        auto topShare = [](std::vector<double> values, const double fraction) {
            if (values.empty())
                return 0.0;
            const double total = std::accumulate(values.begin(), values.end(), 0.0);
            if (total <= 0.0)
                return 0.0;
            std::sort(values.begin(), values.end(), std::greater<double>());
            const size_t count = std::max<size_t>(1u, static_cast<size_t>(std::ceil(values.size() * fraction)));
            const double top = std::accumulate(values.begin(), values.begin() + std::min(count, values.size()), 0.0);
            return top / total;
        };
        auto gini = [](std::vector<double> values) {
            if (values.empty())
                return 0.0;
            std::sort(values.begin(), values.end());
            const double total = std::accumulate(values.begin(), values.end(), 0.0);
            if (total <= 0.0)
                return 0.0;
            double weighted = 0.0;
            for (size_t i = 0u; i < values.size(); ++i)
                weighted += static_cast<double>(i + 1u) * values[i];
            const double n = static_cast<double>(values.size());
            return (2.0 * weighted) / (n * total) - (n + 1.0) / n;
        };

        std::ofstream concentration {concentrationPath};
        if (concentration)
        {
            concentration << std::setprecision(10);
            concentration << "{\n  \"scene\": " << jsonEscape(m_Options.sceneUri) << ",\n";
            concentration << "  \"exact_alphaT_available\": false,\n";
            concentration << "  \"exact_transmittance_available\": false,\n";
            concentration << "  \"proxy\": \"contribution_proxy = opacity_proxy * max(projected_area_px, tile_cost)\",\n";
            concentration << "  \"groups\": [\n";
            bool first = true;
            for (const auto& [key, group] : groups)
            {
                const double total = std::accumulate(group.values.begin(), group.values.end(), 0.0);
                if (!first)
                    concentration << ",\n";
                first = false;
                concentration << "    {\n";
                concentration << "      \"condition\": \"" << formalConditionLabel(group.condition) << "\",\n";
                concentration << "      \"event_type\": \"" << group.eventType << "\",\n";
                concentration << "      \"event_count\": " << group.values.size() << ",\n";
                concentration << "      \"total_contribution_proxy\": " << total << ",\n";
                concentration << "      \"top_1pct_share\": " << topShare(group.values, 0.01) << ",\n";
                concentration << "      \"top_5pct_share\": " << topShare(group.values, 0.05) << ",\n";
                concentration << "      \"top_10pct_share\": " << topShare(group.values, 0.10) << ",\n";
                concentration << "      \"gini\": " << gini(group.values) << "\n";
                concentration << "    }";
            }
            concentration << "\n  ]\n}\n";
        }

        std::ofstream countCdf {countCdfPath};
        std::ofstream energyCdf {energyCdfPath};
        if (countCdf && energyCdf)
        {
            countCdf << std::setprecision(10);
            energyCdf << std::setprecision(10);
            countCdf << "condition,event_type,quantile,contribution_proxy,cumulative_event_fraction\n";
            energyCdf << "condition,event_type,quantile,contribution_proxy,cumulative_energy_fraction\n";
            for (auto& [key, group] : groups)
            {
                if (group.values.empty())
                    continue;
                std::sort(group.values.begin(), group.values.end());
                const double total = std::accumulate(group.values.begin(), group.values.end(), 0.0);
                std::vector<double> prefix(group.values.size());
                std::partial_sum(group.values.begin(), group.values.end(), prefix.begin());
                for (uint32_t q = 0u; q <= 100u; ++q)
                {
                    const double quantile = static_cast<double>(q) / 100.0;
                    const size_t index = std::min<size_t>(
                        group.values.size() - 1u,
                        static_cast<size_t>(std::floor(quantile * static_cast<double>(group.values.size() - 1u))));
                    const double eventFraction =
                        static_cast<double>(index + 1u) / static_cast<double>(group.values.size());
                    const double energyFraction =
                        total > 0.0 ? prefix[index] / total : 0.0;
                    countCdf << formalConditionLabel(group.condition)
                             << ',' << group.eventType
                             << ',' << quantile
                             << ',' << group.values[index]
                             << ',' << eventFraction
                             << '\n';
                    energyCdf << formalConditionLabel(group.condition)
                              << ',' << group.eventType
                              << ',' << quantile
                              << ',' << group.values[index]
                              << ',' << energyFraction
                              << '\n';
                }
            }
        }
    }

    static double giniCoefficient(std::vector<double> values)
    {
        if (values.empty())
            return 0.0;
        std::sort(values.begin(), values.end());
        const double total = std::accumulate(values.begin(), values.end(), 0.0);
        if (total <= 0.0)
            return 0.0;
        double weighted = 0.0;
        for (size_t i = 0u; i < values.size(); ++i)
            weighted += static_cast<double>(i + 1u) * values[i];
        const double n = static_cast<double>(values.size());
        return (2.0 * weighted) / (n * total) - (n + 1.0) / n;
    }

    static double topEnergyShare(std::vector<double> values, const double fraction)
    {
        if (values.empty())
            return 0.0;
        const double total = std::accumulate(values.begin(), values.end(), 0.0);
        if (total <= 0.0)
            return 0.0;
        std::sort(values.begin(), values.end(), std::greater<double>());
        const size_t count = std::max<size_t>(1u, static_cast<size_t>(std::ceil(values.size() * fraction)));
        const double top = std::accumulate(values.begin(),
                                           values.begin() + std::min(count, values.size()),
                                           0.0);
        return top / total;
    }

    static double pearsonCorrelation(const std::vector<double>& xs, const std::vector<double>& ys)
    {
        if (xs.size() != ys.size() || xs.size() < 2u)
            return 0.0;
        const double n = static_cast<double>(xs.size());
        const double meanX = std::accumulate(xs.begin(), xs.end(), 0.0) / n;
        const double meanY = std::accumulate(ys.begin(), ys.end(), 0.0) / n;
        double cov = 0.0;
        double varX = 0.0;
        double varY = 0.0;
        for (size_t i = 0u; i < xs.size(); ++i)
        {
            const double dx = xs[i] - meanX;
            const double dy = ys[i] - meanY;
            cov += dx * dy;
            varX += dx * dx;
            varY += dy * dy;
        }
        if (varX <= 0.0 || varY <= 0.0)
            return 0.0;
        return cov / std::sqrt(varX * varY);
    }

    void writeE3HonestE6Outputs()
    {
        if (!m_Options.e3HonestE6)
            return;

        const auto pixelPath = m_Options.outputDir / "e3_pixel_delta_per_frame.csv";
        ensureOutputParentDirectory(pixelPath, "E3 honest");

        std::ofstream pixel {pixelPath};
        if (pixel)
        {
            pixel << std::setprecision(10);
            pixel << "sample_index,local_frame_index,actual_frame_index,forced_frame_index,condition,"
                     "membership_mode,seed,gaze_x,gaze_y,gaze_speed,selected_count,"
                     "previous_selected_count,forced_selected_count,symmetric_difference,"
                     "normalized_churn,jaccard,selected_set_hash,previous_selected_set_hash,"
                     "forced_selected_set_hash,width,height,band_label,band_min_deg,band_max_deg,"
                     "pixel_count,nonzero_pixel_count,raw_pixel_delta_energy,raw_pixel_delta_mean,"
                     "hp_pixel_delta_energy,hp_pixel_delta_mean,hp_available,actual_frame_image_hash,"
                     "forced_frame_image_hash,forced_ids_requested_count,forced_ids_matched_count,"
                     "forced_ids_all_found,render_log_set_match,shader_side_foveation_disabled,"
                     "delta_nonzero,color_space,capture_target_type,image_format,run_label,notes\n";
            for (const auto& row : m_E3PixelDeltaRows)
            {
                pixel << row.sampleIndex
                      << ',' << row.localFrameIndex
                      << ',' << row.actualFrameIndex
                      << ',' << row.forcedFrameIndex
                      << ',' << formalConditionLabel(row.condition)
                      << ',' << membershipModeLabel(row.membershipMode)
                      << ',' << row.seed
                      << ',' << row.gaze.x
                      << ',' << row.gaze.y
                      << ',' << row.gazeSpeed
                      << ',' << row.selectedCount
                      << ',' << row.previousSelectedCount
                      << ',' << row.forcedSelectedCount
                      << ',' << row.symmetricDifference
                      << ',' << row.normalizedChurn
                      << ',' << row.jaccard
                      << ',' << hexHashText(row.selectedSetHash)
                      << ',' << hexHashText(row.previousSelectedSetHash)
                      << ',' << hexHashText(row.forcedSelectedSetHash)
                      << ',' << row.width
                      << ',' << row.height
                      << ',' << row.bandLabel
                      << ',' << row.bandMinDegrees
                      << ',' << row.bandMaxDegrees
                      << ',' << row.pixelCount
                      << ',' << row.nonzeroPixelCount
                      << ',' << row.rawPixelDeltaEnergy
                      << ',' << row.rawPixelDeltaMean
                      << ',' << row.hpPixelDeltaEnergy
                      << ',' << row.hpPixelDeltaMean
                      << ',' << row.hpAvailable
                      << ',' << hexHashText(row.actualFrameImageHash)
                      << ',' << hexHashText(row.forcedFrameImageHash)
                      << ',' << row.forcedIdsRequestedCount
                      << ',' << row.forcedIdsMatchedCount
                      << ',' << row.forcedIdsAllFound
                      << ',' << row.renderLogSetMatch
                      << ',' << row.shaderSideFoveationDisabled
                      << ',' << row.deltaNonzero
                      << ',' << row.colorSpace
                      << ',' << row.captureTargetType
                      << ',' << row.imageFormat
                      << ',' << csvEscape(row.runLabel)
                      << ',' << csvEscape(row.notes)
                      << '\n';
            }
        }

        struct ProxyFrame
        {
            double proxySum {0.0};
            uint32_t eventCount {0u};
        };
        std::map<std::pair<FormalProtocolCondition, uint32_t>, ProxyFrame> proxyByFrame;
        for (const auto& event : m_E3ChurnEvents)
        {
            auto& proxy = proxyByFrame[{event.condition, event.localFrameIndex}];
            proxy.proxySum += std::max(event.contributionProxy, 0.0);
            ++proxy.eventCount;
        }

        std::map<std::pair<FormalProtocolCondition, uint32_t>, const E3PixelDeltaFrameRow*> fullRows;
        std::map<std::pair<FormalProtocolCondition, uint32_t>, const E3PixelDeltaFrameRow*> peripheralRows;
        for (const auto& row : m_E3PixelDeltaRows)
        {
            if (row.bandLabel == "full_frame")
                fullRows[{row.condition, row.localFrameIndex}] = &row;
            else if (row.bandLabel == "peripheral_all")
                peripheralRows[{row.condition, row.localFrameIndex}] = &row;
        }

        const auto proxyPath = m_Options.outputDir / "e3_proxy_vs_pixel_delta.csv";
        std::ofstream proxyCsv {proxyPath};
        std::vector<double> proxyValues;
        std::vector<double> fullEnergyValues;
        std::vector<double> peripheralEnergyValues;
        if (proxyCsv)
        {
            proxyCsv << std::setprecision(10);
            proxyCsv << "condition,local_frame_index,event_count,proxy_contribution_sum,"
                        "full_raw_pixel_delta_energy,peripheral_raw_pixel_delta_energy,"
                        "full_hp_pixel_delta_energy,peripheral_hp_pixel_delta_energy,"
                        "event_level_pixel_delta_available,attribution_unit,run_label,notes\n";
            for (const auto& [key, fullRow] : fullRows)
            {
                const auto proxyIt = proxyByFrame.find(key);
                const auto periphIt = peripheralRows.find(key);
                const ProxyFrame proxyFrame =
                    proxyIt != proxyByFrame.end() ? proxyIt->second : ProxyFrame {};
                const auto* peripheral = periphIt != peripheralRows.end() ? periphIt->second : nullptr;
                proxyCsv << formalConditionLabel(key.first)
                         << ',' << key.second
                         << ',' << proxyFrame.eventCount
                         << ',' << proxyFrame.proxySum
                         << ',' << fullRow->rawPixelDeltaEnergy
                         << ',' << (peripheral ? peripheral->rawPixelDeltaEnergy : 0.0)
                         << ',' << fullRow->hpPixelDeltaEnergy
                         << ',' << (peripheral ? peripheral->hpPixelDeltaEnergy : 0.0)
                         << ",false,frame_changed_set,"
                         << csvEscape(e3CalibrationRunLabel())
                         << ",\"actual_pixel_delta_is_frame_level;event_level_attribution_unavailable\"\n";
                proxyValues.push_back(proxyFrame.proxySum);
                fullEnergyValues.push_back(fullRow->rawPixelDeltaEnergy);
                peripheralEnergyValues.push_back(peripheral ? peripheral->rawPixelDeltaEnergy : 0.0);
            }
        }

        const auto validationPath = m_Options.outputDir / "e3_proxy_validation_summary.json";
        std::ofstream validation {validationPath};
        if (validation)
        {
            validation << std::setprecision(10);
            validation << "{\n";
            validation << "  \"run_label\": " << jsonEscape(e3CalibrationRunLabel()) << ",\n";
            validation << "  \"exact_alphaT_available\": false,\n";
            validation << "  \"event_level_pixel_delta_available\": false,\n";
            validation << "  \"pixel_delta_attribution_unit\": \"frame_changed_set\",\n";
            validation << "  \"proxy\": \"contribution_proxy = opacity_proxy * max(projected_area_px, tile_cost)\",\n";
            validation << "  \"proxy_vs_full_raw_pixel_delta_pearson\": "
                       << pearsonCorrelation(proxyValues, fullEnergyValues) << ",\n";
            validation << "  \"proxy_vs_peripheral_raw_pixel_delta_pearson\": "
                       << pearsonCorrelation(proxyValues, peripheralEnergyValues) << ",\n";
            validation << "  \"paired_frame_count\": " << proxyValues.size() << ",\n";
            validation << "  \"notes\": \"Proxy correlation is frame/churn-batch level only; per-Gaussian pixel attribution remains missing.\"\n";
            validation << "}\n";
        }

        const auto concentrationPath = m_Options.outputDir / "e3_pixel_delta_energy_concentration.csv";
        const auto lorenzPath = m_Options.outputDir / "e3_pixel_delta_lorenz.csv";
        const auto giniPath = m_Options.outputDir / "e3_pixel_delta_gini.csv";
        std::ofstream concentration {concentrationPath};
        std::ofstream lorenz {lorenzPath};
        std::ofstream giniOut {giniPath};
        if (concentration)
        {
            concentration << std::setprecision(10);
            concentration << "condition,band_label,energy_type,attribution_unit,event_level_pixel_delta_available,"
                             "frame_count,total_energy,top_1pct_share,top_5pct_share,top_10pct_share,gini,"
                             "tail_label,run_label,notes\n";
        }
        if (lorenz)
        {
            lorenz << std::setprecision(10);
            lorenz << "condition,band_label,energy_type,quantile,cumulative_frame_fraction,"
                      "cumulative_energy_fraction,run_label\n";
        }
        if (giniOut)
        {
            giniOut << std::setprecision(10);
            giniOut << "condition,band_label,energy_type,frame_count,total_energy,gini,"
                       "top_1pct_share,top_5pct_share,top_10pct_share,tail_label,run_label\n";
        }

        std::map<std::tuple<FormalProtocolCondition, std::string, std::string>, std::vector<double>> energyGroups;
        for (const auto& row : m_E3PixelDeltaRows)
        {
            energyGroups[{row.condition, row.bandLabel, "raw"}].push_back(row.rawPixelDeltaEnergy);
            energyGroups[{row.condition, row.bandLabel, "hp"}].push_back(row.hpPixelDeltaEnergy);
        }

        for (auto& [key, values] : energyGroups)
        {
            auto [condition, bandLabel, energyType] = key;
            const double total = std::accumulate(values.begin(), values.end(), 0.0);
            const double top1 = topEnergyShare(values, 0.01);
            const double top5 = topEnergyShare(values, 0.05);
            const double top10 = topEnergyShare(values, 0.10);
            const double gini = giniCoefficient(values);
            const char* tailLabel = top1 >= 0.40 || gini >= 0.75 ? "heavy_tailed_pixel_delta_frame_level" :
                                    top1 >= 0.20 || gini >= 0.50 ? "intermediate_pixel_delta_frame_level" :
                                                                   "diffuse_pixel_delta_frame_level";
            if (concentration)
            {
                concentration << formalConditionLabel(condition)
                              << ',' << bandLabel
                              << ',' << energyType
                              << ",frame_changed_set,false,"
                              << values.size()
                              << ',' << total
                              << ',' << top1
                              << ',' << top5
                              << ',' << top10
                              << ',' << gini
                              << ',' << tailLabel
                              << ',' << csvEscape(e3CalibrationRunLabel())
                              << ",\"actual freeze-membership pixel delta; event-level attribution unavailable\"\n";
            }
            if (giniOut)
            {
                giniOut << formalConditionLabel(condition)
                        << ',' << bandLabel
                        << ',' << energyType
                        << ',' << values.size()
                        << ',' << total
                        << ',' << gini
                        << ',' << top1
                        << ',' << top5
                        << ',' << top10
                        << ',' << tailLabel
                        << ',' << csvEscape(e3CalibrationRunLabel())
                        << '\n';
            }
            if (lorenz && !values.empty())
            {
                std::sort(values.begin(), values.end());
                std::vector<double> prefix(values.size(), 0.0);
                std::partial_sum(values.begin(), values.end(), prefix.begin());
                for (uint32_t q = 0u; q <= 100u; ++q)
                {
                    const double quantile = static_cast<double>(q) / 100.0;
                    const size_t index = std::min<size_t>(
                        values.size() - 1u,
                        static_cast<size_t>(std::floor(quantile * static_cast<double>(values.size() - 1u))));
                    lorenz << formalConditionLabel(condition)
                           << ',' << bandLabel
                           << ',' << energyType
                           << ',' << quantile
                           << ',' << static_cast<double>(index + 1u) / static_cast<double>(values.size())
                           << ',' << (total > 0.0 ? prefix[index] / total : 0.0)
                           << ',' << csvEscape(e3CalibrationRunLabel())
                           << '\n';
                }
            }
        }

        auto writeSpectrum = [&](const std::filesystem::path& path,
                                 const std::vector<E6SpectrumBinRow>& rows) {
            std::ofstream out {path};
            if (!out)
                return;
            out << std::setprecision(10);
            out << "sample_index,local_frame_index,actual_frame_index,forced_frame_index,condition,"
                   "field_type,radius_bin,normalized_radius,power,bin_sample_count,run_label,notes\n";
            for (const auto& row : rows)
            {
                out << row.sampleIndex
                    << ',' << row.localFrameIndex
                    << ',' << row.actualFrameIndex
                    << ',' << row.forcedFrameIndex
                    << ',' << formalConditionLabel(row.condition)
                    << ',' << row.fieldType
                    << ',' << row.radiusBin
                    << ',' << row.normalizedRadius
                    << ',' << row.power
                    << ',' << row.binSampleCount
                    << ',' << csvEscape(row.runLabel)
                    << ',' << csvEscape(row.notes)
                    << '\n';
            }
        };
        writeSpectrum(m_Options.outputDir / "e6_flip_field_binary_spectrum.csv", m_E6BinarySpectrumRows);
        writeSpectrum(m_Options.outputDir / "e6_flip_field_weighted_spectrum.csv", m_E6WeightedSpectrumRows);

        const auto bandPath = m_Options.outputDir / "e6_frequency_band_energy.csv";
        std::ofstream bandCsv {bandPath};
        if (bandCsv)
        {
            bandCsv << std::setprecision(10);
            bandCsv << "sample_index,local_frame_index,actual_frame_index,forced_frame_index,condition,"
                       "field_type,selected_count,previous_selected_count,symmetric_difference,"
                       "normalized_churn,low_frequency_fraction,mid_frequency_fraction,"
                       "high_frequency_fraction,total_power,run_label,notes\n";
            for (const auto& row : m_E6FrequencyBandRows)
            {
                bandCsv << row.sampleIndex
                        << ',' << row.localFrameIndex
                        << ',' << row.actualFrameIndex
                        << ',' << row.forcedFrameIndex
                        << ',' << formalConditionLabel(row.condition)
                        << ',' << row.fieldType
                        << ',' << row.selectedCount
                        << ',' << row.previousSelectedCount
                        << ',' << row.symmetricDifference
                        << ',' << row.normalizedChurn
                        << ',' << row.lowFrequencyFraction
                        << ',' << row.midFrequencyFraction
                        << ',' << row.highFrequencyFraction
                        << ',' << row.totalPower
                        << ',' << csvEscape(row.runLabel)
                        << ',' << csvEscape(row.notes)
                        << '\n';
            }
        }

        struct E6GroupSummary
        {
            uint32_t count {0u};
            double lowSum {0.0};
            double midSum {0.0};
            double highSum {0.0};
            double powerSum {0.0};
            double churnSum {0.0};
            double selectedSum {0.0};
        };
        std::map<std::pair<FormalProtocolCondition, std::string>, E6GroupSummary> e6Groups;
        for (const auto& row : m_E6FrequencyBandRows)
        {
            auto& group = e6Groups[{row.condition, row.fieldType}];
            ++group.count;
            group.lowSum += row.lowFrequencyFraction;
            group.midSum += row.midFrequencyFraction;
            group.highSum += row.highFrequencyFraction;
            group.powerSum += row.totalPower;
            group.churnSum += row.symmetricDifference;
            group.selectedSum += row.selectedCount;
        }

        const auto spectrumSummaryPath = m_Options.outputDir / "e6_spectrum_summary.json";
        std::ofstream spectrumSummary {spectrumSummaryPath};
        if (spectrumSummary)
        {
            spectrumSummary << std::setprecision(10);
            spectrumSummary << "{\n";
            spectrumSummary << "  \"run_label\": " << jsonEscape(e3CalibrationRunLabel()) << ",\n";
            spectrumSummary << "  \"binary_field\": \"delta_nonzero_proxy_exact_gaussian_touch_unavailable\",\n";
            spectrumSummary << "  \"weighted_field\": \"abs_freeze_membership_delta_luminance\",\n";
            spectrumSummary << "  \"spectrum_grid_size\": " << m_Options.e6SpectrumGridSize << ",\n";
            spectrumSummary << "  \"groups\": [\n";
            bool first = true;
            for (const auto& [key, group] : e6Groups)
            {
                if (!first)
                    spectrumSummary << ",\n";
                first = false;
                spectrumSummary << "    {\n";
                spectrumSummary << "      \"condition\": \"" << formalConditionLabel(key.first) << "\",\n";
                spectrumSummary << "      \"field_type\": " << jsonEscape(key.second) << ",\n";
                spectrumSummary << "      \"frame_count\": " << group.count << ",\n";
                spectrumSummary << "      \"mean_low_frequency_fraction\": "
                                << (group.count > 0u ? group.lowSum / group.count : 0.0) << ",\n";
                spectrumSummary << "      \"mean_mid_frequency_fraction\": "
                                << (group.count > 0u ? group.midSum / group.count : 0.0) << ",\n";
                spectrumSummary << "      \"mean_high_frequency_fraction\": "
                                << (group.count > 0u ? group.highSum / group.count : 0.0) << ",\n";
                spectrumSummary << "      \"mean_total_power\": "
                                << (group.count > 0u ? group.powerSum / group.count : 0.0) << ",\n";
                spectrumSummary << "      \"mean_symmetric_difference\": "
                                << (group.count > 0u ? group.churnSum / group.count : 0.0) << ",\n";
                spectrumSummary << "      \"mean_selected_count\": "
                                << (group.count > 0u ? group.selectedSum / group.count : 0.0) << "\n";
                spectrumSummary << "    }";
            }
            spectrumSummary << "\n  ]\n";
            spectrumSummary << "}\n";
        }

        const auto churnPath = m_Options.outputDir / "e6_churn_count_c2_vs_c3.csv";
        std::ofstream churn {churnPath};
        if (churn)
        {
            churn << std::setprecision(10);
            churn << "condition,field_type,frame_count,mean_selected_count,mean_symmetric_difference,"
                     "mean_normalized_churn,mean_low_frequency_fraction,mean_mid_frequency_fraction,"
                     "mean_high_frequency_fraction,run_label\n";
            for (const auto& [key, group] : e6Groups)
            {
                churn << formalConditionLabel(key.first)
                      << ',' << key.second
                      << ',' << group.count
                      << ',' << (group.count > 0u ? group.selectedSum / group.count : 0.0)
                      << ',' << (group.count > 0u ? group.churnSum / group.count : 0.0)
                      << ',' << (group.count > 0u && group.selectedSum > 0.0 ?
                                     group.churnSum / group.selectedSum :
                                     0.0)
                      << ',' << (group.count > 0u ? group.lowSum / group.count : 0.0)
                      << ',' << (group.count > 0u ? group.midSum / group.count : 0.0)
                      << ',' << (group.count > 0u ? group.highSum / group.count : 0.0)
                      << ',' << csvEscape(e3CalibrationRunLabel())
                      << '\n';
            }
        }
    }

    void writeFormalProtocolOutputs()
    {
        if (!m_Options.formalE1E2E3 || m_FormalRows.empty())
            return;

        writeFormalE1PerFrameCsv();
        writeFormalE1DerivedCsvs();
        writeFormalE1SummaryJson();
        if (!m_Options.e3HonestE6)
            writeFormalE2Outputs();
        writeFormalE3Outputs();
        writeE3HonestE6Outputs();
    }

    void finishBenchmark()
    {
        if (m_BenchmarkFinished || !m_Options.benchmarkEnabled)
            return;

        m_BenchmarkFinished = true;
        if (m_Samples.empty())
        {
            VULTRA_CLIENT_WARN("Gaussian benchmark produced no samples");
            releaseImageFlickerOffscreenCaptureTarget();
            return;
        }

        writeBenchmarkCsv(m_Options.outputPath, m_Samples);
        writeCachedSelectionOracleCsv();
        writeFormalProtocolOutputs();
        writeImageFlickerOutputs();
        if (m_Options.benchmarkScreenshotOutput)
            saveBenchmarkScreenshot(*m_Options.benchmarkScreenshotOutput);

        const auto cpuFrameStats  = summarizeSeries(collectSeries(m_Samples, &GaussianBenchmarkSample::cpuFrameMs));
        const auto gpuFrameStats  = summarizeSeries(collectSeries(m_Samples, &GaussianBenchmarkSample::gpuFrameMs));
        const auto gpuPreprocess =
            summarizeSeries(collectSeries(m_Samples, &GaussianBenchmarkSample::gpuPreprocessPassMs));
        const auto gpuRenderPass =
            summarizeSeries(collectSeries(m_Samples, &GaussianBenchmarkSample::gpuRenderPassMs));
        const auto cpuClodSelect =
            summarizeSeries(collectSeries(m_Samples, &GaussianBenchmarkSample::cpuClodSelectionMs));

        const auto& last = m_Samples.back();
        std::cout << "\nGaussian benchmark summary\n"
                  << "  samples: " << m_Samples.size() << "\n"
                  << "  output: " << m_Options.outputPath.string() << "\n"
                  << "  mode: " << gaussianModeLabel(last.baselineMode) << "\n"
                  << "  scripted_sweep: camera=" << benchmarkCameraPathLabel(m_Options.benchmarkCameraPath)
                  << ", gaze=" << benchmarkGazePathLabel(m_Options.benchmarkGazePath)
                  << ", view_frames=" << m_Options.benchmarkViewFrames << "\n"
                  << "  stereo_sequential: " << (m_Options.benchmarkStereoSequential ? "yes" : "no")
                  << ", ipd=" << m_Options.benchmarkIpdMeters << "\n"
                  << "  gaze_rendering: " << (last.foveatedClodEnabled ? "yes" : "no") << "\n"
                  << "  gaze_render_mode: " << foveatedRenderModeLabel(last.foveatedRenderMode) << "\n"
                  << "  sh_lod: " << (last.shLodEnabled ? "yes" : "no")
                  << ", degrees=" << last.shDegreeCenter << "/" << last.shDegreeMid << "/"
                  << last.shDegreeOuter << "\n"
                  << "  sh_storage_layout: " << shStorageLayoutLabel(last.shStorageLayout)
                  << ", band_reads=" << last.shBandL1ReadsEst << "/" << last.shBandL2ReadsEst
                  << "/" << last.shBandL3ReadsEst
                  << ", band_bytes=" << last.shBandBytesEst << "\n"
                  << "  sh_lod_guard: " << shLodGuardModeLabel(last.shGuardMode)
                  << ", raised=" << last.shGuardRaisedCount
                  << ", thresholds=" << last.shGuardThresholdMid << "/" << last.shGuardThresholdHigh << "\n"
                  << "  coverage_compensation: " << (last.foveatedCoverageCompensationEnabled ? "yes" : "no") << "\n"
                  << "  coverage_guard: " << foveatedCoverageGuardModeLabel(last.coverageGuardMode)
                  << ", protection=" << last.coverageProtectionDegrees
                  << ", budget_ratio=" << last.coverageGuardBudgetRatio << "\n"
                  << "  gaze_uv: " << last.gazeX << ", " << last.gazeY << "\n"
                  << "  layered_compositor: "
                  << (last.foveatedLayeredCompositeEnabled ? "yes" : "no") << "\n"
                  << "  gaze_adaptation: " << foveatedAdaptationModeLabel(last.foveatedAdaptationMode) << "\n"
                  << "  ring_degrees: " << last.foveaDegrees << ", " << last.midDegrees << "\n"
                  << "  ring_lod: " << last.foveaLod << ", " << last.midLod << ", " << last.outerLod << "\n"
                  << "  ring_res: " << last.foveaResolutionScale << ", " << last.midResolutionScale << ", "
                  << last.outerResolutionScale << "\n"
                  << "  direct_prefix: " << (last.directPrefix ? "yes" : "no") << "\n"
                  << "  shader_antipop: mode=" << shaderAntiPopModeLabel(last.shaderAntiPopMode)
                  << ", enabled=" << (last.shaderAntiPopEnabled ? "yes" : "no")
                  << ", alpha_multiplier=" << (last.shaderAntiPopAlphaMultiplierBased ? "yes" : "no")
                  << ", avoids_cpu_selected_rebuild="
                  << (last.shaderAntiPopAvoidsCpuSelectedSourceRebuild ? "yes" : "no")
                  << ", stable_candidate_set=" << (last.shaderAntiPopStableCandidateSet ? "yes" : "no")
                  << ", direct_prefix=" << (last.shaderAntiPopDirectPrefixStable ? "yes" : "no")
                  << ", guard_proxy=" << shaderAntiPopGuardProxyLabel(last.shaderAntiPopGuardProxyMode)
                  << "\n"
                  << "  splats: total=" << last.totalSplats << ", prepared=" << last.preparedSplats
                  << ", selected_raw=" << last.lodSelectedRawSplats
                  << ", ring_budgets=" << last.foveaSplatBudget << "/" << last.midSplatBudget << "/"
                  << last.outerSplatBudget << "\n"
                  << "  CPU frame: " << statsText(cpuFrameStats) << "\n"
                  << "  GPU frame: " << statsText(gpuFrameStats) << "\n"
                  << "  GPU preprocess pass: " << statsText(gpuPreprocess) << "\n"
                  << "  GPU render pass: " << statsText(gpuRenderPass) << "\n"
                  << "  CPU CLOD prefix build: " << statsText(cpuClodSelect) << "\n";
        if (m_AlphaCapturedFrameCount > 0u)
        {
            const double avgCaptureOverhead =
                m_AlphaCaptureOverheadMs / static_cast<double>(m_AlphaCapturedFrameCount);
            std::cout << "  alpha_capture: frames=" << m_AlphaCapturedFrameCount
                      << ", overhead_total_ms=" << m_AlphaCaptureOverheadMs
                      << ", overhead_avg_ms=" << avgCaptureOverhead
                      << ", timing=post-sample readback outside benchmark frame timings\n";
        }
        std::cout << "\n";

        VULTRA_CLIENT_INFO("Gaussian benchmark wrote {} samples to {}", m_Samples.size(), m_Options.outputPath.string());
        releaseImageFlickerOffscreenCaptureTarget();
    }

    void saveBenchmarkScreenshot(const std::filesystem::path& path)
    {
        if (path.has_parent_path())
        {
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
            if (ec)
            {
                VULTRA_CLIENT_WARN("Failed to create benchmark screenshot directory {}: {}",
                                   path.parent_path().string(),
                                   ec.message());
                return;
            }
        }

        auto& backendService = engineCtx().services.require<IRenderBackendService>();
        auto& rd             = backendService.renderDevice();
        if (!rd.saveTextureToFile(backendService.backbuffer(), path.string()))
        {
            VULTRA_CLIENT_WARN("Failed to save Gaussian benchmark screenshot: {}", path.string());
            return;
        }
        VULTRA_CLIENT_INFO("Gaussian benchmark screenshot wrote {}", path.string());
    }

    bool saveBenchmarkTextureToFile(const rhi::Texture& texture,
                                    const std::filesystem::path& path,
                                    const std::string_view label)
    {
        if (path.has_parent_path())
        {
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
            if (ec)
            {
                VULTRA_CLIENT_WARN("Failed to create {} directory {}: {}",
                                   label,
                                   path.parent_path().string(),
                                   ec.message());
                return false;
            }
        }

        auto& backendService = engineCtx().services.require<IRenderBackendService>();
        auto& rd             = backendService.renderDevice();
        if (!rd.saveTextureToFile(texture, path.string()))
        {
            VULTRA_CLIENT_WARN("Failed to save {}: {}", label, path.string());
            return false;
        }
        VULTRA_CLIENT_INFO("{} wrote {}", label, path.string());
        return true;
    }

    void saveBenchmarkFrameCapture(const GaussianBenchmarkSample& sample)
    {
        if (!m_Options.captureFrameSequence)
            return;
        if (m_Options.captureFrameLimit > 0u && m_CapturedFrameCount >= m_Options.captureFrameLimit)
            return;

        std::ostringstream filename;
        filename << m_Options.captureFramePrefix << "_frame_" << std::setw(6) << std::setfill('0')
                 << sample.frameIndex << ".png";

        const auto path = m_Options.captureFrameDir / filename.str();
        const bool saved = m_ImageFlickerOffscreenTarget ?
                               saveBenchmarkTextureToFile(
                                   m_ImageFlickerOffscreenTarget,
                                   path,
                                   "Gaussian benchmark offscreen scene-color frame capture") :
                               (saveBenchmarkScreenshot(path), true);
        if (saved)
            ++m_CapturedFrameCount;
    }

    void saveBenchmarkAlphaCapture(const GaussianBenchmarkSample& sample)
    {
        if (!m_Options.captureAlpha)
            return;
        if (m_Options.captureAlphaLimit > 0u && m_AlphaCapturedFrameCount >= m_Options.captureAlphaLimit)
            return;

        const auto* alphaTexture = m_RenderService ? m_RenderService->gaussianSplatLastAlphaTexture() : nullptr;
        if (!alphaTexture)
        {
            if (!m_AlphaCaptureUnavailableWarned)
            {
                VULTRA_CLIENT_WARN(
                    "Gaussian alpha capture unavailable for this render mode; use single-pass/non-layered Gaussian rendering");
                m_AlphaCaptureUnavailableWarned = true;
            }
            return;
        }

        if (!m_Options.captureAlphaDir.empty())
        {
            std::error_code ec;
            std::filesystem::create_directories(m_Options.captureAlphaDir, ec);
            if (ec)
            {
                VULTRA_CLIENT_WARN("Failed to create alpha capture directory {}: {}",
                                   m_Options.captureAlphaDir.string(),
                                   ec.message());
                return;
            }
        }

        std::ostringstream filename;
        filename << m_Options.captureFramePrefix << "_alpha_" << std::setw(6) << std::setfill('0')
                 << sample.frameIndex << ".png";
        const auto path = m_Options.captureAlphaDir / filename.str();

        auto& backendService = engineCtx().services.require<IRenderBackendService>();
        auto& rd             = backendService.renderDevice();
        const auto start     = std::chrono::steady_clock::now();
        const bool saved     = rd.saveTextureAlphaToFile(*alphaTexture, path.string());
        const auto end       = std::chrono::steady_clock::now();
        if (!saved)
        {
            if (!m_AlphaCaptureSaveFailureWarned)
            {
                VULTRA_CLIENT_WARN("Failed to save Gaussian alpha capture: {}", path.string());
                m_AlphaCaptureSaveFailureWarned = true;
            }
            return;
        }

        m_AlphaCaptureOverheadMs += std::chrono::duration<double, std::milli>(end - start).count();
        ++m_AlphaCapturedFrameCount;
    }

private:
    GaussianDemoOptions                   m_Options {};
    IRenderService*                       m_RenderService {nullptr};
    RuntimeProfiler*                      m_Profiler {nullptr};
    std::vector<GaussianBenchmarkSample>  m_Samples;
    std::vector<BenchmarkCameraFrame>      m_BenchmarkCameraFrames;
    uint64_t                              m_BenchmarkTicks {0};
    uint64_t                              m_LastCollectedFrame {std::numeric_limits<uint64_t>::max()};
    uint32_t                              m_CapturedFrameCount {0};
    uint32_t                              m_AlphaCapturedFrameCount {0};
    double                                m_AlphaCaptureOverheadMs {0.0};
    bool                                  m_AlphaCaptureUnavailableWarned {false};
    bool                                  m_AlphaCaptureSaveFailureWarned {false};
    uint32_t                              m_CurrentScriptedViewIndex {UINT32_MAX};
    uint32_t                              m_CurrentScriptedGazeIndex {UINT32_MAX};
    uint32_t                              m_CurrentIsGazeMoving {0};
    uint32_t                              m_CurrentGazeJumpEvent {0};
    int32_t                               m_CurrentTransitionWindowId {-1};
    bool                                  m_PreviousSelectedIdLogValid {false};
    bool                                  m_RenderDocCaptureRequested {false};
    uint32_t                              m_PreviousSelectedIdCount {0};
    std::vector<uint32_t>                 m_PreviousSelectedSourceIds;
    std::vector<CachedSelectionOracleLogRow> m_E0OracleRows;
    bool                                  m_E0OraclePreviousValid {false};
    std::vector<uint32_t>                 m_E0OraclePreviousSourceIds;
    bool                                  m_E0OracleFailed {false};
    FormalProtocolCondition               m_CurrentFormalCondition {FormalProtocolCondition::eNone};
    uint32_t                              m_CurrentFormalLocalFrameIndex {0u};
    std::array<FormalProtocolConditionState, 6> m_FormalConditionStates {};
    std::vector<FormalProtocolFrameRow>   m_FormalRows;
    std::vector<FormalProtocolScheduleRow> m_C2ScheduleRows;
    std::vector<FormalChurnEventRow>      m_E3ChurnEvents;
    std::vector<ImageFlickerFrameRow>      m_ImageFlickerRows;
    std::map<std::string, ImageFlickerPreviousFrame> m_ImageFlickerPreviousByCondition;
    ImageFlickerReadback                  m_LastImageFlickerReadback;
    bool                                  m_LastFormalFrameHasPrevious {false};
    FormalProtocolCondition               m_LastFormalCondition {FormalProtocolCondition::eNone};
    uint32_t                              m_LastFormalLocalFrameIndex {0u};
    glm::vec2                             m_LastFormalGazeVelocity {0.0f, 0.0f};
    double                                m_LastFormalJaccard {1.0};
    std::vector<uint32_t>                 m_LastFormalPreviousSourceIds;
    std::vector<uint32_t>                 m_LastFormalCurrentSourceIds;
    std::vector<uint32_t>                 m_LastFormalChangedSourceIds;
    E3HonestPendingCounterfactual         m_E3CounterfactualPending;
    std::vector<E3PixelDeltaFrameRow>     m_E3PixelDeltaRows;
    std::vector<E6SpectrumBinRow>         m_E6BinarySpectrumRows;
    std::vector<E6SpectrumBinRow>         m_E6WeightedSpectrumRows;
    std::vector<E6FrequencyBandRow>       m_E6FrequencyBandRows;
    rhi::Texture                           m_ImageFlickerOffscreenTarget;
    bool                                  m_ImageFlickerReadbackFailedWarned {false};
    bool                                  m_ImageFlickerNonRegimeAWarned {false};
    uint64_t                              m_C2SelectedCountSum {0u};
    uint32_t                              m_C2SelectedCountSampleCount {0u};
    uint32_t                              m_C1StaticMeanCountTarget {UINT32_MAX};
    bool                                  m_BenchmarkFinished {false};
    bool                                  m_BenchmarkExitRequested {false};
};

int main(int argc, char** argv)
{
    GaussianSplattingDemoApp app {};
    return app.run(argc, argv);
}

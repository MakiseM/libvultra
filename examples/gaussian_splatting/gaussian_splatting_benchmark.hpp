#pragma once

#include <vultra/core/base/base.hpp>
#include <vultra/core/base/common_context.hpp>
#include <vultra/function/rendering/render_structs.hpp>
#include <vultra/function/rendering/runtime_profiler.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace vultra::gaussian_splatting_example
{
    struct GaussianBenchmarkSample
    {
        uint32_t sampleIndex {0};
        uint64_t frameIndex {0};
        uint32_t scriptedViewIndex {UINT32_MAX};
        uint32_t scriptedGazeIndex {UINT32_MAX};

        double dtMs {0.0};
        double cpuFrameMs {0.0};
        double cpuRenderMs {0.0};
        double gpuFrameMs {-1.0};

        uint64_t drawCalls {0};
        uint64_t dispatchCalls {0};
        uint64_t copyOps {0};
        uint64_t updateOps {0};
        uint32_t gpuScopeResolvedCount {0};
        uint32_t gpuScopeTokenCount {0};

        GaussianSplatBaselineMode           baselineMode {GaussianSplatBaselineMode::eBaseline};
        GaussianSplatFoveatedRenderMode     foveatedRenderMode {GaussianSplatFoveatedRenderMode::eSinglePass};
        GaussianSplatLodBudgetMode          lodBudgetMode {GaussianSplatLodBudgetMode::eCount};
        GaussianSplatFoveatedAdaptationMode foveatedAdaptationMode {GaussianSplatFoveatedAdaptationMode::eFixed};
        GaussianSplatFoveatedDistribution   foveatedDistribution {GaussianSplatFoveatedDistribution::eGaussian};
        bool                                lodBudgetEnabled {false};
        bool                                foveatedClodEnabled {false};
        bool                                foveatedLayeredCompositeEnabled {false};
        bool                                foveatedCoverageCompensationEnabled {false};
        GaussianSplatFoveatedCoverageGuardMode coverageGuardMode {GaussianSplatFoveatedCoverageGuardMode::eGlobal};
        bool                                directPrefix {false};
        uint32_t                            lodBudget {0};
        uint64_t                            costBudget {0};
        uint64_t                            actualProjectedCost {0};
        uint64_t                            costOvershoot {0};
        float                               projectedCostBudgetRatio {0.0};
        uint32_t                            projectedCostChunkSize {1024};
        uint32_t                            foveatedCoverageBinGridX {12};
        uint32_t                            foveatedCoverageBinGridY {8};
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
        uint32_t                            a4NumChunksScanned {0};
        uint32_t                            a4NumGaussiansOrChunksConsidered {0};
        uint64_t                            a4ProjectedSampleCacheHit {0};
        uint64_t                            a4ProjectedSampleCacheMiss {0};
        bool                                a4ProjectedSampleReused {false};
        uint64_t                            a4NumProjectedSamplesBuilt {0};
        uint64_t                            a4NumProjectedSamplesReused {0};
        uint64_t                            a4CacheEntryCount {0};
        uint64_t                            a4CacheLookupCount {0};
        uint64_t                            a4CacheHitCount {0};
        uint64_t                            a4CacheMissCount {0};
        uint64_t                            a4CacheFullMissCount {0};
        uint64_t                            a4CacheBuildCount {0};
        uint64_t                            a4CacheEvictionCount {0};
        uint64_t                            a4SampleCacheKeyHash {0};
        uint64_t                            a4ChunkCacheHitCount {0};
        uint64_t                            a4ChunkCacheMissCount {0};
        uint64_t                            a4ProjectedSamplesBuilt {0};
        uint64_t                            a4ProjectedSamplesReused {0};
        double                              a4ChunkAggregationCpuMs {-1.0};
        double                              a4CacheLookupCpuMs {-1.0};
        double                              a4CacheHitRate {-1.0};
        bool                                a4GpuCostBuildEnabled {false};
        double                              a4GpuCostBuildGpuMs {-1.0};
        double                              a4GpuCostReadbackCpuMs {-1.0};
        uint32_t                            a4GpuCostChunkCount {0};
        bool                                a4GpuCostValid {false};
        bool                                a4CpuFallbackUsed {false};
        double                              a4GpuCpuCostL1Error {-1.0};
        double                              a4GpuCpuCostMaxError {-1.0};
        double                              a4GpuCpuSelectedRatioDelta {-1.0};
        double                              a4GpuCpuActualCostDelta {-1.0};
        double                              a4GpuCpuOvershootDelta {-1.0};
        float                               gazeX {0.5};
        float                               gazeY {0.5};
        uint32_t                            isGazeMoving {0};
        uint32_t                            gazeJumpEvent {0};
        int32_t                             transitionWindowId {-1};
        float                               foveaLod {1.0};
        float                               midLod {0.40};
        float                               outerLod {0.15};
        float                               foveaDegrees {8.0};
        float                               midDegrees {24.0};
        float                               continuousTheta0Degrees {24.0};
        float                               continuousAlpha {2.0};
        float                               continuousMinLevel {0.12};
        float                               foveaResolutionScale {1.0};
        float                               midResolutionScale {0.75};
        float                               outerResolutionScale {0.50};
        float                               targetFrameMs {11.1};
        float                               coverageProtectionDegrees {0.0};
        float                               coverageGuardBudgetRatio {0.05};
        float                               coverageGuardCenterMin {0.75};
        float                               coverageGuardTransitionMin {0.45};
        float                               coverageGuardPeripheryMin {0.15};
        uint32_t                            coverageGuardMaxAdds {0};
        uint32_t                            guardMaxAddsEffective {0};
        bool                                coverageGuardModeEnabled {false};
        uint32_t                            coverageGuardRiskSectors {0};
        bool                                coverageGuardRiskDetected {false};
        bool                                coverageGuardActive {false};
        bool                                coverageGuardRepairActive {false};
        bool                                coverageGuardAddedAny {false};
        uint32_t                            coverageGuardAddedCount {0};
        uint64_t                            coverageGuardAddedCost {0};
        uint64_t                            coverageGuardBudgetCap {0};
        uint32_t                            riskSectorCountTotal {0};
        uint32_t                            riskSectorCountCenter {0};
        uint32_t                            riskSectorCountMid {0};
        uint32_t                            riskSectorCountOuter {0};
        bool                                riskSectorActiveCenter {false};
        bool                                riskSectorActiveMid {false};
        bool                                riskSectorActiveOuter {false};
        float                               coverageFailureBeforeGuard {0.0};
        float                               coverageFailureAfterGuard {0.0};
        float                               coverageFailureBeforeCenter {-1.0};
        float                               coverageFailureBeforeMid {-1.0};
        float                               coverageFailureBeforeOuter {-1.0};
        float                               coverageFailureAfterCenter {-1.0};
        float                               coverageFailureAfterMid {-1.0};
        float                               coverageFailureAfterOuter {-1.0};
        float                               coverageScoreBeforeGuard {1.0};
        float                               coverageScoreAfterGuard {1.0};
        uint32_t                            guardAddedCountCenter {0};
        uint32_t                            guardAddedCountMid {0};
        uint32_t                            guardAddedCountOuter {0};
        uint64_t                            guardAddedCostCenter {0};
        uint64_t                            guardAddedCostMid {0};
        uint64_t                            guardAddedCostOuter {0};
        bool                                guardRepairCapHit {false};
        bool                                guardRepairCappedByMaxAdds {false};
        bool                                guardRepairCappedByCost {false};
        bool                                guardRepairNoCandidate {false};
        uint32_t                            guardCandidateCount {0};
        uint32_t                            guardCandidateCountCenter {0};
        uint32_t                            guardCandidateCountMid {0};
        uint32_t                            guardCandidateCountOuter {0};
        uint32_t                            guardTrueCandidateCount {0};
        bool                                guardCandidateCountTruncated {false};
        uint32_t                            guardRequiredExtraCountEstimate {UINT32_MAX};
        double                              guardRequiredExtraAreaPx {-1.0};
        double                              guardDeficitAreaBefore {-1.0};
        double                              guardDeficitAreaAfter {-1.0};
        uint32_t                            guardAddedCountToFailingSector {0};
        uint64_t                            guardAddedCostToFailingSector {0};
        uint32_t                            guardFailingSectorId {UINT32_MAX};
        double                              guardRepairSectorMatchRate {-1.0};
        double                              guardAnalysisCpuMs {-1.0};
        double                              guardProjectedSampleBuildCpuMs {-1.0};
        double                              guardCoverageAreaCpuMs {-1.0};
        double                              guardCandidateScanCpuMs {-1.0};
        double                              guardRegionRebuildCpuMs {-1.0};
        uint64_t                            guardCacheFullMissCount {0};
        uint64_t                            guardCacheHitCount {0};
        uint64_t                            guardCacheMissCount {0};
        uint64_t                            guardGeometryCacheHitCount {0};
        uint64_t                            guardGeometryCacheMissCount {0};
        uint64_t                            guardGeometryCacheFullMissCount {0};
        uint64_t                            guardGeometrySamplesBuilt {0};
        uint64_t                            guardGeometrySamplesReused {0};
        bool                                schedulerShrinkEvent {false};
        bool                                schedulerGrowEvent {false};
        float                               schedulerBudgetBefore {-1.0};
        float                               schedulerBudgetAfter {-1.0};
        double                              schedulerP95Ema {-1.0};
        double                              schedulerP95Window {-1.0};
        float                               schedulerTargetMs {-1.0};
        float                               schedulerMarginLow {-1.0};
        float                               schedulerMarginHigh {-1.0};
        float                               projectedCostBudgetRatioBeforeGuard {-1.0};
        float                               projectedCostBudgetRatioAfterScheduler {-1.0};
        float                               selectedRatioBeforeGuard {-1.0};
        float                               selectedRatioAfterGuard {-1.0};
        double                              guardAddedCostRatioToBudget {-1.0};
        double                              guardAddedCountRatioToSelected {-1.0};
        bool                                selectedIdLogEnabled {false};
        uint32_t                            selectedIdLogStride {1};
        uint32_t                            selectedIdCount {0};
        double                              selectedIdIou {-1.0};
        double                              selectedIdChurn {-1.0};
        bool                                selectedIdExactAvailable {false};
        uint32_t                            selectedIdAddedCount {0};
        uint32_t                            selectedIdRemovedCount {0};
        double                              trueSelectedIdIou {-1.0};
        double                              trueSelectedIdChurn {-1.0};
        double                              a4SelectedIdLoggingCpuMs {-1.0};
        std::vector<uint32_t>               selectedSourceIds;
        bool                                temporalHysteresisEnabled {true};
        bool                                boundarySmoothingEnabled {true};
        uint32_t                            temporalResidencyFrames {6};
        float                               temporalHysteresisRatio {0.25};
        float                               boundarySmoothingRatio {0.18};
        float                               temporalPeripheralScale {0.0};
        bool                                shLodEnabled {false};
        bool                                shSmoothSuppressionEnabled {false};
        bool                                peripheralTemporalFilterEnabled {false};
        float                               peripheralTemporalFilterMidDegrees {32.0f};
        float                               peripheralTemporalFilterOuterDegrees {56.0f};
        float                               peripheralTemporalFilterLambdaScale {0.85f};
        float                               peripheralTemporalFilterRejectionThreshold {0.18f};
        float                               peripheralTemporalFilterClampRadius {0.20f};
        uint32_t                            shDegreeCenter {3};
        uint32_t                            shDegreeMid {3};
        uint32_t                            shDegreeOuter {3};
        float                               shSmoothL1StartDegrees {32.0f};
        float                               shSmoothL1EndDegrees {40.0f};
        float                               shSmoothL2StartDegrees {22.0f};
        float                               shSmoothL2EndDegrees {32.0f};
        float                               shSmoothL3StartDegrees {12.0f};
        float                               shSmoothL3EndDegrees {22.0f};
        uint64_t                            estimatedShAcCoeffReads {0};
        double                              estimatedShAcReadReductionVsDegree3 {0.0};
        GaussianSplatShStorageLayout        shStorageLayout {GaussianSplatShStorageLayout::eMonolithic};
        uint64_t                            shBandL1ReadsEst {0};
        uint64_t                            shBandL2ReadsEst {0};
        uint64_t                            shBandL3ReadsEst {0};
        uint64_t                            shBandBytesEst {0};
        double                              shBandBytesReductionVsMonolithic {0.0};
        uint64_t                            shStorageMetadataBytesEst {0};
        bool                                shEnergyMetadataAvailable {false};
        double                              shEnergyMeanAfter0 {0.0};
        double                              shEnergyMeanAfter1 {0.0};
        double                              shEnergyMeanAfter2 {0.0};
        double                              shEnergyP95After0 {0.0};
        double                              shEnergyP95After1 {0.0};
        double                              shEnergyP95After2 {0.0};
        GaussianSplatShLodGuardMode         shGuardMode {GaussianSplatShLodGuardMode::eOff};
        uint32_t                            shGuardRaisedCount {0};
        double                              shGuardRaisedRatio {0.0};
        uint64_t                            shGuardRecoveredAcReads {0};
        uint64_t                            shGuardSavedAcReadsAfterGuard {0};
        float                               shGuardThresholdMid {0.05f};
        float                               shGuardThresholdHigh {0.20f};
        uint32_t                            shDegreeChangedCount {0};
        double                              shDegreeChangedRatio {0.0};
        double                              shPopEnergyProxy {0.0};
        double                              shPopEnergyFovea {0.0};
        double                              shPopEnergyMid {0.0};
        double                              shPopEnergyPeriphery {0.0};
        uint32_t                            shGuardRaiseCount {0};
        uint32_t                            shDelayedDowngradeCount {0};
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
        uint32_t                            shaderAntiPopHashSeed {0};
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
        uint32_t                            shaderAntiPopCandidateCount {0};
        int32_t                             shaderAntiPopEffectiveNonzeroEstimate {-1};
        float                               shaderAntiPopWeightMinEstimate {-1.0f};
        float                               shaderAntiPopWeightMeanEstimate {-1.0f};
        float                               shaderAntiPopWeightMaxEstimate {-1.0f};
        double                              shaderAntiPopAlphaMassEstimate {-1.0};
        uint32_t                            shaderAntiPopGuardProxyMode {0};
        uint32_t                            splatAssets {0};
        uint32_t                            drawRecords {0};
        uint32_t                            totalSplats {0};
        uint32_t                            preparedSplats {0};
        uint32_t                            foveaSplatBudget {0};
        uint32_t                            midSplatBudget {0};
        uint32_t                            outerSplatBudget {0};
        uint32_t                            maxVisibleSplatCap {0};
        uint32_t                            lodSelectedRawSplats {0};
        uint64_t                            numTileIntersections {0};
        double                              sumProjectedAreaPx {0.0};
        bool                                cleanTimingMode {false};
        uint32_t                            visibleSplats {UINT32_MAX};
        uint32_t                            drawnSplats {UINT32_MAX};
        uint32_t                            visibleInstant {UINT32_MAX};
        uint32_t                            tileInstances {UINT32_MAX};
        uint32_t                            coveredPixelCount {UINT32_MAX};

        double cpuRenderFrameMs {-1.0};
        double cpuCookMs {-1.0};
        double cpuGpuSceneRebuildMs {-1.0};
        double cpuLodSelectionMs {-1.0};
        double cpuClodSelectionMs {-1.0};
        double cpuRawSelectionMs {-1.0};
        double cpuLodUploadMs {-1.0};
        double cpuFrameGraphBuildMs {-1.0};
        double cpuFrameGraphExecuteMs {-1.0};

        double gpuPreprocessPassMs {-1.0};
        double gpuProjectCullMs {-1.0};
        double gpuGuideAccumMs {-1.0};
        double gpuSelectionMs {-1.0};
        double gpuSortMs {-1.0};
        double gpuWriteIndirectMs {-1.0};
        double gpuRenderPassMs {-1.0};
        double gpuCompositePassMs {-1.0};
    };

    struct SeriesStats
    {
        double average {0.0};
        double median {0.0};
        double minimum {0.0};
        double maximum {0.0};
    };

    inline std::string_view gaussianModeLabel(const GaussianSplatBaselineMode mode)
    {
        switch (mode)
        {
            case GaussianSplatBaselineMode::eBaseline:
                return "baseline";
            case GaussianSplatBaselineMode::eOrderedClod:
                return "ordered-clod";
        }
        return "unknown";
    }

    inline std::string_view foveatedRenderModeLabel(const GaussianSplatFoveatedRenderMode mode)
    {
        switch (mode)
        {
            case GaussianSplatFoveatedRenderMode::eSinglePass:
                return "single-pass";
            case GaussianSplatFoveatedRenderMode::eLayeredComposite:
                return "layered-composite";
        }
        return "unknown";
    }

    inline std::string_view lodBudgetModeLabel(const GaussianSplatLodBudgetMode mode)
    {
        switch (mode)
        {
            case GaussianSplatLodBudgetMode::eCount:
                return "count";
            case GaussianSplatLodBudgetMode::eProjectedTileCost:
                return "projected-cost";
            case GaussianSplatLodBudgetMode::eFoveatedScore:
                return "foveated-score";
            case GaussianSplatLodBudgetMode::eCoverageBinScore:
                return "coverage-bin-score";
        }
        return "unknown";
    }

    inline std::string_view shaderAntiPopModeLabel(const GaussianSplatShaderAntiPopMode mode)
    {
        switch (mode)
        {
            case GaussianSplatShaderAntiPopMode::eOff:
                return "off";
            case GaussianSplatShaderAntiPopMode::eSoftRamp:
                return "soft_ramp";
            case GaussianSplatShaderAntiPopMode::eHashRamp:
                return "hash_ramp";
            case GaussianSplatShaderAntiPopMode::eHashRampGuarded:
                return "hash_ramp_guarded";
            case GaussianSplatShaderAntiPopMode::eLinearDefault:
                return "linear_default";
            case GaussianSplatShaderAntiPopMode::eGuardedGazeAnchorCrossfade:
                return "guarded_gaze_anchor_crossfade";
            case GaussianSplatShaderAntiPopMode::eEccentricityStochasticTransition:
                return "eccentricity_stochastic_transition";
            case GaussianSplatShaderAntiPopMode::eStableOpticalDepthThinning:
                return "stable_optical_depth_thinning";
            case GaussianSplatShaderAntiPopMode::eCoverageStableLogpolarRelease:
                return "coverage_stable_logpolar_release";
        }
        return "unknown";
    }

    inline std::string_view shaderAntiPopNormalizeModeLabel(const GaussianSplatShaderAntiPopNormalizeMode mode)
    {
        switch (mode)
        {
            case GaussianSplatShaderAntiPopNormalizeMode::eOff:
                return "off";
            case GaussianSplatShaderAntiPopNormalizeMode::eGlobalLuma:
                return "global_luma";
            case GaussianSplatShaderAntiPopNormalizeMode::eAlphaMass:
                return "alpha_mass";
        }
        return "unknown";
    }

    inline std::string_view shaderAntiPopGuardProxyLabel(const uint32_t mode)
    {
        switch (mode)
        {
            case 0u:
                return "none";
            case 1u:
                return "ordered_rank_importance";
            case 2u:
                return "gaze_anchor_opacity_eccentricity_proxy";
            case 3u:
                return "eccentricity_stochastic_opacity_eccentricity_proxy";
        }
        return "unknown";
    }

    inline std::string_view shaderAntiPopPKeepCurveLabel(const GaussianSplatShaderAntiPopPKeepCurve curve)
    {
        switch (curve)
        {
            case GaussianSplatShaderAntiPopPKeepCurve::eCurrent:
                return "current";
            case GaussianSplatShaderAntiPopPKeepCurve::eLinear:
                return "linear";
            case GaussianSplatShaderAntiPopPKeepCurve::eSmoothWide:
                return "smooth_wide";
            case GaussianSplatShaderAntiPopPKeepCurve::eLogisticSoft:
                return "logistic_soft";
            case GaussianSplatShaderAntiPopPKeepCurve::eLogisticSteep:
                return "logistic_steep";
        }
        return "unknown";
    }

    inline std::string_view projectedCostBuildModeLabel(const GaussianProjectedCostBuildMode mode)
    {
        switch (mode)
        {
            case GaussianProjectedCostBuildMode::eCpu:
                return "cpu";
            case GaussianProjectedCostBuildMode::eGpuSync:
                return "gpu-sync";
        }
        return "unknown";
    }

    inline std::string_view foveatedAdaptationModeLabel(const GaussianSplatFoveatedAdaptationMode mode)
    {
        switch (mode)
        {
            case GaussianSplatFoveatedAdaptationMode::eFixed:
                return "fixed";
            case GaussianSplatFoveatedAdaptationMode::eDynamicBudget:
                return "dynamic-budget";
            case GaussianSplatFoveatedAdaptationMode::eStabilityAwareBudget:
                return "stability-aware-budget";
            case GaussianSplatFoveatedAdaptationMode::eDynamicRange:
                return "dynamic-range";
            case GaussianSplatFoveatedAdaptationMode::eProgressiveCenterOut:
                return "progressive-center-out";
            case GaussianSplatFoveatedAdaptationMode::eProgressiveGreedy:
                return "progressive-greedy";
        }
        return "unknown";
    }

    inline std::string_view foveatedDistributionLabel(const GaussianSplatFoveatedDistribution distribution)
    {
        switch (distribution)
        {
            case GaussianSplatFoveatedDistribution::eHardRing:
                return "hard-ring";
            case GaussianSplatFoveatedDistribution::eSmoothstep:
                return "smoothstep";
            case GaussianSplatFoveatedDistribution::eGaussian:
                return "gaussian";
            case GaussianSplatFoveatedDistribution::eExponential:
                return "exponential";
            case GaussianSplatFoveatedDistribution::eInversePower:
                return "inverse-power";
            case GaussianSplatFoveatedDistribution::eLogPolar:
                return "log-polar";
            case GaussianSplatFoveatedDistribution::eContinuousScheduler:
                return "continuous";
            case GaussianSplatFoveatedDistribution::eFoveaProtectedContinuous:
                return "fovea-protected-continuous";
            case GaussianSplatFoveatedDistribution::eCortical:
                return "cortical";
            case GaussianSplatFoveatedDistribution::eConeDensityFitted:
                return "cone-density-fitted";
        }
        return "unknown";
    }

    inline std::string_view foveatedCoverageGuardModeLabel(const GaussianSplatFoveatedCoverageGuardMode mode)
    {
        switch (mode)
        {
            case GaussianSplatFoveatedCoverageGuardMode::eOff:
                return "off";
            case GaussianSplatFoveatedCoverageGuardMode::eGlobal:
                return "global";
            case GaussianSplatFoveatedCoverageGuardMode::eLocalBounded:
                return "local-bounded";
            case GaussianSplatFoveatedCoverageGuardMode::eRiskTriggered:
                return "risk-triggered";
        }
        return "unknown";
    }

    inline std::string_view shLodGuardModeLabel(const GaussianSplatShLodGuardMode mode)
    {
        switch (mode)
        {
            case GaussianSplatShLodGuardMode::eOff:
                return "off";
            case GaussianSplatShLodGuardMode::eEnergy:
                return "energy";
            case GaussianSplatShLodGuardMode::eProjectedCost:
                return "projected-cost";
            case GaussianSplatShLodGuardMode::eEnergyProjectedCost:
                return "energy-projected-cost";
        }
        return "unknown";
    }

    inline std::string_view shStorageLayoutLabel(const GaussianSplatShStorageLayout layout)
    {
        switch (layout)
        {
            case GaussianSplatShStorageLayout::eMonolithic:
                return "monolithic";
            case GaussianSplatShStorageLayout::eSplitBands:
                return "split-bands";
        }
        return "unknown";
    }

    inline double sumScopeMs(const std::vector<RuntimeProfiler::ScopeNode>& scopes,
                             const std::string_view                         needle,
                             const bool                                     useGpuMs)
    {
        double total = 0.0;
        bool   found = false;

        for (const auto& scope : scopes)
        {
            if (scope.name.find(needle) == std::string::npos)
                continue;

            const double ms = useGpuMs ? scope.gpuTotalMs : scope.totalMs;
            if (ms < 0.0)
                continue;

            total += ms;
            found = true;
        }

        return found ? total : -1.0;
    }

    inline GaussianBenchmarkSample makeBenchmarkSample(const uint32_t                         sampleIndex,
                                                       const fsec                             dt,
                                                       const RuntimeProfiler::FrameStats&     frame,
                                                       const GaussianSplatFrameStats&         gaussian,
                                                       const uint32_t                         scriptedViewIndex = UINT32_MAX,
                                                       const uint32_t                         scriptedGazeIndex = UINT32_MAX,
                                                       const uint32_t                         isGazeMoving = 0,
                                                       const uint32_t                         gazeJumpEvent = 0,
                                                       const int32_t                          transitionWindowId = -1)
    {
        GaussianBenchmarkSample sample {};
        sample.sampleIndex           = sampleIndex;
        sample.frameIndex            = frame.frameIndex;
        sample.scriptedViewIndex     = scriptedViewIndex;
        sample.scriptedGazeIndex     = scriptedGazeIndex;
        sample.isGazeMoving          = isGazeMoving;
        sample.gazeJumpEvent         = gazeJumpEvent;
        sample.transitionWindowId    = transitionWindowId;
        sample.dtMs                  = static_cast<double>(dt.count()) * 1000.0;
        sample.cpuFrameMs            = frame.cpuFrameMs;
        sample.cpuRenderMs           = frame.cpuRenderMs;
        sample.gpuFrameMs            = frame.gpuFrameMs;
        sample.drawCalls             = frame.drawCalls;
        sample.dispatchCalls         = frame.dispatchCalls;
        sample.copyOps               = frame.copyOps;
        sample.updateOps             = frame.updateOps;
        sample.gpuScopeResolvedCount = frame.gpuScopeResolvedCount;
        sample.gpuScopeTokenCount    = frame.gpuScopeTokenCount;

        sample.baselineMode                     = gaussian.baselineMode;
        sample.foveatedRenderMode               = gaussian.foveatedRenderMode;
        sample.lodBudgetMode                    = gaussian.lodBudgetMode;
        sample.foveatedAdaptationMode           = gaussian.foveatedAdaptationMode;
        sample.foveatedDistribution             = gaussian.foveatedDistribution;
        sample.lodBudgetEnabled                 = gaussian.lodBudgetEnabled;
        sample.foveatedClodEnabled              = gaussian.foveatedClodEnabled;
        sample.foveatedLayeredCompositeEnabled  = gaussian.foveatedLayeredCompositeEnabled;
        sample.foveatedCoverageCompensationEnabled =
            gaussian.foveatedCoverageCompensationEnabled;
        sample.coverageGuardMode                = gaussian.foveatedCoverageGuardMode;
        sample.directPrefix                     = gaussian.directPrefix;
        sample.lodBudget                        = gaussian.lodBudget;
        sample.costBudget                       = gaussian.costBudget;
        sample.actualProjectedCost              = gaussian.actualProjectedCost;
        sample.costOvershoot                    = gaussian.costOvershoot;
        sample.projectedCostBudgetRatio         = gaussian.projectedCostBudgetRatio;
        sample.projectedCostChunkSize           = gaussian.projectedCostChunkSize;
        sample.foveatedCoverageBinGridX         = gaussian.foveatedCoverageBinGridX;
        sample.foveatedCoverageBinGridY         = gaussian.foveatedCoverageBinGridY;
        sample.foveatedCoverageBinQuotaRatio    = gaussian.foveatedCoverageBinQuotaRatio;
        sample.foveatedCoverageBinPrefixRatio   = gaussian.foveatedCoverageBinPrefixRatio;
        sample.projectedCostBuildMode           = gaussian.projectedCostBuildMode;
        sample.projectedCostSemanticDiffEnabled = gaussian.projectedCostSemanticDiffEnabled;
        sample.selectedCostChunks               = gaussian.selectedCostChunks;
        sample.a4ProjectedCostSelectionCpuMs    = gaussian.a4ProjectedCostSelectionCpuMs;
        sample.a4CostEstimationCpuMs            = gaussian.a4CostEstimationCpuMs;
        sample.a4ChunkSelectionCpuMs            = gaussian.a4ChunkSelectionCpuMs;
        sample.a4ChunkStopCpuMs                 = gaussian.a4ChunkStopCpuMs;
        sample.a4GuardAnalysisCpuMs             = gaussian.a4GuardAnalysisCpuMs;
        sample.a4TotalSelectionCpuMs            = gaussian.a4TotalSelectionCpuMs;
        sample.a4NumChunksScanned               = gaussian.a4NumChunksScanned;
        sample.a4NumGaussiansOrChunksConsidered = gaussian.a4NumGaussiansOrChunksConsidered;
        sample.a4ProjectedSampleCacheHit        = gaussian.a4ProjectedSampleCacheHit;
        sample.a4ProjectedSampleCacheMiss       = gaussian.a4ProjectedSampleCacheMiss;
        sample.a4ProjectedSampleReused          = gaussian.a4ProjectedSampleReused;
        sample.a4NumProjectedSamplesBuilt       = gaussian.a4NumProjectedSamplesBuilt;
        sample.a4NumProjectedSamplesReused      = gaussian.a4NumProjectedSamplesReused;
        sample.a4CacheEntryCount                = gaussian.a4CacheEntryCount;
        sample.a4CacheLookupCount               = gaussian.a4CacheLookupCount;
        sample.a4CacheHitCount                  = gaussian.a4CacheHitCount;
        sample.a4CacheMissCount                 = gaussian.a4CacheMissCount;
        sample.a4CacheFullMissCount             = gaussian.a4CacheFullMissCount;
        sample.a4CacheBuildCount                = gaussian.a4CacheBuildCount;
        sample.a4CacheEvictionCount             = gaussian.a4CacheEvictionCount;
        sample.a4SampleCacheKeyHash             = gaussian.a4SampleCacheKeyHash;
        sample.a4ChunkCacheHitCount             = gaussian.a4ChunkCacheHitCount;
        sample.a4ChunkCacheMissCount            = gaussian.a4ChunkCacheMissCount;
        sample.a4ProjectedSamplesBuilt          = gaussian.a4ProjectedSamplesBuilt;
        sample.a4ProjectedSamplesReused         = gaussian.a4ProjectedSamplesReused;
        sample.a4ChunkAggregationCpuMs          = gaussian.a4ChunkAggregationCpuMs;
        sample.a4CacheLookupCpuMs               = gaussian.a4CacheLookupCpuMs;
        sample.a4CacheHitRate                   = gaussian.a4CacheHitRate;
        sample.a4GpuCostBuildEnabled            = gaussian.a4GpuCostBuildEnabled;
        sample.a4GpuCostBuildGpuMs              = gaussian.a4GpuCostBuildGpuMs;
        sample.a4GpuCostReadbackCpuMs           = gaussian.a4GpuCostReadbackCpuMs;
        sample.a4GpuCostChunkCount              = gaussian.a4GpuCostChunkCount;
        sample.a4GpuCostValid                   = gaussian.a4GpuCostValid;
        sample.a4CpuFallbackUsed                = gaussian.a4CpuFallbackUsed;
        sample.a4GpuCpuCostL1Error              = gaussian.a4GpuCpuCostL1Error;
        sample.a4GpuCpuCostMaxError             = gaussian.a4GpuCpuCostMaxError;
        sample.a4GpuCpuSelectedRatioDelta       = gaussian.a4GpuCpuSelectedRatioDelta;
        sample.a4GpuCpuActualCostDelta          = gaussian.a4GpuCpuActualCostDelta;
        sample.a4GpuCpuOvershootDelta           = gaussian.a4GpuCpuOvershootDelta;
        sample.gazeX                            = gaussian.foveatedGaze.x;
        sample.gazeY                            = gaussian.foveatedGaze.y;
        sample.foveaLod                         = gaussian.foveatedRingLevels.x;
        sample.midLod                           = gaussian.foveatedRingLevels.y;
        sample.outerLod                         = gaussian.foveatedRingLevels.z;
        sample.foveaDegrees                     = gaussian.foveatedRingDegrees.x;
        sample.midDegrees                       = gaussian.foveatedRingDegrees.y;
        sample.continuousTheta0Degrees          = gaussian.foveatedContinuousTheta0Degrees;
        sample.continuousAlpha                  = gaussian.foveatedContinuousAlpha;
        sample.continuousMinLevel               = gaussian.foveatedContinuousMinLevel;
        sample.foveaResolutionScale             = gaussian.foveatedResolutionScales.x;
        sample.midResolutionScale               = gaussian.foveatedResolutionScales.y;
        sample.outerResolutionScale             = gaussian.foveatedResolutionScales.z;
        sample.targetFrameMs                    = gaussian.foveatedTargetFrameMs;
        sample.coverageProtectionDegrees        = gaussian.foveatedCoverageProtectionDegrees;
        sample.coverageGuardBudgetRatio         = gaussian.foveatedCoverageGuardBudgetRatio;
        sample.coverageGuardCenterMin           = gaussian.foveatedCoverageGuardMinLevels.x;
        sample.coverageGuardTransitionMin       = gaussian.foveatedCoverageGuardMinLevels.y;
        sample.coverageGuardPeripheryMin        = gaussian.foveatedCoverageGuardMinLevels.z;
        sample.coverageGuardMaxAdds             = gaussian.foveatedCoverageGuardMaxAdds;
        sample.guardMaxAddsEffective            = gaussian.foveatedCoverageGuardMaxAddsEffective;
        sample.coverageGuardModeEnabled         = gaussian.foveatedCoverageGuardModeEnabled;
        sample.coverageGuardRiskSectors         = gaussian.foveatedCoverageGuardRiskSectors;
        sample.coverageGuardRiskDetected        = gaussian.foveatedCoverageGuardRiskDetected;
        sample.coverageGuardActive              = gaussian.foveatedCoverageGuardActive;
        sample.coverageGuardRepairActive        = gaussian.foveatedCoverageGuardRepairActive;
        sample.coverageGuardAddedAny            = gaussian.foveatedCoverageGuardAddedAny;
        sample.coverageGuardAddedCount          = gaussian.foveatedCoverageGuardAddedCount;
        sample.coverageGuardAddedCost           = gaussian.foveatedCoverageGuardAddedCost;
        sample.coverageGuardBudgetCap           = gaussian.foveatedCoverageGuardBudgetCap;
        sample.riskSectorCountTotal             = gaussian.foveatedCoverageRiskSectorCountTotal;
        sample.riskSectorCountCenter            = gaussian.foveatedCoverageRiskSectorCountCenter;
        sample.riskSectorCountMid               = gaussian.foveatedCoverageRiskSectorCountMid;
        sample.riskSectorCountOuter             = gaussian.foveatedCoverageRiskSectorCountOuter;
        sample.riskSectorActiveCenter           = gaussian.foveatedCoverageRiskSectorActiveCenter;
        sample.riskSectorActiveMid              = gaussian.foveatedCoverageRiskSectorActiveMid;
        sample.riskSectorActiveOuter            = gaussian.foveatedCoverageRiskSectorActiveOuter;
        sample.coverageFailureBeforeGuard       = gaussian.foveatedCoverageFailureBeforeGuard;
        sample.coverageFailureAfterGuard        = gaussian.foveatedCoverageFailureAfterGuard;
        sample.coverageFailureBeforeCenter      = gaussian.foveatedCoverageFailureBeforeCenter;
        sample.coverageFailureBeforeMid         = gaussian.foveatedCoverageFailureBeforeMid;
        sample.coverageFailureBeforeOuter       = gaussian.foveatedCoverageFailureBeforeOuter;
        sample.coverageFailureAfterCenter       = gaussian.foveatedCoverageFailureAfterCenter;
        sample.coverageFailureAfterMid          = gaussian.foveatedCoverageFailureAfterMid;
        sample.coverageFailureAfterOuter        = gaussian.foveatedCoverageFailureAfterOuter;
        sample.coverageScoreBeforeGuard         = gaussian.foveatedCoverageScoreBeforeGuard;
        sample.coverageScoreAfterGuard          = gaussian.foveatedCoverageScoreAfterGuard;
        sample.guardAddedCountCenter            = gaussian.foveatedCoverageGuardAddedCountCenter;
        sample.guardAddedCountMid               = gaussian.foveatedCoverageGuardAddedCountMid;
        sample.guardAddedCountOuter             = gaussian.foveatedCoverageGuardAddedCountOuter;
        sample.guardAddedCostCenter             = gaussian.foveatedCoverageGuardAddedCostCenter;
        sample.guardAddedCostMid                = gaussian.foveatedCoverageGuardAddedCostMid;
        sample.guardAddedCostOuter              = gaussian.foveatedCoverageGuardAddedCostOuter;
        sample.guardRepairCapHit                = gaussian.foveatedCoverageGuardRepairCapHit;
        sample.guardRepairCappedByMaxAdds       = gaussian.foveatedCoverageGuardRepairCappedByMaxAdds;
        sample.guardRepairCappedByCost          = gaussian.foveatedCoverageGuardRepairCappedByCost;
        sample.guardRepairNoCandidate           = gaussian.foveatedCoverageGuardRepairNoCandidate;
        sample.guardCandidateCount              = gaussian.foveatedCoverageGuardCandidateCount;
        sample.guardCandidateCountCenter        = gaussian.foveatedCoverageGuardCandidateCountCenter;
        sample.guardCandidateCountMid           = gaussian.foveatedCoverageGuardCandidateCountMid;
        sample.guardCandidateCountOuter         = gaussian.foveatedCoverageGuardCandidateCountOuter;
        sample.guardTrueCandidateCount          = gaussian.foveatedCoverageGuardTrueCandidateCount;
        sample.guardCandidateCountTruncated     = gaussian.foveatedCoverageGuardCandidateCountTruncated;
        sample.guardRequiredExtraCountEstimate  = gaussian.foveatedCoverageGuardRequiredExtraCountEstimate;
        sample.guardRequiredExtraAreaPx         = gaussian.foveatedCoverageGuardRequiredExtraAreaPx;
        sample.guardDeficitAreaBefore           = gaussian.foveatedCoverageGuardDeficitAreaBefore;
        sample.guardDeficitAreaAfter            = gaussian.foveatedCoverageGuardDeficitAreaAfter;
        sample.guardAddedCountToFailingSector   = gaussian.foveatedCoverageGuardAddedCountToFailingSector;
        sample.guardAddedCostToFailingSector    = gaussian.foveatedCoverageGuardAddedCostToFailingSector;
        sample.guardFailingSectorId             = gaussian.foveatedCoverageGuardFailingSectorId;
        sample.guardRepairSectorMatchRate       = gaussian.foveatedCoverageGuardRepairSectorMatchRate;
        sample.guardAnalysisCpuMs               = gaussian.foveatedCoverageGuardAnalysisCpuMs;
        sample.guardProjectedSampleBuildCpuMs   =
            gaussian.foveatedCoverageGuardProjectedSampleBuildCpuMs;
        sample.guardCoverageAreaCpuMs           = gaussian.foveatedCoverageGuardCoverageAreaCpuMs;
        sample.guardCandidateScanCpuMs          = gaussian.foveatedCoverageGuardCandidateScanCpuMs;
        sample.guardRegionRebuildCpuMs          = gaussian.foveatedCoverageGuardRegionRebuildCpuMs;
        sample.guardCacheFullMissCount          = gaussian.foveatedCoverageGuardCacheFullMissCount;
        sample.guardCacheHitCount               = gaussian.foveatedCoverageGuardCacheHitCount;
        sample.guardCacheMissCount              = gaussian.foveatedCoverageGuardCacheMissCount;
        sample.guardGeometryCacheHitCount       = gaussian.foveatedCoverageGuardGeometryCacheHitCount;
        sample.guardGeometryCacheMissCount      = gaussian.foveatedCoverageGuardGeometryCacheMissCount;
        sample.guardGeometryCacheFullMissCount  = gaussian.foveatedCoverageGuardGeometryCacheFullMissCount;
        sample.guardGeometrySamplesBuilt        = gaussian.foveatedCoverageGuardGeometrySamplesBuilt;
        sample.guardGeometrySamplesReused       = gaussian.foveatedCoverageGuardGeometrySamplesReused;
        sample.schedulerShrinkEvent             = gaussian.schedulerShrinkEvent;
        sample.schedulerGrowEvent               = gaussian.schedulerGrowEvent;
        sample.schedulerBudgetBefore            = gaussian.schedulerBudgetBefore;
        sample.schedulerBudgetAfter             = gaussian.schedulerBudgetAfter;
        sample.schedulerP95Ema                  = gaussian.schedulerP95Ema;
        sample.schedulerP95Window               = gaussian.schedulerP95Window;
        sample.schedulerTargetMs                = gaussian.schedulerTargetMs;
        sample.schedulerMarginLow               = gaussian.schedulerMarginLow;
        sample.schedulerMarginHigh              = gaussian.schedulerMarginHigh;
        sample.projectedCostBudgetRatioBeforeGuard = gaussian.projectedCostBudgetRatioBeforeGuard;
        sample.projectedCostBudgetRatioAfterScheduler = gaussian.projectedCostBudgetRatioAfterScheduler;
        sample.selectedRatioBeforeGuard         = gaussian.selectedRatioBeforeGuard;
        sample.selectedRatioAfterGuard          = gaussian.selectedRatioAfterGuard;
        sample.guardAddedCostRatioToBudget      = gaussian.foveatedCoverageGuardAddedCostRatioToBudget;
        sample.guardAddedCountRatioToSelected   = gaussian.foveatedCoverageGuardAddedCountRatioToSelected;
        sample.temporalHysteresisEnabled        = gaussian.foveatedTemporalHysteresisEnabled;
        sample.boundarySmoothingEnabled         = gaussian.foveatedBoundarySmoothingEnabled;
        sample.temporalResidencyFrames          = gaussian.foveatedTemporalResidencyFrames;
        sample.temporalHysteresisRatio          = gaussian.foveatedTemporalHysteresisRatio;
        sample.boundarySmoothingRatio           = gaussian.foveatedBoundarySmoothingRatio;
        sample.temporalPeripheralScale          = gaussian.foveatedTemporalPeripheralScale;
        sample.shLodEnabled                     = gaussian.foveatedShLodEnabled;
        sample.shSmoothSuppressionEnabled       = gaussian.foveatedShSmoothSuppressionEnabled;
        sample.peripheralTemporalFilterEnabled  = gaussian.peripheralTemporalFilterEnabled;
        sample.peripheralTemporalFilterMidDegrees = gaussian.peripheralTemporalFilterMidDegrees;
        sample.peripheralTemporalFilterOuterDegrees = gaussian.peripheralTemporalFilterOuterDegrees;
        sample.peripheralTemporalFilterLambdaScale = gaussian.peripheralTemporalFilterLambdaScale;
        sample.peripheralTemporalFilterRejectionThreshold =
            gaussian.peripheralTemporalFilterRejectionThreshold;
        sample.peripheralTemporalFilterClampRadius = gaussian.peripheralTemporalFilterClampRadius;
        sample.shDegreeCenter                   = gaussian.shDegreeCenter;
        sample.shDegreeMid                      = gaussian.shDegreeMid;
        sample.shDegreeOuter                    = gaussian.shDegreeOuter;
        sample.shSmoothL1StartDegrees           = gaussian.shSmoothL1StartDegrees;
        sample.shSmoothL1EndDegrees             = gaussian.shSmoothL1EndDegrees;
        sample.shSmoothL2StartDegrees           = gaussian.shSmoothL2StartDegrees;
        sample.shSmoothL2EndDegrees             = gaussian.shSmoothL2EndDegrees;
        sample.shSmoothL3StartDegrees           = gaussian.shSmoothL3StartDegrees;
        sample.shSmoothL3EndDegrees             = gaussian.shSmoothL3EndDegrees;
        sample.estimatedShAcCoeffReads          = gaussian.estimatedShAcCoeffReads;
        sample.estimatedShAcReadReductionVsDegree3 =
            gaussian.estimatedShAcReadReductionVsDegree3;
        sample.shStorageLayout                    = gaussian.shStorageLayout;
        sample.shBandL1ReadsEst                   = gaussian.shBandL1ReadsEst;
        sample.shBandL2ReadsEst                   = gaussian.shBandL2ReadsEst;
        sample.shBandL3ReadsEst                   = gaussian.shBandL3ReadsEst;
        sample.shBandBytesEst                     = gaussian.shBandBytesEst;
        sample.shBandBytesReductionVsMonolithic   = gaussian.shBandBytesReductionVsMonolithic;
        sample.shStorageMetadataBytesEst          = gaussian.shStorageMetadataBytesEst;
        sample.shEnergyMetadataAvailable          = gaussian.shEnergyMetadataAvailable;
        sample.shEnergyMeanAfter0                 = gaussian.shEnergyMeanAfter0;
        sample.shEnergyMeanAfter1                 = gaussian.shEnergyMeanAfter1;
        sample.shEnergyMeanAfter2                 = gaussian.shEnergyMeanAfter2;
        sample.shEnergyP95After0                  = gaussian.shEnergyP95After0;
        sample.shEnergyP95After1                  = gaussian.shEnergyP95After1;
        sample.shEnergyP95After2                  = gaussian.shEnergyP95After2;
        sample.shGuardMode                        = gaussian.shGuardMode;
        sample.shGuardRaisedCount                 = gaussian.shGuardRaisedCount;
        sample.shGuardRaisedRatio                 = gaussian.shGuardRaisedRatio;
        sample.shGuardRecoveredAcReads            = gaussian.shGuardRecoveredAcReads;
        sample.shGuardSavedAcReadsAfterGuard      = gaussian.shGuardSavedAcReadsAfterGuard;
        sample.shGuardThresholdMid                = gaussian.shGuardThresholdMid;
        sample.shGuardThresholdHigh               = gaussian.shGuardThresholdHigh;
        sample.shDegreeChangedCount               = gaussian.shDegreeChangedCount;
        sample.shDegreeChangedRatio               = gaussian.shDegreeChangedRatio;
        sample.shPopEnergyProxy                   = gaussian.shPopEnergyProxy;
        sample.shPopEnergyFovea                   = gaussian.shPopEnergyFovea;
        sample.shPopEnergyMid                     = gaussian.shPopEnergyMid;
        sample.shPopEnergyPeriphery               = gaussian.shPopEnergyPeriphery;
        sample.shGuardRaiseCount                  = gaussian.shPopGuardRaiseCount;
        sample.shDelayedDowngradeCount            = gaussian.shPopDelayedDowngradeCount;
        sample.gazeAnchorCrossfadeEnabled        = gaussian.gazeAnchorCrossfadeEnabled;
        sample.gazeAnchorGpuCrossfadeEnabled     = gaussian.gazeAnchorGpuCrossfadeEnabled;
        sample.gazeAnchorDeadbandDegrees         = gaussian.gazeAnchorDeadbandDegrees;
        sample.gazeAnchorFadeFrames              = gaussian.gazeAnchorFadeFrames;
        sample.gazeAnchorMinUpdateFrames         = gaussian.gazeAnchorMinUpdateFrames;
        sample.gazeAnchorTransitionBudgetRatio   = gaussian.gazeAnchorTransitionBudgetRatio;
        sample.gazeAnchorImmediateFoveaFill      = gaussian.gazeAnchorImmediateFoveaFill;
        sample.gazeAnchorContributionGuard       = gaussian.gazeAnchorContributionGuard;
        sample.gazeAnchorDebugLogEnabled         = gaussian.gazeAnchorDebugLogEnabled;
        sample.gazeAnchorAnchorX                 = gaussian.gazeAnchorAnchorX;
        sample.gazeAnchorAnchorY                 = gaussian.gazeAnchorAnchorY;
        sample.gazeAnchorNewAnchorX              = gaussian.gazeAnchorNewAnchorX;
        sample.gazeAnchorNewAnchorY              = gaussian.gazeAnchorNewAnchorY;
        sample.gazeAnchorDistanceDegrees         = gaussian.gazeAnchorDistanceDegrees;
        sample.gazeAnchorFramesSinceUpdate       = gaussian.gazeAnchorFramesSinceUpdate;
        sample.gazeAnchorFadePhase               = gaussian.gazeAnchorFadePhase;
        sample.gazeAnchorUpdateEvent             = gaussian.gazeAnchorUpdateEvent;
        sample.gazeAnchorUpdateEventCount        = gaussian.gazeAnchorUpdateEventCount;
        sample.gazeAnchorDeadbandViolationCount  = gaussian.gazeAnchorDeadbandViolationCount;
        sample.gazeAnchorUnionCount              = gaussian.gazeAnchorUnionCount;
        sample.gazeAnchorUnionCountRatio         = gaussian.gazeAnchorUnionCountRatio;
        sample.gazeAnchorSharedCount             = gaussian.gazeAnchorSharedCount;
        sample.gazeAnchorSharedRatio             = gaussian.gazeAnchorSharedRatio;
        sample.gazeAnchorOldOnlyCount            = gaussian.gazeAnchorOldOnlyCount;
        sample.gazeAnchorNewOnlyCount            = gaussian.gazeAnchorNewOnlyCount;
        sample.gazeAnchorFadeActiveCount         = gaussian.gazeAnchorFadeActiveCount;
        sample.gazeAnchorImmediateFoveaFillCount = gaussian.gazeAnchorImmediateFoveaFillCount;
        sample.gazeAnchorTransitionBudgetTargetCount =
            gaussian.gazeAnchorTransitionBudgetTargetCount;
        sample.gazeAnchorTransitionBudgetCappedCount =
            gaussian.gazeAnchorTransitionBudgetCappedCount;
        sample.gazeAnchorRenderedOldOnlyCount    = gaussian.gazeAnchorRenderedOldOnlyCount;
        sample.gazeAnchorRenderedNewOnlyCount    = gaussian.gazeAnchorRenderedNewOnlyCount;
        sample.gazeAnchorDroppedOldOnlyCount     = gaussian.gazeAnchorDroppedOldOnlyCount;
        sample.gazeAnchorDroppedNewOnlyCount     = gaussian.gazeAnchorDroppedNewOnlyCount;
        sample.gazeAnchorHighContributionProtectedCount =
            gaussian.gazeAnchorHighContributionProtectedCount;
        sample.gazeAnchorWeightedTvChurn         = gaussian.gazeAnchorWeightedTvChurn;
        sample.eccStochasticTransitionEnabled    = gaussian.eccStochasticTransitionEnabled;
        sample.eccStochasticFadeFrames           = gaussian.eccStochasticFadeFrames;
        sample.eccStochasticProtectOldFoveaDegrees = gaussian.eccStochasticProtectOldFoveaDegrees;
        sample.eccStochasticProtectNewFoveaDegrees = gaussian.eccStochasticProtectNewFoveaDegrees;
        sample.eccStochasticBoundaryBandDegrees  = gaussian.eccStochasticBoundaryBandDegrees;
        sample.eccStochasticMinPDelta            = gaussian.eccStochasticMinPDelta;
        sample.eccStochasticContributionGuard    = gaussian.eccStochasticContributionGuard;
        sample.eccStochasticDebugLogEnabled      = gaussian.eccStochasticDebugLogEnabled;
        sample.eccStochasticOldGazeX             = gaussian.eccStochasticOldGazeX;
        sample.eccStochasticOldGazeY             = gaussian.eccStochasticOldGazeY;
        sample.eccStochasticNewGazeX             = gaussian.eccStochasticNewGazeX;
        sample.eccStochasticNewGazeY             = gaussian.eccStochasticNewGazeY;
        sample.eccStochasticFadePhase            = gaussian.eccStochasticFadePhase;
        sample.eccStochasticFramesSinceUpdate    = gaussian.eccStochasticFramesSinceUpdate;
        sample.eccStochasticUpdateEvent          = gaussian.eccStochasticUpdateEvent;
        sample.eccStochasticUpdateEventCount     = gaussian.eccStochasticUpdateEventCount;
        sample.eccStochasticCounterReadbackValid = gaussian.eccStochasticCounterReadbackValid;
        sample.eccStochasticCounterReadbackCpuMs = gaussian.eccStochasticCounterReadbackCpuMs;
        sample.eccStochasticTotalCandidatesSeen  = gaussian.eccStochasticTotalCandidatesSeen;
        sample.eccStochasticBaseNewSelectedCount = gaussian.eccStochasticBaseNewSelectedCount;
        sample.eccStochasticEffectiveVisibleAfterEcsptCount =
            gaussian.eccStochasticEffectiveVisibleAfterEcsptCount;
        sample.eccStochasticSharedCount           = gaussian.eccStochasticSharedCount;
        sample.eccStochasticUpgradeCount          = gaussian.eccStochasticUpgradeCount;
        sample.eccStochasticDowngradeCount        = gaussian.eccStochasticDowngradeCount;
        sample.eccStochasticProtectedDowngradeCount =
            gaussian.eccStochasticProtectedDowngradeCount;
        sample.eccStochasticDroppedDowngradeCount =
            gaussian.eccStochasticDroppedDowngradeCount;
        sample.eccStochasticOldOnlyFadeVisibleCount =
            gaussian.eccStochasticOldOnlyFadeVisibleCount;
        sample.eccStochasticImmediateNewFoveaCount =
            gaussian.eccStochasticImmediateNewFoveaCount;
        sample.eccStochasticBoundaryProtectedCount =
            gaussian.eccStochasticBoundaryProtectedCount;
        sample.eccStochasticContributionGuardProtectedCount =
            gaussian.eccStochasticContributionGuardProtectedCount;
        sample.eccStochasticZeroWeightDiscardCount =
            gaussian.eccStochasticZeroWeightDiscardCount;
        sample.eccStochasticMinPDeltaDiscardCount =
            gaussian.eccStochasticMinPDeltaDiscardCount;
        sample.eccStochasticFarPeripheryHardDropCount =
            gaussian.eccStochasticFarPeripheryHardDropCount;
        sample.stableOpticalDepthAlphaClampedCount =
            gaussian.stableOpticalDepthAlphaClampedCount;
        sample.coverageStableReleaseFloorActiveCount =
            gaussian.coverageStableReleaseFloorActiveCount;
        sample.coverageStableReleaseFloorRaisedCount =
            gaussian.coverageStableReleaseFloorRaisedCount;
        sample.coverageStableReleaseHeldCount =
            gaussian.coverageStableReleaseHeldCount;
        sample.coverageStableReleaseDroppedCount =
            gaussian.coverageStableReleaseDroppedCount;
        sample.coverageStableReleaseOldOnlyCount =
            gaussian.coverageStableReleaseOldOnlyCount;
        sample.coverageStableReleaseFloorSampleCount =
            gaussian.coverageStableReleaseFloorSampleCount;
        sample.coverageStableReleasePFloorMean =
            gaussian.coverageStableReleasePFloorMean;
        sample.coverageStableReleasePFloorMax =
            gaussian.coverageStableReleasePFloorMax;
        sample.coverageStableReleaseDAlphaMean =
            gaussian.coverageStableReleaseDAlphaMean;
        sample.coverageStableReleaseNEffMean =
            gaussian.coverageStableReleaseNEffMean;
        sample.coverageStableReleaseTileTotalCount =
            gaussian.coverageStableReleaseTileTotalCount;
        sample.coverageStableReleaseTileNonEmptyCount =
            gaussian.coverageStableReleaseTileNonEmptyCount;
        sample.coverageStableReleaseTileEmptyCount =
            gaussian.coverageStableReleaseTileEmptyCount;
        sample.coverageStableReleaseDAlphaMeanAll =
            gaussian.coverageStableReleaseDAlphaMeanAll;
        sample.coverageStableReleaseDAlphaMeanNonEmpty =
            gaussian.coverageStableReleaseDAlphaMeanNonEmpty;
        sample.coverageStableReleaseDAlphaP50NonEmpty =
            gaussian.coverageStableReleaseDAlphaP50NonEmpty;
        sample.coverageStableReleaseDAlphaP90NonEmpty =
            gaussian.coverageStableReleaseDAlphaP90NonEmpty;
        sample.coverageStableReleaseDAlphaP95NonEmpty =
            gaussian.coverageStableReleaseDAlphaP95NonEmpty;
        sample.coverageStableReleaseNEffMeanNonEmpty =
            gaussian.coverageStableReleaseNEffMeanNonEmpty;
        sample.coverageStableReleaseNEffP50NonEmpty =
            gaussian.coverageStableReleaseNEffP50NonEmpty;
        sample.coverageStableReleaseNEffP90NonEmpty =
            gaussian.coverageStableReleaseNEffP90NonEmpty;
        sample.coverageStableReleaseNEffP95NonEmpty =
            gaussian.coverageStableReleaseNEffP95NonEmpty;
        sample.coverageStableReleasePFloorMeanAll =
            gaussian.coverageStableReleasePFloorMeanAll;
        sample.coverageStableReleasePFloorMeanNonEmpty =
            gaussian.coverageStableReleasePFloorMeanNonEmpty;
        sample.coverageStableReleasePFloorP90NonEmpty =
            gaussian.coverageStableReleasePFloorP90NonEmpty;
        sample.coverageStableReleasePFloorP95NonEmpty =
            gaussian.coverageStableReleasePFloorP95NonEmpty;
        sample.coverageStableReleasePLpMean =
            gaussian.coverageStableReleasePLpMean;
        sample.coverageStableReleasePStaticMean =
            gaussian.coverageStableReleasePStaticMean;
        sample.coverageStableReleaseFloorActiveFractionVsCandidates =
            gaussian.coverageStableReleaseFloorActiveFractionVsCandidates;
        sample.coverageStableReleaseFloorRaisedFractionVsEffectiveVisible =
            gaussian.coverageStableReleaseFloorRaisedFractionVsEffectiveVisible;
        sample.coverageStableReleaseDeltaSum =
            gaussian.coverageStableReleaseDeltaSum;
        sample.coverageStableReleaseDeltaMeanOverCandidates =
            gaussian.coverageStableReleaseDeltaMeanOverCandidates;
        sample.coverageStableReleaseDeltaMeanOverHeld =
            gaussian.coverageStableReleaseDeltaMeanOverHeld;
        sample.coverageStableReleaseHeldEffectiveRatio =
            gaussian.coverageStableReleaseHeldEffectiveRatio;
        sample.coverageStableReleaseHeldEffectiveCount =
            gaussian.coverageStableReleaseHeldEffectiveCount;
        sample.coverageStableReleaseHistoryResetCount =
            gaussian.coverageStableReleaseHistoryResetCount;
        sample.coverageStableReleaseReentryResetCount =
            gaussian.coverageStableReleaseReentryResetCount;
        sample.coverageStableReleaseEpsilonCutoffCount =
            gaussian.coverageStableReleaseEpsilonCutoffCount;
        sample.coverageStableReleaseSaturatedFloorEnabled =
            gaussian.coverageStableReleaseSaturatedFloorEnabled;
        sample.coverageStableReleaseFloorCellsSafeSaturated =
            gaussian.coverageStableReleaseFloorCellsSafeSaturated;
        sample.coverageStableReleaseFloorCellsUnsafe =
            gaussian.coverageStableReleaseFloorCellsUnsafe;
        sample.coverageStableReleaseSkippedFloorUpdateCount =
            gaussian.coverageStableReleaseSkippedFloorUpdateCount;
        sample.coverageStableReleaseDAlphaSafeThreshold =
            gaussian.coverageStableReleaseDAlphaSafeThreshold;
        sample.coverageStableReleaseNEffSafeThreshold =
            gaussian.coverageStableReleaseNEffSafeThreshold;
        sample.coverageStableReleaseFloorSaturationRatio =
            gaussian.coverageStableReleaseFloorSaturationRatio;
        sample.coverageStableReleaseCandidateAlphaMass =
            gaussian.coverageStableReleaseCandidateAlphaMass;
        sample.coverageStableReleaseRetainedAlphaMass =
            gaussian.coverageStableReleaseRetainedAlphaMass;
        sample.coverageTextureCurrentAlphaMass =
            gaussian.coverageTextureCurrentAlphaMass;
        sample.coverageTextureCurrentAlpha2Mass =
            gaussian.coverageTextureCurrentAlpha2Mass;
        sample.coverageTextureNonzeroTexelCount =
            gaussian.coverageTextureNonzeroTexelCount;
        sample.coverageTextureWidth = gaussian.coverageTextureWidth;
        sample.coverageTextureHeight = gaussian.coverageTextureHeight;
        sample.coverageTextureHistoryBeta = gaussian.coverageTextureHistoryBeta;
        sample.coverageTextureStrength = gaussian.coverageTextureStrength;
        sample.coverageTexturePFinalMean =
            gaussian.coverageTexturePFinalMean;
        sample.coverageTextureStableHashKeptCount =
            gaussian.coverageTextureStableHashKeptCount;
        sample.guideBeforeDiscardEnabled =
            gaussian.guideBeforeDiscardEnabled;
        sample.guideBeforeDiscardTexelCount =
            gaussian.guideBeforeDiscardTexelCount;
        sample.guideBeforeDiscardCoverageMean =
            gaussian.guideBeforeDiscardCoverageMean;
        sample.guideBeforeDiscardCoverageMin =
            gaussian.guideBeforeDiscardCoverageMin;
        sample.guideBeforeDiscardCoverageMax =
            gaussian.guideBeforeDiscardCoverageMax;
        sample.guideBeforeDiscardLowCoverageBoostedSplats =
            gaussian.guideBeforeDiscardLowCoverageBoostedSplats;
        sample.guideBeforeDiscardHighCoverageReducedSplats =
            gaussian.guideBeforeDiscardHighCoverageReducedSplats;
        sample.guideBeforeDiscardSelectedBeforeTileExpansion =
            gaussian.guideBeforeDiscardSelectedBeforeTileExpansion;
        sample.guideBeforeDiscardTileDuplicateCount =
            gaussian.guideBeforeDiscardTileDuplicateCount;
        sample.delayedGuideLogPolarEnabled =
            gaussian.delayedGuideLogPolarEnabled;
        sample.delayedGuideTemporalLogPolarEnabled =
            gaussian.delayedGuideTemporalLogPolarEnabled;
        sample.delayedGuideReleaseActiveCount =
            gaussian.delayedGuideReleaseActiveCount;
        sample.delayedGuidePHistoryNonzeroCount =
            gaussian.delayedGuidePHistoryNonzeroCount;
        sample.delayedGuidePTargetLessThanHistoryCount =
            gaussian.delayedGuidePTargetLessThanHistoryCount;
        sample.delayedGuidePHistoryBytes =
            gaussian.delayedGuidePHistoryBytes;
        sample.delayedGuideReleasedEffectiveVisibleCount =
            gaussian.delayedGuideReleasedEffectiveVisibleCount;
        sample.delayedGuideReleasedAlphaProxySum =
            gaussian.delayedGuideReleasedAlphaProxySum;
        sample.delayedGuideReleaseCapHitCount =
            gaussian.delayedGuideReleaseCapHitCount;
        sample.delayedGuideRiskProtectedCount =
            gaussian.delayedGuideRiskProtectedCount;
        sample.delayedGuideFootprintFastDecayCount =
            gaussian.delayedGuideFootprintFastDecayCount;
        sample.delayedGuideBaseTileDuplicateProxyCount =
            gaussian.delayedGuideBaseTileDuplicateProxyCount;
        sample.delayedGuideReleasedTileDuplicateProxyCount =
            gaussian.delayedGuideReleasedTileDuplicateProxyCount;
        sample.delayedGuideReleasedDuplicateProxyRatio =
            gaussian.delayedGuideReleasedDuplicateProxyRatio;
        sample.delayedGuideLargeFootprintReleasedCount =
            gaussian.delayedGuideLargeFootprintReleasedCount;
        sample.delayedGuideLargeFootprintReleasedTileDuplicateProxyCount =
            gaussian.delayedGuideLargeFootprintReleasedTileDuplicateProxyCount;
        sample.delayedGuideReleasedRadiusMean =
            gaussian.delayedGuideReleasedRadiusMean;
        sample.delayedGuideReleasedRadiusP95 =
            gaussian.delayedGuideReleasedRadiusP95;
        sample.delayedGuideReleasedRadiusMax =
            gaussian.delayedGuideReleasedRadiusMax;
        sample.delayedGuideHotspotTileIds =
            gaussian.delayedGuideHotspotTileIds;
        sample.delayedGuideHotspotBaseDuplicates =
            gaussian.delayedGuideHotspotBaseDuplicates;
        sample.delayedGuideHotspotReleasedDuplicates =
            gaussian.delayedGuideHotspotReleasedDuplicates;
        sample.eccStochasticEventBinCounters = gaussian.eccStochasticEventBinCounters;
        sample.shaderAntiPopMode                  = gaussian.shaderAntiPopMode;
        sample.shaderAntiPopEnabled               = gaussian.shaderAntiPopEnabled;
        sample.shaderAntiPopAlphaMultiplierBased  = gaussian.shaderAntiPopAlphaMultiplierBased;
        sample.shaderAntiPopAvoidsCpuSelectedSourceRebuild =
            gaussian.shaderAntiPopAvoidsCpuSelectedSourceRebuild;
        sample.shaderAntiPopStableCandidateSet    = gaussian.shaderAntiPopStableCandidateSet;
        sample.shaderAntiPopDirectPrefixStable    = gaussian.shaderAntiPopDirectPrefixStable;
        sample.shaderAntiPopDebugLogEnabled       = gaussian.shaderAntiPopDebugLogEnabled;
        sample.shaderAntiPopHashSeed              = gaussian.shaderAntiPopHashSeed;
        sample.shaderAntiPopRampWidth             = gaussian.shaderAntiPopRampWidth;
        sample.shaderAntiPopGuardThreshold        = gaussian.shaderAntiPopGuardThreshold;
        sample.shaderAntiPopGuardFloor            = gaussian.shaderAntiPopGuardFloor;
        sample.shaderAntiPopPKeepCurve            = gaussian.shaderAntiPopPKeepCurve;
        sample.shaderAntiPopPrefixRatio           = gaussian.shaderAntiPopPrefixRatio;
        sample.shaderAntiPopNormalizeMode         = gaussian.shaderAntiPopNormalizeMode;
        sample.shaderAntiPopNormalizeStrength     = gaussian.shaderAntiPopNormalizeStrength;
        sample.shaderAntiPopNormalizeClampMin     = gaussian.shaderAntiPopNormalizeClampMin;
        sample.shaderAntiPopNormalizeClampMax     = gaussian.shaderAntiPopNormalizeClampMax;
        sample.shaderAntiPopNormalizeFactor       = gaussian.shaderAntiPopNormalizeFactor;
        sample.shaderAntiPopCandidateCount        = gaussian.shaderAntiPopCandidateCount;
        sample.shaderAntiPopEffectiveNonzeroEstimate =
            gaussian.shaderAntiPopEffectiveNonzeroEstimate;
        sample.shaderAntiPopWeightMinEstimate     = gaussian.shaderAntiPopWeightMinEstimate;
        sample.shaderAntiPopWeightMeanEstimate    = gaussian.shaderAntiPopWeightMeanEstimate;
        sample.shaderAntiPopWeightMaxEstimate     = gaussian.shaderAntiPopWeightMaxEstimate;
        sample.shaderAntiPopAlphaMassEstimate     = gaussian.shaderAntiPopAlphaMassEstimate;
        sample.shaderAntiPopGuardProxyMode        = gaussian.shaderAntiPopGuardProxyMode;
        sample.splatAssets                      = gaussian.splatAssets;
        sample.drawRecords                      = gaussian.drawRecords;
        sample.totalSplats                      = gaussian.totalSplats;
        sample.preparedSplats                   = gaussian.preparedSplats;
        sample.foveaSplatBudget                 = gaussian.foveaSplatBudget;
        sample.midSplatBudget                   = gaussian.midSplatBudget;
        sample.outerSplatBudget                 = gaussian.outerSplatBudget;
        sample.maxVisibleSplatCap               = gaussian.maxVisibleSplatCap;
        sample.lodSelectedRawSplats             = gaussian.lodSelectedRawSplats;
        sample.selectedSourceIds                = gaussian.selectedSourceIds;
        sample.numTileIntersections             = gaussian.numTileIntersections;
        sample.sumProjectedAreaPx               = gaussian.sumProjectedAreaPx;
        sample.cleanTimingMode                  = gaussian.cleanTimingMode;
        sample.visibleSplats                    = gaussian.visibleSplats;
        sample.drawnSplats                      = gaussian.drawnSplats;
        sample.visibleInstant                   = gaussian.visibleInstant;
        sample.tileInstances                    = gaussian.tileInstances;
        sample.coveredPixelCount                = gaussian.coveredPixelCount;

        sample.cpuRenderFrameMs       = sumScopeMs(frame.cpuScopeTree, "RenderSystem::renderFrame", false);
        sample.cpuCookMs              = sumScopeMs(frame.cpuScopeTree, "RenderWorldCooker::cook", false);
        sample.cpuGpuSceneRebuildMs   = sumScopeMs(frame.cpuScopeTree, "GpuScene::rebuild", false);
        sample.cpuLodSelectionMs      = sumScopeMs(frame.cpuScopeTree, "GpuScene::gaussian_lod_selection", false);
        sample.cpuClodSelectionMs     = sumScopeMs(frame.cpuScopeTree, "GaussianCLOD::BuildPrefix", false);
        sample.cpuRawSelectionMs      = sumScopeMs(frame.cpuScopeTree, "GaussianSplat::BuildRawSelection", false);
        sample.cpuLodUploadMs         = sumScopeMs(frame.cpuScopeTree, "GaussianLOD::UploadSelected", false);
        sample.cpuFrameGraphBuildMs   = sumScopeMs(frame.cpuScopeTree, "FrameGraph::build", false);
        sample.cpuFrameGraphExecuteMs = sumScopeMs(frame.cpuScopeTree, "FrameGraph::execute", false);

        sample.gpuPreprocessPassMs = sumScopeMs(frame.gpuScopeTree, "GeneralGaussianSplatPreprocessPass", true);
        sample.gpuProjectCullMs =
            sumScopeMs(frame.gpuScopeTree, "GeneralGaussianSplatPreprocess::ProjectCull", true);
        sample.gpuGuideAccumMs =
            sumScopeMs(frame.gpuScopeTree, "GeneralGaussianSplatPreprocess::GuideAccum", true);
        sample.gpuSelectionMs =
            sumScopeMs(frame.gpuScopeTree, "GeneralGaussianSplatPreprocess::Selection", true);
        sample.gpuSortMs           = sumScopeMs(frame.gpuScopeTree, "GeneralGaussianSplatPreprocess::Sort", true);
        sample.gpuWriteIndirectMs =
            sumScopeMs(frame.gpuScopeTree, "GeneralGaussianSplatPreprocess::WriteIndirect", true);
        const double gpuSinglePassRenderMs =
            sumScopeMs(frame.gpuScopeTree, "GeneralGaussianSplatRenderPass", true);
        const double gpuFoveaLayerRenderMs =
            sumScopeMs(frame.gpuScopeTree, "GeneralGaussianSplatFoveaLayerPass", true);
        const double gpuMidLayerRenderMs =
            sumScopeMs(frame.gpuScopeTree, "GeneralGaussianSplatMidLayerPass", true);
        const double gpuOuterLayerRenderMs =
            sumScopeMs(frame.gpuScopeTree, "GeneralGaussianSplatOuterLayerPass", true);
        if (gpuSinglePassRenderMs >= 0.0 || gpuFoveaLayerRenderMs >= 0.0 ||
            gpuMidLayerRenderMs >= 0.0 || gpuOuterLayerRenderMs >= 0.0)
        {
            sample.gpuRenderPassMs = 0.0;
            if (gpuSinglePassRenderMs >= 0.0)
                sample.gpuRenderPassMs += gpuSinglePassRenderMs;
            if (gpuFoveaLayerRenderMs >= 0.0)
                sample.gpuRenderPassMs += gpuFoveaLayerRenderMs;
            if (gpuMidLayerRenderMs >= 0.0)
                sample.gpuRenderPassMs += gpuMidLayerRenderMs;
            if (gpuOuterLayerRenderMs >= 0.0)
                sample.gpuRenderPassMs += gpuOuterLayerRenderMs;
        }
        const double gpuFinalCompositionPassMs =
            sumScopeMs(frame.gpuScopeTree, "FinalCompositionPass", true);
        const double gpuFoveatedCompositePassMs =
            sumScopeMs(frame.gpuScopeTree, "GeneralGaussianSplatFoveatedCompositePass", true);
        if (gpuFinalCompositionPassMs >= 0.0 || gpuFoveatedCompositePassMs >= 0.0)
        {
            sample.gpuCompositePassMs = 0.0;
            if (gpuFinalCompositionPassMs >= 0.0)
                sample.gpuCompositePassMs += gpuFinalCompositionPassMs;
            if (gpuFoveatedCompositePassMs >= 0.0)
                sample.gpuCompositePassMs += gpuFoveatedCompositePassMs;
        }

        return sample;
    }

    inline int64_t csvCounter(const uint32_t value)
    {
        return value == UINT32_MAX ? -1 : static_cast<int64_t>(value);
    }

    inline SeriesStats summarizeSeries(std::vector<double> values)
    {
        values.erase(std::remove_if(values.begin(), values.end(), [](const double value) { return value < 0.0; }),
                     values.end());

        if (values.empty())
            return {};

        std::sort(values.begin(), values.end());

        double sum = 0.0;
        for (const double value : values)
            sum += value;

        const size_t middle = values.size() / 2u;
        const double median = values.size() % 2u == 0u ? (values[middle - 1u] + values[middle]) * 0.5 :
                                                         values[middle];

        return SeriesStats {
            .average = sum / static_cast<double>(values.size()),
            .median  = median,
            .minimum = values.front(),
            .maximum = values.back(),
        };
    }

    inline std::vector<double> collectSeries(const std::vector<GaussianBenchmarkSample>& samples,
                                             double GaussianBenchmarkSample::*           member)
    {
        std::vector<double> values;
        values.reserve(samples.size());
        for (const auto& sample : samples)
            values.push_back(sample.*member);
        return values;
    }

    inline void writeSelectedIdSidecar(const std::filesystem::path&                runtimeCsv,
                                       const std::vector<GaussianBenchmarkSample>& samples)
    {
        const bool enabled = std::any_of(samples.begin(), samples.end(), [](const GaussianBenchmarkSample& sample) {
            return sample.selectedIdLogEnabled && !sample.selectedSourceIds.empty();
        });
        if (!enabled)
            return;

        auto fnv1aHex8 = [](const std::string& text) {
            uint64_t hash = 1469598103934665603ull;
            for (const unsigned char byte : text)
            {
                hash ^= static_cast<uint64_t>(byte);
                hash *= 1099511628211ull;
            }
            std::ostringstream stream;
            stream << std::hex << std::setw(16) << std::setfill('0') << hash;
            return stream.str().substr(0, 8);
        };

        const std::string stem = runtimeCsv.stem().string();
        std::filesystem::path sidecar = runtimeCsv;
        sidecar.replace_filename(stem.substr(0, std::min<size_t>(stem.size(), 32u)) + "__" + fnv1aHex8(stem) +
                                 "_ids.csv");
        if (sidecar.has_parent_path())
        {
            std::error_code ec;
            std::filesystem::create_directories(sidecar.parent_path(), ec);
            if (ec)
            {
                VULTRA_CLIENT_WARN("Failed to create selected-ID sidecar directory {}: {}",
                                   sidecar.parent_path().string(),
                                   ec.message());
            }
        }

        std::ofstream out {sidecar};
        if (!out)
        {
            VULTRA_CLIENT_WARN("Failed to open selected-ID sidecar: {}", sidecar.string());
            return;
        }

        out << "sample,frame,selected_id_count,selected_id_log_stride,selected_source_id_ranges\n";
        for (const auto& sample : samples)
        {
            if (!sample.selectedIdLogEnabled || sample.selectedSourceIds.empty())
                continue;

            std::vector<uint32_t> ids = sample.selectedSourceIds;
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

            out << sample.sampleIndex << ',' << sample.frameIndex << ',' << ids.size() << ','
                << sample.selectedIdLogStride << ',';
            for (size_t i = 0u; i < ids.size();)
            {
                const uint32_t begin = ids[i];
                uint32_t       end   = begin;
                ++i;
                while (i < ids.size() && ids[i] == end + 1u)
                {
                    end = ids[i];
                    ++i;
                }
                if (begin == end)
                    out << begin;
                else
                    out << begin << '-' << end;
                if (i < ids.size())
                    out << ';';
            }
            out << '\n';
        }
    }

    inline constexpr std::array<const char*, kGaussianSplatEcsptEventBinCount> kEcsptEventBinLabels {
        "new_fovea",
        "old_fovea_only",
        "inner_boundary",
        "mid_annulus",
        "far_periphery",
    };

    inline constexpr std::array<const char*, kGaussianSplatEcsptEventTypeCount> kEcsptEventTypeLabels {
        "shared",
        "upgrade",
        "downgrade_protected",
        "downgrade_dropped",
        "instant_hash_flip",
        "immediate_new_fovea",
    };

    inline constexpr std::array<const char*, kGaussianSplatEcsptEventMetricCount> kEcsptEventMetricLabels {
        "count",
        "sum_opacity_scaled",
        "sum_projected_area_proxy_scaled",
        "sum_contribution_proxy_scaled",
        "sum_abs_delta_p_scaled",
        "sum_p_old_scaled",
        "sum_p_new_scaled",
    };

    inline uint32_t ecsptEventBinCounterOffset(const uint32_t bin,
                                               const uint32_t eventType,
                                               const uint32_t metric)
    {
        return ((bin * kGaussianSplatEcsptEventTypeCount + eventType) *
                kGaussianSplatEcsptEventMetricCount +
                metric);
    }

    inline void appendEcsptEventBinCsvHeader(std::ostream& out)
    {
        for (uint32_t bin = 0u; bin < kGaussianSplatEcsptEventBinCount; ++bin)
        {
            for (uint32_t eventType = 0u; eventType < kGaussianSplatEcsptEventTypeCount; ++eventType)
            {
                for (uint32_t metric = 0u; metric < kGaussianSplatEcsptEventMetricCount; ++metric)
                {
                    out << ",ecspt_bin_" << kEcsptEventBinLabels[bin]
                        << '_' << kEcsptEventTypeLabels[eventType]
                        << '_' << kEcsptEventMetricLabels[metric];
                }
            }
        }
    }

    inline void appendEcsptEventBinCsvValues(std::ostream& out, const GaussianBenchmarkSample& sample)
    {
        for (uint32_t bin = 0u; bin < kGaussianSplatEcsptEventBinCount; ++bin)
        {
            for (uint32_t eventType = 0u; eventType < kGaussianSplatEcsptEventTypeCount; ++eventType)
            {
                for (uint32_t metric = 0u; metric < kGaussianSplatEcsptEventMetricCount; ++metric)
                {
                    out << ',' << sample.eccStochasticEventBinCounters[
                        ecsptEventBinCounterOffset(bin, eventType, metric)];
                }
            }
        }
    }

    inline void writeBenchmarkCsv(const std::filesystem::path&                path,
                                  const std::vector<GaussianBenchmarkSample>& samples)
    {
        if (path.has_parent_path())
        {
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
            if (ec)
            {
                VULTRA_CLIENT_WARN("Failed to create benchmark output directory {}: {}", path.parent_path().string(),
                                   ec.message());
            }
        }

        std::ofstream out {path};
        if (!out)
        {
            VULTRA_CLIENT_WARN("Failed to open Gaussian benchmark output: {}", path.string());
            return;
        }

        out << "sample,frame,scripted_view,scripted_gaze,mode,lod_budget_enabled,lod_budget_mode,gaze_rendering,gaze_render_mode,layered_compositor,"
               "coverage_compensation,direct_prefix,lod_budget,cost_budget,actual_projected_cost,cost_overshoot,projected_cost_budget_ratio,projected_cost_chunk_size,coverage_bin_grid_x,coverage_bin_grid_y,coverage_bin_quota_ratio,coverage_bin_prefix_ratio,selected_cost_chunks,"
               "a4_projected_cost_selection_cpu_ms,a4_cost_estimation_cpu_ms,a4_chunk_selection_cpu_ms,a4_chunk_stop_cpu_ms,a4_guard_analysis_cpu_ms,a4_total_selection_cpu_ms,a4_num_chunks_scanned,a4_num_gaussians_or_chunks_considered,"
               "a4_projected_sample_cache_hit,a4_projected_sample_cache_miss,a4_projected_sample_reused,a4_num_projected_samples_built,a4_num_projected_samples_reused,"
               "a4_cache_entry_count,a4_cache_lookup_count,a4_cache_hit_count,a4_cache_miss_count,a4_cache_full_miss_count,a4_cache_build_count,a4_cache_eviction_count,a4_sample_cache_key_hash,"
               "a4_chunk_cache_hit_count,a4_chunk_cache_miss_count,a4_projected_samples_built,a4_projected_samples_reused,a4_chunk_aggregation_cpu_ms,a4_cache_lookup_cpu_ms,a4_cache_hit_rate,"
               "projected_cost_build_mode,projected_cost_semantic_diff,a4_gpu_cost_build_enabled,a4_gpu_cost_build_gpu_ms,a4_gpu_cost_readback_cpu_ms,a4_gpu_cost_chunk_count,a4_gpu_cost_valid,a4_cpu_fallback_used,"
               "a4_gpu_cpu_cost_l1_error,a4_gpu_cpu_cost_max_error,a4_gpu_cpu_selected_ratio_delta,a4_gpu_cpu_actual_cost_delta,a4_gpu_cpu_overshoot_delta,"
               "gaze_x,gaze_y,is_gaze_moving,gaze_jump_event,transition_window_id,fovea_lod,mid_lod,outer_lod,fovea_deg,mid_deg,continuous_theta0_deg,continuous_alpha,continuous_min_lod,fovea_res_scale,mid_res_scale,"
               "outer_res_scale,gaze_adaptation,gaze_distribution,target_frame_ms,dt_ms,cpu_frame_ms,cpu_render_ms,"
               "coverage_guard_mode,coverage_protection_degrees,coverage_guard_budget_ratio,coverage_guard_center_min,"
               "coverage_guard_transition_min,coverage_guard_periphery_min,coverage_guard_max_adds,guard_max_adds_effective,"
               "guard_mode_enabled,guard_risk_sectors,guard_risk_detected,"
               "guard_active,guard_active_rate,guard_repair_active,guard_added_any,guard_added_count,guard_added_cost,guard_budget_cap,coverage_failure_before_guard,coverage_failure_after_guard,"
               "risk_sector_count_total,risk_sector_count_center,risk_sector_count_mid,risk_sector_count_outer,risk_sector_active_center,risk_sector_active_mid,risk_sector_active_outer,"
               "coverage_failure_before_center,coverage_failure_before_mid,coverage_failure_before_outer,coverage_failure_after_center,coverage_failure_after_mid,coverage_failure_after_outer,"
               "guard_added_count_center,guard_added_count_mid,guard_added_count_outer,guard_added_cost_center,guard_added_cost_mid,guard_added_cost_outer,"
               "guard_repair_cap_hit,guard_repair_capped_by_max_adds,guard_repair_capped_by_cost,guard_repair_no_candidate,"
               "guard_candidate_count,guard_candidate_count_center,guard_candidate_count_mid,guard_candidate_count_outer,"
               "guard_true_candidate_count,guard_candidate_count_truncated,guard_required_extra_count_estimate,guard_required_extra_area_px,guard_deficit_area_before,guard_deficit_area_after,"
               "guard_added_count_to_failing_sector,guard_added_cost_to_failing_sector,guard_failing_sector_id,guard_repair_sector_match_rate,"
               "guard_analysis_cpu_ms,guard_projected_sample_build_cpu_ms,guard_coverage_area_cpu_ms,guard_candidate_scan_cpu_ms,guard_region_rebuild_cpu_ms,guard_cache_full_miss_count,guard_cache_hit_count,guard_cache_miss_count,"
               "guard_geometry_cache_hit_count,guard_geometry_cache_miss_count,guard_geometry_cache_full_miss_count,guard_geometry_samples_built,guard_geometry_samples_reused,"
               "scheduler_shrink_event,scheduler_grow_event,scheduler_budget_before,scheduler_budget_after,scheduler_p95_ema,scheduler_p95_window,scheduler_target_ms,scheduler_margin_low,scheduler_margin_high,"
               "projected_cost_budget_ratio_before_guard,projected_cost_budget_ratio_after_scheduler,selected_ratio_before_guard,selected_ratio_after_guard,guard_added_cost_ratio_to_budget,guard_added_count_ratio_to_selected,"
               "selected_id_log_enabled,selected_id_log_stride,selected_id_count,selected_id_iou,selected_id_churn,selected_id_exact_available,selected_id_added_count,selected_id_removed_count,true_selected_id_iou,true_selected_id_churn,a4_selected_id_logging_cpu_ms,"
               "coverage_score_before_guard,coverage_score_after_guard,temporal_hysteresis,boundary_smoothing,temporal_residency_frames,"
               "temporal_hysteresis_ratio,boundary_smoothing_ratio,temporal_peripheral_scale,"
               "peripheral_temporal_filter_enabled,peripheral_temporal_filter_mid_deg,peripheral_temporal_filter_outer_deg,"
               "peripheral_temporal_filter_lambda_scale,peripheral_temporal_filter_rejection_threshold,peripheral_temporal_filter_clamp_radius,"
               "sh_lod_enabled,sh_smooth_suppression_enabled,sh_degree_center,sh_degree_mid,sh_degree_outer,"
               "sh_smooth_l1_start_deg,sh_smooth_l1_end_deg,sh_smooth_l2_start_deg,sh_smooth_l2_end_deg,"
               "sh_smooth_l3_start_deg,sh_smooth_l3_end_deg,estimated_sh_ac_coeff_reads,estimated_sh_ac_read_reduction_vs_degree3,"
               "sh_storage_layout,sh_band_l1_reads_est,sh_band_l2_reads_est,sh_band_l3_reads_est,sh_band_bytes_est,sh_band_bytes_reduction_vs_monolithic,sh_storage_metadata_bytes_est,"
               "sh_energy_metadata_available,sh_energy_mean_after0,sh_energy_mean_after1,sh_energy_mean_after2,sh_energy_p95_after0,sh_energy_p95_after1,sh_energy_p95_after2,"
               "sh_guard_mode,sh_guard_raised_count,sh_guard_raised_ratio,sh_guard_recovered_ac_reads,sh_guard_saved_ac_reads_after_guard,sh_guard_threshold_mid,sh_guard_threshold_high,"
               "sh_degree_changed_count,sh_degree_changed_ratio,sh_pop_energy_proxy,sh_pop_energy_fovea,sh_pop_energy_mid,sh_pop_energy_periphery,sh_guard_raise_count,sh_delayed_downgrade_count,"
               "gaze_anchor_enabled,gaze_anchor_gpu_enabled,gaze_anchor_deadband_deg,gaze_anchor_fade_frames,gaze_anchor_min_update_frames,"
               "gaze_anchor_transition_budget_ratio,gaze_anchor_immediate_fovea_fill,gaze_anchor_contribution_guard,gaze_anchor_debug_log,"
               "gaze_anchor_anchor_x,gaze_anchor_anchor_y,gaze_anchor_new_anchor_x,gaze_anchor_new_anchor_y,gaze_anchor_distance_deg,gaze_anchor_frames_since_update,gaze_anchor_fade_phase,"
               "gaze_anchor_update_event,gaze_anchor_update_event_count,gaze_anchor_deadband_violation_count,"
               "gaze_anchor_union_count,gaze_anchor_union_count_ratio,gaze_anchor_shared_count,gaze_anchor_shared_ratio,"
               "gaze_anchor_old_only_count,gaze_anchor_new_only_count,gaze_anchor_fade_active_count,"
               "gaze_anchor_immediate_fovea_fill_count,gaze_anchor_transition_budget_target_count,"
               "gaze_anchor_transition_budget_capped_count,gaze_anchor_rendered_old_only_count,"
               "gaze_anchor_rendered_new_only_count,gaze_anchor_dropped_old_only_count,gaze_anchor_dropped_new_only_count,"
               "gaze_anchor_high_contribution_protected_count,gaze_anchor_weighted_tv_churn,"
               "ecc_stochastic_enabled,ecc_stochastic_fade_frames,ecc_stochastic_protect_old_fovea_deg,"
               "ecc_stochastic_protect_new_fovea_deg,ecc_stochastic_boundary_band_deg,ecc_stochastic_min_p_delta,"
               "ecc_stochastic_contribution_guard,ecc_stochastic_debug_log,ecc_stochastic_old_gaze_x,"
               "ecc_stochastic_old_gaze_y,ecc_stochastic_new_gaze_x,ecc_stochastic_new_gaze_y,"
               "ecc_stochastic_fade_phase,ecc_stochastic_frames_since_update,ecc_stochastic_update_event,"
               "ecc_stochastic_update_event_count,ecc_stochastic_counter_readback_valid,"
               "ecc_stochastic_counter_readback_cpu_ms,ecc_stochastic_total_candidates_seen,"
               "ecc_stochastic_base_new_selected_count,ecc_stochastic_effective_visible_after_ecspt_count,"
               "ecc_stochastic_shared_count,ecc_stochastic_upgrade_count,ecc_stochastic_downgrade_count,"
               "ecc_stochastic_protected_downgrade_count,ecc_stochastic_dropped_downgrade_count,"
               "ecc_stochastic_old_only_fade_visible_count,ecc_stochastic_immediate_new_fovea_count,"
               "ecc_stochastic_boundary_protected_count,ecc_stochastic_contribution_guard_protected_count,"
               "ecc_stochastic_zero_weight_discard_count,ecc_stochastic_min_p_delta_discard_count,"
               "ecc_stochastic_far_periphery_hard_drop_count,stable_optical_depth_alpha_clamped_count,"
               "coverage_release_floor_active_count,coverage_release_floor_raised_count,"
               "coverage_release_held_count,coverage_release_dropped_count,coverage_release_old_only_count,"
               "coverage_release_floor_sample_count,coverage_release_p_floor_mean,coverage_release_p_floor_max,"
               "coverage_release_d_alpha_mean,coverage_release_n_eff_mean,"
               "coverage_release_tile_total_count,coverage_release_tile_nonempty_count,coverage_release_tile_empty_count,"
               "coverage_release_d_alpha_mean_all,coverage_release_d_alpha_mean_nonempty,"
               "coverage_release_d_alpha_p50_nonempty,coverage_release_d_alpha_p90_nonempty,coverage_release_d_alpha_p95_nonempty,"
               "coverage_release_n_eff_mean_nonempty,coverage_release_n_eff_p50_nonempty,coverage_release_n_eff_p90_nonempty,coverage_release_n_eff_p95_nonempty,"
               "coverage_release_p_floor_mean_all,coverage_release_p_floor_mean_nonempty,coverage_release_p_floor_p90_nonempty,coverage_release_p_floor_p95_nonempty,"
               "coverage_release_p_lp_mean,coverage_release_p_static_mean,coverage_release_floor_active_fraction_vs_candidates,"
               "coverage_release_floor_raised_fraction_vs_effective_visible,coverage_release_delta_sum,"
               "coverage_release_delta_mean_over_candidates,coverage_release_delta_mean_over_held,"
               "coverage_release_held_effective_ratio,coverage_release_held_effective_count,"
               "coverage_release_history_reset_count,coverage_release_reentry_reset_count,"
               "coverage_release_epsilon_cutoff_count,coverage_release_saturated_floor_enabled,"
               "coverage_release_floor_cells_safe_saturated,coverage_release_floor_cells_unsafe,"
               "coverage_release_skipped_floor_update_count,coverage_release_d_alpha_safe_threshold,"
               "coverage_release_n_eff_safe_threshold,coverage_release_floor_saturation_ratio,"
               "coverage_release_candidate_alpha_mass,coverage_release_retained_alpha_mass,"
               "coverage_texture_current_alpha_mass,coverage_texture_current_alpha2_mass,"
               "coverage_texture_nonzero_texel_count,coverage_texture_width,coverage_texture_height,"
               "coverage_texture_history_beta,coverage_texture_strength,"
               "coverage_texture_p_final_mean,coverage_texture_stable_hash_kept_count,"
               "guide_before_discard_enabled,guide_texel_count,mean_guide_coverage,"
               "min_guide_coverage,max_guide_coverage,low_coverage_boosted_splats,"
               "high_coverage_reduced_splats,selected_count_before_tile_expansion,"
               "tile_duplicate_count_if_available,delayed_guide_enabled,"
               "delayed_guide_temporal_enabled,delayed_guide_release_active_count,"
               "delayed_guide_p_history_nonzero_count,"
               "delayed_guide_p_target_less_than_history_count,"
               "delayed_guide_p_history_bytes,"
               "delayed_guide_released_effective_visible_count,"
               "delayed_guide_released_alpha_proxy_sum,"
               "delayed_guide_release_cap_hit_count,"
               "delayed_guide_risk_protected_count,"
               "delayed_guide_footprint_fast_decay_count,"
               "delayed_guide_base_tile_duplicate_proxy_count,"
               "delayed_guide_released_tile_duplicate_proxy_count,"
               "delayed_guide_released_duplicate_proxy_ratio,"
               "delayed_guide_large_footprint_released_count,"
               "delayed_guide_large_footprint_released_tile_duplicate_proxy_count,"
               "delayed_guide_released_radius_mean,"
               "delayed_guide_released_radius_p95,"
               "delayed_guide_released_radius_max,"
               "delayed_guide_hotspot_0_tile_id,delayed_guide_hotspot_0_base_duplicates,delayed_guide_hotspot_0_released_duplicates,"
               "delayed_guide_hotspot_1_tile_id,delayed_guide_hotspot_1_base_duplicates,delayed_guide_hotspot_1_released_duplicates,"
               "delayed_guide_hotspot_2_tile_id,delayed_guide_hotspot_2_base_duplicates,delayed_guide_hotspot_2_released_duplicates,"
               "delayed_guide_hotspot_3_tile_id,delayed_guide_hotspot_3_base_duplicates,delayed_guide_hotspot_3_released_duplicates,"
               "delayed_guide_hotspot_4_tile_id,delayed_guide_hotspot_4_base_duplicates,delayed_guide_hotspot_4_released_duplicates,"
               "delayed_guide_hotspot_5_tile_id,delayed_guide_hotspot_5_base_duplicates,delayed_guide_hotspot_5_released_duplicates,"
               "delayed_guide_hotspot_6_tile_id,delayed_guide_hotspot_6_base_duplicates,delayed_guide_hotspot_6_released_duplicates,"
               "delayed_guide_hotspot_7_tile_id,delayed_guide_hotspot_7_base_duplicates,delayed_guide_hotspot_7_released_duplicates,"
               "shader_antipop_mode,shader_antipop_enabled,shader_antipop_alpha_multiplier_based,"
               "shader_antipop_avoids_cpu_selected_source_rebuild,shader_antipop_stable_candidate_set,"
               "shader_antipop_direct_prefix_stable,shader_antipop_debug_log,shader_antipop_hash_seed,"
               "shader_antipop_ramp_width,shader_antipop_guard_threshold,shader_antipop_guard_floor,"
               "shader_antipop_pkeep_curve,"
               "shader_antipop_prefix_ratio,shader_antipop_normalize_mode,shader_antipop_normalize_strength,"
               "shader_antipop_normalize_clamp_min,shader_antipop_normalize_clamp_max,shader_antipop_normalize_factor,"
               "shader_antipop_candidate_count,shader_antipop_effective_nonzero_estimate,"
               "shader_antipop_weight_min_estimate,shader_antipop_weight_mean_estimate,"
               "shader_antipop_weight_max_estimate,shader_antipop_alpha_mass_estimate,"
               "shader_antipop_guard_proxy,"
               "gpu_frame_ms,gpu_time_ms,draw_calls,dispatch_calls,copy_ops,update_ops,gpu_scope_resolved_count,"
               "gpu_scope_token_count,splat_assets,draw_records,total_splats,prepared_splats,"
               "fovea_splat_budget,mid_splat_budget,outer_splat_budget,max_visible_splat_cap,"
               "lod_selected_raw_splats,selected_count,selected_ratio,num_tile_intersections,sum_projected_area_px,"
               "clean_timing_mode,visible_splats,drawn_splats,visible_instant,tile_instances,covered_pixel_count,"
               "cpu_render_frame_ms,cpu_cook_ms,cpu_gpu_scene_rebuild_ms,cpu_lod_selection_ms,"
               "cpu_clod_prefix_build_ms,cpu_raw_selection_ms,cpu_lod_upload_ms,cpu_framegraph_build_ms,"
               "cpu_framegraph_execute_ms,gpu_preprocess_pass_ms,gpu_project_cull_ms,"
               "gpu_guide_accum_ms,gpu_selection_ms,gpu_sort_ms,"
               "gpu_write_indirect_ms,gpu_render_pass_ms,gpu_composite_pass_ms";
        appendEcsptEventBinCsvHeader(out);
        out << '\n';

        out << std::fixed << std::setprecision(6);
        for (const auto& sample : samples)
        {
            out << sample.sampleIndex << ',' << sample.frameIndex << ',' << csvCounter(sample.scriptedViewIndex)
                << ',' << csvCounter(sample.scriptedGazeIndex) << ',' << gaussianModeLabel(sample.baselineMode)
                << ',' << (sample.lodBudgetEnabled ? 1 : 0) << ','
                << lodBudgetModeLabel(sample.lodBudgetMode) << ','
                << (sample.foveatedClodEnabled ? 1 : 0) << ','
                << foveatedRenderModeLabel(sample.foveatedRenderMode) << ','
                << (sample.foveatedLayeredCompositeEnabled ? 1 : 0) << ','
                << (sample.foveatedCoverageCompensationEnabled ? 1 : 0) << ','
                << (sample.directPrefix ? 1 : 0) << ','
                << sample.lodBudget << ',' << sample.costBudget << ',' << sample.actualProjectedCost << ','
                << sample.costOvershoot << ',' << sample.projectedCostBudgetRatio << ','
                << sample.projectedCostChunkSize << ','
                << sample.foveatedCoverageBinGridX << ',' << sample.foveatedCoverageBinGridY << ','
                << sample.foveatedCoverageBinQuotaRatio << ',' << sample.foveatedCoverageBinPrefixRatio << ','
                << sample.selectedCostChunks << ','
                << sample.a4ProjectedCostSelectionCpuMs << ','
                << sample.a4CostEstimationCpuMs << ',' << sample.a4ChunkSelectionCpuMs << ','
                << sample.a4ChunkStopCpuMs << ','
                << sample.a4GuardAnalysisCpuMs << ',' << sample.a4TotalSelectionCpuMs << ',' << sample.a4NumChunksScanned << ','
                << sample.a4NumGaussiansOrChunksConsidered << ','
                << sample.a4ProjectedSampleCacheHit << ',' << sample.a4ProjectedSampleCacheMiss << ','
                << (sample.a4ProjectedSampleReused ? 1 : 0) << ','
                << sample.a4NumProjectedSamplesBuilt << ',' << sample.a4NumProjectedSamplesReused << ','
                << sample.a4CacheEntryCount << ',' << sample.a4CacheLookupCount << ','
                << sample.a4CacheHitCount << ',' << sample.a4CacheMissCount << ','
                << sample.a4CacheFullMissCount << ',' << sample.a4CacheBuildCount << ','
                << sample.a4CacheEvictionCount << ',' << sample.a4SampleCacheKeyHash << ','
                << sample.a4ChunkCacheHitCount << ',' << sample.a4ChunkCacheMissCount << ','
                << sample.a4ProjectedSamplesBuilt << ',' << sample.a4ProjectedSamplesReused << ','
                << sample.a4ChunkAggregationCpuMs << ',' << sample.a4CacheLookupCpuMs << ','
                << sample.a4CacheHitRate << ','
                << projectedCostBuildModeLabel(sample.projectedCostBuildMode) << ','
                << (sample.projectedCostSemanticDiffEnabled ? 1 : 0) << ','
                << (sample.a4GpuCostBuildEnabled ? 1 : 0) << ','
                << sample.a4GpuCostBuildGpuMs << ','
                << sample.a4GpuCostReadbackCpuMs << ','
                << sample.a4GpuCostChunkCount << ','
                << (sample.a4GpuCostValid ? 1 : 0) << ','
                << (sample.a4CpuFallbackUsed ? 1 : 0) << ','
                << sample.a4GpuCpuCostL1Error << ','
                << sample.a4GpuCpuCostMaxError << ','
                << sample.a4GpuCpuSelectedRatioDelta << ','
                << sample.a4GpuCpuActualCostDelta << ','
                << sample.a4GpuCpuOvershootDelta << ','
                << sample.gazeX << ',' << sample.gazeY << ','
                << sample.isGazeMoving << ',' << sample.gazeJumpEvent << ','
                << sample.transitionWindowId << ','
                << sample.foveaLod << ',' << sample.midLod << ',' << sample.outerLod
                << ',' << sample.foveaDegrees << ',' << sample.midDegrees << ','
                << sample.continuousTheta0Degrees << ',' << sample.continuousAlpha << ','
                << sample.continuousMinLevel << ','
                << sample.foveaResolutionScale << ',' << sample.midResolutionScale << ','
                << sample.outerResolutionScale << ',' << foveatedAdaptationModeLabel(sample.foveatedAdaptationMode)
                << ',' << foveatedDistributionLabel(sample.foveatedDistribution) << ','
                << sample.targetFrameMs << ',' << sample.dtMs << ',' << sample.cpuFrameMs << ',' << sample.cpuRenderMs
                << ',' << foveatedCoverageGuardModeLabel(sample.coverageGuardMode)
                << ',' << sample.coverageProtectionDegrees
                << ',' << sample.coverageGuardBudgetRatio
                << ',' << sample.coverageGuardCenterMin
                << ',' << sample.coverageGuardTransitionMin
                << ',' << sample.coverageGuardPeripheryMin
                << ',' << sample.coverageGuardMaxAdds
                << ',' << sample.guardMaxAddsEffective
                << ',' << (sample.coverageGuardModeEnabled ? 1 : 0)
                << ',' << sample.coverageGuardRiskSectors
                << ',' << (sample.coverageGuardRiskDetected ? 1 : 0)
                << ',' << (sample.coverageGuardActive ? 1 : 0)
                << ',' << (sample.coverageGuardActive ? 1 : 0)
                << ',' << (sample.coverageGuardRepairActive ? 1 : 0)
                << ',' << (sample.coverageGuardAddedAny ? 1 : 0)
                << ',' << sample.coverageGuardAddedCount
                << ',' << sample.coverageGuardAddedCost
                << ',' << sample.coverageGuardBudgetCap
                << ',' << sample.coverageFailureBeforeGuard
                << ',' << sample.coverageFailureAfterGuard
                << ',' << sample.riskSectorCountTotal
                << ',' << sample.riskSectorCountCenter
                << ',' << sample.riskSectorCountMid
                << ',' << sample.riskSectorCountOuter
                << ',' << (sample.riskSectorActiveCenter ? 1 : 0)
                << ',' << (sample.riskSectorActiveMid ? 1 : 0)
                << ',' << (sample.riskSectorActiveOuter ? 1 : 0)
                << ',' << sample.coverageFailureBeforeCenter
                << ',' << sample.coverageFailureBeforeMid
                << ',' << sample.coverageFailureBeforeOuter
                << ',' << sample.coverageFailureAfterCenter
                << ',' << sample.coverageFailureAfterMid
                << ',' << sample.coverageFailureAfterOuter
                << ',' << sample.guardAddedCountCenter
                << ',' << sample.guardAddedCountMid
                << ',' << sample.guardAddedCountOuter
                << ',' << sample.guardAddedCostCenter
                << ',' << sample.guardAddedCostMid
                << ',' << sample.guardAddedCostOuter
                << ',' << (sample.guardRepairCapHit ? 1 : 0)
                << ',' << (sample.guardRepairCappedByMaxAdds ? 1 : 0)
                << ',' << (sample.guardRepairCappedByCost ? 1 : 0)
                << ',' << (sample.guardRepairNoCandidate ? 1 : 0)
                << ',' << sample.guardCandidateCount
                << ',' << sample.guardCandidateCountCenter
                << ',' << sample.guardCandidateCountMid
                << ',' << sample.guardCandidateCountOuter
                << ',' << sample.guardTrueCandidateCount
                << ',' << (sample.guardCandidateCountTruncated ? 1 : 0)
                << ',' << csvCounter(sample.guardRequiredExtraCountEstimate)
                << ',' << sample.guardRequiredExtraAreaPx
                << ',' << sample.guardDeficitAreaBefore
                << ',' << sample.guardDeficitAreaAfter
                << ',' << sample.guardAddedCountToFailingSector
                << ',' << sample.guardAddedCostToFailingSector
                << ',' << csvCounter(sample.guardFailingSectorId)
                << ',' << sample.guardRepairSectorMatchRate
                << ',' << sample.guardAnalysisCpuMs
                << ',' << sample.guardProjectedSampleBuildCpuMs
                << ',' << sample.guardCoverageAreaCpuMs
                << ',' << sample.guardCandidateScanCpuMs
                << ',' << sample.guardRegionRebuildCpuMs
                << ',' << sample.guardCacheFullMissCount
                << ',' << sample.guardCacheHitCount
                << ',' << sample.guardCacheMissCount
                << ',' << sample.guardGeometryCacheHitCount
                << ',' << sample.guardGeometryCacheMissCount
                << ',' << sample.guardGeometryCacheFullMissCount
                << ',' << sample.guardGeometrySamplesBuilt
                << ',' << sample.guardGeometrySamplesReused
                << ',' << (sample.schedulerShrinkEvent ? 1 : 0)
                << ',' << (sample.schedulerGrowEvent ? 1 : 0)
                << ',' << sample.schedulerBudgetBefore
                << ',' << sample.schedulerBudgetAfter
                << ',' << sample.schedulerP95Ema
                << ',' << sample.schedulerP95Window
                << ',' << sample.schedulerTargetMs
                << ',' << sample.schedulerMarginLow
                << ',' << sample.schedulerMarginHigh
                << ',' << sample.projectedCostBudgetRatioBeforeGuard
                << ',' << sample.projectedCostBudgetRatioAfterScheduler
                << ',' << sample.selectedRatioBeforeGuard
                << ',' << sample.selectedRatioAfterGuard
                << ',' << sample.guardAddedCostRatioToBudget
                << ',' << sample.guardAddedCountRatioToSelected
                << ',' << (sample.selectedIdLogEnabled ? 1 : 0)
                << ',' << sample.selectedIdLogStride
                << ',' << sample.selectedIdCount
                << ',' << sample.selectedIdIou
                << ',' << sample.selectedIdChurn
                << ',' << (sample.selectedIdExactAvailable ? 1 : 0)
                << ',' << sample.selectedIdAddedCount
                << ',' << sample.selectedIdRemovedCount
                << ',' << sample.trueSelectedIdIou
                << ',' << sample.trueSelectedIdChurn
                << ',' << sample.a4SelectedIdLoggingCpuMs
                << ',' << sample.coverageScoreBeforeGuard
                << ',' << sample.coverageScoreAfterGuard << ','
                << (sample.temporalHysteresisEnabled ? 1 : 0) << ','
                << (sample.boundarySmoothingEnabled ? 1 : 0) << ','
                << sample.temporalResidencyFrames << ','
                << sample.temporalHysteresisRatio << ','
                << sample.boundarySmoothingRatio
                << ',' << sample.temporalPeripheralScale
                << ',' << (sample.peripheralTemporalFilterEnabled ? 1 : 0)
                << ',' << sample.peripheralTemporalFilterMidDegrees
                << ',' << sample.peripheralTemporalFilterOuterDegrees
                << ',' << sample.peripheralTemporalFilterLambdaScale
                << ',' << sample.peripheralTemporalFilterRejectionThreshold
                << ',' << sample.peripheralTemporalFilterClampRadius
                << ',' << (sample.shLodEnabled ? 1 : 0)
                << ',' << (sample.shSmoothSuppressionEnabled ? 1 : 0)
                << ',' << sample.shDegreeCenter
                << ',' << sample.shDegreeMid
                << ',' << sample.shDegreeOuter
                << ',' << sample.shSmoothL1StartDegrees
                << ',' << sample.shSmoothL1EndDegrees
                << ',' << sample.shSmoothL2StartDegrees
                << ',' << sample.shSmoothL2EndDegrees
                << ',' << sample.shSmoothL3StartDegrees
                << ',' << sample.shSmoothL3EndDegrees
                << ',' << sample.estimatedShAcCoeffReads
                << ',' << sample.estimatedShAcReadReductionVsDegree3
                << ',' << shStorageLayoutLabel(sample.shStorageLayout)
                << ',' << sample.shBandL1ReadsEst
                << ',' << sample.shBandL2ReadsEst
                << ',' << sample.shBandL3ReadsEst
                << ',' << sample.shBandBytesEst
                << ',' << sample.shBandBytesReductionVsMonolithic
                << ',' << sample.shStorageMetadataBytesEst
                << ',' << (sample.shEnergyMetadataAvailable ? 1 : 0)
                << ',' << sample.shEnergyMeanAfter0
                << ',' << sample.shEnergyMeanAfter1
                << ',' << sample.shEnergyMeanAfter2
                << ',' << sample.shEnergyP95After0
                << ',' << sample.shEnergyP95After1
                << ',' << sample.shEnergyP95After2
                << ',' << shLodGuardModeLabel(sample.shGuardMode)
                << ',' << sample.shGuardRaisedCount
                << ',' << sample.shGuardRaisedRatio
                << ',' << sample.shGuardRecoveredAcReads
                << ',' << sample.shGuardSavedAcReadsAfterGuard
                << ',' << sample.shGuardThresholdMid
                << ',' << sample.shGuardThresholdHigh
                << ',' << sample.shDegreeChangedCount
                << ',' << sample.shDegreeChangedRatio
                << ',' << sample.shPopEnergyProxy
                << ',' << sample.shPopEnergyFovea
                << ',' << sample.shPopEnergyMid
                << ',' << sample.shPopEnergyPeriphery
                << ',' << sample.shGuardRaiseCount
                << ',' << sample.shDelayedDowngradeCount
                << ',' << (sample.gazeAnchorCrossfadeEnabled ? 1 : 0)
                << ',' << (sample.gazeAnchorGpuCrossfadeEnabled ? 1 : 0)
                << ',' << sample.gazeAnchorDeadbandDegrees
                << ',' << sample.gazeAnchorFadeFrames
                << ',' << sample.gazeAnchorMinUpdateFrames
                << ',' << sample.gazeAnchorTransitionBudgetRatio
                << ',' << (sample.gazeAnchorImmediateFoveaFill ? 1 : 0)
                << ',' << (sample.gazeAnchorContributionGuard ? 1 : 0)
                << ',' << (sample.gazeAnchorDebugLogEnabled ? 1 : 0)
                << ',' << sample.gazeAnchorAnchorX
                << ',' << sample.gazeAnchorAnchorY
                << ',' << sample.gazeAnchorNewAnchorX
                << ',' << sample.gazeAnchorNewAnchorY
                << ',' << sample.gazeAnchorDistanceDegrees
                << ',' << sample.gazeAnchorFramesSinceUpdate
                << ',' << sample.gazeAnchorFadePhase
                << ',' << (sample.gazeAnchorUpdateEvent ? 1 : 0)
                << ',' << sample.gazeAnchorUpdateEventCount
                << ',' << sample.gazeAnchorDeadbandViolationCount
                << ',' << sample.gazeAnchorUnionCount
                << ',' << sample.gazeAnchorUnionCountRatio
                << ',' << sample.gazeAnchorSharedCount
                << ',' << sample.gazeAnchorSharedRatio
                << ',' << sample.gazeAnchorOldOnlyCount
                << ',' << sample.gazeAnchorNewOnlyCount
                << ',' << sample.gazeAnchorFadeActiveCount
                << ',' << sample.gazeAnchorImmediateFoveaFillCount
                << ',' << sample.gazeAnchorTransitionBudgetTargetCount
                << ',' << sample.gazeAnchorTransitionBudgetCappedCount
                << ',' << sample.gazeAnchorRenderedOldOnlyCount
                << ',' << sample.gazeAnchorRenderedNewOnlyCount
                << ',' << sample.gazeAnchorDroppedOldOnlyCount
                << ',' << sample.gazeAnchorDroppedNewOnlyCount
                << ',' << sample.gazeAnchorHighContributionProtectedCount
                << ',' << sample.gazeAnchorWeightedTvChurn
                << ',' << (sample.eccStochasticTransitionEnabled ? 1 : 0)
                << ',' << sample.eccStochasticFadeFrames
                << ',' << sample.eccStochasticProtectOldFoveaDegrees
                << ',' << sample.eccStochasticProtectNewFoveaDegrees
                << ',' << sample.eccStochasticBoundaryBandDegrees
                << ',' << sample.eccStochasticMinPDelta
                << ',' << (sample.eccStochasticContributionGuard ? 1 : 0)
                << ',' << (sample.eccStochasticDebugLogEnabled ? 1 : 0)
                << ',' << sample.eccStochasticOldGazeX
                << ',' << sample.eccStochasticOldGazeY
                << ',' << sample.eccStochasticNewGazeX
                << ',' << sample.eccStochasticNewGazeY
                << ',' << sample.eccStochasticFadePhase
                << ',' << sample.eccStochasticFramesSinceUpdate
                << ',' << (sample.eccStochasticUpdateEvent ? 1 : 0)
                << ',' << sample.eccStochasticUpdateEventCount
                << ',' << (sample.eccStochasticCounterReadbackValid ? 1 : 0)
                << ',' << sample.eccStochasticCounterReadbackCpuMs
                << ',' << sample.eccStochasticTotalCandidatesSeen
                << ',' << sample.eccStochasticBaseNewSelectedCount
                << ',' << sample.eccStochasticEffectiveVisibleAfterEcsptCount
                << ',' << sample.eccStochasticSharedCount
                << ',' << sample.eccStochasticUpgradeCount
                << ',' << sample.eccStochasticDowngradeCount
                << ',' << sample.eccStochasticProtectedDowngradeCount
                << ',' << sample.eccStochasticDroppedDowngradeCount
                << ',' << sample.eccStochasticOldOnlyFadeVisibleCount
                << ',' << sample.eccStochasticImmediateNewFoveaCount
                << ',' << sample.eccStochasticBoundaryProtectedCount
                << ',' << sample.eccStochasticContributionGuardProtectedCount
                << ',' << sample.eccStochasticZeroWeightDiscardCount
                << ',' << sample.eccStochasticMinPDeltaDiscardCount
                << ',' << sample.eccStochasticFarPeripheryHardDropCount
                << ',' << sample.stableOpticalDepthAlphaClampedCount
                << ',' << sample.coverageStableReleaseFloorActiveCount
                << ',' << sample.coverageStableReleaseFloorRaisedCount
                << ',' << sample.coverageStableReleaseHeldCount
                << ',' << sample.coverageStableReleaseDroppedCount
                << ',' << sample.coverageStableReleaseOldOnlyCount
                << ',' << sample.coverageStableReleaseFloorSampleCount
                << ',' << sample.coverageStableReleasePFloorMean
                << ',' << sample.coverageStableReleasePFloorMax
                << ',' << sample.coverageStableReleaseDAlphaMean
                << ',' << sample.coverageStableReleaseNEffMean
                << ',' << sample.coverageStableReleaseTileTotalCount
                << ',' << sample.coverageStableReleaseTileNonEmptyCount
                << ',' << sample.coverageStableReleaseTileEmptyCount
                << ',' << sample.coverageStableReleaseDAlphaMeanAll
                << ',' << sample.coverageStableReleaseDAlphaMeanNonEmpty
                << ',' << sample.coverageStableReleaseDAlphaP50NonEmpty
                << ',' << sample.coverageStableReleaseDAlphaP90NonEmpty
                << ',' << sample.coverageStableReleaseDAlphaP95NonEmpty
                << ',' << sample.coverageStableReleaseNEffMeanNonEmpty
                << ',' << sample.coverageStableReleaseNEffP50NonEmpty
                << ',' << sample.coverageStableReleaseNEffP90NonEmpty
                << ',' << sample.coverageStableReleaseNEffP95NonEmpty
                << ',' << sample.coverageStableReleasePFloorMeanAll
                << ',' << sample.coverageStableReleasePFloorMeanNonEmpty
                << ',' << sample.coverageStableReleasePFloorP90NonEmpty
                << ',' << sample.coverageStableReleasePFloorP95NonEmpty
                << ',' << sample.coverageStableReleasePLpMean
                << ',' << sample.coverageStableReleasePStaticMean
                << ',' << sample.coverageStableReleaseFloorActiveFractionVsCandidates
                << ',' << sample.coverageStableReleaseFloorRaisedFractionVsEffectiveVisible
                << ',' << sample.coverageStableReleaseDeltaSum
                << ',' << sample.coverageStableReleaseDeltaMeanOverCandidates
                << ',' << sample.coverageStableReleaseDeltaMeanOverHeld
                << ',' << sample.coverageStableReleaseHeldEffectiveRatio
                << ',' << sample.coverageStableReleaseHeldEffectiveCount
                << ',' << sample.coverageStableReleaseHistoryResetCount
                << ',' << sample.coverageStableReleaseReentryResetCount
                << ',' << sample.coverageStableReleaseEpsilonCutoffCount
                << ',' << (sample.coverageStableReleaseSaturatedFloorEnabled ? 1 : 0)
                << ',' << sample.coverageStableReleaseFloorCellsSafeSaturated
                << ',' << sample.coverageStableReleaseFloorCellsUnsafe
                << ',' << sample.coverageStableReleaseSkippedFloorUpdateCount
                << ',' << sample.coverageStableReleaseDAlphaSafeThreshold
                << ',' << sample.coverageStableReleaseNEffSafeThreshold
                << ',' << sample.coverageStableReleaseFloorSaturationRatio
                << ',' << sample.coverageStableReleaseCandidateAlphaMass
                << ',' << sample.coverageStableReleaseRetainedAlphaMass
                << ',' << sample.coverageTextureCurrentAlphaMass
                << ',' << sample.coverageTextureCurrentAlpha2Mass
                << ',' << sample.coverageTextureNonzeroTexelCount
                << ',' << sample.coverageTextureWidth
                << ',' << sample.coverageTextureHeight
                << ',' << sample.coverageTextureHistoryBeta
                << ',' << sample.coverageTextureStrength
                << ',' << sample.coverageTexturePFinalMean
                << ',' << sample.coverageTextureStableHashKeptCount
                << ',' << (sample.guideBeforeDiscardEnabled ? 1 : 0)
                << ',' << sample.guideBeforeDiscardTexelCount
                << ',' << sample.guideBeforeDiscardCoverageMean
                << ',' << sample.guideBeforeDiscardCoverageMin
                << ',' << sample.guideBeforeDiscardCoverageMax
                << ',' << sample.guideBeforeDiscardLowCoverageBoostedSplats
                << ',' << sample.guideBeforeDiscardHighCoverageReducedSplats
                << ',' << sample.guideBeforeDiscardSelectedBeforeTileExpansion
                << ',' << sample.guideBeforeDiscardTileDuplicateCount
                << ',' << (sample.delayedGuideLogPolarEnabled ? 1 : 0)
                << ',' << (sample.delayedGuideTemporalLogPolarEnabled ? 1 : 0)
                << ',' << sample.delayedGuideReleaseActiveCount
                << ',' << sample.delayedGuidePHistoryNonzeroCount
                << ',' << sample.delayedGuidePTargetLessThanHistoryCount
                << ',' << sample.delayedGuidePHistoryBytes
                << ',' << sample.delayedGuideReleasedEffectiveVisibleCount
                << ',' << sample.delayedGuideReleasedAlphaProxySum
                << ',' << sample.delayedGuideReleaseCapHitCount
                << ',' << sample.delayedGuideRiskProtectedCount
                << ',' << sample.delayedGuideFootprintFastDecayCount
                << ',' << sample.delayedGuideBaseTileDuplicateProxyCount
                << ',' << sample.delayedGuideReleasedTileDuplicateProxyCount
                << ',' << sample.delayedGuideReleasedDuplicateProxyRatio
                << ',' << sample.delayedGuideLargeFootprintReleasedCount
                << ',' << sample.delayedGuideLargeFootprintReleasedTileDuplicateProxyCount
                << ',' << sample.delayedGuideReleasedRadiusMean
                << ',' << sample.delayedGuideReleasedRadiusP95
                << ',' << sample.delayedGuideReleasedRadiusMax
                << ',' << sample.delayedGuideHotspotTileIds[0]
                << ',' << sample.delayedGuideHotspotBaseDuplicates[0]
                << ',' << sample.delayedGuideHotspotReleasedDuplicates[0]
                << ',' << sample.delayedGuideHotspotTileIds[1]
                << ',' << sample.delayedGuideHotspotBaseDuplicates[1]
                << ',' << sample.delayedGuideHotspotReleasedDuplicates[1]
                << ',' << sample.delayedGuideHotspotTileIds[2]
                << ',' << sample.delayedGuideHotspotBaseDuplicates[2]
                << ',' << sample.delayedGuideHotspotReleasedDuplicates[2]
                << ',' << sample.delayedGuideHotspotTileIds[3]
                << ',' << sample.delayedGuideHotspotBaseDuplicates[3]
                << ',' << sample.delayedGuideHotspotReleasedDuplicates[3]
                << ',' << sample.delayedGuideHotspotTileIds[4]
                << ',' << sample.delayedGuideHotspotBaseDuplicates[4]
                << ',' << sample.delayedGuideHotspotReleasedDuplicates[4]
                << ',' << sample.delayedGuideHotspotTileIds[5]
                << ',' << sample.delayedGuideHotspotBaseDuplicates[5]
                << ',' << sample.delayedGuideHotspotReleasedDuplicates[5]
                << ',' << sample.delayedGuideHotspotTileIds[6]
                << ',' << sample.delayedGuideHotspotBaseDuplicates[6]
                << ',' << sample.delayedGuideHotspotReleasedDuplicates[6]
                << ',' << sample.delayedGuideHotspotTileIds[7]
                << ',' << sample.delayedGuideHotspotBaseDuplicates[7]
                << ',' << sample.delayedGuideHotspotReleasedDuplicates[7]
                << ',' << shaderAntiPopModeLabel(sample.shaderAntiPopMode)
                << ',' << (sample.shaderAntiPopEnabled ? 1 : 0)
                << ',' << (sample.shaderAntiPopAlphaMultiplierBased ? 1 : 0)
                << ',' << (sample.shaderAntiPopAvoidsCpuSelectedSourceRebuild ? 1 : 0)
                << ',' << (sample.shaderAntiPopStableCandidateSet ? 1 : 0)
                << ',' << (sample.shaderAntiPopDirectPrefixStable ? 1 : 0)
                << ',' << (sample.shaderAntiPopDebugLogEnabled ? 1 : 0)
                << ',' << sample.shaderAntiPopHashSeed
                << ',' << sample.shaderAntiPopRampWidth
                << ',' << sample.shaderAntiPopGuardThreshold
                << ',' << sample.shaderAntiPopGuardFloor
                << ',' << shaderAntiPopPKeepCurveLabel(sample.shaderAntiPopPKeepCurve)
                << ',' << sample.shaderAntiPopPrefixRatio
                << ',' << shaderAntiPopNormalizeModeLabel(sample.shaderAntiPopNormalizeMode)
                << ',' << sample.shaderAntiPopNormalizeStrength
                << ',' << sample.shaderAntiPopNormalizeClampMin
                << ',' << sample.shaderAntiPopNormalizeClampMax
                << ',' << sample.shaderAntiPopNormalizeFactor
                << ',' << sample.shaderAntiPopCandidateCount
                << ',' << sample.shaderAntiPopEffectiveNonzeroEstimate
                << ',' << sample.shaderAntiPopWeightMinEstimate
                << ',' << sample.shaderAntiPopWeightMeanEstimate
                << ',' << sample.shaderAntiPopWeightMaxEstimate
                << ',' << sample.shaderAntiPopAlphaMassEstimate
                << ',' << shaderAntiPopGuardProxyLabel(sample.shaderAntiPopGuardProxyMode)
                << ',' << sample.gpuFrameMs << ',' << sample.gpuFrameMs << ',' << sample.drawCalls << ',' << sample.dispatchCalls << ','
                << sample.copyOps << ',' << sample.updateOps << ',' << sample.gpuScopeResolvedCount << ','
                << sample.gpuScopeTokenCount << ',' << sample.splatAssets << ',' << sample.drawRecords << ','
                << sample.totalSplats << ',' << sample.preparedSplats << ','
                << sample.foveaSplatBudget << ',' << sample.midSplatBudget << ',' << sample.outerSplatBudget << ','
                << sample.maxVisibleSplatCap << ','
                << sample.lodSelectedRawSplats << ',' << sample.lodSelectedRawSplats << ','
                << (sample.totalSplats > 0u ?
                        static_cast<double>(sample.lodSelectedRawSplats) / static_cast<double>(sample.totalSplats) :
                        0.0)
                << ',' << sample.numTileIntersections << ',' << sample.sumProjectedAreaPx << ','
                << (sample.cleanTimingMode ? 1 : 0) << ','
                << csvCounter(sample.visibleSplats) << ','
                << csvCounter(sample.drawnSplats) << ','
                << csvCounter(sample.visibleInstant) << ','
                << csvCounter(sample.tileInstances) << ','
                << csvCounter(sample.coveredPixelCount) << ','
                << sample.cpuRenderFrameMs << ',' << sample.cpuCookMs << ',' << sample.cpuGpuSceneRebuildMs << ','
                << sample.cpuLodSelectionMs << ',' << sample.cpuClodSelectionMs << ',' << sample.cpuRawSelectionMs
                << ',' << sample.cpuLodUploadMs << ',' << sample.cpuFrameGraphBuildMs << ','
                << sample.cpuFrameGraphExecuteMs << ',' << sample.gpuPreprocessPassMs << ','
                << sample.gpuProjectCullMs << ',' << sample.gpuGuideAccumMs << ','
                << sample.gpuSelectionMs << ',' << sample.gpuSortMs << ',' << sample.gpuWriteIndirectMs << ','
                << sample.gpuRenderPassMs << ',' << sample.gpuCompositePassMs;
            appendEcsptEventBinCsvValues(out, sample);
            out << '\n';
        }
        writeSelectedIdSidecar(path, samples);
    }

    inline std::string statsText(const SeriesStats& stats)
    {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(3) << "avg=" << stats.average << "ms, median=" << stats.median
               << "ms, min=" << stats.minimum << "ms, max=" << stats.maximum << "ms";
        return stream.str();
    }
} // namespace vultra::gaussian_splatting_example

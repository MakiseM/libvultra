#include "vultra/function/rendering/render_system.hpp"

#include "vultra/core/base/common_context.hpp"
#include "vultra/core/engine/engine_context.hpp"
#include "vultra/core/rhi/backends/webgpu/webgpu_command_buffer_access.hpp"
#include "vultra/core/rhi/command_buffer.hpp"
#include "vultra/core/rhi/descriptorset_builder.hpp"
#include "vultra/core/services/window_service.hpp"
#include "vultra/function/framegraph/framegraph_context.hpp"
#include "vultra/function/rendering/framework/resource_uploader.hpp"
#include "vultra/function/rendering/render_structs.hpp"
#include "vultra/function/rendering/srp/builtin/upload_resources.hpp"
#include "vultra/function/rendering/srp/render_context.hpp"
#include "vultra/function/services/asset_service.hpp"
#include "vultra/function/services/camera_service.hpp"
#include "vultra/function/services/frame_debugger_service.hpp"
#include "vultra/function/services/gpu_resource_service.hpp"
#include "vultra/function/services/imgui_service.hpp"
#include "vultra/function/services/render_backend_service.hpp"
#include "vultra/function/services/shader_service.hpp"
#include "vultra/function/services/world_service.hpp"
#include "vultra/function/world/components/entity_status_component.hpp"
#include "vultra/function/world/components/gaussian_splat_component.hpp"
#include "vultra/function/world/components/id_component.hpp"
#include "vultra/function/world/components/mesh_component.hpp"
#include "vultra/function/world/components/transform_component.hpp"
#include "vultra/function/world/world.hpp"

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/trigonometric.hpp>

#include <vbase/core/exe_path.hpp>

#include <fg/Blackboard.hpp>
#include <fg/FrameGraph.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>
#ifndef NDEBUG
#include <fstream>
#endif

namespace vultra
{
    namespace
    {
        thread_local rhi::BuiltinProfilerGpuScopeContext g_CurrentBuiltinProfilerGpuScopeContext {};
        std::atomic<uint64_t> g_RenderSystemProjectedCostCacheSaltCounter {1u};

        void clearColorTarget(rhi::CommandBuffer&        cb,
                              rhi::Texture&              target,
                              const rhi::Rect2D&         area,
                              const std::optional<rhi::ClearValue>& clearValue,
                              const bool                 enableMultiview,
                              const uint32_t             multiviewMask)
        {
            rhi::FramebufferInfo clearFbInfo {
                .area             = area,
                .layers           = enableMultiview ? 2u : 1u,
                .viewMask         = enableMultiview ? multiviewMask : 0u,
                .colorAttachments = {rhi::AttachmentInfo {
                    .target     = &target,
                    .clearValue = clearValue.has_value() ? clearValue : std::optional<rhi::ClearValue> {glm::vec4 {0, 0, 0, 1}},
                }},
            };

            rhi::prepareForAttachment(cb, target, false);
            cb.beginRendering(clearFbInfo);
            cb.endRendering();
        }

        float effectiveGaussianAutomaticClodLevel(const GaussianSplatRenderSettings& settings)
        {
            if (settings.foveatedScoreBudgetEnabled())
                return std::clamp(settings.clodLevel, 0.01f, 1.0f);

            if (!settings.foveatedClodActive())
                return std::clamp(settings.clodLevel, 0.01f, 1.0f);

            const glm::vec3 levels {
                std::clamp(settings.foveatedRingLevels.x, 0.0f, 1.0f),
                std::clamp(settings.foveatedRingLevels.y, 0.0f, 1.0f),
                std::clamp(settings.foveatedRingLevels.z, 0.0f, 1.0f),
            };
            float level = std::max(levels.x, std::max(levels.y, levels.z));
            if (settings.foveatedCoverageGuardMode == GaussianSplatFoveatedCoverageGuardMode::eLocalBounded &&
                settings.foveatedCoverageProtectionDegrees > 1e-4f)
            {
                level += level * std::clamp(settings.foveatedCoverageGuardBudgetRatio, 0.0f, 0.25f);
            }
            return std::clamp(level, 0.01f, 1.0f);
        }

        uint32_t effectiveGaussianLodBudget(const GaussianSplatRenderSettings& settings, const uint32_t totalSplatCount)
        {
            // Baseline consumes the full table. Ordered CLOD consumes a prefix of
            // the table that vasset already sorted by importance at import time.
            if (!settings.lodBudgetEnabled())
                return totalSplatCount;

            // An explicit budget is useful for repeatable profiling. With budget 0
            // the UI exposes clodLevel as the paper-style continuous LOD fraction.
            if (settings.lodBudget > 0u)
                return std::min(totalSplatCount, settings.lodBudget);
            const float clodLevel = effectiveGaussianAutomaticClodLevel(settings);
            if (settings.shaderAntiPopActive() && settings.shaderAntiPopPrefixRatio > 0.0f)
            {
                const float prefixLevel =
                    std::clamp(clodLevel * settings.shaderAntiPopPrefixRatio, 0.01f, 1.0f);
                return std::min(totalSplatCount,
                                std::max(1u, static_cast<uint32_t>(
                                                 std::ceil(static_cast<float>(totalSplatCount) * prefixLevel))));
            }
            return std::min(totalSplatCount,
                            std::max(1u, static_cast<uint32_t>(
                                             std::ceil(static_cast<float>(totalSplatCount) * clodLevel))));
        }

        struct GaussianProjectedCostSample
        {
            static constexpr uint32_t kMaxCameraFootprints = 2u;
            uint64_t tileIntersections {0};
            uint64_t cost {1};
            double   projectedAreaPx {0.0};
            std::array<double, kMaxCameraFootprints>    cameraProjectedAreaPx {0.0, 0.0};
            std::array<glm::vec2, kMaxCameraFootprints> cameraCenterNdc {glm::vec2 {0.0f}, glm::vec2 {0.0f}};
            std::array<float, kMaxCameraFootprints>     cameraAngularFootprintRadiusDegrees {0.0f, 0.0f};
            std::array<uint8_t, kMaxCameraFootprints>   cameraVisible {0u, 0u};
            std::array<double, 3> regionProjectedAreaPx {0.0, 0.0, 0.0};
            std::array<float, 3>  regionMinEccentricityDegrees {
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity()};
        };

        struct GaussianProjectedChunkCost
        {
            uint32_t beginRank {0};
            uint32_t endRank {0};
            uint64_t cost {0};
            uint64_t tileIntersections {0};
            double   projectedAreaPx {0.0};
        };

        struct GaussianProjectedCostPrefixStats
        {
            uint32_t selectedCount {0};
            uint32_t selectedChunks {0};
            uint64_t costBudget {0};
            uint64_t actualProjectedCost {0};
            uint64_t costOvershoot {0};
            uint64_t totalProjectedCost {0};
            uint64_t numTileIntersections {0};
            double   sumProjectedAreaPx {0.0};
            float    budgetRatio {0.0f};
            double   costEstimationCpuMs {0.0};
            double   chunkSelectionCpuMs {0.0};
            double   totalSelectionCpuMs {0.0};
            double   chunkAggregationCpuMs {0.0};
            double   cacheLookupCpuMs {0.0};
            uint32_t chunksScanned {0u};
            uint32_t gaussiansOrChunksConsidered {0u};
            double   cacheHitRate {-1.0};
            bool     gpuCostBuildEnabled {false};
            double   gpuCostBuildGpuMs {-1.0};
            double   gpuCostReadbackCpuMs {-1.0};
            uint32_t gpuCostChunkCount {0u};
            bool     gpuCostValid {false};
            bool     cpuFallbackUsed {false};
            double   gpuCpuCostL1Error {-1.0};
            double   gpuCpuCostMaxError {-1.0};
            double   gpuCpuSelectedRatioDelta {-1.0};
            double   gpuCpuActualCostDelta {-1.0};
            double   gpuCpuOvershootDelta {-1.0};
        };

        struct GaussianProjectedCostGpuChunkRecord
        {
            glm::uvec4 costAndTiles {0u};
            glm::uvec4 rangeAndArea {0u};
        };

        static_assert(sizeof(GaussianProjectedCostGpuChunkRecord) == 32,
                      "A4 GPU projected-cost chunk record must match shader layout");

        struct GaussianProjectedCostGpuCamera
        {
            glm::mat4 view {1.0f};
            glm::mat4 projection {1.0f};
            glm::vec4 resolution {1.0f, 1.0f, 1.0f, 1.0f};
            glm::vec4 params {0.1f, 1000.0f, glm::radians(60.0f), 0.0f};
        };

        struct GaussianProjectedCostGpuCameraBlock
        {
            GaussianProjectedCostGpuCamera cameras[2];
        };

        struct GaussianProjectedCostGpuPushConstants
        {
            uint32_t sourceCapacity {0u};
            uint32_t chunkSize {1u};
            uint32_t chunkCount {0u};
            uint32_t cameraCount {1u};
            glm::vec4 gazeAndRings {0.5f, 0.5f, 12.0f, 32.0f};
        };

        struct GaussianProjectedCostGpuBuildContext
        {
            rhi::RenderDevice*          rd {nullptr};
            rhi::ShaderLibraryRuntime* shaderLib {nullptr};
        };

        struct GaussianProjectedCostGpuBuildResult
        {
            std::vector<GaussianProjectedChunkCost> chunks;
            uint64_t totalProjectedCost {0u};
            uint64_t numTileIntersections {0u};
            double   sumProjectedAreaPx {0.0};
            double   buildGpuMs {-1.0};
            double   readbackCpuMs {-1.0};
            bool     valid {false};
            bool     fallbackUsed {false};
        };

        struct GaussianProjectedCostSampleCacheKey
        {
            uint64_t    sceneEpoch {0u};
            uint64_t    hash {0u};
            const void* packedSourcesData {nullptr};
            const void* drawData {nullptr};
            const void* selectedSourcesData {nullptr};
            uint32_t    packedSourcesSize {0u};
            uint32_t    drawCount {0u};
            uint32_t    selectedSourcesSize {0u};
            uint32_t    sourceCapacity {0u};
            uint32_t    cameraCount {0u};
            uint32_t    extentWidth {0u};
            uint32_t    extentHeight {0u};
            bool        directPrefix {false};

            [[nodiscard]] bool operator==(const GaussianProjectedCostSampleCacheKey& other) const
            {
                return sceneEpoch == other.sceneEpoch &&
                       hash == other.hash &&
                       packedSourcesData == other.packedSourcesData &&
                       drawData == other.drawData &&
                       selectedSourcesData == other.selectedSourcesData &&
                       packedSourcesSize == other.packedSourcesSize &&
                       drawCount == other.drawCount &&
                       selectedSourcesSize == other.selectedSourcesSize &&
                       sourceCapacity == other.sourceCapacity &&
                       cameraCount == other.cameraCount &&
                       extentWidth == other.extentWidth &&
                       extentHeight == other.extentHeight &&
                       directPrefix == other.directPrefix;
            }
        };

        struct GaussianProjectedCostPersistentSampleCacheEntry
        {
            GaussianProjectedCostSampleCacheKey key {};
            std::vector<GaussianProjectedCostSample> samples;
            std::vector<GaussianProjectedChunkCost>  chunks;
            uint64_t totalProjectedCost {0u};
            uint64_t regionHash {0u};
            std::array<double, 3> fullRegionAreaPx {0.0, 0.0, 0.0};
            uint32_t chunkSize {0u};
            uint64_t lastUsed {0u};
            bool valid {false};
            bool chunksValid {false};
            bool regionsValid {false};
        };

        struct GaussianProjectedCostPersistentSampleCache
        {
            static constexpr size_t kMaxEntries = 16u;
            std::array<GaussianProjectedCostPersistentSampleCacheEntry, kMaxEntries> entries {};
            uint64_t clock {0u};

            [[nodiscard]] uint64_t entryCount() const
            {
                uint64_t count = 0u;
                for (const auto& entry : entries)
                {
                    if (entry.valid)
                        ++count;
                }
                return count;
            }
        };

        struct GaussianProjectedCostFrameSampleCache
        {
            std::vector<GaussianProjectedCostSample>*       samples {nullptr};
            const std::vector<GaussianProjectedChunkCost>*  chunks {nullptr};
            GaussianProjectedCostPersistentSampleCacheEntry* entry {nullptr};
            GaussianProjectedCostSampleCacheKey             key {};
            bool                                            valid {false};
            bool                                            chunksValid {false};
            uint64_t                                        cacheHit {0u};
            uint64_t                                        cacheMiss {0u};
            bool                                            sampleReused {false};
            uint64_t                                        samplesBuilt {0u};
            uint64_t                                        samplesReused {0u};
            uint64_t                                        cacheEntryCount {0u};
            uint64_t                                        cacheLookupCount {0u};
            uint64_t                                        cacheHitCount {0u};
            uint64_t                                        cacheMissCount {0u};
            uint64_t                                        cacheFullMissCount {0u};
            uint64_t                                        cacheBuildCount {0u};
            uint64_t                                        cacheEvictionCount {0u};
            uint64_t                                        sampleCacheKeyHash {0u};
            uint64_t                                        chunkCacheHitCount {0u};
            uint64_t                                        chunkCacheMissCount {0u};
            uint64_t                                        chunkTotalProjectedCost {0u};
            double                                          chunkAggregationCpuMs {0.0};
            double                                          cacheLookupCpuMs {0.0};
            double                                          regionRebuildCpuMs {0.0};

            [[nodiscard]] double hitRate() const
            {
                const uint64_t total = cacheHit + cacheMiss;
                return total > 0u ? static_cast<double>(cacheHit) / static_cast<double>(total) : -1.0;
            }
        };

        uint64_t hashCombine64(uint64_t seed, const uint64_t value)
        {
            seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u);
            return seed;
        }

        uint64_t hashFloat64(uint64_t seed, const float value)
        {
            return hashCombine64(seed, static_cast<uint64_t>(std::bit_cast<uint32_t>(value)));
        }

        uint64_t hashVec2(uint64_t seed, const glm::vec2 value)
        {
            seed = hashFloat64(seed, value.x);
            return hashFloat64(seed, value.y);
        }

        uint64_t hashVec3(uint64_t seed, const glm::vec3 value)
        {
            seed = hashFloat64(seed, value.x);
            seed = hashFloat64(seed, value.y);
            return hashFloat64(seed, value.z);
        }

        uint64_t hashVec4(uint64_t seed, const glm::vec4 value)
        {
            seed = hashFloat64(seed, value.x);
            seed = hashFloat64(seed, value.y);
            seed = hashFloat64(seed, value.z);
            return hashFloat64(seed, value.w);
        }

        uint64_t hashMat4(uint64_t seed, const glm::mat4& value)
        {
            for (uint32_t column = 0u; column < 4u; ++column)
                seed = hashVec4(seed, value[column]);
            return seed;
        }

        struct GaussianCoverageGuardRiskStats
        {
            uint32_t activePrefixCount {0};
            bool     modeEnabled {false};
            uint32_t riskSectors {0u};
            uint32_t diagnosticRiskSectors {0u};
            bool     riskDetected {false};
            bool     active {false};
            uint32_t addedCount {0u};
            uint64_t addedCost {0u};
            uint64_t budgetCap {0u};
            double   analysisCpuMs {0.0};
            double   projectedSampleBuildCpuMs {0.0};
            double   coverageAreaCpuMs {0.0};
            double   candidateScanCpuMs {0.0};
            double   regionRebuildCpuMs {0.0};
            uint64_t cacheFullMissCount {0u};
            uint64_t cacheHitCount {0u};
            uint64_t cacheMissCount {0u};
            uint64_t geometryCacheFullMissCount {0u};
            uint64_t geometryCacheHitCount {0u};
            uint64_t geometryCacheMissCount {0u};
            uint64_t geometrySamplesBuilt {0u};
            uint64_t geometrySamplesReused {0u};
            std::array<float, 3> coverageFailureBeforeByRegion {-1.0f, -1.0f, -1.0f};
            std::array<float, 3> coverageFailureAfterByRegion {-1.0f, -1.0f, -1.0f};
            std::array<uint32_t, 3> addedCountByRegion {0u, 0u, 0u};
            std::array<uint64_t, 3> addedCostByRegion {0u, 0u, 0u};
            std::array<uint32_t, 3> candidateCountByRegion {0u, 0u, 0u};
            uint32_t candidateCount {0u};
            uint32_t trueCandidateCount {0u};
            uint32_t maxAddsEffective {0u};
            bool     candidateCountTruncated {false};
            uint32_t requiredExtraCountEstimate {UINT32_MAX};
            double   requiredExtraAreaPx {-1.0};
            double   deficitAreaBefore {-1.0};
            double   deficitAreaAfter {-1.0};
            uint32_t addedCountToFailingSector {0u};
            uint64_t addedCostToFailingSector {0u};
            uint32_t failingSectorId {UINT32_MAX};
            double   repairSectorMatchRate {-1.0};
            bool     cappedByMaxAdds {false};
            bool     cappedByCost {false};
            bool     noCandidate {false};
            float    coverageFailureBefore {0.0f};
            float    coverageFailureAfter {0.0f};
            float    coverageScoreBefore {1.0f};
            float    coverageScoreAfter {1.0f};
        };

        double elapsedCpuMs(const std::chrono::steady_clock::time_point start)
        {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }

        float smoothstep01(const float x)
        {
            const float t = std::clamp(x, 0.0f, 1.0f);
            return t * t * (3.0f - 2.0f * t);
        }


        rhi::Extent2D safeGaussianSplatCostExtent(rhi::Extent2D extent);

        GaussianProjectedCostSampleCacheKey makeGaussianProjectedCostSampleCacheKey(
            const GaussianSplatRenderSettings& settings,
            const resource::GpuSceneView&      gpuSceneView,
            const uint32_t                     sourceCapacity,
            const bool                         directPrefix,
            const std::span<const RenderCamera> cameras,
            const rhi::Extent2D                fallbackExtent,
            const uint64_t                     sceneEpoch)
        {
            const rhi::Extent2D extent = safeGaussianSplatCostExtent(fallbackExtent);
            GaussianProjectedCostSampleCacheKey key {
                .sceneEpoch          = sceneEpoch,
                .packedSourcesData   = gpuSceneView.generalGaussianSplatPackedSources.data(),
                .drawData            = gpuSceneView.generalGaussianSplatDraws.data(),
                .selectedSourcesData = gpuSceneView.generalGaussianSplatSelectedSources.data(),
                .packedSourcesSize   =
                    static_cast<uint32_t>(gpuSceneView.generalGaussianSplatPackedSources.size()),
                .drawCount           = static_cast<uint32_t>(gpuSceneView.generalGaussianSplatDraws.size()),
                .selectedSourcesSize =
                    static_cast<uint32_t>(gpuSceneView.generalGaussianSplatSelectedSources.size()),
                .sourceCapacity      = sourceCapacity,
                .cameraCount         = static_cast<uint32_t>(cameras.size()),
                .extentWidth         = extent.width,
                .extentHeight        = extent.height,
                .directPrefix        = directPrefix,
            };

            uint64_t hash = 0xcbf29ce484222325ull;
            hash = hashCombine64(hash, sceneEpoch);
            hash = hashCombine64(hash, reinterpret_cast<uintptr_t>(key.packedSourcesData));
            hash = hashCombine64(hash, reinterpret_cast<uintptr_t>(key.drawData));
            hash = hashCombine64(hash, reinterpret_cast<uintptr_t>(key.selectedSourcesData));
            hash = hashCombine64(hash, key.packedSourcesSize);
            hash = hashCombine64(hash, key.drawCount);
            hash = hashCombine64(hash, key.selectedSourcesSize);
            hash = hashCombine64(hash, sourceCapacity);
            hash = hashCombine64(hash, directPrefix ? 1u : 0u);
            hash = hashCombine64(hash, extent.width);
            hash = hashCombine64(hash, extent.height);
            // The cached sample stores geometry-only screen-space projection.
            // Gaze, ring thresholds, guard thresholds, scheduler budget, and
            // chunk size affect later region/selection stages and are kept out
            // of this key so guard analysis can reuse projection geometry.
            if (cameras.size() > GaussianProjectedCostSample::kMaxCameraFootprints)
            {
                hash = hashVec2(hash, settings.foveatedGaze);
                hash = hashVec2(hash, settings.foveatedRingDegrees);
            }

            for (const RenderCamera& camera : cameras)
            {
                hash = hashCombine64(hash, camera.viewIndex);
                hash = hashCombine64(hash, camera.viewCount);
                hash = hashCombine64(hash, camera.isXRView ? 1u : 0u);
                hash = hashMat4(hash, camera.view);
                hash = hashMat4(hash, camera.projection);
                hash = hashFloat64(hash, camera.zNear);
                hash = hashFloat64(hash, camera.zFar);
                hash = hashFloat64(hash, camera.fovY);
            }
            key.hash = hash;
            return key;
        }

        void noteGaussianProjectedCostSampleReuse(GaussianProjectedCostFrameSampleCache& cache,
                                                  const uint64_t                         sampleCount)
        {
            if (!cache.valid || sampleCount == 0u)
                return;
            cache.cacheHit += sampleCount;
            cache.samplesReused += sampleCount;
            cache.sampleReused = true;
        }

        uint64_t gaussianProjectedCostRegionHash(const GaussianSplatRenderSettings& settings,
                                                 const std::span<const RenderCamera> cameras)
        {
            uint64_t hash = 0x8f17b7c4e3a21d5bull;
            hash = hashVec2(hash, settings.foveatedGaze);
            hash = hashVec2(hash, settings.foveatedRingDegrees);
            for (const RenderCamera& camera : cameras)
            {
                hash = hashCombine64(hash, camera.viewIndex);
                hash = hashCombine64(hash, camera.viewCount);
                hash = hashCombine64(hash, camera.isXRView ? 1u : 0u);
                hash = hashMat4(hash, camera.projection);
            }
            return hash;
        }

        GaussianProjectedCostPersistentSampleCache& gaussianProjectedCostPersistentSampleCache()
        {
            thread_local GaussianProjectedCostPersistentSampleCache cache {};
            return cache;
        }

        glm::vec3 decodeGeneralGaussianSplatPositionCpu(const resource::GpuGeneralGaussianSplatPackedSource& src)
        {
            return glm::vec3 {std::bit_cast<float>(src.posOpacity.x),
                              std::bit_cast<float>(src.posOpacity.y),
                              std::bit_cast<float>(src.posOpacity.z)};
        }

        glm::mat3 decodeGeneralGaussianSplatCovarianceCpu(const resource::GpuGeneralGaussianSplatPackedSource& src)
        {
            const glm::vec2 p0 = glm::unpackHalf2x16(src.covariance0.x);
            const glm::vec2 p1 = glm::unpackHalf2x16(src.covariance0.y);
            const glm::vec2 p2 = glm::unpackHalf2x16(src.covariance0.z);

            glm::mat3 sigma(0.0f);
            sigma[0][0] = p0.x;
            sigma[1][0] = p0.y;
            sigma[0][1] = p0.y;
            sigma[2][0] = p1.x;
            sigma[0][2] = p1.x;
            sigma[1][1] = p1.y;
            sigma[2][1] = p2.x;
            sigma[1][2] = p2.x;
            sigma[2][2] = p2.y;
            return sigma;
        }

        glm::mat3 buildGaussianSplatProjectionJacobianCpu(const glm::vec3 camspace, const glm::vec2 focal)
        {
            float z = camspace.z;
            if (std::abs(z) < 1e-4f)
                z = z < 0.0f ? -1e-4f : 1e-4f;

            glm::mat3 jacobian(0.0f);
            jacobian[0] = glm::vec3 {focal.x / z, 0.0f, -(focal.x * camspace.x) / (z * z)};
            jacobian[1] = glm::vec3 {0.0f, -focal.y / z, (focal.y * camspace.y) / (z * z)};
            return jacobian;
        }

        rhi::Extent2D safeGaussianSplatCostExtent(const rhi::Extent2D extent)
        {
            return rhi::Extent2D {
                .width  = std::max(extent.width, 1u),
                .height = std::max(extent.height, 1u),
            };
        }

        float gaussianSplatFoveatedEccentricityDegreesCpu(const glm::vec2 centerNdc,
                                                          const glm::vec2 gazeUv,
                                                          const RenderCamera& camera)
        {
            const glm::vec2 tanHalfFov {
                1.0f / std::max(std::abs(camera.projection[0][0]), 1e-5f),
                1.0f / std::max(std::abs(camera.projection[1][1]), 1e-5f),
            };
            const float projectionYSign = camera.projection[1][1] < 0.0f ? -1.0f : 1.0f;
            const glm::vec2 gazeNdc {
                gazeUv.x * 2.0f - 1.0f,
                projectionYSign * (1.0f - gazeUv.y * 2.0f),
            };
            const glm::vec2 deltaTan = (centerNdc - gazeNdc) * glm::max(tanHalfFov, glm::vec2 {1e-5f});
            return glm::degrees(std::atan(glm::length(deltaTan)));
        }

        float gaussianSplatGazeAngularDistanceDegreesCpu(const glm::vec2 gazeA,
                                                         const glm::vec2 gazeB,
                                                         const RenderCamera& camera)
        {
            const glm::vec2 tanHalfFov {
                1.0f / std::max(std::abs(camera.projection[0][0]), 1e-5f),
                1.0f / std::max(std::abs(camera.projection[1][1]), 1e-5f),
            };
            const float projectionYSign = camera.projection[1][1] < 0.0f ? -1.0f : 1.0f;
            const glm::vec2 ndcA {
                gazeA.x * 2.0f - 1.0f,
                projectionYSign * (1.0f - gazeA.y * 2.0f),
            };
            const glm::vec2 ndcB {
                gazeB.x * 2.0f - 1.0f,
                projectionYSign * (1.0f - gazeB.y * 2.0f),
            };
            const glm::vec2 deltaTan = (ndcA - ndcB) * glm::max(tanHalfFov, glm::vec2 {1e-5f});
            return glm::degrees(std::atan(glm::length(deltaTan)));
        }

        uint32_t gaussianSplatCoverageSector(const GaussianSplatRenderSettings& settings,
                                             const float                        eccentricityDegrees)
        {
            const float foveaDegrees = std::max(settings.foveatedRingDegrees.x, 0.0f);
            const float midDegrees   = std::max(settings.foveatedRingDegrees.y, foveaDegrees);
            if (eccentricityDegrees <= foveaDegrees)
                return 0u;
            if (eccentricityDegrees <= midDegrees)
                return 1u;
            return 2u;
        }

        void rebuildGaussianProjectedCostSampleRegions(const GaussianSplatRenderSettings&     settings,
                                                       const std::span<const RenderCamera>    cameras,
                                                       GaussianProjectedCostFrameSampleCache& frameCache)
        {
            if (!frameCache.valid || frameCache.entry == nullptr || frameCache.samples == nullptr)
                return;

            auto& entry = *frameCache.entry;
            if (cameras.size() > GaussianProjectedCostSample::kMaxCameraFootprints)
                return;

            const uint64_t regionHash = gaussianProjectedCostRegionHash(settings, cameras);
            if (entry.regionsValid && entry.regionHash == regionHash)
                return;

            const auto rebuildStart = std::chrono::steady_clock::now();
            entry.fullRegionAreaPx = {0.0, 0.0, 0.0};
            for (GaussianProjectedCostSample& sample : *frameCache.samples)
            {
                sample.regionProjectedAreaPx = {0.0, 0.0, 0.0};
                sample.regionMinEccentricityDegrees = {
                    std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity()};

                const uint32_t cameraCount =
                    std::min<uint32_t>(static_cast<uint32_t>(cameras.size()),
                                       GaussianProjectedCostSample::kMaxCameraFootprints);
                for (uint32_t cameraIndex = 0u; cameraIndex < cameraCount; ++cameraIndex)
                {
                    if (sample.cameraVisible[cameraIndex] == 0u ||
                        sample.cameraProjectedAreaPx[cameraIndex] <= 0.0)
                    {
                        continue;
                    }

                    const float eccentricityDegrees =
                        gaussianSplatFoveatedEccentricityDegreesCpu(sample.cameraCenterNdc[cameraIndex],
                                                                     settings.foveatedGaze,
                                                                     cameras[cameraIndex]);
                    const uint32_t coverageSector =
                        std::min<uint32_t>(gaussianSplatCoverageSector(settings, eccentricityDegrees), 2u);
                    sample.regionProjectedAreaPx[coverageSector] += sample.cameraProjectedAreaPx[cameraIndex];
                    entry.fullRegionAreaPx[coverageSector] += sample.cameraProjectedAreaPx[cameraIndex];
                    sample.regionMinEccentricityDegrees[coverageSector] =
                        std::min(sample.regionMinEccentricityDegrees[coverageSector], eccentricityDegrees);
                }
            }

            entry.regionsValid = true;
            entry.regionHash = regionHash;
            frameCache.regionRebuildCpuMs += elapsedCpuMs(rebuildStart);
        }

        GaussianProjectedCostSample estimateGaussianSplatProjectedTileCost(
            const resource::GpuGeneralGaussianSplatPackedSource& src,
            const resource::GpuGeneralGaussianSplatDrawRecord&   draw,
            const GaussianSplatRenderSettings&                    settings,
            const std::span<const RenderCamera>                  cameras,
            const rhi::Extent2D                                  fallbackExtent)
        {
            constexpr float kTileSizePx = 16.0f;

            GaussianProjectedCostSample result {};
            const rhi::Extent2D extent = safeGaussianSplatCostExtent(fallbackExtent);
            const glm::vec2     viewport {static_cast<float>(extent.width), static_cast<float>(extent.height)};
            const glm::vec3     localPos    = decodeGeneralGaussianSplatPositionCpu(src);
            const glm::mat4     model       = draw.model;
            const glm::mat3     modelLinear = glm::mat3(model);
            const glm::vec3     worldPos    = glm::vec3(model * glm::vec4(localPos, 1.0f));
            const glm::mat3     sigmaLocal  = decodeGeneralGaussianSplatCovarianceCpu(src);
            const glm::mat3     sigmaWorld  = modelLinear * sigmaLocal * glm::transpose(modelLinear);

            uint64_t tileIntersections = 0;
            double   projectedAreaPx   = 0.0;

            for (size_t cameraIndex = 0u; cameraIndex < cameras.size(); ++cameraIndex)
            {
                const RenderCamera& camera = cameras[cameraIndex];
                const glm::vec4 posView = camera.view * glm::vec4(worldPos, 1.0f);
                const glm::vec4 posClip = camera.projection * posView;
                if (posClip.w <= 1e-5f)
                    continue;

                const glm::vec3 centerNdc = glm::vec3(posClip) / posClip.w;
                const float     bounds    = 1.2f * posClip.w;
                if (centerNdc.z <= 0.0f || centerNdc.z >= 1.0f)
                    continue;
                if (posClip.x < -bounds || posClip.x > bounds || posClip.y < -bounds || posClip.y > bounds)
                    continue;
                const float eccentricityDegrees =
                    gaussianSplatFoveatedEccentricityDegreesCpu(glm::vec2 {centerNdc.x, centerNdc.y},
                                                                 settings.foveatedGaze,
                                                                 camera);
                const uint32_t coverageSector =
                    std::min<uint32_t>(gaussianSplatCoverageSector(settings, eccentricityDegrees), 2u);

                const glm::vec2 focal {
                    0.5f * std::abs(camera.projection[0][0]) * viewport.x,
                    0.5f * std::abs(camera.projection[1][1]) * viewport.y,
                };
                const glm::mat3 jacobian = buildGaussianSplatProjectionJacobianCpu(glm::vec3(posView), focal);
                const glm::mat3 worldToView =
                    glm::transpose(glm::mat3 {glm::vec3(camera.view[0]),
                                              glm::vec3(camera.view[1]),
                                              glm::vec3(camera.view[2])});
                const glm::mat3 transform = worldToView * jacobian;
                const glm::mat3 cov       = glm::transpose(transform) * sigmaWorld * transform;

                const float kernelSize  = std::max(draw.params0.x, 1e-4f);
                const float diagonal1   = cov[0][0] + kernelSize;
                const float offDiagonal = cov[0][1];
                const float diagonal2   = cov[1][1] + kernelSize;
                const float mid         = 0.5f * (diagonal1 + diagonal2);
                const float radius      = glm::length(glm::vec2 {(diagonal1 - diagonal2) * 0.5f, offDiagonal});
                const float lambda1     = std::max(mid + radius, 1e-4f);
                const float lambda2     = std::max(mid - radius, 0.1f);

                glm::vec2 diagonalVector {offDiagonal, lambda1 - diagonal1};
                if (glm::length(diagonalVector) < 1e-6f)
                    diagonalVector = glm::vec2 {1.0f, 0.0f};
                else
                    diagonalVector = glm::normalize(diagonalVector);

                const float cutoffScale = std::max(draw.params0.y, 1e-3f);
                const glm::vec2 v1      = std::sqrt(2.0f * lambda1) * diagonalVector * cutoffScale;
                const glm::vec2 v2 =
                    std::sqrt(2.0f * lambda2) * glm::vec2 {diagonalVector.y, -diagonalVector.x} * cutoffScale;
                const glm::vec2 halfExtent = glm::abs(v1) + glm::abs(v2);
                if (!std::isfinite(halfExtent.x) || !std::isfinite(halfExtent.y))
                    continue;

                const glm::vec2 centerPx {
                    (centerNdc.x * 0.5f + 0.5f) * viewport.x,
                    (centerNdc.y * 0.5f + 0.5f) * viewport.y,
                };
                const float left   = std::clamp(centerPx.x - halfExtent.x, 0.0f, viewport.x);
                const float right  = std::clamp(centerPx.x + halfExtent.x, 0.0f, viewport.x);
                const float top    = std::clamp(centerPx.y - halfExtent.y, 0.0f, viewport.y);
                const float bottom = std::clamp(centerPx.y + halfExtent.y, 0.0f, viewport.y);
                if (right <= left || bottom <= top)
                    continue;

                const uint32_t tileColumns =
                    std::max(1u, static_cast<uint32_t>(std::ceil(viewport.x / kTileSizePx)));
                const uint32_t tileRows =
                    std::max(1u, static_cast<uint32_t>(std::ceil(viewport.y / kTileSizePx)));
                const uint32_t tileX0 =
                    std::min(tileColumns, static_cast<uint32_t>(std::floor(left / kTileSizePx)));
                const uint32_t tileY0 =
                    std::min(tileRows, static_cast<uint32_t>(std::floor(top / kTileSizePx)));
                const uint32_t tileX1 =
                    std::min(tileColumns,
                             static_cast<uint32_t>(std::ceil(right / kTileSizePx)));
                const uint32_t tileY1 =
                    std::min(tileRows,
                             static_cast<uint32_t>(std::ceil(bottom / kTileSizePx)));
                if (tileX1 <= tileX0 || tileY1 <= tileY0)
                    continue;
                const uint64_t viewTileIntersections =
                    static_cast<uint64_t>(tileX1 - tileX0) * static_cast<uint64_t>(tileY1 - tileY0);
                tileIntersections += viewTileIntersections;
                const double viewProjectedAreaPx =
                    static_cast<double>(right - left) * static_cast<double>(bottom - top);
                projectedAreaPx += viewProjectedAreaPx;
                if (cameraIndex < GaussianProjectedCostSample::kMaxCameraFootprints)
                {
                    result.cameraVisible[cameraIndex] = 1u;
                    result.cameraProjectedAreaPx[cameraIndex] = viewProjectedAreaPx;
                    result.cameraCenterNdc[cameraIndex] = glm::vec2 {centerNdc.x, centerNdc.y};
                }
                result.regionProjectedAreaPx[coverageSector] += viewProjectedAreaPx;
                result.regionMinEccentricityDegrees[coverageSector] =
                    std::min(result.regionMinEccentricityDegrees[coverageSector], eccentricityDegrees);
            }

            result.tileIntersections = tileIntersections;
            result.cost              = std::max<uint64_t>(1u, tileIntersections);
            result.projectedAreaPx   = projectedAreaPx;
            return result;
        }

        resource::GpuGeneralGaussianSplatSelectedSource gaussianSplatPrefixSourceForRank(
            const resource::GpuSceneView& gpuSceneView,
            const uint32_t                rank,
            const bool                    directPrefix)
        {
            resource::GpuGeneralGaussianSplatSelectedSource selection {};
            if (directPrefix)
            {
                selection.sourceIndex  = rank;
                selection.drawIndex    = 0u;
                selection.packedWeight = std::bit_cast<uint32_t>(1.0f);
                selection.flags        = 0u;
                return selection;
            }

            if (rank < gpuSceneView.generalGaussianSplatSelectedSources.size())
                return gpuSceneView.generalGaussianSplatSelectedSources[rank];
            selection.flags = 0x80000000u;
            return selection;
        }

        void collectGaussianSplatSelectedSourceIds(const resource::GpuSceneView& gpuSceneView,
                                                   const uint32_t                activeCount,
                                                   const bool                    directPrefix,
                                                   GaussianSplatFrameStats&      stats)
        {
            stats.selectedSourceIds.clear();
            if (!stats.selectedIdDebugLoggingEnabled || activeCount == 0u)
                return;

            stats.selectedSourceIds.reserve(activeCount);
            for (uint32_t rank = 0u; rank < activeCount; ++rank)
            {
                const auto selection = gaussianSplatPrefixSourceForRank(gpuSceneView, rank, directPrefix);
                if ((selection.flags & 0x80000000u) != 0u)
                    continue;

                uint32_t sourceId = selection.sourceIndex;
                if (selection.sourceIndex < gpuSceneView.generalGaussianSplatPackedSources.size())
                    sourceId = gpuSceneView.generalGaussianSplatPackedSources[selection.sourceIndex].aux0.x;
                stats.selectedSourceIds.push_back(sourceId);
            }
        }

        uint64_t gaussianCachedSelectionOracleMix(uint64_t value)
        {
            value += 0x9e3779b97f4a7c15ull;
            value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
            value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
            return value ^ (value >> 31u);
        }

        uint32_t gaussianCachedSelectionOracleSourceId(
            const resource::GpuSceneView& gpuSceneView,
            const resource::GpuGeneralGaussianSplatSelectedSource& selection)
        {
            if (selection.sourceIndex < gpuSceneView.generalGaussianSplatPackedSources.size())
                return gpuSceneView.generalGaussianSplatPackedSources[selection.sourceIndex].aux0.x;
            return selection.sourceIndex;
        }

        uint64_t gaussianCachedSelectionOracleSelectionKey(
            const resource::GpuSceneView& gpuSceneView,
            const resource::GpuGeneralGaussianSplatSelectedSource& selection,
            const uint32_t seed,
            const uint64_t modeSalt)
        {
            const uint64_t sourceId = gaussianCachedSelectionOracleSourceId(gpuSceneView, selection);
            uint64_t value = sourceId ^
                             (static_cast<uint64_t>(selection.sourceIndex) << 17u) ^
                             (static_cast<uint64_t>(selection.drawIndex) << 33u) ^
                             (static_cast<uint64_t>(seed) << 1u) ^
                             modeSalt;
            return gaussianCachedSelectionOracleMix(value);
        }

        float gaussianCachedSelectionOracleHashThreshold(const uint64_t key)
        {
            return static_cast<float>(static_cast<double>(key >> 11u) *
                                      (1.0 / 9007199254740992.0));
        }

        float gaussianCachedSelectionOracleOpacityProxy(
            const resource::GpuGeneralGaussianSplatPackedSource& source,
            const resource::GpuGeneralGaussianSplatDrawRecord&   draw,
            const resource::GpuGeneralGaussianSplatSelectedSource& selection)
        {
            const glm::vec2 ba = glm::unpackHalf2x16(source.colorSh0.y);
            const float baseOpacity = std::clamp(ba.y, 0.0f, 1.0f);
            const float drawOpacityScale = std::max(draw.params0.z, 0.0f);
            float selectionWeight = std::bit_cast<float>(selection.packedWeight);
            if (!std::isfinite(selectionWeight) || selectionWeight <= 0.0f)
                selectionWeight = 1.0f;
            return std::clamp(baseOpacity * drawOpacityScale * selectionWeight, 0.0f, 1.0f);
        }

        std::vector<GaussianProjectedCostSample>& ensureGaussianProjectedCostSamples(
            const GaussianSplatRenderSettings&          settings,
            const resource::GpuSceneView&               gpuSceneView,
            const uint32_t                              sourceCapacity,
            const bool                                  directPrefix,
            const std::span<const RenderCamera>         cameras,
            const rhi::Extent2D                         fallbackExtent,
            const uint64_t                              sceneEpoch,
            GaussianProjectedCostFrameSampleCache&      frameCache)
        {
            if (frameCache.valid && frameCache.samples != nullptr &&
                frameCache.samples->size() >= sourceCapacity)
            {
                return *frameCache.samples;
            }

            const auto lookupStart = std::chrono::steady_clock::now();
            const auto key = makeGaussianProjectedCostSampleCacheKey(settings,
                                                                     gpuSceneView,
                                                                     sourceCapacity,
                                                                     directPrefix,
                                                                     cameras,
                                                                     fallbackExtent,
                                                                     sceneEpoch);
            auto& persistentCache = gaussianProjectedCostPersistentSampleCache();
            ++persistentCache.clock;
            ++frameCache.cacheLookupCount;
            frameCache.sampleCacheKeyHash = key.hash;

            GaussianProjectedCostPersistentSampleCacheEntry* firstInvalid = nullptr;
            GaussianProjectedCostPersistentSampleCacheEntry* lruEntry     = nullptr;
            for (auto& entry : persistentCache.entries)
            {
                if (!entry.valid)
                {
                    if (firstInvalid == nullptr)
                        firstInvalid = &entry;
                    continue;
                }
                if (entry.key == key && entry.samples.size() >= sourceCapacity)
                {
                    entry.lastUsed = persistentCache.clock;
                    frameCache.samples = &entry.samples;
                    frameCache.entry   = &entry;
                    frameCache.key     = key;
                    frameCache.valid   = true;
                    frameCache.cacheHit += sourceCapacity;
                    ++frameCache.cacheHitCount;
                    frameCache.samplesReused += sourceCapacity;
                    frameCache.sampleReused = true;
                    frameCache.cacheEntryCount = persistentCache.entryCount();
                    frameCache.cacheLookupCpuMs += elapsedCpuMs(lookupStart);
                    return *frameCache.samples;
                }
                if (lruEntry == nullptr || entry.lastUsed < lruEntry->lastUsed)
                    lruEntry = &entry;
            }

            GaussianProjectedCostPersistentSampleCacheEntry* entry =
                firstInvalid != nullptr ? firstInvalid : lruEntry;
            if (entry == nullptr)
            {
                // kMaxEntries is nonzero, so this should be unreachable. Keep a
                // guarded fallback to avoid dereferencing null in release builds.
                static thread_local GaussianProjectedCostPersistentSampleCacheEntry fallback {};
                entry = &fallback;
            }
            if (entry->valid)
            {
                ++frameCache.cacheEvictionCount;
                entry->chunks.clear();
                entry->chunksValid = false;
                entry->regionsValid = false;
                entry->regionHash = 0u;
                entry->fullRegionAreaPx = {0.0, 0.0, 0.0};
            }
            ++frameCache.cacheMissCount;
            ++frameCache.cacheFullMissCount;
            ++frameCache.cacheBuildCount;
            frameCache.cacheLookupCpuMs += elapsedCpuMs(lookupStart);

            entry->samples.clear();
            entry->samples.resize(sourceCapacity);
            std::array<double, 3> fullRegionAreaPx {0.0, 0.0, 0.0};
            for (uint32_t rank = 0u; rank < sourceCapacity; ++rank)
            {
                const auto selection = gaussianSplatPrefixSourceForRank(gpuSceneView, rank, directPrefix);
                GaussianProjectedCostSample sample {};
                if ((selection.flags & 0x80000000u) == 0u &&
                    selection.sourceIndex < gpuSceneView.generalGaussianSplatPackedSources.size() &&
                    selection.drawIndex < gpuSceneView.generalGaussianSplatDraws.size())
                {
                    sample = estimateGaussianSplatProjectedTileCost(
                        gpuSceneView.generalGaussianSplatPackedSources[selection.sourceIndex],
                        gpuSceneView.generalGaussianSplatDraws[selection.drawIndex],
                        settings,
                        cameras,
                        fallbackExtent);
                }
                for (uint32_t region = 0u; region < 3u; ++region)
                    fullRegionAreaPx[region] += sample.regionProjectedAreaPx[region];
                entry->samples[rank] = sample;
            }
            entry->key                = key;
            entry->lastUsed           = persistentCache.clock;
            entry->valid              = true;
            entry->chunksValid        = false;
            entry->regionsValid       = true;
            entry->regionHash         = gaussianProjectedCostRegionHash(settings, cameras);
            entry->fullRegionAreaPx   = fullRegionAreaPx;
            entry->chunkSize          = 0u;
            entry->totalProjectedCost = 0u;
            entry->chunks.clear();

            frameCache.samples = &entry->samples;
            frameCache.entry   = entry;
            frameCache.key     = key;
            frameCache.valid   = true;
            frameCache.cacheMiss += sourceCapacity;
            frameCache.samplesBuilt += sourceCapacity;
            frameCache.cacheEntryCount = persistentCache.entryCount();
            return *frameCache.samples;
        }

        const std::vector<GaussianProjectedChunkCost>& ensureGaussianProjectedCostChunks(
            const std::vector<GaussianProjectedCostSample>& projectedSamples,
            const uint32_t                                  sourceCapacity,
            const uint32_t                                  chunkSize,
            GaussianProjectedCostFrameSampleCache&          frameCache)
        {
            if (frameCache.chunksValid && frameCache.chunks != nullptr)
                return *frameCache.chunks;

            auto* entry = frameCache.entry;
            if (entry != nullptr && entry->valid && entry->chunksValid && entry->chunkSize == chunkSize)
            {
                frameCache.chunks = &entry->chunks;
                frameCache.chunksValid = true;
                frameCache.chunkTotalProjectedCost = entry->totalProjectedCost;
                ++frameCache.chunkCacheHitCount;
                return *frameCache.chunks;
            }

            ++frameCache.chunkCacheMissCount;
            const auto aggregationStart = std::chrono::steady_clock::now();
            std::vector<GaussianProjectedChunkCost>* chunks = nullptr;
            if (entry != nullptr)
            {
                entry->chunks.clear();
                entry->chunks.reserve((sourceCapacity + chunkSize - 1u) / chunkSize);
                entry->totalProjectedCost = 0u;
                chunks = &entry->chunks;
            }
            else
            {
                static thread_local std::vector<GaussianProjectedChunkCost> fallbackChunks {};
                fallbackChunks.clear();
                fallbackChunks.reserve((sourceCapacity + chunkSize - 1u) / chunkSize);
                chunks = &fallbackChunks;
            }

            uint64_t totalProjectedCost = 0u;
            for (uint32_t rank = 0u; rank < sourceCapacity; ++rank)
            {
                const GaussianProjectedCostSample& sample = projectedSamples[rank];
                totalProjectedCost += sample.cost;

                if (rank % chunkSize == 0u)
                {
                    chunks->push_back(GaussianProjectedChunkCost {
                        .beginRank = rank,
                        .endRank   = rank,
                    });
                }
                GaussianProjectedChunkCost& chunk = chunks->back();
                chunk.endRank = rank + 1u;
                chunk.cost += sample.cost;
                chunk.tileIntersections += sample.tileIntersections;
                chunk.projectedAreaPx += sample.projectedAreaPx;
            }

            if (entry != nullptr)
            {
                entry->chunkSize = chunkSize;
                entry->totalProjectedCost = totalProjectedCost;
                entry->chunksValid = true;
            }
            frameCache.chunks = chunks;
            frameCache.chunksValid = true;
            frameCache.chunkTotalProjectedCost = totalProjectedCost;
            frameCache.chunkAggregationCpuMs += elapsedCpuMs(aggregationStart);
            return *frameCache.chunks;
        }

        float gaussianFoveatedRingBlendCpu(const float edge0, const float edge1, const float value)
        {
            if (edge1 <= edge0)
                return value >= edge1 ? 1.0f : 0.0f;
            const float t = std::clamp((value - edge0) / (edge1 - edge0), 0.0f, 1.0f);
            return t * t * (3.0f - 2.0f * t);
        }

        float gaussianFoveatedBaseClodLevelCpu(const GaussianSplatRenderSettings& settings,
                                               const float                        eccentricityDegrees)
        {
            const float foveaDegrees = std::max(settings.foveatedRingDegrees.x, 0.0f);
            const float midDegrees   = std::max(settings.foveatedRingDegrees.y, foveaDegrees);
            const glm::vec3 levels =
                glm::clamp(settings.foveatedRingLevels, glm::vec3 {0.0f}, glm::vec3 {1.0f});
            const auto distribution = settings.foveatedDistribution;

            if (distribution == GaussianSplatFoveatedDistribution::eHardRing ||
                (distribution != GaussianSplatFoveatedDistribution::eContinuousScheduler &&
                 distribution != GaussianSplatFoveatedDistribution::eFoveaProtectedContinuous &&
                 settings.foveatedTransitionDegrees <= 1e-4f))
            {
                if (eccentricityDegrees <= foveaDegrees)
                    return levels.x;
                if (eccentricityDegrees <= midDegrees)
                    return levels.y;
                return levels.z;
            }

            if (distribution == GaussianSplatFoveatedDistribution::eContinuousScheduler)
            {
                const float theta0 = std::max(settings.foveatedContinuousTheta0Degrees, 1e-4f);
                const float alpha  = std::max(settings.foveatedContinuousAlpha, 0.0f);
                const float minLevel =
                    std::clamp(settings.foveatedContinuousMinLevel, 0.0f, std::clamp(levels.x, 0.0f, 1.0f));
                const float budget = std::max(levels.x, minLevel);
                const float falloff =
                    std::pow(1.0f + std::max(eccentricityDegrees, 0.0f) / theta0, alpha);
                return minLevel + (budget - minLevel) / std::max(falloff, 1e-5f);
            }

            if (distribution == GaussianSplatFoveatedDistribution::eFoveaProtectedContinuous)
            {
                const float protectedDegrees = std::max(settings.foveatedRingDegrees.x, 0.0f);
                const float theta0 = std::max(settings.foveatedContinuousTheta0Degrees, 1e-4f);
                const float alpha  = std::max(settings.foveatedContinuousAlpha, 0.0f);
                const float foveaLevel = std::clamp(std::max(levels.x, 0.95f), 0.0f, 1.0f);
                const float peripheryLevel =
                    std::clamp(settings.foveatedContinuousMinLevel, 0.0f, foveaLevel);
                const float offset =
                    std::max(eccentricityDegrees - protectedDegrees, 0.0f);
                const float falloff =
                    1.0f / std::max(std::pow(1.0f + offset / theta0, alpha), 1e-5f);
                return peripheryLevel + (foveaLevel - peripheryLevel) * std::clamp(falloff, 0.0f, 1.0f);
            }

            if (distribution == GaussianSplatFoveatedDistribution::eGaussian ||
                distribution == GaussianSplatFoveatedDistribution::eExponential ||
                distribution == GaussianSplatFoveatedDistribution::eInversePower ||
                distribution == GaussianSplatFoveatedDistribution::eLogPolar ||
                distribution == GaussianSplatFoveatedDistribution::eCortical ||
                distribution == GaussianSplatFoveatedDistribution::eConeDensityFitted)
            {
                const float offset    = std::max(eccentricityDegrees - foveaDegrees, 0.0f);
                const float midOffset = std::max(midDegrees - foveaDegrees, 1e-3f);
                const float ratio =
                    std::clamp((levels.y - levels.z) / std::max(levels.x - levels.z, 1e-5f), 1e-3f, 0.999f);
                float weight = 0.0f;
                if (distribution == GaussianSplatFoveatedDistribution::eGaussian)
                {
                    const float sigma = midOffset / std::sqrt(std::max(-2.0f * std::log(ratio), 1e-4f));
                    const float x     = offset / std::max(sigma, 1e-4f);
                    weight            = std::exp(-0.5f * x * x);
                }
                else if (distribution == GaussianSplatFoveatedDistribution::eExponential)
                {
                    const float lambda = -std::log(ratio) / midOffset;
                    weight             = std::exp(-lambda * offset);
                }
                else if (distribution == GaussianSplatFoveatedDistribution::eInversePower)
                {
                    constexpr float power = 2.0f;
                    const float scale =
                        midOffset / std::pow(std::max(1.0f / ratio - 1.0f, 1e-4f), 1.0f / power);
                    const float x = offset / std::max(scale, 1e-4f);
                    weight        = 1.0f / (1.0f + std::pow(x, power));
                }
                else if (distribution == GaussianSplatFoveatedDistribution::eLogPolar)
                {
                    const float scale =
                        midOffset / std::max(std::exp(1.0f / ratio - 1.0f) - 1.0f, 1e-4f);
                    weight = 1.0f / (1.0f + std::log1p(offset / std::max(scale, 1e-4f)));
                }
                else if (distribution == GaussianSplatFoveatedDistribution::eCortical)
                {
                    const float e2 = ratio * midOffset / std::max(1.0f - ratio, 1e-4f);
                    weight = e2 / std::max(offset + e2, 1e-4f);
                }
                else
                {
                    constexpr float densityFloor = 0.04f;
                    constexpr float conePower = 2.0f;
                    constexpr float beta = 1.0f;
                    const float densityMid = std::clamp(std::pow(ratio, 1.0f / beta),
                                                        densityFloor + 1.0e-4f,
                                                        1.0f);
                    const float thetaC =
                        midOffset /
                        std::pow(std::max((1.0f - densityFloor) / (densityMid - densityFloor) - 1.0f,
                                          1.0e-4f),
                                 1.0f / conePower);
                    const float density =
                        densityFloor +
                        (1.0f - densityFloor) /
                            (1.0f + std::pow(offset / std::max(thetaC, 1.0e-4f), conePower));
                    weight = std::pow(std::clamp(density, 0.0f, 1.0f), beta);
                }
                return levels.z + (levels.x - levels.z) * std::clamp(weight, 0.0f, 1.0f);
            }

            const float halfTransition = std::max(settings.foveatedTransitionDegrees, 0.0f) * 0.5f;
            const float blend0 = gaussianFoveatedRingBlendCpu(std::max(foveaDegrees - halfTransition, 0.0f),
                                                              foveaDegrees + halfTransition,
                                                              eccentricityDegrees);
            const float blend1 = gaussianFoveatedRingBlendCpu(std::max(midDegrees - halfTransition, 0.0f),
                                                              midDegrees + halfTransition,
                                                              eccentricityDegrees);
            const float inner = levels.x + (levels.y - levels.x) * std::clamp(blend0, 0.0f, 1.0f);
            return inner + (levels.z - inner) * std::clamp(blend1, 0.0f, 1.0f);
        }

        float gaussianFoveatedGuardedClodLevelCpu(const GaussianSplatRenderSettings& settings,
                                                  const float                        eccentricityDegrees,
                                                  const uint32_t                     riskSectors)
        {
            const float baseLevel = gaussianFoveatedBaseClodLevelCpu(settings, eccentricityDegrees);
            const auto  guardMode = settings.foveatedCoverageGuardMode;
            if (guardMode == GaussianSplatFoveatedCoverageGuardMode::eOff)
                return baseLevel;

            const uint32_t sector = gaussianSplatCoverageSector(settings, eccentricityDegrees);
            if (guardMode == GaussianSplatFoveatedCoverageGuardMode::eRiskTriggered)
            {
                if ((riskSectors & (1u << sector)) == 0u)
                    return baseLevel;
            }

            const float protectedLevel =
                settings.foveatedCoverageProtectionDegrees > 1e-4f ?
                    gaussianFoveatedBaseClodLevelCpu(
                        settings,
                        std::max(eccentricityDegrees - settings.foveatedCoverageProtectionDegrees, 0.0f)) :
                    baseLevel;

            if (guardMode == GaussianSplatFoveatedCoverageGuardMode::eGlobal)
                return std::max(baseLevel, protectedLevel);

            const float localFloor = sector == 0u ? settings.foveatedCoverageGuardMinLevels.x :
                                     sector == 1u ? settings.foveatedCoverageGuardMinLevels.y :
                                                    settings.foveatedCoverageGuardMinLevels.z;
            const float requestedLevel = std::max(baseLevel, std::max(std::clamp(localFloor, 0.0f, 1.0f),
                                                                       protectedLevel));
            const float baseBudget = std::max(settings.foveatedRingLevels.x, 1e-5f);
            const float guardCapLevel =
                baseLevel + baseBudget * std::clamp(settings.foveatedCoverageGuardBudgetRatio, 0.0f, 0.25f);
            return std::min(requestedLevel, guardCapLevel);
        }

        bool gaussianRankPassesLevel(const uint32_t rank, const uint32_t rankTotalCount, const float level)
        {
            if (rankTotalCount == 0u)
                return false;
            const uint32_t budget =
                std::min(rankTotalCount,
                         std::max(1u, static_cast<uint32_t>(
                                          std::ceil(static_cast<float>(rankTotalCount) *
                                                    std::clamp(level, 0.0f, 1.0f)))));
            return rank < budget;
        }

        uint64_t effectiveGaussianProjectedCostBudget(const GaussianSplatRenderSettings& settings,
                                                      const uint64_t                     totalProjectedCost,
                                                      float&                             budgetRatio)
        {
            budgetRatio = 0.0f;
            if (totalProjectedCost == 0u)
                return 0u;

            if (settings.projectedCostBudget > 0u)
                return std::min(totalProjectedCost, settings.projectedCostBudget);

            budgetRatio = settings.projectedCostBudgetRatio > 0.0f ?
                              std::clamp(settings.projectedCostBudgetRatio, 0.0f, 1.0f) :
                              effectiveGaussianAutomaticClodLevel(settings);
            const double budget =
                std::ceil(static_cast<double>(totalProjectedCost) * static_cast<double>(budgetRatio));
            return std::min(totalProjectedCost, std::max<uint64_t>(1u, static_cast<uint64_t>(budget)));
        }

        GaussianProjectedCostPrefixStats selectGaussianSplatProjectedCostChunksFromChunkCosts(
            const GaussianSplatRenderSettings&              settings,
            const std::span<const GaussianProjectedChunkCost> chunks,
            const uint64_t                                  totalProjectedCost,
            const uint32_t                                  sourceCapacity,
            const std::chrono::steady_clock::time_point     totalStart)
        {
            GaussianProjectedCostPrefixStats stats {};
            stats.totalProjectedCost = totalProjectedCost;
            stats.costBudget = effectiveGaussianProjectedCostBudget(settings, stats.totalProjectedCost, stats.budgetRatio);
            stats.gaussiansOrChunksConsidered = static_cast<uint32_t>(chunks.size());
            if (stats.costBudget == 0u)
            {
                stats.totalSelectionCpuMs = elapsedCpuMs(totalStart);
                return stats;
            }

            const auto chunkSelectionStart = std::chrono::steady_clock::now();
            for (const auto& chunk : chunks)
            {
                stats.actualProjectedCost += chunk.cost;
                stats.numTileIntersections += chunk.tileIntersections;
                stats.sumProjectedAreaPx += chunk.projectedAreaPx;
                stats.selectedCount = std::min(sourceCapacity, chunk.endRank);
                ++stats.selectedChunks;
                ++stats.chunksScanned;
                if (stats.actualProjectedCost >= stats.costBudget)
                    break;
            }
            stats.chunkSelectionCpuMs = elapsedCpuMs(chunkSelectionStart);
            stats.costOvershoot = stats.actualProjectedCost > stats.costBudget ?
                                      stats.actualProjectedCost - stats.costBudget :
                                      0u;
            stats.totalSelectionCpuMs = elapsedCpuMs(totalStart);
            return stats;
        }

        GaussianProjectedCostGpuCameraBlock makeGaussianProjectedCostGpuCameraBlock(
            const std::span<const RenderCamera> cameras,
            const rhi::Extent2D                fallbackExtent)
        {
            const rhi::Extent2D extent = safeGaussianSplatCostExtent(fallbackExtent);
            GaussianProjectedCostGpuCameraBlock block {};
            const uint32_t cameraCount = std::min<uint32_t>(static_cast<uint32_t>(cameras.size()), 2u);
            for (uint32_t i = 0u; i < cameraCount; ++i)
            {
                const RenderCamera& camera = cameras[i];
                block.cameras[i].view       = camera.view;
                block.cameras[i].projection = camera.projection;
                block.cameras[i].resolution =
                    glm::vec4 {static_cast<float>(extent.width),
                               static_cast<float>(extent.height),
                               1.0f / static_cast<float>(std::max(extent.width, 1u)),
                               1.0f / static_cast<float>(std::max(extent.height, 1u))};
                block.cameras[i].params = glm::vec4 {camera.zNear, camera.zFar, camera.fovY, 0.0f};
            }
            return block;
        }

        uint64_t combineU64(const uint32_t lo, const uint32_t hi)
        {
            return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32u);
        }

        GaussianProjectedCostGpuBuildResult tryBuildGaussianProjectedCostChunksGpuSync(
            const GaussianSplatRenderSettings&         settings,
            resource::GpuSceneView&                    gpuSceneView,
            const uint32_t                             sourceCapacity,
            const bool                                 directPrefix,
            const std::span<const RenderCamera>        cameras,
            const rhi::Extent2D                        fallbackExtent,
            GaussianProjectedCostGpuBuildContext*      gpuContext)
        {
            GaussianProjectedCostGpuBuildResult result {};
            result.fallbackUsed = true;
            if (!gpuContext || !gpuContext->rd || !gpuContext->shaderLib)
                return result;
            auto& rd = *gpuContext->rd;
            if (rd.getBackendApi() != rhi::RenderBackendApi::eVulkan)
                return result;
            if (!directPrefix || sourceCapacity == 0u || cameras.empty())
                return result;
            if (!gpuSceneView.generalGaussianSplatDrawBuffer ||
                !gpuSceneView.generalGaussianSplatPackedSourceBuffer)
            {
                return result;
            }

            const uint32_t chunkSize = std::max(settings.projectedCostChunkSize, 1u);
            const uint32_t chunkCount = (sourceCapacity + chunkSize - 1u) / chunkSize;
            if (chunkCount == 0u)
                return result;
            gpuSceneView.ensureGeneralGaussianSplatProjectedCostBuffers(rd, chunkCount);
            if (!gpuSceneView.generalGaussianSplatProjectedCostChunkBuffer ||
                !gpuSceneView.generalGaussianSplatProjectedCostReadbackBuffer)
            {
                return result;
            }

            rhi::ShaderLibraryRuntime::KeywordValues keywords {
                {"USE_DIRECT_PREFIX", 1u},
            };
            auto shader = gpuContext->shaderLib->load(
                rhi::ShaderLibraryRuntime::computeVariantHash("gaussian_splat_projected_cost.comp",
                                                              vshadersystem::ShaderStage::eComp,
                                                              keywords),
                vshadersystem::ShaderStage::eComp);
            if (!shader)
                return result;
            auto pipeline = rd.createComputePipelineBuiltin(shader->spirv);
            if (!pipeline)
                return result;

            auto cameraBuffer = rd.createUniformBuffer(sizeof(GaussianProjectedCostGpuCameraBlock),
                                                       rhi::AllocationHints::eSequentialWrite);
            if (!cameraBuffer)
                return result;
            const GaussianProjectedCostGpuCameraBlock cameraBlock =
                makeGaussianProjectedCostGpuCameraBlock(cameras, fallbackExtent);
            GaussianProjectedCostGpuPushConstants pc {};
            pc.sourceCapacity = sourceCapacity;
            pc.chunkSize      = chunkSize;
            pc.chunkCount     = chunkCount;
            pc.cameraCount    = std::min<uint32_t>(static_cast<uint32_t>(cameras.size()), 2u);
            pc.gazeAndRings =
                glm::vec4 {settings.foveatedGaze.x,
                           settings.foveatedGaze.y,
                           std::max(settings.foveatedRingDegrees.x, 0.0f),
                           std::max(settings.foveatedRingDegrees.y, settings.foveatedRingDegrees.x)};

            const uint64_t bytes =
                static_cast<uint64_t>(chunkCount) * sizeof(GaussianProjectedCostGpuChunkRecord);
            const auto readbackStart = std::chrono::steady_clock::now();
            rd.execute(
                [&](rhi::CommandBuffer& cb) {
                    cb.update(cameraBuffer, 0u, sizeof(cameraBlock), &cameraBlock);
                    cb.clear(*gpuSceneView.generalGaussianSplatProjectedCostChunkBuffer, 0u);
                    cb.getBarrierBuilder().memoryBarrier(
                        {
                            .srcStage  = rhi::PipelineStages::eTransfer,
                            .srcAccess = rhi::Access::eTransferWrite,
                        },
                        {
                            .dstStage  = rhi::PipelineStages::eComputeShader,
                            .dstAccess = rhi::Access::eShaderRead | rhi::Access::eShaderWrite,
                        });

                    auto descriptorSetBuilder = cb.createDescriptorSetBuilder();
                    descriptorSetBuilder.bind(0u,
                                              rhi::bindings::UniformBuffer {
                                                  .buffer = &cameraBuffer,
                                                  .offset = 0u,
                                                  .range  = sizeof(GaussianProjectedCostGpuCameraBlock),
                                              });
                    descriptorSetBuilder.bind(13u,
                                              rhi::bindings::StorageBuffer {
                                                  .buffer = gpuSceneView.generalGaussianSplatDrawBuffer.get(),
                                                  .offset = 0u,
                                                  .range  = std::nullopt,
                                              });
                    descriptorSetBuilder.bind(14u,
                                              rhi::bindings::StorageBuffer {
                                                  .buffer = gpuSceneView.generalGaussianSplatPackedSourceBuffer.get(),
                                                  .offset = 0u,
                                                  .range  = std::nullopt,
                                              });
                    descriptorSetBuilder.bind(47u,
                                              rhi::bindings::StorageBuffer {
                                                  .buffer =
                                                      gpuSceneView.generalGaussianSplatProjectedCostChunkBuffer.get(),
                                                  .offset = 0u,
                                                  .range  = bytes,
                                              });
                    const auto descriptors = descriptorSetBuilder.build(pipeline.getDescriptorSetLayout(0u));
                    cb.bindPipeline(pipeline);
                    cb.bindDescriptorSet(0u, descriptors);
                    cb.pushConstants(rhi::ShaderStages::eCompute, 0u, &pc);
                    cb.dispatch({chunkCount, 1u, 1u});
                    // The synchronous readback copy consumes shader writes as transfer reads.
                    cb.insertBufferBarrier(*gpuSceneView.generalGaussianSplatProjectedCostChunkBuffer,
                                           {
                                               .srcStage  = rhi::PipelineStages::eComputeShader,
                                               .srcAccess = rhi::Access::eShaderWrite,
                                           },
                                           {
                                               .dstStage  = rhi::PipelineStages::eTransfer,
                                               .dstAccess = rhi::Access::eTransferRead,
                                           },
                                           0u,
                                           bytes);
                    cb.copyBuffer(*gpuSceneView.generalGaussianSplatProjectedCostChunkBuffer,
                                  *gpuSceneView.generalGaussianSplatProjectedCostReadbackBuffer,
                                  rhi::BufferCopy {0u, 0u, bytes});
                },
                true);
            rd.waitIdle();

            auto* mapped = static_cast<const GaussianProjectedCostGpuChunkRecord*>(
                gpuSceneView.generalGaussianSplatProjectedCostReadbackBuffer->map());
            if (!mapped)
                return result;

            result.chunks.clear();
            result.chunks.reserve(chunkCount);
            for (uint32_t chunk = 0u; chunk < chunkCount; ++chunk)
            {
                const auto& record = mapped[chunk];
                const uint64_t cost = combineU64(record.costAndTiles.x, record.costAndTiles.y);
                const uint64_t tiles = combineU64(record.costAndTiles.z, record.costAndTiles.w);
                const uint32_t beginRank = record.rangeAndArea.x;
                const uint32_t endRank   = std::min(sourceCapacity, record.rangeAndArea.y);
                const float area = std::bit_cast<float>(record.rangeAndArea.z);
                result.chunks.push_back(GaussianProjectedChunkCost {
                    .beginRank        = beginRank,
                    .endRank          = endRank,
                    .cost             = cost,
                    .tileIntersections = tiles,
                    .projectedAreaPx  = static_cast<double>(std::max(area, 0.0f)),
                });
                result.totalProjectedCost += cost;
                result.numTileIntersections += tiles;
                result.sumProjectedAreaPx += static_cast<double>(std::max(area, 0.0f));
            }
            gpuSceneView.generalGaussianSplatProjectedCostReadbackBuffer->unmap();

            result.readbackCpuMs = elapsedCpuMs(readbackStart);
            result.valid = !result.chunks.empty();
            result.fallbackUsed = !result.valid;
            return result;
        }

        bool shouldBuildGaussianProjectedCostChunksOnGpu(
            const GaussianSplatRenderSettings&               settings,
            const bool                                       directPrefix,
            const GaussianProjectedCostGpuBuildContext* const gpuContext)
        {
            if (gpuContext == nullptr)
                return false;
            if (settings.cleanTimingMode)
                return false;
            // Keep CPU mode on the persistent CPU sample/chunk cache. The GPU path
            // performs a synchronous readback and is only suitable as an explicit
            // diagnostic build mode.
            return directPrefix &&
                   settings.projectedCostBuildMode == GaussianProjectedCostBuildMode::eGpuSync;
        }

        GaussianProjectedCostPrefixStats selectGaussianSplatProjectedCostChunks(
            const GaussianSplatRenderSettings& settings,
            resource::GpuSceneView&            gpuSceneView,
            const uint32_t                     sourceCapacity,
            const bool                         directPrefix,
            const std::span<const RenderCamera> cameras,
            const rhi::Extent2D                fallbackExtent,
            const uint64_t                     sceneEpoch,
            GaussianProjectedCostFrameSampleCache& projectedSampleCache,
            RuntimeProfiler&                   profiler,
            GaussianProjectedCostGpuBuildContext* gpuContext)
        {
            RuntimeProfiler::Scope scope {profiler, "GaussianLOD::ProjectedCostChunks"};
            const auto totalStart = std::chrono::steady_clock::now();

            GaussianProjectedCostPrefixStats stats {};
            if (sourceCapacity == 0u)
                return stats;

            const bool useGpuBuild =
                shouldBuildGaussianProjectedCostChunksOnGpu(settings, directPrefix, gpuContext);
            if (useGpuBuild)
            {
                const auto gpuBuild = tryBuildGaussianProjectedCostChunksGpuSync(settings,
                                                                                 gpuSceneView,
                                                                                 sourceCapacity,
                                                                                 directPrefix,
                                                                                 cameras,
                                                                                 fallbackExtent,
                                                                                 gpuContext);
                stats.gpuCostBuildEnabled = true;
                stats.gpuCostBuildGpuMs = gpuBuild.buildGpuMs;
                stats.gpuCostReadbackCpuMs = gpuBuild.readbackCpuMs;
                stats.gpuCostChunkCount = static_cast<uint32_t>(gpuBuild.chunks.size());
                stats.gpuCostValid = gpuBuild.valid;
                stats.cpuFallbackUsed = gpuBuild.fallbackUsed;
                if (gpuBuild.valid)
                {
                    stats = selectGaussianSplatProjectedCostChunksFromChunkCosts(settings,
                                                                                 gpuBuild.chunks,
                                                                                 gpuBuild.totalProjectedCost,
                                                                                 sourceCapacity,
                                                                                 totalStart);
                    if (settings.projectedCostSemanticDiffEnabled)
                    {
                        const auto& projectedSamples = ensureGaussianProjectedCostSamples(settings,
                                                                                          gpuSceneView,
                                                                                          sourceCapacity,
                                                                                          directPrefix,
                                                                                          cameras,
                                                                                          fallbackExtent,
                                                                                          sceneEpoch,
                                                                                          projectedSampleCache);
                        const auto& cpuChunks = ensureGaussianProjectedCostChunks(projectedSamples,
                                                                                  sourceCapacity,
                                                                                  std::max(settings.projectedCostChunkSize,
                                                                                           1u),
                                                                                  projectedSampleCache);
                        const auto cpuStats =
                            selectGaussianSplatProjectedCostChunksFromChunkCosts(settings,
                                                                                 cpuChunks,
                                                                                 projectedSampleCache.chunkTotalProjectedCost,
                                                                                 sourceCapacity,
                                                                                 totalStart);
                        double maxError = 0.0;
                        double sumError = 0.0;
                        const size_t compareCount = std::min(gpuBuild.chunks.size(), cpuChunks.size());
                        for (size_t i = 0u; i < compareCount; ++i)
                        {
                            const double error =
                                std::abs(static_cast<double>(gpuBuild.chunks[i].cost) -
                                         static_cast<double>(cpuChunks[i].cost));
                            sumError += error;
                            maxError = std::max(maxError, error);
                        }
                        stats.gpuCpuCostL1Error = compareCount > 0u ?
                                                      sumError / static_cast<double>(compareCount) :
                                                      -1.0;
                        stats.gpuCpuCostMaxError = compareCount > 0u ? maxError : -1.0;
                        const double denom = static_cast<double>(std::max(sourceCapacity, 1u));
                        stats.gpuCpuSelectedRatioDelta =
                            std::abs(static_cast<double>(stats.selectedCount) -
                                     static_cast<double>(cpuStats.selectedCount)) / denom;
                        stats.gpuCpuActualCostDelta =
                            std::abs(static_cast<double>(stats.actualProjectedCost) -
                                     static_cast<double>(cpuStats.actualProjectedCost));
                        stats.gpuCpuOvershootDelta =
                            std::abs(static_cast<double>(stats.costOvershoot) -
                                     static_cast<double>(cpuStats.costOvershoot));
                    }
                    stats.gpuCostBuildEnabled = true;
                    stats.gpuCostBuildGpuMs = gpuBuild.buildGpuMs;
                    stats.gpuCostReadbackCpuMs = gpuBuild.readbackCpuMs;
                    stats.gpuCostChunkCount = static_cast<uint32_t>(gpuBuild.chunks.size());
                    stats.gpuCostValid = true;
                    stats.cpuFallbackUsed = false;
                    stats.costEstimationCpuMs = gpuBuild.readbackCpuMs;
                    stats.chunkAggregationCpuMs = 0.0;
                    stats.cacheLookupCpuMs = 0.0;
                    stats.cacheHitRate = -1.0;
                    return stats;
                }
            }

            const uint32_t chunkSize = std::max(settings.projectedCostChunkSize, 1u);

            const auto costEstimationStart = std::chrono::steady_clock::now();
            const auto& projectedSamples = ensureGaussianProjectedCostSamples(settings,
                                                                              gpuSceneView,
                                                                              sourceCapacity,
                                                                              directPrefix,
                                                                              cameras,
                                                                              fallbackExtent,
                                                                              sceneEpoch,
                                                                              projectedSampleCache);
            stats.costEstimationCpuMs = elapsedCpuMs(costEstimationStart);
            stats.gaussiansOrChunksConsidered = sourceCapacity;

            const auto& chunks = ensureGaussianProjectedCostChunks(projectedSamples,
                                                                   sourceCapacity,
                                                                   chunkSize,
                                                                   projectedSampleCache);
            stats.chunkAggregationCpuMs = projectedSampleCache.chunkAggregationCpuMs;
            stats.totalProjectedCost = projectedSampleCache.chunkTotalProjectedCost;
            const auto chunkSelectionStart = std::chrono::steady_clock::now();
            stats.costBudget = effectiveGaussianProjectedCostBudget(settings, stats.totalProjectedCost, stats.budgetRatio);
            if (stats.costBudget == 0u)
            {
                stats.chunkSelectionCpuMs = elapsedCpuMs(chunkSelectionStart);
                stats.cacheLookupCpuMs = projectedSampleCache.cacheLookupCpuMs;
                stats.cacheHitRate = projectedSampleCache.hitRate();
                stats.totalSelectionCpuMs = elapsedCpuMs(totalStart);
                return stats;
            }

            for (const auto& chunk : chunks)
            {
                stats.actualProjectedCost += chunk.cost;
                stats.numTileIntersections += chunk.tileIntersections;
                stats.sumProjectedAreaPx += chunk.projectedAreaPx;
                stats.selectedCount = chunk.endRank;
                ++stats.selectedChunks;
                ++stats.chunksScanned;
                if (stats.actualProjectedCost >= stats.costBudget)
                    break;
            }
            stats.chunkSelectionCpuMs = elapsedCpuMs(chunkSelectionStart);

            stats.costOvershoot = stats.actualProjectedCost > stats.costBudget ?
                                      stats.actualProjectedCost - stats.costBudget :
                                      0u;
            stats.cacheLookupCpuMs = projectedSampleCache.cacheLookupCpuMs;
            stats.cacheHitRate = projectedSampleCache.hitRate();
            stats.totalSelectionCpuMs = elapsedCpuMs(totalStart);
            if (useGpuBuild)
            {
                stats.gpuCostBuildEnabled = true;
                stats.gpuCostValid = false;
                stats.cpuFallbackUsed = true;
            }
            return stats;
        }

        GaussianProjectedCostPrefixStats selectGaussianSplatFoveatedScoreSources(
            const GaussianSplatRenderSettings& settings,
            resource::GpuSceneView&            gpuSceneView,
            const uint32_t                     sourceCapacity,
            const uint32_t                     totalSplatCount,
            const std::span<const RenderCamera> cameras,
            const rhi::Extent2D                fallbackExtent,
            RuntimeProfiler&                   profiler,
            std::vector<uint16_t>&             scoreResidency)
        {
            RuntimeProfiler::Scope scope {profiler, "GaussianLOD::FoveatedScoreSelect"};
            const auto totalStart = std::chrono::steady_clock::now();

            GaussianProjectedCostPrefixStats stats {};
            if (sourceCapacity == 0u || gpuSceneView.generalGaussianSplatSelectedSources.empty())
                return stats;

            const uint32_t targetCount =
                std::min(sourceCapacity, effectiveGaussianLodBudget(settings, totalSplatCount));
            if (targetCount == 0u)
                return stats;
            if (settings.foveatedTemporalHysteresisEnabled && scoreResidency.size() != sourceCapacity)
                scoreResidency.assign(sourceCapacity, 0u);

            struct ScoredChunk
            {
                double score {0.0};
                uint32_t beginRank {0u};
                uint32_t endRank {0u};
                uint32_t chunkIndex {0u};
                uint32_t binIndex {std::numeric_limits<uint32_t>::max()};
                uint64_t cost {1u};
                uint64_t tileIntersections {0u};
                double projectedAreaPx {0.0};
                bool visible {false};
            };

            const auto& orderedSources = gpuSceneView.generalGaussianSplatSelectedSources;
            const uint32_t orderedSourceCount =
                std::min<uint32_t>(sourceCapacity, static_cast<uint32_t>(orderedSources.size()));
            const rhi::Extent2D extent = safeGaussianSplatCostExtent(fallbackExtent);
            const double viewportArea =
                static_cast<double>(std::max(extent.width, 1u)) *
                static_cast<double>(std::max(extent.height, 1u));
            const double areaNormalizer = std::max(std::log1p(viewportArea), 1.0);

            const uint32_t chunkSize = std::clamp(settings.projectedCostChunkSize, 256u, 8192u);
            const uint32_t chunkCount =
                static_cast<uint32_t>((orderedSourceCount + chunkSize - 1u) / chunkSize);
            const bool coverageBinMode = settings.coverageBinScoreBudgetEnabled();
            const uint32_t coverageBinGridX = std::clamp(settings.foveatedCoverageBinGridX, 1u, 64u);
            const uint32_t coverageBinGridY = std::clamp(settings.foveatedCoverageBinGridY, 1u, 64u);
            const uint32_t coverageBinCount = coverageBinGridX * coverageBinGridY;
            std::vector<ScoredChunk> chunks;
            chunks.reserve(chunkCount);

            const auto estimationStart = std::chrono::steady_clock::now();
            uint32_t validCount = 0u;
            for (uint32_t chunkIndex = 0u; chunkIndex < chunkCount; ++chunkIndex)
            {
                const uint32_t beginRank = chunkIndex * chunkSize;
                const uint32_t endRank =
                    std::min<uint32_t>(beginRank + chunkSize, orderedSourceCount);
                if (beginRank >= endRank)
                    continue;

                uint32_t representativeRank = beginRank + (endRank - beginRank) / 2u;
                while (representativeRank < endRank)
                {
                    const auto& candidate = orderedSources[representativeRank];
                    if ((candidate.flags & 0x80000000u) == 0u &&
                        candidate.sourceIndex < gpuSceneView.generalGaussianSplatPackedSources.size() &&
                        candidate.drawIndex < gpuSceneView.generalGaussianSplatDraws.size())
                    {
                        break;
                    }
                    ++representativeRank;
                }
                if (representativeRank >= endRank)
                {
                    continue;
                }
                const auto& selection = orderedSources[representativeRank];

                const GaussianProjectedCostSample sample = estimateGaussianSplatProjectedTileCost(
                    gpuSceneView.generalGaussianSplatPackedSources[selection.sourceIndex],
                    gpuSceneView.generalGaussianSplatDraws[selection.drawIndex],
                    settings,
                    cameras,
                    fallbackExtent);

                float minEccentricity = std::numeric_limits<float>::infinity();
                for (const float regionEccentricity : sample.regionMinEccentricityDegrees)
                {
                    if (std::isfinite(regionEccentricity))
                        minEccentricity = std::min(minEccentricity, regionEccentricity);
                }
                if (!std::isfinite(minEccentricity))
                    minEccentricity = 180.0f;

                const float desiredLevel =
                    gaussianFoveatedGuardedClodLevelCpu(settings, minEccentricity, 0x7u);
                const double rankImportance =
                    orderedSourceCount > 1u ?
                        1.0 - static_cast<double>(representativeRank) /
                                  static_cast<double>(orderedSourceCount - 1u) :
                        1.0;
                const double projectedTerm =
                    std::clamp(std::log1p(std::max(sample.projectedAreaPx, 0.0)) / areaNormalizer,
                               0.0,
                               1.0);
                const double visibilityTerm = sample.projectedAreaPx > 0.0 ? 1.0 : 0.0;
                uint32_t binIndex = std::numeric_limits<uint32_t>::max();
                if (coverageBinMode && sample.projectedAreaPx > 0.0)
                {
                    for (uint32_t cameraIndex = 0u;
                         cameraIndex < GaussianProjectedCostSample::kMaxCameraFootprints;
                         ++cameraIndex)
                    {
                        if (sample.cameraVisible[cameraIndex] == 0u)
                            continue;
                        const glm::vec2 ndc = sample.cameraCenterNdc[cameraIndex];
                        const float u = std::clamp(ndc.x * 0.5f + 0.5f, 0.0f, 0.999999f);
                        const float v = std::clamp(1.0f - (ndc.y * 0.5f + 0.5f), 0.0f, 0.999999f);
                        const uint32_t binX =
                            std::min<uint32_t>(coverageBinGridX - 1u,
                                               static_cast<uint32_t>(u * static_cast<float>(coverageBinGridX)));
                        const uint32_t binY =
                            std::min<uint32_t>(coverageBinGridY - 1u,
                                               static_cast<uint32_t>(v * static_cast<float>(coverageBinGridY)));
                        binIndex = binY * coverageBinGridX + binX;
                        break;
                    }
                }
                const double foveaTerm =
                    minEccentricity <= std::max(settings.foveatedRingDegrees.x, 0.0f) ? 1.0 : 0.0;
                const float peripheralFactor = [&settings, minEccentricity]() {
                    const float foveaDegrees = std::max(settings.foveatedRingDegrees.x, 0.0f);
                    const float midDegrees = std::max(settings.foveatedRingDegrees.y, foveaDegrees + 1.0e-3f);
                    const float t = std::clamp((minEccentricity - foveaDegrees) /
                                                   std::max(midDegrees - foveaDegrees, 1.0e-3f),
                                               0.0f,
                                               1.0f);
                    return t * t * (3.0f - 2.0f * t);
                }();
                const double temporalBonus =
                    settings.foveatedTemporalHysteresisEnabled && representativeRank < scoreResidency.size() &&
                            scoreResidency[representativeRank] > 0u ?
                        static_cast<double>(settings.foveatedTemporalHysteresisRatio) *
                            (1.0 + static_cast<double>(settings.foveatedTemporalPeripheralScale) *
                                       static_cast<double>(peripheralFactor)) :
                        0.0;
                const double score =
                    6.0 * static_cast<double>(std::clamp(desiredLevel, 0.0f, 1.0f)) +
                    1.25 * rankImportance +
                    0.75 * projectedTerm +
                    0.50 * foveaTerm +
                    0.25 * visibilityTerm +
                    temporalBonus;

                const uint32_t chunkPopulation = endRank - beginRank;
                const uint64_t chunkCost =
                    std::max<uint64_t>(1u, sample.cost) * static_cast<uint64_t>(chunkPopulation);
                stats.totalProjectedCost += chunkCost;
                ++validCount;
                chunks.push_back(ScoredChunk {
                    .score             = score,
                    .beginRank         = beginRank,
                    .endRank           = endRank,
                    .chunkIndex        = static_cast<uint32_t>(chunks.size()),
                    .binIndex          = binIndex,
                    .cost              = chunkCost,
                    .tileIntersections = sample.tileIntersections * static_cast<uint64_t>(chunkPopulation),
                    .projectedAreaPx   = sample.projectedAreaPx * static_cast<double>(chunkPopulation),
                    .visible           = sample.projectedAreaPx > 0.0,
                });
            }
            stats.costEstimationCpuMs = elapsedCpuMs(estimationStart);
            stats.gaussiansOrChunksConsidered = validCount;

            const auto selectionStart = std::chrono::steady_clock::now();
            std::vector<ScoredChunk> byScore = chunks;
            std::stable_sort(byScore.begin(), byScore.end(), [](const auto& lhs, const auto& rhs) {
                if (lhs.score != rhs.score)
                    return lhs.score > rhs.score;
                return lhs.beginRank < rhs.beginRank;
            });

            const uint32_t selectedCount =
                std::min<uint32_t>(targetCount, orderedSourceCount);
            const float prefixRatio =
                coverageBinMode ?
                    std::clamp(settings.foveatedCoverageBinPrefixRatio, 0.0f, 0.90f) :
                    0.50f;
            const uint32_t prefixFloorCount =
                std::min<uint32_t>(selectedCount,
                                   static_cast<uint32_t>(std::ceil(static_cast<float>(selectedCount) *
                                                                   prefixRatio)));
            uint32_t selectedSoFar = 0u;
            uint32_t selectedChunkCount = 0u;
            std::vector<uint8_t> selectedChunkMask(chunks.size(), 0u);
            std::vector<std::pair<uint32_t, uint32_t>> selectedRanges;
            selectedRanges.reserve(byScore.size() + 1u);
            auto selectRange = [&](const uint32_t beginRank, const uint32_t endRank) {
                if (selectedSoFar >= selectedCount || beginRank >= endRank)
                    return;
                const uint32_t clampedEnd =
                    std::min<uint32_t>(endRank, beginRank + (selectedCount - selectedSoFar));
                if (beginRank >= clampedEnd)
                    return;
                selectedRanges.emplace_back(beginRank, clampedEnd);
                selectedSoFar += clampedEnd - beginRank;
            };
            auto selectChunkTail = [&](const uint32_t chunkVectorIndex) {
                if (chunkVectorIndex >= chunks.size() || selectedSoFar >= selectedCount ||
                    selectedChunkMask[chunkVectorIndex] != 0u)
                {
                    return;
                }
                selectedChunkMask[chunkVectorIndex] = 1u;
                ++selectedChunkCount;
                const auto& chunk = chunks[chunkVectorIndex];
                selectRange(std::max(chunk.beginRank, prefixFloorCount), chunk.endRank);
            };
            selectRange(0u, prefixFloorCount);
            if (coverageBinMode && !chunks.empty() && selectedSoFar < selectedCount)
            {
                std::vector<std::vector<uint32_t>> binChunks(coverageBinCount);
                for (uint32_t chunkVectorIndex = 0u;
                     chunkVectorIndex < static_cast<uint32_t>(chunks.size());
                     ++chunkVectorIndex)
                {
                    const auto& chunk = chunks[chunkVectorIndex];
                    if (!chunk.visible || chunk.binIndex >= coverageBinCount)
                        continue;
                    binChunks[chunk.binIndex].push_back(chunkVectorIndex);
                }

                std::vector<uint32_t> activeBins;
                activeBins.reserve(coverageBinCount);
                for (uint32_t bin = 0u; bin < coverageBinCount; ++bin)
                {
                    if (binChunks[bin].empty())
                        continue;
                    auto& binList = binChunks[bin];
                    std::stable_sort(binList.begin(), binList.end(), [&chunks](const uint32_t lhs, const uint32_t rhs) {
                        if (chunks[lhs].score != chunks[rhs].score)
                            return chunks[lhs].score > chunks[rhs].score;
                        return chunks[lhs].beginRank < chunks[rhs].beginRank;
                    });
                    activeBins.push_back(bin);
                }

                std::stable_sort(activeBins.begin(), activeBins.end(), [coverageBinGridX, coverageBinGridY, &settings](
                                                                         const uint32_t lhs,
                                                                         const uint32_t rhs) {
                    auto distance2 = [coverageBinGridX, coverageBinGridY, &settings](const uint32_t bin) {
                        const uint32_t x = bin % coverageBinGridX;
                        const uint32_t y = bin / coverageBinGridX;
                        const float u = (static_cast<float>(x) + 0.5f) /
                                        static_cast<float>(std::max(coverageBinGridX, 1u));
                        const float v = (static_cast<float>(y) + 0.5f) /
                                        static_cast<float>(std::max(coverageBinGridY, 1u));
                        const glm::vec2 delta = glm::vec2 {u, v} - settings.foveatedGaze;
                        return glm::dot(delta, delta);
                    };
                    const float lhsDistance = distance2(lhs);
                    const float rhsDistance = distance2(rhs);
                    if (lhsDistance != rhsDistance)
                        return lhsDistance < rhsDistance;
                    return lhs < rhs;
                });

                if (!activeBins.empty())
                {
                    const uint32_t quotaBudget =
                        std::min<uint32_t>(selectedCount - selectedSoFar,
                                           static_cast<uint32_t>(
                                               std::ceil(static_cast<float>(selectedCount) *
                                                         std::clamp(settings.foveatedCoverageBinQuotaRatio,
                                                                    0.0f,
                                                                    0.80f))));
                    const uint32_t quotaPerBin =
                        std::max<uint32_t>(1u,
                                           static_cast<uint32_t>(
                                               std::ceil(static_cast<float>(quotaBudget) /
                                                         static_cast<float>(activeBins.size()))));
                    std::vector<uint32_t> binCursors(coverageBinCount, 0u);
                    uint32_t quotaSelected = 0u;
                    for (uint32_t pass = 0u;
                         pass < quotaPerBin && quotaSelected < quotaBudget && selectedSoFar < selectedCount;
                         ++pass)
                    {
                        for (const uint32_t bin : activeBins)
                        {
                            if (quotaSelected >= quotaBudget || selectedSoFar >= selectedCount)
                                break;
                            auto& cursor = binCursors[bin];
                            auto& list = binChunks[bin];
                            while (cursor < list.size() && selectedChunkMask[list[cursor]] != 0u)
                                ++cursor;
                            if (cursor >= list.size())
                                continue;
                            const uint32_t before = selectedSoFar;
                            selectChunkTail(list[cursor]);
                            ++cursor;
                            if (selectedSoFar > before)
                                quotaSelected += selectedSoFar - before;
                        }
                    }
                }
            }
            for (const auto& chunk : byScore)
            {
                if (selectedSoFar >= selectedCount)
                    break;
                if (coverageBinMode)
                    selectChunkTail(chunk.chunkIndex);
                else
                    selectRange(std::max(chunk.beginRank, prefixFloorCount), chunk.endRank);
            }
            std::stable_sort(selectedRanges.begin(), selectedRanges.end(), [](const auto& lhs, const auto& rhs) {
                if (lhs.first != rhs.first)
                    return lhs.first < rhs.first;
                return lhs.second < rhs.second;
            });
            std::vector<resource::GpuGeneralGaussianSplatSelectedSource> finalSelected;
            finalSelected.reserve(selectedSoFar);
            for (const auto& [beginRank, endRank] : selectedRanges)
            {
                for (uint32_t rank = beginRank; rank < endRank; ++rank)
                {
                    auto selection = orderedSources[rank];
                    selection.packedWeight = std::bit_cast<uint32_t>(1.0f);
                    finalSelected.push_back(selection);
                }
            }
            gpuSceneView.generalGaussianSplatSelectedSources.clear();
            gpuSceneView.generalGaussianSplatSelectedSources.reserve(finalSelected.size());
            for (const auto& selection : finalSelected)
                gpuSceneView.pushGeneralGaussianSplatSelectedSource(selection);

            for (const auto& chunk : chunks)
            {
                uint32_t selectedInChunk = 0u;
                for (const auto& [beginRank, endRank] : selectedRanges)
                {
                    const uint32_t overlapBegin = std::max(beginRank, chunk.beginRank);
                    const uint32_t overlapEnd = std::min(endRank, chunk.endRank);
                    if (overlapBegin < overlapEnd)
                        selectedInChunk += overlapEnd - overlapBegin;
                }
                if (selectedInChunk == 0u)
                    continue;
                const double ratio = static_cast<double>(selectedInChunk) /
                                     static_cast<double>(std::max(chunk.endRank - chunk.beginRank, 1u));
                stats.actualProjectedCost += static_cast<uint64_t>(static_cast<double>(chunk.cost) * ratio);
                stats.numTileIntersections += static_cast<uint64_t>(static_cast<double>(chunk.tileIntersections) * ratio);
                stats.sumProjectedAreaPx += chunk.projectedAreaPx * ratio;
            }
            if (settings.foveatedTemporalHysteresisEnabled)
            {
                for (auto& residency : scoreResidency)
                {
                    if (residency > 0u)
                        --residency;
                }
                const uint16_t selectedResidency =
                    static_cast<uint16_t>(std::min<uint32_t>(
                        std::max(settings.foveatedTemporalResidencyFrames, 1u),
                        std::numeric_limits<uint16_t>::max()));
                for (const auto& [beginRank, endRank] : selectedRanges)
                {
                    for (uint32_t rank = beginRank; rank < endRank && rank < scoreResidency.size(); ++rank)
                        scoreResidency[rank] = selectedResidency;
                }
            }
            stats.selectedCount = selectedSoFar;
            stats.budgetRatio =
                static_cast<float>(static_cast<double>(selectedSoFar) /
                                   static_cast<double>(std::max(sourceCapacity, 1u)));
            stats.costBudget = stats.actualProjectedCost;
            stats.selectedChunks =
                coverageBinMode ? selectedChunkCount : static_cast<uint32_t>(byScore.size());
            stats.chunkSelectionCpuMs = elapsedCpuMs(selectionStart);
            stats.totalSelectionCpuMs = elapsedCpuMs(totalStart);
            return stats;
        }

        GaussianCoverageGuardRiskStats analyzeGaussianSplatCoverageGuardRisk(
            const GaussianSplatRenderSettings& settings,
            const resource::GpuSceneView&      gpuSceneView,
            const uint32_t                     sourceCapacity,
            const uint32_t                     rankTotalCount,
            const uint32_t                     mainActiveCount,
            const uint64_t                     mainProjectedCost,
            const bool                         directPrefix,
            const std::span<const RenderCamera> cameras,
            const rhi::Extent2D                fallbackExtent,
            const uint64_t                     sceneEpoch,
            GaussianProjectedCostFrameSampleCache& projectedSampleCache,
            RuntimeProfiler&                   profiler)
        {
            RuntimeProfiler::Scope scope {profiler, "GaussianLOD::CoverageGuardRisk"};
            const auto analysisStart = std::chrono::steady_clock::now();
            const uint64_t initialCacheFullMissCount = projectedSampleCache.cacheFullMissCount;
            const uint64_t initialCacheHitCount      = projectedSampleCache.cacheHitCount;
            const uint64_t initialCacheMissCount     = projectedSampleCache.cacheMissCount;
            const uint64_t initialSamplesBuilt       = projectedSampleCache.samplesBuilt;
            const double   initialRegionRebuildCpuMs = projectedSampleCache.regionRebuildCpuMs;

            GaussianCoverageGuardRiskStats stats {};
            stats.activePrefixCount = mainActiveCount;
            stats.maxAddsEffective = settings.foveatedCoverageGuardMaxAdds;
            stats.modeEnabled =
                settings.foveatedCoverageGuardMode != GaussianSplatFoveatedCoverageGuardMode::eOff;
            auto finish = [&]() {
                stats.cacheFullMissCount =
                    projectedSampleCache.cacheFullMissCount >= initialCacheFullMissCount ?
                        projectedSampleCache.cacheFullMissCount - initialCacheFullMissCount :
                        0u;
                stats.cacheHitCount =
                    projectedSampleCache.cacheHitCount >= initialCacheHitCount ?
                        projectedSampleCache.cacheHitCount - initialCacheHitCount :
                        0u;
                stats.cacheMissCount =
                    projectedSampleCache.cacheMissCount >= initialCacheMissCount ?
                        projectedSampleCache.cacheMissCount - initialCacheMissCount :
                        0u;
                stats.regionRebuildCpuMs =
                    projectedSampleCache.regionRebuildCpuMs >= initialRegionRebuildCpuMs ?
                        projectedSampleCache.regionRebuildCpuMs - initialRegionRebuildCpuMs :
                        0.0;
                stats.geometryCacheFullMissCount = stats.cacheFullMissCount;
                stats.geometryCacheHitCount      = stats.cacheHitCount;
                stats.geometryCacheMissCount     = stats.cacheMissCount;
                stats.geometrySamplesBuilt =
                    projectedSampleCache.samplesBuilt >= initialSamplesBuilt ?
                        projectedSampleCache.samplesBuilt - initialSamplesBuilt :
                        0u;
                stats.geometrySamplesReused = stats.geometryCacheHitCount > 0u ? sourceCapacity : 0u;
                stats.analysisCpuMs = elapsedCpuMs(analysisStart);
                return stats;
            };
            if (!settings.foveatedClodActive() ||
                settings.foveatedCoverageGuardMode != GaussianSplatFoveatedCoverageGuardMode::eRiskTriggered ||
                sourceCapacity == 0u || rankTotalCount == 0u)
            {
                return finish();
            }

            constexpr uint32_t kCoverageGuardAreaScanHardCap = 16384u;
            constexpr uint32_t kCoverageGuardCandidatePrefixProbeLimit = 16384u;
            constexpr uint32_t kCoverageGuardCandidateExtraScanHardCap = 32768u;

            std::array<double, 3> fullArea {0.0, 0.0, 0.0};
            std::array<double, 3> beforeArea {0.0, 0.0, 0.0};
            uint64_t beforeSelectedCost = 0u;

            const auto sampleBuildStart = std::chrono::steady_clock::now();
            const auto& projectedSamples = ensureGaussianProjectedCostSamples(settings,
                                                                              gpuSceneView,
                                                                              sourceCapacity,
                                                                              directPrefix,
                                                                              cameras,
                                                                              fallbackExtent,
                                                                              sceneEpoch,
                                                                              projectedSampleCache);
            stats.projectedSampleBuildCpuMs = elapsedCpuMs(sampleBuildStart);
            if (projectedSampleCache.entry == nullptr || !projectedSampleCache.entry->regionsValid)
                rebuildGaussianProjectedCostSampleRegions(settings, cameras, projectedSampleCache);
            noteGaussianProjectedCostSampleReuse(projectedSampleCache, sourceCapacity);
            const bool fullAreaCached =
                projectedSampleCache.entry != nullptr && projectedSampleCache.entry->regionsValid;
            if (fullAreaCached)
                fullArea = projectedSampleCache.entry->fullRegionAreaPx;

            const auto coverageAreaStart = std::chrono::steady_clock::now();
            const uint32_t coverageScanCount = std::min({sourceCapacity,
                                                         mainActiveCount,
                                                         kCoverageGuardAreaScanHardCap});
            for (uint32_t rank = 0u; rank < coverageScanCount; ++rank)
            {
                const GaussianProjectedCostSample& sample = projectedSamples[rank];

                bool selectedBeforeAnyRegion = false;
                for (uint32_t region = 0u; region < 3u; ++region)
                {
                    const double regionArea = sample.regionProjectedAreaPx[region];
                    if (regionArea <= 0.0)
                        continue;
                    if (!fullAreaCached)
                        fullArea[region] += regionArea;
                    const float eccentricity = sample.regionMinEccentricityDegrees[region];
                    if (!std::isfinite(eccentricity))
                        continue;
                    const bool selectedBefore =
                        rank < mainActiveCount &&
                        gaussianRankPassesLevel(rank,
                                                rankTotalCount,
                                                gaussianFoveatedBaseClodLevelCpu(settings, eccentricity));
                    if (selectedBefore)
                    {
                        beforeArea[region] += regionArea;
                        selectedBeforeAnyRegion = true;
                    }
                }
                if (selectedBeforeAnyRegion)
                    beforeSelectedCost += std::max<uint64_t>(1u, sample.cost);
            }
            if (coverageScanCount > 0u && coverageScanCount < mainActiveCount)
            {
                const double prefixScale =
                    static_cast<double>(mainActiveCount) / static_cast<double>(coverageScanCount);
                for (uint32_t region = 0u; region < 3u; ++region)
                    beforeArea[region] = std::min(fullArea[region], beforeArea[region] * prefixScale);
                const double scaledCost = static_cast<double>(beforeSelectedCost) * prefixScale;
                beforeSelectedCost =
                    scaledCost >= static_cast<double>(std::numeric_limits<uint64_t>::max()) ?
                        std::numeric_limits<uint64_t>::max() :
                        static_cast<uint64_t>(std::max(0.0, scaledCost));
            }
            stats.coverageAreaCpuMs = elapsedCpuMs(coverageAreaStart);

            const glm::vec3 thresholds =
                glm::clamp(settings.foveatedCoverageGuardMinLevels, glm::vec3 {0.0f}, glm::vec3 {1.0f});
            const std::array<float, 3> thresholdByRegion {thresholds.x, thresholds.y, thresholds.z};
            auto regionRatio = [](const double selectedArea, const double totalArea) {
                return totalArea > 1e-6 ? static_cast<float>(selectedArea / totalArea) : 1.0f;
            };
            auto scoreFor = [&](const std::array<double, 3>& selectedArea) {
                double weightedScore = 0.0;
                double totalWeight   = 0.0;
                for (uint32_t region = 0u; region < 3u; ++region)
                {
                    weightedScore += static_cast<double>(regionRatio(selectedArea[region], fullArea[region])) *
                                     fullArea[region];
                    totalWeight += fullArea[region];
                }
                return totalWeight > 1e-6 ? static_cast<float>(weightedScore / totalWeight) : 1.0f;
            };
            auto failureFor = [&](const std::array<double, 3>& selectedArea) {
                for (uint32_t region = 0u; region < 3u; ++region)
                {
                    if (fullArea[region] <= 1e-6)
                        continue;
                    if (regionRatio(selectedArea[region], fullArea[region]) < thresholdByRegion[region])
                        return 1.0f;
                }
                return 0.0f;
            };
            auto failureForRegion = [&](const uint32_t region, const std::array<double, 3>& selectedArea) {
                if (region >= 3u || fullArea[region] <= 1e-6)
                    return -1.0f;
                return regionRatio(selectedArea[region], fullArea[region]) < thresholdByRegion[region] ? 1.0f : 0.0f;
            };
            auto deficitForRegion = [&](const uint32_t region, const std::array<double, 3>& selectedArea) {
                if (region >= 3u || fullArea[region] <= 1e-6)
                    return 0.0;
                return std::max(0.0,
                                static_cast<double>(thresholdByRegion[region]) * fullArea[region] -
                                    selectedArea[region]);
            };
            auto deficitFor = [&](const std::array<double, 3>& selectedArea, const uint32_t sectorMask) {
                double deficit = 0.0;
                for (uint32_t region = 0u; region < 3u; ++region)
                {
                    if ((sectorMask & (1u << region)) == 0u)
                        continue;
                    deficit += deficitForRegion(region, selectedArea);
                }
                return deficit;
            };

            stats.coverageScoreBefore   = scoreFor(beforeArea);
            stats.coverageFailureBefore = failureFor(beforeArea);
            for (uint32_t region = 0u; region < 3u; ++region)
                stats.coverageFailureBeforeByRegion[region] = failureForRegion(region, beforeArea);

            for (uint32_t region = 0u; region < 3u; ++region)
            {
                if (fullArea[region] <= 1e-6)
                    continue;
                if (regionRatio(beforeArea[region], fullArea[region]) < thresholdByRegion[region])
                    stats.riskSectors |= (1u << region);
            }
            stats.diagnosticRiskSectors = stats.riskSectors;
            stats.riskDetected = stats.riskSectors != 0u;
            stats.deficitAreaBefore = deficitFor(beforeArea, stats.riskSectors);
            stats.requiredExtraAreaPx = stats.deficitAreaBefore;
            stats.deficitAreaAfter = stats.deficitAreaBefore;
            if (stats.riskSectors != 0u)
            {
                double bestDeficit = -1.0;
                for (uint32_t region = 0u; region < 3u; ++region)
                {
                    if ((stats.riskSectors & (1u << region)) == 0u)
                        continue;
                    const double deficit = deficitForRegion(region, beforeArea);
                    if (deficit > bestDeficit)
                    {
                        bestDeficit = deficit;
                        stats.failingSectorId = region;
                    }
                }
            }
            else
            {
                stats.requiredExtraCountEstimate = 0u;
                stats.deficitAreaAfter = 0.0;
            }

            if (stats.riskSectors == 0u)
            {
                stats.coverageScoreAfter   = stats.coverageScoreBefore;
                stats.coverageFailureAfter = stats.coverageFailureBefore;
                stats.coverageFailureAfterByRegion = stats.coverageFailureBeforeByRegion;
                return finish();
            }

            const uint64_t mainCost = std::max<uint64_t>(
                1u,
                mainProjectedCost > 0u ? mainProjectedCost : beforeSelectedCost);
            const uint64_t guardCostLimit =
                static_cast<uint64_t>(std::ceil(static_cast<double>(mainCost) *
                                                static_cast<double>(
                                                    std::clamp(settings.foveatedCoverageGuardBudgetRatio,
                                                               0.0f,
                                                               0.25f))));
            stats.budgetCap = guardCostLimit;
            if (guardCostLimit == 0u)
            {
                stats.riskSectors = 0u;
                stats.coverageScoreAfter   = stats.coverageScoreBefore;
                stats.coverageFailureAfter = stats.coverageFailureBefore;
                stats.coverageFailureAfterByRegion = stats.coverageFailureBeforeByRegion;
                return finish();
            }

            const auto candidateScanStart = std::chrono::steady_clock::now();
            std::vector<uint32_t> includedGuardRanks;
            includedGuardRanks.reserve(settings.foveatedCoverageGuardMaxAdds > 0u ?
                                           std::min(settings.foveatedCoverageGuardMaxAdds, sourceCapacity) :
                                           std::min<uint32_t>(sourceCapacity, 4096u));
            bool   repairClosed = false;
            double addedAreaToFailingSectorPx = 0.0;
            const uint32_t guardMaxAddsEffective = settings.foveatedCoverageGuardMaxAdds;
            stats.maxAddsEffective = guardMaxAddsEffective;
            constexpr bool exactTrueCandidateDiagnostics = false;
            auto finalizeRepairDiagnostics = [&]() {
                stats.candidateCountTruncated =
                    exactTrueCandidateDiagnostics ? stats.trueCandidateCount > stats.candidateCount :
                                                    stats.cappedByMaxAdds;
                stats.repairSectorMatchRate =
                    stats.addedCount > 0u ?
                        static_cast<double>(stats.addedCountToFailingSector) /
                            static_cast<double>(stats.addedCount) :
                        -1.0;
                if (stats.deficitAreaBefore <= 1e-6)
                {
                    stats.requiredExtraCountEstimate = 0u;
                }
                else if (addedAreaToFailingSectorPx > 1e-6 && stats.addedCountToFailingSector > 0u)
                {
                    const double averageAddedArea =
                        addedAreaToFailingSectorPx / static_cast<double>(stats.addedCountToFailingSector);
                    const double estimate = std::ceil(stats.deficitAreaBefore / std::max(averageAddedArea, 1e-6));
                    stats.requiredExtraCountEstimate =
                        estimate >= static_cast<double>(UINT32_MAX) ?
                            UINT32_MAX :
                            static_cast<uint32_t>(std::max(0.0, estimate));
                }
            };
            const float guardBudgetRatio =
                std::clamp(settings.foveatedCoverageGuardBudgetRatio, 0.0f, 0.25f);
            const float maxGuardedLevel =
                std::clamp(std::max(settings.foveatedRingLevels.x, 0.0f) * (1.0f + guardBudgetRatio),
                           0.0f,
                           1.0f);
            const uint32_t maxGuardedRank =
                std::min(rankTotalCount,
                         std::max(1u,
                                  static_cast<uint32_t>(std::ceil(static_cast<float>(rankTotalCount) *
                                                                  maxGuardedLevel))));
            const uint64_t requestedExtraScan =
                guardMaxAddsEffective > 0u ?
                    std::max<uint64_t>(kCoverageGuardAreaScanHardCap,
                                       static_cast<uint64_t>(guardMaxAddsEffective) * 8u) :
                    kCoverageGuardAreaScanHardCap;
            const uint32_t extraScanLimit =
                static_cast<uint32_t>(std::min<uint64_t>(kCoverageGuardCandidateExtraScanHardCap,
                                                         requestedExtraScan));
            const uint32_t prefixProbeEnd =
                std::min({sourceCapacity, mainActiveCount, kCoverageGuardCandidatePrefixProbeLimit});
            const uint32_t extensionBegin = std::min(sourceCapacity, mainActiveCount);
            const uint32_t extensionEnd =
                std::min({sourceCapacity,
                          maxGuardedRank,
                          static_cast<uint32_t>(
                              std::min<uint64_t>(sourceCapacity,
                                                 static_cast<uint64_t>(extensionBegin) + extraScanLimit))});
            auto scanCandidate = [&](const uint32_t index) {
                const GaussianProjectedCostSample& sample = projectedSamples[index];
                bool candidate = false;
                std::array<bool, 3> candidateByRegion {false, false, false};
                for (uint32_t region = 0u; region < 3u; ++region)
                {
                    if ((stats.riskSectors & (1u << region)) == 0u ||
                        sample.regionProjectedAreaPx[region] <= 0.0)
                    {
                        continue;
                    }
                    const float eccentricity = sample.regionMinEccentricityDegrees[region];
                    if (!std::isfinite(eccentricity))
                        continue;
                    const bool selectedBefore =
                        index < mainActiveCount &&
                        gaussianRankPassesLevel(index,
                                                rankTotalCount,
                                                gaussianFoveatedBaseClodLevelCpu(settings, eccentricity));
                    const bool selectedAfter =
                        gaussianRankPassesLevel(index,
                                                rankTotalCount,
                                                gaussianFoveatedGuardedClodLevelCpu(settings,
                                                                                    eccentricity,
                                                                                    stats.riskSectors));
                    if (selectedAfter && !selectedBefore)
                    {
                        candidate = true;
                        candidateByRegion[region] = true;
                    }
                }
                if (!candidate)
                    return;
                if (exactTrueCandidateDiagnostics || !repairClosed)
                    ++stats.trueCandidateCount;
                if (!repairClosed)
                {
                    ++stats.candidateCount;
                    for (uint32_t region = 0u; region < 3u; ++region)
                    {
                        if (candidateByRegion[region])
                            ++stats.candidateCountByRegion[region];
                    }
                }

                if (repairClosed)
                {
                    return;
                }

                if (guardMaxAddsEffective > 0u && stats.addedCount >= guardMaxAddsEffective)
                {
                    stats.cappedByMaxAdds = true;
                    repairClosed = true;
                    if (!exactTrueCandidateDiagnostics)
                    {
                        stats.candidateCountTruncated = true;
                    }
                    return;
                }

                const uint64_t sampleCost = std::max<uint64_t>(1u, sample.cost);
                if (stats.addedCost + sampleCost > guardCostLimit)
                {
                    stats.cappedByCost = true;
                    repairClosed = true;
                    return;
                }

                includedGuardRanks.push_back(index);
                stats.addedCost += sampleCost;
                ++stats.addedCount;
                double sampleFailingSectorArea = 0.0;
                for (uint32_t region = 0u; region < 3u; ++region)
                {
                    if (!candidateByRegion[region])
                        continue;
                    ++stats.addedCountByRegion[region];
                    stats.addedCostByRegion[region] += sampleCost;
                    if ((stats.diagnosticRiskSectors & (1u << region)) != 0u)
                        sampleFailingSectorArea += sample.regionProjectedAreaPx[region];
                }
                if (sampleFailingSectorArea > 0.0)
                {
                    ++stats.addedCountToFailingSector;
                    stats.addedCostToFailingSector += sampleCost;
                    addedAreaToFailingSectorPx += sampleFailingSectorArea;
                }
                stats.activePrefixCount = std::max(stats.activePrefixCount, index + 1u);
            };
            for (uint32_t index = 0u; index < prefixProbeEnd; ++index)
            {
                scanCandidate(index);
                if (repairClosed && !exactTrueCandidateDiagnostics)
                    break;
            }
            if (!repairClosed || exactTrueCandidateDiagnostics)
            {
                for (uint32_t index = extensionBegin; index < extensionEnd; ++index)
                {
                    scanCandidate(index);
                    if (repairClosed && !exactTrueCandidateDiagnostics)
                        break;
                }
            }
            if (guardMaxAddsEffective > 0u && stats.addedCount >= guardMaxAddsEffective)
            {
                stats.cappedByMaxAdds = true;
            }

            if (stats.addedCount == 0u)
            {
                stats.candidateScanCpuMs = elapsedCpuMs(candidateScanStart);
                stats.noCandidate = stats.candidateCount == 0u;
                stats.riskSectors = 0u;
                stats.coverageScoreAfter   = stats.coverageScoreBefore;
                stats.coverageFailureAfter = stats.coverageFailureBefore;
                stats.coverageFailureAfterByRegion = stats.coverageFailureBeforeByRegion;
                stats.deficitAreaAfter = stats.deficitAreaBefore;
                finalizeRepairDiagnostics();
                return finish();
            }

            std::array<double, 3> afterArea = beforeArea;
            for (const uint32_t index : includedGuardRanks)
            {
                const GaussianProjectedCostSample& sample = projectedSamples[index];
                for (uint32_t region = 0u; region < 3u; ++region)
                {
                    if ((stats.riskSectors & (1u << region)) == 0u ||
                        sample.regionProjectedAreaPx[region] <= 0.0)
                    {
                        continue;
                    }
                    const float eccentricity = sample.regionMinEccentricityDegrees[region];
                    if (!std::isfinite(eccentricity))
                        continue;
                    const bool selectedBefore =
                        index < mainActiveCount &&
                        gaussianRankPassesLevel(index,
                                                rankTotalCount,
                                                gaussianFoveatedBaseClodLevelCpu(settings, eccentricity));
                    const bool selectedAfter =
                        index < stats.activePrefixCount &&
                        gaussianRankPassesLevel(index,
                                                rankTotalCount,
                                                gaussianFoveatedGuardedClodLevelCpu(settings,
                                                                                    eccentricity,
                                                                                    stats.riskSectors));
                    if (selectedAfter && !selectedBefore)
                        afterArea[region] += sample.regionProjectedAreaPx[region];
                }
            }

            stats.active               = stats.addedCount > 0u;
            stats.coverageScoreAfter   = scoreFor(afterArea);
            stats.coverageFailureAfter = failureFor(afterArea);
            for (uint32_t region = 0u; region < 3u; ++region)
                stats.coverageFailureAfterByRegion[region] = failureForRegion(region, afterArea);
            stats.deficitAreaAfter = deficitFor(afterArea, stats.diagnosticRiskSectors);
            finalizeRepairDiagnostics();
            stats.candidateScanCpuMs = elapsedCpuMs(candidateScanStart);
            return finish();
        }

        uint32_t selectGaussianSplatPrefixBudget(GaussianSplatRenderSettings&       settings,
                                                 resource::GpuSceneView&            gpuSceneView,
                                                 const uint32_t                     sourceCapacity,
                                                 const uint32_t                     totalSplatCount,
                                                 const bool                         directPrefix,
                                                 const std::span<const RenderCamera> cameras,
                                                 const rhi::Extent2D                fallbackExtent,
                                                  const uint64_t                     projectedCostCacheEpoch,
                                                 GaussianSplatFrameStats&           stats,
                                                 RuntimeProfiler&                   profiler,
                                                 std::vector<uint16_t>&             scoreResidency,
                                                  GaussianProjectedCostGpuBuildContext* gpuContext = nullptr)
        {
            const auto prefixSelectionStart = std::chrono::steady_clock::now();
            const bool recordA4Profiling = settings.projectedCostBudgetEnabled();
            GaussianProjectedCostFrameSampleCache projectedSampleCache {};
            uint32_t mainActiveCount = 0u;
            uint64_t mainProjectedCost = 0u;
            if (settings.foveatedScoreBudgetEnabled())
            {
                const auto scoreStats = selectGaussianSplatFoveatedScoreSources(settings,
                                                                                 gpuSceneView,
                                                                                 sourceCapacity,
                                                                                 totalSplatCount,
                                                                                 cameras,
                                                                                 fallbackExtent,
                                                                                 profiler,
                                                                                 scoreResidency);
                stats.costBudget                = scoreStats.costBudget;
                stats.actualProjectedCost        = scoreStats.actualProjectedCost;
                stats.projectedCostBudgetRatio  = scoreStats.budgetRatio;
                stats.selectedCostChunks        = 0u;
                stats.numTileIntersections      = scoreStats.numTileIntersections;
                stats.sumProjectedAreaPx        = scoreStats.sumProjectedAreaPx;
                stats.a4ProjectedCostSelectionCpuMs = scoreStats.totalSelectionCpuMs;
                stats.a4CostEstimationCpuMs     = scoreStats.costEstimationCpuMs;
                stats.a4ChunkSelectionCpuMs     = scoreStats.chunkSelectionCpuMs;
                stats.a4ChunkStopCpuMs          = scoreStats.chunkSelectionCpuMs;
                stats.a4NumGaussiansOrChunksConsidered = scoreStats.gaussiansOrChunksConsidered;
                mainActiveCount                 = std::min(sourceCapacity, scoreStats.selectedCount);
                mainProjectedCost               = scoreStats.actualProjectedCost;
            }
            else if (!settings.projectedCostBudgetEnabled())
            {
                mainActiveCount = std::min(sourceCapacity, effectiveGaussianLodBudget(settings, totalSplatCount));
            }
            else
            {
                const auto costStats = selectGaussianSplatProjectedCostChunks(settings,
                                                                              gpuSceneView,
                                                                              sourceCapacity,
                                                                              directPrefix,
                                                                              cameras,
                                                                              fallbackExtent,
                                                                              projectedCostCacheEpoch,
                                                                              projectedSampleCache,
                                                                              profiler,
                                                                              gpuContext);
                stats.costBudget                = costStats.costBudget;
                stats.actualProjectedCost        = costStats.actualProjectedCost;
                stats.costOvershoot             = costStats.costOvershoot;
                stats.projectedCostBudgetRatio  = costStats.budgetRatio;
                stats.projectedCostChunkSize    = std::max(settings.projectedCostChunkSize, 1u);
                stats.selectedCostChunks        = costStats.selectedChunks;
                stats.numTileIntersections      = costStats.numTileIntersections;
                stats.sumProjectedAreaPx        = costStats.sumProjectedAreaPx;
                stats.a4ProjectedCostSelectionCpuMs = costStats.totalSelectionCpuMs;
                stats.a4CostEstimationCpuMs     = costStats.costEstimationCpuMs;
                stats.a4ChunkSelectionCpuMs     = costStats.chunkSelectionCpuMs;
                stats.a4ChunkStopCpuMs          = costStats.chunkSelectionCpuMs;
                stats.a4ChunkAggregationCpuMs   = costStats.chunkAggregationCpuMs;
                stats.a4CacheLookupCpuMs        = costStats.cacheLookupCpuMs;
                stats.a4NumChunksScanned        = costStats.chunksScanned;
                stats.a4NumGaussiansOrChunksConsidered = costStats.gaussiansOrChunksConsidered;
                stats.a4CacheHitRate            = costStats.cacheHitRate;
                stats.a4GpuCostBuildEnabled     = costStats.gpuCostBuildEnabled;
                stats.a4GpuCostBuildGpuMs       = costStats.gpuCostBuildGpuMs;
                stats.a4GpuCostReadbackCpuMs    = costStats.gpuCostReadbackCpuMs;
                stats.a4GpuCostChunkCount       = costStats.gpuCostChunkCount;
                stats.a4GpuCostValid            = costStats.gpuCostValid;
                stats.a4CpuFallbackUsed         = costStats.cpuFallbackUsed;
                stats.a4GpuCpuCostL1Error       = costStats.gpuCpuCostL1Error;
                stats.a4GpuCpuCostMaxError      = costStats.gpuCpuCostMaxError;
                stats.a4GpuCpuSelectedRatioDelta = costStats.gpuCpuSelectedRatioDelta;
                stats.a4GpuCpuActualCostDelta   = costStats.gpuCpuActualCostDelta;
                stats.a4GpuCpuOvershootDelta    = costStats.gpuCpuOvershootDelta;
                mainActiveCount                 = std::min(sourceCapacity, costStats.selectedCount);
                mainProjectedCost               = costStats.actualProjectedCost;
            }

            const auto guardStats = analyzeGaussianSplatCoverageGuardRisk(settings,
                                                                          gpuSceneView,
                                                                          sourceCapacity,
                                                                          totalSplatCount,
                                                                          mainActiveCount,
                                                                          mainProjectedCost,
                                                                          directPrefix,
                                                                          cameras,
                                                                          fallbackExtent,
                                                                          projectedCostCacheEpoch,
                                                                          projectedSampleCache,
                                                                          profiler);
            stats.foveatedCoverageGuardModeEnabled       = guardStats.modeEnabled;
            stats.foveatedCoverageGuardRiskSectors        = guardStats.riskSectors;
            stats.foveatedCoverageGuardRiskDetected       = guardStats.riskDetected;
            stats.foveatedCoverageGuardActive             = guardStats.active;
            stats.foveatedCoverageGuardRepairActive       = guardStats.active;
            stats.foveatedCoverageGuardAddedAny           = guardStats.addedCount > 0u;
            stats.foveatedCoverageGuardAddedCount         = guardStats.addedCount;
            stats.foveatedCoverageGuardAddedCost          = guardStats.addedCost;
            stats.foveatedCoverageGuardBudgetCap          = guardStats.budgetCap;
            stats.foveatedCoverageGuardMaxAdds            = guardStats.maxAddsEffective;
            stats.foveatedCoverageGuardMaxAddsEffective   = guardStats.maxAddsEffective;
            const uint32_t diagnosticRiskSectors          = guardStats.diagnosticRiskSectors;
            stats.foveatedCoverageRiskSectorCountTotal    = std::popcount(diagnosticRiskSectors & 0x7u);
            stats.foveatedCoverageRiskSectorActiveCenter  = (diagnosticRiskSectors & 0x1u) != 0u;
            stats.foveatedCoverageRiskSectorActiveMid     = (diagnosticRiskSectors & 0x2u) != 0u;
            stats.foveatedCoverageRiskSectorActiveOuter   = (diagnosticRiskSectors & 0x4u) != 0u;
            stats.foveatedCoverageRiskSectorCountCenter   =
                stats.foveatedCoverageRiskSectorActiveCenter ? 1u : 0u;
            stats.foveatedCoverageRiskSectorCountMid      =
                stats.foveatedCoverageRiskSectorActiveMid ? 1u : 0u;
            stats.foveatedCoverageRiskSectorCountOuter    =
                stats.foveatedCoverageRiskSectorActiveOuter ? 1u : 0u;
            stats.a4GuardAnalysisCpuMs                    = guardStats.analysisCpuMs;
            stats.foveatedCoverageGuardAnalysisCpuMs       = guardStats.analysisCpuMs;
            stats.foveatedCoverageGuardProjectedSampleBuildCpuMs =
                guardStats.projectedSampleBuildCpuMs;
            stats.foveatedCoverageGuardCoverageAreaCpuMs   = guardStats.coverageAreaCpuMs;
            stats.foveatedCoverageGuardCandidateScanCpuMs  = guardStats.candidateScanCpuMs;
            stats.foveatedCoverageGuardRegionRebuildCpuMs  = guardStats.regionRebuildCpuMs;
            stats.foveatedCoverageGuardCacheFullMissCount  = guardStats.cacheFullMissCount;
            stats.foveatedCoverageGuardCacheHitCount       = guardStats.cacheHitCount;
            stats.foveatedCoverageGuardCacheMissCount      = guardStats.cacheMissCount;
            stats.foveatedCoverageGuardGeometryCacheFullMissCount = guardStats.geometryCacheFullMissCount;
            stats.foveatedCoverageGuardGeometryCacheHitCount = guardStats.geometryCacheHitCount;
            stats.foveatedCoverageGuardGeometryCacheMissCount = guardStats.geometryCacheMissCount;
            stats.foveatedCoverageGuardGeometrySamplesBuilt = guardStats.geometrySamplesBuilt;
            stats.foveatedCoverageGuardGeometrySamplesReused = guardStats.geometrySamplesReused;
            stats.foveatedCoverageFailureBeforeGuard      = guardStats.coverageFailureBefore;
            stats.foveatedCoverageFailureAfterGuard       = guardStats.coverageFailureAfter;
            stats.foveatedCoverageFailureBeforeCenter     = guardStats.coverageFailureBeforeByRegion[0];
            stats.foveatedCoverageFailureBeforeMid        = guardStats.coverageFailureBeforeByRegion[1];
            stats.foveatedCoverageFailureBeforeOuter      = guardStats.coverageFailureBeforeByRegion[2];
            stats.foveatedCoverageFailureAfterCenter      = guardStats.coverageFailureAfterByRegion[0];
            stats.foveatedCoverageFailureAfterMid         = guardStats.coverageFailureAfterByRegion[1];
            stats.foveatedCoverageFailureAfterOuter       = guardStats.coverageFailureAfterByRegion[2];
            stats.foveatedCoverageScoreBeforeGuard        = guardStats.coverageScoreBefore;
            stats.foveatedCoverageScoreAfterGuard         = guardStats.coverageScoreAfter;
            stats.foveatedCoverageGuardAddedCountCenter   = guardStats.addedCountByRegion[0];
            stats.foveatedCoverageGuardAddedCountMid      = guardStats.addedCountByRegion[1];
            stats.foveatedCoverageGuardAddedCountOuter    = guardStats.addedCountByRegion[2];
            stats.foveatedCoverageGuardAddedCostCenter    = guardStats.addedCostByRegion[0];
            stats.foveatedCoverageGuardAddedCostMid       = guardStats.addedCostByRegion[1];
            stats.foveatedCoverageGuardAddedCostOuter     = guardStats.addedCostByRegion[2];
            stats.foveatedCoverageGuardRepairCappedByMaxAdds = guardStats.cappedByMaxAdds;
            stats.foveatedCoverageGuardRepairCappedByCost = guardStats.cappedByCost;
            stats.foveatedCoverageGuardRepairCapHit       =
                guardStats.cappedByMaxAdds || guardStats.cappedByCost;
            stats.foveatedCoverageGuardRepairNoCandidate  = guardStats.noCandidate;
            stats.foveatedCoverageGuardCandidateCount     = guardStats.candidateCount;
            stats.foveatedCoverageGuardCandidateCountCenter = guardStats.candidateCountByRegion[0];
            stats.foveatedCoverageGuardCandidateCountMid  = guardStats.candidateCountByRegion[1];
            stats.foveatedCoverageGuardCandidateCountOuter = guardStats.candidateCountByRegion[2];
            stats.foveatedCoverageGuardTrueCandidateCount = guardStats.trueCandidateCount;
            stats.foveatedCoverageGuardCandidateCountTruncated = guardStats.candidateCountTruncated;
            stats.foveatedCoverageGuardRequiredExtraCountEstimate =
                guardStats.requiredExtraCountEstimate;
            stats.foveatedCoverageGuardRequiredExtraAreaPx = guardStats.requiredExtraAreaPx;
            stats.foveatedCoverageGuardDeficitAreaBefore = guardStats.deficitAreaBefore;
            stats.foveatedCoverageGuardDeficitAreaAfter = guardStats.deficitAreaAfter;
            stats.foveatedCoverageGuardAddedCountToFailingSector =
                guardStats.addedCountToFailingSector;
            stats.foveatedCoverageGuardAddedCostToFailingSector =
                guardStats.addedCostToFailingSector;
            stats.foveatedCoverageGuardFailingSectorId = guardStats.failingSectorId;
            stats.foveatedCoverageGuardRepairSectorMatchRate = guardStats.repairSectorMatchRate;

            const uint32_t activePrefixCount =
                std::min(sourceCapacity, std::max(mainActiveCount, guardStats.activePrefixCount));
            const double ratioDenominator = static_cast<double>(std::max(totalSplatCount, 1u));
            stats.selectedRatioBeforeGuard = static_cast<float>(static_cast<double>(mainActiveCount) / ratioDenominator);
            stats.selectedRatioAfterGuard  = static_cast<float>(static_cast<double>(activePrefixCount) / ratioDenominator);
            stats.projectedCostBudgetRatioBeforeGuard =
                settings.projectedCostBudgetEnabled() ? stats.projectedCostBudgetRatio : stats.selectedRatioBeforeGuard;
            const uint64_t budgetForGuardRatio =
                stats.costBudget > 0u ? stats.costBudget :
                stats.actualProjectedCost > 0u ? stats.actualProjectedCost :
                guardStats.budgetCap > 0u ? guardStats.budgetCap :
                0u;
            stats.foveatedCoverageGuardAddedCostRatioToBudget =
                budgetForGuardRatio > 0u ?
                    static_cast<double>(guardStats.addedCost) / static_cast<double>(budgetForGuardRatio) :
                    -1.0;
            stats.foveatedCoverageGuardAddedCountRatioToSelected =
                mainActiveCount > 0u ?
                    static_cast<double>(guardStats.addedCount) / static_cast<double>(mainActiveCount) :
                    -1.0;
            stats.a4ProjectedSampleCacheHit     = projectedSampleCache.cacheHit;
            stats.a4ProjectedSampleCacheMiss    = projectedSampleCache.cacheMiss;
            stats.a4ProjectedSampleReused       = projectedSampleCache.sampleReused;
            stats.a4NumProjectedSamplesBuilt    = projectedSampleCache.samplesBuilt;
            stats.a4NumProjectedSamplesReused   = projectedSampleCache.samplesReused;
            stats.a4CacheEntryCount             = projectedSampleCache.cacheEntryCount;
            stats.a4CacheLookupCount            = projectedSampleCache.cacheLookupCount;
            stats.a4CacheHitCount               = projectedSampleCache.cacheHitCount;
            stats.a4CacheMissCount              = projectedSampleCache.cacheMissCount;
            stats.a4CacheFullMissCount          = projectedSampleCache.cacheFullMissCount;
            stats.a4CacheBuildCount             = projectedSampleCache.cacheBuildCount;
            stats.a4CacheEvictionCount          = projectedSampleCache.cacheEvictionCount;
            stats.a4SampleCacheKeyHash          = projectedSampleCache.sampleCacheKeyHash;
            stats.a4ChunkCacheHitCount          = projectedSampleCache.chunkCacheHitCount;
            stats.a4ChunkCacheMissCount         = projectedSampleCache.chunkCacheMissCount;
            stats.a4ProjectedSamplesBuilt       = projectedSampleCache.samplesBuilt;
            stats.a4ProjectedSamplesReused      = projectedSampleCache.samplesReused;
            stats.a4ChunkAggregationCpuMs       = projectedSampleCache.chunkAggregationCpuMs;
            stats.a4CacheLookupCpuMs            = projectedSampleCache.cacheLookupCpuMs;
            stats.a4CacheHitRate                = projectedSampleCache.hitRate();
            if (recordA4Profiling)
                stats.a4TotalSelectionCpuMs = elapsedCpuMs(prefixSelectionStart);
            return activePrefixCount;
        }

        void applyGaussianSplatFoveatedClodSettings(resource::GpuSceneView&            gpuSceneView,
                                                    const GaussianSplatRenderSettings& settings,
                                                    const GaussianSplatFrameStats&     stats)
        {
            const auto layers = settings.foveatedLayers();
            const bool shaderGazeAnchorCrossfade =
                settings.shaderAntiPopActive() &&
                settings.shaderAntiPopMode == GaussianSplatShaderAntiPopMode::eGuardedGazeAnchorCrossfade;
            const bool shaderEccStochasticTransition =
                settings.shaderAntiPopActive() &&
                settings.shaderAntiPopMode ==
                    GaussianSplatShaderAntiPopMode::eEccentricityStochasticTransition;
            const bool shaderCoverageStableRelease =
                settings.shaderAntiPopActive() &&
                settings.shaderAntiPopMode ==
                    GaussianSplatShaderAntiPopMode::eCoverageStableLogpolarRelease;
            const bool shaderOrderFreeFoveatedActive =
                settings.shaderAntiPopActive() && settings.shaderAntiPopOrderFree && settings.foveatedClodEnabled;
            const bool foveatedSelectionActive = settings.foveatedClodActive() || shaderOrderFreeFoveatedActive;
            const glm::vec2 shaderGaze =
                shaderGazeAnchorCrossfade ?
                    glm::vec2 {stats.gazeAnchorNewAnchorX, stats.gazeAnchorNewAnchorY} :
                    shaderEccStochasticTransition ?
                    glm::vec2 {stats.eccStochasticNewGazeX, stats.eccStochasticNewGazeY} :
                    settings.foveatedGaze;
            gpuSceneView.setGeneralGaussianSplatFoveatedClod(foveatedSelectionActive,
                                                             settings.foveatedLayeredCompositeActive(),
                                                             settings.foveatedCoverageCompensationEnabled,
                                                             shaderGaze,
                                                             glm::vec2 {layers[0].eccentricityDegrees,
                                                                        layers[1].eccentricityDegrees},
                                                             glm::vec3 {layers[0].lodLevel,
                                                                        layers[1].lodLevel,
                                                                        layers[2].lodLevel},
                                                             glm::vec3 {layers[0].resolutionScale,
                                                                        layers[1].resolutionScale,
                                                                        layers[2].resolutionScale},
                                                             std::max(settings.foveatedTransitionDegrees, 0.0f),
                                                             static_cast<uint32_t>(settings.foveatedCoverageGuardMode),
                                                             std::max(settings.foveatedCoverageProtectionDegrees, 0.0f),
                                                             std::clamp(settings.foveatedCoverageGuardBudgetRatio,
                                                                        0.0f,
                                                                        0.25f),
                                                             glm::clamp(settings.foveatedCoverageGuardMinLevels,
                                                                        glm::vec3 {0.0f},
                                                                        glm::vec3 {1.0f}),
                                                             settings.foveatedCoverageGuardMode ==
                                                                     GaussianSplatFoveatedCoverageGuardMode::eRiskTriggered ?
                                                                 stats.foveatedCoverageGuardRiskSectors :
                                                                 0x7u,
                                                             static_cast<uint32_t>(settings.foveatedDistribution),
                                                             std::max(settings.foveatedContinuousTheta0Degrees, 1e-4f),
                                                             std::max(settings.foveatedContinuousAlpha, 0.0f),
                                                             std::clamp(settings.foveatedContinuousMinLevel, 0.0f, 1.0f),
                                                             std::max(settings.foveatedTemporalPeripheralScale, 0.0f),
                                                             settings.foveatedTemporalHysteresisEnabled,
                                                             settings.foveatedBoundarySmoothingEnabled,
                                                             std::clamp(settings.foveatedTemporalResidencyFrames, 0u, 255u),
                                                             std::clamp(settings.foveatedTemporalHysteresisRatio, 0.0f, 1.0f),
                                                             std::clamp(settings.foveatedBoundarySmoothingRatio, 0.0f, 1.0f),
                                                             settings.foveatedShLodActive(),
                                                             settings.foveatedShSmoothSuppressionActive(),
                                                             settings.peripheralTemporalFilterEnabled,
                                                             settings.peripheralTemporalFilterOuterDegrees,
                                                             settings.peripheralTemporalFilterLambdaScale,
                                                             settings.peripheralTemporalFilterRejectionThreshold,
                                                             settings.peripheralTemporalFilterClampRadius,
                                                             settings.foveatedShLodDegrees,
                                                             static_cast<uint32_t>(
                                                                      settings.foveatedShLodGuardMode),
                                                             std::max(settings.foveatedShLodGuardThresholdMid, 0.0f),
                                                             std::max(settings.foveatedShLodGuardThresholdHigh,
                                                                      settings.foveatedShLodGuardThresholdMid));
            gpuSceneView.generalGaussianSplatFoveatedScoreSelected = settings.foveatedScoreBudgetEnabled();
            const float shaderGuardThreshold =
                shaderEccStochasticTransition ?
                    std::max(settings.eccStochasticMinPDelta, 0.0f) :
                    settings.shaderAntiPopGuardThreshold;
            gpuSceneView.setGeneralGaussianSplatShaderAntiPop(
                settings.shaderAntiPopActive() ? static_cast<uint32_t>(settings.shaderAntiPopMode) : 0u,
                settings.shaderAntiPopHashSeed,
                settings.shaderAntiPopRampWidth,
                shaderGuardThreshold,
                settings.shaderAntiPopGuardFloor,
                static_cast<uint32_t>(settings.shaderAntiPopPKeepCurve),
                settings.shaderAntiPopPrefixRatio,
                static_cast<uint32_t>(settings.shaderAntiPopNormalizeMode),
                settings.shaderAntiPopNormalizeStrength,
                settings.shaderAntiPopNormalizeClampMin,
                settings.shaderAntiPopNormalizeClampMax,
                settings.shaderAntiPopNormalizeFactor);
            uint32_t coverageReleaseFlags = 0u;
            const bool coverageTexturePathEnabled = settings.coverageTextureFloorEnabled;
            if (coverageTexturePathEnabled)
                coverageReleaseFlags |= 32u;
            else if (settings.coverageStableReleaseCoverageFloorEnabled)
                coverageReleaseFlags |= 1u;
            if (settings.guideBeforeDiscardEnabled)
                coverageReleaseFlags |= 1u | 128u;
            if (settings.delayedGuideLogPolarEnabled)
                coverageReleaseFlags |= 128u;
            if (settings.coverageStableReleaseStableHashEnabled)
                coverageReleaseFlags |= 2u;
            if (settings.coverageStableReleaseStaggeredReleaseEnabled)
                coverageReleaseFlags |= 4u;
            const bool coverageReleaseDiagnosticsRequested =
                !settings.cleanTimingMode &&
                shaderCoverageStableRelease &&
                (settings.shaderAntiPopDebugLogEnabled || settings.coverageTextureFloorDebugEnabled);
            if (coverageReleaseDiagnosticsRequested)
                coverageReleaseFlags |= 8u;
            if (settings.coverageStableReleaseSaturatedFloorEnabled)
                coverageReleaseFlags |= 16u;
            gpuSceneView.setGeneralGaussianSplatCoverageStableRelease(
                coverageReleaseFlags,
                coverageTexturePathEnabled ?
                    settings.coverageTextureFloorWidth :
                    settings.coverageStableReleaseTileGridX,
                coverageTexturePathEnabled ?
                    settings.coverageTextureFloorHeight :
                    settings.coverageStableReleaseTileGridY,
                static_cast<uint32_t>(stats.frameIndex & 0xffffu),
                settings.coverageStableReleaseLambda,
                coverageTexturePathEnabled ?
                    settings.coverageTextureFloorDMin :
                    settings.coverageStableReleaseDMin,
                coverageTexturePathEnabled ?
                    settings.coverageTextureFloorSigmaMax :
                    settings.coverageStableReleaseSigmaMax,
                settings.coverageStableReleaseReleaseEpsilon);
            gpuSceneView.setGeneralGaussianSplatCoverageTextureFloor(
                settings.coverageTextureFloorStrength,
                settings.coverageTextureFloorHistoryBeta,
                settings.coverageTextureFloorUpdateInterval);
            gpuSceneView.setGeneralGaussianSplatDelayedGuideTemporalReleaseCost(
                static_cast<uint32_t>(settings.delayedGuideTemporalReleasePolicy),
                settings.delayedGuideTemporalReleaseCapRatio,
                settings.delayedGuideTemporalRiskThreshold,
                settings.delayedGuideTemporalFootprintDecayScale,
                settings.delayedGuideTemporalLargeFootprintPx);
            gpuSceneView.generalGaussianSplatEcsptCounterReadbackEnabled =
                !settings.cleanTimingMode &&
                (settings.shaderAntiPopDebugLogEnabled ||
                 settings.eccStochasticDebugLogEnabled ||
                 coverageReleaseDiagnosticsRequested);
            uint32_t gazeAnchorFlags = 0u;
            if (shaderGazeAnchorCrossfade)
            {
                gazeAnchorFlags |= 1u;
                if (settings.gazeAnchorImmediateFoveaFill)
                    gazeAnchorFlags |= 2u;
                if (settings.gazeAnchorContributionGuard)
                    gazeAnchorFlags |= 4u;
                if (settings.gazeAnchorTransitionBudgetRatio > 0.0f &&
                    !settings.gazeAnchorContributionGuard)
                    gazeAnchorFlags |= 8u;
            }
            gpuSceneView.setGeneralGaussianSplatGazeAnchorCrossfade(
                gazeAnchorFlags,
                glm::vec2 {stats.gazeAnchorAnchorX, stats.gazeAnchorAnchorY},
                glm::vec2 {stats.gazeAnchorNewAnchorX, stats.gazeAnchorNewAnchorY},
                stats.gazeAnchorFadePhase,
                settings.gazeAnchorTransitionBudgetRatio,
                settings.gazeAnchorFadeFrames,
                settings.gazeAnchorMinUpdateFrames);
            if (shaderEccStochasticTransition)
            {
                uint32_t eccFlags = 1u;
                if (settings.eccStochasticProtectOldFoveaDegrees >= 0.0f)
                    eccFlags |= 2u;
                if (settings.eccStochasticProtectNewFoveaDegrees >= 0.0f)
                    eccFlags |= 4u;
                if (settings.eccStochasticBoundaryBandDegrees > 0.0f)
                    eccFlags |= 8u;
                if (settings.eccStochasticContributionGuard)
                    eccFlags |= 16u;
                gpuSceneView.setGeneralGaussianSplatEccentricityStochasticTransition(
                    eccFlags,
                    glm::vec2 {stats.eccStochasticOldGazeX, stats.eccStochasticOldGazeY},
                    glm::vec2 {stats.eccStochasticNewGazeX, stats.eccStochasticNewGazeY},
                    stats.eccStochasticFadePhase,
                    settings.eccStochasticProtectOldFoveaDegrees,
                    settings.eccStochasticProtectNewFoveaDegrees,
                    settings.eccStochasticBoundaryBandDegrees,
                    settings.eccStochasticFadeFrames);
            }
            if (settings.shaderAntiPopDebugLogEnabled && settings.shaderAntiPopActive())
            {
                VULTRA_CLIENT_INFO(
                    "GPU anti-pop scene-view uniforms: mode={}, hash_seed={}, ramp_width={}, guard_threshold={}, "
                    "guard_floor={}, pkeep_curve={}, prefix_ratio={}, normalize_mode={}, normalize_factor={}",
                    gpuSceneView.generalGaussianSplatShaderAntiPopMode,
                    gpuSceneView.generalGaussianSplatShaderAntiPopHashSeed,
                    gpuSceneView.generalGaussianSplatShaderAntiPopRampWidth,
                    gpuSceneView.generalGaussianSplatShaderAntiPopGuardThreshold,
                    gpuSceneView.generalGaussianSplatShaderAntiPopGuardFloor,
                    gpuSceneView.generalGaussianSplatShaderAntiPopPKeepCurve,
                    gpuSceneView.generalGaussianSplatShaderAntiPopPrefixRatio,
                    gpuSceneView.generalGaussianSplatShaderAntiPopNormalizeMode,
                    gpuSceneView.generalGaussianSplatShaderAntiPopNormalizeFactor);
            }
            gpuSceneView.generalGaussianSplatDeterministicSourceOrderSort =
                settings.cachedSelectionOracleDeterministicSourceOrderSort;
        }

        bool updateGaussianSplatShaderGazeAnchorCrossfadeState(
            const GaussianSplatRenderSettings& settings,
            const std::span<const RenderCamera> cameras,
            GaussianSplatFrameStats& stats,
            bool& anchorValid,
            glm::vec2& anchorGaze,
            glm::vec2& previousAnchorGaze,
            uint64_t& anchorLastUpdateFrame,
            uint32_t& anchorUpdateEventCount,
            uint32_t& anchorDeadbandViolationCount)
        {
            const bool active =
                settings.shaderAntiPopActive() &&
                settings.shaderAntiPopMode == GaussianSplatShaderAntiPopMode::eGuardedGazeAnchorCrossfade;
            if (!active)
                return false;

            const glm::vec2 currentGaze = glm::clamp(settings.foveatedGaze, glm::vec2 {0.0f}, glm::vec2 {1.0f});
            if (!anchorValid)
            {
                anchorValid = true;
                anchorGaze = currentGaze;
                previousAnchorGaze = currentGaze;
                anchorLastUpdateFrame = stats.frameIndex;
                anchorUpdateEventCount = 0u;
                anchorDeadbandViolationCount = 0u;
            }

            const glm::vec2 oldAnchorBeforeUpdate = anchorGaze;
            float anchorDistance = 0.0f;
            if (!cameras.empty())
                anchorDistance =
                    gaussianSplatGazeAngularDistanceDegreesCpu(oldAnchorBeforeUpdate, currentGaze, cameras.front());
            else
                anchorDistance = glm::length((currentGaze - oldAnchorBeforeUpdate) * glm::vec2 {90.0f, 90.0f});

            const uint64_t frameDelta =
                stats.frameIndex >= anchorLastUpdateFrame ? stats.frameIndex - anchorLastUpdateFrame : 0u;
            const bool deadbandViolated = anchorDistance > std::max(settings.gazeAnchorDeadbandDegrees, 0.0f);
            const bool minUpdateSatisfied = frameDelta >= static_cast<uint64_t>(settings.gazeAnchorMinUpdateFrames);
            const bool updateEvent = deadbandViolated && minUpdateSatisfied;
            if (deadbandViolated)
                ++anchorDeadbandViolationCount;
            if (updateEvent)
            {
                previousAnchorGaze = anchorGaze;
                anchorGaze = currentGaze;
                anchorLastUpdateFrame = stats.frameIndex;
                ++anchorUpdateEventCount;
            }

            const uint64_t framesSinceUpdate =
                stats.frameIndex >= anchorLastUpdateFrame ? stats.frameIndex - anchorLastUpdateFrame : 0u;
            const float fadeFrames = static_cast<float>(std::max(settings.gazeAnchorFadeFrames, 1u));
            const float fadePhase = updateEvent ? 0.0f :
                std::clamp(static_cast<float>(framesSinceUpdate) / fadeFrames, 0.0f, 1.0f);

            stats.gazeAnchorCrossfadeEnabled = true;
            stats.gazeAnchorGpuCrossfadeEnabled = true;
            stats.gazeAnchorDeadbandDegrees = std::max(settings.gazeAnchorDeadbandDegrees, 0.0f);
            stats.gazeAnchorFadeFrames = settings.gazeAnchorFadeFrames;
            stats.gazeAnchorMinUpdateFrames = settings.gazeAnchorMinUpdateFrames;
            stats.gazeAnchorTransitionBudgetRatio = settings.gazeAnchorTransitionBudgetRatio;
            stats.gazeAnchorImmediateFoveaFill = settings.gazeAnchorImmediateFoveaFill;
            stats.gazeAnchorContributionGuard = settings.gazeAnchorContributionGuard;
            stats.gazeAnchorDebugLogEnabled = settings.gazeAnchorDebugLogEnabled;
            stats.gazeAnchorAnchorX = previousAnchorGaze.x;
            stats.gazeAnchorAnchorY = previousAnchorGaze.y;
            stats.gazeAnchorNewAnchorX = anchorGaze.x;
            stats.gazeAnchorNewAnchorY = anchorGaze.y;
            stats.gazeAnchorDistanceDegrees = anchorDistance;
            stats.gazeAnchorFramesSinceUpdate =
                static_cast<uint32_t>(std::min<uint64_t>(framesSinceUpdate, std::numeric_limits<uint32_t>::max()));
            stats.gazeAnchorFadePhase = fadePhase;
            stats.gazeAnchorUpdateEvent = updateEvent;
            stats.gazeAnchorUpdateEventCount = anchorUpdateEventCount;
            stats.gazeAnchorDeadbandViolationCount = anchorDeadbandViolationCount;
            return true;
        }

        bool updateGaussianSplatShaderEccStochasticTransitionState(
            const GaussianSplatRenderSettings& settings,
            GaussianSplatFrameStats&           stats,
            bool&                              stateValid,
            glm::vec2&                         oldGaze,
            glm::vec2&                         newGaze,
            uint64_t&                          lastUpdateFrame,
            uint32_t&                          updateEventCount)
        {
            const bool active =
                settings.shaderAntiPopActive() &&
                settings.shaderAntiPopMode ==
                    GaussianSplatShaderAntiPopMode::eEccentricityStochasticTransition;
            if (!active)
                return false;

            const glm::vec2 currentGaze =
                glm::clamp(settings.foveatedGaze, glm::vec2 {0.0f}, glm::vec2 {1.0f});
            bool updateEvent = false;
            if (!stateValid)
            {
                stateValid = true;
                oldGaze = currentGaze;
                newGaze = currentGaze;
                lastUpdateFrame = stats.frameIndex;
                updateEventCount = 0u;
            }
            else if (glm::length(currentGaze - newGaze) > 1e-5f)
            {
                oldGaze = newGaze;
                newGaze = currentGaze;
                lastUpdateFrame = stats.frameIndex;
                ++updateEventCount;
                updateEvent = true;
            }

            const uint64_t framesSinceUpdate =
                stats.frameIndex >= lastUpdateFrame ? stats.frameIndex - lastUpdateFrame : 0u;
            const float fadePhase =
                settings.eccStochasticFadeFrames == 0u ?
                    1.0f :
                    std::clamp(static_cast<float>(framesSinceUpdate) /
                                   static_cast<float>(std::max(settings.eccStochasticFadeFrames, 1u)),
                               0.0f,
                               1.0f);

            stats.eccStochasticTransitionEnabled = true;
            stats.eccStochasticFadeFrames = settings.eccStochasticFadeFrames;
            stats.eccStochasticProtectOldFoveaDegrees = settings.eccStochasticProtectOldFoveaDegrees;
            stats.eccStochasticProtectNewFoveaDegrees = settings.eccStochasticProtectNewFoveaDegrees;
            stats.eccStochasticBoundaryBandDegrees = std::max(settings.eccStochasticBoundaryBandDegrees, 0.0f);
            stats.eccStochasticMinPDelta = std::max(settings.eccStochasticMinPDelta, 0.0f);
            stats.eccStochasticContributionGuard = settings.eccStochasticContributionGuard;
            stats.eccStochasticDebugLogEnabled = settings.eccStochasticDebugLogEnabled;
            stats.eccStochasticOldGazeX = oldGaze.x;
            stats.eccStochasticOldGazeY = oldGaze.y;
            stats.eccStochasticNewGazeX = newGaze.x;
            stats.eccStochasticNewGazeY = newGaze.y;
            stats.eccStochasticFadePhase = fadePhase;
            stats.eccStochasticFramesSinceUpdate =
                static_cast<uint32_t>(std::min<uint64_t>(framesSinceUpdate,
                                                         std::numeric_limits<uint32_t>::max()));
            stats.eccStochasticUpdateEvent = updateEvent;
            stats.eccStochasticUpdateEventCount = updateEventCount;
            return true;
        }

        void applyGaussianSplatShaderGazeAnchorProxyStats(const GaussianSplatRenderSettings& settings,
                                                          const uint32_t activeGaussianSplats,
                                                          GaussianSplatFrameStats& stats)
        {
            if (!(settings.shaderAntiPopActive() &&
                  settings.shaderAntiPopMode == GaussianSplatShaderAntiPopMode::eGuardedGazeAnchorCrossfade))
            {
                return;
            }

            stats.gazeAnchorUnionCount = activeGaussianSplats;
            stats.gazeAnchorUnionCountRatio = 1.0f;
            stats.gazeAnchorTransitionBudgetTargetCount =
                settings.gazeAnchorTransitionBudgetRatio > 0.0f ?
                    static_cast<uint32_t>(std::ceil(static_cast<double>(activeGaussianSplats) *
                                                    static_cast<double>(settings.gazeAnchorTransitionBudgetRatio))) :
                    0u;
            if (settings.gazeAnchorTransitionBudgetRatio > 0.0f &&
                stats.gazeAnchorTransitionBudgetTargetCount < activeGaussianSplats)
            {
                stats.gazeAnchorTransitionBudgetCappedCount =
                    activeGaussianSplats - stats.gazeAnchorTransitionBudgetTargetCount;
                stats.gazeAnchorDroppedOldOnlyCount = stats.gazeAnchorTransitionBudgetCappedCount / 2u;
                stats.gazeAnchorDroppedNewOnlyCount =
                    stats.gazeAnchorTransitionBudgetCappedCount - stats.gazeAnchorDroppedOldOnlyCount;
            }
            if (settings.gazeAnchorContributionGuard && stats.gazeAnchorTransitionBudgetCappedCount > 0u)
            {
                stats.gazeAnchorHighContributionProtectedCount =
                    std::min<uint32_t>(stats.gazeAnchorTransitionBudgetCappedCount,
                                       std::max<uint32_t>(1u, activeGaussianSplats / 20u));
            }
        }

        void applyGaussianSplatShaderEccStochasticProxyStats(const GaussianSplatRenderSettings& settings,
                                                             const uint32_t activeGaussianSplats,
                                                             GaussianSplatFrameStats& stats)
        {
            if (!(settings.shaderAntiPopActive() &&
                  (settings.shaderAntiPopMode ==
                       GaussianSplatShaderAntiPopMode::eEccentricityStochasticTransition ||
                   settings.shaderAntiPopMode ==
                       GaussianSplatShaderAntiPopMode::eStableOpticalDepthThinning)))
            {
                return;
            }

        }

        uint32_t gaussianSplatShaderAntiPopGuardProxyMode(const GaussianSplatRenderSettings& settings)
        {
            if (settings.shaderAntiPopMode == GaussianSplatShaderAntiPopMode::eHashRampGuarded)
                return 1u;
            if (settings.shaderAntiPopMode == GaussianSplatShaderAntiPopMode::eGuardedGazeAnchorCrossfade &&
                settings.gazeAnchorContributionGuard)
                return 2u;
            if (settings.shaderAntiPopMode ==
                    GaussianSplatShaderAntiPopMode::eEccentricityStochasticTransition &&
                settings.eccStochasticContributionGuard)
                return 3u;
            return 0u;
        }

        void copyGaussianSplatShaderGazeAnchorStats(const GaussianSplatFrameStats& source,
                                                    GaussianSplatFrameStats& target)
        {
            target.gazeAnchorCrossfadeEnabled = source.gazeAnchorCrossfadeEnabled;
            target.gazeAnchorGpuCrossfadeEnabled = source.gazeAnchorGpuCrossfadeEnabled;
            target.gazeAnchorDeadbandDegrees = source.gazeAnchorDeadbandDegrees;
            target.gazeAnchorFadeFrames = source.gazeAnchorFadeFrames;
            target.gazeAnchorMinUpdateFrames = source.gazeAnchorMinUpdateFrames;
            target.gazeAnchorTransitionBudgetRatio = source.gazeAnchorTransitionBudgetRatio;
            target.gazeAnchorImmediateFoveaFill = source.gazeAnchorImmediateFoveaFill;
            target.gazeAnchorContributionGuard = source.gazeAnchorContributionGuard;
            target.gazeAnchorDebugLogEnabled = source.gazeAnchorDebugLogEnabled;
            target.gazeAnchorAnchorX = source.gazeAnchorAnchorX;
            target.gazeAnchorAnchorY = source.gazeAnchorAnchorY;
            target.gazeAnchorNewAnchorX = source.gazeAnchorNewAnchorX;
            target.gazeAnchorNewAnchorY = source.gazeAnchorNewAnchorY;
            target.gazeAnchorDistanceDegrees = source.gazeAnchorDistanceDegrees;
            target.gazeAnchorFramesSinceUpdate = source.gazeAnchorFramesSinceUpdate;
            target.gazeAnchorFadePhase = source.gazeAnchorFadePhase;
            target.gazeAnchorUpdateEvent = source.gazeAnchorUpdateEvent;
            target.gazeAnchorUpdateEventCount = source.gazeAnchorUpdateEventCount;
            target.gazeAnchorDeadbandViolationCount = source.gazeAnchorDeadbandViolationCount;
        }

        void copyGaussianSplatShaderEccStochasticStats(const GaussianSplatFrameStats& source,
                                                       GaussianSplatFrameStats&       target)
        {
            target.eccStochasticTransitionEnabled = source.eccStochasticTransitionEnabled;
            target.eccStochasticFadeFrames = source.eccStochasticFadeFrames;
            target.eccStochasticProtectOldFoveaDegrees = source.eccStochasticProtectOldFoveaDegrees;
            target.eccStochasticProtectNewFoveaDegrees = source.eccStochasticProtectNewFoveaDegrees;
            target.eccStochasticBoundaryBandDegrees = source.eccStochasticBoundaryBandDegrees;
            target.eccStochasticMinPDelta = source.eccStochasticMinPDelta;
            target.eccStochasticContributionGuard = source.eccStochasticContributionGuard;
            target.eccStochasticDebugLogEnabled = source.eccStochasticDebugLogEnabled;
            target.eccStochasticOldGazeX = source.eccStochasticOldGazeX;
            target.eccStochasticOldGazeY = source.eccStochasticOldGazeY;
            target.eccStochasticNewGazeX = source.eccStochasticNewGazeX;
            target.eccStochasticNewGazeY = source.eccStochasticNewGazeY;
            target.eccStochasticFadePhase = source.eccStochasticFadePhase;
            target.eccStochasticFramesSinceUpdate = source.eccStochasticFramesSinceUpdate;
            target.eccStochasticUpdateEvent = source.eccStochasticUpdateEvent;
            target.eccStochasticUpdateEventCount = source.eccStochasticUpdateEventCount;
            target.eccStochasticCounterReadbackValid = source.eccStochasticCounterReadbackValid;
            target.eccStochasticCounterReadbackCpuMs = source.eccStochasticCounterReadbackCpuMs;
            target.eccStochasticTotalCandidatesSeen = source.eccStochasticTotalCandidatesSeen;
            target.eccStochasticBaseNewSelectedCount = source.eccStochasticBaseNewSelectedCount;
            target.eccStochasticEffectiveVisibleAfterEcsptCount =
                source.eccStochasticEffectiveVisibleAfterEcsptCount;
            target.eccStochasticSharedCount = source.eccStochasticSharedCount;
            target.eccStochasticUpgradeCount = source.eccStochasticUpgradeCount;
            target.eccStochasticDowngradeCount = source.eccStochasticDowngradeCount;
            target.eccStochasticProtectedDowngradeCount = source.eccStochasticProtectedDowngradeCount;
            target.eccStochasticDroppedDowngradeCount = source.eccStochasticDroppedDowngradeCount;
            target.eccStochasticOldOnlyFadeVisibleCount = source.eccStochasticOldOnlyFadeVisibleCount;
            target.eccStochasticImmediateNewFoveaCount = source.eccStochasticImmediateNewFoveaCount;
            target.eccStochasticBoundaryProtectedCount = source.eccStochasticBoundaryProtectedCount;
            target.eccStochasticContributionGuardProtectedCount =
                source.eccStochasticContributionGuardProtectedCount;
            target.eccStochasticZeroWeightDiscardCount = source.eccStochasticZeroWeightDiscardCount;
            target.eccStochasticMinPDeltaDiscardCount = source.eccStochasticMinPDeltaDiscardCount;
            target.eccStochasticFarPeripheryHardDropCount = source.eccStochasticFarPeripheryHardDropCount;
            target.stableOpticalDepthAlphaClampedCount = source.stableOpticalDepthAlphaClampedCount;
            target.coverageStableReleaseFloorActiveCount = source.coverageStableReleaseFloorActiveCount;
            target.coverageStableReleaseFloorRaisedCount = source.coverageStableReleaseFloorRaisedCount;
            target.coverageStableReleaseHeldCount = source.coverageStableReleaseHeldCount;
            target.coverageStableReleaseDroppedCount = source.coverageStableReleaseDroppedCount;
            target.coverageStableReleaseOldOnlyCount = source.coverageStableReleaseOldOnlyCount;
            target.coverageStableReleaseFloorSampleCount = source.coverageStableReleaseFloorSampleCount;
            target.coverageStableReleasePFloorMean = source.coverageStableReleasePFloorMean;
            target.coverageStableReleasePFloorMax = source.coverageStableReleasePFloorMax;
            target.coverageStableReleaseDAlphaMean = source.coverageStableReleaseDAlphaMean;
            target.coverageStableReleaseNEffMean = source.coverageStableReleaseNEffMean;
            target.coverageStableReleaseTileTotalCount = source.coverageStableReleaseTileTotalCount;
            target.coverageStableReleaseTileNonEmptyCount = source.coverageStableReleaseTileNonEmptyCount;
            target.coverageStableReleaseTileEmptyCount = source.coverageStableReleaseTileEmptyCount;
            target.coverageStableReleaseDAlphaMeanAll = source.coverageStableReleaseDAlphaMeanAll;
            target.coverageStableReleaseDAlphaMeanNonEmpty = source.coverageStableReleaseDAlphaMeanNonEmpty;
            target.coverageStableReleaseDAlphaP50NonEmpty = source.coverageStableReleaseDAlphaP50NonEmpty;
            target.coverageStableReleaseDAlphaP90NonEmpty = source.coverageStableReleaseDAlphaP90NonEmpty;
            target.coverageStableReleaseDAlphaP95NonEmpty = source.coverageStableReleaseDAlphaP95NonEmpty;
            target.coverageStableReleaseNEffMeanNonEmpty = source.coverageStableReleaseNEffMeanNonEmpty;
            target.coverageStableReleaseNEffP50NonEmpty = source.coverageStableReleaseNEffP50NonEmpty;
            target.coverageStableReleaseNEffP90NonEmpty = source.coverageStableReleaseNEffP90NonEmpty;
            target.coverageStableReleaseNEffP95NonEmpty = source.coverageStableReleaseNEffP95NonEmpty;
            target.coverageStableReleasePFloorMeanAll = source.coverageStableReleasePFloorMeanAll;
            target.coverageStableReleasePFloorMeanNonEmpty = source.coverageStableReleasePFloorMeanNonEmpty;
            target.coverageStableReleasePFloorP90NonEmpty = source.coverageStableReleasePFloorP90NonEmpty;
            target.coverageStableReleasePFloorP95NonEmpty = source.coverageStableReleasePFloorP95NonEmpty;
            target.coverageStableReleasePLpMean = source.coverageStableReleasePLpMean;
            target.coverageStableReleasePStaticMean = source.coverageStableReleasePStaticMean;
            target.coverageStableReleaseFloorActiveFractionVsCandidates =
                source.coverageStableReleaseFloorActiveFractionVsCandidates;
            target.coverageStableReleaseFloorRaisedFractionVsEffectiveVisible =
                source.coverageStableReleaseFloorRaisedFractionVsEffectiveVisible;
            target.coverageStableReleaseDeltaSum = source.coverageStableReleaseDeltaSum;
            target.coverageStableReleaseDeltaMeanOverCandidates =
                source.coverageStableReleaseDeltaMeanOverCandidates;
            target.coverageStableReleaseDeltaMeanOverHeld = source.coverageStableReleaseDeltaMeanOverHeld;
            target.coverageStableReleaseHeldEffectiveRatio = source.coverageStableReleaseHeldEffectiveRatio;
            target.coverageStableReleaseHeldEffectiveCount = source.coverageStableReleaseHeldEffectiveCount;
            target.coverageStableReleaseHistoryResetCount = source.coverageStableReleaseHistoryResetCount;
            target.coverageStableReleaseReentryResetCount = source.coverageStableReleaseReentryResetCount;
            target.coverageStableReleaseEpsilonCutoffCount =
                source.coverageStableReleaseEpsilonCutoffCount;
            target.coverageStableReleaseSaturatedFloorEnabled =
                source.coverageStableReleaseSaturatedFloorEnabled;
            target.coverageStableReleaseFloorCellsSafeSaturated =
                source.coverageStableReleaseFloorCellsSafeSaturated;
            target.coverageStableReleaseFloorCellsUnsafe =
                source.coverageStableReleaseFloorCellsUnsafe;
            target.coverageStableReleaseSkippedFloorUpdateCount =
                source.coverageStableReleaseSkippedFloorUpdateCount;
            target.coverageStableReleaseDAlphaSafeThreshold =
                source.coverageStableReleaseDAlphaSafeThreshold;
            target.coverageStableReleaseNEffSafeThreshold =
                source.coverageStableReleaseNEffSafeThreshold;
            target.coverageStableReleaseFloorSaturationRatio =
                source.coverageStableReleaseFloorSaturationRatio;
            target.coverageStableReleaseCandidateAlphaMass =
                source.coverageStableReleaseCandidateAlphaMass;
            target.coverageStableReleaseRetainedAlphaMass =
                source.coverageStableReleaseRetainedAlphaMass;
            target.coverageTextureCurrentAlphaMass =
                source.coverageTextureCurrentAlphaMass;
            target.coverageTextureCurrentAlpha2Mass =
                source.coverageTextureCurrentAlpha2Mass;
            target.coverageTextureNonzeroTexelCount =
                source.coverageTextureNonzeroTexelCount;
            target.coverageTextureWidth = source.coverageTextureWidth;
            target.coverageTextureHeight = source.coverageTextureHeight;
            target.coverageTextureHistoryBeta = source.coverageTextureHistoryBeta;
            target.coverageTextureStrength = source.coverageTextureStrength;
            target.coverageTexturePFinalMean =
                source.coverageTexturePFinalMean;
            target.coverageTextureStableHashKeptCount =
                source.coverageTextureStableHashKeptCount;
            target.guideBeforeDiscardEnabled =
                source.guideBeforeDiscardEnabled;
            target.guideBeforeDiscardTexelCount =
                source.guideBeforeDiscardTexelCount;
            target.guideBeforeDiscardCoverageMean =
                source.guideBeforeDiscardCoverageMean;
            target.guideBeforeDiscardCoverageMin =
                source.guideBeforeDiscardCoverageMin;
            target.guideBeforeDiscardCoverageMax =
                source.guideBeforeDiscardCoverageMax;
            target.guideBeforeDiscardLowCoverageBoostedSplats =
                source.guideBeforeDiscardLowCoverageBoostedSplats;
            target.guideBeforeDiscardHighCoverageReducedSplats =
                source.guideBeforeDiscardHighCoverageReducedSplats;
            target.guideBeforeDiscardSelectedBeforeTileExpansion =
                source.guideBeforeDiscardSelectedBeforeTileExpansion;
            target.guideBeforeDiscardTileDuplicateCount =
                source.guideBeforeDiscardTileDuplicateCount;
            target.delayedGuideLogPolarEnabled =
                source.delayedGuideLogPolarEnabled;
            target.delayedGuideTemporalLogPolarEnabled =
                source.delayedGuideTemporalLogPolarEnabled;
            target.delayedGuideReleaseActiveCount =
                source.delayedGuideReleaseActiveCount;
            target.delayedGuidePHistoryNonzeroCount =
                source.delayedGuidePHistoryNonzeroCount;
            target.delayedGuidePTargetLessThanHistoryCount =
                source.delayedGuidePTargetLessThanHistoryCount;
            target.delayedGuidePHistoryBytes =
                source.delayedGuidePHistoryBytes;
            target.delayedGuideReleasedEffectiveVisibleCount =
                source.delayedGuideReleasedEffectiveVisibleCount;
            target.delayedGuideReleasedAlphaProxySum =
                source.delayedGuideReleasedAlphaProxySum;
            target.delayedGuideReleaseCapHitCount =
                source.delayedGuideReleaseCapHitCount;
            target.delayedGuideRiskProtectedCount =
                source.delayedGuideRiskProtectedCount;
            target.delayedGuideFootprintFastDecayCount =
                source.delayedGuideFootprintFastDecayCount;
            target.delayedGuideBaseTileDuplicateProxyCount =
                source.delayedGuideBaseTileDuplicateProxyCount;
            target.delayedGuideReleasedTileDuplicateProxyCount =
                source.delayedGuideReleasedTileDuplicateProxyCount;
            target.delayedGuideReleasedDuplicateProxyRatio =
                source.delayedGuideReleasedDuplicateProxyRatio;
            target.delayedGuideLargeFootprintReleasedCount =
                source.delayedGuideLargeFootprintReleasedCount;
            target.delayedGuideLargeFootprintReleasedTileDuplicateProxyCount =
                source.delayedGuideLargeFootprintReleasedTileDuplicateProxyCount;
            target.delayedGuideReleasedRadiusMean =
                source.delayedGuideReleasedRadiusMean;
            target.delayedGuideReleasedRadiusP95 =
                source.delayedGuideReleasedRadiusP95;
            target.delayedGuideReleasedRadiusMax =
                source.delayedGuideReleasedRadiusMax;
            target.delayedGuideHotspotTileIds =
                source.delayedGuideHotspotTileIds;
            target.delayedGuideHotspotBaseDuplicates =
                source.delayedGuideHotspotBaseDuplicates;
            target.delayedGuideHotspotReleasedDuplicates =
                source.delayedGuideHotspotReleasedDuplicates;
            target.visibleInstant = source.visibleInstant;
            target.tileInstances = source.tileInstances;
            target.coveredPixelCount = source.coveredPixelCount;
            target.eccStochasticEventBinCounters = source.eccStochasticEventBinCounters;
        }

        bool gaussianSplatSplitShStorageAvailable(const resource::GpuResourcePool& pool)
        {
            const auto& storage = pool.gaussianStorage;
            return storage.shL1Buffer && storage.shL2Buffer && storage.shL3Buffer &&
                   !storage.cpuShL1.empty() && !storage.cpuShL2.empty() && !storage.cpuShL3.empty();
        }

        GaussianSplatShStorageLayout effectiveGaussianSplatShStorageLayout(
            const GaussianSplatRenderSettings& settings,
            const resource::GpuResourcePool&   pool)
        {
            if (settings.shStorageLayout == GaussianSplatShStorageLayout::eSplitBands &&
                gaussianSplatSplitShStorageAvailable(pool))
            {
                return GaussianSplatShStorageLayout::eSplitBands;
            }
            return GaussianSplatShStorageLayout::eMonolithic;
        }

        void bindGaussianSplatShStorageBuffers(resource::GpuSceneView&            gpuSceneView,
                                               const resource::GpuResourcePool&   pool,
                                               const GaussianSplatShStorageLayout layout)
        {
            gpuSceneView.generalGaussianSplatShStorageLayout = layout;
            gpuSceneView.generalGaussianSplatShBuffer        = pool.gaussianStorage.shBuffer;
            gpuSceneView.generalGaussianSplatShL1Buffer      =
                layout == GaussianSplatShStorageLayout::eSplitBands ? pool.gaussianStorage.shL1Buffer : nullptr;
            gpuSceneView.generalGaussianSplatShL2Buffer      =
                layout == GaussianSplatShStorageLayout::eSplitBands ? pool.gaussianStorage.shL2Buffer : nullptr;
            gpuSceneView.generalGaussianSplatShL3Buffer      =
                layout == GaussianSplatShStorageLayout::eSplitBands ? pool.gaussianStorage.shL3Buffer : nullptr;
            gpuSceneView.generalGaussianSplatShEnergyMetadataBuffer =
                pool.gaussianStorage.shEnergyMetadataBuffer;
        }

        void resetGaussianSplatIndirectBuffer(rhi::RenderDevice& rd, rhi::DrawIndirectBuffer& buffer)
        {
            std::vector<rhi::DrawIndirectCommand> indirect(1u);
            indirect[0].type          = rhi::DrawIndirectType::eNonIndexed;
            indirect[0].count         = 4u;
            indirect[0].instanceCount = 0u;
            indirect[0].first         = 0u;
            indirect[0].vertexOffset  = 0;
            indirect[0].firstInstance = 0u;
            rd.uploadDrawIndirect(buffer, indirect);
        }

        void resetGaussianSplatIndirectBuffers(rhi::RenderDevice& rd, resource::GpuSceneView& gpuSceneView)
        {
            if (gpuSceneView.generalGaussianSplatIndirectBuffer.has_value())
                resetGaussianSplatIndirectBuffer(rd, gpuSceneView.generalGaussianSplatIndirectBuffer.value());

            for (auto& buffer : gpuSceneView.generalGaussianSplatFoveatedIndirectBuffers)
            {
                if (buffer.has_value())
                    resetGaussianSplatIndirectBuffer(rd, buffer.value());
            }
        }

        void resetGaussianSplatTemporalStateIfNeeded(rhi::CommandBuffer& cb, resource::GpuSceneView& gpuSceneView)
        {
            if (!gpuSceneView.generalGaussianSplatTemporalStateBuffer ||
                !gpuSceneView.generalGaussianSplatTemporalStateNeedsReset ||
                gpuSceneView.generalGaussianSplatTemporalStateCapacity == 0u)
            {
                return;
            }

            std::vector<uint32_t> zeroStates(gpuSceneView.generalGaussianSplatTemporalStateCapacity, 0u);
            cb.update(*gpuSceneView.generalGaussianSplatTemporalStateBuffer,
                      0,
                      static_cast<uint64_t>(zeroStates.size()) * sizeof(uint32_t),
                      zeroStates.data());
            gpuSceneView.generalGaussianSplatTemporalStateNeedsReset = false;
        }

        bool syncGaussianSplatXrGaze(const IRenderBackendService& backendService,
                                      GaussianSplatRenderSettings& settings)
        {
            if (!settings.foveatedClodActive() || !settings.foveatedXrGazeEnabled ||
                settings.foveatedManualGazeControlEnabled)
            {
                return false;
            }

            const auto xrEyeViews = backendService.xrEyeViews();
            glm::vec2  gazeUvSum {0.0f};
            uint32_t   gazeUvCount {0u};
            for (const auto& eyeView : xrEyeViews)
            {
                if (!eyeView.gazeValid)
                    continue;
                gazeUvSum += eyeView.gazeUv;
                ++gazeUvCount;
            }

            if (gazeUvCount == 0u)
                return false;

            settings.foveatedGaze =
                glm::clamp(gazeUvSum / static_cast<float>(gazeUvCount), glm::vec2 {0.0f}, glm::vec2 {1.0f});
            settings.foveatedGaze.y = 1.0f - settings.foveatedGaze.y;
            return true;
        }

        bool gaussianSplatSelectionSettingsDirty(const GaussianSplatRenderSettings& current,
                                                 const GaussianSplatRenderSettings& applied)
        {
            return current.lodBudget != applied.lodBudget ||
                   current.clodLevel != applied.clodLevel ||
                   current.lodBudgetMode != applied.lodBudgetMode ||
                   current.projectedCostBudget != applied.projectedCostBudget ||
                   current.projectedCostBudgetRatio != applied.projectedCostBudgetRatio ||
                   current.projectedCostChunkSize != applied.projectedCostChunkSize ||
                   current.foveatedScoreSelectionStride != applied.foveatedScoreSelectionStride ||
                   current.foveatedCoverageBinGridX != applied.foveatedCoverageBinGridX ||
                   current.foveatedCoverageBinGridY != applied.foveatedCoverageBinGridY ||
                   current.foveatedCoverageBinQuotaRatio != applied.foveatedCoverageBinQuotaRatio ||
                   current.foveatedCoverageBinPrefixRatio != applied.foveatedCoverageBinPrefixRatio ||
                   current.projectedCostBuildMode != applied.projectedCostBuildMode ||
                   current.projectedCostSemanticDiffEnabled != applied.projectedCostSemanticDiffEnabled ||
                   current.foveatedClodEnabled != applied.foveatedClodEnabled ||
                   current.foveatedXrGazeEnabled != applied.foveatedXrGazeEnabled ||
                   current.foveatedCoverageCompensationEnabled != applied.foveatedCoverageCompensationEnabled ||
                   current.foveatedRenderMode != applied.foveatedRenderMode ||
                   current.foveatedGaze != applied.foveatedGaze ||
                   current.foveatedRingDegrees != applied.foveatedRingDegrees ||
                   current.foveatedRingLevels != applied.foveatedRingLevels ||
                   current.foveatedResolutionScales != applied.foveatedResolutionScales ||
                   current.foveatedDistribution != applied.foveatedDistribution ||
                   current.foveatedContinuousTheta0Degrees != applied.foveatedContinuousTheta0Degrees ||
                   current.foveatedContinuousAlpha != applied.foveatedContinuousAlpha ||
                   current.foveatedContinuousMinLevel != applied.foveatedContinuousMinLevel ||
                   current.foveatedCoverageGuardMode != applied.foveatedCoverageGuardMode ||
                   current.foveatedCoverageProtectionDegrees != applied.foveatedCoverageProtectionDegrees ||
                   current.foveatedCoverageGuardBudgetRatio != applied.foveatedCoverageGuardBudgetRatio ||
                   current.foveatedCoverageGuardMinLevels != applied.foveatedCoverageGuardMinLevels ||
                   current.foveatedCoverageGuardMaxAdds != applied.foveatedCoverageGuardMaxAdds ||
                   current.foveatedTemporalHysteresisEnabled != applied.foveatedTemporalHysteresisEnabled ||
                   current.foveatedBoundarySmoothingEnabled != applied.foveatedBoundarySmoothingEnabled ||
                   current.foveatedTemporalResidencyFrames != applied.foveatedTemporalResidencyFrames ||
                   current.foveatedTemporalHysteresisRatio != applied.foveatedTemporalHysteresisRatio ||
                   current.foveatedTemporalPeripheralScale != applied.foveatedTemporalPeripheralScale ||
                   current.foveatedBoundarySmoothingRatio != applied.foveatedBoundarySmoothingRatio ||
                   current.peripheralTemporalFilterEnabled != applied.peripheralTemporalFilterEnabled ||
                   current.peripheralTemporalFilterOuterDegrees != applied.peripheralTemporalFilterOuterDegrees ||
                   current.peripheralTemporalFilterLambdaScale != applied.peripheralTemporalFilterLambdaScale ||
                   current.peripheralTemporalFilterRejectionThreshold !=
                       applied.peripheralTemporalFilterRejectionThreshold ||
                   current.peripheralTemporalFilterClampRadius != applied.peripheralTemporalFilterClampRadius ||
                   current.foveatedTransitionDegrees != applied.foveatedTransitionDegrees ||
                   current.gazeAnchorDeadbandDegrees != applied.gazeAnchorDeadbandDegrees ||
                   current.gazeAnchorFadeFrames != applied.gazeAnchorFadeFrames ||
                   current.gazeAnchorMinUpdateFrames != applied.gazeAnchorMinUpdateFrames ||
                   current.gazeAnchorTransitionBudgetRatio != applied.gazeAnchorTransitionBudgetRatio ||
                   current.gazeAnchorImmediateFoveaFill != applied.gazeAnchorImmediateFoveaFill ||
                   current.gazeAnchorContributionGuard != applied.gazeAnchorContributionGuard ||
                   current.gazeAnchorDebugLogEnabled != applied.gazeAnchorDebugLogEnabled ||
                   current.eccStochasticFadeFrames != applied.eccStochasticFadeFrames ||
                   current.eccStochasticProtectOldFoveaDegrees !=
                       applied.eccStochasticProtectOldFoveaDegrees ||
                   current.eccStochasticProtectNewFoveaDegrees !=
                       applied.eccStochasticProtectNewFoveaDegrees ||
                   current.eccStochasticBoundaryBandDegrees != applied.eccStochasticBoundaryBandDegrees ||
                   current.eccStochasticMinPDelta != applied.eccStochasticMinPDelta ||
                   current.eccStochasticContributionGuard != applied.eccStochasticContributionGuard ||
                   current.eccStochasticDebugLogEnabled != applied.eccStochasticDebugLogEnabled ||
                   current.shaderAntiPopMode != applied.shaderAntiPopMode ||
                   current.shaderAntiPopHashSeed != applied.shaderAntiPopHashSeed ||
                   current.shaderAntiPopRampWidth != applied.shaderAntiPopRampWidth ||
                   current.shaderAntiPopGuardThreshold != applied.shaderAntiPopGuardThreshold ||
                   current.shaderAntiPopGuardFloor != applied.shaderAntiPopGuardFloor ||
                   current.shaderAntiPopPKeepCurve != applied.shaderAntiPopPKeepCurve ||
                   current.shaderAntiPopPrefixRatio != applied.shaderAntiPopPrefixRatio ||
                   current.shaderAntiPopNormalizeMode != applied.shaderAntiPopNormalizeMode ||
                   current.shaderAntiPopNormalizeStrength != applied.shaderAntiPopNormalizeStrength ||
                   current.shaderAntiPopNormalizeClampMin != applied.shaderAntiPopNormalizeClampMin ||
                   current.shaderAntiPopNormalizeClampMax != applied.shaderAntiPopNormalizeClampMax ||
                   current.shaderAntiPopNormalizeFactor != applied.shaderAntiPopNormalizeFactor ||
                   current.shaderAntiPopOrderFree != applied.shaderAntiPopOrderFree ||
                   current.shaderAntiPopDebugLogEnabled != applied.shaderAntiPopDebugLogEnabled ||
                   current.guideBeforeDiscardEnabled != applied.guideBeforeDiscardEnabled ||
                   current.delayedGuideLogPolarEnabled != applied.delayedGuideLogPolarEnabled ||
                   current.delayedGuideTemporalLogPolarEnabled != applied.delayedGuideTemporalLogPolarEnabled ||
                   current.delayedGuideTemporalReleasePolicy != applied.delayedGuideTemporalReleasePolicy ||
                   current.delayedGuideTemporalReleaseCapRatio != applied.delayedGuideTemporalReleaseCapRatio ||
                   current.delayedGuideTemporalRiskThreshold != applied.delayedGuideTemporalRiskThreshold ||
                   current.delayedGuideTemporalFootprintDecayScale !=
                       applied.delayedGuideTemporalFootprintDecayScale ||
                   current.delayedGuideTemporalLargeFootprintPx != applied.delayedGuideTemporalLargeFootprintPx ||
                   current.cachedSelectionOracleEnabled != applied.cachedSelectionOracleEnabled ||
                   current.cachedSelectionMembershipMode != applied.cachedSelectionMembershipMode ||
                   current.cachedSelectionOracleSeed != applied.cachedSelectionOracleSeed ||
                   current.cachedSelectionOracleStaticTargetCount !=
                       applied.cachedSelectionOracleStaticTargetCount ||
                   current.cachedSelectionOracleMatchedNullFrameIndex !=
                       applied.cachedSelectionOracleMatchedNullFrameIndex ||
                   current.cachedSelectionOracleMatchedNullTargetAddCount !=
                       applied.cachedSelectionOracleMatchedNullTargetAddCount ||
                   current.cachedSelectionOracleMatchedNullTargetRemoveCount !=
                       applied.cachedSelectionOracleMatchedNullTargetRemoveCount ||
                   current.cachedSelectionOracleMatchedNullTargetSelectedCount !=
                       applied.cachedSelectionOracleMatchedNullTargetSelectedCount ||
                   current.cachedSelectionOracleMatchedNullTargetSymmetricDiff !=
                       applied.cachedSelectionOracleMatchedNullTargetSymmetricDiff ||
                   current.cachedSelectionOracleForcedSourceIdsEnabled !=
                       applied.cachedSelectionOracleForcedSourceIdsEnabled ||
                   current.cachedSelectionOracleForcedSourceIds !=
                       applied.cachedSelectionOracleForcedSourceIds;
        }

        bool gaussianSplatSelectionSettingsDirtyIgnoringGaze(const GaussianSplatRenderSettings& current,
                                                             const GaussianSplatRenderSettings& applied)
        {
            auto currentWithoutGaze = current;
            auto appliedWithoutGaze = applied;
            currentWithoutGaze.foveatedGaze = glm::vec2 {0.5f, 0.5f};
            appliedWithoutGaze.foveatedGaze = currentWithoutGaze.foveatedGaze;
            return gaussianSplatSelectionSettingsDirty(currentWithoutGaze, appliedWithoutGaze);
        }

        bool gaussianSplatShLodSettingsDirty(const GaussianSplatRenderSettings& current,
                                             const GaussianSplatRenderSettings& applied)
        {
            return current.foveatedShLodEnabled != applied.foveatedShLodEnabled ||
                   current.foveatedShSmoothSuppressionEnabled !=
                       applied.foveatedShSmoothSuppressionEnabled ||
                   current.foveatedShLodDegrees != applied.foveatedShLodDegrees ||
                   current.foveatedShLodGuardMode != applied.foveatedShLodGuardMode ||
                   current.foveatedShLodGuardThresholdMid != applied.foveatedShLodGuardThresholdMid ||
                   current.foveatedShLodGuardThresholdHigh != applied.foveatedShLodGuardThresholdHigh ||
                   current.shPopLogEnabled != applied.shPopLogEnabled ||
                   current.shDegreeHysteresisEnabled != applied.shDegreeHysteresisEnabled ||
                   current.shDegreeDowngradeDelay != applied.shDegreeDowngradeDelay ||
                   current.shDegreeGuardBandDegrees != applied.shDegreeGuardBandDegrees ||
                   current.shStorageLayout != applied.shStorageLayout;
        }

        void applyGaussianShSmoothSuppressionStats(const GaussianSplatRenderSettings& settings,
                                                   GaussianSplatFrameStats&           stats)
        {
            const float fovea = std::max(settings.foveatedRingDegrees.x, 0.0f);
            const float mid = std::max(settings.foveatedRingDegrees.y, fovea + 1.0f);
            const float transition = std::max(settings.foveatedTransitionDegrees, 1.0f);
            const float half = fovea + (mid - fovea) * 0.5f;
            stats.foveatedShSmoothSuppressionEnabled =
                settings.foveatedShSmoothSuppressionActive();
            stats.shSmoothL3StartDegrees = fovea;
            stats.shSmoothL3EndDegrees = std::max(half, fovea + 1.0f);
            stats.shSmoothL2StartDegrees = half;
            stats.shSmoothL2EndDegrees = mid;
            stats.shSmoothL1StartDegrees = mid;
            stats.shSmoothL1EndDegrees = mid + transition;
        }

        uint32_t gaussianSplatBudgetFromLevel(const uint32_t totalSplatCount, const float level)
        {
            if (totalSplatCount == 0u)
                return 0u;
            const float clamped = std::clamp(level, 0.0f, 1.0f);
            return std::min(totalSplatCount,
                            std::max(1u, static_cast<uint32_t>(
                                             std::ceil(static_cast<float>(totalSplatCount) * clamped))));
        }

        uint32_t gaussianShAcCoeffCount(const uint32_t degree)
        {
            switch (std::min(degree, 3u))
            {
                case 0u:
                    return 0u;
                case 1u:
                    return 3u;
                case 2u:
                    return 8u;
                default:
                    return 15u;
            }
        }

        void resetGaussianShPopFrameStats(const GaussianSplatRenderSettings& settings,
                                          GaussianSplatFrameStats&           stats)
        {
            stats.shPopLogEnabled              = settings.shPopLogEnabled;
            stats.shDegreeHysteresisEnabled    = settings.shDegreeHysteresisEnabled;
            stats.shDegreeDowngradeDelay       = settings.shDegreeDowngradeDelay;
            stats.shDegreeGuardBandDegrees     = std::max(settings.shDegreeGuardBandDegrees, 0.0f);
            stats.shDegreeChangedCount         = 0u;
            stats.shDegreeChangedRatio         = 0.0;
            stats.shPopEnergyProxy             = 0.0;
            stats.shPopEnergyFovea             = 0.0;
            stats.shPopEnergyMid               = 0.0;
            stats.shPopEnergyPeriphery         = 0.0;
            stats.shPopGuardRaiseCount         = 0u;
            stats.shPopDelayedDowngradeCount   = 0u;
        }

        double gaussianShEnergyAfterDegree(const GaussianSplatFrameStats& stats, const uint32_t degree)
        {
            const uint32_t clampedDegree = std::min(degree, 3u);
            if (stats.shEnergyMetadataAvailable)
            {
                if (clampedDegree == 0u)
                    return std::max(stats.shEnergyP95After0, 0.0);
                if (clampedDegree == 1u)
                    return std::max(stats.shEnergyP95After1, 0.0);
                if (clampedDegree == 2u)
                    return std::max(stats.shEnergyP95After2, 0.0);
                return 0.0;
            }

            const double fullReads = static_cast<double>(gaussianShAcCoeffCount(3u));
            const double keptReads = static_cast<double>(gaussianShAcCoeffCount(clampedDegree));
            return fullReads > 0.0 ? std::max(fullReads - keptReads, 0.0) / fullReads : 0.0;
        }

        double gaussianAffectedShEnergy(const GaussianSplatFrameStats& stats,
                                        const uint32_t                 oldDegree,
                                        const uint32_t                 newDegree)
        {
            const uint32_t lowDegree  = std::min(std::min(oldDegree, newDegree), 3u);
            const uint32_t highDegree = std::min(std::max(oldDegree, newDegree), 3u);
            if (lowDegree == highDegree)
                return 0.0;

            return std::max(gaussianShEnergyAfterDegree(stats, lowDegree) -
                                gaussianShEnergyAfterDegree(stats, highDegree),
                            0.0);
        }

        double gaussianBoundaryAffectedShEnergy(const GaussianSplatFrameStats& stats,
                                                const glm::uvec3               shDegrees)
        {
            double total = 0.0;
            uint32_t count = 0u;
            if (shDegrees.x != shDegrees.y)
            {
                total += gaussianAffectedShEnergy(stats, shDegrees.x, shDegrees.y);
                ++count;
            }
            if (shDegrees.y != shDegrees.z)
            {
                total += gaussianAffectedShEnergy(stats, shDegrees.y, shDegrees.z);
                ++count;
            }
            return count > 0u ? total / static_cast<double>(count) : 0.0;
        }

        double gaussianWeightedAverageShDegree(const glm::uvec3                    shDegrees,
                                               const std::array<uint32_t, 3>&      ringCounts)
        {
            const uint64_t selected = static_cast<uint64_t>(ringCounts[0]) +
                                      static_cast<uint64_t>(ringCounts[1]) +
                                      static_cast<uint64_t>(ringCounts[2]);
            if (selected == 0u)
                return 0.0;
            const double weighted =
                static_cast<double>(ringCounts[0]) * static_cast<double>(std::min(shDegrees.x, 3u)) +
                static_cast<double>(ringCounts[1]) * static_cast<double>(std::min(shDegrees.y, 3u)) +
                static_cast<double>(ringCounts[2]) * static_cast<double>(std::min(shDegrees.z, 3u));
            return weighted / static_cast<double>(selected);
        }

        void storeGaussianShPopAggregateState(GaussianSplatShPopAggregateState& state,
                                              const GaussianSplatRenderSettings& settings,
                                              const glm::uvec3                   shDegrees,
                                              const std::array<uint32_t, 3>&      ringCounts,
                                              const uint32_t                     selected,
                                              const uint32_t                     guardRaisedCount)
        {
            state.valid            = true;
            state.gaze             = settings.foveatedGaze;
            state.shDegrees        = shDegrees;
            state.ringCounts       = ringCounts;
            state.selectedCount    = selected;
            state.guardRaisedCount = guardRaisedCount;
        }

        void estimateGaussianShPopAggregate(const GaussianSplatRenderSettings& settings,
                                            GaussianSplatFrameStats&           stats,
                                            GaussianSplatShPopAggregateState&  state,
                                            const uint32_t                     selected,
                                            const uint32_t                     centerCount,
                                            const uint32_t                     midCount,
                                            const uint32_t                     outerCount,
                                            const glm::uvec3                   shDegrees)
        {
            resetGaussianShPopFrameStats(settings, stats);
            if (!settings.shPopLogEnabled || !settings.foveatedShLodActive() || selected == 0u)
            {
                state = {};
                return;
            }

            const std::array<uint32_t, 3> ringCounts {centerCount, midCount, outerCount};
            stats.shPopGuardRaiseCount = stats.shGuardRaisedCount;

            if (!state.valid)
            {
                state.downgradeCandidateFrames = 0u;
                storeGaussianShPopAggregateState(state,
                                                 settings,
                                                 shDegrees,
                                                 ringCounts,
                                                 selected,
                                                 stats.shGuardRaisedCount);
                return;
            }

            uint32_t profileChangedCount = 0u;
            double   profileChangedEnergy = 0.0;
            const std::array<uint32_t, 3> currentDegrees {shDegrees.x, shDegrees.y, shDegrees.z};
            const std::array<uint32_t, 3> previousDegrees {
                state.shDegrees.x,
                state.shDegrees.y,
                state.shDegrees.z,
            };
            for (size_t ring = 0u; ring < ringCounts.size(); ++ring)
            {
                const uint32_t commonCount = std::min(ringCounts[ring], state.ringCounts[ring]);
                if (commonCount == 0u || currentDegrees[ring] == previousDegrees[ring])
                    continue;
                const double affectedEnergy =
                    gaussianAffectedShEnergy(stats, previousDegrees[ring], currentDegrees[ring]);
                profileChangedCount += commonCount;
                profileChangedEnergy += static_cast<double>(commonCount) * affectedEnergy;
            }

            const uint32_t selectionDelta =
                selected > state.selectedCount ? selected - state.selectedCount : state.selectedCount - selected;
            const double averageDegreeScale =
                std::clamp(gaussianWeightedAverageShDegree(shDegrees, ringCounts) / 3.0, 0.0, 1.0);
            const uint32_t selectionChangedCount = static_cast<uint32_t>(
                std::ceil(static_cast<double>(selectionDelta) * averageDegreeScale));

            const double gazeDelta = static_cast<double>(glm::length(settings.foveatedGaze - state.gaze));
            const double boundaryContrast =
                (static_cast<double>(std::abs(static_cast<int>(shDegrees.x) - static_cast<int>(shDegrees.y))) +
                 static_cast<double>(std::abs(static_cast<int>(shDegrees.y) - static_cast<int>(shDegrees.z)))) /
                6.0;
            const double guardBandScale =
                1.0 / (1.0 + static_cast<double>(std::max(settings.shDegreeGuardBandDegrees, 0.0f)) /
                                  std::max(static_cast<double>(settings.foveatedRingDegrees.y), 1.0));
            const uint32_t gazeChangedCount = static_cast<uint32_t>(
                std::ceil(static_cast<double>(selected) *
                          std::clamp(gazeDelta * 4.0, 0.0, 1.0) *
                          std::clamp(boundaryContrast, 0.0, 1.0) *
                          guardBandScale));

            const uint32_t guardChangedCount =
                stats.shGuardRaisedCount > state.guardRaisedCount ?
                    stats.shGuardRaisedCount - state.guardRaisedCount :
                    state.guardRaisedCount - stats.shGuardRaisedCount;

            uint32_t changedCount =
                std::min(selected,
                         profileChangedCount + selectionChangedCount + gazeChangedCount + guardChangedCount);
            double affectedEnergy =
                profileChangedCount > 0u ?
                    profileChangedEnergy / static_cast<double>(profileChangedCount) :
                    gaussianBoundaryAffectedShEnergy(stats, shDegrees);

            const double previousAverageDegree =
                gaussianWeightedAverageShDegree(state.shDegrees, state.ringCounts);
            const double targetAverageDegree = gaussianWeightedAverageShDegree(shDegrees, ringCounts);
            bool acceptTargetProfile = true;
            if (settings.shDegreeHysteresisEnabled && targetAverageDegree + 1e-6 < previousAverageDegree)
            {
                ++state.downgradeCandidateFrames;
                if (state.downgradeCandidateFrames <= settings.shDegreeDowngradeDelay)
                {
                    stats.shPopDelayedDowngradeCount = changedCount;
                    changedCount = 0u;
                    affectedEnergy = 0.0;
                    acceptTargetProfile = false;
                }
            }
            else
            {
                state.downgradeCandidateFrames = 0u;
            }

            stats.shDegreeChangedCount = changedCount;
            stats.shDegreeChangedRatio =
                selected > 0u ? static_cast<double>(changedCount) / static_cast<double>(selected) : 0.0;

            const double areaPerSplat =
                selected > 0u && stats.sumProjectedAreaPx > 0.0 ?
                    std::max(stats.sumProjectedAreaPx / static_cast<double>(selected), 1.0) :
                    1.0;
            const double changed = static_cast<double>(changedCount);
            const double selectedD = static_cast<double>(selected);
            const double foveaFraction = selectedD > 0.0 ? static_cast<double>(centerCount) / selectedD : 0.0;
            const double midFraction = selectedD > 0.0 ? static_cast<double>(midCount) / selectedD : 0.0;
            const double peripheryFraction = selectedD > 0.0 ? static_cast<double>(outerCount) / selectedD : 0.0;

            stats.shPopEnergyFovea = changed * foveaFraction * areaPerSplat * affectedEnergy;
            stats.shPopEnergyMid = changed * midFraction * areaPerSplat * affectedEnergy * 0.45;
            stats.shPopEnergyPeriphery = changed * peripheryFraction * areaPerSplat * affectedEnergy * 0.15;
            stats.shPopEnergyProxy =
                stats.shPopEnergyFovea + stats.shPopEnergyMid + stats.shPopEnergyPeriphery;

            if (acceptTargetProfile)
            {
                storeGaussianShPopAggregateState(state,
                                                 settings,
                                                 shDegrees,
                                                 ringCounts,
                                                 selected,
                                                 stats.shGuardRaisedCount);
            }
        }

        void addGaussianShBandReadsForDegree(const uint64_t count,
                                             const uint32_t degree,
                                             uint64_t&      l1Reads,
                                             uint64_t&      l2Reads,
                                             uint64_t&      l3Reads)
        {
            const uint32_t clampedDegree = std::min(degree, 3u);
            if (clampedDegree >= 1u)
                l1Reads += count * resource::GpuGaussianSplat::s_PackedShL1Coeffs;
            if (clampedDegree >= 2u)
                l2Reads += count * resource::GpuGaussianSplat::s_PackedShL2Coeffs;
            if (clampedDegree >= 3u)
                l3Reads += count * resource::GpuGaussianSplat::s_PackedShL3Coeffs;
        }

        void addRecoveredGaussianShReadsToBands(GaussianSplatFrameStats& stats,
                                                const uint64_t           selected,
                                                uint64_t                 recoveredReads)
        {
            const uint64_t l1Capacity = selected * resource::GpuGaussianSplat::s_PackedShL1Coeffs;
            const uint64_t l2Capacity = selected * resource::GpuGaussianSplat::s_PackedShL2Coeffs;
            const uint64_t l3Capacity = selected * resource::GpuGaussianSplat::s_PackedShL3Coeffs;

            auto recoverIntoBand = [&](uint64_t& reads, const uint64_t capacity) {
                if (recoveredReads == 0u || reads >= capacity)
                    return;
                const uint64_t added = std::min(recoveredReads, capacity - reads);
                reads += added;
                recoveredReads -= added;
            };

            recoverIntoBand(stats.shBandL1ReadsEst, l1Capacity);
            recoverIntoBand(stats.shBandL2ReadsEst, l2Capacity);
            recoverIntoBand(stats.shBandL3ReadsEst, l3Capacity);
        }

        void finalizeGaussianShBandStats(GaussianSplatFrameStats& stats, const uint64_t degree3Reads)
        {
            const uint64_t totalBandReads =
                stats.shBandL1ReadsEst + stats.shBandL2ReadsEst + stats.shBandL3ReadsEst;
            stats.shBandBytesEst =
                totalBandReads * static_cast<uint64_t>(sizeof(glm::uvec2));
            stats.shStorageMetadataBytesEst = 0u;

            const uint64_t degree3Bytes =
                degree3Reads * static_cast<uint64_t>(sizeof(glm::uvec2));
            if (degree3Bytes == 0u)
            {
                stats.shBandBytesReductionVsMonolithic = 0.0;
                return;
            }

            stats.shBandBytesReductionVsMonolithic =
                1.0 - static_cast<double>(stats.shBandBytesEst) /
                          static_cast<double>(degree3Bytes);
            stats.shBandBytesReductionVsMonolithic =
                std::clamp(stats.shBandBytesReductionVsMonolithic, 0.0, 1.0);
        }

        void estimateGaussianShLodRuntimeReads(const GaussianSplatRenderSettings& settings,
                                               GaussianSplatFrameStats&           stats,
                                               GaussianSplatShPopAggregateState&  shPopState)
        {
            stats.shGuardMode =
                settings.foveatedShLodActive() ? settings.foveatedShLodGuardMode :
                                                  GaussianSplatShLodGuardMode::eOff;
            stats.shGuardThresholdMid = std::max(settings.foveatedShLodGuardThresholdMid, 0.0f);
            stats.shGuardThresholdHigh =
                std::max(settings.foveatedShLodGuardThresholdHigh, stats.shGuardThresholdMid);
            stats.shGuardRaisedCount            = 0u;
            stats.shGuardRaisedRatio            = 0.0;
            stats.shGuardRecoveredAcReads       = 0u;
            stats.shGuardSavedAcReadsAfterGuard = 0u;
            stats.shBandL1ReadsEst              = 0u;
            stats.shBandL2ReadsEst              = 0u;
            stats.shBandL3ReadsEst              = 0u;
            stats.shBandBytesEst                = 0u;
            stats.shBandBytesReductionVsMonolithic = 0.0;
            stats.shStorageMetadataBytesEst     = 0u;
            resetGaussianShPopFrameStats(settings, stats);

            const uint32_t selected =
                stats.lodSelectedRawSplats > 0u ? stats.lodSelectedRawSplats : stats.preparedSplats;
            const uint64_t degree3Reads = static_cast<uint64_t>(selected) * gaussianShAcCoeffCount(3u);
            stats.estimatedShAcCoeffReads = degree3Reads;
            stats.estimatedShAcReadReductionVsDegree3 = 0.0;
            addGaussianShBandReadsForDegree(selected,
                                            3u,
                                            stats.shBandL1ReadsEst,
                                            stats.shBandL2ReadsEst,
                                            stats.shBandL3ReadsEst);
            if (!settings.foveatedShLodActive() || selected == 0u || degree3Reads == 0u)
            {
                shPopState = {};
                finalizeGaussianShBandStats(stats, degree3Reads);
                return;
            }

            const uint32_t totalForBudget = std::max(stats.totalSplats, selected);
            const uint32_t outerPrefix =
                std::min(selected, gaussianSplatBudgetFromLevel(totalForBudget, settings.foveatedRingLevels.z));
            const uint32_t midPrefix =
                std::min(selected, gaussianSplatBudgetFromLevel(totalForBudget, settings.foveatedRingLevels.y));
            const uint32_t outerCount = outerPrefix;
            const uint32_t midCount = midPrefix > outerPrefix ? midPrefix - outerPrefix : 0u;
            const uint32_t centerCount = selected > midPrefix ? selected - midPrefix : 0u;
            const auto shDegrees = glm::clamp(settings.foveatedShLodDegrees, glm::uvec3 {0u}, glm::uvec3 {3u});

            stats.shBandL1ReadsEst = 0u;
            stats.shBandL2ReadsEst = 0u;
            stats.shBandL3ReadsEst = 0u;
            addGaussianShBandReadsForDegree(centerCount,
                                            shDegrees.x,
                                            stats.shBandL1ReadsEst,
                                            stats.shBandL2ReadsEst,
                                            stats.shBandL3ReadsEst);
            addGaussianShBandReadsForDegree(midCount,
                                            shDegrees.y,
                                            stats.shBandL1ReadsEst,
                                            stats.shBandL2ReadsEst,
                                            stats.shBandL3ReadsEst);
            addGaussianShBandReadsForDegree(outerCount,
                                            shDegrees.z,
                                            stats.shBandL1ReadsEst,
                                            stats.shBandL2ReadsEst,
                                            stats.shBandL3ReadsEst);

            stats.estimatedShAcCoeffReads =
                static_cast<uint64_t>(centerCount) * gaussianShAcCoeffCount(shDegrees.x) +
                static_cast<uint64_t>(midCount) * gaussianShAcCoeffCount(shDegrees.y) +
                static_cast<uint64_t>(outerCount) * gaussianShAcCoeffCount(shDegrees.z);

            const uint64_t gazeOnlyReads = stats.estimatedShAcCoeffReads;
            stats.shGuardSavedAcReadsAfterGuard =
                degree3Reads > gazeOnlyReads ? degree3Reads - gazeOnlyReads : 0u;
            if (stats.shGuardMode != GaussianSplatShLodGuardMode::eOff &&
                stats.shGuardSavedAcReadsAfterGuard > 0u)
            {
                const bool usesEnergy =
                    stats.shGuardMode == GaussianSplatShLodGuardMode::eEnergy ||
                    stats.shGuardMode == GaussianSplatShLodGuardMode::eEnergyProjectedCost;
                const bool usesProjectedCost =
                    stats.shGuardMode == GaussianSplatShLodGuardMode::eProjectedCost ||
                    stats.shGuardMode == GaussianSplatShLodGuardMode::eEnergyProjectedCost;

                if (!usesEnergy || stats.shEnergyMetadataAvailable)
                {
                    const auto energySignalForDegree = [&](const uint32_t degree) {
                        if (degree == 0u)
                            return stats.shEnergyP95After0;
                        if (degree == 1u)
                            return stats.shEnergyP95After1;
                        if (degree == 2u)
                            return stats.shEnergyP95After2;
                        return 0.0;
                    };
                    const auto ringSignal = [&](const uint32_t count, const uint32_t baseDegree) {
                        if (count == 0u || baseDegree >= 3u)
                            return 0.0;
                        const double energyTerm =
                            usesEnergy ? energySignalForDegree(baseDegree) : 1.0;
                        const double projectedTerm =
                            usesProjectedCost ?
                                (stats.sumProjectedAreaPx > 0.0 ?
                                     std::max(stats.sumProjectedAreaPx / static_cast<double>(selected), 1.0) :
                                     1.0) :
                                1.0;
                        return energyTerm * projectedTerm;
                    };

                    const double signal =
                        std::max({ringSignal(centerCount, shDegrees.x),
                                  ringSignal(midCount, shDegrees.y),
                                  ringSignal(outerCount, shDegrees.z)});
                    if (signal > static_cast<double>(stats.shGuardThresholdMid))
                    {
                        const double thresholdSpan =
                            std::max(static_cast<double>(
                                         stats.shGuardThresholdHigh - stats.shGuardThresholdMid),
                                     1e-8);
                        const double severity =
                            signal > static_cast<double>(stats.shGuardThresholdHigh) ?
                                1.0 :
                                std::clamp((signal - static_cast<double>(stats.shGuardThresholdMid)) /
                                               thresholdSpan,
                                           0.0,
                                           1.0);
                        const double maxRaisedFraction = usesProjectedCost && !usesEnergy ? 0.25 : 0.05;
                        stats.shGuardRaisedCount =
                            std::min(selected,
                                     static_cast<uint32_t>(
                                         std::ceil(static_cast<double>(selected) *
                                                   maxRaisedFraction * severity)));
                        stats.shGuardRaisedRatio =
                            selected > 0u ?
                                static_cast<double>(stats.shGuardRaisedCount) /
                                    static_cast<double>(selected) :
                                0.0;

                        const double averageRecoveredReads =
                            std::max(static_cast<double>(degree3Reads - gazeOnlyReads) /
                                         static_cast<double>(selected),
                                     0.0);
                        stats.shGuardRecoveredAcReads =
                            std::min(stats.shGuardSavedAcReadsAfterGuard,
                                     static_cast<uint64_t>(std::llround(
                                         averageRecoveredReads *
                                         static_cast<double>(stats.shGuardRaisedCount))));
                        stats.estimatedShAcCoeffReads += stats.shGuardRecoveredAcReads;
                        addRecoveredGaussianShReadsToBands(stats,
                                                            selected,
                                                            stats.shGuardRecoveredAcReads);
                        stats.shGuardSavedAcReadsAfterGuard =
                            degree3Reads > stats.estimatedShAcCoeffReads ?
                                degree3Reads - stats.estimatedShAcCoeffReads :
                                0u;
                    }
                }
            }

            stats.estimatedShAcReadReductionVsDegree3 =
                1.0 - static_cast<double>(stats.estimatedShAcCoeffReads) /
                          static_cast<double>(degree3Reads);
            stats.estimatedShAcReadReductionVsDegree3 =
                std::clamp(stats.estimatedShAcReadReductionVsDegree3, 0.0, 1.0);
            estimateGaussianShPopAggregate(settings,
                                           stats,
                                           shPopState,
                                           selected,
                                           centerCount,
                                           midCount,
                                           outerCount,
                                           shDegrees);
            finalizeGaussianShBandStats(stats, degree3Reads);
        }

        void summarizeGaussianShEnergyMetadata(const RenderWorld&                     world,
                                               const resource::GpuResourcePool&       pool,
                                               GaussianSplatFrameStats&               stats)
        {
            uint64_t totalPointsWithAssets = 0u;
            uint64_t pointsWithMetadata    = 0u;
            double   weightedMeanAfter0    = 0.0;
            double   weightedMeanAfter1    = 0.0;
            double   weightedMeanAfter2    = 0.0;
            double   p95After0             = 0.0;
            double   p95After1             = 0.0;
            double   p95After2             = 0.0;

            for (const auto& splatInst : world.gaussianSplats)
            {
                if (splatInst.splatIndex >= pool.gaussianSplats.size())
                    continue;

                const auto& gpuSplat = pool.gaussianSplats[splatInst.splatIndex];
                if (gpuSplat.pointCount == 0u)
                    continue;

                totalPointsWithAssets += gpuSplat.pointCount;
                const uint64_t metadataEnd =
                    static_cast<uint64_t>(gpuSplat.pointOffset) + static_cast<uint64_t>(gpuSplat.pointCount);
                const bool metadataAvailable =
                    gpuSplat.shEnergyMetadataAvailable && pool.gaussianStorage.shEnergyMetadataBuffer &&
                    metadataEnd <= static_cast<uint64_t>(pool.gaussianStorage.cpuShEnergyMetadata.size());
                if (!metadataAvailable)
                    continue;

                pointsWithMetadata += gpuSplat.pointCount;
                weightedMeanAfter0 += static_cast<double>(gpuSplat.shEnergyMean.x) * gpuSplat.pointCount;
                weightedMeanAfter1 += static_cast<double>(gpuSplat.shEnergyMean.y) * gpuSplat.pointCount;
                weightedMeanAfter2 += static_cast<double>(gpuSplat.shEnergyMean.z) * gpuSplat.pointCount;
                p95After0 = std::max(p95After0, static_cast<double>(gpuSplat.shEnergyP95.x));
                p95After1 = std::max(p95After1, static_cast<double>(gpuSplat.shEnergyP95.y));
                p95After2 = std::max(p95After2, static_cast<double>(gpuSplat.shEnergyP95.z));
            }

            stats.shEnergyMetadataAvailable =
                totalPointsWithAssets > 0u && pointsWithMetadata == totalPointsWithAssets;
            if (pointsWithMetadata == 0u)
                return;

            stats.shEnergyMeanAfter0 = weightedMeanAfter0 / static_cast<double>(pointsWithMetadata);
            stats.shEnergyMeanAfter1 = weightedMeanAfter1 / static_cast<double>(pointsWithMetadata);
            stats.shEnergyMeanAfter2 = weightedMeanAfter2 / static_cast<double>(pointsWithMetadata);
            stats.shEnergyP95After0  = p95After0;
            stats.shEnergyP95After1  = p95After1;
            stats.shEnergyP95After2  = p95After2;
        }

        float smooth01(const float edge0, const float edge1, const float value)
        {
            if (edge1 <= edge0)
                return value >= edge1 ? 1.0f : 0.0f;

            const float t = std::clamp((value - edge0) / (edge1 - edge0), 0.0f, 1.0f);
            return t * t * (3.0f - 2.0f * t);
        }

        float lerpFloat(const float a, const float b, const float t)
        {
            return a + (b - a) * std::clamp(t, 0.0f, 1.0f);
        }

        void applyProgressiveCenterOutProfile(GaussianSplatRenderSettings& settings, const float budgetLevel)
        {
            constexpr float kMinFoveaBudget = 0.18f;
            constexpr float kOuterFloor     = 0.12f;
            constexpr float kBudgetQuantum  = 0.02f;

            const float budget = std::clamp(
                std::round(std::clamp(budgetLevel, kMinFoveaBudget, 1.0f) / kBudgetQuantum) * kBudgetQuantum,
                kMinFoveaBudget,
                1.0f);
            const float t      = smooth01(kMinFoveaBudget, 1.0f, budget);

            // The fovea level is the active prefix/scan budget. Mid/outer are
            // derived from it so a minimum full-frame base is always present,
            // and extra budget expands from gaze center before improving outer.
            const float outerGrowth = smooth01(0.58f, 1.0f, t);
            const float outer       = std::min(budget, kOuterFloor + 0.24f * outerGrowth);
            const float midMix      = smooth01(0.16f, 0.80f, t);
            const float mid         = std::clamp(outer + (budget - outer) * (0.38f + 0.50f * midMix), outer, budget);

            const float foveaRadius = lerpFloat(8.0f, 32.0f, smooth01(0.08f, 0.72f, t));
            const float midRadius   = std::max(foveaRadius + 10.0f, lerpFloat(24.0f, 76.0f, smooth01(0.35f, 1.0f, t)));

            settings.foveatedRingLevels  = glm::vec3 {budget, mid, outer};
            settings.foveatedRingDegrees = glm::vec2 {foveaRadius, midRadius};
            settings.foveatedTransitionDegrees = std::max(settings.foveatedTransitionDegrees, 8.0f);
            settings.foveatedCoverageProtectionDegrees =
                std::clamp(settings.foveatedCoverageProtectionDegrees, 0.0f, 3.0f);
        }

        double percentileValue(std::vector<double> values, const double percentile)
        {
            values.erase(std::remove_if(values.begin(), values.end(), [](const double value) {
                             return value <= 0.0 || !std::isfinite(value);
                         }),
                         values.end());
            if (values.empty())
                return -1.0;

            std::sort(values.begin(), values.end());
            const double rank = std::clamp(percentile, 0.0, 100.0) * 0.01 * static_cast<double>(values.size() - 1u);
            const auto   lo   = static_cast<size_t>(std::floor(rank));
            const auto   hi   = static_cast<size_t>(std::ceil(rank));
            const double t    = rank - static_cast<double>(lo);
            return values[lo] + (values[hi] - values[lo]) * t;
        }

        void resetGaussianSplatP95Controller(GaussianSplatFoveatedP95ControllerState& state,
                                             const GaussianSplatFoveatedAdaptationMode mode)
        {
            state.gpuWindow.clear();
            state.p95Ema        = -1.0;
            state.growCounter   = 0u;
            state.shrinkCounter = 0u;
            state.dynamicFrameMsEma   = -1.0;
            state.dynamicGrowCounter   = 0u;
            state.dynamicShrinkCounter = 0u;
            state.mode                 = mode;
        }

        bool updateProgressiveP95Controller(GaussianSplatRenderSettings&                settings,
                                            GaussianSplatFoveatedP95ControllerState&    state,
                                            const double                                gpuFrameMs)
        {
            constexpr size_t kWindowFrames = 30u;
            constexpr float  kAlpha        = 0.85f;
            constexpr float  kMarginLowMs  = 0.30f;
            constexpr float  kMarginHighMs = 0.20f;
            constexpr uint32_t kGrowDelay  = 8u;
            constexpr uint32_t kShrinkDelay = 2u;
            constexpr uint32_t kStableGrowFrames = 5u;
            constexpr float kBudgetMin     = 0.35f;
            constexpr float kBudgetMax     = 0.75f;
            constexpr float kCoverageGuardMaxDegrees = 3.0f;

            const float budgetMin =
                [&]() {
                    if (settings.foveatedCoverageGuardMode !=
                        GaussianSplatFoveatedCoverageGuardMode::eRiskTriggered)
                    {
                        return kBudgetMin;
                    }

                    const float guardRatio =
                        std::clamp(settings.foveatedCoverageGuardBudgetRatio, 0.0f, 0.25f);
                    const float centerFloor =
                        std::clamp(settings.foveatedCoverageGuardMinLevels.x, 0.0f, 1.0f);
                    const float centerReachableBudget =
                        centerFloor > 0.0f ? centerFloor / (1.0f + guardRatio) : kBudgetMin;
                    return std::clamp(std::max(kBudgetMin, centerReachableBudget), kBudgetMin, kBudgetMax);
                }();

            auto currentBudgetForDiagnostics = [&]() {
                return settings.projectedCostBudgetEnabled() ?
                           std::clamp(settings.projectedCostBudgetRatio > 0.0f ?
                                          settings.projectedCostBudgetRatio :
                                          effectiveGaussianAutomaticClodLevel(settings),
                                      budgetMin,
                                      kBudgetMax) :
                           std::clamp(settings.foveatedRingLevels.x, budgetMin, kBudgetMax);
            };
            settings.schedulerShrinkEvent = false;
            settings.schedulerGrowEvent   = false;
            settings.schedulerBudgetBefore = currentBudgetForDiagnostics();
            settings.schedulerBudgetAfter  = settings.schedulerBudgetBefore;
            settings.schedulerP95Ema       = state.p95Ema;
            settings.schedulerP95Window    = -1.0;
            settings.schedulerTargetMs     = std::max(settings.foveatedTargetFrameMs, 0.1f);
            settings.schedulerMarginLow    = kMarginLowMs;
            settings.schedulerMarginHigh   = kMarginHighMs;
            if (!settings.foveatedClodActive())
            {
                state.dynamicFrameMsEma    = -1.0;
                state.dynamicGrowCounter   = 0u;
                state.dynamicShrinkCounter = 0u;
                return false;
            }
            if (gpuFrameMs <= 0.0 || !std::isfinite(gpuFrameMs))
                return false;

            if (state.mode != settings.foveatedAdaptationMode)
            {
                resetGaussianSplatP95Controller(state, settings.foveatedAdaptationMode);
                settings.schedulerP95Ema = state.p95Ema;
            }

            state.gpuWindow.push_back(gpuFrameMs);
            if (state.gpuWindow.size() > kWindowFrames)
                state.gpuWindow.erase(state.gpuWindow.begin());
            if (state.gpuWindow.size() < std::min<size_t>(10u, kWindowFrames))
                return false;

            const double p95Window = percentileValue(state.gpuWindow, 95.0);
            if (p95Window <= 0.0)
                return false;
            settings.schedulerP95Window = p95Window;

            state.p95Ema = state.p95Ema <= 0.0 ? p95Window :
                                                     static_cast<double>(kAlpha) * state.p95Ema +
                                                         (1.0 - static_cast<double>(kAlpha)) * p95Window;
            settings.schedulerP95Ema = state.p95Ema;

            const float targetMs   = std::max(settings.foveatedTargetFrameMs, 0.1f);
            const float growRate   = std::clamp(settings.foveatedBudgetAdjustRate, 0.005f, 0.01f);
            const float shrinkRate = std::clamp(settings.foveatedBudgetAdjustRate * 3.0f, 0.02f, 0.05f);
            bool        changed    = false;

            bool stableUnderTarget = state.gpuWindow.size() >= kStableGrowFrames;
            if (stableUnderTarget)
            {
                const size_t start = state.gpuWindow.size() - kStableGrowFrames;
                for (size_t i = start; i < state.gpuWindow.size(); ++i)
                {
                    if (state.gpuWindow[i] >= static_cast<double>(targetMs))
                    {
                        stableUnderTarget = false;
                        break;
                    }
                }
            }

            if (state.p95Ema > static_cast<double>(targetMs + kMarginHighMs))
            {
                ++state.shrinkCounter;
                state.growCounter = 0u;
            }
            else if (state.p95Ema < static_cast<double>(targetMs - kMarginLowMs) && stableUnderTarget)
            {
                ++state.growCounter;
                state.shrinkCounter = 0u;
            }
            else
            {
                state.growCounter   = 0u;
                state.shrinkCounter = 0u;
            }

            const bool controlsProjectedCostBudget = settings.projectedCostBudgetEnabled();
            const float currentBudget =
                controlsProjectedCostBudget ?
                    std::clamp(settings.projectedCostBudgetRatio > 0.0f ?
                                   settings.projectedCostBudgetRatio :
                                   effectiveGaussianAutomaticClodLevel(settings),
                                budgetMin,
                                kBudgetMax) :
                    std::clamp(settings.foveatedRingLevels.x, budgetMin, kBudgetMax);
            float nextBudget = currentBudget;
            settings.schedulerBudgetBefore = currentBudget;
            if (state.shrinkCounter >= kShrinkDelay)
            {
                state.shrinkCounter = 0u;
                state.growCounter = 0u;
                nextBudget = std::clamp(nextBudget * (1.0f - shrinkRate), budgetMin, kBudgetMax);
                changed = true;
                settings.schedulerShrinkEvent = true;
            }
            else if (state.growCounter >= kGrowDelay)
            {
                nextBudget = std::clamp(nextBudget * (1.0f + growRate), budgetMin, kBudgetMax);
                state.growCounter = 0u;
                changed = true;
                settings.schedulerGrowEvent = true;
            }

            if (!changed && std::abs(nextBudget - currentBudget) < 1e-4f)
                return false;
            settings.schedulerBudgetAfter = nextBudget;

            if (controlsProjectedCostBudget)
            {
                // B_t is the latency controller state. The spatial foveated
                // profile follows B_t, but projected-cost stop remains the hard
                // runtime budget.
                settings.projectedCostBudget      = 0u;
                settings.projectedCostBudgetRatio = nextBudget;
            }
            applyProgressiveCenterOutProfile(settings, nextBudget);
            settings.foveatedRingLevels.x = std::clamp(settings.foveatedRingLevels.x, budgetMin, kBudgetMax);
            settings.foveatedRingLevels.y =
                std::clamp(settings.foveatedRingLevels.y, settings.foveatedRingLevels.z, settings.foveatedRingLevels.x);
            settings.foveatedRingLevels.z = std::clamp(settings.foveatedRingLevels.z, 0.08f, settings.foveatedRingLevels.y);
            if (controlsProjectedCostBudget)
                settings.projectedCostBudgetRatio = std::clamp(settings.projectedCostBudgetRatio, budgetMin, kBudgetMax);
            settings.foveatedCoverageProtectionDegrees =
                std::min(settings.foveatedCoverageProtectionDegrees, kCoverageGuardMaxDegrees);
            return true;
        }

        bool updateStabilityAwareBudgetController(GaussianSplatRenderSettings&             settings,
                                                  GaussianSplatFoveatedP95ControllerState& state,
                                                  const double                             gpuFrameMs)
        {
            constexpr size_t   kWindowFrames = 30u;
            constexpr float    kAlpha        = 0.90f;
            constexpr float    kMarginHighMs = 0.18f;
            constexpr float    kMarginLowMs  = 0.35f;
            constexpr uint32_t kShrinkDelay  = 4u;
            constexpr uint32_t kGrowDelay    = 14u;
            constexpr uint32_t kStableGrowFrames = 8u;
            constexpr float    kFoveaBudgetMin = 0.12f;
            constexpr float    kMidBudgetMin   = 0.07f;
            constexpr float    kOuterBudgetMin = 0.04f;

            auto currentBudgetForDiagnostics = [&settings]() {
                return settings.projectedCostBudgetEnabled() ?
                           std::clamp(settings.projectedCostBudgetRatio > 0.0f ?
                                          settings.projectedCostBudgetRatio :
                                          effectiveGaussianAutomaticClodLevel(settings),
                                      kFoveaBudgetMin,
                                      1.0f) :
                           std::clamp(settings.foveatedRingLevels.x, kFoveaBudgetMin, 1.0f);
            };

            settings.schedulerShrinkEvent  = false;
            settings.schedulerGrowEvent    = false;
            settings.schedulerBudgetBefore = currentBudgetForDiagnostics();
            settings.schedulerBudgetAfter  = settings.schedulerBudgetBefore;
            settings.schedulerP95Ema       = state.p95Ema;
            settings.schedulerP95Window    = -1.0;
            settings.schedulerTargetMs     = std::max(settings.foveatedTargetFrameMs, 0.1f);
            settings.schedulerMarginLow    = kMarginLowMs;
            settings.schedulerMarginHigh   = kMarginHighMs;

            if (!settings.foveatedClodActive())
                return false;
            if (gpuFrameMs <= 0.0 || !std::isfinite(gpuFrameMs))
                return false;

            state.gpuWindow.push_back(gpuFrameMs);
            if (state.gpuWindow.size() > kWindowFrames)
                state.gpuWindow.erase(state.gpuWindow.begin());
            if (state.gpuWindow.size() < std::min<size_t>(12u, kWindowFrames))
                return false;

            const double p95Window = percentileValue(state.gpuWindow, 95.0);
            if (p95Window <= 0.0)
                return false;
            settings.schedulerP95Window = p95Window;

            state.p95Ema = state.p95Ema <= 0.0 ? p95Window :
                                                     static_cast<double>(kAlpha) * state.p95Ema +
                                                         (1.0 - static_cast<double>(kAlpha)) * p95Window;
            settings.schedulerP95Ema = state.p95Ema;

            const float targetMs = std::max(settings.foveatedTargetFrameMs, 0.1f);
            bool stableUnderTarget = state.gpuWindow.size() >= kStableGrowFrames;
            if (stableUnderTarget)
            {
                const size_t start = state.gpuWindow.size() - kStableGrowFrames;
                for (size_t i = start; i < state.gpuWindow.size(); ++i)
                {
                    if (state.gpuWindow[i] >= static_cast<double>(targetMs - 0.05f))
                    {
                        stableUnderTarget = false;
                        break;
                    }
                }
            }

            if (state.p95Ema > static_cast<double>(targetMs + kMarginHighMs))
            {
                ++state.shrinkCounter;
                state.growCounter = 0u;
            }
            else if (state.p95Ema < static_cast<double>(targetMs - kMarginLowMs) && stableUnderTarget)
            {
                ++state.growCounter;
                state.shrinkCounter = 0u;
            }
            else
            {
                state.growCounter   = 0u;
                state.shrinkCounter = 0u;
            }

            float budgetStep = 0.0f;
            const float shrinkStep = std::clamp(settings.foveatedBudgetAdjustRate * 0.16f, 0.0025f, 0.0080f);
            const float growStep   = std::clamp(settings.foveatedBudgetAdjustRate * 0.08f, 0.0015f, 0.0040f);
            if (state.shrinkCounter >= kShrinkDelay)
            {
                const float severity =
                    std::clamp(static_cast<float>(
                                   (state.p95Ema - static_cast<double>(targetMs + kMarginHighMs)) /
                                   static_cast<double>(targetMs)),
                               0.0f,
                               1.0f);
                budgetStep = -shrinkStep * (0.70f + 0.30f * severity);
                state.shrinkCounter = 0u;
                state.growCounter   = 0u;
                settings.schedulerShrinkEvent = true;
            }
            else if (state.growCounter >= kGrowDelay)
            {
                budgetStep = growStep;
                state.growCounter = 0u;
                settings.schedulerGrowEvent = true;
            }

            if (std::abs(budgetStep) < 1e-5f)
                return false;

            const float currentBudget = currentBudgetForDiagnostics();
            const float nextBudget    = std::clamp(currentBudget + budgetStep, kFoveaBudgetMin, 1.0f);
            settings.schedulerBudgetBefore = currentBudget;
            settings.schedulerBudgetAfter  = nextBudget;
            if (std::abs(nextBudget - currentBudget) < 1e-5f)
            {
                settings.schedulerShrinkEvent = false;
                settings.schedulerGrowEvent   = false;
                return false;
            }

            const float scale = currentBudget > 0.0f ? nextBudget / currentBudget : 1.0f;
            settings.foveatedRingLevels.x = nextBudget;
            settings.foveatedRingLevels.y =
                std::clamp(settings.foveatedRingLevels.y * scale, kMidBudgetMin, settings.foveatedRingLevels.x);
            settings.foveatedRingLevels.z =
                std::clamp(settings.foveatedRingLevels.z * scale, kOuterBudgetMin, settings.foveatedRingLevels.y);
            settings.foveatedRingDegrees.x = std::clamp(settings.foveatedRingDegrees.x, 6.0f, 36.0f);
            settings.foveatedRingDegrees.y =
                std::clamp(settings.foveatedRingDegrees.y, settings.foveatedRingDegrees.x + 8.0f, 78.0f);
            settings.foveatedTransitionDegrees = std::max(settings.foveatedTransitionDegrees, 8.0f);
            settings.foveatedDistribution      = GaussianSplatFoveatedDistribution::eGaussian;

            if (settings.projectedCostBudgetEnabled())
            {
                settings.projectedCostBudget      = 0u;
                settings.projectedCostBudgetRatio = nextBudget;
            }

            return true;
        }

        bool gaussianSplatFoveatedFrameTimeFeedback(const GaussianSplatRenderSettings& settings,
                                                    const double                       gpuFrameMs,
                                                    float&                             error,
                                                    float&                             step)
        {
            if (!settings.foveatedClodActive() || gpuFrameMs <= 0.0)
                return false;

            const float targetMs = std::max(settings.foveatedTargetFrameMs, 0.1f);
            const float maxStep = std::clamp(settings.foveatedBudgetAdjustRate, 0.001f, 0.25f);
            error = static_cast<float>((targetMs - gpuFrameMs) / targetMs);
            if (std::abs(error) < 0.03f)
                return false;

            step = std::clamp(error * 0.5f, -maxStep, maxStep);
            return true;
        }

        bool gaussianSplatFoveatedDynamicRangeFeedback(const GaussianSplatRenderSettings&          settings,
                                                       GaussianSplatFoveatedP95ControllerState&    state,
                                                       const double                                gpuFrameMs,
                                                       float&                                      error,
                                                       float&                                      foveaDeltaDegrees,
                                                       float&                                      midDeltaDegrees)
        {
            if (!settings.foveatedClodActive() || gpuFrameMs <= 0.0 || !std::isfinite(gpuFrameMs))
                return false;

            constexpr double   kAlpha       = 0.90;
            constexpr float    kDeadband    = 0.06f;
            constexpr uint32_t kGrowDelay   = 6u;
            constexpr uint32_t kShrinkDelay = 3u;

            const float targetMs = std::max(settings.foveatedTargetFrameMs, 0.1f);
            state.dynamicFrameMsEma =
                state.dynamicFrameMsEma <= 0.0 ?
                    gpuFrameMs :
                    kAlpha * state.dynamicFrameMsEma + (1.0 - kAlpha) * gpuFrameMs;

            error = static_cast<float>((targetMs - state.dynamicFrameMsEma) / targetMs);
            if (std::abs(error) < kDeadband)
            {
                state.dynamicGrowCounter   = 0u;
                state.dynamicShrinkCounter = 0u;
                return false;
            }

            if (error > 0.0f)
            {
                ++state.dynamicGrowCounter;
                state.dynamicShrinkCounter = 0u;
                if (state.dynamicGrowCounter < kGrowDelay)
                    return false;
                state.dynamicGrowCounter = 0u;
            }
            else
            {
                ++state.dynamicShrinkCounter;
                state.dynamicGrowCounter = 0u;
                if (state.dynamicShrinkCounter < kShrinkDelay)
                    return false;
                state.dynamicShrinkCounter = 0u;
            }

            const float maxDeltaDegrees = std::clamp(settings.foveatedBudgetAdjustRate * 12.0f, 0.15f, 0.75f);
            foveaDeltaDegrees = std::clamp(error * 6.0f, -maxDeltaDegrees, maxDeltaDegrees);
            midDeltaDegrees   = foveaDeltaDegrees * 1.5f;
            return std::abs(foveaDeltaDegrees) > 1e-4f;
        }

        void updateGaussianSplatFoveatedAdaptation(GaussianSplatRenderSettings&             settings,
                                                   GaussianSplatFoveatedP95ControllerState& p95State,
                                                   const double                             gpuFrameMs)
        {
            if (p95State.mode != settings.foveatedAdaptationMode)
                resetGaussianSplatP95Controller(p95State, settings.foveatedAdaptationMode);

            if (settings.foveatedAdaptationMode == GaussianSplatFoveatedAdaptationMode::eProgressiveCenterOut)
            {
                updateProgressiveP95Controller(settings, p95State, gpuFrameMs);
                return;
            }

            if (settings.foveatedAdaptationMode == GaussianSplatFoveatedAdaptationMode::eStabilityAwareBudget)
            {
                updateStabilityAwareBudgetController(settings, p95State, gpuFrameMs);
                return;
            }

            auto currentBudgetForDiagnostics = [&settings]() {
                return settings.projectedCostBudgetEnabled() ?
                           std::clamp(settings.projectedCostBudgetRatio > 0.0f ?
                                          settings.projectedCostBudgetRatio :
                                          effectiveGaussianAutomaticClodLevel(settings),
                                      0.0f,
                                      1.0f) :
                           std::clamp(settings.foveatedRingLevels.x, 0.0f, 1.0f);
            };
            settings.schedulerShrinkEvent = false;
            settings.schedulerGrowEvent   = false;
            settings.schedulerBudgetBefore = currentBudgetForDiagnostics();
            settings.schedulerBudgetAfter  = settings.schedulerBudgetBefore;
            settings.schedulerP95Ema       = p95State.p95Ema;
            settings.schedulerP95Window    = -1.0;
            settings.schedulerTargetMs     = std::max(settings.foveatedTargetFrameMs, 0.1f);
            settings.schedulerMarginLow    = -1.0f;
            settings.schedulerMarginHigh   = 0.03f * settings.schedulerTargetMs;

            auto clampRings = [&settings]() {
                settings.foveatedRingLevels.z = std::clamp(settings.foveatedRingLevels.z, 0.01f, 1.0f);
                settings.foveatedRingLevels.y =
                    std::clamp(settings.foveatedRingLevels.y, settings.foveatedRingLevels.z, 1.0f);
                settings.foveatedRingLevels.x =
                    std::clamp(settings.foveatedRingLevels.x, settings.foveatedRingLevels.y, 1.0f);
                settings.foveatedRingDegrees.x = std::clamp(settings.foveatedRingDegrees.x, 1.0f, 45.0f);
                settings.foveatedRingDegrees.y =
                    std::clamp(settings.foveatedRingDegrees.y, settings.foveatedRingDegrees.x, 90.0f);
            };

            if (settings.foveatedAdaptationMode == GaussianSplatFoveatedAdaptationMode::eDynamicRange)
            {
                float error             = 0.0f;
                float foveaDeltaDegrees = 0.0f;
                float midDeltaDegrees   = 0.0f;
                if (!gaussianSplatFoveatedDynamicRangeFeedback(
                        settings, p95State, gpuFrameMs, error, foveaDeltaDegrees, midDeltaDegrees))
                    return;

                settings.foveatedRingDegrees.x =
                    std::clamp(settings.foveatedRingDegrees.x + foveaDeltaDegrees, 1.0f, 45.0f);
                settings.foveatedRingDegrees.y = std::clamp(settings.foveatedRingDegrees.y + midDeltaDegrees,
                                                             settings.foveatedRingDegrees.x,
                                                             90.0f);
                clampRings();
                return;
            }

            float error = 0.0f;
            float step  = 0.0f;
            if (!gaussianSplatFoveatedFrameTimeFeedback(settings, gpuFrameMs, error, step))
                return;

            auto adjustBudget = [](float value, const float step, const float floorValue) {
                return std::clamp(value + step * std::max(value, 0.1f), floorValue, 1.0f);
            };
            switch (settings.foveatedAdaptationMode)
            {
                case GaussianSplatFoveatedAdaptationMode::eFixed:
                    break;
                case GaussianSplatFoveatedAdaptationMode::eDynamicBudget:
                    settings.foveatedRingLevels.z = adjustBudget(settings.foveatedRingLevels.z, step, 0.01f);
                    settings.foveatedRingLevels.y =
                        adjustBudget(settings.foveatedRingLevels.y, step, settings.foveatedRingLevels.z);
                    if (error < -0.35f)
                        settings.foveatedRingLevels.x =
                            adjustBudget(settings.foveatedRingLevels.x, step, settings.foveatedRingLevels.y);
                    else
                        settings.foveatedRingLevels.x =
                            std::max(settings.foveatedRingLevels.x, settings.foveatedRingLevels.y);
                    clampRings();
                    settings.schedulerBudgetAfter = currentBudgetForDiagnostics();
                    settings.schedulerShrinkEvent =
                        settings.schedulerBudgetAfter < settings.schedulerBudgetBefore - 1e-5f;
                    settings.schedulerGrowEvent =
                        settings.schedulerBudgetAfter > settings.schedulerBudgetBefore + 1e-5f;
                    break;
                case GaussianSplatFoveatedAdaptationMode::eStabilityAwareBudget:
                    break;
                case GaussianSplatFoveatedAdaptationMode::eDynamicRange:
                    break;
                case GaussianSplatFoveatedAdaptationMode::eProgressiveCenterOut:
                    break;
                case GaussianSplatFoveatedAdaptationMode::eProgressiveGreedy:
                {
                    const float nextBudget = std::clamp(settings.foveatedRingLevels.x + step, 0.12f, 1.0f);
                    applyProgressiveCenterOutProfile(settings, nextBudget);
                    clampRings();
                    settings.schedulerBudgetAfter = currentBudgetForDiagnostics();
                    settings.schedulerShrinkEvent =
                        settings.schedulerBudgetAfter < settings.schedulerBudgetBefore - 1e-5f;
                    settings.schedulerGrowEvent =
                        settings.schedulerBudgetAfter > settings.schedulerBudgetBefore + 1e-5f;
                    break;
                }
            }
        }

        void rebuildGaussianSplatOrderedClodPrefixSources(resource::GpuSceneView&                       gpuSceneView,
                                                          const std::vector<RenderGaussianSplatInstance>& gaussianSplats,
                                                          const resource::GpuResourcePool&                 pool,
                                                          RuntimeProfiler&                                 profiler)
        {
            gpuSceneView.generalGaussianSplatSelectedSources.clear();

            struct OrderedSource
            {
                uint32_t rankNumerator {0};
                uint32_t pointCount {1};
                uint32_t drawIndex {0};
                uint32_t rank {0};
                resource::GpuGeneralGaussianSplatSelectedSource selection {};
            };

            RuntimeProfiler::Scope scope {profiler, "GaussianCLOD::BuildPrefix"};
            const bool             singleDraw = gpuSceneView.generalGaussianSplatDraws.size() <= 1u;
            std::vector<OrderedSource> orderedSources;
            if (!singleDraw)
            {
                uint32_t reserveCount = 0u;
                for (const auto& splatInst : gaussianSplats)
                {
                    if (splatInst.splatIndex < pool.gaussianSplats.size())
                        reserveCount += pool.gaussianSplats[splatInst.splatIndex].pointCount;
                }
                orderedSources.reserve(reserveCount);
            }

            uint32_t drawIndex = 0u;
            for (const auto& splatInst : gaussianSplats)
            {
                if (splatInst.splatIndex >= pool.gaussianSplats.size())
                    continue;
                if (drawIndex >= gpuSceneView.generalGaussianSplatDraws.size())
                    break;

                const auto& gpuSplat = pool.gaussianSplats[splatInst.splatIndex];
                if (gpuSplat.pointCount == 0u)
                    continue;

                const auto&  drawRecord      = gpuSceneView.generalGaussianSplatDraws[drawIndex];
                const uint32_t sourceOffset  = drawRecord.pointOffset;
                const uint32_t rankCount     = gpuSplat.pointCount;

                for (uint32_t rank = 0u; rank < rankCount; ++rank)
                {
                    const uint32_t localPoint = rank;

                    resource::GpuGeneralGaussianSplatSelectedSource selection {};
                    selection.sourceIndex  = sourceOffset + localPoint;
                    selection.drawIndex    = drawIndex;
                    selection.packedWeight = std::bit_cast<uint32_t>(1.0f);
                    selection.flags        = 0u;

                    if (singleDraw)
                    {
                        gpuSceneView.pushGeneralGaussianSplatSelectedSource(selection);
                    }
                    else
                    {
                        orderedSources.push_back(OrderedSource {
                            .rankNumerator = rank + 1u,
                            .pointCount    = gpuSplat.pointCount,
                            .drawIndex     = drawIndex,
                            .rank          = rank,
                            .selection     = selection,
                        });
                    }
                }

                ++drawIndex;
            }

            if (!singleDraw)
            {
                std::stable_sort(orderedSources.begin(), orderedSources.end(), [](const auto& a, const auto& b) {
                    const uint64_t lhs =
                        static_cast<uint64_t>(a.rankNumerator) * static_cast<uint64_t>(b.pointCount);
                    const uint64_t rhs =
                        static_cast<uint64_t>(b.rankNumerator) * static_cast<uint64_t>(a.pointCount);
                    if (lhs != rhs)
                        return lhs < rhs;
                    if (a.drawIndex != b.drawIndex)
                        return a.drawIndex < b.drawIndex;
                    return a.rank < b.rank;
                });

                gpuSceneView.generalGaussianSplatSelectedSources.reserve(orderedSources.size());
                for (const auto& source : orderedSources)
                    gpuSceneView.pushGeneralGaussianSplatSelectedSource(source.selection);
            }
        }

        void rebuildGaussianSplatSelectedSources(resource::GpuSceneView&                       gpuSceneView,
                                                 const std::vector<RenderGaussianSplatInstance>& gaussianSplats,
                                                 const resource::GpuResourcePool&                 pool,
                                                 GaussianSplatFrameStats&                         stats,
                                                 RuntimeProfiler&                                 profiler)
        {
            gpuSceneView.generalGaussianSplatSelectedSources.clear();

            // Baseline still builds a selected-source table so the preprocess
            // shader can share one path with Ordered CLOD.
            RuntimeProfiler::Scope scope {profiler, "GaussianSplat::BuildRawSelection"};
            uint32_t               drawIndex = 0u;
            for (const auto& splatInst : gaussianSplats)
            {
                if (splatInst.splatIndex >= pool.gaussianSplats.size())
                    continue;
                if (drawIndex >= gpuSceneView.generalGaussianSplatDraws.size())
                    break;

                const auto& gpuSplat = pool.gaussianSplats[splatInst.splatIndex];
                if (gpuSplat.pointCount == 0u)
                    continue;

                const auto&  drawRecord      = gpuSceneView.generalGaussianSplatDraws[drawIndex];
                const uint32_t sourceOffset  = drawRecord.pointOffset;
                for (uint32_t localPoint = 0u; localPoint < gpuSplat.pointCount; ++localPoint)
                {
                    resource::GpuGeneralGaussianSplatSelectedSource selection {};
                    selection.sourceIndex  = sourceOffset + localPoint;
                    selection.drawIndex    = drawIndex;
                    selection.packedWeight = std::bit_cast<uint32_t>(1.0f);
                    selection.flags        = 0u;
                    gpuSceneView.pushGeneralGaussianSplatSelectedSource(selection);
                    ++stats.lodSelectedRawSplats;
                }

                ++drawIndex;
            }
        }
    } // namespace

    void RenderWorldCooker::cook(World& world, IAssetService& assets, RenderWorld& out)
    {
        out.clear();

        auto& reg = world.registry();

        auto view = reg.view<IDComponent, TransformComponent, MeshComponent>();
        out.instances.reserve(view.size_hint());
        for (auto e : view)
        {
            const auto& id   = view.get<IDComponent>(e);
            const auto& tr   = view.get<TransformComponent>(e);
            const auto& mesh = view.get<MeshComponent>(e);
            if (auto* status = reg.try_get<EntityStatusComponent>(e); status && (!status->active || !status->visible))
                continue;

            auto h = assets.loadMeshSync(mesh.mesh);
            if (!h.ready())
                continue;

            RenderInstance inst {};
            inst.entity      = id.uuid;
            inst.meshIndex   = h.gpuIndex();
            inst.worldMatrix = tr.worldMatrix;
            out.instances.push_back(inst);
        }

        auto splatView = reg.view<IDComponent, TransformComponent, GaussianSplatComponent>();
        out.gaussianSplats.reserve(splatView.size_hint());
        for (auto e : splatView)
        {
            const auto& id    = splatView.get<IDComponent>(e);
            const auto& tr    = splatView.get<TransformComponent>(e);
            const auto& splat = splatView.get<GaussianSplatComponent>(e);
            if (auto* status = reg.try_get<EntityStatusComponent>(e); status && (!status->active || !status->visible))
                continue;

            auto h = assets.loadGaussianSplatSync(splat.gaussianSplat);
            if (!h.ready())
                continue;

            RenderGaussianSplatInstance inst {};
            inst.entity      = id.uuid;
            inst.splatIndex  = h.gpuIndex();
            inst.worldMatrix = tr.worldMatrix;
            out.gaussianSplats.push_back(inst);
        }
    }

    bool RenderSystem::onInit()
    {
        VULTRA_CORE_INFO("[RenderSystem] Initializing...");
        m_GaussianProjectedCostCacheSalt =
            g_RenderSystemProjectedCostCacheSaltCounter.fetch_add(1u, std::memory_order_relaxed);
        m_GaussianProjectedCostCacheEpoch = m_GaussianProjectedCostCacheSalt;

#if defined(__ANDROID__)
        // Android paths currently rely on the CPU-driven renderer.
        m_EnableGpuDrivenMeshletPipeline = false;
#endif

        VULTRA_CORE_TRACE("[RenderSystem] Getting render backend service");
        auto& backendService = ctx().services.require<IRenderBackendService>();

        VULTRA_CORE_TRACE("[RenderSystem] Getting window service");
        auto& windowService = ctx().services.require<IWindowService>();

        VULTRA_CORE_TRACE("[RenderSystem] Creating transient resources");
        m_TransientResources = createScope<framegraph::TransientResources>(backendService.renderDevice());

        VULTRA_CORE_TRACE("[RenderSystem] Initializing renderers");
        for (auto& [key, renderer] : m_Renderers)
        {
            VULTRA_CORE_TRACE("[RenderSystem]     Initializing renderer: {}", key);
            Services services = ctx().services;
            renderer->setupServices(services);
            renderer->init();
        }

        VULTRA_CORE_TRACE("[RenderSystem] Initializing samplers");
        m_Samplers["default"] = backendService.renderDevice().getSampler(rhi::SamplerInfo {});
        m_Samplers["linear"]  = backendService.renderDevice().getSampler(
            rhi::SamplerInfo {.magFilter = rhi::TexelFilter::eLinear, .minFilter = rhi::TexelFilter::eLinear});
        m_Samplers["nearest"] = backendService.renderDevice().getSampler(
            rhi::SamplerInfo {.magFilter = rhi::TexelFilter::eNearest, .minFilter = rhi::TexelFilter::eNearest});

        VULTRA_CORE_TRACE("[RenderSystem] Providing IRenderService");
        ctx().services.provide<IRenderService>(this);

        VULTRA_CORE_INFO("[RenderSystem] Initialized!");

        return true;
    }

    void RenderSystem::onShutdown()
    {
        VULTRA_CORE_INFO("[RenderSystem] Shutting down");

        auto& backendService = ctx().services.require<IRenderBackendService>();
        backendService.renderDevice().waitIdle();

        m_GpuSceneViewBack.clear();
        m_GpuSceneViewFront.clear();
        m_GpuSceneDatabaseBack.clear();
        m_GpuSceneDatabaseFront.clear();
        m_GpuSceneDirtyTracker.reset();
        gaussianProjectedCostPersistentSampleCache() = {};

        for (auto& [key, renderer] : m_Renderers)
            renderer = nullptr;
        m_Renderers.clear();

        m_TransientResources.reset();

        m_FrameResources.clear();
    }

    void RenderSystem::registerRenderer(Ref<Renderer> renderer)
    {
        if (!renderer)
            return;
        if (m_Renderers.contains(std::string(renderer->name())))
            return;
        m_Renderers[std::string(renderer->name())] = renderer;
    }

    Ref<Renderer> RenderSystem::resolveRenderer(const RenderCamera& cam) const
    {
        if (auto it = m_Renderers.find(cam.rendererKey); it != m_Renderers.end())
            return it->second;

        if (auto it2 = m_Renderers.find(m_DefaultRendererKey); it2 != m_Renderers.end())
            return it2->second;

        return nullptr;
    }

    void RenderSystem::onResize(uint32_t width, uint32_t height)
    {
        for (auto& [key, renderer] : m_Renderers)
        {
            if (renderer)
                renderer->onResize(width, height);
        }
    }

    void RenderSystem::renderFrame()
    {
        const auto renderFrameCpuStart = std::chrono::steady_clock::now();

        auto& backendService     = ctx().services.require<IRenderBackendService>();
        auto& worldService       = ctx().services.require<IWorldService>();
        auto& camService         = ctx().services.require<ICameraService>();
        auto& gpuResourceService = ctx().services.require<IGpuResourceService>();
        auto& assetService       = ctx().services.require<IAssetService>();
        auto& shaderService      = ctx().services.require<IShaderService>();
        auto& window             = ctx().services.require<IWindowService>().window();

        // Optional ImGui service for rendering ImGui on top of frame.
        auto* imguiService = ctx().services.tryGet<IImGuiService>();

        // Optional frame debugger service for GPU capture.
        auto* frameDebuggerService = ctx().services.tryGet<IFrameDebuggerService>();

        auto& rd = backendService.renderDevice();

        m_RuntimeProfiler.beginFrame(m_FrameCounter);
        m_RuntimeProfiler.setVsyncEnabled(ctx().config.render.vSyncConfig != rhi::VerticalSync::eDisabled);
        rhi::CommandBuffer::resetFrameStats();

        // Begin frame first so downstream systems can consume per-frame backend state (e.g. XR eye views).
        if (!backendService.beginFrame())
        {
            m_SkipRender = true;
            m_RuntimeProfiler.endFrame();
            return;
        }

        auto& cb = backendService.commandBuffer();
        RuntimeProfiler::Scope scopeRenderFrame {m_RuntimeProfiler, "RenderSystem::renderFrame"};
        rd.beginFrameGpuQuery(cb);

        // Default target for cameras without explicit RT
        auto& defaultTarget = backendService.backbuffer();

        if (frameDebuggerService)
        {
            frameDebuggerService->captureStart();
        }

        World&     world = worldService.world();
        const auto cams  = camService.cameras();

        // Asset upload/update stage (main thread)
        {
            RuntimeProfiler::Scope scope {m_RuntimeProfiler, "AssetService::update"};
            assetService.update(m_FrameCounter);
        }
        // Cook render instances
        RenderWorldCooker cooker {};
        {
            RuntimeProfiler::Scope scope {m_RuntimeProfiler, "RenderWorldCooker::cook"};
            cooker.cook(world, assetService, m_RenderWorldBack);
        }
        m_RenderWorldBack.frameIndex = m_FrameCounter;

        const bool gaussianOrderedClodMode = m_GaussianSplatSettings.orderedClodEnabled();

        const uint64_t resourceRevision = gpuResourceService.contentRevision();
        const auto&    pool             = gpuResourceService.pool();

        uint32_t maxGeneralGaussianSplatPoints = 0;
        uint32_t maxGeneralGaussianSplatSourceCount = 0;
        for (const auto& splatInst : m_RenderWorldBack.gaussianSplats)
        {
            if (splatInst.splatIndex >= pool.gaussianSplats.size())
                continue;
            const auto& gpuSplat = pool.gaussianSplats[splatInst.splatIndex];
            maxGeneralGaussianSplatPoints += gpuSplat.pointCount;
            maxGeneralGaussianSplatSourceCount += gpuSplat.pointCount;
        }

        const bool gaussianXrGazeActive = syncGaussianSplatXrGaze(backendService, m_GaussianSplatSettings);

        GaussianSplatFrameStats gaussianStats {};
        gaussianStats.frameIndex                       = m_FrameCounter;
        gaussianStats.baselineMode                     = m_GaussianSplatSettings.baselineMode;
        gaussianStats.foveatedRenderMode               = m_GaussianSplatSettings.foveatedRenderMode;
        gaussianStats.lodBudgetMode                    = m_GaussianSplatSettings.lodBudgetMode;
        gaussianStats.lodBudgetEnabled                 = m_GaussianSplatSettings.lodBudgetEnabled();
        gaussianStats.foveatedClodEnabled              = m_GaussianSplatSettings.foveatedClodActive();
        gaussianStats.foveatedLayeredCompositeEnabled = m_GaussianSplatSettings.foveatedLayeredCompositeActive();
        gaussianStats.foveatedXrGazeEnabled            = m_GaussianSplatSettings.foveatedXrGazeEnabled;
        gaussianStats.foveatedXrGazeActive             = gaussianXrGazeActive;
        gaussianStats.foveatedCoverageCompensationEnabled =
            m_GaussianSplatSettings.foveatedCoverageCompensationEnabled;
        gaussianStats.foveatedAdaptationMode           = m_GaussianSplatSettings.foveatedAdaptationMode;
        gaussianStats.foveatedDistribution             = m_GaussianSplatSettings.foveatedDistribution;
        gaussianStats.lodBudget                        = m_GaussianSplatSettings.lodBudget;
        gaussianStats.costBudget                       = m_GaussianSplatSettings.projectedCostBudget;
        gaussianStats.projectedCostBudgetRatio         = m_GaussianSplatSettings.projectedCostBudgetRatio;
        gaussianStats.projectedCostChunkSize           = m_GaussianSplatSettings.projectedCostChunkSize;
        gaussianStats.foveatedCoverageBinGridX         = m_GaussianSplatSettings.foveatedCoverageBinGridX;
        gaussianStats.foveatedCoverageBinGridY         = m_GaussianSplatSettings.foveatedCoverageBinGridY;
        gaussianStats.foveatedCoverageBinQuotaRatio    = m_GaussianSplatSettings.foveatedCoverageBinQuotaRatio;
        gaussianStats.foveatedCoverageBinPrefixRatio   = m_GaussianSplatSettings.foveatedCoverageBinPrefixRatio;
        gaussianStats.projectedCostBuildMode           = m_GaussianSplatSettings.projectedCostBuildMode;
        gaussianStats.projectedCostSemanticDiffEnabled =
            m_GaussianSplatSettings.projectedCostSemanticDiffEnabled;
        gaussianStats.cleanTimingMode                  = m_GaussianSplatSettings.cleanTimingMode;
        gaussianStats.foveatedRingLevels               = m_GaussianSplatSettings.foveatedRingLevels;
        gaussianStats.foveatedResolutionScales         = m_GaussianSplatSettings.foveatedResolutionScales;
        gaussianStats.foveatedGaze                     = m_GaussianSplatSettings.foveatedGaze;
        gaussianStats.foveatedRingDegrees              = m_GaussianSplatSettings.foveatedRingDegrees;
        gaussianStats.foveatedTargetFrameMs            = m_GaussianSplatSettings.foveatedTargetFrameMs;
        gaussianStats.foveatedContinuousTheta0Degrees  =
            m_GaussianSplatSettings.foveatedContinuousTheta0Degrees;
        gaussianStats.foveatedContinuousAlpha          = m_GaussianSplatSettings.foveatedContinuousAlpha;
        gaussianStats.foveatedContinuousMinLevel       = m_GaussianSplatSettings.foveatedContinuousMinLevel;
        gaussianStats.schedulerShrinkEvent             = m_GaussianSplatSettings.schedulerShrinkEvent;
        gaussianStats.schedulerGrowEvent               = m_GaussianSplatSettings.schedulerGrowEvent;
        gaussianStats.schedulerBudgetBefore            = m_GaussianSplatSettings.schedulerBudgetBefore;
        gaussianStats.schedulerBudgetAfter             = m_GaussianSplatSettings.schedulerBudgetAfter;
        gaussianStats.schedulerP95Ema                  = m_GaussianSplatSettings.schedulerP95Ema;
        gaussianStats.schedulerP95Window               = m_GaussianSplatSettings.schedulerP95Window;
        gaussianStats.schedulerTargetMs                = m_GaussianSplatSettings.schedulerTargetMs;
        gaussianStats.schedulerMarginLow               = m_GaussianSplatSettings.schedulerMarginLow;
        gaussianStats.schedulerMarginHigh              = m_GaussianSplatSettings.schedulerMarginHigh;
        gaussianStats.projectedCostBudgetRatioAfterScheduler =
            m_GaussianSplatSettings.projectedCostBudgetRatio;
        gaussianStats.foveatedCoverageGuardMode =
            m_GaussianSplatSettings.foveatedCoverageGuardMode;
        gaussianStats.foveatedCoverageProtectionDegrees =
            m_GaussianSplatSettings.foveatedCoverageProtectionDegrees;
        gaussianStats.foveatedCoverageGuardBudgetRatio =
            m_GaussianSplatSettings.foveatedCoverageGuardBudgetRatio;
        gaussianStats.foveatedCoverageGuardMinLevels =
            m_GaussianSplatSettings.foveatedCoverageGuardMinLevels;
        gaussianStats.foveatedCoverageGuardMaxAdds =
            m_GaussianSplatSettings.foveatedCoverageGuardMaxAdds;
        gaussianStats.foveatedCoverageGuardMaxAddsEffective =
            m_GaussianSplatSettings.foveatedCoverageGuardMaxAdds;
        gaussianStats.foveatedTemporalHysteresisEnabled =
            m_GaussianSplatSettings.foveatedTemporalHysteresisEnabled;
        gaussianStats.foveatedBoundarySmoothingEnabled =
            m_GaussianSplatSettings.foveatedBoundarySmoothingEnabled;
        gaussianStats.foveatedTemporalResidencyFrames =
            m_GaussianSplatSettings.foveatedTemporalResidencyFrames;
        gaussianStats.foveatedTemporalHysteresisRatio =
            m_GaussianSplatSettings.foveatedTemporalHysteresisRatio;
        gaussianStats.foveatedBoundarySmoothingRatio =
            m_GaussianSplatSettings.foveatedBoundarySmoothingRatio;
        gaussianStats.foveatedTemporalPeripheralScale =
            m_GaussianSplatSettings.foveatedTemporalPeripheralScale;
        gaussianStats.peripheralTemporalFilterEnabled =
            m_GaussianSplatSettings.peripheralTemporalFilterEnabled;
        gaussianStats.peripheralTemporalFilterMidDegrees =
            std::max(m_GaussianSplatSettings.foveatedRingDegrees.y,
                     m_GaussianSplatSettings.foveatedRingDegrees.x + 1.0f);
        gaussianStats.peripheralTemporalFilterOuterDegrees =
            std::max(m_GaussianSplatSettings.peripheralTemporalFilterOuterDegrees,
                     gaussianStats.peripheralTemporalFilterMidDegrees + 1.0f);
        gaussianStats.peripheralTemporalFilterLambdaScale =
            std::clamp(m_GaussianSplatSettings.peripheralTemporalFilterLambdaScale, 0.0f, 1.0f);
        gaussianStats.peripheralTemporalFilterRejectionThreshold =
            std::max(m_GaussianSplatSettings.peripheralTemporalFilterRejectionThreshold, 1e-4f);
        gaussianStats.peripheralTemporalFilterClampRadius =
            std::max(m_GaussianSplatSettings.peripheralTemporalFilterClampRadius, 0.0f);
        gaussianStats.shaderAntiPopMode = m_GaussianSplatSettings.shaderAntiPopMode;
        gaussianStats.shaderAntiPopEnabled = m_GaussianSplatSettings.shaderAntiPopActive();
        gaussianStats.shaderAntiPopAlphaMultiplierBased =
            gaussianStats.shaderAntiPopEnabled &&
            m_GaussianSplatSettings.shaderAntiPopMode !=
                GaussianSplatShaderAntiPopMode::eCoverageStableLogpolarRelease;
        gaussianStats.shaderAntiPopAvoidsCpuSelectedSourceRebuild =
            gaussianStats.shaderAntiPopEnabled &&
            m_GaussianSplatSettings.shaderAntiPopOrderFree;
        gaussianStats.shaderAntiPopHashSeed = m_GaussianSplatSettings.shaderAntiPopHashSeed;
        gaussianStats.shaderAntiPopRampWidth =
            std::clamp(m_GaussianSplatSettings.shaderAntiPopRampWidth, 0.0f, 1.0f);
        gaussianStats.shaderAntiPopGuardThreshold =
            std::clamp(m_GaussianSplatSettings.shaderAntiPopGuardThreshold, 0.0f, 1.0f);
        gaussianStats.shaderAntiPopGuardFloor =
            std::clamp(m_GaussianSplatSettings.shaderAntiPopGuardFloor, 0.0f, 1.0f);
        gaussianStats.shaderAntiPopPKeepCurve =
            m_GaussianSplatSettings.shaderAntiPopPKeepCurve;
        gaussianStats.shaderAntiPopPrefixRatio =
            m_GaussianSplatSettings.shaderAntiPopPrefixRatio;
        gaussianStats.shaderAntiPopNormalizeMode =
            m_GaussianSplatSettings.shaderAntiPopNormalizeMode;
        gaussianStats.shaderAntiPopNormalizeStrength =
            std::clamp(m_GaussianSplatSettings.shaderAntiPopNormalizeStrength, 0.0f, 1.0f);
        gaussianStats.shaderAntiPopNormalizeClampMin =
            std::max(m_GaussianSplatSettings.shaderAntiPopNormalizeClampMin, 0.0f);
        gaussianStats.shaderAntiPopNormalizeClampMax =
            std::max(m_GaussianSplatSettings.shaderAntiPopNormalizeClampMax,
                     gaussianStats.shaderAntiPopNormalizeClampMin);
        gaussianStats.shaderAntiPopNormalizeFactor =
            std::clamp(m_GaussianSplatSettings.shaderAntiPopNormalizeFactor,
                       gaussianStats.shaderAntiPopNormalizeClampMin,
                       gaussianStats.shaderAntiPopNormalizeClampMax);
        gaussianStats.coverageStableReleasePFloorMean = 0.0f;
        gaussianStats.coverageStableReleasePFloorMax = 0.0f;
        gaussianStats.coverageStableReleaseDAlphaMean = 0.0f;
        gaussianStats.coverageStableReleaseNEffMean = 0.0f;
        gaussianStats.coverageStableReleaseTileTotalCount = 0u;
        gaussianStats.coverageStableReleaseTileNonEmptyCount = 0u;
        gaussianStats.coverageStableReleaseTileEmptyCount = 0u;
        gaussianStats.coverageStableReleaseDAlphaMeanAll = 0.0f;
        gaussianStats.coverageStableReleaseDAlphaMeanNonEmpty = 0.0f;
        gaussianStats.coverageStableReleaseDAlphaP50NonEmpty = 0.0f;
        gaussianStats.coverageStableReleaseDAlphaP90NonEmpty = 0.0f;
        gaussianStats.coverageStableReleaseDAlphaP95NonEmpty = 0.0f;
        gaussianStats.coverageStableReleaseNEffMeanNonEmpty = 0.0f;
        gaussianStats.coverageStableReleaseNEffP50NonEmpty = 0.0f;
        gaussianStats.coverageStableReleaseNEffP90NonEmpty = 0.0f;
        gaussianStats.coverageStableReleaseNEffP95NonEmpty = 0.0f;
        gaussianStats.coverageStableReleasePFloorMeanAll = 0.0f;
        gaussianStats.coverageStableReleasePFloorMeanNonEmpty = 0.0f;
        gaussianStats.coverageStableReleasePFloorP90NonEmpty = 0.0f;
        gaussianStats.coverageStableReleasePFloorP95NonEmpty = 0.0f;
        gaussianStats.coverageStableReleasePLpMean = 0.0f;
        gaussianStats.coverageStableReleasePStaticMean = 0.0f;
        gaussianStats.coverageStableReleaseFloorActiveFractionVsCandidates = 0.0f;
        gaussianStats.coverageStableReleaseFloorRaisedFractionVsEffectiveVisible = 0.0f;
        gaussianStats.coverageStableReleaseDeltaSum = 0.0f;
        gaussianStats.coverageStableReleaseDeltaMeanOverCandidates = 0.0f;
        gaussianStats.coverageStableReleaseDeltaMeanOverHeld = 0.0f;
        gaussianStats.coverageStableReleaseHeldEffectiveRatio = 0.0f;
        gaussianStats.coverageStableReleaseHeldEffectiveCount = 0u;
        gaussianStats.coverageStableReleaseHistoryResetCount = 0u;
        gaussianStats.coverageStableReleaseReentryResetCount = 0u;
        gaussianStats.coverageStableReleaseEpsilonCutoffCount = 0u;
        gaussianStats.coverageStableReleaseSaturatedFloorEnabled =
            m_GaussianSplatSettings.coverageStableReleaseSaturatedFloorEnabled;
        gaussianStats.coverageStableReleaseFloorCellsSafeSaturated = 0u;
        gaussianStats.coverageStableReleaseFloorCellsUnsafe = 0u;
        gaussianStats.coverageStableReleaseSkippedFloorUpdateCount = 0u;
        gaussianStats.coverageStableReleaseDAlphaSafeThreshold = 0.0f;
        gaussianStats.coverageStableReleaseNEffSafeThreshold = 0.0f;
        gaussianStats.coverageStableReleaseFloorSaturationRatio = 0.0f;
        gaussianStats.coverageStableReleaseCandidateAlphaMass = 0.0f;
        gaussianStats.coverageStableReleaseRetainedAlphaMass = 0.0f;
        gaussianStats.coverageTextureCurrentAlphaMass = 0.0f;
        gaussianStats.coverageTextureCurrentAlpha2Mass = 0.0f;
        gaussianStats.coverageTextureNonzeroTexelCount = 0u;
        gaussianStats.coverageTextureWidth =
            m_GaussianSplatSettings.coverageTextureFloorEnabled ?
                std::clamp(m_GaussianSplatSettings.coverageTextureFloorWidth, 1u, 1024u) :
                0u;
        gaussianStats.coverageTextureHeight =
            m_GaussianSplatSettings.coverageTextureFloorEnabled ?
                std::clamp(m_GaussianSplatSettings.coverageTextureFloorHeight, 1u, 1024u) :
                0u;
        gaussianStats.coverageTextureHistoryBeta =
            m_GaussianSplatSettings.coverageTextureFloorEnabled ?
                m_GaussianSplatSettings.coverageTextureFloorHistoryBeta :
                0.0f;
        gaussianStats.coverageTextureStrength =
            m_GaussianSplatSettings.coverageTextureFloorEnabled ?
                m_GaussianSplatSettings.coverageTextureFloorStrength :
                0.0f;
        gaussianStats.coverageTexturePFinalMean = 0.0f;
        gaussianStats.coverageTextureStableHashKeptCount = 0u;
        gaussianStats.guideBeforeDiscardEnabled =
            m_GaussianSplatSettings.guideBeforeDiscardEnabled;
        gaussianStats.delayedGuideLogPolarEnabled =
            m_GaussianSplatSettings.delayedGuideLogPolarEnabled;
        gaussianStats.delayedGuideTemporalLogPolarEnabled =
            m_GaussianSplatSettings.delayedGuideTemporalLogPolarEnabled;
        gaussianStats.delayedGuideReleaseActiveCount = 0u;
        gaussianStats.delayedGuidePHistoryNonzeroCount = 0u;
        gaussianStats.delayedGuidePTargetLessThanHistoryCount = 0u;
        gaussianStats.delayedGuidePHistoryBytes =
            m_GaussianSplatSettings.delayedGuideTemporalLogPolarEnabled ?
                static_cast<uint64_t>(maxGeneralGaussianSplatSourceCount) * sizeof(uint32_t) :
                0u;
        gaussianStats.delayedGuideReleasedEffectiveVisibleCount = 0u;
        gaussianStats.delayedGuideReleasedAlphaProxySum = 0.0f;
        gaussianStats.delayedGuideReleaseCapHitCount = 0u;
        gaussianStats.delayedGuideRiskProtectedCount = 0u;
        gaussianStats.delayedGuideFootprintFastDecayCount = 0u;
        gaussianStats.delayedGuideBaseTileDuplicateProxyCount = 0u;
        gaussianStats.delayedGuideReleasedTileDuplicateProxyCount = 0u;
        gaussianStats.delayedGuideReleasedDuplicateProxyRatio = 0.0f;
        gaussianStats.delayedGuideLargeFootprintReleasedCount = 0u;
        gaussianStats.delayedGuideLargeFootprintReleasedTileDuplicateProxyCount = 0u;
        gaussianStats.delayedGuideReleasedRadiusMean = 0.0f;
        gaussianStats.delayedGuideReleasedRadiusP95 = 0.0f;
        gaussianStats.delayedGuideReleasedRadiusMax = 0.0f;
        gaussianStats.delayedGuideHotspotTileIds.fill(0u);
        gaussianStats.delayedGuideHotspotBaseDuplicates.fill(0u);
        gaussianStats.delayedGuideHotspotReleasedDuplicates.fill(0u);
        gaussianStats.guideBeforeDiscardTexelCount =
            m_GaussianSplatSettings.delayedGuideLogPolarEnabled ?
                std::clamp(m_GaussianSplatSettings.coverageTextureFloorWidth, 1u, 1024u) *
                    std::clamp(m_GaussianSplatSettings.coverageTextureFloorHeight, 1u, 1024u) :
                m_GaussianSplatSettings.guideBeforeDiscardEnabled ?
                std::clamp(m_GaussianSplatSettings.coverageStableReleaseTileGridX, 1u, 32u) *
                    std::clamp(m_GaussianSplatSettings.coverageStableReleaseTileGridY, 1u, 16u) :
                0u;
        gaussianStats.guideBeforeDiscardCoverageMean = 0.0f;
        gaussianStats.guideBeforeDiscardCoverageMin = 0.0f;
        gaussianStats.guideBeforeDiscardCoverageMax = 0.0f;
        gaussianStats.guideBeforeDiscardLowCoverageBoostedSplats = 0u;
        gaussianStats.guideBeforeDiscardHighCoverageReducedSplats = 0u;
        gaussianStats.guideBeforeDiscardSelectedBeforeTileExpansion = 0u;
        gaussianStats.guideBeforeDiscardTileDuplicateCount = UINT32_MAX;
        gaussianStats.shaderAntiPopDebugLogEnabled =
            m_GaussianSplatSettings.shaderAntiPopDebugLogEnabled;
        gaussianStats.shaderAntiPopGuardProxyMode =
            gaussianSplatShaderAntiPopGuardProxyMode(m_GaussianSplatSettings);
        gaussianStats.eccStochasticFadeFrames = m_GaussianSplatSettings.eccStochasticFadeFrames;
        gaussianStats.eccStochasticProtectOldFoveaDegrees =
            m_GaussianSplatSettings.eccStochasticProtectOldFoveaDegrees;
        gaussianStats.eccStochasticProtectNewFoveaDegrees =
            m_GaussianSplatSettings.eccStochasticProtectNewFoveaDegrees;
        gaussianStats.eccStochasticBoundaryBandDegrees =
            std::max(m_GaussianSplatSettings.eccStochasticBoundaryBandDegrees, 0.0f);
        gaussianStats.eccStochasticMinPDelta =
            std::max(m_GaussianSplatSettings.eccStochasticMinPDelta, 0.0f);
        gaussianStats.eccStochasticContributionGuard =
            m_GaussianSplatSettings.eccStochasticContributionGuard;
        gaussianStats.eccStochasticDebugLogEnabled =
            m_GaussianSplatSettings.eccStochasticDebugLogEnabled;
        gaussianStats.foveatedShLodEnabled =
            m_GaussianSplatSettings.foveatedShLodActive();
        gaussianStats.selectedIdDebugLoggingEnabled =
            m_GaussianSplatSettings.selectedIdDebugLoggingEnabled;
        const auto shLodDegrees =
            glm::clamp(m_GaussianSplatSettings.foveatedShLodDegrees,
                       glm::uvec3 {0u},
                       glm::uvec3 {3u});
        gaussianStats.shDegreeCenter = shLodDegrees.x;
        gaussianStats.shDegreeMid    = shLodDegrees.y;
        gaussianStats.shDegreeOuter  = shLodDegrees.z;
        gaussianStats.shGuardMode =
            m_GaussianSplatSettings.foveatedShLodActive() ?
                m_GaussianSplatSettings.foveatedShLodGuardMode :
                GaussianSplatShLodGuardMode::eOff;
        gaussianStats.shGuardThresholdMid =
            std::max(m_GaussianSplatSettings.foveatedShLodGuardThresholdMid, 0.0f);
        gaussianStats.shGuardThresholdHigh =
            std::max(m_GaussianSplatSettings.foveatedShLodGuardThresholdHigh,
                     gaussianStats.shGuardThresholdMid);
        gaussianStats.shStorageLayout =
            effectiveGaussianSplatShStorageLayout(m_GaussianSplatSettings, pool);
        applyGaussianShSmoothSuppressionStats(m_GaussianSplatSettings, gaussianStats);
        gaussianStats.splatAssets                      = static_cast<uint32_t>(m_RenderWorldBack.gaussianSplats.size());
        gaussianStats.totalSplats                      = maxGeneralGaussianSplatPoints;
        summarizeGaussianShEnergyMetadata(m_RenderWorldBack, pool, gaussianStats);
        if (m_GaussianSplatSettings.foveatedClodActive())
        {
            gaussianStats.foveaSplatBudget =
                gaussianSplatBudgetFromLevel(maxGeneralGaussianSplatPoints, m_GaussianSplatSettings.foveatedRingLevels.x);
            gaussianStats.midSplatBudget =
                gaussianSplatBudgetFromLevel(maxGeneralGaussianSplatPoints, m_GaussianSplatSettings.foveatedRingLevels.y);
            gaussianStats.outerSplatBudget =
                gaussianSplatBudgetFromLevel(maxGeneralGaussianSplatPoints, m_GaussianSplatSettings.foveatedRingLevels.z);
        }

        const bool gaussianModeSettingsDirty =
            m_GaussianSplatSettings.baselineMode != m_AppliedGaussianSplatSettings.baselineMode;
        const bool gaussianSelectionSettingsDirty =
            gaussianSplatSelectionSettingsDirty(m_GaussianSplatSettings, m_AppliedGaussianSplatSettings);
        const bool gaussianShLodSettingsDirty =
            gaussianSplatShLodSettingsDirty(m_GaussianSplatSettings, m_AppliedGaussianSplatSettings);
        const bool gaussianTemporalStateSettingsDirty =
            m_GaussianSplatSettings.foveatedTemporalHysteresisEnabled !=
                m_AppliedGaussianSplatSettings.foveatedTemporalHysteresisEnabled ||
            m_GaussianSplatSettings.foveatedBoundarySmoothingEnabled !=
                m_AppliedGaussianSplatSettings.foveatedBoundarySmoothingEnabled ||
            m_GaussianSplatSettings.foveatedTemporalResidencyFrames !=
                m_AppliedGaussianSplatSettings.foveatedTemporalResidencyFrames ||
            m_GaussianSplatSettings.foveatedTemporalHysteresisRatio !=
                m_AppliedGaussianSplatSettings.foveatedTemporalHysteresisRatio ||
            m_GaussianSplatSettings.foveatedBoundarySmoothingRatio !=
                m_AppliedGaussianSplatSettings.foveatedBoundarySmoothingRatio ||
            m_GaussianSplatSettings.shaderAntiPopMode !=
                m_AppliedGaussianSplatSettings.shaderAntiPopMode ||
            m_GaussianSplatSettings.coverageStableReleaseStaggeredReleaseEnabled !=
                m_AppliedGaussianSplatSettings.coverageStableReleaseStaggeredReleaseEnabled ||
            m_GaussianSplatSettings.coverageStableReleaseLambda !=
                m_AppliedGaussianSplatSettings.coverageStableReleaseLambda ||
            m_GaussianSplatSettings.coverageStableReleaseReleaseEpsilon !=
                m_AppliedGaussianSplatSettings.coverageStableReleaseReleaseEpsilon ||
            m_GaussianSplatSettings.delayedGuideTemporalReleasePolicy !=
                m_AppliedGaussianSplatSettings.delayedGuideTemporalReleasePolicy ||
            m_GaussianSplatSettings.delayedGuideTemporalReleaseCapRatio !=
                m_AppliedGaussianSplatSettings.delayedGuideTemporalReleaseCapRatio ||
            m_GaussianSplatSettings.delayedGuideTemporalRiskThreshold !=
                m_AppliedGaussianSplatSettings.delayedGuideTemporalRiskThreshold ||
            m_GaussianSplatSettings.delayedGuideTemporalFootprintDecayScale !=
                m_AppliedGaussianSplatSettings.delayedGuideTemporalFootprintDecayScale ||
            m_GaussianSplatSettings.delayedGuideTemporalLargeFootprintPx !=
                m_AppliedGaussianSplatSettings.delayedGuideTemporalLargeFootprintPx;
        const bool gpuSceneDirty =
            gaussianModeSettingsDirty ||
            m_GpuSceneDirtyTracker.shouldRebuild(m_RenderWorldBack, resourceRevision, m_EnableGpuDrivenMeshletPipeline);
        const bool cachedSelectionOracleActive =
            gaussianOrderedClodMode && m_GaussianSplatSettings.cachedSelectionOracleEnabled;
        const bool gaussianSelectionDirty =
            gaussianOrderedClodMode && (gaussianSelectionSettingsDirty || cachedSelectionOracleActive);
        const bool shaderAntiPopOrderFreeUniformDirty =
            !gaussianOrderedClodMode &&
            m_GaussianSplatSettings.shaderAntiPopOrderFree &&
            m_GaussianSplatSettings.shaderAntiPopActive() &&
            gaussianSelectionSettingsDirty;
        const bool shSmoothOrderFreeUniformDirty =
            !gaussianOrderedClodMode &&
            m_GaussianSplatSettings.foveatedShSmoothSuppressionActive() &&
            gaussianSelectionSettingsDirty;
        const bool shaderGazeAnchorCrossfadeActive = false;
        const bool shaderEccStochasticTransitionActive = false;
        const bool shaderGazeAnchorUniformDirty = false;
        const bool shaderEccStochasticUniformDirty = false;
        const bool gaussianProjectedCostSelectionActive =
            gaussianOrderedClodMode && m_GaussianSplatSettings.projectedCostBudgetEnabled() &&
            !cachedSelectionOracleActive;
        if (!m_GaussianSplatSettings.cachedSelectionOracleEnabled)
        {
            m_GaussianCachedSelectionOracleValid = false;
            m_GaussianCachedSelectionOracleCandidates.clear();
            m_GaussianCachedSelectionOracleStaticTargetCount = UINT32_MAX;
            m_GaussianCachedSelectionOracleMatchedNullValid = false;
            m_GaussianCachedSelectionOracleMatchedNullFrameIndex = UINT32_MAX;
        }
        if (!shaderGazeAnchorCrossfadeActive)
        {
            m_GaussianGazeAnchorValid = false;
            m_GaussianGazeAnchorGaze = glm::vec2 {0.5f, 0.5f};
            m_GaussianGazeAnchorPreviousGaze = glm::vec2 {0.5f, 0.5f};
            m_GaussianGazeAnchorLastUpdateFrame = 0u;
            m_GaussianGazeAnchorUpdateEventCount = 0u;
            m_GaussianGazeAnchorDeadbandViolationCount = 0u;
        }
        if (!shaderEccStochasticTransitionActive)
        {
            m_GaussianEccStochasticTransitionValid = false;
            m_GaussianEccStochasticOldGaze = glm::vec2 {0.5f, 0.5f};
            m_GaussianEccStochasticNewGaze = glm::vec2 {0.5f, 0.5f};
            m_GaussianEccStochasticLastUpdateFrame = 0u;
            m_GaussianEccStochasticUpdateEventCount = 0u;
        }

        auto applyCachedSelectionOracle =
            [this](resource::GpuSceneView& gpuSceneView,
                   uint32_t&               activeGaussianSplats,
                   uint32_t&               selectedSourceCapacity,
                   const std::span<const RenderCamera> cameras,
                   const rhi::Extent2D fallbackExtent,
                   GaussianSplatFrameStats& stats) -> bool {
            const auto& settings = m_GaussianSplatSettings;
            if (!settings.cachedSelectionOracleEnabled)
                return false;

            const uint32_t sourceCount =
                static_cast<uint32_t>(gpuSceneView.generalGaussianSplatPackedSources.size());
            const uint32_t currentCandidateCount =
                static_cast<uint32_t>(gpuSceneView.generalGaussianSplatSelectedSources.size());
            const bool cacheUsable =
                m_GaussianCachedSelectionOracleValid &&
                m_GaussianCachedSelectionOracleSeed == settings.cachedSelectionOracleSeed &&
                m_GaussianCachedSelectionOracleSourceCount == sourceCount &&
                m_GaussianCachedSelectionOracleCandidateCount > 0u;
            const uint32_t targetCount =
                cacheUsable ? m_GaussianCachedSelectionOracleActiveCount :
                              std::min(activeGaussianSplats, currentCandidateCount);
            const bool cacheMatches =
                cacheUsable &&
                m_GaussianCachedSelectionOracleSourceCount == sourceCount &&
                m_GaussianCachedSelectionOracleCandidateCount > 0u &&
                m_GaussianCachedSelectionOracleActiveCount == targetCount;

            if (!cacheMatches)
            {
                const auto candidates = gpuSceneView.generalGaussianSplatSelectedSources;
                m_GaussianCachedSelectionOracleCandidates.clear();
                m_GaussianCachedSelectionOracleCandidates.reserve(candidates.size());

                for (uint32_t rank = 0u; rank < static_cast<uint32_t>(candidates.size()); ++rank)
                {
                    const auto& selection = candidates[rank];
                    if ((selection.flags & 0x80000000u) != 0u ||
                        selection.sourceIndex >= gpuSceneView.generalGaussianSplatPackedSources.size() ||
                        selection.drawIndex >= gpuSceneView.generalGaussianSplatDraws.size())
                    {
                        continue;
                    }

                    const auto& source = gpuSceneView.generalGaussianSplatPackedSources[selection.sourceIndex];
                    const auto& draw = gpuSceneView.generalGaussianSplatDraws[selection.drawIndex];
                    const GaussianProjectedCostSample projected =
                        estimateGaussianSplatProjectedTileCost(source, draw, settings, cameras, fallbackExtent);

                    uint32_t visibleCameraIndex = GaussianProjectedCostSample::kMaxCameraFootprints;
                    for (uint32_t cameraIndex = 0u;
                         cameraIndex < GaussianProjectedCostSample::kMaxCameraFootprints;
                         ++cameraIndex)
                    {
                        if (projected.cameraVisible[cameraIndex] != 0u)
                        {
                            visibleCameraIndex = cameraIndex;
                            break;
                        }
                    }

                    const uint64_t hashKey =
                        gaussianCachedSelectionOracleSelectionKey(gpuSceneView,
                                                                  selection,
                                                                  settings.cachedSelectionOracleSeed,
                                                                  0xd6e8feb86659fd93ull);
                    m_GaussianCachedSelectionOracleCandidates.push_back(
                        GaussianCachedSelectionOracleCandidate {
                            .selection = selection,
                            .sourceId = gaussianCachedSelectionOracleSourceId(gpuSceneView, selection),
                            .rank = rank,
                            .centerNdc = visibleCameraIndex < GaussianProjectedCostSample::kMaxCameraFootprints ?
                                             projected.cameraCenterNdc[visibleCameraIndex] :
                                             glm::vec2 {0.0f},
                            .opacityProxy = gaussianCachedSelectionOracleOpacityProxy(source, draw, selection),
                            .projectedAreaPx = projected.projectedAreaPx,
                            .tileCost = projected.cost,
                            .hashKey = hashKey,
                            .hashThreshold = gaussianCachedSelectionOracleHashThreshold(hashKey),
                            .visibleAtCache =
                                visibleCameraIndex < GaussianProjectedCostSample::kMaxCameraFootprints,
                            .staticRandomSelected = false,
                        });
                }

                const uint32_t cachedCandidateCount =
                    static_cast<uint32_t>(m_GaussianCachedSelectionOracleCandidates.size());

                m_GaussianCachedSelectionOracleValid = true;
                m_GaussianCachedSelectionOracleSourceCount = sourceCount;
                m_GaussianCachedSelectionOracleCandidateCount = cachedCandidateCount;
                m_GaussianCachedSelectionOracleActiveCount = targetCount;
                m_GaussianCachedSelectionOracleCacheFrameIndex =
                    static_cast<uint32_t>(std::min<uint64_t>(m_FrameCounter,
                                                             std::numeric_limits<uint32_t>::max()));
                m_GaussianCachedSelectionOracleStaticTargetCount = UINT32_MAX;
                m_GaussianCachedSelectionOracleMatchedNullValid = false;
                m_GaussianCachedSelectionOracleMatchedNullFrameIndex = UINT32_MAX;
            }

            m_GaussianCachedSelectionOracleMode = settings.cachedSelectionMembershipMode;
            m_GaussianCachedSelectionOracleSeed = settings.cachedSelectionOracleSeed;

            const uint32_t cachedCandidateCount =
                static_cast<uint32_t>(m_GaussianCachedSelectionOracleCandidates.size());
            const uint32_t requestedStaticTargetCount =
                settings.cachedSelectionOracleStaticTargetCount == UINT32_MAX ?
                    std::min(m_GaussianCachedSelectionOracleActiveCount, cachedCandidateCount) :
                    std::min(settings.cachedSelectionOracleStaticTargetCount, cachedCandidateCount);
            if (m_GaussianCachedSelectionOracleStaticTargetCount != requestedStaticTargetCount)
            {
                for (auto& candidate : m_GaussianCachedSelectionOracleCandidates)
                    candidate.staticRandomSelected = false;

                std::vector<uint32_t> staticOrder(cachedCandidateCount);
                std::iota(staticOrder.begin(), staticOrder.end(), 0u);
                std::stable_sort(staticOrder.begin(), staticOrder.end(), [&](const uint32_t lhs, const uint32_t rhs) {
                    const uint64_t lhsKey = m_GaussianCachedSelectionOracleCandidates[lhs].hashKey ^
                                            0x51c37c2d9a1f734bull;
                    const uint64_t rhsKey = m_GaussianCachedSelectionOracleCandidates[rhs].hashKey ^
                                            0x51c37c2d9a1f734bull;
                    if (lhsKey != rhsKey)
                        return lhsKey < rhsKey;
                    return lhs < rhs;
                });
                for (uint32_t i = 0u; i < requestedStaticTargetCount; ++i)
                    m_GaussianCachedSelectionOracleCandidates[staticOrder[i]].staticRandomSelected = true;
                m_GaussianCachedSelectionOracleStaticTargetCount = requestedStaticTargetCount;
            }

            uint32_t matchedNullRealizedAddCount = 0u;
            uint32_t matchedNullRealizedRemoveCount = 0u;
            uint32_t matchedNullRealizedSelectedCount = 0u;
            uint32_t matchedNullRealizedSymmetricDiff = 0u;
            bool     matchedNullScheduleMatched = true;

            if (settings.cachedSelectionMembershipMode ==
                GaussianSplatCachedSelectionMembershipMode::eMatchedRateRandomNull)
            {
                const uint32_t scheduleFrame = settings.cachedSelectionOracleMatchedNullFrameIndex;
                const uint32_t targetSelected =
                    std::min(settings.cachedSelectionOracleMatchedNullTargetSelectedCount, cachedCandidateCount);
                const bool resetNullState =
                    !m_GaussianCachedSelectionOracleMatchedNullValid ||
                    scheduleFrame == 0u ||
                    scheduleFrame == UINT32_MAX ||
                    scheduleFrame <= m_GaussianCachedSelectionOracleMatchedNullFrameIndex;

                auto orderBySalt = [&](std::vector<uint32_t>& indices, const uint64_t salt) {
                    std::stable_sort(indices.begin(), indices.end(), [&](const uint32_t lhs, const uint32_t rhs) {
                        const uint64_t lhsKey =
                            gaussianCachedSelectionOracleMix(
                                m_GaussianCachedSelectionOracleCandidates[lhs].hashKey ^
                                (static_cast<uint64_t>(settings.cachedSelectionOracleSeed) << 32u) ^
                                salt);
                        const uint64_t rhsKey =
                            gaussianCachedSelectionOracleMix(
                                m_GaussianCachedSelectionOracleCandidates[rhs].hashKey ^
                                (static_cast<uint64_t>(settings.cachedSelectionOracleSeed) << 32u) ^
                                salt);
                        if (lhsKey != rhsKey)
                            return lhsKey < rhsKey;
                        return lhs < rhs;
                    });
                };

                if (resetNullState)
                {
                    for (auto& candidate : m_GaussianCachedSelectionOracleCandidates)
                        candidate.matchedNullSelected = false;

                    std::vector<uint32_t> order(cachedCandidateCount);
                    std::iota(order.begin(), order.end(), 0u);
                    orderBySalt(order, 0xb4e3a74a4f9c21d0ull);
                    for (uint32_t i = 0u; i < targetSelected; ++i)
                        m_GaussianCachedSelectionOracleCandidates[order[i]].matchedNullSelected = true;

                    matchedNullRealizedSelectedCount = targetSelected;
                    matchedNullRealizedAddCount = 0u;
                    matchedNullRealizedRemoveCount = 0u;
                    matchedNullRealizedSymmetricDiff = 0u;
                }
                else if (scheduleFrame != m_GaussianCachedSelectionOracleMatchedNullFrameIndex)
                {
                    std::vector<uint32_t> selectedBefore;
                    std::vector<uint32_t> unselectedBefore;
                    selectedBefore.reserve(cachedCandidateCount);
                    unselectedBefore.reserve(cachedCandidateCount);
                    for (uint32_t i = 0u; i < cachedCandidateCount; ++i)
                    {
                        if (m_GaussianCachedSelectionOracleCandidates[i].matchedNullSelected)
                            selectedBefore.push_back(i);
                        else
                            unselectedBefore.push_back(i);
                    }

                    orderBySalt(selectedBefore,
                                0xf27bb1340f2ab9c1ull ^
                                    (static_cast<uint64_t>(scheduleFrame) << 1u));
                    const uint32_t targetRemove =
                        std::min(settings.cachedSelectionOracleMatchedNullTargetRemoveCount,
                                 static_cast<uint32_t>(selectedBefore.size()));
                    for (uint32_t i = 0u; i < targetRemove; ++i)
                        m_GaussianCachedSelectionOracleCandidates[selectedBefore[i]].matchedNullSelected = false;
                    matchedNullRealizedRemoveCount = targetRemove;
                    if (targetRemove != settings.cachedSelectionOracleMatchedNullTargetRemoveCount)
                        matchedNullScheduleMatched = false;

                    orderBySalt(unselectedBefore,
                                0x8ad62f351c177bd5ull ^
                                    (static_cast<uint64_t>(scheduleFrame) << 1u));
                    const uint32_t targetAdd =
                        std::min(settings.cachedSelectionOracleMatchedNullTargetAddCount,
                                 static_cast<uint32_t>(unselectedBefore.size()));
                    for (uint32_t i = 0u; i < targetAdd; ++i)
                        m_GaussianCachedSelectionOracleCandidates[unselectedBefore[i]].matchedNullSelected = true;
                    matchedNullRealizedAddCount = targetAdd;
                    if (targetAdd != settings.cachedSelectionOracleMatchedNullTargetAddCount)
                        matchedNullScheduleMatched = false;

                    uint32_t currentSelected = 0u;
                    for (const auto& candidate : m_GaussianCachedSelectionOracleCandidates)
                        currentSelected += candidate.matchedNullSelected ? 1u : 0u;
                    if (currentSelected < targetSelected)
                    {
                        std::vector<uint32_t> fill;
                        fill.reserve(cachedCandidateCount - currentSelected);
                        for (uint32_t i = 0u; i < cachedCandidateCount; ++i)
                        {
                            if (!m_GaussianCachedSelectionOracleCandidates[i].matchedNullSelected)
                                fill.push_back(i);
                        }
                        orderBySalt(fill,
                                    0x651f71ed2f8b6b41ull ^
                                        (static_cast<uint64_t>(scheduleFrame) << 1u));
                        const uint32_t extra =
                            std::min(targetSelected - currentSelected, static_cast<uint32_t>(fill.size()));
                        for (uint32_t i = 0u; i < extra; ++i)
                            m_GaussianCachedSelectionOracleCandidates[fill[i]].matchedNullSelected = true;
                        matchedNullRealizedAddCount += extra;
                        currentSelected += extra;
                        matchedNullScheduleMatched = false;
                    }
                    else if (currentSelected > targetSelected)
                    {
                        std::vector<uint32_t> trim;
                        trim.reserve(currentSelected);
                        for (uint32_t i = 0u; i < cachedCandidateCount; ++i)
                        {
                            if (m_GaussianCachedSelectionOracleCandidates[i].matchedNullSelected)
                                trim.push_back(i);
                        }
                        orderBySalt(trim,
                                    0x3ff0d9734500abafull ^
                                        (static_cast<uint64_t>(scheduleFrame) << 1u));
                        const uint32_t extra = std::min(currentSelected - targetSelected,
                                                        static_cast<uint32_t>(trim.size()));
                        for (uint32_t i = 0u; i < extra; ++i)
                            m_GaussianCachedSelectionOracleCandidates[trim[i]].matchedNullSelected = false;
                        matchedNullRealizedRemoveCount += extra;
                        currentSelected -= extra;
                        matchedNullScheduleMatched = false;
                    }

                    matchedNullRealizedSelectedCount = currentSelected;
                    matchedNullRealizedSymmetricDiff =
                        matchedNullRealizedAddCount + matchedNullRealizedRemoveCount;
                }

                if (resetNullState)
                {
                    m_GaussianCachedSelectionOracleMatchedNullValid = true;
                    m_GaussianCachedSelectionOracleMatchedNullFrameIndex = scheduleFrame;
                }
                else if (scheduleFrame != m_GaussianCachedSelectionOracleMatchedNullFrameIndex)
                {
                    m_GaussianCachedSelectionOracleMatchedNullFrameIndex = scheduleFrame;
                }
            }

            std::vector<resource::GpuGeneralGaussianSplatSelectedSource> selected;
            selected.reserve(cachedCandidateCount);
            std::unordered_set<uint32_t> forcedSourceIds;
            if (settings.cachedSelectionOracleForcedSourceIdsEnabled)
            {
                forcedSourceIds.reserve(settings.cachedSelectionOracleForcedSourceIds.size());
                for (const uint32_t sourceId : settings.cachedSelectionOracleForcedSourceIds)
                    forcedSourceIds.insert(sourceId);
            }

            float eccentricityMin = std::numeric_limits<float>::infinity();
            float eccentricityMax = 0.0f;
            float keepMin = std::numeric_limits<float>::infinity();
            float keepMax = 0.0f;
            double eccentricitySum = 0.0;
            double keepSum = 0.0;
            uint32_t forcedMatchedCount = 0u;
            stats.cachedSelectionOracleCandidateDebugSamples.clear();
            if (settings.selectedIdDebugLoggingEnabled)
                stats.cachedSelectionOracleCandidateDebugSamples.reserve(cachedCandidateCount);
            const RenderCamera* referenceCamera = cameras.empty() ? nullptr : &cameras.front();
            for (const auto& candidate : m_GaussianCachedSelectionOracleCandidates)
            {
                float eccentricity = 180.0f;
                if (candidate.visibleAtCache && referenceCamera != nullptr)
                {
                    eccentricity =
                        gaussianSplatFoveatedEccentricityDegreesCpu(candidate.centerNdc,
                                                                     settings.foveatedGaze,
                                                                     *referenceCamera);
                }
                const float keepProbability =
                    std::clamp(gaussianFoveatedBaseClodLevelCpu(settings, eccentricity), 0.0f, 1.0f);
                eccentricityMin = std::min(eccentricityMin, eccentricity);
                eccentricityMax = std::max(eccentricityMax, eccentricity);
                keepMin = std::min(keepMin, keepProbability);
                keepMax = std::max(keepMax, keepProbability);
                eccentricitySum += static_cast<double>(eccentricity);
                keepSum += static_cast<double>(keepProbability);

                bool keep = false;
                if (settings.cachedSelectionOracleForcedSourceIdsEnabled)
                {
                    keep = forcedSourceIds.contains(candidate.sourceId);
                    forcedMatchedCount += keep ? 1u : 0u;
                }
                else
                {
                    switch (settings.cachedSelectionMembershipMode)
                    {
                        case GaussianSplatCachedSelectionMembershipMode::eStaticRandom:
                            keep = candidate.staticRandomSelected;
                            break;
                        case GaussianSplatCachedSelectionMembershipMode::eDynamicDeterministic:
                        {
                            const float rankRatio =
                                (static_cast<float>(candidate.rank) + 0.5f) /
                                static_cast<float>(std::max(cachedCandidateCount, 1u));
                            keep = rankRatio <= keepProbability;
                            break;
                        }
                        case GaussianSplatCachedSelectionMembershipMode::eDynamicFrozenHash:
                            keep = candidate.hashThreshold <= keepProbability;
                            break;
                        case GaussianSplatCachedSelectionMembershipMode::eMatchedRateRandomNull:
                            keep = candidate.matchedNullSelected;
                            break;
                    }
                }
                if (keep)
                    selected.push_back(candidate.selection);

                if (settings.selectedIdDebugLoggingEnabled)
                {
                    stats.cachedSelectionOracleCandidateDebugSamples.push_back(
                        GaussianSplatCachedSelectionOracleDebugSample {
                            .sourceId = candidate.sourceId,
                            .eccentricityDegrees = eccentricity,
                            .keepProbability = keepProbability,
                            .opacityProxy = candidate.opacityProxy,
                            .hashThreshold = candidate.hashThreshold,
                            .projectedAreaPx = candidate.projectedAreaPx,
                            .tileCost = candidate.tileCost,
                            .visibleAtCache = candidate.visibleAtCache,
                            .selected = keep,
                        });
                }
            }

            if (!std::isfinite(eccentricityMin))
                eccentricityMin = 0.0f;
            if (!std::isfinite(keepMin))
                keepMin = 0.0f;

            gpuSceneView.generalGaussianSplatSelectedSources = std::move(selected);
            selectedSourceCapacity =
                static_cast<uint32_t>(gpuSceneView.generalGaussianSplatSelectedSources.size());
            activeGaussianSplats = selectedSourceCapacity;

            stats.cachedSelectionOracleEnabled = true;
            stats.cachedSelectionOracleCandidateCount = cachedCandidateCount;
            stats.cachedSelectionOracleCacheFrameIndex = m_GaussianCachedSelectionOracleCacheFrameIndex;
            stats.cachedSelectionOracleMembershipRecomputedThisFrame =
                !settings.cachedSelectionOracleForcedSourceIdsEnabled;
            stats.cachedSelectionOracleShaderFoveationDisabled =
                !settings.foveatedClodActive() &&
                !settings.foveatedCoverageCompensationEnabled &&
                !settings.foveatedTemporalHysteresisEnabled &&
                !settings.foveatedBoundarySmoothingEnabled;
            stats.cachedSelectionOracleSelectedFraction =
                cachedCandidateCount > 0u ?
                    static_cast<float>(activeGaussianSplats) / static_cast<float>(cachedCandidateCount) :
                    0.0f;
            stats.cachedSelectionOracleThresholdFoveaDegrees = std::max(settings.foveatedRingDegrees.x, 0.0f);
            stats.cachedSelectionOracleThresholdMidDegrees =
                std::max(settings.foveatedRingDegrees.y, settings.foveatedRingDegrees.x);
            stats.cachedSelectionOracleFalloffTransitionDegrees =
                std::max(settings.foveatedTransitionDegrees, 0.0f);
            stats.cachedSelectionOracleLevelCenter = std::clamp(settings.foveatedRingLevels.x, 0.0f, 1.0f);
            stats.cachedSelectionOracleLevelMid = std::clamp(settings.foveatedRingLevels.y, 0.0f, 1.0f);
            stats.cachedSelectionOracleLevelOuter = std::clamp(settings.foveatedRingLevels.z, 0.0f, 1.0f);
            stats.cachedSelectionOracleEccentricityMin = eccentricityMin;
            stats.cachedSelectionOracleEccentricityMean =
                cachedCandidateCount > 0u ?
                    static_cast<float>(eccentricitySum / static_cast<double>(cachedCandidateCount)) :
                    0.0f;
            stats.cachedSelectionOracleEccentricityMax = eccentricityMax;
            stats.cachedSelectionOracleKeepProbabilityMin = keepMin;
            stats.cachedSelectionOracleKeepProbabilityMean =
                cachedCandidateCount > 0u ?
                    static_cast<float>(keepSum / static_cast<double>(cachedCandidateCount)) :
                    0.0f;
            stats.cachedSelectionOracleKeepProbabilityMax = keepMax;
            stats.cachedSelectionOracleMatchedNullTargetAddCount =
                settings.cachedSelectionOracleMatchedNullTargetAddCount;
            stats.cachedSelectionOracleMatchedNullTargetRemoveCount =
                settings.cachedSelectionOracleMatchedNullTargetRemoveCount;
            stats.cachedSelectionOracleMatchedNullTargetSelectedCount =
                settings.cachedSelectionOracleMatchedNullTargetSelectedCount;
            stats.cachedSelectionOracleMatchedNullTargetSymmetricDiff =
                settings.cachedSelectionOracleMatchedNullTargetSymmetricDiff;
            stats.cachedSelectionOracleMatchedNullRealizedAddCount = matchedNullRealizedAddCount;
            stats.cachedSelectionOracleMatchedNullRealizedRemoveCount = matchedNullRealizedRemoveCount;
            stats.cachedSelectionOracleMatchedNullRealizedSelectedCount =
                settings.cachedSelectionMembershipMode ==
                        GaussianSplatCachedSelectionMembershipMode::eMatchedRateRandomNull ?
                    activeGaussianSplats :
                    0u;
            stats.cachedSelectionOracleMatchedNullRealizedSymmetricDiff =
                matchedNullRealizedSymmetricDiff;
            stats.cachedSelectionOracleMatchedNullScheduleMatched =
                matchedNullScheduleMatched &&
                (settings.cachedSelectionMembershipMode !=
                     GaussianSplatCachedSelectionMembershipMode::eMatchedRateRandomNull ||
                 activeGaussianSplats == settings.cachedSelectionOracleMatchedNullTargetSelectedCount);
            stats.cachedSelectionOracleForcedSourceIdsEnabled =
                settings.cachedSelectionOracleForcedSourceIdsEnabled;
            stats.cachedSelectionOracleForcedSourceIdsRequestedCount =
                static_cast<uint32_t>(settings.cachedSelectionOracleForcedSourceIds.size());
            stats.cachedSelectionOracleForcedSourceIdsMatchedCount = forcedMatchedCount;
            stats.cachedSelectionOracleForcedSourceIdsAllFound =
                !settings.cachedSelectionOracleForcedSourceIdsEnabled ||
                forcedMatchedCount == forcedSourceIds.size();
            return true;
        };

        if (gpuSceneDirty || !m_GaussianSplatSettings.shPopLogEnabled)
            m_GaussianShPopAggregateState = {};
        if (gpuSceneDirty)
            ++m_GaussianProjectedCostCacheEpoch;
        // Build GPU scene database + per-view draw state.
        //
        // Database layer:
        // - stable pointer to global resource pool
        // - scene/instance tables
        //
        // View layer:
        // - draw table
        // - indirect commands
        if (gpuSceneDirty)
        {
            RuntimeProfiler::Scope scope {m_RuntimeProfiler, "GpuScene::rebuild"};
            auto        packGaussianCovariance = [](const glm::uvec4 packed) {
                const glm::vec2 p0 = glm::unpackHalf2x16(packed.x);
                const glm::vec2 p1 = glm::unpackHalf2x16(packed.y);
                const glm::vec2 p2 = glm::unpackHalf2x16(packed.z);

                glm::mat3 sigma(0.0f);
                sigma[0][0] = p0.x;
                sigma[1][0] = p0.y;
                sigma[0][1] = p0.y;
                sigma[2][0] = p1.x;
                sigma[0][2] = p1.x;
                sigma[1][1] = p1.y;
                sigma[2][1] = p2.x;
                sigma[1][2] = p2.x;
                sigma[2][2] = p2.y;
                return sigma;
            };
            auto repackGaussianCovariance = [](const glm::mat3& sigma) {
                return glm::uvec4 {
                    glm::packHalf2x16(glm::vec2(sigma[0][0], sigma[1][0])),
                    glm::packHalf2x16(glm::vec2(sigma[2][0], sigma[1][1])),
                    glm::packHalf2x16(glm::vec2(sigma[2][1], sigma[2][2])),
                    0u,
                };
            };

            m_GpuSceneDatabaseBack.beginFrame(pool);
            m_GpuSceneDatabaseBack.instances.reserve(m_RenderWorldBack.instances.size());
            m_GpuSceneDatabaseBack.transforms.reserve(m_RenderWorldBack.instances.size());
            m_GpuSceneDatabaseBack.rebuildMeshTableFromResources();

            // Keep CPU staging mirrors even though the current render path is still
            // CPU-driven. The upcoming GPU-driven cluster pipeline will consume the
            // same scene database buffers directly.
            for (const auto& inst : m_RenderWorldBack.instances)
            {
                const uint32_t transformIndex = m_GpuSceneDatabaseBack.pushTransform(inst.worldMatrix);

                resource::GpuInstance gpuInst {};
                gpuInst.meshIndex      = inst.meshIndex;
                gpuInst.materialIndex  = inst.materialIndex;
                gpuInst.transformIndex = transformIndex;
                gpuInst.flags          = 0;
                m_GpuSceneDatabaseBack.pushInstance(gpuInst);
            }
            m_GpuSceneDatabaseBack.uploadSceneTables(rd, cb);

            uint32_t maxMeshletDraws = 0;
            for (const auto& inst : m_RenderWorldBack.instances)
            {
                if (inst.meshIndex >= pool.meshes.size())
                    continue;
                maxMeshletDraws += pool.meshes[inst.meshIndex].meshletCount;
            }

            if (m_EnableGpuDrivenMeshletPipeline)
            {
                m_GpuSceneViewBack.beginFrame(m_GpuSceneDatabaseBack, resource::GpuSceneBuildMode::eGpuDriven);
                m_GpuSceneViewBack.setGpuDrivenCaps(
                    static_cast<uint32_t>(m_GpuSceneDatabaseBack.instances.size()), maxMeshletDraws, maxMeshletDraws);
            }
            else
            {
                m_GpuSceneViewBack.beginFrame(m_GpuSceneDatabaseBack, resource::GpuSceneBuildMode::eCpuDriven);
                m_GpuSceneViewBack.setGpuDrivenCaps(
                    static_cast<uint32_t>(m_GpuSceneDatabaseBack.instances.size()), maxMeshletDraws, maxMeshletDraws);
                m_GpuSceneViewBack.ensureVisibleMeshletBuffers(rd);
                m_GpuSceneViewBack.draws.reserve(maxMeshletDraws);

                for (uint32_t instanceIndex = 0;
                     instanceIndex < static_cast<uint32_t>(m_RenderWorldBack.instances.size());
                     ++instanceIndex)
                {
                    const auto& inst = m_RenderWorldBack.instances[instanceIndex];
                    if (inst.meshIndex >= pool.meshes.size())
                        continue;
                    if (instanceIndex >= m_GpuSceneDatabaseBack.instances.size())
                        continue;

                    const auto& mesh    = pool.meshes[inst.meshIndex];
                    if (mesh.meshletCount == 0)
                        continue;

                    for (uint32_t localMeshlet = 0; localMeshlet < mesh.meshletCount; ++localMeshlet)
                    {
                        const uint32_t globalMeshletIndex = mesh.meshletOffset + localMeshlet;
                        if (globalMeshletIndex >= pool.meshlets.cpuMeshlets.size())
                            continue;

                        const auto& meshlet = pool.meshlets.cpuMeshlets[globalMeshletIndex];

                        resource::GpuDrawRecord dr;
                        dr.primitiveIndex    = globalMeshletIndex;
                        dr.materialIndex     = meshlet.materialIndex;
                        dr.vertexStrideBytes = mesh.vertexStrideBytes;
                        dr.flags             = resource::gpuDrawFlagsToMask(resource::GpuDrawFlags::eMeshlet);
                        dr.vertexAddress     = pool.geometry.vertexBytesAddress;
                        dr.instanceIndex     = instanceIndex;
                        dr.padding0          = 0;
                        dr.model             = inst.worldMatrix;
                        m_GpuSceneViewBack.pushMeshletDraw(std::move(dr));
                    }
                }

                std::stable_sort(m_GpuSceneViewBack.draws.begin(), m_GpuSceneViewBack.draws.end(), [](const auto& a, const auto& b) {
                    if (a.materialIndex != b.materialIndex)
                        return a.materialIndex < b.materialIndex;
                    return a.primitiveIndex < b.primitiveIndex;
                });

                m_GpuSceneViewBack.uploadDraws(rd, cb);
                m_GpuSceneViewBack.buildIndirectFromDraws(pool);
                m_GpuSceneViewBack.uploadIndirect(rd);
            }

            m_GpuSceneViewBack.generalGaussianSplatDraws.clear();
            m_GpuSceneViewBack.generalGaussianSplatPackedSources.clear();
            m_GpuSceneViewBack.generalGaussianSplatSelectedSources.clear();
            m_GpuSceneViewBack.generalGaussianSplatDirectPrefix = false;
            m_GpuSceneViewBack.generalGaussianSplatDraws.reserve(m_RenderWorldBack.gaussianSplats.size());
            m_GpuSceneViewBack.generalGaussianSplatPackedSources.reserve(maxGeneralGaussianSplatSourceCount);

            for (const auto& splatInst : m_RenderWorldBack.gaussianSplats)
            {
                if (splatInst.splatIndex >= pool.gaussianSplats.size())
                    continue;

                const auto& gpuSplat = pool.gaussianSplats[splatInst.splatIndex];
                if (gpuSplat.pointCount == 0u)
                    continue;

                const uint32_t drawIndex = static_cast<uint32_t>(m_GpuSceneViewBack.generalGaussianSplatDraws.size());
                const uint32_t pointBase = gpuSplat.pointOffset;
                const uint32_t shBaseStride = std::max(gpuSplat.shRestCoeffCount, 1u);
                const uint32_t rawSourceOffset =
                    static_cast<uint32_t>(m_GpuSceneViewBack.generalGaussianSplatPackedSources.size());

                resource::GpuGeneralGaussianSplatDrawRecord drawRecord {};
                drawRecord.splatIndex  = splatInst.splatIndex;
                drawRecord.pointOffset = rawSourceOffset;
                drawRecord.pointCount  = gpuSplat.pointCount;
                drawRecord.shDegree    = static_cast<uint32_t>(std::max(gpuSplat.shDegree, 0));
                // x: kernel size, y: cutoff scale, z: opacity scale, w: reserved sort order.
                drawRecord.params0 = glm::vec4 {0.3f,
                                                1.0f,
                                                1.0f,
                                                0.0f};
                drawRecord.model   = splatInst.worldMatrix;

                for (uint32_t localPoint = 0; localPoint < gpuSplat.pointCount; ++localPoint)
                {
                    const uint32_t globalPoint = pointBase + localPoint;
                    resource::GpuGeneralGaussianSplatPackedSource packed {};
                    if (globalPoint < pool.gaussianStorage.cpuCenters.size() &&
                        globalPoint < pool.gaussianStorage.cpuCovariances.size() &&
                        globalPoint < pool.gaussianStorage.cpuColors.size())
                    {
                        const glm::vec4 localCenter = pool.gaussianStorage.cpuCenters[globalPoint];
                        const uint32_t shOffset = globalPoint * shBaseStride;
                        const glm::uvec2 sh0 =
                            shOffset < pool.gaussianStorage.cpuSh.size() ?
                                pool.gaussianStorage.cpuSh[shOffset] :
                                glm::uvec2 {0u};

                        packed.posOpacity = glm::uvec4 {
                            std::bit_cast<uint32_t>(localCenter.x),
                            std::bit_cast<uint32_t>(localCenter.y),
                            std::bit_cast<uint32_t>(localCenter.z),
                            pool.gaussianStorage.cpuColors[globalPoint].y,
                        };
                        packed.covariance0 = pool.gaussianStorage.cpuCovariances[globalPoint];
                        packed.colorSh0    = glm::uvec4 {
                            pool.gaussianStorage.cpuColors[globalPoint].x,
                            pool.gaussianStorage.cpuColors[globalPoint].y,
                            sh0.x,
                            sh0.y,
                        };
                        packed.aux0 = glm::uvec4 {globalPoint, 0u, shOffset, 0u};
                    }
                    m_GpuSceneViewBack.pushGeneralGaussianSplatSource(packed);
                }

                m_GpuSceneViewBack.pushGeneralGaussianSplatDraw(drawRecord);
            }

            uint32_t selectedSourceCapacity = 0u;
            uint32_t activeGaussianSplats   = 0u;
            const uint32_t packedGaussianSources =
                static_cast<uint32_t>(m_GpuSceneViewBack.generalGaussianSplatPackedSources.size());
            const bool gaussianDirectPrefix =
                gaussianOrderedClodMode &&
                m_GpuSceneViewBack.generalGaussianSplatDraws.size() == 1u &&
                !m_GaussianSplatSettings.foveatedScoreBudgetEnabled() &&
                !m_GaussianSplatSettings.cachedSelectionOracleEnabled;
            gaussianStats.directPrefix = gaussianDirectPrefix;
            if (gaussianDirectPrefix)
            {
                selectedSourceCapacity = 0u;
                activeGaussianSplats = selectGaussianSplatPrefixBudget(m_GaussianSplatSettings,
                                                                        m_GpuSceneViewBack,
                                                                        packedGaussianSources,
                                                                        maxGeneralGaussianSplatPoints,
                                                                        gaussianDirectPrefix,
                                                                        cams,
                                                                        defaultTarget.getExtent(),
                                                                        m_GaussianProjectedCostCacheEpoch,
                                                                        gaussianStats,
                                                                        m_RuntimeProfiler,
                                                                        m_GaussianFoveatedScoreResidency);
                gaussianStats.lodSelectedRawSplats = activeGaussianSplats;
            }
            else if (gaussianOrderedClodMode)
            {
                rebuildGaussianSplatOrderedClodPrefixSources(m_GpuSceneViewBack,
                                                             m_RenderWorldBack.gaussianSplats,
                                                             pool,
                                                             m_RuntimeProfiler);
                selectedSourceCapacity =
                    static_cast<uint32_t>(m_GpuSceneViewBack.generalGaussianSplatSelectedSources.size());
                activeGaussianSplats = selectGaussianSplatPrefixBudget(m_GaussianSplatSettings,
                                                                        m_GpuSceneViewBack,
                                                                        selectedSourceCapacity,
                                                                        maxGeneralGaussianSplatPoints,
                                                                        gaussianDirectPrefix,
                                                                        cams,
                                                                        defaultTarget.getExtent(),
                                                                        m_GaussianProjectedCostCacheEpoch,
                                                                        gaussianStats,
                                                                        m_RuntimeProfiler,
                                                                        m_GaussianFoveatedScoreResidency);
                gaussianStats.lodSelectedRawSplats = activeGaussianSplats;
            }
            else
            {
                rebuildGaussianSplatSelectedSources(m_GpuSceneViewBack,
                                                    m_RenderWorldBack.gaussianSplats,
                                                    pool,
                                                    gaussianStats,
                                                    m_RuntimeProfiler);
                selectedSourceCapacity =
                    static_cast<uint32_t>(m_GpuSceneViewBack.generalGaussianSplatSelectedSources.size());
                activeGaussianSplats = selectedSourceCapacity;
            }

            applyCachedSelectionOracle(m_GpuSceneViewBack,
                                       activeGaussianSplats,
                                       selectedSourceCapacity,
                                       cams,
                                       defaultTarget.getExtent(),
                                       gaussianStats);
            gaussianStats.lodSelectedRawSplats = activeGaussianSplats;

            uint32_t maxVisibleGaussianSplats = activeGaussianSplats;
            if (!m_GaussianSplatSettings.cachedSelectionOracleEnabled &&
                m_GaussianSplatSettings.lodBudgetEnabled() &&
                m_GaussianSplatSettings.lodBudgetMode == GaussianSplatLodBudgetMode::eCount &&
                m_GaussianSplatSettings.lodBudget > 0u)
            {
                maxVisibleGaussianSplats = std::min(maxVisibleGaussianSplats, m_GaussianSplatSettings.lodBudget);
            }

            gaussianStats.drawRecords        = static_cast<uint32_t>(m_GpuSceneViewBack.generalGaussianSplatDraws.size());
            gaussianStats.preparedSplats     = activeGaussianSplats;
            gaussianStats.maxVisibleSplatCap = maxVisibleGaussianSplats;
            applyGaussianSplatShaderGazeAnchorProxyStats(m_GaussianSplatSettings,
                                                         activeGaussianSplats,
                                                         gaussianStats);
            applyGaussianSplatShaderEccStochasticProxyStats(m_GaussianSplatSettings,
                                                            activeGaussianSplats,
                                                            gaussianStats);
            if (gaussianStats.shaderAntiPopEnabled)
            {
                gaussianStats.shaderAntiPopCandidateCount = activeGaussianSplats;
                gaussianStats.shaderAntiPopStableCandidateSet = true;
                gaussianStats.shaderAntiPopDirectPrefixStable = gaussianDirectPrefix;
                gaussianStats.shaderAntiPopAvoidsCpuSelectedSourceRebuild =
                    m_GaussianSplatSettings.shaderAntiPopOrderFree || gaussianDirectPrefix ||
                    selectedSourceCapacity == 0u;
            }
            collectGaussianSplatSelectedSourceIds(m_GpuSceneViewBack,
                                                  activeGaussianSplats,
                                                  gaussianDirectPrefix,
                                                  gaussianStats);
            if (m_GaussianSplatSettings.cachedSelectionOracleEnabled)
            {
                gaussianStats.cachedSelectionOracleRenderBufferMatchesLoggedSet =
                    gaussianStats.selectedSourceIds.size() == activeGaussianSplats;
            }
            estimateGaussianShLodRuntimeReads(m_GaussianSplatSettings,
                                              gaussianStats,
                                              m_GaussianShPopAggregateState);
            m_GaussianSplatStats             = gaussianStats;

            m_GpuSceneViewBack.setGeneralGaussianSplatCaps(
                gaussianStats.drawRecords,
                packedGaussianSources,
                selectedSourceCapacity,
                activeGaussianSplats,
                maxVisibleGaussianSplats,
                gaussianDirectPrefix);
            applyGaussianSplatFoveatedClodSettings(m_GpuSceneViewBack, m_GaussianSplatSettings, gaussianStats);
            bindGaussianSplatShStorageBuffers(m_GpuSceneViewBack,
                                              pool,
                                              gaussianStats.shStorageLayout);
            m_GpuSceneViewBack.ensureGeneralGaussianSplatBuffers(rd);
            resetGaussianSplatTemporalStateIfNeeded(cb, m_GpuSceneViewBack);

            auto uploadStorage = [&](rhi::Buffer& buffer, const uint64_t size, const void* data) {
                constexpr uint64_t kInlineUpdateLimit = 64ull * 1024ull;
                if (size > kInlineUpdateLimit)
                    rd.uploadS(buffer, 0, size, data);
                else
                    cb.update(buffer, 0, size, data);
            };

            if (!m_GpuSceneViewBack.generalGaussianSplatDraws.empty() && m_GpuSceneViewBack.generalGaussianSplatDrawBuffer)
            {
                uploadStorage(*m_GpuSceneViewBack.generalGaussianSplatDrawBuffer,
                              static_cast<uint64_t>(m_GpuSceneViewBack.generalGaussianSplatDraws.size()) *
                                  sizeof(resource::GpuGeneralGaussianSplatDrawRecord),
                              m_GpuSceneViewBack.generalGaussianSplatDraws.data());
            }

            if (!m_GpuSceneViewBack.generalGaussianSplatPackedSources.empty() &&
                m_GpuSceneViewBack.generalGaussianSplatPackedSourceBuffer)
            {
                uploadStorage(*m_GpuSceneViewBack.generalGaussianSplatPackedSourceBuffer,
                              static_cast<uint64_t>(m_GpuSceneViewBack.generalGaussianSplatPackedSources.size()) *
                                  sizeof(resource::GpuGeneralGaussianSplatPackedSource),
                              m_GpuSceneViewBack.generalGaussianSplatPackedSources.data());
            }

            if (!m_GpuSceneViewBack.generalGaussianSplatSelectedSources.empty() &&
                m_GpuSceneViewBack.generalGaussianSplatSelectedSourceBuffer)
            {
                RuntimeProfiler::Scope scope {m_RuntimeProfiler, "GaussianLOD::UploadSelected"};
                uploadStorage(*m_GpuSceneViewBack.generalGaussianSplatSelectedSourceBuffer,
                              static_cast<uint64_t>(m_GpuSceneViewBack.generalGaussianSplatSelectedSources.size()) *
                                  sizeof(resource::GpuGeneralGaussianSplatSelectedSource),
                              m_GpuSceneViewBack.generalGaussianSplatSelectedSources.data());
            }

            if (m_GpuSceneViewBack.generalGaussianSplatVisibleCountBuffer)
            {
                const uint32_t zero = 0u;
                cb.update(*m_GpuSceneViewBack.generalGaussianSplatVisibleCountBuffer, 0, sizeof(uint32_t), &zero);
            }

            if (m_GpuSceneViewBack.generalGaussianSplatDispatchArgsBuffer)
            {
                const uint32_t zeroArgs[4] = {0u, 1u, 1u, 0u};
                cb.update(*m_GpuSceneViewBack.generalGaussianSplatDispatchArgsBuffer,
                          0,
                          sizeof(zeroArgs),
                          zeroArgs);
            }

            resetGaussianSplatIndirectBuffers(rd, m_GpuSceneViewBack);

            m_RenderWorldBack.gpuSceneDatabase = &m_GpuSceneDatabaseBack;
            m_RenderWorldBack.gpuSceneView     = &m_GpuSceneViewBack;
        }
        else
        {
            // Reuse previous snapshot when neither cooked world nor resource pool changed.
            m_RenderWorldBack.gpuSceneDatabase = &m_GpuSceneDatabaseFront;
            m_RenderWorldBack.gpuSceneView     = &m_GpuSceneViewFront;
        }

        if (!gpuSceneDirty && (gaussianSelectionDirty || gaussianProjectedCostSelectionActive))
        {
            RuntimeProfiler::Scope scope {m_RuntimeProfiler, "GpuScene::gaussian_lod_selection"};
            auto& gpuSceneView = m_GpuSceneViewFront;
            if (gaussianTemporalStateSettingsDirty)
                gpuSceneView.generalGaussianSplatTemporalStateNeedsReset = true;

            uint32_t selectedSourceCapacity = static_cast<uint32_t>(gpuSceneView.generalGaussianSplatSelectedSources.size());
            uint32_t activeGaussianSplats   = 0u;
            bool     uploadSelectedSources  = false;
            const bool gaussianDirectPrefix = gpuSceneView.generalGaussianSplatDirectPrefix;
            gaussianStats.directPrefix      = gaussianDirectPrefix;
            const uint32_t scoreSelectionStride =
                std::max(m_GaussianSplatSettings.foveatedScoreSelectionStride, 1u);
            const bool foveatedScoreOnlyGazeDirty =
                m_GaussianSplatSettings.foveatedScoreBudgetEnabled() &&
                m_GaussianSplatSettings.foveatedGaze != m_AppliedGaussianSplatSettings.foveatedGaze &&
                !gaussianSplatSelectionSettingsDirtyIgnoringGaze(m_GaussianSplatSettings,
                                                                  m_AppliedGaussianSplatSettings);
            const bool reuseFoveatedScoreSelection =
                foveatedScoreOnlyGazeDirty &&
                scoreSelectionStride > 1u &&
                (m_FrameCounter % scoreSelectionStride) != 0u &&
                !gaussianDirectPrefix &&
                selectedSourceCapacity > 0u;
            const bool reuseCachedSelectionOracleCandidates =
                m_GaussianSplatSettings.cachedSelectionOracleEnabled &&
                m_GaussianCachedSelectionOracleValid &&
                m_GaussianCachedSelectionOracleCandidateCount > 0u;
            if (!reuseCachedSelectionOracleCandidates &&
                !reuseFoveatedScoreSelection && !gaussianDirectPrefix &&
                selectedSourceCapacity < maxGeneralGaussianSplatPoints)
            {
                rebuildGaussianSplatOrderedClodPrefixSources(gpuSceneView,
                                                             m_RenderWorldBack.gaussianSplats,
                                                             pool,
                                                             m_RuntimeProfiler);
                selectedSourceCapacity =
                    static_cast<uint32_t>(gpuSceneView.generalGaussianSplatSelectedSources.size());
                uploadSelectedSources = true;
            }

            if (reuseCachedSelectionOracleCandidates)
            {
                const auto previousStats = m_GaussianSplatStats;
                selectedSourceCapacity = m_GaussianCachedSelectionOracleCandidateCount;
                activeGaussianSplats = m_GaussianCachedSelectionOracleActiveCount;
                gaussianStats.costBudget = previousStats.costBudget;
                gaussianStats.actualProjectedCost = previousStats.actualProjectedCost;
                gaussianStats.projectedCostBudgetRatio =
                    previousStats.projectedCostBudgetRatio > 0.0f ?
                        previousStats.projectedCostBudgetRatio :
                        m_GaussianSplatSettings.clodLevel;
                gaussianStats.selectedCostChunks = previousStats.selectedCostChunks;
                gaussianStats.numTileIntersections = previousStats.numTileIntersections;
                gaussianStats.sumProjectedAreaPx = previousStats.sumProjectedAreaPx;
                gaussianStats.a4ProjectedCostSelectionCpuMs = 0.0;
                gaussianStats.a4CostEstimationCpuMs = 0.0;
                gaussianStats.a4ChunkSelectionCpuMs = 0.0;
                gaussianStats.a4ChunkStopCpuMs = 0.0;
                gaussianStats.a4NumGaussiansOrChunksConsidered = m_GaussianCachedSelectionOracleCandidateCount;
            }
            else if (reuseFoveatedScoreSelection)
            {
                const auto previousStats = m_GaussianSplatStats;
                activeGaussianSplats = std::min(selectedSourceCapacity, maxGeneralGaussianSplatPoints);
                gaussianStats.costBudget = previousStats.costBudget;
                gaussianStats.actualProjectedCost = previousStats.actualProjectedCost;
                gaussianStats.projectedCostBudgetRatio =
                    previousStats.projectedCostBudgetRatio > 0.0f ?
                        previousStats.projectedCostBudgetRatio :
                        m_GaussianSplatSettings.clodLevel;
                gaussianStats.selectedCostChunks = 0u;
                gaussianStats.numTileIntersections = previousStats.numTileIntersections;
                gaussianStats.sumProjectedAreaPx = previousStats.sumProjectedAreaPx;
                gaussianStats.a4ProjectedCostSelectionCpuMs = 0.0;
                gaussianStats.a4CostEstimationCpuMs = 0.0;
                gaussianStats.a4ChunkSelectionCpuMs = 0.0;
                gaussianStats.a4ChunkStopCpuMs = 0.0;
                gaussianStats.a4NumGaussiansOrChunksConsidered = 0u;
            }
            else
            {
                const uint32_t activeBudgetSourceCount =
                    gaussianDirectPrefix ? static_cast<uint32_t>(gpuSceneView.generalGaussianSplatPackedSources.size()) :
                                           selectedSourceCapacity;
                GaussianProjectedCostGpuBuildContext projectedCostGpuContext {
                    .rd        = &rd,
                    .shaderLib = &shaderService.builtinLibrary(),
                };
                activeGaussianSplats = selectGaussianSplatPrefixBudget(m_GaussianSplatSettings,
                                                                        gpuSceneView,
                                                                        activeBudgetSourceCount,
                                                                        maxGeneralGaussianSplatPoints,
                                                                        gaussianDirectPrefix,
                                                                        cams,
                                                                        defaultTarget.getExtent(),
                                                                        m_GaussianProjectedCostCacheEpoch,
                                                                        gaussianStats,
                                                                        m_RuntimeProfiler,
                                                                        m_GaussianFoveatedScoreResidency,
                                                                        &projectedCostGpuContext);
            }
            gaussianStats.lodSelectedRawSplats = activeGaussianSplats;

            uploadSelectedSources =
                applyCachedSelectionOracle(gpuSceneView,
                                           activeGaussianSplats,
                                           selectedSourceCapacity,
                                           cams,
                                           defaultTarget.getExtent(),
                                           gaussianStats) ||
                uploadSelectedSources;
            gaussianStats.lodSelectedRawSplats = activeGaussianSplats;

            uint32_t maxVisibleGaussianSplats = activeGaussianSplats;
            if (!m_GaussianSplatSettings.cachedSelectionOracleEnabled &&
                m_GaussianSplatSettings.lodBudgetEnabled() &&
                m_GaussianSplatSettings.lodBudgetMode == GaussianSplatLodBudgetMode::eCount &&
                m_GaussianSplatSettings.lodBudget > 0u)
            {
                maxVisibleGaussianSplats = std::min(maxVisibleGaussianSplats, m_GaussianSplatSettings.lodBudget);
            }

            gaussianStats.drawRecords        = static_cast<uint32_t>(gpuSceneView.generalGaussianSplatDraws.size());
            gaussianStats.preparedSplats     = activeGaussianSplats;
            gaussianStats.maxVisibleSplatCap = maxVisibleGaussianSplats;
            applyGaussianSplatShaderGazeAnchorProxyStats(m_GaussianSplatSettings,
                                                         activeGaussianSplats,
                                                         gaussianStats);
            applyGaussianSplatShaderEccStochasticProxyStats(m_GaussianSplatSettings,
                                                            activeGaussianSplats,
                                                            gaussianStats);
            if (gaussianStats.shaderAntiPopEnabled)
            {
                gaussianStats.shaderAntiPopCandidateCount = activeGaussianSplats;
                gaussianStats.shaderAntiPopStableCandidateSet = true;
                gaussianStats.shaderAntiPopDirectPrefixStable = gaussianDirectPrefix;
                gaussianStats.shaderAntiPopAvoidsCpuSelectedSourceRebuild =
                    m_GaussianSplatSettings.shaderAntiPopOrderFree || gaussianDirectPrefix ||
                    !uploadSelectedSources;
            }
            collectGaussianSplatSelectedSourceIds(gpuSceneView,
                                                  activeGaussianSplats,
                                                  gaussianDirectPrefix,
                                                  gaussianStats);
            if (m_GaussianSplatSettings.cachedSelectionOracleEnabled)
            {
                gaussianStats.cachedSelectionOracleRenderBufferMatchesLoggedSet =
                    gaussianStats.selectedSourceIds.size() == activeGaussianSplats;
            }
            estimateGaussianShLodRuntimeReads(m_GaussianSplatSettings,
                                              gaussianStats,
                                              m_GaussianShPopAggregateState);
            m_GaussianSplatStats             = gaussianStats;

            gpuSceneView.setGeneralGaussianSplatCaps(
                gaussianStats.drawRecords,
                static_cast<uint32_t>(gpuSceneView.generalGaussianSplatPackedSources.size()),
                selectedSourceCapacity,
                activeGaussianSplats,
                maxVisibleGaussianSplats,
                gaussianDirectPrefix);
            applyGaussianSplatFoveatedClodSettings(gpuSceneView, m_GaussianSplatSettings, gaussianStats);
            bindGaussianSplatShStorageBuffers(gpuSceneView,
                                              pool,
                                              gaussianStats.shStorageLayout);
            gpuSceneView.ensureGeneralGaussianSplatBuffers(rd);
            resetGaussianSplatTemporalStateIfNeeded(cb, gpuSceneView);

            if (uploadSelectedSources && !gpuSceneView.generalGaussianSplatSelectedSources.empty() &&
                gpuSceneView.generalGaussianSplatSelectedSourceBuffer)
            {
                RuntimeProfiler::Scope scope {m_RuntimeProfiler, "GaussianLOD::UploadSelected"};
                cb.update(*gpuSceneView.generalGaussianSplatSelectedSourceBuffer,
                          0,
                          static_cast<uint64_t>(gpuSceneView.generalGaussianSplatSelectedSources.size()) *
                              sizeof(resource::GpuGeneralGaussianSplatSelectedSource),
                          gpuSceneView.generalGaussianSplatSelectedSources.data());
            }

            if (gpuSceneView.generalGaussianSplatVisibleCountBuffer)
            {
                const uint32_t zero = 0u;
                cb.update(*gpuSceneView.generalGaussianSplatVisibleCountBuffer, 0, sizeof(uint32_t), &zero);
            }

            if (gpuSceneView.generalGaussianSplatDispatchArgsBuffer)
            {
                const uint32_t zeroArgs[4] = {0u, 1u, 1u, 0u};
                cb.update(*gpuSceneView.generalGaussianSplatDispatchArgsBuffer, 0, sizeof(zeroArgs), zeroArgs);
            }

            resetGaussianSplatIndirectBuffers(rd, gpuSceneView);
        }

        if (!gpuSceneDirty && !gaussianSelectionDirty && !gaussianProjectedCostSelectionActive &&
            gaussianShLodSettingsDirty)
        {
            auto& gpuSceneView = m_GpuSceneViewFront;
            auto  updatedStats = m_GaussianSplatStats;
            if (gaussianTemporalStateSettingsDirty)
            {
                gpuSceneView.generalGaussianSplatTemporalStateNeedsReset = true;
                resetGaussianSplatTemporalStateIfNeeded(cb, gpuSceneView);
            }
            updatedStats.frameIndex = m_FrameCounter;
            updatedStats.foveatedShLodEnabled =
                m_GaussianSplatSettings.foveatedShLodActive();
            const auto shLodDegrees =
                glm::clamp(m_GaussianSplatSettings.foveatedShLodDegrees,
                           glm::uvec3 {0u},
                           glm::uvec3 {3u});
            updatedStats.shDegreeCenter = shLodDegrees.x;
            updatedStats.shDegreeMid    = shLodDegrees.y;
            updatedStats.shDegreeOuter  = shLodDegrees.z;
            updatedStats.shGuardMode =
                m_GaussianSplatSettings.foveatedShLodActive() ?
                    m_GaussianSplatSettings.foveatedShLodGuardMode :
                    GaussianSplatShLodGuardMode::eOff;
            updatedStats.shGuardThresholdMid =
                std::max(m_GaussianSplatSettings.foveatedShLodGuardThresholdMid, 0.0f);
            updatedStats.shGuardThresholdHigh =
                std::max(m_GaussianSplatSettings.foveatedShLodGuardThresholdHigh,
                         updatedStats.shGuardThresholdMid);
            updatedStats.shStorageLayout =
                effectiveGaussianSplatShStorageLayout(m_GaussianSplatSettings, pool);
            applyGaussianShSmoothSuppressionStats(m_GaussianSplatSettings, updatedStats);
            if (shaderGazeAnchorCrossfadeActive)
                copyGaussianSplatShaderGazeAnchorStats(gaussianStats, updatedStats);
            if (shaderEccStochasticTransitionActive)
                copyGaussianSplatShaderEccStochasticStats(gaussianStats, updatedStats);
            estimateGaussianShLodRuntimeReads(m_GaussianSplatSettings,
                                              updatedStats,
                                              m_GaussianShPopAggregateState);
            m_GaussianSplatStats = updatedStats;

            applyGaussianSplatFoveatedClodSettings(gpuSceneView, m_GaussianSplatSettings, updatedStats);
            bindGaussianSplatShStorageBuffers(gpuSceneView,
                                              pool,
                                              updatedStats.shStorageLayout);
        }

        if (!gpuSceneDirty && !gaussianSelectionDirty && !gaussianProjectedCostSelectionActive &&
            !gaussianShLodSettingsDirty &&
            (shaderAntiPopOrderFreeUniformDirty || shSmoothOrderFreeUniformDirty ||
             shaderGazeAnchorUniformDirty || shaderEccStochasticUniformDirty))
        {
            auto& gpuSceneView = m_GpuSceneViewFront;
            auto  updatedStats = m_GaussianSplatStats;
            updatedStats.frameIndex = m_FrameCounter;
            updatedStats.foveatedClodEnabled = m_GaussianSplatSettings.foveatedClodActive();
            updatedStats.foveatedLayeredCompositeEnabled =
                m_GaussianSplatSettings.foveatedLayeredCompositeActive();
            updatedStats.foveatedCoverageCompensationEnabled =
                m_GaussianSplatSettings.foveatedCoverageCompensationEnabled;
            updatedStats.foveatedGaze = m_GaussianSplatSettings.foveatedGaze;
            updatedStats.foveatedRingDegrees = m_GaussianSplatSettings.foveatedRingDegrees;
            updatedStats.foveatedRingLevels = m_GaussianSplatSettings.foveatedRingLevels;
            updatedStats.foveatedResolutionScales = m_GaussianSplatSettings.foveatedResolutionScales;
            updatedStats.foveatedDistribution = m_GaussianSplatSettings.foveatedDistribution;
            updatedStats.foveatedContinuousTheta0Degrees =
                m_GaussianSplatSettings.foveatedContinuousTheta0Degrees;
            updatedStats.foveatedContinuousAlpha = m_GaussianSplatSettings.foveatedContinuousAlpha;
            updatedStats.foveatedContinuousMinLevel = m_GaussianSplatSettings.foveatedContinuousMinLevel;
            updatedStats.foveatedTemporalHysteresisEnabled =
                m_GaussianSplatSettings.foveatedTemporalHysteresisEnabled;
            updatedStats.foveatedBoundarySmoothingEnabled =
                m_GaussianSplatSettings.foveatedBoundarySmoothingEnabled;
            updatedStats.foveatedTemporalResidencyFrames =
                m_GaussianSplatSettings.foveatedTemporalResidencyFrames;
            updatedStats.foveatedTemporalHysteresisRatio =
                m_GaussianSplatSettings.foveatedTemporalHysteresisRatio;
            updatedStats.foveatedBoundarySmoothingRatio =
                m_GaussianSplatSettings.foveatedBoundarySmoothingRatio;
            updatedStats.foveatedTemporalPeripheralScale =
                m_GaussianSplatSettings.foveatedTemporalPeripheralScale;
            updatedStats.peripheralTemporalFilterEnabled =
                m_GaussianSplatSettings.peripheralTemporalFilterEnabled;
            updatedStats.peripheralTemporalFilterMidDegrees =
                std::max(m_GaussianSplatSettings.foveatedRingDegrees.y,
                         m_GaussianSplatSettings.foveatedRingDegrees.x + 1.0f);
            updatedStats.peripheralTemporalFilterOuterDegrees =
                std::max(m_GaussianSplatSettings.peripheralTemporalFilterOuterDegrees,
                         updatedStats.peripheralTemporalFilterMidDegrees + 1.0f);
            updatedStats.peripheralTemporalFilterLambdaScale =
                std::clamp(m_GaussianSplatSettings.peripheralTemporalFilterLambdaScale, 0.0f, 1.0f);
            updatedStats.peripheralTemporalFilterRejectionThreshold =
                std::max(m_GaussianSplatSettings.peripheralTemporalFilterRejectionThreshold, 1e-4f);
            updatedStats.peripheralTemporalFilterClampRadius =
                std::max(m_GaussianSplatSettings.peripheralTemporalFilterClampRadius, 0.0f);
            updatedStats.foveatedShLodEnabled =
                m_GaussianSplatSettings.foveatedShLodActive();
            const auto shLodDegrees =
                glm::clamp(m_GaussianSplatSettings.foveatedShLodDegrees,
                           glm::uvec3 {0u},
                           glm::uvec3 {3u});
            updatedStats.shDegreeCenter = shLodDegrees.x;
            updatedStats.shDegreeMid = shLodDegrees.y;
            updatedStats.shDegreeOuter = shLodDegrees.z;
            applyGaussianShSmoothSuppressionStats(m_GaussianSplatSettings, updatedStats);
            updatedStats.shaderAntiPopMode = m_GaussianSplatSettings.shaderAntiPopMode;
            updatedStats.shaderAntiPopEnabled = m_GaussianSplatSettings.shaderAntiPopActive();
            updatedStats.shaderAntiPopAlphaMultiplierBased =
                updatedStats.shaderAntiPopEnabled &&
                m_GaussianSplatSettings.shaderAntiPopMode !=
                    GaussianSplatShaderAntiPopMode::eCoverageStableLogpolarRelease;
            updatedStats.shaderAntiPopAvoidsCpuSelectedSourceRebuild = true;
            updatedStats.shaderAntiPopHashSeed = m_GaussianSplatSettings.shaderAntiPopHashSeed;
            updatedStats.shaderAntiPopRampWidth =
                std::clamp(m_GaussianSplatSettings.shaderAntiPopRampWidth, 0.0f, 1.0f);
            updatedStats.shaderAntiPopGuardThreshold =
                std::clamp(m_GaussianSplatSettings.shaderAntiPopGuardThreshold, 0.0f, 1.0f);
            updatedStats.shaderAntiPopGuardFloor =
                std::clamp(m_GaussianSplatSettings.shaderAntiPopGuardFloor, 0.0f, 1.0f);
            updatedStats.shaderAntiPopPKeepCurve = m_GaussianSplatSettings.shaderAntiPopPKeepCurve;
            updatedStats.shaderAntiPopPrefixRatio = m_GaussianSplatSettings.shaderAntiPopPrefixRatio;
            updatedStats.shaderAntiPopNormalizeMode = m_GaussianSplatSettings.shaderAntiPopNormalizeMode;
            updatedStats.shaderAntiPopNormalizeStrength =
                std::clamp(m_GaussianSplatSettings.shaderAntiPopNormalizeStrength, 0.0f, 1.0f);
            updatedStats.shaderAntiPopNormalizeClampMin =
                std::max(m_GaussianSplatSettings.shaderAntiPopNormalizeClampMin, 0.0f);
            updatedStats.shaderAntiPopNormalizeClampMax =
                std::max(m_GaussianSplatSettings.shaderAntiPopNormalizeClampMax,
                         updatedStats.shaderAntiPopNormalizeClampMin);
            updatedStats.shaderAntiPopNormalizeFactor =
                std::clamp(m_GaussianSplatSettings.shaderAntiPopNormalizeFactor,
                           updatedStats.shaderAntiPopNormalizeClampMin,
                           updatedStats.shaderAntiPopNormalizeClampMax);
            updatedStats.shaderAntiPopDebugLogEnabled =
                m_GaussianSplatSettings.shaderAntiPopDebugLogEnabled;
            updatedStats.shaderAntiPopGuardProxyMode =
                gaussianSplatShaderAntiPopGuardProxyMode(m_GaussianSplatSettings);
            if (shaderGazeAnchorCrossfadeActive)
                copyGaussianSplatShaderGazeAnchorStats(gaussianStats, updatedStats);
            if (shaderEccStochasticTransitionActive)
                copyGaussianSplatShaderEccStochasticStats(gaussianStats, updatedStats);
            m_GaussianSplatStats = updatedStats;
            applyGaussianSplatFoveatedClodSettings(gpuSceneView, m_GaussianSplatSettings, updatedStats);
        }

        if (!gpuSceneDirty && !gaussianSelectionDirty && !gaussianProjectedCostSelectionActive &&
            !gaussianShLodSettingsDirty && m_GaussianSplatSettings.shPopLogEnabled)
        {
            auto updatedStats = m_GaussianSplatStats;
            updatedStats.frameIndex = m_FrameCounter;
            updatedStats.foveatedGaze = m_GaussianSplatSettings.foveatedGaze;
            estimateGaussianShLodRuntimeReads(m_GaussianSplatSettings,
                                              updatedStats,
                                              m_GaussianShPopAggregateState);
            m_GaussianSplatStats = updatedStats;
        }

        m_GpuSceneDirtyTracker.markBuilt(m_RenderWorldBack, resourceRevision, m_EnableGpuDrivenMeshletPipeline);
        if (gpuSceneDirty || gaussianSelectionDirty || gaussianProjectedCostSelectionActive ||
            gaussianShLodSettingsDirty || shaderAntiPopOrderFreeUniformDirty || shSmoothOrderFreeUniformDirty ||
            shaderEccStochasticUniformDirty)
        {
            m_AppliedGaussianSplatSettings = m_GaussianSplatSettings;
        }

        std::swap(m_RenderWorldFront, m_RenderWorldBack);
        if (gpuSceneDirty)
        {
            std::swap(m_GpuSceneDatabaseFront, m_GpuSceneDatabaseBack);
            std::swap(m_GpuSceneViewFront, m_GpuSceneViewBack);
        }
        m_RenderWorldFront.gpuSceneDatabase = &m_GpuSceneDatabaseFront;
        m_RenderWorldFront.gpuSceneView     = &m_GpuSceneViewFront;
        m_RenderWorldBack.gpuSceneDatabase  = &m_GpuSceneDatabaseBack;
        m_RenderWorldBack.gpuSceneView      = &m_GpuSceneViewBack;
        m_GpuSceneViewFront.generalGaussianSplatLastAlphaTexture = nullptr;

        m_FrameResources.beginFrame(m_FrameCounter);
        {
            ImmediateResourceUploader frameUploader {m_FrameResources, rd};
            prepareFrameData(frameUploader, m_PreparedFrameData, m_FrameCounter, 0.0f, 0.0f);
        }

        ++m_FrameCounter;

        std::vector<size_t> cameraOrder(cams.size());
        std::iota(cameraOrder.begin(), cameraOrder.end(), 0u);
        std::stable_sort(cameraOrder.begin(), cameraOrder.end(), [&cams](size_t a, size_t b) {
            return cams[a].priority < cams[b].priority;
        });

        const bool supportsMultiview =
            HasFlagValues(rd.getFeatureReport().flags, rhi::RenderDeviceFeatureReportFlagBits::eMultiview);
        const auto xrEyeViews               = backendService.xrEyeViews();
        bool       skipRemainingStereoViews = false;
        bool       backbufferClearedThisFrame = false;
        auto*      gaussianCounterReadbackView = m_RenderWorldFront.gpuSceneView;
        bool       gaussianCounterReadbackPending = false;
        m_RuntimeProfiler.setGpuScopeCpuFallback(rd.getBackendApi() == rhi::RenderBackendApi::eWebGPU);

        m_RuntimeProfiler.setGpuScopeCallbacks(
            [this, &rd, &cb]() {
                if (g_CurrentBuiltinProfilerGpuScopeContext.commandBufferHandle == 0)
                    return uint64_t {0};

                if (rd.getBackendApi() == rhi::RenderBackendApi::eWebGPU)
                {
                    return uint64_t {0};
                }

                // WebGPU compute encoders are kept open lazily. At a framegraph pass boundary the next
                // top-level scope may still observe the previous compute pass as active, which would
                // incorrectly suppress or mis-attribute the new pass timing.
                if (rd.getBackendApi() == rhi::RenderBackendApi::eWebGPU &&
                    g_CurrentBuiltinProfilerGpuScopeContext.renderPassEncoderHandle == 0 &&
                    g_CurrentBuiltinProfilerGpuScopeContext.computePassEncoderHandle != 0 &&
                    m_RuntimeProfiler.gpuScopeDepth() <= 1)
                {
                    rhi::WebGPUCommandBufferAccess::closeActiveComputePassForProfilingBoundary(cb);
                    g_CurrentBuiltinProfilerGpuScopeContext.renderPassEncoderHandle = cb.getCurrentRenderPassEncoderHandle();
                    g_CurrentBuiltinProfilerGpuScopeContext.computePassEncoderHandle =
                        cb.getCurrentComputePassEncoderHandle();
                }

                // WebGPU fallback timestamps are pass-bound; ignore nested scopes inside an active pass.
                if (rd.getBackendApi() == rhi::RenderBackendApi::eWebGPU &&
                    (g_CurrentBuiltinProfilerGpuScopeContext.renderPassEncoderHandle != 0 ||
                     g_CurrentBuiltinProfilerGpuScopeContext.computePassEncoderHandle != 0))
                {
                    return uint64_t {0};
                }
                return rd.beginScopeGpuQuery(g_CurrentBuiltinProfilerGpuScopeContext.commandBufferHandle);
            },
            [&rd](const uint64_t token) {
                if (g_CurrentBuiltinProfilerGpuScopeContext.commandBufferHandle == 0 || token == 0)
                    return;
                if (rd.getBackendApi() == rhi::RenderBackendApi::eWebGPU)
                    return;
                rd.endScopeGpuQuery(g_CurrentBuiltinProfilerGpuScopeContext.commandBufferHandle, token);
            },
            [&rd](const uint64_t token) {
                if (rd.getBackendApi() == rhi::RenderBackendApi::eWebGPU)
                    return -1.0;
                return rd.consumeScopeGpuMs(token);
            });
        rhi::setBuiltinProfilerGpuScopeCallbacks(
            [](const rhi::BuiltinProfilerGpuScopeContext& ctx) { g_CurrentBuiltinProfilerGpuScopeContext = ctx; },
            [this](const rhi::BuiltinProfilerGpuScopeContext& ctx, const char* label) {
                g_CurrentBuiltinProfilerGpuScopeContext = ctx;
                (void)m_RuntimeProfiler.beginGpuScope(label ? label : "GPU Scope");
            },
            [this](const rhi::BuiltinProfilerGpuScopeContext& ctx) {
                g_CurrentBuiltinProfilerGpuScopeContext = ctx;
                m_RuntimeProfiler.endGpuScope();
            });

        // TODO: TimeSystem, for now use 0
        const fsec dt {0};
        static_cast<void>(dt);

        for (const size_t cameraIdx : cameraOrder)
        {
            RuntimeProfiler::Scope scopeCamera {m_RuntimeProfiler, "RenderCamera::execute"};
            const auto& cam = cams[cameraIdx];

            if (skipRemainingStereoViews && cam.isXRView && !cam.isXRPrimaryView)
                continue;
            if (!cam.isXRView || cam.isXRPrimaryView)
                skipRemainingStereoViews = false;

            auto renderer = resolveRenderer(cam);
            if (!renderer)
                continue;

            FrameGraph             fg {};
            FrameGraphBlackboard   bb {};
            FrameGraphDataRegistry dataRegistry {};
            const bool             useFrameGraph = renderer->usesFrameGraph();

            const bool canUseXrMultiview = supportsMultiview && cam.isXRView && cam.isXRPrimaryView &&
                                           cam.viewCount == 2u && m_RenderWorldFront.instances.empty() &&
                                           !xrEyeViews.empty() && xrEyeViews[0].stereoTarget;

            bool      xrGazeValid = false;
            glm::vec2 xrGazeUv {0.5f, 0.5f};
            if (cam.isXRView && !xrEyeViews.empty())
            {
                if (canUseXrMultiview)
                {
                    glm::vec2 gazeUvSum {0.0f};
                    uint32_t  gazeUvCount {0u};
                    for (const auto& eyeView : xrEyeViews)
                    {
                        if (!eyeView.gazeValid)
                            continue;
                        gazeUvSum += eyeView.gazeUv;
                        ++gazeUvCount;
                    }

                    if (gazeUvCount > 0u)
                    {
                        xrGazeValid = true;
                        xrGazeUv =
                            glm::clamp(gazeUvSum / static_cast<float>(gazeUvCount), glm::vec2 {0.0f}, glm::vec2 {1.0f});
                    }
                }
                else if (cam.viewIndex < xrEyeViews.size() && xrEyeViews[cam.viewIndex].gazeValid)
                {
                    xrGazeValid = true;
                    xrGazeUv    = glm::clamp(xrEyeViews[cam.viewIndex].gazeUv, glm::vec2 {0.0f}, glm::vec2 {1.0f});
                }
            }

            rhi::Texture* target =
                canUseXrMultiview ? xrEyeViews[0].stereoTarget : (cam.target ? cam.target : &defaultTarget);
            if (!target)
                continue;

            const bool isBackbufferTarget = !cam.isXRView && cam.target == nullptr && target == &defaultTarget;
            const bool useWindowContentArea =
                isBackbufferTarget && window.platformType() == os::Window::PlatformType::eAndroidNativeWindow;
            const rhi::Rect2D renderArea = useWindowContentArea ?
                                               window.getContentArea() :
                                               rhi::Rect2D {.offset = {0, 0}, .extent = target->getExtent()};

            RenderView view {
                .renderWorld          = &m_RenderWorldFront,
                .camera               = &cam,
                .target               = target,
                .extent               = renderArea.extent,
                .clearValue           = cam.clearValue,
                .enableMultiview      = canUseXrMultiview,
                .multiviewMask        = canUseXrMultiview ? 0x3u : 0u,
                .multiviewCameras     = {&cam, nullptr},
                .multiviewCameraCount = canUseXrMultiview ? 2u : 0u,
                .xrGazeValid          = xrGazeValid,
                .xrGazeUv             = xrGazeUv,
                .gpuSceneDatabase     = m_RenderWorldFront.gpuSceneDatabase,
                .gpuSceneView         = m_RenderWorldFront.gpuSceneView,
            };

            if (canUseXrMultiview)
            {
                const auto secondEyeIt = std::find_if(cameraOrder.begin(), cameraOrder.end(), [&](size_t idx) {
                    return cams[idx].isXRView && !cams[idx].isXRPrimaryView && cams[idx].viewCount == cam.viewCount;
                });
                if (secondEyeIt != cameraOrder.end())
                    view.multiviewCameras[1] = &cams[*secondEyeIt];
            }

            rhi::FramebufferInfo fbInfo {
                .area             = renderArea,
                .layers           = canUseXrMultiview ? 2u : 1u,
                .viewMask         = canUseXrMultiview ? 0x3u : 0u,
                .colorAttachments = {rhi::AttachmentInfo {.target = target, .clearValue = cam.clearValue}},
            };

            ViewRenderData viewData {
                .view            = view,
                .framebufferInfo = fbInfo,
            };

            // Fallback clear for backbuffer cameras.
            // This guarantees a deterministic background even when renderer contributes no color pass
            // (e.g. pure ImGui examples with no framegraph features).
            if (isBackbufferTarget && !backbufferClearedThisFrame)
            {
                clearColorTarget(cb,
                                 *target,
                                 renderArea,
                                 cam.clearValue,
                                 canUseXrMultiview,
                                 0x3u);
                backbufferClearedThisFrame = true;
            }

            {
                ImmediateResourceUploader immediateUploader {m_FrameResources, rd};
                prepareCameraData(immediateUploader, viewData, renderArea.extent, cam, rd.getBackendApi());
            }

            ImmediateRenderContext immediateCtx {
                .cb          = cb,
                .rd          = rd,
                .frame       = m_PreparedFrameData,
                .viewData    = viewData,
                .resourceSet = {},
            };

            rhi::prepareForAttachment(cb, *target, false);
            renderer->render(immediateCtx);
            if (useFrameGraph)
            {
                FrameGraphResourceUploader fgUploader {fg};
                prepareFrameData(fgUploader, m_PreparedFrameData, m_RenderWorldFront.frameIndex, 0.0f, 0.0f);
                prepareCameraData(fgUploader, viewData, renderArea.extent, cam, rd.getBackendApi());
                bb.add<FrameData>(m_PreparedFrameData.frameData);
                bb.add<CameraData>(viewData.cameraData);
            }

            if (useFrameGraph)
            {
                RuntimeProfiler::Scope scopeFrameGraphBuild {m_RuntimeProfiler, "FrameGraph::build"};
                FrameGraphBuildContext buildCtx {
                    .fg       = fg,
                    .bb       = bb,
                    .rd       = rd,
                    .data     = dataRegistry,
                    .frame    = m_PreparedFrameData,
                    .viewData = viewData,
                };

                // This sets up the frame graph using a feature renderer or a custom graph-aware renderer.
                rhi::prepareForAttachment(cb, *target, false);
                renderer->buildFrameGraph(buildCtx);
                fg.compile();

#ifndef NDEBUG
                {
                    const std::filesystem::path debugRoot = !ctx().config.writableRoot.empty() ?
                                                                std::filesystem::path(ctx().config.writableRoot) :
                                                                vbase::executable_dir();
                    const std::filesystem::path debugPath = debugRoot / "framegraph.dot";
                    std::ofstream               ofs(debugPath);
                    if (ofs.is_open())
                    {
                        ofs << fg;
                    }
                    else
                    {
                        VULTRA_CORE_WARN("[RenderSystem] Failed to write framegraph dot file: {}",
                                         debugPath.generic_string());
                    }
                }
#endif

                viewData.framebufferInfo = std::nullopt; // Clear framebuffer info for execution phase, will be set by
                                                         // FrameGraphTexture preRead callback if needed.
                FrameGraphExecContext frameGraphExecCtx {
                    .cb          = cb,
                    .rd          = rd,
                    .frame       = m_PreparedFrameData,
                    .viewData    = viewData,
                    .resourceSet = {},
                    .ext         = {.builtinShaderLib = &shaderService.builtinLibrary(), .samplers = m_Samplers},
                };

                {
                    RuntimeProfiler::Scope scopeFrameGraphExec {m_RuntimeProfiler, "FrameGraph::execute"};
                    FG_GPU_ZONE(cb);
                    fg.execute(&frameGraphExecCtx, m_TransientResources.get());
                }
            }

            if (canUseXrMultiview)
                skipRemainingStereoViews = true;

            // Optional ImGui rendering per non-XR camera
            if (imguiService && !cam.isXRView && cam.renderImGui)
            {
                imguiService->begin();
                renderer->onImGui();
                imguiService->end();

                rhi::prepareForAttachment(cb, *target, false);
                imguiService->render(cb, fbInfo);
            }
        }

        if (imguiService && backendService.isXREnabled() && backendService.isXRMirrorEnabled() && !xrEyeViews.empty())
        {
            for (const auto& eyeView : xrEyeViews)
            {
                if (!eyeView.target || !eyeView.mirrorTarget)
                    continue;

                rhi::prepareForReading(cb, *eyeView.target);
                cb.blit(*eyeView.target, *eyeView.mirrorTarget, rhi::TexelFilter::eLinear);
            }

            imguiService->begin();

            std::unordered_set<Renderer*> imguiRenderers;
            for (const size_t cameraIdx : cameraOrder)
            {
                auto renderer = resolveRenderer(cams[cameraIdx]);
                if (!renderer || imguiRenderers.contains(renderer.get()))
                    continue;
                imguiRenderers.insert(renderer.get());
                renderer->onImGui();
            }

            imguiService->end();

            for (const auto& xrEyeView : xrEyeViews)
            {
                if (xrEyeView.mirrorTarget)
                    rhi::prepareForReading(cb, *xrEyeView.mirrorTarget);
            }

            const rhi::Rect2D imguiArea = window.platformType() == os::Window::PlatformType::eAndroidNativeWindow ?
                                              window.getContentArea() :
                                              rhi::Rect2D {.offset = {0, 0}, .extent = defaultTarget.getExtent()};

            rhi::FramebufferInfo imguiFbInfo {
                .area             = imguiArea,
                .colorAttachments = {rhi::AttachmentInfo {.target = &defaultTarget}},
            };

            rhi::prepareForAttachment(cb, defaultTarget, false);
            imguiService->render(cb, imguiFbInfo);
        }

        // Stop issuing begin/end scope queries after rendering submission building is done,
        // but keep resolve callback alive so endFrame can harvest ready GPU samples.
        m_RuntimeProfiler.setGpuScopeCallbacks(
            []() { return uint64_t {0}; },
            [](const uint64_t) {},
            [&rd](const uint64_t token) { return rd.consumeScopeGpuMs(token); });

        if (gaussianCounterReadbackView &&
            gaussianCounterReadbackView->generalGaussianSplatEcsptCounterBuffer &&
            gaussianCounterReadbackView->generalGaussianSplatEcsptCounterReadbackBuffer &&
            gaussianCounterReadbackView->generalGaussianSplatEcsptCounterReadbackEnabled)
        {
            constexpr uint64_t kEcsptCounterReadbackBytes = sizeof(uint32_t) * 4096ull;
            cb.insertBufferBarrier(*gaussianCounterReadbackView->generalGaussianSplatEcsptCounterBuffer,
                                   {
                                       .srcStage  = rhi::PipelineStages::eComputeShader,
                                       .srcAccess = rhi::Access::eShaderWrite,
                                   },
                                   {
                                       .dstStage  = rhi::PipelineStages::eTransfer,
                                       .dstAccess = rhi::Access::eTransferRead,
                                   },
                                   0u,
                                   kEcsptCounterReadbackBytes);
            cb.copyBuffer(*gaussianCounterReadbackView->generalGaussianSplatEcsptCounterBuffer,
                          *gaussianCounterReadbackView->generalGaussianSplatEcsptCounterReadbackBuffer,
                          rhi::BufferCopy {0u, 0u, kEcsptCounterReadbackBytes});
            gaussianCounterReadbackPending = true;
        }

        m_TransientResources->update();
        rd.endFrameGpuQuery(cb);
        backendService.endFrame();

        if (gaussianCounterReadbackPending &&
            gaussianCounterReadbackView &&
            gaussianCounterReadbackView->generalGaussianSplatEcsptCounterReadbackBuffer)
        {
            const auto readbackStart = std::chrono::steady_clock::now();
            rd.waitIdle();
            const auto* counters = static_cast<const uint32_t*>(
                gaussianCounterReadbackView->generalGaussianSplatEcsptCounterReadbackBuffer->map());
            if (counters)
            {
                auto updatedStats = m_GaussianSplatStats;
                updatedStats.eccStochasticCounterReadbackValid = true;
                updatedStats.eccStochasticCounterReadbackCpuMs = elapsedCpuMs(readbackStart);
                updatedStats.eccStochasticTotalCandidatesSeen = counters[0];
                updatedStats.eccStochasticBaseNewSelectedCount = counters[1];
                updatedStats.eccStochasticEffectiveVisibleAfterEcsptCount = counters[2];
                updatedStats.eccStochasticSharedCount = counters[3];
                updatedStats.eccStochasticUpgradeCount = counters[4];
                updatedStats.eccStochasticDowngradeCount = counters[5];
                updatedStats.eccStochasticProtectedDowngradeCount = counters[6];
                updatedStats.eccStochasticDroppedDowngradeCount = counters[7];
                updatedStats.eccStochasticOldOnlyFadeVisibleCount = counters[8];
                updatedStats.eccStochasticImmediateNewFoveaCount = counters[9];
                updatedStats.eccStochasticBoundaryProtectedCount = counters[10];
                updatedStats.eccStochasticContributionGuardProtectedCount = counters[11];
                updatedStats.eccStochasticZeroWeightDiscardCount = counters[12];
                updatedStats.eccStochasticMinPDeltaDiscardCount = counters[13];
                updatedStats.eccStochasticFarPeripheryHardDropCount = counters[14];
                updatedStats.stableOpticalDepthAlphaClampedCount = counters[15];
                updatedStats.coverageStableReleaseFloorActiveCount = counters[16];
                updatedStats.coverageStableReleaseFloorRaisedCount = counters[17];
                updatedStats.coverageStableReleaseHeldCount = counters[18];
                updatedStats.coverageStableReleaseDroppedCount = counters[19];
                updatedStats.coverageStableReleaseOldOnlyCount = counters[20];
                updatedStats.coverageStableReleaseFloorSampleCount = counters[25];
                const float coverageScale = 1024.0f;
                const float floorSamples =
                    static_cast<float>(std::max(updatedStats.coverageStableReleaseFloorSampleCount, 1u));
                updatedStats.coverageStableReleasePFloorMean =
                    static_cast<float>(counters[21]) / coverageScale / floorSamples;
                updatedStats.coverageStableReleasePFloorMax =
                    static_cast<float>(counters[22]) / coverageScale;
                updatedStats.coverageStableReleaseDAlphaMean =
                    static_cast<float>(counters[23]) / coverageScale / floorSamples;
                updatedStats.coverageStableReleaseNEffMean =
                    static_cast<float>(counters[24]) / coverageScale / floorSamples;
                updatedStats.coverageStableReleasePLpMean =
                    static_cast<float>(counters[26]) / coverageScale / floorSamples;
                updatedStats.coverageStableReleasePStaticMean =
                    static_cast<float>(counters[27]) / coverageScale / floorSamples;
                updatedStats.coverageStableReleaseDeltaSum =
                    static_cast<float>(counters[28]) / coverageScale;
                updatedStats.coverageStableReleaseHistoryResetCount = counters[29];
                updatedStats.coverageStableReleaseReentryResetCount = counters[30];
                updatedStats.coverageStableReleaseHeldEffectiveCount = counters[31];
                updatedStats.coverageStableReleaseEpsilonCutoffCount = counters[32];
                updatedStats.coverageStableReleaseSkippedFloorUpdateCount = counters[33];
                updatedStats.coverageStableReleaseCandidateAlphaMass =
                    static_cast<float>(counters[34]) / coverageScale;
                updatedStats.coverageStableReleaseRetainedAlphaMass =
                    static_cast<float>(counters[35]) / coverageScale;
                updatedStats.visibleInstant = counters[36];
                updatedStats.tileInstances = counters[37];
                updatedStats.coveredPixelCount = counters[38] == 0u ? UINT32_MAX : counters[38];
                updatedStats.coverageTextureCurrentAlphaMass =
                    static_cast<float>(counters[39]) / coverageScale;
                updatedStats.coverageTextureCurrentAlpha2Mass =
                    static_cast<float>(counters[40]) / coverageScale;
                updatedStats.coverageTextureNonzeroTexelCount = counters[41];
                updatedStats.coverageTextureWidth = counters[42];
                updatedStats.coverageTextureHeight = counters[43];
                updatedStats.coverageTextureHistoryBeta =
                    gaussianCounterReadbackView->generalGaussianSplatCoverageTextureHistoryBeta;
                updatedStats.coverageTextureStrength =
                    gaussianCounterReadbackView->generalGaussianSplatCoverageTextureStrength;
                updatedStats.coverageTexturePFinalMean =
                    static_cast<float>(counters[45]) / coverageScale / floorSamples;
                updatedStats.coverageTextureStableHashKeptCount = counters[54];
                updatedStats.guideBeforeDiscardEnabled =
                    m_GaussianSplatSettings.guideBeforeDiscardEnabled;
                updatedStats.delayedGuideLogPolarEnabled =
                    m_GaussianSplatSettings.delayedGuideLogPolarEnabled;
                updatedStats.delayedGuideTemporalLogPolarEnabled =
                    m_GaussianSplatSettings.delayedGuideTemporalLogPolarEnabled;
                updatedStats.delayedGuideReleaseActiveCount =
                    updatedStats.delayedGuideTemporalLogPolarEnabled ? counters[18] : 0u;
                updatedStats.delayedGuidePHistoryNonzeroCount =
                    updatedStats.delayedGuideTemporalLogPolarEnabled ? counters[57] : 0u;
                updatedStats.delayedGuidePTargetLessThanHistoryCount =
                    updatedStats.delayedGuideTemporalLogPolarEnabled ? counters[58] : 0u;
                updatedStats.delayedGuidePHistoryBytes =
                    updatedStats.delayedGuideTemporalLogPolarEnabled ?
                        static_cast<uint64_t>(maxGeneralGaussianSplatSourceCount) * sizeof(uint32_t) :
                        0u;
                updatedStats.delayedGuideReleasedEffectiveVisibleCount =
                    updatedStats.delayedGuideTemporalLogPolarEnabled ? counters[59] : 0u;
                updatedStats.delayedGuideReleasedAlphaProxySum =
                    updatedStats.delayedGuideTemporalLogPolarEnabled ?
                        static_cast<float>(counters[60]) / coverageScale :
                        0.0f;
                updatedStats.delayedGuideReleaseCapHitCount =
                    updatedStats.delayedGuideTemporalLogPolarEnabled ? counters[61] : 0u;
                updatedStats.delayedGuideRiskProtectedCount =
                    updatedStats.delayedGuideTemporalLogPolarEnabled ? counters[62] : 0u;
                updatedStats.delayedGuideFootprintFastDecayCount =
                    updatedStats.delayedGuideTemporalLogPolarEnabled ? counters[63] : 0u;
                updatedStats.guideBeforeDiscardLowCoverageBoostedSplats = counters[56];
                updatedStats.guideBeforeDiscardHighCoverageReducedSplats = counters[55];
                updatedStats.guideBeforeDiscardSelectedBeforeTileExpansion = counters[54];
                updatedStats.guideBeforeDiscardTileDuplicateCount =
                    counters[37] == 0u ? UINT32_MAX : counters[37];
                const float candidates =
                    static_cast<float>(std::max(updatedStats.eccStochasticTotalCandidatesSeen, 1u));
                const float effectiveVisible =
                    static_cast<float>(std::max(updatedStats.eccStochasticEffectiveVisibleAfterEcsptCount, 1u));
                const float releaseHeld =
                    static_cast<float>(std::max(updatedStats.coverageStableReleaseHeldCount, 1u));
                updatedStats.coverageStableReleaseFloorActiveFractionVsCandidates =
                    static_cast<float>(updatedStats.coverageStableReleaseFloorActiveCount) / candidates;
                updatedStats.coverageStableReleaseFloorRaisedFractionVsEffectiveVisible =
                    static_cast<float>(updatedStats.coverageStableReleaseFloorRaisedCount) / effectiveVisible;
                updatedStats.coverageStableReleaseDeltaMeanOverCandidates =
                    updatedStats.coverageStableReleaseDeltaSum / candidates;
                updatedStats.coverageStableReleaseDeltaMeanOverHeld =
                    updatedStats.coverageStableReleaseDeltaSum / releaseHeld;
                updatedStats.coverageStableReleaseHeldEffectiveRatio =
                    static_cast<float>(updatedStats.coverageStableReleaseHeldEffectiveCount) / releaseHeld;

                constexpr uint32_t kCoverageReleaseTileCounterOffset = 256u;
                constexpr uint32_t kCoverageReleaseMaxTileCount = 512u;
                constexpr float    kCoverageReleaseTileCounterScale = 2048.0f;
                constexpr uint32_t kCoverageReleaseTileSaturatedOffset =
                    kCoverageReleaseTileCounterOffset + kCoverageReleaseMaxTileCount * 2u;
                constexpr uint32_t kReleaseCostTileBaseDuplicateOffset =
                    kCoverageReleaseTileCounterOffset + kCoverageReleaseMaxTileCount * 3u;
                constexpr uint32_t kReleaseCostTileReleasedDuplicateOffset =
                    kCoverageReleaseTileCounterOffset + kCoverageReleaseMaxTileCount * 4u;
                constexpr uint32_t kReleaseCostSummaryOffset =
                    kCoverageReleaseTileCounterOffset + kCoverageReleaseMaxTileCount * 5u;
                constexpr uint32_t kReleaseCostBaseDuplicateCounter =
                    kReleaseCostSummaryOffset + 0u;
                constexpr uint32_t kReleaseCostReleasedDuplicateCounter =
                    kReleaseCostSummaryOffset + 1u;
                constexpr uint32_t kReleaseCostLargeReleasedCounter =
                    kReleaseCostSummaryOffset + 2u;
                constexpr uint32_t kReleaseCostLargeReleasedDuplicateCounter =
                    kReleaseCostSummaryOffset + 3u;
                constexpr uint32_t kReleaseCostReleasedRadiusSumCounter =
                    kReleaseCostSummaryOffset + 4u;
                constexpr uint32_t kReleaseCostReleasedRadiusMaxCounter =
                    kReleaseCostSummaryOffset + 5u;
                constexpr uint32_t kReleaseCostReleasedRadiusHistOffset =
                    kReleaseCostSummaryOffset + 8u;
                constexpr uint32_t kReleaseCostReleasedRadiusHistCount = 8u;
                constexpr float    kReleaseCostRadiusCounterScale = 4.0f;
                const bool releaseCostDiagnosticsActive =
                    updatedStats.delayedGuideLogPolarEnabled ||
                    updatedStats.delayedGuideTemporalLogPolarEnabled;
                updatedStats.delayedGuideBaseTileDuplicateProxyCount =
                    releaseCostDiagnosticsActive ? counters[kReleaseCostBaseDuplicateCounter] : 0u;
                updatedStats.delayedGuideReleasedTileDuplicateProxyCount =
                    releaseCostDiagnosticsActive ? counters[kReleaseCostReleasedDuplicateCounter] : 0u;
                const uint64_t releaseCostDuplicateProxyTotal =
                    static_cast<uint64_t>(updatedStats.delayedGuideBaseTileDuplicateProxyCount) +
                    static_cast<uint64_t>(updatedStats.delayedGuideReleasedTileDuplicateProxyCount);
                updatedStats.delayedGuideReleasedDuplicateProxyRatio =
                    releaseCostDuplicateProxyTotal > 0u ?
                        static_cast<float>(updatedStats.delayedGuideReleasedTileDuplicateProxyCount) /
                            static_cast<float>(releaseCostDuplicateProxyTotal) :
                        0.0f;
                updatedStats.delayedGuideLargeFootprintReleasedCount =
                    releaseCostDiagnosticsActive ? counters[kReleaseCostLargeReleasedCounter] : 0u;
                updatedStats.delayedGuideLargeFootprintReleasedTileDuplicateProxyCount =
                    releaseCostDiagnosticsActive ? counters[kReleaseCostLargeReleasedDuplicateCounter] : 0u;
                const uint32_t releaseRadiusSampleCount =
                    std::max(updatedStats.delayedGuideReleasedEffectiveVisibleCount, 1u);
                updatedStats.delayedGuideReleasedRadiusMean =
                    releaseCostDiagnosticsActive ?
                        (static_cast<float>(counters[kReleaseCostReleasedRadiusSumCounter]) /
                         kReleaseCostRadiusCounterScale) /
                            static_cast<float>(releaseRadiusSampleCount) :
                        0.0f;
                updatedStats.delayedGuideReleasedRadiusMax =
                    releaseCostDiagnosticsActive ?
                        static_cast<float>(counters[kReleaseCostReleasedRadiusMaxCounter]) /
                            kReleaseCostRadiusCounterScale :
                        0.0f;
                updatedStats.delayedGuideReleasedRadiusP95 = 0.0f;
                if (releaseCostDiagnosticsActive &&
                    updatedStats.delayedGuideReleasedEffectiveVisibleCount > 0u)
                {
                    constexpr std::array<float, kReleaseCostReleasedRadiusHistCount> kRadiusHistUpperBounds {
                        4.0f,
                        8.0f,
                        16.0f,
                        32.0f,
                        48.0f,
                        64.0f,
                        96.0f,
                        128.0f};
                    const uint32_t p95Target = std::max(
                        1u,
                        static_cast<uint32_t>(
                            std::ceil(static_cast<float>(
                                          updatedStats.delayedGuideReleasedEffectiveVisibleCount) *
                                      0.95f)));
                    uint32_t cumulativeRadiusSamples = 0u;
                    updatedStats.delayedGuideReleasedRadiusP95 =
                        updatedStats.delayedGuideReleasedRadiusMax;
                    for (uint32_t i = 0u; i < kReleaseCostReleasedRadiusHistCount; ++i)
                    {
                        cumulativeRadiusSamples +=
                            counters[kReleaseCostReleasedRadiusHistOffset + i];
                        if (cumulativeRadiusSamples >= p95Target)
                        {
                            updatedStats.delayedGuideReleasedRadiusP95 =
                                kRadiusHistUpperBounds[i];
                            break;
                        }
                    }
                }
                struct ReleaseCostHotspot
                {
                    uint32_t tileId {0u};
                    uint32_t baseDuplicates {0u};
                    uint32_t releasedDuplicates {0u};
                    uint64_t totalDuplicates {0u};
                };
                std::vector<ReleaseCostHotspot> releaseCostHotspots;
                releaseCostHotspots.reserve(kCoverageReleaseMaxTileCount);
                for (uint32_t i = 0u; i < kCoverageReleaseMaxTileCount; ++i)
                {
                    const uint32_t baseDuplicates =
                        counters[kReleaseCostTileBaseDuplicateOffset + i];
                    const uint32_t releasedDuplicates =
                        counters[kReleaseCostTileReleasedDuplicateOffset + i];
                    const uint64_t totalDuplicates =
                        static_cast<uint64_t>(baseDuplicates) +
                        static_cast<uint64_t>(releasedDuplicates);
                    if (totalDuplicates == 0u)
                        continue;
                    releaseCostHotspots.push_back(
                        ReleaseCostHotspot {i, baseDuplicates, releasedDuplicates, totalDuplicates});
                }
                std::sort(releaseCostHotspots.begin(),
                          releaseCostHotspots.end(),
                          [](const ReleaseCostHotspot& lhs, const ReleaseCostHotspot& rhs) {
                              if (lhs.totalDuplicates != rhs.totalDuplicates)
                                  return lhs.totalDuplicates > rhs.totalDuplicates;
                              return lhs.releasedDuplicates > rhs.releasedDuplicates;
                          });
                updatedStats.delayedGuideHotspotTileIds.fill(0u);
                updatedStats.delayedGuideHotspotBaseDuplicates.fill(0u);
                updatedStats.delayedGuideHotspotReleasedDuplicates.fill(0u);
                for (uint32_t i = 0u;
                     i < kGaussianSplatReleaseCostHotspotCount &&
                     i < releaseCostHotspots.size();
                     ++i)
                {
                    updatedStats.delayedGuideHotspotTileIds[i] =
                        releaseCostHotspots[i].tileId;
                    updatedStats.delayedGuideHotspotBaseDuplicates[i] =
                        releaseCostHotspots[i].baseDuplicates;
                    updatedStats.delayedGuideHotspotReleasedDuplicates[i] =
                        releaseCostHotspots[i].releasedDuplicates;
                }
                updatedStats.coverageStableReleaseSaturatedFloorEnabled =
                    m_GaussianSplatSettings.coverageStableReleaseSaturatedFloorEnabled;
                updatedStats.coverageStableReleaseFloorCellsSafeSaturated = 0u;
                updatedStats.coverageStableReleaseFloorCellsUnsafe = 0u;
                updatedStats.coverageStableReleaseFloorSaturationRatio = 0.0f;
                const bool coverageTextureFloorActive =
                    (gaussianCounterReadbackView->generalGaussianSplatCoverageStableReleaseFlags & 32u) != 0u;
                const uint32_t tileGridX =
                    coverageTextureFloorActive ?
                        1u :
                        std::clamp(gaussianCounterReadbackView->generalGaussianSplatCoverageStableReleaseTileGridX,
                                   1u,
                                   32u);
                const uint32_t tileGridY =
                    coverageTextureFloorActive ?
                        1u :
                        std::clamp(gaussianCounterReadbackView->generalGaussianSplatCoverageStableReleaseTileGridY,
                                   1u,
                                   16u);
                const uint32_t tileCount =
                    coverageTextureFloorActive ? 0u : std::min(tileGridX * tileGridY, kCoverageReleaseMaxTileCount);
                std::vector<float> dAlphaAll;
                std::vector<float> dAlphaNonEmpty;
                std::vector<float> nEffNonEmpty;
                std::vector<float> pFloorAll;
                std::vector<float> pFloorNonEmpty;
                dAlphaAll.reserve(tileCount);
                dAlphaNonEmpty.reserve(tileCount);
                nEffNonEmpty.reserve(tileCount);
                pFloorAll.reserve(tileCount);
                pFloorNonEmpty.reserve(tileCount);
                float dAlphaSumAll = 0.0f;
                float dAlphaSumNonEmpty = 0.0f;
                float nEffSumNonEmpty = 0.0f;
                float pFloorSumAll = 0.0f;
                float pFloorSumNonEmpty = 0.0f;
                const float dMin = std::max(m_GaussianSplatSettings.coverageStableReleaseDMin, 0.0f);
                const float sigmaMax =
                    std::max(m_GaussianSplatSettings.coverageStableReleaseSigmaMax, 0.0f);
                const auto curveKeepProbability = [](const float keepProbability,
                                                     const GaussianSplatShaderAntiPopPKeepCurve curve) {
                    const float p = std::clamp(keepProbability, 0.0f, 1.0f);
                    if (curve == GaussianSplatShaderAntiPopPKeepCurve::eSmoothWide)
                    {
                        const float centered = std::clamp((p - 0.5f) * 2.0f, -1.0f, 1.0f);
                        const float shaped = std::pow(std::abs(centered), 1.6f);
                        return std::clamp(0.5f + 0.5f * (centered < 0.0f ? -shaped : shaped), 0.0f, 1.0f);
                    }
                    const auto normalizedLogistic = [](const float x, const float slope) {
                        const float lo = 1.0f / (1.0f + std::exp(0.5f * slope));
                        const float hi = 1.0f / (1.0f + std::exp(-0.5f * slope));
                        const float y = 1.0f / (1.0f + std::exp(-slope * (std::clamp(x, 0.0f, 1.0f) - 0.5f)));
                        return std::clamp((y - lo) / std::max(hi - lo, 1e-5f), 0.0f, 1.0f);
                    };
                    if (curve == GaussianSplatShaderAntiPopPKeepCurve::eLogisticSoft)
                        return normalizedLogistic(p, 4.0f);
                    if (curve == GaussianSplatShaderAntiPopPKeepCurve::eLogisticSteep)
                        return normalizedLogistic(p, 10.0f);
                    return p;
                };
                const float rMin = std::max(
                    curveKeepProbability(m_GaussianSplatSettings.foveatedRingLevels.z,
                                         m_GaussianSplatSettings.shaderAntiPopPKeepCurve),
                    1e-4f);
                updatedStats.coverageStableReleaseDAlphaSafeThreshold =
                    dMin > 0.0f ? dMin / rMin : 0.0f;
                updatedStats.coverageStableReleaseNEffSafeThreshold =
                    sigmaMax > 1e-6f ? (1.0f / rMin - 1.0f) / (sigmaMax * sigmaMax) : 0.0f;
                for (uint32_t i = 0u; i < tileCount; ++i)
                {
                    const bool saturated =
                        counters[kCoverageReleaseTileSaturatedOffset + i] != 0u;
                    if (saturated)
                        ++updatedStats.coverageStableReleaseFloorCellsSafeSaturated;
                    const float dAlpha =
                        static_cast<float>(counters[kCoverageReleaseTileCounterOffset + i]) /
                        kCoverageReleaseTileCounterScale;
                    const float alpha2 =
                        static_cast<float>(counters[kCoverageReleaseTileCounterOffset +
                                                    kCoverageReleaseMaxTileCount + i]) /
                        kCoverageReleaseTileCounterScale;
                    float nEff = 0.0f;
                    float pFloor = 0.0f;
                    const bool nonEmpty = dAlpha > 1e-6f;
                    if (nonEmpty)
                    {
                        nEff = alpha2 > 1e-6f ? std::max(1.0f, (dAlpha * dAlpha) / alpha2) : 1.0f;
                        if (!saturated)
                        {
                            const float pMean = dMin > 0.0f ? std::min(1.0f, dMin / dAlpha) : 0.0f;
                            const float pVar = 1.0f / (1.0f + sigmaMax * sigmaMax * nEff);
                            pFloor = std::clamp(std::max(pMean, pVar), 0.0f, 1.0f);
                        }
                    }
                    dAlphaAll.push_back(dAlpha);
                    pFloorAll.push_back(pFloor);
                    dAlphaSumAll += dAlpha;
                    pFloorSumAll += pFloor;
                    if (nonEmpty)
                    {
                        dAlphaNonEmpty.push_back(dAlpha);
                        nEffNonEmpty.push_back(nEff);
                        pFloorNonEmpty.push_back(pFloor);
                        dAlphaSumNonEmpty += dAlpha;
                        nEffSumNonEmpty += nEff;
                        pFloorSumNonEmpty += pFloor;
                    }
                }
                const auto percentile = [](std::vector<float> values, const float pct) {
                    if (values.empty())
                        return 0.0f;
                    std::sort(values.begin(), values.end());
                    if (values.size() == 1u)
                        return values.front();
                    const float position = (static_cast<float>(values.size() - 1u)) * pct;
                    const auto  lo = static_cast<size_t>(std::floor(position));
                    const auto  hi = static_cast<size_t>(std::ceil(position));
                    if (lo == hi)
                        return values[lo];
                    const float t = position - static_cast<float>(lo);
                    return values[lo] * (1.0f - t) + values[hi] * t;
                };
                const uint32_t nonEmptyCount = static_cast<uint32_t>(dAlphaNonEmpty.size());
                const uint32_t emptyCount = tileCount >= nonEmptyCount ? tileCount - nonEmptyCount : 0u;
                const float tileCountFloat = static_cast<float>(std::max(tileCount, 1u));
                const float nonEmptyCountFloat = static_cast<float>(std::max(nonEmptyCount, 1u));
                updatedStats.coverageStableReleaseTileTotalCount = tileCount;
                updatedStats.coverageStableReleaseTileNonEmptyCount = nonEmptyCount;
                updatedStats.coverageStableReleaseTileEmptyCount = emptyCount;
                updatedStats.coverageStableReleaseFloorCellsUnsafe =
                    tileCount >= updatedStats.coverageStableReleaseFloorCellsSafeSaturated ?
                        tileCount - updatedStats.coverageStableReleaseFloorCellsSafeSaturated :
                        0u;
                updatedStats.coverageStableReleaseFloorSaturationRatio =
                    static_cast<float>(updatedStats.coverageStableReleaseFloorCellsSafeSaturated) /
                    tileCountFloat;
                updatedStats.coverageStableReleaseDAlphaMeanAll = dAlphaSumAll / tileCountFloat;
                updatedStats.guideBeforeDiscardTexelCount =
                    coverageTextureFloorActive ?
                        std::clamp(gaussianCounterReadbackView->generalGaussianSplatCoverageStableReleaseTileGridX,
                                   1u,
                                   1024u) *
                            std::clamp(gaussianCounterReadbackView->generalGaussianSplatCoverageStableReleaseTileGridY,
                                       1u,
                                       1024u) :
                        tileCount;
                updatedStats.guideBeforeDiscardCoverageMean =
                    updatedStats.coverageStableReleaseDAlphaMeanAll;
                updatedStats.guideBeforeDiscardCoverageMin =
                    dAlphaAll.empty() ? 0.0f : *std::min_element(dAlphaAll.begin(), dAlphaAll.end());
                updatedStats.guideBeforeDiscardCoverageMax =
                    dAlphaAll.empty() ? 0.0f : *std::max_element(dAlphaAll.begin(), dAlphaAll.end());
                updatedStats.coverageStableReleaseDAlphaMeanNonEmpty =
                    dAlphaSumNonEmpty / nonEmptyCountFloat;
                updatedStats.coverageStableReleaseDAlphaP50NonEmpty =
                    percentile(dAlphaNonEmpty, 0.50f);
                updatedStats.coverageStableReleaseDAlphaP90NonEmpty =
                    percentile(dAlphaNonEmpty, 0.90f);
                updatedStats.coverageStableReleaseDAlphaP95NonEmpty =
                    percentile(dAlphaNonEmpty, 0.95f);
                updatedStats.coverageStableReleaseNEffMeanNonEmpty =
                    nEffSumNonEmpty / nonEmptyCountFloat;
                updatedStats.coverageStableReleaseNEffP50NonEmpty =
                    percentile(nEffNonEmpty, 0.50f);
                updatedStats.coverageStableReleaseNEffP90NonEmpty =
                    percentile(nEffNonEmpty, 0.90f);
                updatedStats.coverageStableReleaseNEffP95NonEmpty =
                    percentile(nEffNonEmpty, 0.95f);
                updatedStats.coverageStableReleasePFloorMeanAll = pFloorSumAll / tileCountFloat;
                updatedStats.coverageStableReleasePFloorMeanNonEmpty =
                    pFloorSumNonEmpty / nonEmptyCountFloat;
                updatedStats.coverageStableReleasePFloorP90NonEmpty =
                    percentile(pFloorNonEmpty, 0.90f);
                updatedStats.coverageStableReleasePFloorP95NonEmpty =
                    percentile(pFloorNonEmpty, 0.95f);
                for (uint32_t i = 0u; i < kGaussianSplatEcsptEventBinCounterCount; ++i)
                    updatedStats.eccStochasticEventBinCounters[i] =
                        counters[kGaussianSplatEcsptScalarCounterCount + i];
                updatedStats.visibleSplats = counters[2];
                updatedStats.drawnSplats = counters[2];
                m_GaussianSplatStats = updatedStats;
                gaussianCounterReadbackView->generalGaussianSplatEcsptCounterReadbackBuffer->unmap();
            }
        }

        const auto commandStats = rhi::CommandBuffer::consumeFrameStats();
        m_RuntimeProfiler.setCommandStats(commandStats.drawCalls,
                                          commandStats.dispatchCalls,
                                          commandStats.traceRaysCalls,
                                          commandStats.copyOps,
                                          commandStats.updateOps);
        const auto assetMemoryStats = assetService.memoryStats();
        const auto memoryStats      = rd.getMemoryStats();
        m_RuntimeProfiler.setMemoryStats(assetMemoryStats.cpuCacheBytes,
                                          memoryStats.cpuCacheBytes,
                                          memoryStats.gpuDeviceLocalBytes,
                                          memoryStats.gpuHostVisibleBytes);
        const double gpuFrameMs = rd.consumeGpuFrameMs();
        m_RuntimeProfiler.setGpuFrameMs(gpuFrameMs);
        const auto renderFrameCpuEnd = std::chrono::steady_clock::now();
        m_RuntimeProfiler.setCpuRenderMs(
            std::chrono::duration<double, std::milli>(renderFrameCpuEnd - renderFrameCpuStart).count());
        m_RuntimeProfiler.endFrame();
        updateGaussianSplatFoveatedAdaptation(m_GaussianSplatSettings,
                                               m_FoveatedP95ControllerState,
                                               gpuFrameMs);
        rhi::setBuiltinProfilerGpuScopeCallbacks({}, {});
        m_RuntimeProfiler.setGpuScopeCallbacks({}, {}, {});

        if (frameDebuggerService)
            frameDebuggerService->captureEnd();
    }

    void RenderSystem::onPreRender() { m_SkipRender = false; }

    void RenderSystem::onRender() { renderFrame(); }

    void RenderSystem::onPostRender()
    {
        auto* imguiService = ctx().services.tryGet<IImGuiService>();
        if (imguiService)
            imguiService->postRender();
    }

    void RenderSystem::onPresent()
    {
        if (m_SkipRender)
            return;

        auto& backendService = ctx().services.require<IRenderBackendService>();
        backendService.present();
    }
} // namespace vultra

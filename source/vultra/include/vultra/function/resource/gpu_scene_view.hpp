#pragma once

#include "vultra/core/base/base.hpp"
#include "vultra/core/rhi/draw_indirect_buffer.hpp"
#include "vultra/core/rhi/structs/draw_indirect_command.hpp"
#include "vultra/core/rhi/structs/draw_indirect_type.hpp"
#include "vultra/core/rhi/structs/buffer_usage.hpp"
#include "vultra/core/rhi/radix_sorter.hpp"
#include "vultra/core/rhi/render_device.hpp"
#include "vultra/core/rhi/storage_buffer.hpp"
#include "vultra/function/resource/gpu_draw.hpp"
#include "vultra/function/resource/gpu_scene_database.hpp"
#include "vultra/function/resource/gpu_visible_meshlet.hpp"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/common.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace vultra::rhi
{
    class CommandBuffer;
}

namespace vultra::resource
{
    inline constexpr uint32_t kGeneralGaussianSplatFoveatedLayerCount = 3u;

    enum class GpuSceneBuildMode : uint8_t
    {
        eCpuDriven,
        eGpuDriven,
    };

    // Per-view / per-frame GPU scene state.
    //
    // Responsibilities:
    // - CPU-driven fallback staging for meshlet draws
    // - GPU-driven transient buffers (meshlet cull/output + per-primitive draw/indirect targets)
    // - Reference to a GpuSceneDatabase that owns scene-level tables/resources
    struct GpuSceneView
    {
        const GpuSceneDatabase* database {nullptr};
        GpuSceneBuildMode       mode {GpuSceneBuildMode::eCpuDriven};

        // CPU staging (fallback/debug/per-frame)
        // Meshlet path
        std::vector<GpuDrawRecord>            draws;
        std::vector<rhi::DrawIndirectCommand> indirectCommands;

        // Unified eGeneral gaussian splat path
        std::vector<GpuGeneralGaussianSplatDrawRecord>  generalGaussianSplatDraws;
        std::vector<GpuGeneralGaussianSplatPackedSource> generalGaussianSplatPackedSources;
        std::vector<GpuGeneralGaussianSplatSelectedSource> generalGaussianSplatSelectedSources;

        // Optional CPU mirror for GPU-driven intermediate visibility.
        std::vector<GpuVisibleMeshlet> visibleMeshlets;

        // GPU buffers (per-view/per-frame)
        // Meshlet path
        Ref<rhi::StorageBuffer>                visibleInstanceBuffer {nullptr};
        Ref<rhi::StorageBuffer>                visibleInstanceCountBuffer {nullptr};
        Ref<rhi::StorageBuffer>                meshletCullDispatchArgsBuffer {nullptr};
        Ref<rhi::StorageBuffer>                visibleMeshletBuffer {nullptr};
        Ref<rhi::DrawIndirectBuffer>           visibleMeshletCountBuffer {nullptr};
        Ref<rhi::StorageBuffer>                drawBuffer {nullptr};
        Ref<rhi::DrawIndirectBuffer>           drawSetBuffer {nullptr};
        std::optional<rhi::DrawIndirectBuffer> indirectBuffer;

        // General gaussian splat path
        Ref<rhi::StorageBuffer>                generalGaussianSplatDrawBuffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatPackedSourceBuffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatSelectedSourceBuffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatVisibleSplatBuffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatSortKeyBuffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatSortIndexBuffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatVisibleCountBuffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatEcsptCounterBuffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatEcsptCounterReadbackBuffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatDispatchArgsBuffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatSortStorageBuffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatShBuffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatShL1Buffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatShL2Buffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatShL3Buffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatShEnergyMetadataBuffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatTemporalStateBuffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatCoverageTextureBuffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatProjectedCostChunkBuffer {nullptr};
        Ref<rhi::StorageBuffer>                generalGaussianSplatProjectedCostReadbackBuffer {nullptr};
        rhi::Texture*                          generalGaussianSplatLastAlphaTexture {nullptr};
        std::optional<rhi::DrawIndirectBuffer> generalGaussianSplatIndirectBuffer;
        std::optional<rhi::RadixSorter>        generalGaussianSplatSorter;
        bool                                   generalGaussianSplatDeterministicSourceOrderSort {false};
        std::array<Ref<rhi::StorageBuffer>, kGeneralGaussianSplatFoveatedLayerCount>
            generalGaussianSplatFoveatedVisibleSplatBuffers {};
        std::array<Ref<rhi::StorageBuffer>, kGeneralGaussianSplatFoveatedLayerCount>
            generalGaussianSplatFoveatedSortKeyBuffers {};
        std::array<Ref<rhi::StorageBuffer>, kGeneralGaussianSplatFoveatedLayerCount>
            generalGaussianSplatFoveatedSortIndexBuffers {};
        std::array<Ref<rhi::StorageBuffer>, kGeneralGaussianSplatFoveatedLayerCount>
            generalGaussianSplatFoveatedVisibleCountBuffers {};
        std::array<std::optional<rhi::DrawIndirectBuffer>, kGeneralGaussianSplatFoveatedLayerCount>
            generalGaussianSplatFoveatedIndirectBuffers {};

        uint32_t maxVisibleInstances {0};
        uint32_t maxVisibleMeshlets {0};
        uint32_t maxDraws {0};
        uint32_t maxGeneralGaussianSplatDraws {0};
        // Source count is the full packed raw table. Point count sizes the
        // selected-source buffer, while active point count is the dispatch prefix
        // currently consumed by preprocess.
        uint32_t maxGeneralGaussianSplatSourceCount {0};
        uint32_t maxGeneralGaussianSplatPoints {0};
        uint32_t activeGeneralGaussianSplatPoints {0};
        uint32_t maxGeneralGaussianSplatVisibleSplats {0};
        uint32_t generalGaussianSplatTemporalStateCapacity {0};
        uint32_t generalGaussianSplatCoverageTextureCapacity {0};
        uint32_t generalGaussianSplatProjectedCostChunkCapacity {0};
        bool      generalGaussianSplatTemporalStateNeedsReset {false};
        bool      generalGaussianSplatCoverageTextureNeedsReset {false};
        bool      generalGaussianSplatDirectPrefix {false};
        GpuGaussianSplatShStorageLayout generalGaussianSplatShStorageLayout {
            GpuGaussianSplatShStorageLayout::eMonolithic};
        bool      generalGaussianSplatFoveatedClodEnabled {false};
        bool      generalGaussianSplatFoveatedLayeredCompositeEnabled {false};
        bool      generalGaussianSplatFoveatedCoverageCompensationEnabled {false};
        bool      generalGaussianSplatFoveatedScoreSelected {false};
        bool      generalGaussianSplatFoveatedTemporalHysteresisEnabled {true};
        bool      generalGaussianSplatFoveatedBoundarySmoothingEnabled {true};
        glm::vec2 generalGaussianSplatFoveatedGaze {0.5f, 0.5f};
        glm::vec2 generalGaussianSplatFoveatedRingDegrees {12.0f, 32.0f};
        glm::vec3 generalGaussianSplatFoveatedRingLevels {1.0f, 0.40f, 0.15f};
        glm::vec3 generalGaussianSplatFoveatedResolutionScales {1.0f, 0.75f, 0.50f};
        float     generalGaussianSplatFoveatedTransitionDegrees {8.0f};
        uint32_t  generalGaussianSplatFoveatedCoverageGuardMode {1u};
        float     generalGaussianSplatFoveatedCoverageProtectionDegrees {0.0f};
        float     generalGaussianSplatFoveatedCoverageGuardBudgetRatio {0.05f};
        glm::vec3 generalGaussianSplatFoveatedCoverageGuardMinLevels {0.75f, 0.45f, 0.15f};
        uint32_t  generalGaussianSplatFoveatedCoverageGuardRiskSectors {0x7u};
        float     generalGaussianSplatFoveatedTemporalHysteresisRatio {0.25f};
        float     generalGaussianSplatFoveatedBoundarySmoothingRatio {0.18f};
        uint32_t  generalGaussianSplatFoveatedTemporalResidencyFrames {6u};
        uint32_t  generalGaussianSplatFoveatedDistribution {2u};
        float     generalGaussianSplatFoveatedContinuousTheta0Degrees {24.0f};
        float     generalGaussianSplatFoveatedContinuousAlpha {2.0f};
        float     generalGaussianSplatFoveatedContinuousMinLevel {0.12f};
        float     generalGaussianSplatFoveatedTemporalPeripheralScale {0.0f};
        bool      generalGaussianSplatFoveatedShLodEnabled {false};
        bool      generalGaussianSplatFoveatedShSmoothSuppressionEnabled {false};
        bool      generalGaussianSplatPeripheralTemporalFilterEnabled {false};
        float     generalGaussianSplatPeripheralTemporalFilterOuterDegrees {56.0f};
        float     generalGaussianSplatPeripheralTemporalFilterLambdaScale {0.85f};
        float     generalGaussianSplatPeripheralTemporalFilterRejectionThreshold {0.18f};
        float     generalGaussianSplatPeripheralTemporalFilterClampRadius {0.20f};
        glm::uvec3 generalGaussianSplatFoveatedShLodDegrees {3u, 3u, 3u};
        uint32_t  generalGaussianSplatFoveatedShLodGuardMode {0u};
        float     generalGaussianSplatFoveatedShLodGuardThresholdMid {0.05f};
        float     generalGaussianSplatFoveatedShLodGuardThresholdHigh {0.20f};
        uint32_t  generalGaussianSplatShaderAntiPopMode {0u};
        uint32_t  generalGaussianSplatShaderAntiPopHashSeed {0u};
        float     generalGaussianSplatShaderAntiPopRampWidth {0.05f};
        float     generalGaussianSplatShaderAntiPopGuardThreshold {0.0f};
        float     generalGaussianSplatShaderAntiPopGuardFloor {0.0f};
        uint32_t  generalGaussianSplatShaderAntiPopPKeepCurve {0u};
        float     generalGaussianSplatShaderAntiPopPrefixRatio {-1.0f};
        uint32_t  generalGaussianSplatShaderAntiPopNormalizeMode {0u};
        float     generalGaussianSplatShaderAntiPopNormalizeStrength {1.0f};
        float     generalGaussianSplatShaderAntiPopNormalizeClampMin {0.75f};
        float     generalGaussianSplatShaderAntiPopNormalizeClampMax {1.15f};
        float     generalGaussianSplatShaderAntiPopNormalizeFactor {1.0f};
        bool      generalGaussianSplatEcsptCounterReadbackEnabled {false};
        uint32_t  generalGaussianSplatCoverageStableReleaseFlags {0u};
        uint32_t  generalGaussianSplatCoverageStableReleaseTileGridX {16u};
        uint32_t  generalGaussianSplatCoverageStableReleaseTileGridY {8u};
        uint32_t  generalGaussianSplatCoverageStableReleaseFrameIndex {0u};
        float     generalGaussianSplatCoverageStableReleaseLambda {0.7743f};
        float     generalGaussianSplatCoverageStableReleaseDMin {4.0f};
        float     generalGaussianSplatCoverageStableReleaseSigmaMax {0.65f};
        float     generalGaussianSplatCoverageStableReleaseReleaseEpsilon {1e-4f};
        float     generalGaussianSplatCoverageTextureStrength {0.25f};
        float     generalGaussianSplatCoverageTextureHistoryBeta {0.80f};
        uint32_t  generalGaussianSplatCoverageTextureUpdateInterval {1u};
        uint32_t  generalGaussianSplatDelayedGuideTemporalReleasePolicy {0u};
        float     generalGaussianSplatDelayedGuideTemporalReleaseCapRatio {0.12f};
        float     generalGaussianSplatDelayedGuideTemporalRiskThreshold {0.35f};
        float     generalGaussianSplatDelayedGuideTemporalFootprintDecayScale {0.55f};
        float     generalGaussianSplatDelayedGuideTemporalLargeFootprintPx {48.0f};
        uint32_t  generalGaussianSplatGazeAnchorFlags {0u};
        uint32_t  generalGaussianSplatGazeAnchorFadeFrames {4u};
        uint32_t  generalGaussianSplatGazeAnchorMinUpdateFrames {2u};
        float     generalGaussianSplatGazeAnchorFadePhase {1.0f};
        float     generalGaussianSplatGazeAnchorTransitionBudgetRatio {-1.0f};
        float     generalGaussianSplatGazeAnchorAux0 {0.0f};
        float     generalGaussianSplatGazeAnchorAux1 {0.0f};
        glm::vec2 generalGaussianSplatGazeAnchorOldGaze {0.5f, 0.5f};
        glm::vec2 generalGaussianSplatGazeAnchorNewGaze {0.5f, 0.5f};

        void clear()
        {
            database = nullptr;
            mode     = GpuSceneBuildMode::eCpuDriven;
            draws.clear();
            indirectCommands.clear();
            generalGaussianSplatDraws.clear();
            generalGaussianSplatPackedSources.clear();
            generalGaussianSplatSelectedSources.clear();
            visibleMeshlets.clear();
            visibleInstanceBuffer         = nullptr;
            visibleInstanceCountBuffer    = nullptr;
            meshletCullDispatchArgsBuffer = nullptr;
            visibleMeshletBuffer          = nullptr;
            visibleMeshletCountBuffer     = nullptr;
            drawBuffer                    = nullptr;
            drawSetBuffer                 = nullptr;
            indirectBuffer.reset();
            generalGaussianSplatDrawBuffer        = nullptr;
            generalGaussianSplatPackedSourceBuffer = nullptr;
            generalGaussianSplatSelectedSourceBuffer = nullptr;
            generalGaussianSplatVisibleSplatBuffer = nullptr;
            generalGaussianSplatSortKeyBuffer      = nullptr;
            generalGaussianSplatSortIndexBuffer    = nullptr;
            generalGaussianSplatVisibleCountBuffer = nullptr;
            generalGaussianSplatEcsptCounterBuffer = nullptr;
            generalGaussianSplatEcsptCounterReadbackBuffer = nullptr;
            generalGaussianSplatDispatchArgsBuffer = nullptr;
            generalGaussianSplatSortStorageBuffer  = nullptr;
            generalGaussianSplatShBuffer           = nullptr;
            generalGaussianSplatShL1Buffer         = nullptr;
            generalGaussianSplatShL2Buffer         = nullptr;
            generalGaussianSplatShL3Buffer         = nullptr;
            generalGaussianSplatShEnergyMetadataBuffer = nullptr;
            generalGaussianSplatTemporalStateBuffer = nullptr;
            generalGaussianSplatCoverageTextureBuffer = nullptr;
            generalGaussianSplatProjectedCostChunkBuffer = nullptr;
            generalGaussianSplatProjectedCostReadbackBuffer = nullptr;
            generalGaussianSplatLastAlphaTexture = nullptr;
            generalGaussianSplatEcsptCounterReadbackEnabled = false;
            generalGaussianSplatIndirectBuffer.reset();
            generalGaussianSplatSorter.reset();
            for (auto& buffer : generalGaussianSplatFoveatedVisibleSplatBuffers)
                buffer = nullptr;
            for (auto& buffer : generalGaussianSplatFoveatedSortKeyBuffers)
                buffer = nullptr;
            for (auto& buffer : generalGaussianSplatFoveatedSortIndexBuffers)
                buffer = nullptr;
            for (auto& buffer : generalGaussianSplatFoveatedVisibleCountBuffers)
                buffer = nullptr;
            for (auto& buffer : generalGaussianSplatFoveatedIndirectBuffers)
                buffer.reset();
            maxVisibleInstances          = 0;
            maxVisibleMeshlets           = 0;
            maxDraws                     = 0;
            maxGeneralGaussianSplatDraws         = 0;
            maxGeneralGaussianSplatSourceCount   = 0;
            maxGeneralGaussianSplatPoints        = 0;
            activeGeneralGaussianSplatPoints     = 0;
            maxGeneralGaussianSplatVisibleSplats = 0;
            generalGaussianSplatTemporalStateCapacity   = 0;
            generalGaussianSplatCoverageTextureCapacity = 0;
            generalGaussianSplatProjectedCostChunkCapacity = 0;
            generalGaussianSplatTemporalStateNeedsReset = false;
            generalGaussianSplatCoverageTextureNeedsReset = false;
            generalGaussianSplatDirectPrefix     = false;
            generalGaussianSplatShStorageLayout  = GpuGaussianSplatShStorageLayout::eMonolithic;
            resetGeneralGaussianSplatFoveatedClod();
        }

        void beginFrame(const GpuSceneDatabase& db, GpuSceneBuildMode buildMode = GpuSceneBuildMode::eCpuDriven)
        {
            database = &db;
            mode     = buildMode;
            draws.clear();
            indirectCommands.clear();
            generalGaussianSplatDraws.clear();
            generalGaussianSplatPackedSources.clear();
            generalGaussianSplatSelectedSources.clear();
            visibleMeshlets.clear();
            maxVisibleInstances          = 0;
            maxVisibleMeshlets           = 0;
            maxDraws                     = 0;
            maxGeneralGaussianSplatDraws         = 0;
            maxGeneralGaussianSplatSourceCount   = 0;
            maxGeneralGaussianSplatPoints        = 0;
            activeGeneralGaussianSplatPoints     = 0;
            maxGeneralGaussianSplatVisibleSplats = 0;
            generalGaussianSplatTemporalStateNeedsReset = false;
            generalGaussianSplatProjectedCostChunkCapacity = 0;
            generalGaussianSplatDirectPrefix     = false;
            generalGaussianSplatShStorageLayout  = GpuGaussianSplatShStorageLayout::eMonolithic;
            generalGaussianSplatEcsptCounterReadbackEnabled = false;
            resetGeneralGaussianSplatFoveatedClod();
        }

        [[nodiscard]] bool isCpuDriven() const { return mode == GpuSceneBuildMode::eCpuDriven; }
        [[nodiscard]] bool isGpuDriven() const { return mode == GpuSceneBuildMode::eGpuDriven; }

        void setGpuDrivenCaps(uint32_t maxVisibleInstancesCount, uint32_t maxVisible, uint32_t maxDrawCount)
        {
            maxVisibleInstances = maxVisibleInstancesCount;
            maxVisibleMeshlets  = maxVisible;
            maxDraws            = maxDrawCount;
        }

        void ensureVisibleInstanceBuffers(rhi::RenderDevice& rd)
        {
            if (maxVisibleInstances == 0)
                return;

            const uint64_t idsBytes = static_cast<uint64_t>(maxVisibleInstances) * sizeof(uint32_t);
            if (!visibleInstanceBuffer || static_cast<uint64_t>(visibleInstanceBuffer->getSize()) < idsBytes)
                visibleInstanceBuffer = createRef<rhi::StorageBuffer>(rd.createStorageBuffer(idsBytes));

            if (!visibleInstanceCountBuffer || visibleInstanceCountBuffer->getSize() < sizeof(uint32_t))
                visibleInstanceCountBuffer = createRef<rhi::StorageBuffer>(rd.createStorageBuffer(sizeof(uint32_t)));

            constexpr uint64_t kDispatchArgsBytes = sizeof(uint32_t) * 3ull;
            if (!meshletCullDispatchArgsBuffer ||
                static_cast<uint64_t>(meshletCullDispatchArgsBuffer->getSize()) < kDispatchArgsBytes)
            {
                meshletCullDispatchArgsBuffer = createRef<rhi::StorageBuffer>(
                    rd.createStorageBufferWithUsage(kDispatchArgsBytes, rhi::BufferUsage::eIndirectBuffer));
            }
        }

        [[nodiscard]] uint32_t getDispatchableDrawCount() const
        {
            return isGpuDriven() ? maxDraws : static_cast<uint32_t>(indirectCommands.size());
        }

        [[nodiscard]] uint32_t countMeshletDraws() const { return static_cast<uint32_t>(draws.size()); }
        [[nodiscard]] uint32_t countGeneralGaussianSplatDraws() const
        {
            return static_cast<uint32_t>(generalGaussianSplatDraws.size());
        }
        [[nodiscard]] bool hasGeneralGaussianSplats() const
        {
            return !generalGaussianSplatDraws.empty() || maxGeneralGaussianSplatDraws > 0u;
        }

        uint32_t pushDraw(const GpuDrawRecord& dr)
        {
            const uint32_t index = static_cast<uint32_t>(draws.size());
            draws.push_back(dr);
            return index;
        }

        uint32_t pushDraw(GpuDrawRecord&& dr)
        {
            const uint32_t index = static_cast<uint32_t>(draws.size());
            draws.push_back(std::move(dr));
            return index;
        }

        uint32_t pushMeshletDraw(GpuDrawRecord dr)
        {
            dr.flags |= gpuDrawFlagsToMask(GpuDrawFlags::eMeshlet);
            return pushDraw(std::move(dr));
        }

        uint32_t pushGeneralGaussianSplatDraw(const GpuGeneralGaussianSplatDrawRecord& dr)
        {
            const uint32_t index = static_cast<uint32_t>(generalGaussianSplatDraws.size());
            generalGaussianSplatDraws.push_back(dr);
            return index;
        }

        uint32_t pushGeneralGaussianSplatSource(const GpuGeneralGaussianSplatPackedSource& src)
        {
            const uint32_t index = static_cast<uint32_t>(generalGaussianSplatPackedSources.size());
            generalGaussianSplatPackedSources.push_back(src);
            return index;
        }

        uint32_t pushGeneralGaussianSplatSelectedSource(const GpuGeneralGaussianSplatSelectedSource& src)
        {
            const uint32_t index = static_cast<uint32_t>(generalGaussianSplatSelectedSources.size());
            generalGaussianSplatSelectedSources.push_back(src);
            return index;
        }

        uint32_t pushVisibleMeshlet(const GpuVisibleMeshlet& vm)
        {
            const uint32_t index = static_cast<uint32_t>(visibleMeshlets.size());
            visibleMeshlets.push_back(vm);
            return index;
        }

        void ensureVisibleMeshletBuffers(rhi::RenderDevice& rd)
        {
            if (maxVisibleMeshlets == 0)
                return;

            const uint64_t idsBytes = static_cast<uint64_t>(maxVisibleMeshlets) * sizeof(GpuVisibleMeshlet);
            if (!visibleMeshletBuffer || static_cast<uint64_t>(visibleMeshletBuffer->getSize()) < idsBytes)
                visibleMeshletBuffer = createRef<rhi::StorageBuffer>(rd.createStorageBuffer(idsBytes));

            if (!visibleMeshletCountBuffer || visibleMeshletCountBuffer->getSize() < sizeof(uint32_t))
                visibleMeshletCountBuffer = createRef<rhi::DrawIndirectBuffer>(
                    rd.createDrawIndirectBufferBySize(sizeof(uint32_t), rhi::DrawIndirectType::eNonIndexed));
        }

        void ensureDrawBuffer(rhi::RenderDevice& rd)
        {
            const uint32_t count = isGpuDriven() ? maxDraws : static_cast<uint32_t>(draws.size());
            const uint64_t bytes = static_cast<uint64_t>(count) * sizeof(GpuDrawRecord);
            if (bytes == 0)
                return;

            if (!drawBuffer || static_cast<uint64_t>(drawBuffer->getSize()) < bytes)
                drawBuffer = createRef<rhi::StorageBuffer>(rd.createStorageBuffer(bytes));
        }

        void ensureDrawSetBuffer(rhi::RenderDevice& rd)
        {
            // 8 counters to keep alignment and allow future queue expansion.
            constexpr uint64_t kDrawSetBytes = sizeof(uint32_t) * 8ull;
            if (!drawSetBuffer || static_cast<uint64_t>(drawSetBuffer->getSize()) < kDrawSetBytes)
                drawSetBuffer = createRef<rhi::DrawIndirectBuffer>(
                    rd.createDrawIndirectBufferBySize(kDrawSetBytes, rhi::DrawIndirectType::eNonIndexed));
        }

        void uploadDraws(rhi::RenderDevice& rd, rhi::CommandBuffer& cb)
        {
            ensureDrawBuffer(rd);

            const size_t drawBytes = draws.size() * sizeof(GpuDrawRecord);
            if (drawBytes > 0)
                cb.update(*drawBuffer, 0, static_cast<uint64_t>(drawBytes), draws.data());
        }

        void buildIndirectFromDraws(const GpuResourcePool& pool)
        {
            indirectCommands.clear();
            indirectCommands.reserve(draws.size());

            for (uint32_t drawId = 0; drawId < static_cast<uint32_t>(draws.size()); ++drawId)
            {
                const auto& d = draws[drawId];
                if (!gpuDrawHasFlag(d.flags, GpuDrawFlags::eMeshlet))
                    continue;
                if (d.primitiveIndex >= pool.meshlets.cpuMeshlets.size())
                    continue;
                const auto& m = pool.meshlets.cpuMeshlets[d.primitiveIndex];

                rhi::DrawIndirectCommand cmd {};
                cmd.type          = rhi::DrawIndirectType::eNonIndexed;
                cmd.count         = m.triangleCount * 3u;
                cmd.instanceCount = 1;
                cmd.first         = 0;
                cmd.vertexOffset  = 0;
                cmd.firstInstance = drawId; // MoltenVK-friendly drawId path

                indirectCommands.push_back(cmd);
            }

            maxDraws = static_cast<uint32_t>(indirectCommands.size());
        }

        void ensureIndirectBuffer(rhi::RenderDevice& rd)
        {
            const uint32_t cmdCount = isGpuDriven() ?
                                          (maxDraws == 0 ? 5u : maxDraws * 5u) :
                                          static_cast<uint32_t>(indirectCommands.empty() ? 1 : indirectCommands.size());

            if (!indirectBuffer.has_value() ||
                indirectBuffer->getDrawIndirectType() != rhi::DrawIndirectType::eNonIndexed ||
                indirectBuffer->getCapacity() < cmdCount)
            {
                indirectBuffer = rd.createDrawIndirectBufferByCount(cmdCount, rhi::DrawIndirectType::eNonIndexed);
            }
        }

        void uploadIndirect(rhi::RenderDevice& rd)
        {
            ensureIndirectBuffer(rd);
            if (!indirectBuffer.has_value())
                return;

            if (isCpuDriven())
            {
                rd.uploadDrawIndirect(*indirectBuffer, indirectCommands);
            }
        }

        void prepareGpuDrivenBuffers(rhi::RenderDevice& rd,
                                     uint32_t           maxVisibleInstancesCount,
                                     uint32_t           maxVisible,
                                     uint32_t           maxDrawCount)
        {
            mode = GpuSceneBuildMode::eGpuDriven;
            setGpuDrivenCaps(maxVisibleInstancesCount, maxVisible, maxDrawCount);
            ensureVisibleInstanceBuffers(rd);
            ensureVisibleMeshletBuffers(rd);
            ensureDrawBuffer(rd);
            ensureDrawSetBuffer(rd);
            ensureIndirectBuffer(rd);
        }

        // maxSourceCount sizes immutable packed raw data. maxPointCount sizes the
        // selected-source indirection buffer when the fallback path is active;
        // directPrefix skips that buffer and consumes a packed-source prefix.
        void setGeneralGaussianSplatCaps(uint32_t maxDrawCount,
                                         uint32_t maxSourceCount,
                                         uint32_t maxPointCount,
                                         uint32_t activePointCount,
                                         uint32_t maxVisibleSplatCount,
                                         bool     directPrefix = false)
        {
            maxGeneralGaussianSplatDraws         = maxDrawCount;
            maxGeneralGaussianSplatSourceCount   = maxSourceCount;
            maxGeneralGaussianSplatPoints        = maxPointCount;
            generalGaussianSplatDirectPrefix     = directPrefix;
            const uint32_t activeCapacity        = directPrefix ? maxSourceCount : maxPointCount;
            activeGeneralGaussianSplatPoints     = std::min(activePointCount, activeCapacity);
            maxGeneralGaussianSplatVisibleSplats = maxVisibleSplatCount;
        }

        void setGeneralGaussianSplatFoveatedClod(bool enabled,
                                                 bool layeredCompositeEnabled,
                                                 bool coverageCompensationEnabled,
                                                 glm::vec2 gaze,
                                                 glm::vec2 ringDegrees,
                                                 glm::vec3 ringLevels,
                                                 glm::vec3 resolutionScales,
                                                 float     transitionDegrees,
                                                 uint32_t  coverageGuardMode,
                                                 float     coverageProtectionDegrees,
                                                 float     coverageGuardBudgetRatio,
                                                 glm::vec3 coverageGuardMinLevels,
                                                 uint32_t  coverageGuardRiskSectors,
                                                 uint32_t  distribution,
                                                 float     continuousTheta0Degrees,
                                                 float     continuousAlpha,
                                                 float     continuousMinLevel,
                                                 float     temporalPeripheralScale,
                                                 bool      temporalHysteresisEnabled,
                                                 bool      boundarySmoothingEnabled,
                                                 uint32_t  temporalResidencyFrames,
                                                 float     temporalHysteresisRatio,
                                                 float     boundarySmoothingRatio,
                                                 bool      shLodEnabled,
                                                 bool      shSmoothSuppressionEnabled,
                                                 bool      peripheralTemporalFilterEnabled,
                                                 float     peripheralTemporalFilterOuterDegrees,
                                                 float     peripheralTemporalFilterLambdaScale,
                                                 float     peripheralTemporalFilterRejectionThreshold,
                                                 float     peripheralTemporalFilterClampRadius,
                                                 glm::uvec3 shLodDegrees,
                                                 uint32_t  shLodGuardMode,
                                                 float     shLodGuardThresholdMid,
                                                 float     shLodGuardThresholdHigh)
        {
            generalGaussianSplatFoveatedClodEnabled              = enabled;
            generalGaussianSplatFoveatedLayeredCompositeEnabled = layeredCompositeEnabled;
            generalGaussianSplatFoveatedCoverageCompensationEnabled =
                coverageCompensationEnabled;
            generalGaussianSplatFoveatedScoreSelected = false;
            generalGaussianSplatFoveatedTemporalHysteresisEnabled =
                temporalHysteresisEnabled;
            generalGaussianSplatFoveatedBoundarySmoothingEnabled =
                boundarySmoothingEnabled;
            generalGaussianSplatFoveatedGaze                     = gaze;
            generalGaussianSplatFoveatedRingDegrees              = ringDegrees;
            generalGaussianSplatFoveatedRingLevels               = ringLevels;
            generalGaussianSplatFoveatedResolutionScales         = resolutionScales;
            generalGaussianSplatFoveatedTransitionDegrees        = transitionDegrees;
            generalGaussianSplatFoveatedCoverageGuardMode        = coverageGuardMode;
            generalGaussianSplatFoveatedCoverageProtectionDegrees =
                coverageProtectionDegrees;
            generalGaussianSplatFoveatedCoverageGuardBudgetRatio =
                coverageGuardBudgetRatio;
            generalGaussianSplatFoveatedCoverageGuardMinLevels =
                coverageGuardMinLevels;
            generalGaussianSplatFoveatedCoverageGuardRiskSectors =
                coverageGuardRiskSectors & 0x7u;
            generalGaussianSplatFoveatedTemporalResidencyFrames  = temporalResidencyFrames;
            generalGaussianSplatFoveatedTemporalHysteresisRatio  = temporalHysteresisRatio;
            generalGaussianSplatFoveatedBoundarySmoothingRatio   = boundarySmoothingRatio;
            generalGaussianSplatFoveatedDistribution             = distribution;
            generalGaussianSplatFoveatedContinuousTheta0Degrees  = std::max(continuousTheta0Degrees, 1e-4f);
            generalGaussianSplatFoveatedContinuousAlpha          = std::max(continuousAlpha, 0.0f);
            generalGaussianSplatFoveatedContinuousMinLevel       =
                std::clamp(continuousMinLevel, 0.0f, 1.0f);
            generalGaussianSplatFoveatedTemporalPeripheralScale  =
                std::max(temporalPeripheralScale, 0.0f);
            generalGaussianSplatFoveatedShLodEnabled             = shLodEnabled;
            generalGaussianSplatFoveatedShSmoothSuppressionEnabled =
                shSmoothSuppressionEnabled;
            generalGaussianSplatPeripheralTemporalFilterEnabled =
                peripheralTemporalFilterEnabled;
            generalGaussianSplatPeripheralTemporalFilterOuterDegrees =
                std::max(peripheralTemporalFilterOuterDegrees, generalGaussianSplatFoveatedRingDegrees.y + 1.0f);
            generalGaussianSplatPeripheralTemporalFilterLambdaScale =
                std::clamp(peripheralTemporalFilterLambdaScale, 0.0f, 1.0f);
            generalGaussianSplatPeripheralTemporalFilterRejectionThreshold =
                std::max(peripheralTemporalFilterRejectionThreshold, 1e-4f);
            generalGaussianSplatPeripheralTemporalFilterClampRadius =
                std::max(peripheralTemporalFilterClampRadius, 0.0f);
            generalGaussianSplatFoveatedShLodDegrees             = glm::clamp(shLodDegrees,
                                                                               glm::uvec3 {0u},
                                                                               glm::uvec3 {3u});
            generalGaussianSplatFoveatedShLodGuardMode           = std::min(shLodGuardMode, 3u);
            generalGaussianSplatFoveatedShLodGuardThresholdMid   = std::max(shLodGuardThresholdMid, 0.0f);
            generalGaussianSplatFoveatedShLodGuardThresholdHigh  =
                std::max(shLodGuardThresholdHigh, generalGaussianSplatFoveatedShLodGuardThresholdMid);
        }

        void setGeneralGaussianSplatShaderAntiPop(uint32_t mode,
                                                  uint32_t hashSeed,
                                                  float    rampWidth,
                                                  float    guardThreshold,
                                                  float    guardFloor,
                                                  uint32_t pKeepCurve,
                                                  float    prefixRatio = -1.0f,
                                                  uint32_t normalizeMode = 0u,
                                                  float    normalizeStrength = 1.0f,
                                                  float    normalizeClampMin = 0.75f,
                                                  float    normalizeClampMax = 1.15f,
                                                  float    normalizeFactor = 1.0f)
        {
            generalGaussianSplatShaderAntiPopMode = std::min(mode, 9u);
            generalGaussianSplatShaderAntiPopHashSeed = hashSeed;
            generalGaussianSplatShaderAntiPopRampWidth = std::clamp(rampWidth, 0.0f, 1.0f);
            generalGaussianSplatShaderAntiPopGuardThreshold = std::clamp(guardThreshold, 0.0f, 1.0f);
            generalGaussianSplatShaderAntiPopGuardFloor = std::clamp(guardFloor, 0.0f, 1.0f);
            generalGaussianSplatShaderAntiPopPKeepCurve = std::min(pKeepCurve, 4u);
            generalGaussianSplatShaderAntiPopPrefixRatio = prefixRatio;
            generalGaussianSplatShaderAntiPopNormalizeMode = std::min(normalizeMode, 2u);
            generalGaussianSplatShaderAntiPopNormalizeStrength = std::clamp(normalizeStrength, 0.0f, 1.0f);
            generalGaussianSplatShaderAntiPopNormalizeClampMin = std::max(normalizeClampMin, 0.0f);
            generalGaussianSplatShaderAntiPopNormalizeClampMax =
                std::max(normalizeClampMax, generalGaussianSplatShaderAntiPopNormalizeClampMin);
            generalGaussianSplatShaderAntiPopNormalizeFactor =
                std::clamp(normalizeFactor,
                           generalGaussianSplatShaderAntiPopNormalizeClampMin,
                           generalGaussianSplatShaderAntiPopNormalizeClampMax);
        }

        void setGeneralGaussianSplatCoverageStableRelease(uint32_t flags,
                                                          uint32_t tileGridX,
                                                          uint32_t tileGridY,
                                                          uint32_t frameIndex,
                                                          float    releaseLambda,
                                                          float    dMin,
                                                          float    sigmaMax,
                                                          float    releaseEpsilon)
        {
            constexpr uint32_t kCoverageReleaseTextureFloorFlag = 32u;
            generalGaussianSplatCoverageStableReleaseFlags = flags;
            const bool textureFloor = (flags & kCoverageReleaseTextureFloorFlag) != 0u;
            generalGaussianSplatCoverageStableReleaseTileGridX =
                textureFloor ? std::clamp(tileGridX, 1u, 1024u) : std::clamp(tileGridX, 1u, 32u);
            generalGaussianSplatCoverageStableReleaseTileGridY =
                textureFloor ? std::clamp(tileGridY, 1u, 1024u) : std::clamp(tileGridY, 1u, 16u);
            generalGaussianSplatCoverageStableReleaseFrameIndex = frameIndex;
            generalGaussianSplatCoverageStableReleaseLambda = std::clamp(releaseLambda, 0.0f, 1.0f);
            generalGaussianSplatCoverageStableReleaseDMin = std::max(dMin, 0.0f);
            generalGaussianSplatCoverageStableReleaseSigmaMax = std::max(sigmaMax, 0.0f);
            generalGaussianSplatCoverageStableReleaseReleaseEpsilon = std::max(releaseEpsilon, 0.0f);
        }

        void setGeneralGaussianSplatCoverageTextureFloor(float strength, float historyBeta, uint32_t updateInterval)
        {
            generalGaussianSplatCoverageTextureStrength = std::clamp(strength, 0.0f, 1.0f);
            generalGaussianSplatCoverageTextureHistoryBeta = std::clamp(historyBeta, 0.0f, 1.0f);
            generalGaussianSplatCoverageTextureUpdateInterval = std::max(updateInterval, 1u);
        }

        void setGeneralGaussianSplatDelayedGuideTemporalReleaseCost(uint32_t policy,
                                                                    float    capRatio,
                                                                    float    riskThreshold,
                                                                    float    footprintDecayScale,
                                                                    float    largeFootprintPx)
        {
            generalGaussianSplatDelayedGuideTemporalReleasePolicy = std::min(policy, 4u);
            generalGaussianSplatDelayedGuideTemporalReleaseCapRatio = std::clamp(capRatio, 0.0f, 1.0f);
            generalGaussianSplatDelayedGuideTemporalRiskThreshold = std::clamp(riskThreshold, 0.0f, 1.0f);
            generalGaussianSplatDelayedGuideTemporalFootprintDecayScale =
                std::clamp(footprintDecayScale, 0.0f, 1.0f);
            generalGaussianSplatDelayedGuideTemporalLargeFootprintPx = std::max(largeFootprintPx, 1.0f);
        }

        void setGeneralGaussianSplatGazeAnchorCrossfade(uint32_t  flags,
                                                        glm::vec2 oldGaze,
                                                        glm::vec2 newGaze,
                                                        float     fadePhase,
                                                        float     transitionBudgetRatio,
                                                        uint32_t  fadeFrames,
                                                        uint32_t  minUpdateFrames)
        {
            generalGaussianSplatGazeAnchorFlags = flags;
            generalGaussianSplatGazeAnchorOldGaze = glm::clamp(oldGaze, glm::vec2 {0.0f}, glm::vec2 {1.0f});
            generalGaussianSplatGazeAnchorNewGaze = glm::clamp(newGaze, glm::vec2 {0.0f}, glm::vec2 {1.0f});
            generalGaussianSplatGazeAnchorFadePhase = std::clamp(fadePhase, 0.0f, 1.0f);
            generalGaussianSplatGazeAnchorTransitionBudgetRatio = transitionBudgetRatio;
            generalGaussianSplatGazeAnchorAux0 = 0.0f;
            generalGaussianSplatGazeAnchorAux1 = 0.0f;
            generalGaussianSplatGazeAnchorFadeFrames = std::max(fadeFrames, 1u);
            generalGaussianSplatGazeAnchorMinUpdateFrames = minUpdateFrames;
        }

        void setGeneralGaussianSplatEccentricityStochasticTransition(uint32_t  flags,
                                                                      glm::vec2 oldGaze,
                                                                      glm::vec2 newGaze,
                                                                      float     fadePhase,
                                                                      float     protectOldFoveaDegrees,
                                                                      float     protectNewFoveaDegrees,
                                                                      float     boundaryBandDegrees,
                                                                      uint32_t  fadeFrames)
        {
            generalGaussianSplatGazeAnchorFlags = flags;
            generalGaussianSplatGazeAnchorOldGaze = glm::clamp(oldGaze, glm::vec2 {0.0f}, glm::vec2 {1.0f});
            generalGaussianSplatGazeAnchorNewGaze = glm::clamp(newGaze, glm::vec2 {0.0f}, glm::vec2 {1.0f});
            generalGaussianSplatGazeAnchorFadePhase = std::clamp(fadePhase, 0.0f, 1.0f);
            generalGaussianSplatGazeAnchorTransitionBudgetRatio = std::max(protectOldFoveaDegrees, -1.0f);
            generalGaussianSplatGazeAnchorAux0 = std::max(protectNewFoveaDegrees, -1.0f);
            generalGaussianSplatGazeAnchorAux1 = std::max(boundaryBandDegrees, 0.0f);
            generalGaussianSplatGazeAnchorFadeFrames = std::max(fadeFrames, 1u);
            generalGaussianSplatGazeAnchorMinUpdateFrames = 0u;
        }

        void resetGeneralGaussianSplatFoveatedClod()
        {
            setGeneralGaussianSplatFoveatedClod(false,
                                                false,
                                                false,
                                                glm::vec2 {0.5f, 0.5f},
                                                glm::vec2 {12.0f, 32.0f},
                                                glm::vec3 {1.0f, 0.40f, 0.15f},
                                                glm::vec3 {1.0f, 0.75f, 0.50f},
                                                8.0f,
                                                1u,
                                                0.0f,
                                                0.05f,
                                                glm::vec3 {0.75f, 0.45f, 0.15f},
                                                0x7u,
                                                2u,
                                                24.0f,
                                                2.0f,
                                                0.12f,
                                                0.0f,
                                                true,
                                                true,
                                                6u,
                                                0.25f,
                                                0.18f,
                                                false,
                                                false,
                                                false,
                                                56.0f,
                                                0.85f,
                                                0.18f,
                                                0.20f,
                                                glm::uvec3 {3u, 3u, 3u},
                                                0u,
                                                0.05f,
                                                0.20f);
            setGeneralGaussianSplatShaderAntiPop(0u, 0u, 0.05f, 0.0f, 0.0f, 0u);
            setGeneralGaussianSplatCoverageStableRelease(0x7u, 16u, 8u, 0u, 0.7743f, 4.0f, 0.65f, 1e-4f);
            setGeneralGaussianSplatDelayedGuideTemporalReleaseCost(0u, 0.12f, 0.35f, 0.55f, 48.0f);
        }

        void ensureGeneralGaussianSplatBuffers(rhi::RenderDevice& rd)
        {
            if (maxGeneralGaussianSplatDraws > 0u)
            {
                const uint64_t drawBytes =
                    static_cast<uint64_t>(maxGeneralGaussianSplatDraws) * sizeof(GpuGeneralGaussianSplatDrawRecord);
                if (!generalGaussianSplatDrawBuffer ||
                    static_cast<uint64_t>(generalGaussianSplatDrawBuffer->getSize()) < drawBytes)
                {
                    generalGaussianSplatDrawBuffer = createRef<rhi::StorageBuffer>(rd.createStorageBuffer(drawBytes));
                }
            }

            if (maxGeneralGaussianSplatSourceCount > 0u)
            {
                const uint64_t packedBytes =
                    static_cast<uint64_t>(maxGeneralGaussianSplatSourceCount) * sizeof(GpuGeneralGaussianSplatPackedSource);
                if (!generalGaussianSplatPackedSourceBuffer ||
                    static_cast<uint64_t>(generalGaussianSplatPackedSourceBuffer->getSize()) < packedBytes)
                {
                    generalGaussianSplatPackedSourceBuffer =
                        createRef<rhi::StorageBuffer>(rd.createStorageBuffer(packedBytes));
                }

                const uint64_t temporalStateBytes =
                    static_cast<uint64_t>(maxGeneralGaussianSplatSourceCount) * sizeof(uint32_t);
                if (!generalGaussianSplatTemporalStateBuffer ||
                    static_cast<uint64_t>(generalGaussianSplatTemporalStateBuffer->getSize()) < temporalStateBytes)
                {
                    generalGaussianSplatTemporalStateBuffer =
                        createRef<rhi::StorageBuffer>(rd.createStorageBuffer(temporalStateBytes));
                    generalGaussianSplatTemporalStateNeedsReset = true;
                }
                generalGaussianSplatTemporalStateCapacity = maxGeneralGaussianSplatSourceCount;
            }

            if (maxGeneralGaussianSplatPoints > 0u)
            {
                const uint64_t selectedBytes =
                    static_cast<uint64_t>(maxGeneralGaussianSplatPoints) * sizeof(GpuGeneralGaussianSplatSelectedSource);
                if (!generalGaussianSplatSelectedSourceBuffer ||
                    static_cast<uint64_t>(generalGaussianSplatSelectedSourceBuffer->getSize()) < selectedBytes)
                {
                    generalGaussianSplatSelectedSourceBuffer =
                        createRef<rhi::StorageBuffer>(rd.createStorageBuffer(selectedBytes));
                }
            }

            if (maxGeneralGaussianSplatVisibleSplats == 0u)
                return;

            {
                constexpr uint32_t kCoverageReleaseTextureFloorFlag = 32u;
                const bool coverageTextureFloor =
                    (generalGaussianSplatCoverageStableReleaseFlags & kCoverageReleaseTextureFloorFlag) != 0u;
                const uint32_t coverageWidth =
                    coverageTextureFloor ?
                        std::clamp(generalGaussianSplatCoverageStableReleaseTileGridX, 1u, 1024u) :
                        1u;
                const uint32_t coverageHeight =
                    coverageTextureFloor ?
                        std::clamp(generalGaussianSplatCoverageStableReleaseTileGridY, 1u, 1024u) :
                        1u;
                const uint32_t coverageTexels = coverageWidth * coverageHeight;
                const uint64_t coverageBytes =
                    static_cast<uint64_t>(coverageTexels) * 4ull * sizeof(uint32_t);
                if (!generalGaussianSplatCoverageTextureBuffer ||
                    static_cast<uint64_t>(generalGaussianSplatCoverageTextureBuffer->getSize()) < coverageBytes ||
                    generalGaussianSplatCoverageTextureCapacity != coverageTexels)
                {
                    generalGaussianSplatCoverageTextureBuffer =
                        createRef<rhi::StorageBuffer>(rd.createStorageBuffer(coverageBytes));
                    generalGaussianSplatCoverageTextureCapacity = coverageTexels;
                    generalGaussianSplatCoverageTextureNeedsReset = true;
                }
            }

            const uint64_t visibleBytes = static_cast<uint64_t>(maxGeneralGaussianSplatVisibleSplats) *
                                          sizeof(GpuGeneralGaussianSplatVisibleSplat);
            // vk_radix_sort may bind a small implementation-side sentinel tail for
            // key/value buffers, so keep the external sort inputs padded beyond
            // the visible splat cap.
            const uint64_t sortBytes =
                (static_cast<uint64_t>(maxGeneralGaussianSplatVisibleSplats) + 3ull) * sizeof(uint32_t);

            if (!generalGaussianSplatVisibleSplatBuffer ||
                static_cast<uint64_t>(generalGaussianSplatVisibleSplatBuffer->getSize()) < visibleBytes)
            {
                generalGaussianSplatVisibleSplatBuffer = createRef<rhi::StorageBuffer>(rd.createStorageBuffer(visibleBytes));
            }

            if (!generalGaussianSplatSortKeyBuffer ||
                static_cast<uint64_t>(generalGaussianSplatSortKeyBuffer->getSize()) < sortBytes)
            {
                generalGaussianSplatSortKeyBuffer = createRef<rhi::StorageBuffer>(rd.createStorageBuffer(sortBytes));
            }

            if (!generalGaussianSplatSortIndexBuffer ||
                static_cast<uint64_t>(generalGaussianSplatSortIndexBuffer->getSize()) < sortBytes)
            {
                generalGaussianSplatSortIndexBuffer = createRef<rhi::StorageBuffer>(rd.createStorageBuffer(sortBytes));
            }

            if (!generalGaussianSplatVisibleCountBuffer ||
                static_cast<uint64_t>(generalGaussianSplatVisibleCountBuffer->getSize()) < sizeof(uint32_t))
            {
                generalGaussianSplatVisibleCountBuffer =
                    createRef<rhi::StorageBuffer>(rd.createStorageBuffer(sizeof(uint32_t)));
            }

            constexpr uint64_t kEcsptCounterBytes = sizeof(uint32_t) * 4096ull;
            if (!generalGaussianSplatEcsptCounterBuffer ||
                static_cast<uint64_t>(generalGaussianSplatEcsptCounterBuffer->getSize()) < kEcsptCounterBytes)
            {
                generalGaussianSplatEcsptCounterBuffer =
                    createRef<rhi::StorageBuffer>(
                        rd.createStorageBufferWithUsage(kEcsptCounterBytes, rhi::BufferUsage::eTransferSrc));
            }
            if (!generalGaussianSplatEcsptCounterReadbackBuffer ||
                static_cast<uint64_t>(generalGaussianSplatEcsptCounterReadbackBuffer->getSize()) < kEcsptCounterBytes)
            {
                generalGaussianSplatEcsptCounterReadbackBuffer =
                    createRef<rhi::StorageBuffer>(
                        rd.createStorageBufferWithUsage(kEcsptCounterBytes,
                                                        rhi::BufferUsage::eTransferDst,
                                                        rhi::AllocationHints::eRandomAccess));
            }

            constexpr uint64_t kDispatchArgsBytes = sizeof(uint32_t) * 4ull;
            if (!generalGaussianSplatDispatchArgsBuffer ||
                static_cast<uint64_t>(generalGaussianSplatDispatchArgsBuffer->getSize()) < kDispatchArgsBytes)
            {
                generalGaussianSplatDispatchArgsBuffer = createRef<rhi::StorageBuffer>(
                    rd.createStorageBufferWithUsage(kDispatchArgsBytes, rhi::BufferUsage::eIndirectBuffer));
            }

            if (!generalGaussianSplatIndirectBuffer.has_value() || generalGaussianSplatIndirectBuffer->getCapacity() < 1u)
            {
                generalGaussianSplatIndirectBuffer =
                    rd.createDrawIndirectBufferByCount(1u, rhi::DrawIndirectType::eNonIndexed);
            }

            if (!generalGaussianSplatSorter.has_value() || !static_cast<bool>(*generalGaussianSplatSorter) ||
                generalGaussianSplatSorter->getMaxElementCount() < maxGeneralGaussianSplatVisibleSplats)
            {
                generalGaussianSplatSorter = rd.createRadixSorter(maxGeneralGaussianSplatVisibleSplats);
            }

            if (generalGaussianSplatFoveatedLayeredCompositeEnabled)
            {
                for (auto& buffer : generalGaussianSplatFoveatedVisibleSplatBuffers)
                {
                    if (!buffer || static_cast<uint64_t>(buffer->getSize()) < visibleBytes)
                        buffer = createRef<rhi::StorageBuffer>(rd.createStorageBuffer(visibleBytes));
                }
                for (auto& buffer : generalGaussianSplatFoveatedSortKeyBuffers)
                {
                    if (!buffer || static_cast<uint64_t>(buffer->getSize()) < sortBytes)
                        buffer = createRef<rhi::StorageBuffer>(rd.createStorageBuffer(sortBytes));
                }
                for (auto& buffer : generalGaussianSplatFoveatedSortIndexBuffers)
                {
                    if (!buffer || static_cast<uint64_t>(buffer->getSize()) < sortBytes)
                        buffer = createRef<rhi::StorageBuffer>(rd.createStorageBuffer(sortBytes));
                }
                for (auto& buffer : generalGaussianSplatFoveatedVisibleCountBuffers)
                {
                    if (!buffer || static_cast<uint64_t>(buffer->getSize()) < sizeof(uint32_t))
                        buffer = createRef<rhi::StorageBuffer>(rd.createStorageBuffer(sizeof(uint32_t)));
                }
                for (auto& buffer : generalGaussianSplatFoveatedIndirectBuffers)
                {
                    if (!buffer.has_value() || buffer->getCapacity() < 1u)
                        buffer = rd.createDrawIndirectBufferByCount(1u, rhi::DrawIndirectType::eNonIndexed);
                }
            }

            if (generalGaussianSplatSorter.has_value() && static_cast<bool>(*generalGaussianSplatSorter))
            {
                const auto req = generalGaussianSplatSorter->getKeyValueStorageRequirements();
                if (req.size > 0u &&
                    (!generalGaussianSplatSortStorageBuffer ||
                     static_cast<uint64_t>(generalGaussianSplatSortStorageBuffer->getSize()) < req.size))
                {
                    generalGaussianSplatSortStorageBuffer = createRef<rhi::StorageBuffer>(
                        rd.createStorageBufferWithUsage(req.size, req.usage));
                }
            }
        }

        void ensureGeneralGaussianSplatProjectedCostBuffers(rhi::RenderDevice& rd, const uint32_t chunkCount)
        {
            if (chunkCount == 0u)
                return;

            constexpr uint64_t kProjectedCostChunkRecordBytes = 32ull;
            const uint64_t bytes = static_cast<uint64_t>(chunkCount) * kProjectedCostChunkRecordBytes;
            if (!generalGaussianSplatProjectedCostChunkBuffer ||
                static_cast<uint64_t>(generalGaussianSplatProjectedCostChunkBuffer->getSize()) < bytes)
            {
                generalGaussianSplatProjectedCostChunkBuffer =
                    createRef<rhi::StorageBuffer>(rd.createStorageBufferWithUsage(bytes, rhi::BufferUsage::eTransferSrc));
            }
            if (!generalGaussianSplatProjectedCostReadbackBuffer ||
                static_cast<uint64_t>(generalGaussianSplatProjectedCostReadbackBuffer->getSize()) < bytes)
            {
                generalGaussianSplatProjectedCostReadbackBuffer =
                    createRef<rhi::StorageBuffer>(rd.createStorageBufferWithUsage(bytes,
                                                                                  rhi::BufferUsage::eTransferDst,
                                                                                  rhi::AllocationHints::eRandomAccess));
            }
            generalGaussianSplatProjectedCostChunkCapacity = chunkCount;
        }

    };
} // namespace vultra::resource

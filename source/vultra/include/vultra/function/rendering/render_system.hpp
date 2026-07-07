#pragma once

#include "vultra/core/engine/engine_subsystem.hpp"
#include "vultra/function/framegraph/transient_resources.hpp"
#include "vultra/function/rendering/gpu_scene_dirty_tracker.hpp"
#include "vultra/function/rendering/framework/prepared_render_data.hpp"
#include "vultra/function/rendering/framework/render_frame_resources.hpp"
#include "vultra/function/rendering/runtime_profiler.hpp"
#include "vultra/function/rendering/render_structs.hpp"
#include "vultra/function/rendering/srp/renderer.hpp"
#include "vultra/function/resource/gpu_scene_database.hpp"
#include "vultra/function/resource/gpu_scene_view.hpp"
#include "vultra/function/services/asset_service.hpp"
#include "vultra/function/services/camera_service.hpp"
#include "vultra/function/services/render_service.hpp"
#include "vultra/function/services/world_service.hpp"

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace vultra
{
    namespace rhi
    {
        class Texture;
    }

    struct GaussianSplatFoveatedP95ControllerState
    {
        std::vector<double> gpuWindow;
        double              p95Ema {-1.0};
        uint32_t            growCounter {0};
        uint32_t            shrinkCounter {0};
        double              dynamicFrameMsEma {-1.0};
        uint32_t            dynamicGrowCounter {0};
        uint32_t            dynamicShrinkCounter {0};
        GaussianSplatFoveatedAdaptationMode mode {GaussianSplatFoveatedAdaptationMode::eFixed};
    };

    struct GaussianSplatShPopAggregateState
    {
        bool                    valid {false};
        glm::vec2               gaze {0.5f, 0.5f};
        glm::uvec3              shDegrees {3u, 3u, 3u};
        std::array<uint32_t, 3> ringCounts {0u, 0u, 0u};
        uint32_t                selectedCount {0u};
        uint32_t                guardRaisedCount {0u};
        uint32_t                downgradeCandidateFrames {0u};
    };

    // RenderSystem (SRP host):
    // - Reads cooked cameras from CameraSystem
    // - Resolves renderer per camera.rendererKey
    // - Builds FrameGraph per camera using RenderView + build/exec contexts
    // - Compiles & executes
    class RenderSystem final : public EngineSubsystem, public IRenderService
    {
    public:
        ENGINE_SUBSYSTEM(RenderSystem)

        bool onInit() override;
        void onShutdown() override;

        void onPreRender() override;
        void onRender() override;
        void onPostRender() override;
        void onPresent() override;

        // IRenderService
        void registerRenderer(Ref<Renderer> renderer) override;
        void renderFrame() override;
        void onResize(uint32_t width, uint32_t height) override;

        // Optional: set default renderer key used if camera.rendererKey not found
        void setDefaultRendererKey(std::string key) { m_DefaultRendererKey = std::move(key); }

        // Optional: set fallback backbuffer target (can be used for offline rendering)
        void setBackbufferTarget(rhi::Texture* tex) { m_Backbuffer = tex; }

        // Cooked render world (read-only for renderer)
        const RenderWorld& renderWorld() const { return m_RenderWorldFront; }
        RuntimeProfiler*   runtimeProfiler() override { return &m_RuntimeProfiler; }
        GaussianSplatRenderSettings&       gaussianSplatSettings() override { return m_GaussianSplatSettings; }
        const GaussianSplatRenderSettings& gaussianSplatSettings() const override { return m_GaussianSplatSettings; }
        const GaussianSplatFrameStats&     gaussianSplatFrameStats() const override { return m_GaussianSplatStats; }
        const rhi::Texture*                gaussianSplatLastAlphaTexture() const override
        {
            return m_GpuSceneViewFront.generalGaussianSplatLastAlphaTexture;
        }

    private:
        Ref<Renderer> resolveRenderer(const RenderCamera& cam) const;

    private:
        bool m_SkipRender {false};

        std::unordered_map<std::string, Ref<Renderer>> m_Renderers;
        std::string                                    m_DefaultRendererKey {"builtin"};

        rhi::Texture*                                   m_Backbuffer {nullptr};
        std::unique_ptr<framegraph::TransientResources> m_TransientResources {nullptr};

        RenderWorld m_RenderWorldFront {};
        RenderWorld m_RenderWorldBack {};

        resource::GpuSceneDatabase m_GpuSceneDatabaseFront {};
        resource::GpuSceneDatabase m_GpuSceneDatabaseBack {};

        resource::GpuSceneView m_GpuSceneViewFront {};
        resource::GpuSceneView m_GpuSceneViewBack {};

        uint64_t m_FrameCounter {0};

        RenderFrameResources m_FrameResources {};
        FrameRenderData      m_PreparedFrameData {};

        Samplers m_Samplers;

        bool                 m_EnableGpuDrivenMeshletPipeline {true};
        GpuSceneDirtyTracker m_GpuSceneDirtyTracker;
        RuntimeProfiler      m_RuntimeProfiler;
        GaussianSplatRenderSettings m_GaussianSplatSettings;
        GaussianSplatRenderSettings m_AppliedGaussianSplatSettings;
        GaussianSplatFrameStats     m_GaussianSplatStats;
        GaussianSplatShPopAggregateState m_GaussianShPopAggregateState;
        GaussianSplatFoveatedP95ControllerState m_FoveatedP95ControllerState;
        std::vector<uint16_t> m_GaussianFoveatedScoreResidency;

        bool m_GaussianGazeAnchorValid {false};
        glm::vec2 m_GaussianGazeAnchorGaze {0.5f, 0.5f};
        glm::vec2 m_GaussianGazeAnchorPreviousGaze {0.5f, 0.5f};
        uint64_t m_GaussianGazeAnchorLastUpdateFrame {0u};
        uint32_t m_GaussianGazeAnchorUpdateEventCount {0u};
        uint32_t m_GaussianGazeAnchorDeadbandViolationCount {0u};
        bool m_GaussianEccStochasticTransitionValid {false};
        glm::vec2 m_GaussianEccStochasticOldGaze {0.5f, 0.5f};
        glm::vec2 m_GaussianEccStochasticNewGaze {0.5f, 0.5f};
        uint64_t m_GaussianEccStochasticLastUpdateFrame {0u};
        uint32_t m_GaussianEccStochasticUpdateEventCount {0u};
        struct GaussianCachedSelectionOracleCandidate
        {
            resource::GpuGeneralGaussianSplatSelectedSource selection {};
            uint32_t sourceId {0u};
            uint32_t rank {0u};
            glm::vec2 centerNdc {0.0f};
            float angularFootprintRadiusDegrees {0.0f};
            float opacityProxy {0.0f};
            double projectedAreaPx {0.0};
            uint64_t tileCost {0u};
            uint64_t hashKey {0u};
            float hashThreshold {0.0f};
            bool visibleAtCache {false};
            bool staticRandomSelected {false};
            bool matchedNullSelected {false};
        };

        bool m_GaussianCachedSelectionOracleValid {false};
        GaussianSplatCachedSelectionMembershipMode m_GaussianCachedSelectionOracleMode {
            GaussianSplatCachedSelectionMembershipMode::eDynamicDeterministic};
        uint32_t m_GaussianCachedSelectionOracleSeed {0u};
        uint32_t m_GaussianCachedSelectionOracleSourceCount {0u};
        uint32_t m_GaussianCachedSelectionOracleCandidateCount {0u};
        uint32_t m_GaussianCachedSelectionOracleActiveCount {0u};
        uint32_t m_GaussianCachedSelectionOracleCacheFrameIndex {UINT32_MAX};
        uint32_t m_GaussianCachedSelectionOracleStaticTargetCount {UINT32_MAX};
        bool     m_GaussianCachedSelectionOracleMatchedNullValid {false};
        uint32_t m_GaussianCachedSelectionOracleMatchedNullFrameIndex {UINT32_MAX};
        std::vector<GaussianCachedSelectionOracleCandidate> m_GaussianCachedSelectionOracleCandidates;
        uint64_t m_GaussianProjectedCostCacheSalt {0u};
        uint64_t m_GaussianProjectedCostCacheEpoch {0u};
    };

    // Cook World into RenderWorld.
    class RenderWorldCooker
    {
    public:
        static void cook(World& world, IAssetService& assets, RenderWorld& out);
    };
} // namespace vultra

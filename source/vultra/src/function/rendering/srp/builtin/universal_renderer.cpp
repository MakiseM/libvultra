#include "vultra/function/rendering/srp/builtin/universal_renderer.hpp"
#include "vultra/core/rhi/structs/render_backend_api.hpp"
#include "vultra/function/rendering/runtime_profiler.hpp"
#include "vultra/function/rendering/render_structs.hpp"
#include "vultra/function/rendering/srp/builtin/features/compatibility_basecolor_feature.hpp"
#include "vultra/function/rendering/srp/builtin/features/final_composition_feature.hpp"
#include "vultra/function/rendering/srp/builtin/features/general_gaussian_splat_feature.hpp"
#include "vultra/function/rendering/srp/builtin/features/meshlet_feature.hpp"
#include "vultra/function/rendering/srp/builtin/features/test_feature.hpp"
#include "vultra/function/services/camera_service.hpp"
#include "vultra/function/services/gpu_resource_service.hpp"
#include "vultra/function/services/render_backend_service.hpp"
#include "vultra/function/services/render_service.hpp"
#ifdef VULTRA_ENABLE_RENDERDOC
#include "vultra/function/services/frame_debugger_service.hpp"
#endif

#include <IconsMaterialDesignIcons.h>
#include <imgui.h>
#include <implot/implot.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>

namespace vultra
{
    namespace
    {
        [[nodiscard]] std::string formatBytes(const uint64_t bytes)
        {
            constexpr double kKB = 1024.0;
            constexpr double kMB = 1024.0 * 1024.0;
            constexpr double kGB = 1024.0 * 1024.0 * 1024.0;

            const double v = static_cast<double>(bytes);
            char         buf[64] {};
            if (v >= kGB)
            {
                std::snprintf(buf, sizeof(buf), "%.2f GB", v / kGB);
            }
            else if (v >= kMB)
            {
                std::snprintf(buf, sizeof(buf), "%.2f MB", v / kMB);
            }
            else if (v >= kKB)
            {
                std::snprintf(buf, sizeof(buf), "%.2f KB", v / kKB);
            }
            else
            {
                std::snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(bytes));
            }
            return std::string(buf);
        }

        [[nodiscard]] const char* gaussianBaselineModeLabel(const GaussianSplatBaselineMode mode)
        {
            switch (mode)
            {
                case GaussianSplatBaselineMode::eBaseline:
                    return "Baseline";
                case GaussianSplatBaselineMode::eOrderedClod:
                    return "Ordered CLOD";
            }
            return "Unknown";
        }

        [[nodiscard]] const char* gaussianFoveatedRenderModeLabel(const GaussianSplatFoveatedRenderMode mode)
        {
            switch (mode)
            {
                case GaussianSplatFoveatedRenderMode::eSinglePass:
                    return "Single Pass";
                case GaussianSplatFoveatedRenderMode::eLayeredComposite:
                    return "Layered Composite";
            }
            return "Unknown";
        }

        [[nodiscard]] const char* gaussianLodBudgetModeLabel(const GaussianSplatLodBudgetMode mode)
        {
            switch (mode)
            {
                case GaussianSplatLodBudgetMode::eCount:
                    return "Count";
                case GaussianSplatLodBudgetMode::eProjectedTileCost:
                    return "Projected Tile Cost";
                case GaussianSplatLodBudgetMode::eFoveatedScore:
                    return "Foveated Score";
                case GaussianSplatLodBudgetMode::eCoverageBinScore:
                    return "Coverage Bin Score";
            }
            return "Unknown";
        }

        [[nodiscard]] const char* gaussianProjectedCostBuildModeLabel(const GaussianProjectedCostBuildMode mode)
        {
            switch (mode)
            {
                case GaussianProjectedCostBuildMode::eCpu:
                    return "CPU";
                case GaussianProjectedCostBuildMode::eGpuSync:
                    return "GPU Sync";
            }
            return "Unknown";
        }

        [[nodiscard]] const char* gaussianFoveatedCoverageGuardModeLabel(
            const GaussianSplatFoveatedCoverageGuardMode mode)
        {
            switch (mode)
            {
                case GaussianSplatFoveatedCoverageGuardMode::eOff:
                    return "Off";
                case GaussianSplatFoveatedCoverageGuardMode::eGlobal:
                    return "Global";
                case GaussianSplatFoveatedCoverageGuardMode::eLocalBounded:
                    return "Local Bounded";
                case GaussianSplatFoveatedCoverageGuardMode::eRiskTriggered:
                    return "Risk Triggered";
            }
            return "Unknown";
        }

        [[nodiscard]] const char* gaussianFoveatedAdaptationModeLabel(
            const GaussianSplatFoveatedAdaptationMode mode)
        {
            switch (mode)
            {
                case GaussianSplatFoveatedAdaptationMode::eFixed:
                    return "Fixed";
                case GaussianSplatFoveatedAdaptationMode::eDynamicBudget:
                    return "Dynamic Budget";
                case GaussianSplatFoveatedAdaptationMode::eStabilityAwareBudget:
                    return "Stability-Aware Budget";
                case GaussianSplatFoveatedAdaptationMode::eDynamicRange:
                    return "Dynamic Range";
                case GaussianSplatFoveatedAdaptationMode::eProgressiveCenterOut:
                    return "Progressive Center-Out";
                case GaussianSplatFoveatedAdaptationMode::eProgressiveGreedy:
                    return "Progressive Greedy";
            }
            return "Unknown";
        }

        [[nodiscard]] const char* gaussianFoveatedDistributionLabel(
            const GaussianSplatFoveatedDistribution distribution)
        {
            switch (distribution)
            {
                case GaussianSplatFoveatedDistribution::eHardRing:
                    return "Hard Ring";
                case GaussianSplatFoveatedDistribution::eSmoothstep:
                    return "Smoothstep";
                case GaussianSplatFoveatedDistribution::eGaussian:
                    return "Gaussian";
                case GaussianSplatFoveatedDistribution::eExponential:
                    return "Exponential";
                case GaussianSplatFoveatedDistribution::eInversePower:
                    return "Inverse Power";
                case GaussianSplatFoveatedDistribution::eLogPolar:
                    return "Log Polar";
                case GaussianSplatFoveatedDistribution::eContinuousScheduler:
                    return "Continuous Scheduler";
                case GaussianSplatFoveatedDistribution::eFoveaProtectedContinuous:
                    return "Fovea-Protected Continuous";
                case GaussianSplatFoveatedDistribution::eCortical:
                    return "Cortical";
                case GaussianSplatFoveatedDistribution::eConeDensityFitted:
                    return "Cone Density Fitted";
            }
            return "Unknown";
        }

        [[nodiscard]] const char* gaussianShLodGuardModeLabel(const GaussianSplatShLodGuardMode mode)
        {
            switch (mode)
            {
                case GaussianSplatShLodGuardMode::eOff:
                    return "Off";
                case GaussianSplatShLodGuardMode::eEnergy:
                    return "Energy";
                case GaussianSplatShLodGuardMode::eProjectedCost:
                    return "Projected Cost";
                case GaussianSplatShLodGuardMode::eEnergyProjectedCost:
                    return "Energy + Projected Cost";
            }
            return "Unknown";
        }

        [[nodiscard]] const char* gaussianShStorageLayoutLabel(const GaussianSplatShStorageLayout layout)
        {
            switch (layout)
            {
                case GaussianSplatShStorageLayout::eMonolithic:
                    return "Monolithic";
                case GaussianSplatShStorageLayout::eSplitBands:
                    return "Split Bands";
            }
            return "Unknown";
        }

        [[nodiscard]] bool shDegreesEqual(const glm::uvec3 degrees,
                                          const uint32_t   center,
                                          const uint32_t   mid,
                                          const uint32_t   outer)
        {
            return degrees.x == center && degrees.y == mid && degrees.z == outer;
        }

        [[nodiscard]] bool gaussianFloatEqual(const float a, const float b)
        {
            return std::abs(a - b) <= 1e-4f;
        }

        [[nodiscard]] bool gaussianVec3Equal(const glm::vec3 a, const glm::vec3 b)
        {
            return gaussianFloatEqual(a.x, b.x) && gaussianFloatEqual(a.y, b.y) &&
                   gaussianFloatEqual(a.z, b.z);
        }

        [[nodiscard]] bool gaussianFullSelectionActive(const GaussianSplatRenderSettings& settings)
        {
            return settings.orderedClodEnabled() && settings.lodBudgetMode == GaussianSplatLodBudgetMode::eCount &&
                   settings.lodBudget == 0u && gaussianFloatEqual(settings.clodLevel, 1.0f);
        }

        void applyGaussianFullSelectionSettings(GaussianSplatRenderSettings& settings)
        {
            settings.baselineMode                         = GaussianSplatBaselineMode::eOrderedClod;
            settings.lodBudget                            = 0u;
            settings.clodLevel                            = 1.0f;
            settings.lodBudgetMode                        = GaussianSplatLodBudgetMode::eCount;
            settings.projectedCostBudget                  = 0u;
            settings.projectedCostBudgetRatio             = 0.0f;
            settings.foveatedClodEnabled                  = false;
            settings.foveatedManualGazeControlEnabled     = false;
            settings.foveatedXrGazeEnabled                = false;
            settings.foveatedCoverageCompensationEnabled  = false;
            settings.foveatedRenderMode                   = GaussianSplatFoveatedRenderMode::eSinglePass;
            settings.foveatedRingLevels                   = glm::vec3 {1.0f, 1.0f, 1.0f};
            settings.foveatedResolutionScales             = glm::vec3 {1.0f, 1.0f, 1.0f};
            settings.foveatedCoverageGuardMode            = GaussianSplatFoveatedCoverageGuardMode::eOff;
            settings.foveatedCoverageProtectionDegrees    = 0.0f;
            settings.foveatedCoverageGuardBudgetRatio     = 0.0f;
            settings.foveatedCoverageGuardMaxAdds         = 0u;
            settings.foveatedTemporalHysteresisEnabled    = false;
            settings.foveatedBoundarySmoothingEnabled     = false;
            settings.foveatedAdaptationMode               = GaussianSplatFoveatedAdaptationMode::eFixed;
        }

        void applyGaussianFullSelectionGazeSettings(GaussianSplatRenderSettings& settings)
        {
            applyGaussianFullSelectionSettings(settings);
            settings.foveatedClodEnabled                  = true;
            settings.foveatedManualGazeControlEnabled     = true;
        }

        void applyGaussianSystemGazeSettings(GaussianSplatRenderSettings& settings)
        {
            applyGaussianFullSelectionSettings(settings);
            settings.foveatedClodEnabled              = true;
            settings.foveatedManualGazeControlEnabled = true;
            settings.foveatedRingDegrees              = glm::vec2 {12.0f, 32.0f};
            settings.foveatedRingLevels               = glm::vec3 {1.0f, 0.40f, 0.15f};
            settings.foveatedResolutionScales         = glm::vec3 {1.0f, 0.75f, 0.50f};
            settings.foveatedTransitionDegrees        = 8.0f;
            settings.foveatedDistribution             = GaussianSplatFoveatedDistribution::eGaussian;
            settings.foveatedTemporalHysteresisEnabled = true;
            settings.foveatedBoundarySmoothingEnabled  = true;
            settings.foveatedTemporalResidencyFrames   = 6u;
            settings.foveatedTemporalHysteresisRatio   = 0.25f;
            settings.foveatedBoundarySmoothingRatio    = 0.18f;
        }

        void applyGaussianFullSh3(GaussianSplatRenderSettings& settings)
        {
            settings.foveatedShLodEnabled      = false;
            settings.foveatedShLodDegrees      = glm::uvec3 {3u, 3u, 3u};
            settings.foveatedShLodGuardMode    = GaussianSplatShLodGuardMode::eOff;
            settings.shStorageLayout           = GaussianSplatShStorageLayout::eMonolithic;
            settings.shDegreeHysteresisEnabled = false;
            settings.shPopLogEnabled           = false;
        }

        void applyGaussianUniformShDegree(GaussianSplatRenderSettings& settings, const uint32_t degree)
        {
            const uint32_t clampedDegree       = std::clamp(degree, 0u, 3u);
            settings.foveatedShLodEnabled      = true;
            settings.foveatedShLodDegrees      = glm::uvec3 {clampedDegree, clampedDegree, clampedDegree};
            settings.foveatedShLodGuardMode    = GaussianSplatShLodGuardMode::eOff;
            settings.shStorageLayout           = GaussianSplatShStorageLayout::eMonolithic;
            settings.shDegreeHysteresisEnabled = false;
            settings.shPopLogEnabled           = false;
        }

        void applyGaussianGazeShDegree(GaussianSplatRenderSettings& settings,
                                       const glm::uvec3             degrees,
                                       const GaussianSplatShLodGuardMode guardMode)
        {
            settings.foveatedShLodEnabled      = true;
            settings.foveatedShLodDegrees      = glm::clamp(degrees, glm::uvec3 {0u}, glm::uvec3 {3u});
            settings.foveatedShLodGuardMode    = guardMode;
            settings.shStorageLayout           = GaussianSplatShStorageLayout::eMonolithic;
            settings.shDegreeHysteresisEnabled = false;
            settings.shPopLogEnabled           = false;
        }

        [[nodiscard]] bool gaussianSystemGazePresetActive(const GaussianSplatRenderSettings& settings)
        {
            return gaussianFullSelectionActive(settings) && settings.foveatedClodEnabled &&
                   gaussianVec3Equal(settings.foveatedRingLevels, glm::vec3 {1.0f, 0.40f, 0.15f}) &&
                   gaussianVec3Equal(settings.foveatedResolutionScales, glm::vec3 {1.0f, 0.75f, 0.50f});
        }

        [[nodiscard]] int gaussianDemoPresetIndex(const GaussianSplatRenderSettings& settings)
        {
            constexpr int kCustomPreset = 8;
            const auto    degrees       = glm::clamp(settings.foveatedShLodDegrees,
                                                     glm::uvec3 {0u},
                                                     glm::uvec3 {3u});
            const bool guardOff =
                settings.foveatedShLodGuardMode == GaussianSplatShLodGuardMode::eOff;
            const bool energyGuard =
                settings.foveatedShLodGuardMode == GaussianSplatShLodGuardMode::eEnergy;
            const bool fullSelection = gaussianFullSelectionActive(settings);
            const bool noGazeSelection = !settings.foveatedClodEnabled;

            if (fullSelection && noGazeSelection && !settings.foveatedShLodEnabled)
                return 0;
            if (fullSelection && noGazeSelection && settings.foveatedShLodEnabled && guardOff)
            {
                if (shDegreesEqual(degrees, 2u, 2u, 2u))
                    return 1;
                if (shDegreesEqual(degrees, 1u, 1u, 1u))
                    return 2;
                if (shDegreesEqual(degrees, 0u, 0u, 0u))
                    return 3;
            }

            if (fullSelection && settings.foveatedClodEnabled &&
                gaussianVec3Equal(settings.foveatedRingLevels, glm::vec3 {1.0f, 1.0f, 1.0f}))
            {
                if (settings.foveatedShLodEnabled && guardOff && shDegreesEqual(degrees, 3u, 2u, 1u))
                    return 4;
            }

            if (gaussianSystemGazePresetActive(settings))
            {
                if (!settings.foveatedShLodEnabled)
                    return 5;
                if (shDegreesEqual(degrees, 3u, 2u, 1u))
                {
                    if (guardOff)
                        return 6;
                    if (energyGuard)
                        return 7;
                }
            }

            return kCustomPreset;
        }

        void applyGaussianDemoPreset(GaussianSplatRenderSettings& settings, const int presetIndex)
        {
            switch (presetIndex)
            {
                case 0:
                    applyGaussianFullSelectionSettings(settings);
                    applyGaussianFullSh3(settings);
                    break;
                case 1:
                    applyGaussianFullSelectionSettings(settings);
                    applyGaussianUniformShDegree(settings, 2u);
                    break;
                case 2:
                    applyGaussianFullSelectionSettings(settings);
                    applyGaussianUniformShDegree(settings, 1u);
                    break;
                case 3:
                    applyGaussianFullSelectionSettings(settings);
                    applyGaussianUniformShDegree(settings, 0u);
                    break;
                case 4:
                    applyGaussianFullSelectionGazeSettings(settings);
                    applyGaussianGazeShDegree(settings, glm::uvec3 {3u, 2u, 1u}, GaussianSplatShLodGuardMode::eOff);
                    break;
                case 5:
                    applyGaussianSystemGazeSettings(settings);
                    applyGaussianFullSh3(settings);
                    break;
                case 6:
                    applyGaussianSystemGazeSettings(settings);
                    applyGaussianGazeShDegree(settings, glm::uvec3 {3u, 2u, 1u}, GaussianSplatShLodGuardMode::eOff);
                    break;
                case 7:
                    applyGaussianSystemGazeSettings(settings);
                    applyGaussianGazeShDegree(settings, glm::uvec3 {3u, 2u, 1u}, GaussianSplatShLodGuardMode::eEnergy);
                    break;
                default:
                    break;
            }
        }

        [[nodiscard]] int gaussianShStagePresetIndex(const GaussianSplatRenderSettings& settings)
        {
            constexpr int kCustomPreset = 6;
            if (!settings.foveatedShLodEnabled)
                return 0;

            const auto degrees = glm::clamp(settings.foveatedShLodDegrees, glm::uvec3 {0u}, glm::uvec3 {3u});
            const bool monolithic = settings.shStorageLayout == GaussianSplatShStorageLayout::eMonolithic;
            const bool split      = settings.shStorageLayout == GaussianSplatShStorageLayout::eSplitBands;
            const bool guardOff   = settings.foveatedShLodGuardMode == GaussianSplatShLodGuardMode::eOff;
            const bool energyGuard = settings.foveatedShLodGuardMode == GaussianSplatShLodGuardMode::eEnergy;
            const bool stabilized = settings.shDegreeHysteresisEnabled && settings.shPopLogEnabled;

            if (shDegreesEqual(degrees, 3u, 2u, 1u) && guardOff && monolithic && !stabilized)
                return 1;
            if (shDegreesEqual(degrees, 3u, 2u, 1u) && energyGuard && monolithic && !stabilized)
                return 2;
            if (shDegreesEqual(degrees, 3u, 2u, 1u) && energyGuard && split && !stabilized)
                return 3;
            if (shDegreesEqual(degrees, 3u, 1u, 0u) && energyGuard && split && !stabilized)
                return 4;
            if (shDegreesEqual(degrees, 3u, 2u, 1u) && energyGuard && split && stabilized)
                return 5;
            return kCustomPreset;
        }

        void applyGaussianShStagePreset(GaussianSplatRenderSettings& settings, const int presetIndex)
        {
            applyGaussianFullSelectionGazeSettings(settings);

            if (presetIndex == 0)
            {
                settings.foveatedShLodEnabled      = false;
                settings.foveatedShLodDegrees      = glm::uvec3 {3u, 3u, 3u};
                settings.foveatedShLodGuardMode    = GaussianSplatShLodGuardMode::eOff;
                settings.shDegreeHysteresisEnabled = false;
                settings.shPopLogEnabled           = false;
                return;
            }

            if (presetIndex < 1 || presetIndex > 5)
                return;

            settings.foveatedShLodEnabled      = true;
            settings.foveatedShLodDegrees      = glm::uvec3 {3u, 2u, 1u};
            settings.foveatedShLodGuardMode    = GaussianSplatShLodGuardMode::eEnergy;
            settings.shStorageLayout           = GaussianSplatShStorageLayout::eMonolithic;
            settings.shDegreeHysteresisEnabled = false;
            settings.shPopLogEnabled           = false;

            switch (presetIndex)
            {
                case 1:
                    settings.foveatedShLodGuardMode = GaussianSplatShLodGuardMode::eOff;
                    break;
                case 2:
                    break;
                case 3:
                    settings.shStorageLayout = GaussianSplatShStorageLayout::eSplitBands;
                    break;
                case 4:
                    settings.foveatedShLodDegrees = glm::uvec3 {3u, 1u, 0u};
                    settings.shStorageLayout      = GaussianSplatShStorageLayout::eSplitBands;
                    break;
                case 5:
                    settings.shStorageLayout           = GaussianSplatShStorageLayout::eSplitBands;
                    settings.shDegreeHysteresisEnabled = true;
                    settings.shPopLogEnabled           = true;
                    break;
                default:
                    break;
            }
        }

        [[nodiscard]] bool gaussianManualGazeControlActive(const GaussianSplatRenderSettings& settings)
        {
            return (settings.foveatedClodActive() ||
                    (settings.shaderAntiPopActive() && settings.shaderAntiPopOrderFree &&
                     settings.foveatedClodEnabled)) &&
                   settings.foveatedManualGazeControlEnabled;
        }

        void drawGaussianManualGazeOverlay(GaussianSplatRenderSettings& settings)
        {
            if (!gaussianManualGazeControlActive(settings))
                return;

            const ImGuiViewport* viewport = ImGui::GetMainViewport();
            if (!viewport || viewport->Size.x <= 0.0f || viewport->Size.y <= 0.0f)
                return;

            auto& io = ImGui::GetIO();
            const ImVec2 mouse = io.MousePos;
            const ImVec2 viewMin = viewport->Pos;
            const ImVec2 viewSize = io.DisplaySize.x > 0.0f && io.DisplaySize.y > 0.0f ? io.DisplaySize :
                                                                                         viewport->Size;
            const ImVec2 viewMax {viewMin.x + viewSize.x, viewMin.y + viewSize.y};
            const bool   mouseInside =
                mouse.x >= viewMin.x && mouse.x <= viewMax.x && mouse.y >= viewMin.y && mouse.y <= viewMax.y;

            if (mouseInside && !io.WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            {
                settings.foveatedGaze.x = std::clamp((mouse.x - viewMin.x) / viewSize.x, 0.0f, 1.0f);
                settings.foveatedGaze.y = std::clamp((mouse.y - viewMin.y) / viewSize.y, 0.0f, 1.0f);
            }

            const ImVec2 gaze {
                viewMin.x + settings.foveatedGaze.x * viewSize.x,
                viewMin.y + settings.foveatedGaze.y * viewSize.y,
            };
            ImDrawList* drawList = ImGui::GetForegroundDrawList();
            drawList->AddCircleFilled(gaze, 5.0f, IM_COL32(255, 32, 32, 255), 24);
            drawList->AddCircle(gaze, 8.0f, IM_COL32(255, 255, 255, 220), 24, 1.5f);
        }

        void drawGazeMarker(ImDrawList& drawList, const ImVec2 imageMin, const ImVec2 imageSize, const glm::vec2 gazeUv)
        {
            if (imageSize.x <= 0.0f || imageSize.y <= 0.0f)
                return;

            const ImVec2 gaze {
                imageMin.x + std::clamp(gazeUv.x, 0.0f, 1.0f) * imageSize.x,
                imageMin.y + std::clamp(gazeUv.y, 0.0f, 1.0f) * imageSize.y,
            };
            drawList.AddCircleFilled(gaze, 5.0f, IM_COL32(255, 32, 32, 255), 24);
            drawList.AddCircle(gaze, 8.0f, IM_COL32(255, 255, 255, 220), 24, 1.5f);
        }

        [[nodiscard]] double findScopeGpuMs(const std::vector<RuntimeProfiler::ScopeNode>& nodes,
                                            const char*                                    namePart)
        {
            for (const auto& node : nodes)
            {
                if (node.name.find(namePart) != std::string::npos && node.gpuTotalMs >= 0.0)
                    return node.gpuTotalMs;
            }
            return -1.0;
        }

        void drawOptionalCounter(const char* label, const uint32_t value)
        {
            if (value == UINT32_MAX)
                ImGui::Text("%s: N/A", label);
            else
                ImGui::Text("%s: %u", label, value);
        }

        void drawGaussianSplatBaselinePanel(IRenderService& renderService, const IRenderBackendService& backendService)
        {
            if (!ImGui::CollapsingHeader("Gaussian Splat Baseline", ImGuiTreeNodeFlags_DefaultOpen))
                return;

            auto&       settings = renderService.gaussianSplatSettings();
            const auto& stats    = renderService.gaussianSplatFrameStats();

            constexpr const char* kModeLabels[] = {"Baseline", "Ordered CLOD"};
            int modeIndex = static_cast<int>(settings.baselineMode);
            if (ImGui::Combo("Mode", &modeIndex, kModeLabels, IM_ARRAYSIZE(kModeLabels)))
            {
                modeIndex             = std::clamp(modeIndex, 0, IM_ARRAYSIZE(kModeLabels) - 1);
                settings.baselineMode = static_cast<GaussianSplatBaselineMode>(modeIndex);
            }

            const bool lodControlsEnabled = settings.lodBudgetEnabled();
            const bool orderedClodEnabled  = settings.orderedClodEnabled();
            if (!orderedClodEnabled)
                ImGui::BeginDisabled();
            constexpr const char* kLodBudgetModeLabels[] = {
                "Count",
                "Projected Tile Cost",
                "Foveated Score",
                "Coverage Bin Score",
            };
            int lodBudgetModeIndex = static_cast<int>(settings.lodBudgetMode);
            if (ImGui::Combo("LOD Budget Mode",
                             &lodBudgetModeIndex,
                             kLodBudgetModeLabels,
                             IM_ARRAYSIZE(kLodBudgetModeLabels)))
            {
                lodBudgetModeIndex = std::clamp(lodBudgetModeIndex, 0, IM_ARRAYSIZE(kLodBudgetModeLabels) - 1);
                settings.lodBudgetMode = static_cast<GaussianSplatLodBudgetMode>(lodBudgetModeIndex);
            }
            if (!orderedClodEnabled)
                ImGui::EndDisabled();

            const uint32_t budgetSliderMax =
                std::min(stats.totalSplats, static_cast<uint32_t>(std::numeric_limits<int>::max()));
            settings.lodBudget = std::min(settings.lodBudget, budgetSliderMax);
            int budget = static_cast<int>(std::min(settings.lodBudget, budgetSliderMax));
            if (!lodControlsEnabled)
                ImGui::BeginDisabled();
            if (ImGui::SliderInt("LOD Budget", &budget, 0, static_cast<int>(budgetSliderMax), budget == 0 ? "Auto" : "%d"))
                settings.lodBudget = static_cast<uint32_t>(std::clamp(budget, 0, static_cast<int>(budgetSliderMax)));
            if (!lodControlsEnabled)
                ImGui::EndDisabled();

            if (!orderedClodEnabled)
                ImGui::BeginDisabled();
            ImGui::SliderFloat("CLOD Level", &settings.clodLevel, 0.01f, 1.0f, "%.2f");
            settings.clodLevel = std::clamp(settings.clodLevel, 0.01f, 1.0f);
            if (!orderedClodEnabled)
                ImGui::EndDisabled();

            constexpr const char* kDemoPresetLabels[] = {
                "0 Reference: Full Gaussian + SH3",
                "1 Full-frame SH2",
                "2 Full-frame SH1",
                "3 Full-frame SH0",
                "4 SH gaze 3/2/1 + Full Gaussian",
                "5 System gaze only + SH3",
                "6 System gaze + SH 3/2/1",
                "7 System gaze + Guarded SH 3/2/1",
                "Custom",
            };
            int demoPresetIndex = gaussianDemoPresetIndex(settings);
            if (ImGui::Combo("Demo Stage", &demoPresetIndex, kDemoPresetLabels, IM_ARRAYSIZE(kDemoPresetLabels)))
            {
                demoPresetIndex = std::clamp(demoPresetIndex, 0, IM_ARRAYSIZE(kDemoPresetLabels) - 1);
                applyGaussianDemoPreset(settings, demoPresetIndex);
            }

            constexpr const char* kShStagePresetLabels[] = {
                "Full SH3 / Full Gaussian",
                "SH gaze 3/2/1 / Full Gaussian",
                "SH gaze guard 3/2/1 / Full Gaussian",
                "SH gaze guard 3/2/1 split / Full Gaussian",
                "SH gaze guard 3/1/0 split / Full Gaussian",
                "SH gaze stabilized 3/2/1 split / Full Gaussian",
                "Custom",
            };

            const bool projectedCostControlsEnabled =
                orderedClodEnabled && settings.lodBudgetMode == GaussianSplatLodBudgetMode::eProjectedTileCost;
            if (!projectedCostControlsEnabled)
                ImGui::BeginDisabled();
            uint64_t projectedCostBudget = settings.projectedCostBudget;
            if (ImGui::InputScalar("Projected Cost Budget", ImGuiDataType_U64, &projectedCostBudget))
                settings.projectedCostBudget = projectedCostBudget;
            ImGui::SliderFloat(
                "Projected Cost Ratio", &settings.projectedCostBudgetRatio, 0.0f, 1.0f, "%.2f");
            int projectedCostChunkSize =
                static_cast<int>(std::min(settings.projectedCostChunkSize,
                                          static_cast<uint32_t>(std::numeric_limits<int>::max())));
            if (ImGui::SliderInt("Projected Cost Chunk", &projectedCostChunkSize, 1, 8192, "%d splats"))
                settings.projectedCostChunkSize =
                    static_cast<uint32_t>(std::clamp(projectedCostChunkSize, 1, 8192));
            constexpr const char* kProjectedCostBuildModeLabels[] = {"CPU", "GPU Sync"};
            int projectedCostBuildModeIndex = static_cast<int>(settings.projectedCostBuildMode);
            if (ImGui::Combo("Projected Cost Builder",
                             &projectedCostBuildModeIndex,
                             kProjectedCostBuildModeLabels,
                             IM_ARRAYSIZE(kProjectedCostBuildModeLabels)))
            {
                projectedCostBuildModeIndex =
                    std::clamp(projectedCostBuildModeIndex, 0, IM_ARRAYSIZE(kProjectedCostBuildModeLabels) - 1);
                settings.projectedCostBuildMode =
                    static_cast<GaussianProjectedCostBuildMode>(projectedCostBuildModeIndex);
            }
            ImGui::Checkbox("Semantic Difference Cost", &settings.projectedCostSemanticDiffEnabled);
            settings.projectedCostBudgetRatio = std::clamp(settings.projectedCostBudgetRatio, 0.0f, 1.0f);
            settings.projectedCostChunkSize   = std::max(settings.projectedCostChunkSize, 1u);
            if (!projectedCostControlsEnabled)
                ImGui::EndDisabled();

            ImGui::SeparatorText("Gaze Rendering");
            if (!orderedClodEnabled)
                ImGui::BeginDisabled();
            ImGui::Checkbox("Enable Gaze Rendering", &settings.foveatedClodEnabled);
            if (!orderedClodEnabled)
                ImGui::EndDisabled();

            const bool gazeControlsEnabled = orderedClodEnabled && settings.foveatedClodEnabled;
            if (!gazeControlsEnabled)
                ImGui::BeginDisabled();
            constexpr const char* kFoveatedModeLabels[] = {"Single Pass", "Layered Composite"};
            int foveatedModeIndex = static_cast<int>(settings.foveatedRenderMode);
            if (ImGui::Combo("Gaze Render Path",
                             &foveatedModeIndex,
                             kFoveatedModeLabels,
                             IM_ARRAYSIZE(kFoveatedModeLabels)))
            {
                foveatedModeIndex = std::clamp(foveatedModeIndex, 0, IM_ARRAYSIZE(kFoveatedModeLabels) - 1);
                settings.foveatedRenderMode = static_cast<GaussianSplatFoveatedRenderMode>(foveatedModeIndex);
            }
            ImGui::Checkbox("Coverage Compensation", &settings.foveatedCoverageCompensationEnabled);
            ImGui::Checkbox("Manual Gaze Point", &settings.foveatedManualGazeControlEnabled);
            const bool xrGazeAvailable = backendService.isXREnabled();
            if (!xrGazeAvailable)
                ImGui::BeginDisabled();
            ImGui::Checkbox("XR Eye Gaze", &settings.foveatedXrGazeEnabled);
            if (!xrGazeAvailable)
                ImGui::EndDisabled();
            if (settings.foveatedManualGazeControlEnabled)
                settings.foveatedXrGazeEnabled = false;
            ImGui::SliderFloat2("Gaze UV", &settings.foveatedGaze.x, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Fovea Degrees", &settings.foveatedRingDegrees.x, 0.0f, 45.0f, "%.1f");
            ImGui::SliderFloat("Mid Degrees", &settings.foveatedRingDegrees.y, 0.0f, 90.0f, "%.1f");
            settings.foveatedRingDegrees.x = std::max(settings.foveatedRingDegrees.x, 0.0f);
            settings.foveatedRingDegrees.y =
                std::max(settings.foveatedRingDegrees.y, settings.foveatedRingDegrees.x);
            ImGui::SliderFloat("Fovea LOD", &settings.foveatedRingLevels.x, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Mid LOD", &settings.foveatedRingLevels.y, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Outer LOD", &settings.foveatedRingLevels.z, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Fovea Resolution", &settings.foveatedResolutionScales.x, 0.05f, 1.0f, "%.2f");
            ImGui::SliderFloat("Mid Resolution", &settings.foveatedResolutionScales.y, 0.05f, 1.0f, "%.2f");
            ImGui::SliderFloat("Outer Resolution", &settings.foveatedResolutionScales.z, 0.05f, 1.0f, "%.2f");
            ImGui::SliderFloat("Transition Degrees", &settings.foveatedTransitionDegrees, 0.0f, 20.0f, "%.1f");
            constexpr const char* kCoverageGuardLabels[] = {
                "Off",
                "Global",
                "Local Bounded",
                "Risk Triggered",
            };
            int coverageGuardIndex = static_cast<int>(settings.foveatedCoverageGuardMode);
            if (ImGui::Combo("Coverage Guard",
                             &coverageGuardIndex,
                             kCoverageGuardLabels,
                             IM_ARRAYSIZE(kCoverageGuardLabels)))
            {
                coverageGuardIndex = std::clamp(coverageGuardIndex, 0, IM_ARRAYSIZE(kCoverageGuardLabels) - 1);
                settings.foveatedCoverageGuardMode =
                    static_cast<GaussianSplatFoveatedCoverageGuardMode>(coverageGuardIndex);
            }
            ImGui::SliderFloat(
                "Coverage Protection", &settings.foveatedCoverageProtectionDegrees, 0.0f, 12.0f, "%.1f deg");
            ImGui::SliderFloat(
                "Guard Budget", &settings.foveatedCoverageGuardBudgetRatio, 0.0f, 0.10f, "%.3f");
            ImGui::SliderFloat(
                "Guard Center Min", &settings.foveatedCoverageGuardMinLevels.x, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat(
                "Guard Transition Min", &settings.foveatedCoverageGuardMinLevels.y, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat(
                "Guard Periphery Min", &settings.foveatedCoverageGuardMinLevels.z, 0.0f, 1.0f, "%.2f");
            ImGui::Checkbox("Temporal Hysteresis", &settings.foveatedTemporalHysteresisEnabled);
            ImGui::Checkbox("Boundary Smoothing", &settings.foveatedBoundarySmoothingEnabled);
            int temporalResidencyFrames = static_cast<int>(settings.foveatedTemporalResidencyFrames);
            if (ImGui::SliderInt("Temporal Residency", &temporalResidencyFrames, 0, 8, "%d frames"))
                settings.foveatedTemporalResidencyFrames =
                    static_cast<uint32_t>(std::clamp(temporalResidencyFrames, 0, 255));
            ImGui::SliderFloat(
                "Temporal Hysteresis Ratio", &settings.foveatedTemporalHysteresisRatio, 0.0f, 0.5f, "%.2f");
            ImGui::SliderFloat(
                "Boundary Smoothing Ratio", &settings.foveatedBoundarySmoothingRatio, 0.0f, 0.5f, "%.2f");
            constexpr const char* kDistributionLabels[] = {
                "Hard Ring",
                "Smoothstep",
                "Gaussian",
                "Exponential",
                "Inverse Power",
                "Log Polar",
                "Continuous Scheduler",
            };
            int distributionIndex = static_cast<int>(settings.foveatedDistribution);
            if (ImGui::Combo("Gaze Distribution", &distributionIndex, kDistributionLabels, IM_ARRAYSIZE(kDistributionLabels)))
            {
                distributionIndex = std::clamp(distributionIndex, 0, IM_ARRAYSIZE(kDistributionLabels) - 1);
                settings.foveatedDistribution =
                    static_cast<GaussianSplatFoveatedDistribution>(distributionIndex);
            }
            constexpr const char* kAdaptationLabels[] = {
                "Fixed",
                "Dynamic Budget",
                "Dynamic Range",
                "Progressive Center-Out",
                "Progressive Greedy",
            };
            int adaptationIndex = static_cast<int>(settings.foveatedAdaptationMode);
            if (ImGui::Combo("Gaze Adaptation", &adaptationIndex, kAdaptationLabels, IM_ARRAYSIZE(kAdaptationLabels)))
            {
                adaptationIndex = std::clamp(adaptationIndex, 0, IM_ARRAYSIZE(kAdaptationLabels) - 1);
                settings.foveatedAdaptationMode =
                    static_cast<GaussianSplatFoveatedAdaptationMode>(adaptationIndex);
            }
            const bool adaptationEnabled =
                settings.foveatedAdaptationMode != GaussianSplatFoveatedAdaptationMode::eFixed;
            if (!adaptationEnabled)
                ImGui::BeginDisabled();
            ImGui::SliderFloat("Target Frame", &settings.foveatedTargetFrameMs, 1.0f, 33.3f, "%.1f ms");
            ImGui::SliderFloat("Adaptation Step", &settings.foveatedBudgetAdjustRate, 0.001f, 0.25f, "%.3f");
            if (!adaptationEnabled)
                ImGui::EndDisabled();
            settings.foveatedGaze.x = std::clamp(settings.foveatedGaze.x, 0.0f, 1.0f);
            settings.foveatedGaze.y = std::clamp(settings.foveatedGaze.y, 0.0f, 1.0f);
            settings.foveatedRingLevels.x = std::clamp(settings.foveatedRingLevels.x, 0.0f, 1.0f);
            settings.foveatedRingLevels.y = std::clamp(settings.foveatedRingLevels.y, 0.0f, 1.0f);
            settings.foveatedRingLevels.z = std::clamp(settings.foveatedRingLevels.z, 0.0f, 1.0f);
            settings.foveatedResolutionScales.x = std::clamp(settings.foveatedResolutionScales.x, 0.05f, 1.0f);
            settings.foveatedResolutionScales.y = std::clamp(settings.foveatedResolutionScales.y, 0.05f, 1.0f);
            settings.foveatedResolutionScales.z = std::clamp(settings.foveatedResolutionScales.z, 0.05f, 1.0f);
            settings.foveatedTransitionDegrees = std::max(settings.foveatedTransitionDegrees, 0.0f);
            settings.foveatedTargetFrameMs = std::max(settings.foveatedTargetFrameMs, 0.1f);
            settings.foveatedBudgetAdjustRate = std::clamp(settings.foveatedBudgetAdjustRate, 0.001f, 0.25f);
            settings.foveatedCoverageProtectionDegrees =
                std::clamp(settings.foveatedCoverageProtectionDegrees, 0.0f, 12.0f);
            settings.foveatedCoverageGuardBudgetRatio =
                std::clamp(settings.foveatedCoverageGuardBudgetRatio, 0.0f, 0.25f);
            settings.foveatedCoverageGuardMinLevels =
                glm::clamp(settings.foveatedCoverageGuardMinLevels, glm::vec3 {0.0f}, glm::vec3 {1.0f});
            settings.foveatedTemporalResidencyFrames =
                std::clamp(settings.foveatedTemporalResidencyFrames, 0u, 255u);
            settings.foveatedTemporalHysteresisRatio =
                std::clamp(settings.foveatedTemporalHysteresisRatio, 0.0f, 1.0f);
            settings.foveatedBoundarySmoothingRatio =
                std::clamp(settings.foveatedBoundarySmoothingRatio, 0.0f, 1.0f);
            if (!gazeControlsEnabled)
                ImGui::EndDisabled();

            ImGui::SeparatorText("SH Appearance LOD");
            int shStagePresetIndex = gaussianShStagePresetIndex(settings);
            if (ImGui::Combo("SH Stage Preset",
                             &shStagePresetIndex,
                             kShStagePresetLabels,
                             IM_ARRAYSIZE(kShStagePresetLabels)))
            {
                shStagePresetIndex =
                    std::clamp(shStagePresetIndex, 0, IM_ARRAYSIZE(kShStagePresetLabels) - 1);
                applyGaussianShStagePreset(settings, shStagePresetIndex);
            }

            bool shLodEnabled = settings.foveatedShLodEnabled;
            if (ImGui::Checkbox("Enable SH LOD", &shLodEnabled))
            {
                settings.foveatedShLodEnabled = shLodEnabled;
                if (settings.foveatedShLodEnabled)
                    settings.baselineMode = GaussianSplatBaselineMode::eOrderedClod;
            }

            if (!settings.foveatedShLodEnabled)
                ImGui::BeginDisabled();
            int shDegreeCenter = static_cast<int>(std::clamp(settings.foveatedShLodDegrees.x, 0u, 3u));
            int shDegreeMid    = static_cast<int>(std::clamp(settings.foveatedShLodDegrees.y, 0u, 3u));
            int shDegreeOuter  = static_cast<int>(std::clamp(settings.foveatedShLodDegrees.z, 0u, 3u));
            if (ImGui::SliderInt("Center SH Degree", &shDegreeCenter, 0, 3))
                settings.foveatedShLodDegrees.x = static_cast<uint32_t>(std::clamp(shDegreeCenter, 0, 3));
            if (ImGui::SliderInt("Mid SH Degree", &shDegreeMid, 0, 3))
                settings.foveatedShLodDegrees.y = static_cast<uint32_t>(std::clamp(shDegreeMid, 0, 3));
            if (ImGui::SliderInt("Outer SH Degree", &shDegreeOuter, 0, 3))
                settings.foveatedShLodDegrees.z = static_cast<uint32_t>(std::clamp(shDegreeOuter, 0, 3));

            constexpr const char* kShGuardLabels[] = {
                "Off",
                "Energy",
                "Projected Cost",
                "Energy + Projected Cost",
            };
            int shGuardIndex = static_cast<int>(settings.foveatedShLodGuardMode);
            if (ImGui::Combo("SH Guard", &shGuardIndex, kShGuardLabels, IM_ARRAYSIZE(kShGuardLabels)))
            {
                shGuardIndex = std::clamp(shGuardIndex, 0, IM_ARRAYSIZE(kShGuardLabels) - 1);
                settings.foveatedShLodGuardMode = static_cast<GaussianSplatShLodGuardMode>(shGuardIndex);
            }
            const bool shGuardEnabled = settings.foveatedShLodGuardMode != GaussianSplatShLodGuardMode::eOff;
            if (!shGuardEnabled)
                ImGui::BeginDisabled();
            ImGui::SliderFloat(
                "SH Guard Mid Threshold", &settings.foveatedShLodGuardThresholdMid, 0.0f, 1.0f, "%.3f");
            ImGui::SliderFloat(
                "SH Guard High Threshold", &settings.foveatedShLodGuardThresholdHigh, 0.0f, 1.0f, "%.3f");
            if (!shGuardEnabled)
                ImGui::EndDisabled();

            constexpr const char* kShStorageLabels[] = {"Monolithic", "Split Bands"};
            int shStorageIndex = static_cast<int>(settings.shStorageLayout);
            if (ImGui::Combo("SH Storage", &shStorageIndex, kShStorageLabels, IM_ARRAYSIZE(kShStorageLabels)))
            {
                shStorageIndex = std::clamp(shStorageIndex, 0, IM_ARRAYSIZE(kShStorageLabels) - 1);
                settings.shStorageLayout = static_cast<GaussianSplatShStorageLayout>(shStorageIndex);
            }
            ImGui::Checkbox("SH Pop Log", &settings.shPopLogEnabled);
            ImGui::Checkbox("SH Degree Hysteresis", &settings.shDegreeHysteresisEnabled);
            if (!settings.shDegreeHysteresisEnabled)
                ImGui::BeginDisabled();
            int shDegreeDowngradeDelay = static_cast<int>(settings.shDegreeDowngradeDelay);
            if (ImGui::SliderInt("SH Downgrade Delay", &shDegreeDowngradeDelay, 0, 16, "%d frames"))
                settings.shDegreeDowngradeDelay =
                    static_cast<uint32_t>(std::clamp(shDegreeDowngradeDelay, 0, 255));
            ImGui::SliderFloat("SH Guard Band", &settings.shDegreeGuardBandDegrees, 0.0f, 12.0f, "%.1f deg");
            if (!settings.shDegreeHysteresisEnabled)
                ImGui::EndDisabled();
            if (!settings.foveatedShLodEnabled)
                ImGui::EndDisabled();

            settings.foveatedShLodDegrees =
                glm::clamp(settings.foveatedShLodDegrees, glm::uvec3 {0u}, glm::uvec3 {3u});
            settings.foveatedShLodGuardThresholdMid =
                std::max(settings.foveatedShLodGuardThresholdMid, 0.0f);
            settings.foveatedShLodGuardThresholdHigh =
                std::max(settings.foveatedShLodGuardThresholdHigh,
                         settings.foveatedShLodGuardThresholdMid);
            settings.shDegreeDowngradeDelay = std::clamp(settings.shDegreeDowngradeDelay, 0u, 255u);
            settings.shDegreeGuardBandDegrees = std::max(settings.shDegreeGuardBandDegrees, 0.0f);

            ImGui::SeparatorText("Counters");
            ImGui::Text("Mode: %s", gaussianBaselineModeLabel(stats.baselineMode));
            ImGui::Text("LOD Budget Mode: %s", gaussianLodBudgetModeLabel(stats.lodBudgetMode));
            ImGui::Text("LOD Budget: %s / %u", stats.lodBudgetEnabled ? "enabled" : "disabled", stats.lodBudget);
            ImGui::Text("Projected Cost: budget=%llu actual=%llu overshoot=%llu chunks=%u",
                        static_cast<unsigned long long>(stats.costBudget),
                        static_cast<unsigned long long>(stats.actualProjectedCost),
                        static_cast<unsigned long long>(stats.costOvershoot),
                        stats.selectedCostChunks);
            ImGui::Text("Projected Builder: %s / chunk=%u / ratio=%.2f / semantic=%s",
                        gaussianProjectedCostBuildModeLabel(stats.projectedCostBuildMode),
                        stats.projectedCostChunkSize,
                        stats.projectedCostBudgetRatio,
                        stats.projectedCostSemanticDiffEnabled ? "yes" : "no");
            ImGui::Text("Gaze Rendering: %s", stats.foveatedClodEnabled ? "yes" : "no");
            ImGui::Text("Gaze Render Path: %s", gaussianFoveatedRenderModeLabel(stats.foveatedRenderMode));
            ImGui::Text("Layered Framebuffers: %s", stats.foveatedLayeredCompositeEnabled ? "yes" : "no");
            ImGui::Text("XR Eye Gaze: %s%s",
                        stats.foveatedXrGazeEnabled ? "enabled" : "disabled",
                        stats.foveatedXrGazeActive ? " / active" : "");
            ImGui::Text("Coverage Compensation: %s",
                        stats.foveatedCoverageCompensationEnabled ? "yes" : "no");
            ImGui::Text("Gaze UV: %.2f / %.2f", stats.foveatedGaze.x, stats.foveatedGaze.y);
            ImGui::Text("Gaze Adaptation: %s", gaussianFoveatedAdaptationModeLabel(stats.foveatedAdaptationMode));
            ImGui::Text("Gaze Distribution: %s", gaussianFoveatedDistributionLabel(stats.foveatedDistribution));
            ImGui::Text("Coverage Guard: %s / %.1f deg / %.1f%%",
                        gaussianFoveatedCoverageGuardModeLabel(stats.foveatedCoverageGuardMode),
                        stats.foveatedCoverageProtectionDegrees,
                        stats.foveatedCoverageGuardBudgetRatio * 100.0f);
            ImGui::Text("Guard Risk: sectors=%u active=%s added=%u cost=%llu",
                        stats.foveatedCoverageGuardRiskSectors,
                        stats.foveatedCoverageGuardActive ? "yes" : "no",
                        stats.foveatedCoverageGuardAddedCount,
                        static_cast<unsigned long long>(stats.foveatedCoverageGuardAddedCost));
            ImGui::Text("Temporal: %s / smoothing: %s",
                        stats.foveatedTemporalHysteresisEnabled ? "on" : "off",
                        stats.foveatedBoundarySmoothingEnabled ? "on" : "off");
            ImGui::Text("Direct Prefix: %s", stats.directPrefix ? "yes" : "no");
            ImGui::Text("Ring Degrees: %.1f / %.1f", stats.foveatedRingDegrees.x, stats.foveatedRingDegrees.y);
            ImGui::Text("Ring LODs: %.2f / %.2f / %.2f",
                        stats.foveatedRingLevels.x,
                        stats.foveatedRingLevels.y,
                        stats.foveatedRingLevels.z);
            ImGui::Text("Ring Res: %.2f / %.2f / %.2f",
                        stats.foveatedResolutionScales.x,
                        stats.foveatedResolutionScales.y,
                        stats.foveatedResolutionScales.z);
            ImGui::Text("SH LOD: %s / degrees=%u/%u/%u / guard=%s / storage=%s",
                        stats.foveatedShLodEnabled ? "yes" : "no",
                        stats.shDegreeCenter,
                        stats.shDegreeMid,
                        stats.shDegreeOuter,
                        gaussianShLodGuardModeLabel(stats.shGuardMode),
                        gaussianShStorageLayoutLabel(stats.shStorageLayout));
            ImGui::Text("SH AC Reads: %llu coeffs / saved %.1f%%",
                        static_cast<unsigned long long>(stats.estimatedShAcCoeffReads),
                        stats.estimatedShAcReadReductionVsDegree3 * 100.0);
            ImGui::Text("SH Split Bands: L1=%llu L2=%llu L3=%llu bytes=%s saved %.1f%%",
                        static_cast<unsigned long long>(stats.shBandL1ReadsEst),
                        static_cast<unsigned long long>(stats.shBandL2ReadsEst),
                        static_cast<unsigned long long>(stats.shBandL3ReadsEst),
                        formatBytes(stats.shBandBytesEst).c_str(),
                        stats.shBandBytesReductionVsMonolithic * 100.0);
            ImGui::Text("SH Guard: raised=%u (%.2f%%) recovered=%llu saved=%llu",
                        stats.shGuardRaisedCount,
                        stats.shGuardRaisedRatio * 100.0,
                        static_cast<unsigned long long>(stats.shGuardRecoveredAcReads),
                        static_cast<unsigned long long>(stats.shGuardSavedAcReadsAfterGuard));
            ImGui::Text("SH Pop: log=%s changed=%u (%.3f%%) proxy=%.3f guard=%u delayed=%u",
                        stats.shPopLogEnabled ? "on" : "off",
                        stats.shDegreeChangedCount,
                        stats.shDegreeChangedRatio * 100.0,
                        stats.shPopEnergyProxy,
                        stats.shPopGuardRaiseCount,
                        stats.shPopDelayedDowngradeCount);
            ImGui::Text("Total Splats: %u", stats.totalSplats);
            ImGui::Text("Prepared Splats: %u", stats.preparedSplats);
            const uint32_t selectedSplats =
                stats.lodSelectedRawSplats > 0u ? stats.lodSelectedRawSplats : stats.preparedSplats;
            const double selectedRatio =
                stats.totalSplats > 0u ? static_cast<double>(selectedSplats) / static_cast<double>(stats.totalSplats) :
                                         0.0;
            ImGui::Text("Selected Ratio: %.3f (%u / %u)", selectedRatio, selectedSplats, stats.totalSplats);
            ImGui::Text("Ring Budgets: %u / %u / %u",
                        stats.foveaSplatBudget,
                        stats.midSplatBudget,
                        stats.outerSplatBudget);
            ImGui::Text("Visible Cap: %u", stats.maxVisibleSplatCap);
            ImGui::Text("Draw Records: %u", stats.drawRecords);
            ImGui::Text("LOD Raw Splats: %u", stats.lodSelectedRawSplats);
            drawOptionalCounter("Visible Splats", stats.visibleSplats);
            drawOptionalCounter("Drawn Splats", stats.drawnSplats);

            if (auto* profiler = renderService.runtimeProfiler())
            {
                const auto* selected = profiler->selectedFrame();
                ImGui::SeparatorText("Timings");
                if (selected && selected->gpuFrameMs >= 0.0)
                    ImGui::Text("GPU Frame: %.3f ms", selected->gpuFrameMs);
                else
                    ImGui::TextUnformatted("GPU Frame: N/A");

                const double preprocessMs =
                    selected ? findScopeGpuMs(selected->gpuScopeTree, "GeneralGaussianSplatPreprocess") : -1.0;
                if (preprocessMs >= 0.0)
                    ImGui::Text("Preprocess: %.3f ms", preprocessMs);
                else
                    ImGui::TextUnformatted("Preprocess: N/A");

                const double sortMs = selected ? findScopeGpuMs(selected->gpuScopeTree, "Sort") : -1.0;
                if (sortMs >= 0.0)
                    ImGui::Text("Sort: %.3f ms", sortMs);
                else
                    ImGui::TextUnformatted("Sort: N/A");
            }
        }

        void drawRuntimeProfilerPanel(IRenderService& renderService)
        {
            auto* profiler = renderService.runtimeProfiler();
            if (!profiler)
                return;

            if (!ImGui::CollapsingHeader("Built-in Profiler", ImGuiTreeNodeFlags_DefaultOpen))
                return;

            bool enabled = profiler->isEnabled();
            if (ImGui::Checkbox("Enable Internal Profiler", &enabled))
                profiler->setEnabled(enabled);

            if (!enabled)
                return;

            bool paused = profiler->isPaused();
            if (ImGui::Checkbox("Pause", &paused))
                profiler->setPaused(paused);

            const char* sortLabels[] = {"Total (ms)", "Self (ms)", "Calls", "Name"};
            int         sortIndex    = static_cast<int>(profiler->getSortKey());
            if (ImGui::Combo("Sort", &sortIndex, sortLabels, IM_ARRAYSIZE(sortLabels)))
                profiler->setSortKey(static_cast<RuntimeProfiler::SortKey>(sortIndex));

            if (paused && profiler->historySize() > 0)
            {
                int frozen = profiler->getFrozenHistoryIndex();
                if (frozen < 0)
                    frozen = static_cast<int>(profiler->historySize()) - 1;
                if (ImGui::SliderInt("Frozen Frame", &frozen, 0, static_cast<int>(profiler->historySize()) - 1))
                    profiler->setFrozenHistoryIndex(frozen);
            }

            const auto* selected = profiler->selectedFrame();
            if (!selected)
            {
                ImGui::TextUnformatted("No profiler frames yet.");
                return;
            }

            ImGui::SeparatorText("Frame Summary");
            ImGui::Text("Frame: %llu", static_cast<unsigned long long>(selected->frameIndex));
            ImGui::Text("CPU Frame: %.3f ms", selected->cpuFrameMs);
            ImGui::Text("CPU Render: %.3f ms", selected->cpuRenderMs);
            if (selected->gpuFrameMs >= 0.0)
                ImGui::Text("GPU Frame: %.3f ms", selected->gpuFrameMs);
            else
                ImGui::TextUnformatted("GPU Frame: N/A");

            ImGui::Text("Draw Calls: %llu", static_cast<unsigned long long>(selected->drawCalls));
            ImGui::Text("Dispatch: %llu", static_cast<unsigned long long>(selected->dispatchCalls));
            ImGui::Text("Trace Rays: %llu", static_cast<unsigned long long>(selected->traceRaysCalls));
            ImGui::Text("Copy Ops: %llu", static_cast<unsigned long long>(selected->copyOps));
            ImGui::Text("Update Ops: %llu", static_cast<unsigned long long>(selected->updateOps));
            ImGui::Text("VSync: %s", selected->vsyncEnabled ? "On" : "Off");
            ImGui::Text("Asset CPU Cache: %s", formatBytes(selected->assetCpuCacheBytes).c_str());
            ImGui::Text("Render CPU Cache: %s", formatBytes(selected->renderCpuCacheBytes).c_str());
            ImGui::Text("GPU Device Local: %s", formatBytes(selected->gpuDeviceLocalBytes).c_str());
            ImGui::Text("GPU Host Visible: %s", formatBytes(selected->gpuHostVisibleBytes).c_str());
            ImGui::Text("GPU Scope Begin/Token/Resolved: %u / %u / %u",
                        selected->gpuScopeBeginCount,
                        selected->gpuScopeTokenCount,
                        selected->gpuScopeResolvedCount);

            const auto& history = profiler->history();
            if (!history.empty() && ImPlot::BeginPlot("Frame Times", ImVec2(-1, 180)))
            {
                static std::vector<double> x;
                static std::vector<double> cpu;
                static std::vector<double> gpu;
                x.resize(history.size());
                cpu.resize(history.size());
                gpu.resize(history.size());

                for (size_t i = 0; i < history.size(); ++i)
                {
                    x[i]   = static_cast<double>(i);
                    cpu[i] = history[i].cpuFrameMs;
                    gpu[i] = history[i].gpuFrameMs >= 0.0 ? history[i].gpuFrameMs : 0.0;
                }

                ImPlot::SetupAxes("Frame", "ms", ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_AutoFit);
                ImPlot::PlotLine("CPU", x.data(), cpu.data(), static_cast<int>(cpu.size()));
                if (std::any_of(history.begin(), history.end(), [](const auto& f) { return f.gpuFrameMs >= 0.0; }))
                    ImPlot::PlotLine("GPU", x.data(), gpu.data(), static_cast<int>(gpu.size()));
                ImPlot::EndPlot();
            }

            ImGui::SeparatorText("Scope Trees");

            auto drawScopeTreeTable = [&](const char*                                    title,
                                          const char*                                    tableId,
                                          const std::vector<RuntimeProfiler::ScopeNode>& nodes,
                                          const bool                                     gpuTree) {
                ImGui::TextUnformatted(title);
                if (nodes.empty())
                {
                    ImGui::TextUnformatted("No scope data.");
                    return;
                }

                if (!ImGui::BeginTable(tableId,
                                       4,
                                       ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                           ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY,
                                       ImVec2(-1.0f, 180.0f)))
                    return;

                ImGui::TableSetupColumn("Scope");
                ImGui::TableSetupColumn(
                    gpuTree ? "GPU Total (ms)" : "CPU Total (ms)", ImGuiTableColumnFlags_WidthFixed, 130.0f);
                ImGui::TableSetupColumn(
                    gpuTree ? "GPU Self (ms)" : "CPU Self (ms)", ImGuiTableColumnFlags_WidthFixed, 130.0f);
                ImGui::TableSetupColumn("Calls", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                ImGui::TableHeadersRow();

                std::vector<std::vector<int>> children(nodes.size());
                for (size_t i = 1; i < nodes.size(); ++i)
                {
                    const int parent = nodes[i].parent;
                    if (parent >= 0 && static_cast<size_t>(parent) < children.size())
                        children[static_cast<size_t>(parent)].push_back(static_cast<int>(i));
                }

                auto sortChildren = [&](std::vector<int>& list) {
                    const auto sortKey = profiler->getSortKey();
                    std::stable_sort(list.begin(), list.end(), [&](const int ia, const int ib) {
                        const auto& a = nodes[static_cast<size_t>(ia)];
                        const auto& b = nodes[static_cast<size_t>(ib)];
                        switch (sortKey)
                        {
                            case RuntimeProfiler::SortKey::eTotalMs:
                                return gpuTree ? a.gpuTotalMs > b.gpuTotalMs : a.totalMs > b.totalMs;
                            case RuntimeProfiler::SortKey::eSelfMs:
                                return gpuTree ? a.gpuSelfMs > b.gpuSelfMs : a.selfMs > b.selfMs;
                            case RuntimeProfiler::SortKey::eCalls:
                                return a.callCount > b.callCount;
                            case RuntimeProfiler::SortKey::eName:
                                return a.name < b.name;
                        }
                        return a.totalMs > b.totalMs;
                    });
                };

                std::function<void(int)> drawNode = [&](const int idx) {
                    const auto& node      = nodes[static_cast<size_t>(idx)];
                    auto&       childList = children[static_cast<size_t>(idx)];
                    sortChildren(childList);

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);

                    const bool         isLeaf = childList.empty();
                    ImGuiTreeNodeFlags flags =
                        isLeaf ? (ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen) : 0;
                    const bool opened = ImGui::TreeNodeEx(
                        reinterpret_cast<void*>(static_cast<intptr_t>(idx)), flags, "%s", node.name.c_str());

                    ImGui::TableSetColumnIndex(1);
                    if (gpuTree)
                    {
                        if (node.gpuTotalMs >= 0.0)
                            ImGui::Text("%.3f", node.gpuTotalMs);
                        else
                            ImGui::TextUnformatted("N/A");
                    }
                    else
                    {
                        ImGui::Text("%.3f", node.totalMs);
                    }

                    ImGui::TableSetColumnIndex(2);
                    if (gpuTree)
                    {
                        if (node.gpuSelfMs >= 0.0)
                            ImGui::Text("%.3f", node.gpuSelfMs);
                        else
                            ImGui::TextUnformatted("N/A");
                    }
                    else
                    {
                        ImGui::Text("%.3f", node.selfMs);
                    }

                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%u", node.callCount);

                    if (!isLeaf && opened)
                    {
                        for (const int child : childList)
                            drawNode(child);
                        ImGui::TreePop();
                    }
                };

                drawNode(0);
                ImGui::EndTable();
            };

            drawScopeTreeTable("CPU Tree", "##RuntimeProfilerCpuTree", selected->cpuScopeTree, false);
            drawScopeTreeTable("GPU Tree", "##RuntimeProfilerGpuTree", selected->gpuScopeTree, true);
        }

        void drawHintRow(const char* icon, const char* text)
        {
            ImGui::TextUnformatted(icon);
            ImGui::SameLine();
            ImGui::TextUnformatted(text);
        }

        void syncImGuiTextureRegistration(IImGuiService&            imguiService,
                                          const rhi::Texture*       texture,
                                          const rhi::Texture*&      registeredTexture,
                                          IImGuiService::TextureID& textureId)
        {
            const bool alreadyCleared = registeredTexture == nullptr && texture == nullptr && textureId == 0;
            if (alreadyCleared)
                return;
            const bool sameTexture = texture != nullptr && textureId != 0 && registeredTexture == texture;
            if (sameTexture)
                return;

            if (textureId)
                imguiService.removeTexture(textureId);

            registeredTexture = texture;
            textureId         = texture ? imguiService.addTexture(*texture) : 0;
        }

        void drawFpsOverlay()
        {
            const ImGuiViewport* viewport = ImGui::GetMainViewport();
            if (!viewport)
                return;

            constexpr float kPadding = 10.0f;

            const ImVec2 windowPos {viewport->WorkPos.x + viewport->WorkSize.x - kPadding,
                                    viewport->WorkPos.y + kPadding};
            const ImVec2 windowPivot {1.0f, 0.0f};

            ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always, windowPivot);
            ImGui::SetNextWindowBgAlpha(0.35f);

            constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                                ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                                ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;

            if (ImGui::Begin("##UniversalRendererFpsOverlay", nullptr, kFlags))
            {
                const float fps = ImGui::GetIO().Framerate;
                const float ms  = fps > 0.0f ? (1000.0f / fps) : 0.0f;

                ImGui::Text("FPS: %.1f", fps);
                ImGui::Text("Frame: %.2f ms", ms);
            }
            ImGui::End();
        }

        void drawCameraHintOverlay(const std::optional<CameraControlOverlayInfo>& infoOpt)
        {
            if (!infoOpt.has_value())
                return;
            const auto info = *infoOpt;
            if (!info.enabled)
                return;

            const ImGuiViewport* viewport = ImGui::GetMainViewport();
            if (!viewport)
                return;

            constexpr float kPadding = 10.0f;
            ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + kPadding, viewport->WorkPos.y + kPadding),
                                    ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.35f);
            constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                                ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                                ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;

            if (ImGui::Begin("##CameraControlHintsOverlay", nullptr, kFlags))
            {
                if (info.mode == CameraControlMode::eFly)
                {
                    drawHintRow(ICON_MDI_MOUSE_RIGHT_CLICK " " ICON_MDI_EYE_OUTLINE, "Look");
                    drawHintRow(ICON_MDI_ALPHA_W_BOX " " ICON_MDI_ALPHA_A_BOX " " ICON_MDI_ALPHA_S_BOX
                                                     " " ICON_MDI_ALPHA_D_BOX,
                                "Move");
                    drawHintRow(ICON_MDI_ALPHA_Q_BOX " " ICON_MDI_CHEVRON_DOWN_BOX "   " ICON_MDI_ALPHA_E_BOX
                                                     " " ICON_MDI_CHEVRON_UP_BOX,
                                "Down / Up");
                    drawHintRow(ICON_MDI_APPLE_KEYBOARD_SHIFT " " ICON_MDI_RUN_FAST, "Faster");
                    drawHintRow(ICON_MDI_APPLE_KEYBOARD_CONTROL " " ICON_MDI_TURTLE, "Slower");
                }
                else
                {
                    drawHintRow(ICON_MDI_MOUSE_LEFT_CLICK " " ICON_MDI_ROTATE_ORBIT, "Rotate");
                    drawHintRow(ICON_MDI_MOUSE_SCROLL_WHEEL " " ICON_MDI_PAN, "Pan");
                    drawHintRow(ICON_MDI_MOUSE_SCROLL_WHEEL " " ICON_MDI_MAGNIFY, "Zoom");
                    drawHintRow(ICON_MDI_MOUSE_RIGHT_CLICK " " ICON_MDI_EYE_OUTLINE, "Fly");
                }
            }
            ImGui::End();
        }

        void drawXrMirrorControls(bool&  fitToPanel,
                                  float& manualScale,
                                  bool&  swapEyes,
                                  bool&  singleEye,
                                  int&   eyeIndex,
                                  int    maxEyeIndex)
        {
            if (ImGui::Button("Fit"))
                fitToPanel = true;

            ImGui::SameLine();
            if (ImGui::Button("1:1"))
            {
                fitToPanel  = false;
                manualScale = 1.0f;
            }

            ImGui::SameLine();
            ImGui::Checkbox("Swap Eyes", &swapEyes);

            ImGui::SameLine();
            ImGui::Checkbox("Single Eye", &singleEye);

            if (!fitToPanel)
                ImGui::SliderFloat("Scale", &manualScale, 0.1f, 2.0f, "%.2fx");

            if (singleEye)
                ImGui::SliderInt("Eye", &eyeIndex, 0, std::max(0, maxEyeIndex));
        }

        void syncTextureViewerRegistration(IImGuiService&                         imguiService,
                                           const resource::GpuResourcePool&       pool,
                                           std::vector<const rhi::Texture*>&      registeredTextures,
                                           std::vector<IImGuiService::TextureID>& textureIds)
        {
            uint32_t maxBindlessIndex = 0u;
            for (const auto& gpuTexture : pool.textures)
                maxBindlessIndex = std::max(maxBindlessIndex, gpuTexture.bindlessIndex);

            const size_t requiredSize = static_cast<size_t>(maxBindlessIndex) + 1u;
            if (registeredTextures.size() < requiredSize)
                registeredTextures.resize(requiredSize, nullptr);
            if (textureIds.size() < requiredSize)
                textureIds.resize(requiredSize, 0);

            std::vector<bool> alive(requiredSize, false);
            for (const auto& gpuTexture : pool.textures)
            {
                const auto index = static_cast<size_t>(gpuTexture.bindlessIndex);
                alive[index]     = true;
                syncImGuiTextureRegistration(
                    imguiService, gpuTexture.texture.get(), registeredTextures[index], textureIds[index]);
            }

            for (size_t i = 0; i < registeredTextures.size(); ++i)
            {
                const bool isAlive = i < alive.size() ? alive[i] : false;
                if (!isAlive && textureIds[i])
                {
                    imguiService.removeTexture(textureIds[i]);
                    registeredTextures[i] = nullptr;
                }
            }
        }

        void drawTextureViewer(const resource::GpuResourcePool&       pool,
                               std::vector<const rhi::Texture*>&      registeredTextures,
                               std::vector<IImGuiService::TextureID>& textureIds,
                               int&                                   columns)
        {
            if (!ImGui::CollapsingHeader("Texture Viewer", ImGuiTreeNodeFlags_DefaultOpen))
                return;

            ImGui::Text("Loaded GPU textures: %zu", pool.textures.size());
            ImGui::SliderInt("Columns", &columns, 1, 8);

            if (!ImGui::BeginTable("##TextureViewerTable", columns, ImGuiTableFlags_SizingStretchSame))
                return;

            for (size_t bindless = 0; bindless < registeredTextures.size(); ++bindless)
            {
                const auto* texture = registeredTextures[bindless];
                const auto  texId   = bindless < textureIds.size() ? textureIds[bindless] : 0;
                if (!texture || !texId)
                    continue;

                ImGui::TableNextColumn();
                ImGui::BeginGroup();
                ImGui::Image(texId, ImVec2(96.0f, 96.0f), ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f));
                const auto extent = texture->getExtent();
                const auto format = rhi::toString(texture->getPixelFormat());
                ImGui::Text("Slot: %zu", bindless);
                ImGui::Text("Size: %ux%u", extent.width, extent.height);
                ImGui::Text("Mips: %u", texture->getNumMipLevels());
                ImGui::Text("Fmt : %.*s", static_cast<int>(format.size()), format.data());
                ImGui::EndGroup();
            }

            ImGui::EndTable();
        }
    } // namespace

    void UniversalRenderer::init()
    {
        auto* services = getServices();
        if (!services)
            return;

        const auto backendApi = services->require<IRenderBackendService>().renderDevice().getBackendApi();
#if defined(__ANDROID__)
        constexpr bool kForceCompatibilityFeature = true;
#else
        constexpr bool kForceCompatibilityFeature = false;
#endif
        const bool forceCompatibilityByCli = m_RenderProfile == RenderProfile::eCompatibility;
        const bool useCompatibilityFeature =
            kForceCompatibilityFeature || forceCompatibilityByCli || backendApi == rhi::RenderBackendApi::eWebGPU;
        if (useCompatibilityFeature)
        {
            emplaceFeature<CompatibilityBaseColorFeature>();
            emplaceFeature<GeneralGaussianSplatFeature>();
            emplaceFeature<FinalCompositionFeature>();
            return;
        }

        // Add features in the desired order.
        emplaceFeature<MeshletFeature>();
        emplaceFeature<TestFeature>();
        emplaceFeature<GeneralGaussianSplatFeature>();
        emplaceFeature<FinalCompositionFeature>();
    }

    void UniversalRenderer::onImGui()
    {
        drawFpsOverlay();

        auto* services       = getServices();
        auto& backendService = services->require<IRenderBackendService>();
        auto& cameraService  = services->require<ICameraService>();
        auto& imguiService   = services->require<IImGuiService>();
        auto& renderService  = services->require<IRenderService>();
        auto& gpuResourceSvc = services->require<IGpuResourceService>();

        const bool suppressCameraInput = gaussianManualGazeControlActive(renderService.gaussianSplatSettings()) ||
                                         ImGui::GetIO().WantCaptureMouse || ImGui::IsAnyItemHovered() ||
                                         ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);
        cameraService.setCameraControlInputSuppressed(suppressCameraInput);

        drawCameraHintOverlay(cameraService.cameraControlOverlayInfo());

        ImGui::Begin("Universal Renderer");

        drawGaussianSplatBaselinePanel(renderService, backendService);
        drawGaussianManualGazeOverlay(renderService.gaussianSplatSettings());

        if (backendService.isXREnabled() && backendService.isXRMirrorEnabled())
        {
            static bool  fitToPanel     = true;
            static float manualScale    = 1.0f;
            static bool  swapEyes       = false;
            static bool  singleEye      = false;
            static int   singleEyeIndex = 0;

            if (ImGui::CollapsingHeader("XR Mirror", ImGuiTreeNodeFlags_DefaultOpen))
            {
                const auto eyeViews    = backendService.xrEyeViews();
                const int  maxEyeIndex = static_cast<int>(eyeViews.empty() ? 0u : (eyeViews.size() - 1u));
                drawXrMirrorControls(fitToPanel, manualScale, swapEyes, singleEye, singleEyeIndex, maxEyeIndex);

                const size_t mirrorCount = std::min<std::size_t>(eyeViews.size(), m_XRMirrorTextureIds.size());
                for (size_t eyeIndex = 0; eyeIndex < mirrorCount; ++eyeIndex)
                {
                    const auto& eyeView = eyeViews[eyeIndex];
                    if (!eyeView.mirrorTarget)
                        continue;

                    syncImGuiTextureRegistration(imguiService,
                                                 eyeView.mirrorTarget,
                                                 m_XRMirrorTextures[eyeIndex],
                                                 m_XRMirrorTextureIds[eyeIndex]);
                }

                const bool   drawSingleEye = singleEye || mirrorCount <= 1u;
                const size_t primaryEye    = swapEyes && mirrorCount > 1u ? 1u : 0u;
                const size_t secondaryEye  = swapEyes && mirrorCount > 1u ? 0u : 1u;

                auto drawEye = [&](size_t eyeIndex, float slotWidth) {
                    if (eyeIndex >= mirrorCount)
                        return;

                    const auto& eyeView   = eyeViews[eyeIndex];
                    const auto  textureId = m_XRMirrorTextureIds[eyeIndex];
                    if (!eyeView.mirrorTarget || !textureId)
                        return;

                    const auto   extent       = eyeView.mirrorTarget->getExtent();
                    const float  nativeWidth  = static_cast<float>(std::max(extent.width, 1u));
                    const float  nativeHeight = static_cast<float>(std::max(extent.height, 1u));
                    const float  aspect       = nativeHeight / nativeWidth;
                    const float  drawWidth    = fitToPanel ? slotWidth : std::min(slotWidth, nativeWidth * manualScale);
                    const ImVec2 imageSize {drawWidth, drawWidth * aspect};

                    ImGui::BeginGroup();
                    ImGui::Text("Eye %u  %ux%u", eyeView.eyeIndex, extent.width, extent.height);
                    ImGui::Image(textureId, imageSize, ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f));
                    if (eyeView.gazeValid)
                    {
                        const ImVec2 imageMin = ImGui::GetItemRectMin();
                        const ImVec2 imageMax = ImGui::GetItemRectMax();
                        drawGazeMarker(*ImGui::GetWindowDrawList(),
                                       imageMin,
                                       {imageMax.x - imageMin.x, imageMax.y - imageMin.y},
                                       eyeView.gazeUv);
                    }
                    ImGui::EndGroup();
                };

                const float spacing    = ImGui::GetStyle().ItemSpacing.x;
                const float availWidth = ImGui::GetContentRegionAvail().x;

                if (drawSingleEye)
                {
                    const size_t eyeIndex = static_cast<size_t>(
                        std::clamp(singleEyeIndex, 0, static_cast<int>(mirrorCount > 0 ? mirrorCount - 1u : 0u)));
                    drawEye(eyeIndex, std::max(1.0f, availWidth));
                }
                else
                {
                    const float slotWidth = std::max(1.0f, (availWidth - spacing) * 0.5f);
                    drawEye(primaryEye, slotWidth);
                    ImGui::SameLine();
                    drawEye(secondaryEye, slotWidth);
                }
            }
        }
        else
        {
            for (auto& textureId : m_XRMirrorTextureIds)
            {
                if (textureId)
                    imguiService.removeTexture(textureId);
            }
            m_XRMirrorTextures.fill(nullptr);
        }

        syncTextureViewerRegistration(
            imguiService, gpuResourceSvc.pool(), m_TextureViewerRegisteredTextures, m_TextureViewerTextureIds);
        drawTextureViewer(gpuResourceSvc.pool(),
                          m_TextureViewerRegisteredTextures,
                          m_TextureViewerTextureIds,
                          m_TextureViewerColumns);

        drawRuntimeProfilerPanel(renderService);

#ifdef VULTRA_ENABLE_RENDERDOC
        ImGui::Button("Capture One Frame");
        if (ImGui::IsItemClicked())
        {
            getServices()->require<IFrameDebuggerService>().captureSingleFrame();
        }
#endif
        ImGui::End();
    }
} // namespace vultra

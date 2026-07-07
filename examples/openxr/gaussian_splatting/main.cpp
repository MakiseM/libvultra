#include <vultra/core/app/demo_app_host.hpp>
#include <vultra/core/base/common_context.hpp>
#include <vultra/function/rendering/render_structs.hpp>
#include <vultra/function/services/asset_service.hpp>
#include <vultra/function/services/render_service.hpp>
#include <vultra/function/services/scene_service.hpp>
#include <vultra/function/services/world_service.hpp>
#include <vultra/function/world/world.hpp>
#include <vultra/function/world/components/gaussian_splat_component.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <span>
#include <string>
#include <string_view>

using namespace vultra;

namespace
{
    struct OpenXRGaussianOptions
    {
        std::optional<std::string>                 splatUri {
            "res://models/3dgs/playroom_point_cloud.ply"};
        bool                                       xrGazeEnabled {true};
        GaussianSplatFoveatedRenderMode            renderMode {GaussianSplatFoveatedRenderMode::eSinglePass};
        GaussianSplatFoveatedAdaptationMode        adaptationMode {GaussianSplatFoveatedAdaptationMode::eFixed};
        GaussianSplatFoveatedDistribution          distribution {GaussianSplatFoveatedDistribution::eGaussian};
        GaussianSplatFoveatedCoverageGuardMode     coverageGuardMode {GaussianSplatFoveatedCoverageGuardMode::eOff};
        std::optional<float>                       targetFrameMs;
        std::optional<float>                       coverageProtectionDegrees;
        std::optional<float>                       gazeX;
        std::optional<float>                       gazeY;
        float                                      coverageGuardBudgetRatio {0.0f};
        bool                                       temporalHysteresisEnabled {true};
        bool                                       boundarySmoothingEnabled {true};
        std::optional<bool>                        shLodEnabled;
        std::optional<uint32_t>                    shDegreeCenter;
        std::optional<uint32_t>                    shDegreeMid;
        std::optional<uint32_t>                    shDegreeOuter;
        std::optional<GaussianSplatShLodGuardMode> shLodGuardMode;
        std::optional<float>                       shLodGuardThresholdMid;
        std::optional<float>                       shLodGuardThresholdHigh;
        std::optional<GaussianSplatShStorageLayout> shStorageLayout;
    };

    std::string normalizeToken(const std::string_view value)
    {
        std::string normalized;
        normalized.reserve(value.size());
        for (const char ch : value)
            normalized.push_back(ch == '_' ? '-' : static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        return normalized;
    }

    std::optional<std::string_view> takeOptionValue(std::span<const std::string> args, size_t& i, std::string_view name)
    {
        const std::string_view arg = args[i];
        if (arg == name)
        {
            if (i + 1u < args.size())
                return std::string_view {args[++i]};
            return std::nullopt;
        }

        if (arg.starts_with(name) && arg.size() > name.size() && arg[name.size()] == '=')
            return arg.substr(name.size() + 1u);

        return std::nullopt;
    }

    std::optional<float> parseFloat(std::string_view value)
    {
        std::string text {value};
        char*       end = nullptr;
        const float parsed = std::strtof(text.c_str(), &end);
        if (end == text.c_str() || *end != '\0' || !std::isfinite(parsed))
            return std::nullopt;
        return parsed;
    }

    std::optional<uint32_t> parseU32(std::string_view value)
    {
        std::string text {value};
        try
        {
            size_t         parsedChars = 0;
            const uint32_t parsed      = static_cast<uint32_t>(std::stoul(text, &parsedChars, 10));
            if (parsedChars != text.size())
                return std::nullopt;
            return parsed;
        }
        catch (...)
        {
            return std::nullopt;
        }
    }

    std::optional<GaussianSplatFoveatedRenderMode> parseRenderMode(std::string_view value)
    {
        if (value == "single-pass" || value == "single")
            return GaussianSplatFoveatedRenderMode::eSinglePass;
        return std::nullopt;
    }

    std::optional<GaussianSplatFoveatedAdaptationMode> parseAdaptationMode(std::string_view value)
    {
        if (value == "fixed")
            return GaussianSplatFoveatedAdaptationMode::eFixed;
        if (value == "dynamic-budget" || value == "budget")
            return GaussianSplatFoveatedAdaptationMode::eDynamicBudget;
        if (value == "stability-aware" || value == "stable-budget" ||
            value == "stability-aware-budget" || value == "temporal-stability-aware")
            return GaussianSplatFoveatedAdaptationMode::eStabilityAwareBudget;
        if (value == "dynamic-range" || value == "range")
            return GaussianSplatFoveatedAdaptationMode::eDynamicRange;
        if (value == "progressive" || value == "center-out" || value == "progressive-center-out")
            return GaussianSplatFoveatedAdaptationMode::eProgressiveCenterOut;
        if (value == "progressive-greedy" || value == "old-progressive")
            return GaussianSplatFoveatedAdaptationMode::eProgressiveGreedy;
        return std::nullopt;
    }

    const char* adaptationModeLabel(const GaussianSplatFoveatedAdaptationMode mode)
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

    std::optional<GaussianSplatFoveatedDistribution> parseDistribution(std::string_view value)
    {
        if (value == "hard-ring" || value == "hard" || value == "ring")
            return GaussianSplatFoveatedDistribution::eHardRing;
        if (value == "smoothstep" || value == "smooth")
            return GaussianSplatFoveatedDistribution::eSmoothstep;
        if (value == "gaussian" || value == "normal")
            return GaussianSplatFoveatedDistribution::eGaussian;
        if (value == "exponential" || value == "exp")
            return GaussianSplatFoveatedDistribution::eExponential;
        if (value == "inverse-power" || value == "inverse" || value == "hyperbolic")
            return GaussianSplatFoveatedDistribution::eInversePower;
        if (value == "log-polar" || value == "logpolar")
            return GaussianSplatFoveatedDistribution::eLogPolar;
        if (value == "cortical" || value == "cortical-style" || value == "cortical-magnification")
            return GaussianSplatFoveatedDistribution::eCortical;
        if (value == "cone-density" || value == "cone-density-fitted" || value == "cone" ||
            value == "photoreceptor")
            return GaussianSplatFoveatedDistribution::eConeDensityFitted;
        if (value == "continuous" || value == "continuous-scheduler" ||
            value == "eccentricity-continuous" || value == "eccentricity")
        {
            return GaussianSplatFoveatedDistribution::eContinuousScheduler;
        }
        return std::nullopt;
    }

    std::optional<GaussianSplatFoveatedCoverageGuardMode> parseCoverageGuardMode(std::string_view value)
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

    const char* coverageGuardModeLabel(const GaussianSplatFoveatedCoverageGuardMode mode)
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

    std::optional<GaussianSplatShLodGuardMode> parseShLodGuardMode(std::string_view value)
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

    std::optional<GaussianSplatShStorageLayout> parseShStorageLayout(std::string_view value)
    {
        const auto normalized = normalizeToken(value);
        if (normalized == "monolithic" || normalized == "default")
            return GaussianSplatShStorageLayout::eMonolithic;
        if (normalized == "split-bands" || normalized == "split" || normalized == "bands")
            return GaussianSplatShStorageLayout::eSplitBands;
        return std::nullopt;
    }

    std::string splatUriFromCliValue(std::string_view value)
    {
        if (value.find("://") != std::string_view::npos)
            return std::string {value};

        if (value == "hornedlizard" || value == "lizard")
            return "res://models/3dgs/hornedlizard.spz";
        if (value == "racoonfamily" || value == "racoon")
            return "res://models/3dgs/racoonfamily.spz";

        if (value == "playroom")
            return "res://models/3dgs/playroom_point_cloud.ply";
        if (value == "drjohnson")
            return "res://../datasets/3dgs-pretrained/Voxel51/drjohnson/point_cloud/iteration_30000/point_cloud.ply";
        if (value == "train")
            return "res://../datasets/3dgs-pretrained/Voxel51/train/point_cloud/iteration_30000/point_cloud.ply";
        if (value == "truck")
            return "res://../datasets/3dgs-pretrained/Voxel51/truck/point_cloud/iteration_30000/point_cloud.ply";

        return std::string {value};
    }

    OpenXRGaussianOptions parseOptions(std::span<const std::string> args)
    {
        OpenXRGaussianOptions options {};
        for (size_t i = 0; i < args.size(); ++i)
        {
            const std::string_view arg = args[i];
            if (const auto value = takeOptionValue(args, i, "--splat"))
            {
                options.splatUri = splatUriFromCliValue(*value);
                continue;
            }
            if (arg == "--xr-gaze")
            {
                options.xrGazeEnabled = true;
                continue;
            }
            if (arg == "--no-xr-gaze")
            {
                options.xrGazeEnabled = false;
                continue;
            }
            if (arg == "--foveated-temporal-hysteresis" || arg == "--temporal-hysteresis")
            {
                options.temporalHysteresisEnabled = true;
                continue;
            }
            if (arg == "--no-foveated-temporal-hysteresis" || arg == "--no-temporal-hysteresis")
            {
                options.temporalHysteresisEnabled = false;
                continue;
            }
            if (arg == "--foveated-boundary-smoothing" || arg == "--boundary-smoothing")
            {
                options.boundarySmoothingEnabled = true;
                continue;
            }
            if (arg == "--no-foveated-boundary-smoothing" || arg == "--no-boundary-smoothing")
            {
                options.boundarySmoothingEnabled = false;
                continue;
            }
            if (arg == "--sh-lod" || arg == "--foveated-sh-lod")
            {
                options.shLodEnabled = true;
                continue;
            }
            if (arg == "--no-sh-lod" || arg == "--no-foveated-sh-lod")
            {
                options.shLodEnabled = false;
                continue;
            }
            if (const auto value = takeOptionValue(args, i, "--gaze-render-mode"))
            {
                if (const auto mode = parseRenderMode(*value))
                    options.renderMode = *mode;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --gaze-render-mode value: {}", *value);
                continue;
            }
            if (const auto value = takeOptionValue(args, i, "--gaze-adaptation"))
            {
                if (const auto mode = parseAdaptationMode(*value))
                    options.adaptationMode = *mode;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --gaze-adaptation value: {}", *value);
                continue;
            }
            if (const auto value = takeOptionValue(args, i, "--gaze-distribution"))
            {
                if (const auto distribution = parseDistribution(*value))
                    options.distribution = *distribution;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --gaze-distribution value: {}", *value);
                continue;
            }
            if (const auto value = takeOptionValue(args, i, "--foveated-distribution"))
            {
                if (const auto distribution = parseDistribution(*value))
                    options.distribution = *distribution;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --foveated-distribution value: {}", *value);
                continue;
            }
            if (const auto value = takeOptionValue(args, i, "--coverage-guard"))
            {
                if (const auto mode = parseCoverageGuardMode(*value))
                    options.coverageGuardMode = *mode;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-guard value: {}", *value);
                continue;
            }
            if (const auto value = takeOptionValue(args, i, "--sh-lod-guard"))
            {
                if (const auto mode = parseShLodGuardMode(*value))
                    options.shLodGuardMode = *mode;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --sh-lod-guard value: {}", *value);
                continue;
            }
            if (const auto value = takeOptionValue(args, i, "--sh-storage-layout"))
            {
                if (const auto layout = parseShStorageLayout(*value))
                    options.shStorageLayout = *layout;
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --sh-storage-layout value: {}", *value);
                continue;
            }
            if (const auto value = takeOptionValue(args, i, "--target-frame-ms"))
            {
                options.targetFrameMs = parseFloat(*value);
                if (!options.targetFrameMs)
                    VULTRA_CLIENT_WARN("Ignoring invalid --target-frame-ms value: {}", *value);
                continue;
            }
            if (const auto value = takeOptionValue(args, i, "--coverage-protection-degrees"))
            {
                options.coverageProtectionDegrees = parseFloat(*value);
                if (!options.coverageProtectionDegrees)
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-protection-degrees value: {}", *value);
                continue;
            }
            if (const auto value = takeOptionValue(args, i, "--gaze-x"))
            {
                options.gazeX = parseFloat(*value);
                if (!options.gazeX)
                    VULTRA_CLIENT_WARN("Ignoring invalid --gaze-x value: {}", *value);
                continue;
            }
            if (const auto value = takeOptionValue(args, i, "--gaze-y"))
            {
                options.gazeY = parseFloat(*value);
                if (!options.gazeY)
                    VULTRA_CLIENT_WARN("Ignoring invalid --gaze-y value: {}", *value);
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
            if (const auto value = takeOptionValue(args, i, "--sh-lod-guard-threshold-mid"))
            {
                options.shLodGuardThresholdMid = parseFloat(*value);
                if (!options.shLodGuardThresholdMid)
                    VULTRA_CLIENT_WARN("Ignoring invalid --sh-lod-guard-threshold-mid value: {}", *value);
                continue;
            }
            if (const auto value = takeOptionValue(args, i, "--sh-guard-threshold-mid"))
            {
                options.shLodGuardThresholdMid = parseFloat(*value);
                if (!options.shLodGuardThresholdMid)
                    VULTRA_CLIENT_WARN("Ignoring invalid --sh-guard-threshold-mid value: {}", *value);
                continue;
            }
            if (const auto value = takeOptionValue(args, i, "--sh-lod-guard-threshold-high"))
            {
                options.shLodGuardThresholdHigh = parseFloat(*value);
                if (!options.shLodGuardThresholdHigh)
                    VULTRA_CLIENT_WARN("Ignoring invalid --sh-lod-guard-threshold-high value: {}", *value);
                continue;
            }
            if (const auto value = takeOptionValue(args, i, "--sh-guard-threshold-high"))
            {
                options.shLodGuardThresholdHigh = parseFloat(*value);
                if (!options.shLodGuardThresholdHigh)
                    VULTRA_CLIENT_WARN("Ignoring invalid --sh-guard-threshold-high value: {}", *value);
                continue;
            }
            if (const auto value = takeOptionValue(args, i, "--coverage-guard-budget-ratio"))
            {
                if (const auto ratio = parseFloat(*value))
                    options.coverageGuardBudgetRatio = std::clamp(*ratio, 0.0f, 0.25f);
                else
                    VULTRA_CLIENT_WARN("Ignoring invalid --coverage-guard-budget-ratio value: {}", *value);
                continue;
            }
        }
        return options;
    }

    void applyGaussianSplatOverride(World& world, IAssetService& assets, const std::string& uri)
    {
        auto handle = assets.loadGaussianSplatSync(uri);
        if (!handle.ready())
        {
            VULTRA_CLIENT_WARN("Ignoring --splat override; failed to resolve {}", uri);
            return;
        }

        auto view = world.registry().view<GaussianSplatComponent>();
        uint32_t replacedSplats {0u};
        for (auto entity : view)
        {
            view.get<GaussianSplatComponent>(entity).gaussianSplat = handle.uuid();
            ++replacedSplats;
        }

        VULTRA_CLIENT_INFO("Applied Gaussian splat override: {} (uuid={}, entities={})",
                           uri,
                           handle.uuid().toString(),
                           replacedSplats);
    }

    void applyOpenXRFoveatedDefaults(GaussianSplatRenderSettings& settings, const OpenXRGaussianOptions& options)
    {
        settings.baselineMode                         = GaussianSplatBaselineMode::eOrderedClod;
        settings.lodBudget                            = 0u;
        settings.clodLevel                            = 1.0f;
        settings.foveatedClodEnabled                  = true;
        settings.foveatedManualGazeControlEnabled     = false;
        settings.foveatedXrGazeEnabled                = options.xrGazeEnabled;
        settings.foveatedCoverageCompensationEnabled  = false;
        settings.foveatedRenderMode                   = options.renderMode;
        settings.foveatedGaze                         = glm::vec2 {0.5f, 0.5f};
        if (options.gazeX || options.gazeY)
        {
            settings.foveatedManualGazeControlEnabled = true;
            settings.foveatedXrGazeEnabled            = false;
            if (options.gazeX)
                settings.foveatedGaze.x = std::clamp(*options.gazeX, 0.0f, 1.0f);
            if (options.gazeY)
                settings.foveatedGaze.y = std::clamp(*options.gazeY, 0.0f, 1.0f);
        }
        settings.foveatedRingDegrees                  = glm::vec2 {12.0f, 32.0f};
        settings.foveatedRingLevels                   = glm::vec3 {1.0f, 0.40f, 0.15f};
        settings.foveatedResolutionScales             = glm::vec3 {1.0f, 0.75f, 0.50f};
        settings.foveatedTransitionDegrees            = 8.0f;
        settings.foveatedDistribution                 = options.distribution;
        settings.foveatedAdaptationMode               = options.adaptationMode;
        settings.foveatedTargetFrameMs                = std::max(options.targetFrameMs.value_or(11.1f), 0.1f);
        settings.foveatedBudgetAdjustRate             = 0.05f;
        settings.foveatedCoverageGuardMode            = options.coverageGuardMode;
        settings.foveatedCoverageProtectionDegrees =
            std::clamp(options.coverageProtectionDegrees.value_or(0.0f), 0.0f, 12.0f);
        settings.foveatedCoverageGuardBudgetRatio =
            std::clamp(options.coverageGuardBudgetRatio, 0.0f, 0.25f);
        settings.foveatedCoverageGuardMinLevels       = glm::vec3 {0.75f, 0.45f, 0.15f};
        settings.foveatedTemporalHysteresisEnabled    = options.temporalHysteresisEnabled;
        settings.foveatedBoundarySmoothingEnabled     = options.boundarySmoothingEnabled;
        settings.foveatedTemporalResidencyFrames      = 6u;
        settings.foveatedTemporalHysteresisRatio      = 0.25f;
        settings.foveatedBoundarySmoothingRatio       = 0.18f;
        if (options.shLodEnabled)
            settings.foveatedShLodEnabled = *options.shLodEnabled;
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

        if (settings.foveatedAdaptationMode == GaussianSplatFoveatedAdaptationMode::eProgressiveCenterOut ||
            settings.foveatedAdaptationMode == GaussianSplatFoveatedAdaptationMode::eProgressiveGreedy ||
            settings.foveatedAdaptationMode == GaussianSplatFoveatedAdaptationMode::eStabilityAwareBudget)
        {
            settings.foveatedRingDegrees       = glm::vec2 {8.0f, 24.0f};
            settings.foveatedRingLevels        = glm::vec3 {0.35f, 0.18f, 0.08f};
            settings.foveatedResolutionScales  = glm::vec3 {1.0f, 1.0f, 1.0f};
            settings.foveatedTransitionDegrees = 8.0f;
            settings.foveatedDistribution      = options.distribution;
        }
    }
} // namespace

class OpenXRGaussianSplattingApp final : public DemoAppHost
{
protected:
    std::string_view demoWindowTitle() const override { return "OpenXR Gaussian Splatting"; }

    void onPostConfigureDemo(Engine& engine) override
    {
        const auto options = parseOptions(commandLineArgs());

        auto& sceneService = engine.ctx().services.require<ISceneService>();
        auto& worldService = engine.ctx().services.require<IWorldService>();
        auto& assetService = engine.ctx().services.require<IAssetService>();
        auto& renderService = engine.ctx().services.require<IRenderService>();

        auto& world = worldService.world();
        sceneService.instantiateScene(world, "res://scenes/3dgs_example.vmanifest");
        if (options.splatUri)
            applyGaussianSplatOverride(world, assetService, *options.splatUri);

        applyOpenXRFoveatedDefaults(renderService.gaussianSplatSettings(), options);

        VULTRA_CLIENT_INFO("Loaded world from scene manifest: \"res://scenes/3dgs_example.vmanifest\"");
        const auto& settings = renderService.gaussianSplatSettings();
        VULTRA_CLIENT_INFO("OpenXR Gaussian foveated path enabled: ordered CLOD, XR gaze={}, manual gaze={}, "
                           "gaze_uv=({:.2f}, {:.2f}), adaptation={}, coverage compensation={}, coverage guard={}",
                           settings.foveatedXrGazeEnabled ? "yes" : "no",
                           settings.foveatedManualGazeControlEnabled ? "yes" : "no",
                           settings.foveatedGaze.x,
                           settings.foveatedGaze.y,
                           adaptationModeLabel(settings.foveatedAdaptationMode),
                           settings.foveatedCoverageCompensationEnabled ? "yes" : "no",
                           coverageGuardModeLabel(settings.foveatedCoverageGuardMode));
    }

    rhi::RenderDeviceFeatureFlagBits demoRenderDeviceFeatureFlag() const override
    {
        return rhi::RenderDeviceFeatureFlagBits::eXR;
    }
};

int main(int argc, char** argv)
{
    OpenXRGaussianSplattingApp app {};
    return app.run(argc, argv);
}

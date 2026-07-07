[vshader]
language = glsl
version = 460

[keywords]
USE_MULTIVIEW : bool permute
USE_DIRECT_PREFIX : bool permute
USE_FOVEATED_LAYER_OUTPUT : bool permute
USE_SH_LOD_GUARD : bool permute
USE_SH_LOD_ENERGY_GUARD : bool permute
USE_SH_SPLIT_BANDS : bool permute
USE_DETERMINISTIC_SOURCE_ORDER_SORT : bool permute

[comp]
#define VULTRA_DECLARE_CAMERA
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_DRAW_BUFFER
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_PACKED_SOURCE_BUFFER
#if !USE_DIRECT_PREFIX
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_SELECTED_SOURCE_BUFFER
#endif
#if USE_FOVEATED_LAYER_OUTPUT
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_FOVEATED_LAYER_BUFFERS
#else
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_VISIBLE_SPLAT_BUFFER_READWRITE
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_SORT_KEY_BUFFER_READWRITE
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_SORT_INDEX_BUFFER_READWRITE
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_VISIBLE_COUNT_BUFFER
#endif
#if USE_SH_SPLIT_BANDS
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_SH_SPLIT_BAND_BUFFERS
#else
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_SH_BUFFER
#endif
#if USE_SH_LOD_ENERGY_GUARD
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_SH_ENERGY_METADATA_BUFFER
#endif
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_TEMPORAL_STATE_BUFFER_READWRITE
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_ECSPT_COUNTER_BUFFER
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_COVERAGE_TEXTURE_BUFFER
#if USE_MULTIVIEW
#define VULTRA_DECLARE_STEREO_CAMERA
#endif
#include "include/common/gpu_scene.glsl"
#include "include/common/gaussian_splat_foveated.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

const float MIN_VISIBLE_OPACITY = 0.02;
const uint SORT_ORDER_Z_DEPTH = 0u;
const uint SORT_ORDER_DISTANCE = 1u;
const uint SORT_ORDER_VIEW_DEPTH = 2u;
const uint SORT_ORDER_CONSERVATIVE_DEPTH = 3u;
const uint FOVEATED_CLOD_ENABLED_FLAG = 1u;
const uint FOVEATED_CLOD_COVERAGE_COMPENSATION_FLAG = 8u;
const uint FOVEATED_CLOD_TEMPORAL_HYSTERESIS_FLAG = 16u;
const uint FOVEATED_CLOD_BOUNDARY_SMOOTHING_FLAG = 32u;
const uint FOVEATED_CLOD_SCORE_SELECTED_FLAG = 64u;
const uint FOVEATED_SH_LOD_ENABLED_FLAG = 1u;
const uint FOVEATED_SH_SMOOTH_SUPPRESSION_FLAG = 2u;
const uint FOVEATED_SH_LOD_GUARD_OFF = 0u;
const uint FOVEATED_SH_LOD_GUARD_ENERGY = 1u;
const uint FOVEATED_SH_LOD_GUARD_PROJECTED_COST = 2u;
const uint FOVEATED_SH_LOD_GUARD_ENERGY_PROJECTED_COST = 3u;
const uint SHADER_ANTIPOP_OFF = 0u;
const uint SHADER_ANTIPOP_SOFT_RAMP = 1u;
const uint SHADER_ANTIPOP_HASH_RAMP = 2u;
const uint SHADER_ANTIPOP_HASH_RAMP_GUARDED = 3u;
const uint SHADER_ANTIPOP_LINEAR_DEFAULT = 4u;
const uint SHADER_ANTIPOP_GUARDED_GAZE_ANCHOR_CROSSFADE = 5u;
const uint SHADER_ANTIPOP_ECCENTRICITY_STOCHASTIC_TRANSITION = 6u;
const uint SHADER_ANTIPOP_STABLE_OPTICAL_DEPTH_THINNING = 7u;
const uint SHADER_ANTIPOP_COVERAGE_STABLE_LOGPOLAR_RELEASE = 8u;
const uint SHADER_ANTIPOP_PKEEP_CURRENT = 0u;
const uint SHADER_ANTIPOP_PKEEP_LINEAR = 1u;
const uint SHADER_ANTIPOP_PKEEP_SMOOTH_WIDE = 2u;
const uint SHADER_ANTIPOP_PKEEP_LOGISTIC_SOFT = 3u;
const uint SHADER_ANTIPOP_PKEEP_LOGISTIC_STEEP = 4u;
const uint SHADER_ANTIPOP_NORMALIZE_OFF = 0u;
const uint SHADER_ANTIPOP_NORMALIZE_GLOBAL_LUMA = 1u;
const uint SHADER_ANTIPOP_NORMALIZE_ALPHA_MASS = 2u;
const uint GAZE_ANCHOR_FLAG_ENABLED = 1u;
const uint GAZE_ANCHOR_FLAG_IMMEDIATE_FOVEA_FILL = 2u;
const uint GAZE_ANCHOR_FLAG_CONTRIBUTION_GUARD = 4u;
const uint GAZE_ANCHOR_FLAG_NAIVE_CAP = 8u;
const uint ECC_STOCHASTIC_FLAG_ENABLED = 1u;
const uint ECC_STOCHASTIC_FLAG_PROTECT_OLD_FOVEA = 2u;
const uint ECC_STOCHASTIC_FLAG_PROTECT_NEW_FOVEA = 4u;
const uint ECC_STOCHASTIC_FLAG_PROTECT_BOUNDARY = 8u;
const uint ECC_STOCHASTIC_FLAG_CONTRIBUTION_GUARD = 16u;
const uint ECSPT_COUNTER_TOTAL_CANDIDATES_SEEN = 0u;
const uint ECSPT_COUNTER_BASE_NEW_SELECTED_COUNT = 1u;
const uint ECSPT_COUNTER_EFFECTIVE_VISIBLE_AFTER_ECSPT_COUNT = 2u;
const uint ECSPT_COUNTER_SHARED_COUNT = 3u;
const uint ECSPT_COUNTER_UPGRADE_COUNT = 4u;
const uint ECSPT_COUNTER_DOWNGRADE_COUNT = 5u;
const uint ECSPT_COUNTER_PROTECTED_DOWNGRADE_COUNT = 6u;
const uint ECSPT_COUNTER_DROPPED_DOWNGRADE_COUNT = 7u;
const uint ECSPT_COUNTER_OLD_ONLY_FADE_VISIBLE_COUNT = 8u;
const uint ECSPT_COUNTER_IMMEDIATE_NEW_FOVEA_COUNT = 9u;
const uint ECSPT_COUNTER_BOUNDARY_PROTECTED_COUNT = 10u;
const uint ECSPT_COUNTER_CONTRIBUTION_GUARD_PROTECTED_COUNT = 11u;
const uint ECSPT_COUNTER_ZERO_WEIGHT_DISCARD_COUNT = 12u;
const uint ECSPT_COUNTER_MIN_P_DELTA_DISCARD_COUNT = 13u;
const uint ECSPT_COUNTER_FAR_PERIPHERY_HARD_DROP_COUNT = 14u;
const uint ECSPT_COUNTER_STABLE_OPTICAL_DEPTH_ALPHA_CLAMPED_COUNT = 15u;
const uint ECSPT_COUNTER_COVERAGE_FLOOR_ACTIVE_COUNT = 16u;
const uint ECSPT_COUNTER_COVERAGE_FLOOR_RAISED_COUNT = 17u;
const uint ECSPT_COUNTER_COVERAGE_RELEASE_HELD_COUNT = 18u;
const uint ECSPT_COUNTER_COVERAGE_FINAL_DROPPED_COUNT = 19u;
const uint ECSPT_COUNTER_COVERAGE_OLD_ONLY_RELEASE_COUNT = 20u;
const uint ECSPT_COUNTER_COVERAGE_P_FLOOR_SUM = 21u;
const uint ECSPT_COUNTER_COVERAGE_P_FLOOR_MAX = 22u;
const uint ECSPT_COUNTER_COVERAGE_D_ALPHA_SUM = 23u;
const uint ECSPT_COUNTER_COVERAGE_N_EFF_SUM = 24u;
const uint ECSPT_COUNTER_COVERAGE_FLOOR_SAMPLE_COUNT = 25u;
const uint ECSPT_COUNTER_COVERAGE_P_LP_SUM = 26u;
const uint ECSPT_COUNTER_COVERAGE_P_STATIC_SUM = 27u;
const uint ECSPT_COUNTER_COVERAGE_RELEASE_DELTA_SUM = 28u;
const uint ECSPT_COUNTER_COVERAGE_HISTORY_RESET_COUNT = 29u;
const uint ECSPT_COUNTER_COVERAGE_REENTRY_RESET_COUNT = 30u;
const uint ECSPT_COUNTER_COVERAGE_RELEASE_HELD_EFFECTIVE_COUNT = 31u;
const uint ECSPT_COUNTER_COVERAGE_RELEASE_EPSILON_CUTOFF_COUNT = 32u;
const uint ECSPT_COUNTER_COVERAGE_SKIPPED_FLOOR_UPDATE_COUNT = 33u;
const uint ECSPT_COUNTER_COVERAGE_CANDIDATE_ALPHA_SUM = 34u;
const uint ECSPT_COUNTER_COVERAGE_RETAINED_ALPHA_SUM = 35u;
const uint ECSPT_COUNTER_VISIBLE_INSTANT_COUNT = 36u;
const uint ECSPT_COUNTER_TILE_INSTANCE_COUNT = 37u;
const uint ECSPT_COUNTER_COVERED_PIXEL_COUNT = 38u;
const uint ECSPT_COUNTER_COVERAGE_TEXTURE_ALPHA_SUM = 39u;
const uint ECSPT_COUNTER_COVERAGE_TEXTURE_ALPHA2_SUM = 40u;
const uint ECSPT_COUNTER_COVERAGE_TEXTURE_NONZERO_TEXEL_COUNT = 41u;
const uint ECSPT_COUNTER_COVERAGE_TEXTURE_WIDTH = 42u;
const uint ECSPT_COUNTER_COVERAGE_TEXTURE_HEIGHT = 43u;
const uint ECSPT_COUNTER_COVERAGE_P_FINAL_SUM = 45u;
const uint ECSPT_COUNTER_COVERAGE_STABLE_HASH_KEPT_COUNT = 54u;
const uint ECSPT_COUNTER_COVERAGE_GUIDE_REDUCED_COUNT = 55u;
const uint ECSPT_COUNTER_COVERAGE_GUIDE_BOOSTED_COUNT = 56u;
const uint ECSPT_COUNTER_COVERAGE_P_HISTORY_NONZERO_COUNT = 57u;
const uint ECSPT_COUNTER_COVERAGE_P_TARGET_LESS_THAN_HISTORY_COUNT = 58u;
const uint ECSPT_COUNTER_COVERAGE_RELEASED_EFFECTIVE_VISIBLE_COUNT = 59u;
const uint ECSPT_COUNTER_COVERAGE_RELEASED_ALPHA_PROXY_SUM = 60u;
const uint ECSPT_COUNTER_COVERAGE_RELEASE_CAP_HIT_COUNT = 61u;
const uint ECSPT_COUNTER_COVERAGE_RISK_PROTECTED_COUNT = 62u;
const uint ECSPT_COUNTER_COVERAGE_FOOTPRINT_FAST_DECAY_COUNT = 63u;
const uint ECSPT_COUNTER_SCALAR_COUNT = 64u;
const uint ECSPT_COUNTER_EVENT_BIN_COUNT = 5u;
const uint ECSPT_COUNTER_EVENT_TYPE_COUNT = 6u;
const uint ECSPT_COUNTER_EVENT_METRIC_COUNT = 7u;
const uint ECSPT_EVENT_SHARED = 0u;
const uint ECSPT_EVENT_UPGRADE = 1u;
const uint ECSPT_EVENT_DOWNGRADE_PROTECTED = 2u;
const uint ECSPT_EVENT_DOWNGRADE_DROPPED = 3u;
const uint ECSPT_EVENT_INSTANT_HASH_FLIP = 4u;
const uint ECSPT_EVENT_IMMEDIATE_NEW_FOVEA = 5u;
const uint ECSPT_METRIC_COUNT = 0u;
const uint ECSPT_METRIC_SUM_OPACITY = 1u;
const uint ECSPT_METRIC_SUM_PROJECTED_AREA_PROXY = 2u;
const uint ECSPT_METRIC_SUM_CONTRIBUTION_PROXY = 3u;
const uint ECSPT_METRIC_SUM_ABS_DELTA_P = 4u;
const uint ECSPT_METRIC_SUM_P_OLD = 5u;
const uint ECSPT_METRIC_SUM_P_NEW = 6u;
const uint ECSPT_EVAL_REACHED = 1u << 0u;
const uint ECSPT_EVAL_OLD_SELECTED = 1u << 1u;
const uint ECSPT_EVAL_NEW_SELECTED = 1u << 2u;
const uint ECSPT_EVAL_SHARED = 1u << 3u;
const uint ECSPT_EVAL_UPGRADE = 1u << 4u;
const uint ECSPT_EVAL_DOWNGRADE = 1u << 5u;
const uint ECSPT_EVAL_PROTECTED_DOWNGRADE = 1u << 6u;
const uint ECSPT_EVAL_DROPPED_DOWNGRADE = 1u << 7u;
const uint ECSPT_EVAL_IMMEDIATE_NEW_FOVEA = 1u << 8u;
const uint ECSPT_EVAL_BOUNDARY_PROTECTED = 1u << 9u;
const uint ECSPT_EVAL_CONTRIBUTION_GUARD_PROTECTED = 1u << 10u;
const uint ECSPT_EVAL_ZERO_WEIGHT = 1u << 11u;
const uint ECSPT_EVAL_MIN_P_DELTA_DISCARD = 1u << 12u;
const uint ECSPT_EVAL_FAR_PERIPHERY_HARD_DROP = 1u << 13u;
const uint ECSPT_EVAL_STABLE_OPTICAL_DEPTH_ALPHA_CLAMPED = 1u << 14u;
const uint ECSPT_EVAL_COVERAGE_FLOOR_ACTIVE = 1u << 15u;
const uint ECSPT_EVAL_COVERAGE_FLOOR_RAISED = 1u << 16u;
const uint ECSPT_EVAL_COVERAGE_RELEASE_HELD = 1u << 17u;
const uint ECSPT_EVAL_COVERAGE_OLD_ONLY_RELEASE = 1u << 18u;
const uint COVERAGE_RELEASE_FLAG_COVERAGE_FLOOR = 1u;
const uint COVERAGE_RELEASE_FLAG_STABLE_HASH = 2u;
const uint COVERAGE_RELEASE_FLAG_STAGGERED_RELEASE = 4u;
const uint COVERAGE_RELEASE_FLAG_DIAGNOSTICS = 8u;
const uint COVERAGE_RELEASE_FLAG_SATURATED_FLOOR = 16u;
const uint COVERAGE_RELEASE_FLAG_TEXTURE_FLOOR = 32u;
const uint COVERAGE_RELEASE_FLAG_TEXTURE_UPDATE_THIS_FRAME = 64u;
const uint COVERAGE_RELEASE_FLAG_GUIDE_MODIFIER = 128u;
const uint COVERAGE_RELEASE_FLAG_GUIDE_HISTORY_VALID = 256u;
const uint COVERAGE_RELEASE_POLICY_CURRENT = 0u;
const uint COVERAGE_RELEASE_POLICY_RISK_ONLY = 1u;
const uint COVERAGE_RELEASE_POLICY_CAPPED = 2u;
const uint COVERAGE_RELEASE_POLICY_FOOTPRINT_DECAY = 3u;
const uint COVERAGE_RELEASE_POLICY_TILE_LOCAL_CAPPED = 4u;
const uint COVERAGE_RELEASE_PHASE_RESOLVE = 1u << 30u;
const uint COVERAGE_RELEASE_PHASE_ACCUMULATE = 1u << 31u;
const uint COVERAGE_RELEASE_TILE_COUNTER_OFFSET = 256u;
const uint COVERAGE_RELEASE_MAX_TILE_COUNT = 512u;
const uint COVERAGE_RELEASE_TILE_SATURATED_OFFSET =
    COVERAGE_RELEASE_TILE_COUNTER_OFFSET + COVERAGE_RELEASE_MAX_TILE_COUNT * 2u;
const uint RELEASE_COST_TILE_BASE_DUPLICATE_OFFSET =
    COVERAGE_RELEASE_TILE_COUNTER_OFFSET + COVERAGE_RELEASE_MAX_TILE_COUNT * 3u;
const uint RELEASE_COST_TILE_RELEASED_DUPLICATE_OFFSET =
    COVERAGE_RELEASE_TILE_COUNTER_OFFSET + COVERAGE_RELEASE_MAX_TILE_COUNT * 4u;
const uint RELEASE_COST_SUMMARY_OFFSET =
    COVERAGE_RELEASE_TILE_COUNTER_OFFSET + COVERAGE_RELEASE_MAX_TILE_COUNT * 5u;
const uint RELEASE_COST_BASE_DUPLICATE_COUNTER =
    RELEASE_COST_SUMMARY_OFFSET + 0u;
const uint RELEASE_COST_RELEASED_DUPLICATE_COUNTER =
    RELEASE_COST_SUMMARY_OFFSET + 1u;
const uint RELEASE_COST_LARGE_RELEASED_COUNTER =
    RELEASE_COST_SUMMARY_OFFSET + 2u;
const uint RELEASE_COST_LARGE_RELEASED_DUPLICATE_COUNTER =
    RELEASE_COST_SUMMARY_OFFSET + 3u;
const uint RELEASE_COST_RELEASED_RADIUS_SUM_COUNTER =
    RELEASE_COST_SUMMARY_OFFSET + 4u;
const uint RELEASE_COST_RELEASED_RADIUS_MAX_COUNTER =
    RELEASE_COST_SUMMARY_OFFSET + 5u;
const uint RELEASE_COST_RELEASED_RADIUS_HIST_OFFSET =
    RELEASE_COST_SUMMARY_OFFSET + 8u;
const uint RELEASE_COST_RELEASED_RADIUS_HIST_COUNT = 8u;
const float COVERAGE_RELEASE_COUNTER_SCALE = 1024.0;
const float COVERAGE_RELEASE_TILE_COUNTER_SCALE = 2048.0;
const float RELEASE_COST_RADIUS_COUNTER_SCALE = 4.0;
const float RELEASE_COST_PROJECTED_TILE_SIZE_PX = 16.0;
const uint FOVEATED_COVERAGE_GUARD_OFF = 0u;
const uint FOVEATED_COVERAGE_GUARD_GLOBAL = 1u;
const uint FOVEATED_COVERAGE_GUARD_LOCAL_BOUNDED = 2u;
const uint FOVEATED_COVERAGE_GUARD_RISK_TRIGGERED = 3u;
const float FOVEATED_COVERAGE_MIN_LEVEL = 0.12;
const float FOVEATED_COVERAGE_MAX_ALPHA_BOOST = 2.0;
const float FOVEATED_COVERAGE_MAX_RADIUS_SCALE = 1.35;
const uint FOVEATED_TEMPORAL_INACTIVE = 0u;
const uint FOVEATED_TEMPORAL_CANDIDATE = 1u;
const uint FOVEATED_TEMPORAL_ACTIVE = 2u;
const uint FOVEATED_TEMPORAL_COOLDOWN = 3u;
const uint FOVEATED_TEMPORAL_PHASE_MASK = 3u;
const uint FOVEATED_TEMPORAL_RESIDENCY_SHIFT = 8u;
const uint FOVEATED_TEMPORAL_RESIDENCY_MASK = 255u;
const float SH_C1 = 0.4886025119029199;
const float SH_C2[5] = float[5](1.0925484305920792,
                                -1.0925484305920792,
                                0.31539156525252005,
                                -1.0925484305920792,
                                0.5462742152960396);
const float SH_C3[7] = float[7](-0.5900435899266435,
                                2.890611442640554,
                                -0.4570457994644658,
                                0.3731763325901154,
                                -0.4570457994644658,
                                1.445305721320277,
                                -0.5900435899266435);

layout(push_constant) uniform GeneralGaussianSplatPreprocessPushConstants
{
    uint pointCount;
    uint maxVisibleSplats;
    uint rankTotalCount;
    uint foveatedClodEnabled;
    vec4 foveatedGazeAndRings;
    vec4 foveatedLevelsAndTransition;
    uvec4 foveatedLayerParams;
    vec4 foveatedCoverageGuardParams;
    vec4 foveatedTemporalParams;
    vec4 foveatedContinuousParams;
    uvec4 shLodParams;
    vec4 shGuardParams;
    uvec4 shaderAntiPopParams;
    vec4 shaderAntiPopFloats;
    uvec4 gazeAnchorParams;
    vec4 gazeAnchorGazes;
    vec4 gazeAnchorFloats;
    uvec4 coverageReleaseParams;
    vec4 coverageReleaseFloats;
    vec4 coverageTextureFloats;
    uvec4 coverageReleaseCostParams;
    vec4 coverageReleaseCostFloats;
} u_PC;

struct EyePreprocessResult
{
    vec2 v1;
    vec2 v2;
    vec2 centerNdc;
    float eccentricityDegrees;
    float depth;
    vec4 colorOpacity;
    float sortDepth;
    float eccStochasticWeight;
    float eccStochasticOpacity;
    float eccStochasticContributionProxy;
    float eccStochasticPOld;
    float eccStochasticPNew;
    float eccStochasticPDelta;
    float eccStochasticOldEccentricityDegrees;
    float eccStochasticNewEccentricityDegrees;
    uint eccStochasticFlags;
    float projectedRadiusPx;
    uint tileDuplicateProxy;
    bool visible;
};

struct EccStochasticEval
{
    float weight;
    float opacity;
    float contributionProxy;
    float pOld;
    float pNew;
    float pDelta;
    float oldEccentricityDegrees;
    float newEccentricityDegrees;
    uint flags;
};

float computeFoveatedEccentricityDegreesForGaze(const vec2 centerNdc,
                                                const CameraData camera,
                                                const vec2 gazeUv)
{
    const vec2 tanHalfFov = vec2(1.0 / max(abs(camera.projection[0][0]), 1e-5),
                                 1.0 / max(abs(camera.projection[1][1]), 1e-5));
    const float projectionYSign = camera.projection[1][1] < 0.0 ? -1.0 : 1.0;
    return gaussianFoveatedEccentricityDegreesFromNdc(
        centerNdc,
        clamp(gazeUv, vec2(0.0), vec2(1.0)),
        tanHalfFov,
        projectionYSign);
}

float computeFoveatedEccentricityDegrees(const vec2 centerNdc, const CameraData camera)
{
    return computeFoveatedEccentricityDegreesForGaze(centerNdc, camera, u_PC.foveatedGazeAndRings.xy);
}

float foveatedBaseClodLevelForEccentricity(const float eccentricityDegrees)
{
    return gaussianFoveatedClodLevel(eccentricityDegrees,
                                     u_PC.foveatedGazeAndRings.zw,
                                     u_PC.foveatedLevelsAndTransition.xyz,
                                     u_PC.foveatedLevelsAndTransition.w,
                                     u_PC.foveatedLayerParams.y,
                                     u_PC.foveatedContinuousParams);
}

float foveatedLocalCoverageFloor(const float eccentricityDegrees)
{
    const float foveaDegrees = max(u_PC.foveatedGazeAndRings.z, 0.0);
    const float midDegrees = max(u_PC.foveatedGazeAndRings.w, foveaDegrees);
    const vec3 floors = clamp(u_PC.foveatedCoverageGuardParams.yzw, vec3(0.0), vec3(1.0));

    if (eccentricityDegrees <= foveaDegrees)
        return floors.x;
    if (eccentricityDegrees <= midDegrees)
        return floors.y;
    return floors.z;
}

uint foveatedCoverageSector(const float eccentricityDegrees)
{
    const float foveaDegrees = max(u_PC.foveatedGazeAndRings.z, 0.0);
    const float midDegrees = max(u_PC.foveatedGazeAndRings.w, foveaDegrees);
    if (eccentricityDegrees <= foveaDegrees)
        return 0u;
    if (eccentricityDegrees <= midDegrees)
        return 1u;
    return 2u;
}

bool foveatedCoverageGuardRiskActive(const float eccentricityDegrees)
{
    const uint riskSectors = floatBitsToUint(u_PC.foveatedTemporalParams.w) & 0x7u;
    const uint sector = foveatedCoverageSector(eccentricityDegrees);
    return (riskSectors & (1u << sector)) != 0u;
}

float foveatedClodLevelForEccentricity(const float eccentricityDegrees)
{
    const uint guardMode = u_PC.foveatedLayerParams.w;
    const float coverageProtectionDegrees = max(uintBitsToFloat(u_PC.foveatedLayerParams.z), 0.0);
    const float baseLevel = foveatedBaseClodLevelForEccentricity(eccentricityDegrees);

    if (guardMode == FOVEATED_COVERAGE_GUARD_OFF)
        return baseLevel;

    const float protectedLevel =
        coverageProtectionDegrees > 1e-4 ?
            foveatedBaseClodLevelForEccentricity(max(eccentricityDegrees - coverageProtectionDegrees, 0.0)) :
            baseLevel;

    if (guardMode != FOVEATED_COVERAGE_GUARD_LOCAL_BOUNDED)
    {
        if (guardMode == FOVEATED_COVERAGE_GUARD_RISK_TRIGGERED &&
            !foveatedCoverageGuardRiskActive(eccentricityDegrees))
        {
            return baseLevel;
        }
        if (guardMode == FOVEATED_COVERAGE_GUARD_GLOBAL)
            return max(baseLevel, protectedLevel);
    }

    if (guardMode != FOVEATED_COVERAGE_GUARD_LOCAL_BOUNDED &&
        guardMode != FOVEATED_COVERAGE_GUARD_RISK_TRIGGERED)
    {
        return max(baseLevel, protectedLevel);
    }

    const float localFloor = foveatedLocalCoverageFloor(eccentricityDegrees);
    if (baseLevel >= localFloor)
        return baseLevel;

    const float baseBudget = max(u_PC.foveatedLevelsAndTransition.x, 1e-5);
    const float guardBudgetRatio = clamp(u_PC.foveatedCoverageGuardParams.x, 0.0, 0.25);
    const float guardCapLevel = baseLevel + baseBudget * guardBudgetRatio;
    const float requestedLevel = max(baseLevel, max(localFloor, protectedLevel));
    return min(requestedLevel, guardCapLevel);
}

float foveatedRankRatio(const uint rank)
{
    return (float(rank) + 0.5) / float(max(u_PC.rankTotalCount, 1u));
}

bool foveatedTemporalHysteresisEnabled()
{
    return (u_PC.foveatedClodEnabled & FOVEATED_CLOD_TEMPORAL_HYSTERESIS_FLAG) != 0u;
}

bool foveatedBoundarySmoothingEnabled()
{
    return (u_PC.foveatedClodEnabled & FOVEATED_CLOD_BOUNDARY_SMOOTHING_FLAG) != 0u;
}

uint packFoveatedTemporalState(const uint phase, const uint residency)
{
    return (phase & FOVEATED_TEMPORAL_PHASE_MASK) |
           ((min(residency, FOVEATED_TEMPORAL_RESIDENCY_MASK) & FOVEATED_TEMPORAL_RESIDENCY_MASK) <<
            FOVEATED_TEMPORAL_RESIDENCY_SHIFT);
}

float foveatedBoundaryOpacityWeight(const float rankRatio, const float level, const float offLevel)
{
    if (!foveatedBoundarySmoothingEnabled())
        return 1.0;
    if (rankRatio <= level)
        return 1.0;

    const float smoothingRatio = max(u_PC.foveatedTemporalParams.y, 0.0);
    const float smoothingBand = max(level * smoothingRatio, 1e-4);
    const float edge1 = max(min(offLevel, level + smoothingBand), level + 1e-4);
    return clamp(1.0 - smoothstep(level, edge1, rankRatio), 0.0, 1.0);
}

float foveatedTemporalPeripheralFactor(const float eccentricityDegrees)
{
    const float foveaDegrees = max(u_PC.foveatedGazeAndRings.z, 0.0);
    const float midDegrees = max(u_PC.foveatedGazeAndRings.w, foveaDegrees + 1e-3);
    return clamp(smoothstep(foveaDegrees, midDegrees, eccentricityDegrees), 0.0, 1.0);
}

bool shaderAntiPopActive()
{
    return u_PC.shaderAntiPopParams.x != SHADER_ANTIPOP_OFF;
}

uint shaderAntiPopHashUint(uint value)
{
    value ^= u_PC.shaderAntiPopParams.y;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

float shaderAntiPopHash01(const uint sourceIndex)
{
    return float(shaderAntiPopHashUint(sourceIndex) & 0x00ffffffu) / 16777215.0;
}

void ecsptCounterAdd(const uint counterIndex, const uint value)
{
    if (value == 0u)
        return;
    atomicAdd(s_GeneralGaussianSplatEcsptCounters.counters[counterIndex], value);
}

uint ecsptScaledCounterValue(const float value)
{
    return uint(round(clamp(value, 0.0, 1024.0) * 1024.0));
}

uint ecsptEventBinCounterIndex(const uint bin, const uint eventType, const uint metric)
{
    return ECSPT_COUNTER_SCALAR_COUNT +
           ((min(bin, ECSPT_COUNTER_EVENT_BIN_COUNT - 1u) * ECSPT_COUNTER_EVENT_TYPE_COUNT +
             min(eventType, ECSPT_COUNTER_EVENT_TYPE_COUNT - 1u)) *
                ECSPT_COUNTER_EVENT_METRIC_COUNT +
            min(metric, ECSPT_COUNTER_EVENT_METRIC_COUNT - 1u));
}

void ecsptEventBinAdd(const uint eventType,
                      const uint bin,
                      const float opacity,
                      const float contributionProxy,
                      const float pDelta,
                      const float pOld,
                      const float pNew)
{
    ecsptCounterAdd(ecsptEventBinCounterIndex(bin, eventType, ECSPT_METRIC_COUNT), 1u);
    ecsptCounterAdd(ecsptEventBinCounterIndex(bin, eventType, ECSPT_METRIC_SUM_OPACITY),
                    ecsptScaledCounterValue(opacity));
    ecsptCounterAdd(ecsptEventBinCounterIndex(bin, eventType, ECSPT_METRIC_SUM_PROJECTED_AREA_PROXY), 0u);
    ecsptCounterAdd(ecsptEventBinCounterIndex(bin, eventType, ECSPT_METRIC_SUM_CONTRIBUTION_PROXY),
                    ecsptScaledCounterValue(contributionProxy));
    ecsptCounterAdd(ecsptEventBinCounterIndex(bin, eventType, ECSPT_METRIC_SUM_ABS_DELTA_P),
                    ecsptScaledCounterValue(pDelta));
    ecsptCounterAdd(ecsptEventBinCounterIndex(bin, eventType, ECSPT_METRIC_SUM_P_OLD),
                    ecsptScaledCounterValue(pOld));
    ecsptCounterAdd(ecsptEventBinCounterIndex(bin, eventType, ECSPT_METRIC_SUM_P_NEW),
                    ecsptScaledCounterValue(pNew));
}

float shaderAntiPopHash01Salted(const uint sourceIndex, const uint salt)
{
    return float(shaderAntiPopHashUint(sourceIndex ^ salt) & 0x00ffffffu) / 16777215.0;
}

float shaderAntiPopNormalizedLogistic(const float x, const float slope)
{
    const float lo = 1.0 / (1.0 + exp(0.5 * slope));
    const float hi = 1.0 / (1.0 + exp(-0.5 * slope));
    const float y = 1.0 / (1.0 + exp(-slope * (clamp(x, 0.0, 1.0) - 0.5)));
    return clamp((y - lo) / max(hi - lo, 1e-5), 0.0, 1.0);
}

float shaderAntiPopCurveKeepProbability(const float keepProbability)
{
    const float p = clamp(keepProbability, 0.0, 1.0);
    const uint curve = min(u_PC.shaderAntiPopParams.z, SHADER_ANTIPOP_PKEEP_LOGISTIC_STEEP);
    if (curve == SHADER_ANTIPOP_PKEEP_SMOOTH_WIDE)
    {
        const float centered = clamp((p - 0.5) * 2.0, -1.0, 1.0);
        return clamp(0.5 + 0.5 * sign(centered) * pow(abs(centered), 1.6), 0.0, 1.0);
    }
    if (curve == SHADER_ANTIPOP_PKEEP_LOGISTIC_SOFT)
        return shaderAntiPopNormalizedLogistic(p, 4.0);
    if (curve == SHADER_ANTIPOP_PKEEP_LOGISTIC_STEEP)
        return shaderAntiPopNormalizedLogistic(p, 10.0);
    return p;
}

float shaderAntiPopRampWeight(const uint sourceIndex,
                              const uint rank,
                              const float keepProbability)
{
    const uint mode = u_PC.shaderAntiPopParams.x;
    const uint curve = min(u_PC.shaderAntiPopParams.z, SHADER_ANTIPOP_PKEEP_LOGISTIC_STEEP);
    const uint normalizeMode = min(u_PC.shaderAntiPopParams.w, SHADER_ANTIPOP_NORMALIZE_ALPHA_MASS);
    const float pKeep = shaderAntiPopCurveKeepProbability(keepProbability);
    const float width = max(u_PC.shaderAntiPopFloats.x, 1e-5);
    const float normalizeFactor =
        normalizeMode == SHADER_ANTIPOP_NORMALIZE_OFF ? 1.0 : max(u_PC.shaderAntiPopFloats.w, 0.0);

    if (mode == SHADER_ANTIPOP_LINEAR_DEFAULT)
        return clamp(pKeep * normalizeFactor, 0.0, 1.0);

    float weight = 1.0;
    if (mode == SHADER_ANTIPOP_SOFT_RAMP)
    {
        if (curve == SHADER_ANTIPOP_PKEEP_LINEAR)
            return clamp(pKeep * normalizeFactor, 0.0, 1.0);
        const float rankRatio = foveatedRankRatio(rank);
        weight = 1.0 - smoothstep(pKeep - width, pKeep + width, rankRatio);
        return clamp(weight * normalizeFactor, 0.0, 1.0);
    }

    const float threshold = shaderAntiPopHash01(sourceIndex);
    weight = smoothstep(threshold - width, threshold + width, pKeep);
    if (mode == SHADER_ANTIPOP_HASH_RAMP_GUARDED)
    {
        const float guardThreshold = clamp(u_PC.shaderAntiPopFloats.y, 0.0, 1.0);
        if (guardThreshold > 0.0)
        {
            const float rankImportance = 1.0 - foveatedRankRatio(rank);
            if (rankImportance >= guardThreshold)
                weight = max(weight, clamp(u_PC.shaderAntiPopFloats.z, 0.0, 1.0));
        }
    }
    return clamp(weight * normalizeFactor, 0.0, 1.0);
}

float shaderGazeAnchorContributionProxy(const float opacity,
                                        const float oldEccentricityDegrees,
                                        const float newEccentricityDegrees,
                                        const float fadePhase,
                                        const bool oldOnly)
{
    const float foveaDegrees = max(u_PC.foveatedGazeAndRings.z, 0.0);
    const float midDegrees = max(u_PC.foveatedGazeAndRings.w, foveaDegrees + 1e-3);
    const float eccentricity = min(oldEccentricityDegrees, newEccentricityDegrees);
    const float fovealWeight = 1.0 - clamp((eccentricity - foveaDegrees) / max(midDegrees - foveaDegrees, 1e-3), 0.0, 1.0);
    const float fadeRisk = oldOnly ? (1.0 - fadePhase) : fadePhase;
    return clamp(0.58 * clamp(opacity, 0.0, 1.0) + 0.27 * fovealWeight + 0.15 * fadeRisk, 0.0, 1.0);
}

float shaderGazeAnchorTransitionCapWeight(const uint sourceIndex,
                                          const float baseWeight,
                                          const float opacity,
                                          const float oldEccentricityDegrees,
                                          const float newEccentricityDegrees,
                                          const bool oldOnly,
                                          const bool immediateFoveal)
{
    if (baseWeight <= 0.0)
        return 0.0;
    const uint flags = u_PC.gazeAnchorParams.x;
    const float budgetRatio = u_PC.gazeAnchorFloats.y;
    if (budgetRatio <= 0.0 || budgetRatio >= 8.0 || immediateFoveal)
        return baseWeight;

    const float allowance = clamp(budgetRatio - 1.0, 0.0, 1.0);
    if (allowance >= 0.999)
        return baseWeight;

    const float transitionHash = shaderAntiPopHash01Salted(sourceIndex, 0x9e3779b9u);
    if ((flags & GAZE_ANCHOR_FLAG_CONTRIBUTION_GUARD) == 0u)
    {
        return transitionHash <= allowance ? baseWeight : 0.0;
    }

    const float fadePhase = smoothstep(0.0, 1.0, clamp(u_PC.gazeAnchorFloats.x, 0.0, 1.0));
    const float priority = shaderGazeAnchorContributionProxy(opacity,
                                                             oldEccentricityDegrees,
                                                             newEccentricityDegrees,
                                                             fadePhase,
                                                             oldOnly);
    if (priority >= 0.55)
        return baseWeight;

    const float priorityBoost = mix(0.35, 1.0, priority);
    return transitionHash <= allowance * priorityBoost ? baseWeight : 0.0;
}

float shaderGazeAnchorCrossfadeWeight(const uint sourceIndex,
                                      const uint rank,
                                      const vec2 centerNdc,
                                      const CameraData camera,
                                      const float opacity)
{
    const uint flags = u_PC.gazeAnchorParams.x;
    if ((flags & GAZE_ANCHOR_FLAG_ENABLED) == 0u)
        return 1.0;

    const vec2 oldGaze = u_PC.gazeAnchorGazes.xy;
    const vec2 newGaze = u_PC.gazeAnchorGazes.zw;
    const float oldEccentricity = computeFoveatedEccentricityDegreesForGaze(centerNdc, camera, oldGaze);
    const float newEccentricity = computeFoveatedEccentricityDegreesForGaze(centerNdc, camera, newGaze);
    const float oldKeepProbability =
        shaderAntiPopCurveKeepProbability(clamp(foveatedBaseClodLevelForEccentricity(oldEccentricity), 0.0, 1.0));
    const float newKeepProbability =
        shaderAntiPopCurveKeepProbability(clamp(foveatedBaseClodLevelForEccentricity(newEccentricity), 0.0, 1.0));
    const float threshold = shaderAntiPopHash01(sourceIndex);
    const bool oldKeep = threshold <= oldKeepProbability;
    const bool newKeep = threshold <= newKeepProbability;
    const float fadePhase = smoothstep(0.0, 1.0, clamp(u_PC.gazeAnchorFloats.x, 0.0, 1.0));

    if (oldKeep && newKeep)
        return 1.0;
    if (!oldKeep && !newKeep)
        return 0.0;

    const float foveaDegrees = max(u_PC.foveatedGazeAndRings.z, 0.0);
    const bool immediateFoveal =
        newKeep &&
        ((flags & GAZE_ANCHOR_FLAG_IMMEDIATE_FOVEA_FILL) != 0u) &&
        newEccentricity <= foveaDegrees;
    float weight = 0.0;
    if (!oldKeep && newKeep)
        weight = immediateFoveal ? 1.0 : fadePhase;
    else
        weight = 1.0 - fadePhase;

    return shaderGazeAnchorTransitionCapWeight(sourceIndex,
                                               weight,
                                               opacity,
                                               oldEccentricity,
                                               newEccentricity,
                                               oldKeep && !newKeep,
                                               immediateFoveal);
}

float shaderEccStochasticContributionProxy(const float opacity,
                                           const float oldEccentricityDegrees,
                                           const float newEccentricityDegrees,
                                           const float pDelta)
{
    const float foveaDegrees = max(u_PC.foveatedGazeAndRings.z, 0.0);
    const float midDegrees = max(u_PC.foveatedGazeAndRings.w, foveaDegrees + 1e-3);
    const float eccentricity = min(oldEccentricityDegrees, newEccentricityDegrees);
    const float fovealWeight =
        1.0 - clamp((eccentricity - foveaDegrees) / max(midDegrees - foveaDegrees, 1e-3), 0.0, 1.0);
    return clamp(0.62 * clamp(opacity, 0.0, 1.0) + 0.25 * fovealWeight + 0.13 * clamp(pDelta * 8.0, 0.0, 1.0),
                 0.0,
                 1.0);
}

bool shaderEccStochasticNearTransitionBoundary(const float oldEccentricityDegrees,
                                               const float newEccentricityDegrees,
                                               const float boundaryBandDegrees)
{
    if (boundaryBandDegrees <= 0.0)
        return false;
    const float foveaDegrees = max(u_PC.foveatedGazeAndRings.z, 0.0);
    const float midDegrees = max(u_PC.foveatedGazeAndRings.w, foveaDegrees);
    const float oldBoundaryDistance =
        min(abs(oldEccentricityDegrees - foveaDegrees), abs(oldEccentricityDegrees - midDegrees));
    const float newBoundaryDistance =
        min(abs(newEccentricityDegrees - foveaDegrees), abs(newEccentricityDegrees - midDegrees));
    return min(oldBoundaryDistance, newBoundaryDistance) <= boundaryBandDegrees;
}

EccStochasticEval shaderEccStochasticTransitionEval(const uint sourceIndex,
                                                    const vec2 centerNdc,
                                                    const CameraData camera,
                                                    const float opacity)
{
    EccStochasticEval eval;
    eval.weight = 1.0;
    eval.opacity = clamp(opacity, 0.0, 1.0);
    eval.contributionProxy = 0.0;
    eval.pOld = 0.0;
    eval.pNew = 0.0;
    eval.pDelta = 0.0;
    eval.oldEccentricityDegrees = 0.0;
    eval.newEccentricityDegrees = 0.0;
    eval.flags = 0u;

    const uint flags = u_PC.gazeAnchorParams.x;
    if ((flags & ECC_STOCHASTIC_FLAG_ENABLED) == 0u)
        return eval;

    const vec2 oldGaze = u_PC.gazeAnchorGazes.xy;
    const vec2 newGaze = u_PC.gazeAnchorGazes.zw;
    const float oldEccentricity = computeFoveatedEccentricityDegreesForGaze(centerNdc, camera, oldGaze);
    const float newEccentricity = computeFoveatedEccentricityDegreesForGaze(centerNdc, camera, newGaze);
    const float pOld =
        shaderAntiPopCurveKeepProbability(clamp(foveatedBaseClodLevelForEccentricity(oldEccentricity), 0.0, 1.0));
    const float pNew =
        shaderAntiPopCurveKeepProbability(clamp(foveatedBaseClodLevelForEccentricity(newEccentricity), 0.0, 1.0));
    const float threshold = shaderAntiPopHash01(sourceIndex);
    const bool oldKeep = threshold <= pOld;
    const bool newKeep = threshold <= pNew;
    const float pDelta = abs(pNew - pOld);
    const float contributionProxy =
        shaderEccStochasticContributionProxy(opacity, oldEccentricity, newEccentricity, pDelta);

    eval.pOld = pOld;
    eval.pNew = pNew;
    eval.pDelta = pDelta;
    eval.oldEccentricityDegrees = oldEccentricity;
    eval.newEccentricityDegrees = newEccentricity;
    eval.contributionProxy = contributionProxy;

    eval.flags |= ECSPT_EVAL_REACHED;
    if (oldKeep)
        eval.flags |= ECSPT_EVAL_OLD_SELECTED;
    if (newKeep)
        eval.flags |= ECSPT_EVAL_NEW_SELECTED;

    const float foveaDegrees = max(u_PC.foveatedGazeAndRings.z, 0.0);
    const bool immediateNewFovea = newKeep && newEccentricity <= foveaDegrees;
    if (immediateNewFovea)
        eval.flags |= ECSPT_EVAL_IMMEDIATE_NEW_FOVEA;

    if (newKeep)
    {
        eval.weight = 1.0;
        eval.flags |= oldKeep ? ECSPT_EVAL_SHARED : ECSPT_EVAL_UPGRADE;
        return eval;
    }
    if (!oldKeep)
    {
        eval.weight = 0.0;
        eval.flags |= ECSPT_EVAL_ZERO_WEIGHT;
        return eval;
    }

    eval.flags |= ECSPT_EVAL_DOWNGRADE;
    const float minPDelta = max(u_PC.shaderAntiPopFloats.y, 0.0);
    if (pDelta < minPDelta)
    {
        eval.weight = 0.0;
        eval.flags |= ECSPT_EVAL_DROPPED_DOWNGRADE |
                      ECSPT_EVAL_MIN_P_DELTA_DISCARD |
                      ECSPT_EVAL_ZERO_WEIGHT;
        return eval;
    }

    const float midDegrees = max(u_PC.foveatedGazeAndRings.w, foveaDegrees + 1e-3);
    const float protectOldFoveaDegrees = u_PC.gazeAnchorFloats.y >= 0.0 ? u_PC.gazeAnchorFloats.y : foveaDegrees;
    const float protectNewFoveaDegrees = u_PC.gazeAnchorFloats.z >= 0.0 ? u_PC.gazeAnchorFloats.z : foveaDegrees;
    const float boundaryBandDegrees = max(u_PC.gazeAnchorFloats.w, 0.0);
    const bool nearOldFovea =
        ((flags & ECC_STOCHASTIC_FLAG_PROTECT_OLD_FOVEA) != 0u) &&
        oldEccentricity <= protectOldFoveaDegrees;
    const bool nearNewFovea =
        ((flags & ECC_STOCHASTIC_FLAG_PROTECT_NEW_FOVEA) != 0u) &&
        newEccentricity <= protectNewFoveaDegrees;
    const bool nearBoundary =
        ((flags & ECC_STOCHASTIC_FLAG_PROTECT_BOUNDARY) != 0u) &&
        shaderEccStochasticNearTransitionBoundary(oldEccentricity, newEccentricity, boundaryBandDegrees);
    const bool farPeriphery = min(oldEccentricity, newEccentricity) > midDegrees + max(boundaryBandDegrees, 1.0);
    const bool contributionProtected =
        ((flags & ECC_STOCHASTIC_FLAG_CONTRIBUTION_GUARD) != 0u) &&
        !farPeriphery &&
        contributionProxy >= 0.55;

    if (nearBoundary)
        eval.flags |= ECSPT_EVAL_BOUNDARY_PROTECTED;
    if (contributionProtected)
        eval.flags |= ECSPT_EVAL_CONTRIBUTION_GUARD_PROTECTED;
    if (farPeriphery)
        eval.flags |= ECSPT_EVAL_FAR_PERIPHERY_HARD_DROP;

    if (!(nearOldFovea || nearNewFovea || nearBoundary || contributionProtected))
    {
        eval.weight = 0.0;
        eval.flags |= ECSPT_EVAL_DROPPED_DOWNGRADE | ECSPT_EVAL_ZERO_WEIGHT;
        return eval;
    }

    const float fadePhase = smoothstep(0.0, 1.0, clamp(u_PC.gazeAnchorFloats.x, 0.0, 1.0));
    eval.weight = clamp(1.0 - fadePhase, 0.0, 1.0);
    eval.flags |= ECSPT_EVAL_PROTECTED_DOWNGRADE;
    if (eval.weight <= 0.0)
        eval.flags |= ECSPT_EVAL_ZERO_WEIGHT;
    return eval;
}

EccStochasticEval shaderStableOpticalDepthThinningEval(const uint sourceIndex,
                                                       const vec2 centerNdc,
                                                       const CameraData camera,
                                                       const float opacity)
{
    EccStochasticEval eval;
    eval.weight = 1.0;
    eval.opacity = clamp(opacity, 0.0, 1.0);
    eval.contributionProxy = 0.0;
    eval.pOld = 0.0;
    eval.pNew = 0.0;
    eval.pDelta = 0.0;
    eval.oldEccentricityDegrees = 0.0;
    eval.newEccentricityDegrees = 0.0;
    eval.flags = ECSPT_EVAL_REACHED;

    const float eccentricity = computeFoveatedEccentricityDegrees(centerNdc, camera);
    const float pKeep =
        shaderAntiPopCurveKeepProbability(clamp(foveatedBaseClodLevelForEccentricity(eccentricity), 0.0, 1.0));
    const float pSafe = max(pKeep, clamp(u_PC.shaderAntiPopFloats.x, 1e-4, 1.0));
    const float alphaMax = clamp(u_PC.shaderAntiPopFloats.y, MIN_VISIBLE_OPACITY, 0.999);
    const float threshold = shaderAntiPopHash01(sourceIndex);

    eval.pOld = pKeep;
    eval.pNew = pKeep;
    eval.oldEccentricityDegrees = eccentricity;
    eval.newEccentricityDegrees = eccentricity;
    eval.contributionProxy = clamp(eval.opacity / max(pSafe, 1e-4), 0.0, 1.0);

    if (threshold > pKeep)
    {
        eval.weight = 0.0;
        eval.flags |= ECSPT_EVAL_ZERO_WEIGHT;
        return eval;
    }

    eval.flags |= ECSPT_EVAL_NEW_SELECTED | ECSPT_EVAL_SHARED;

    const bool exactTau = u_PC.shaderAntiPopFloats.z >= 0.5;
    float compensatedAlpha = eval.opacity / pSafe;
    if (exactTau)
    {
        const float tau = -log(max(1.0 - eval.opacity, 1e-6));
        compensatedAlpha = 1.0 - exp(-tau / pSafe);
    }
    const float clampedAlpha = min(compensatedAlpha, alphaMax);
    if (compensatedAlpha > alphaMax + 1e-6)
        eval.flags |= ECSPT_EVAL_STABLE_OPTICAL_DEPTH_ALPHA_CLAMPED;
    eval.opacity = clampedAlpha;
    eval.weight = eval.opacity / max(opacity, 1e-6);
    return eval;
}

bool coverageReleaseAccumulationPhase()
{
    return (u_PC.coverageReleaseParams.w & COVERAGE_RELEASE_PHASE_ACCUMULATE) != 0u;
}

bool coverageReleaseResolvePhase()
{
    return (u_PC.coverageReleaseParams.w & COVERAGE_RELEASE_PHASE_RESOLVE) != 0u;
}

bool coverageReleaseDiagnosticsEnabled()
{
    return (u_PC.coverageReleaseParams.x & COVERAGE_RELEASE_FLAG_DIAGNOSTICS) != 0u;
}

bool coverageReleaseSaturatedFloorEnabled()
{
    return (u_PC.coverageReleaseParams.x & COVERAGE_RELEASE_FLAG_SATURATED_FLOOR) != 0u;
}

bool coverageTextureFloorEnabled()
{
    return (u_PC.coverageReleaseParams.x & COVERAGE_RELEASE_FLAG_TEXTURE_FLOOR) != 0u;
}

bool coverageTextureUpdateThisFrame()
{
    return (u_PC.coverageReleaseParams.x & COVERAGE_RELEASE_FLAG_TEXTURE_UPDATE_THIS_FRAME) != 0u;
}

bool coverageGuideModifierEnabled()
{
    return (u_PC.coverageReleaseParams.x & COVERAGE_RELEASE_FLAG_GUIDE_MODIFIER) != 0u;
}

bool coverageGuideHistoryValid()
{
    return (u_PC.coverageReleaseParams.x & COVERAGE_RELEASE_FLAG_GUIDE_HISTORY_VALID) != 0u;
}

uint coverageReleasePolicy()
{
    return min(u_PC.coverageReleaseCostParams.x, COVERAGE_RELEASE_POLICY_TILE_LOCAL_CAPPED);
}

float coverageReleaseCapRatio()
{
    return clamp(u_PC.coverageReleaseCostFloats.x, 0.0, 1.0);
}

float coverageReleaseRiskThreshold()
{
    return clamp(u_PC.coverageReleaseCostFloats.y, 0.0, 1.0);
}

float coverageReleaseFootprintDecayScale()
{
    return clamp(u_PC.coverageReleaseCostFloats.z, 0.0, 1.0);
}

float coverageReleaseLargeFootprintPx()
{
    return max(u_PC.coverageReleaseCostFloats.w, 1.0);
}

bool ecsptCounterDiagnosticsEnabled()
{
    if (u_PC.shaderAntiPopParams.x == SHADER_ANTIPOP_COVERAGE_STABLE_LOGPOLAR_RELEASE)
        return coverageReleaseDiagnosticsEnabled();
    return true;
}

uint coverageReleaseFrameIndex16()
{
    return u_PC.coverageReleaseParams.w & 0xffffu;
}

uint coverageReleaseTileGridX()
{
    return clamp(u_PC.coverageReleaseParams.y, 1u, 32u);
}

uint coverageReleaseTileGridY()
{
    return clamp(u_PC.coverageReleaseParams.z, 1u, 16u);
}

uint coverageReleaseTileCount()
{
    return min(coverageReleaseTileGridX() * coverageReleaseTileGridY(), COVERAGE_RELEASE_MAX_TILE_COUNT);
}

uint coverageReleaseTileIndexFromNdc(const vec2 centerNdc)
{
    const vec2 uv = clamp(centerNdc * 0.5 + vec2(0.5), vec2(0.0), vec2(0.999999));
    const uint gridX = coverageReleaseTileGridX();
    const uint gridY = coverageReleaseTileGridY();
    const uint x = min(uint(floor(uv.x * float(gridX))), gridX - 1u);
    const uint y = min(uint(floor(uv.y * float(gridY))), gridY - 1u);
    return min(y * gridX + x, coverageReleaseTileCount() - 1u);
}

uint coverageReleaseAlphaCounterIndex(const uint tileIndex)
{
    return COVERAGE_RELEASE_TILE_COUNTER_OFFSET + tileIndex;
}

uint coverageReleaseAlpha2CounterIndex(const uint tileIndex)
{
    return COVERAGE_RELEASE_TILE_COUNTER_OFFSET + COVERAGE_RELEASE_MAX_TILE_COUNT + tileIndex;
}

uint coverageReleaseSaturatedCounterIndex(const uint tileIndex)
{
    return COVERAGE_RELEASE_TILE_SATURATED_OFFSET + tileIndex;
}

float coverageReleaseCounterToFloat(const uint counterValue)
{
    return float(counterValue) / COVERAGE_RELEASE_TILE_COUNTER_SCALE;
}

uint coverageReleaseFloatToCounter(const float value)
{
    uint scaled = uint(round(max(value, 0.0) * COVERAGE_RELEASE_TILE_COUNTER_SCALE));
    if (value > 0.0)
        scaled = max(scaled, 1u);
    return scaled;
}

void coverageReleaseCounterAddFloat(const uint counterIndex, const float value)
{
    uint scaled = coverageReleaseFloatToCounter(value);
    atomicAdd(s_GeneralGaussianSplatEcsptCounters.counters[counterIndex],
              scaled);
}

uint releaseCostBaseDuplicateCounterIndex(const uint tileIndex)
{
    return RELEASE_COST_TILE_BASE_DUPLICATE_OFFSET + tileIndex;
}

uint releaseCostReleasedDuplicateCounterIndex(const uint tileIndex)
{
    return RELEASE_COST_TILE_RELEASED_DUPLICATE_OFFSET + tileIndex;
}

uint releaseCostRadiusBin(const float radiusPx)
{
    if (radiusPx < 4.0)
        return 0u;
    if (radiusPx < 8.0)
        return 1u;
    if (radiusPx < 16.0)
        return 2u;
    if (radiusPx < 32.0)
        return 3u;
    if (radiusPx < 48.0)
        return 4u;
    if (radiusPx < 64.0)
        return 5u;
    if (radiusPx < 96.0)
        return 6u;
    return 7u;
}

uint releaseCostScaledRadius(const float radiusPx)
{
    return uint(round(clamp(radiusPx, 0.0, 1024.0) * RELEASE_COST_RADIUS_COUNTER_SCALE));
}

uint estimateProjectedTileDuplicateProxy(const vec2 centerNdc,
                                         const vec2 v1,
                                         const vec2 v2,
                                         const vec2 viewport)
{
    const vec2 safeViewport = max(viewport, vec2(1.0));
    const vec2 centerPx = (centerNdc * 0.5 + vec2(0.5)) * safeViewport;
    const vec2 halfExtentPx = abs(v1) + abs(v2);
    const float maxX = max(safeViewport.x - 0.001, 0.0);
    const float maxY = max(safeViewport.y - 0.001, 0.0);
    const uint tileMaxX =
        max(uint(ceil(safeViewport.x / RELEASE_COST_PROJECTED_TILE_SIZE_PX)), 1u) - 1u;
    const uint tileMaxY =
        max(uint(ceil(safeViewport.y / RELEASE_COST_PROJECTED_TILE_SIZE_PX)), 1u) - 1u;
    const uint tileX0 =
        min(uint(floor(clamp(centerPx.x - halfExtentPx.x, 0.0, maxX) /
                       RELEASE_COST_PROJECTED_TILE_SIZE_PX)),
            tileMaxX);
    const uint tileX1 =
        min(uint(floor(clamp(centerPx.x + halfExtentPx.x, 0.0, maxX) /
                       RELEASE_COST_PROJECTED_TILE_SIZE_PX)),
            tileMaxX);
    const uint tileY0 =
        min(uint(floor(clamp(centerPx.y - halfExtentPx.y, 0.0, maxY) /
                       RELEASE_COST_PROJECTED_TILE_SIZE_PX)),
            tileMaxY);
    const uint tileY1 =
        min(uint(floor(clamp(centerPx.y + halfExtentPx.y, 0.0, maxY) /
                       RELEASE_COST_PROJECTED_TILE_SIZE_PX)),
            tileMaxY);
    return max(tileX1 - tileX0 + 1u, 1u) * max(tileY1 - tileY0 + 1u, 1u);
}

void recordReleaseCostDiagnostics(const vec2 centerNdc,
                                  const bool released,
                                  const uint tileDuplicateProxy,
                                  const float radiusPx)
{
    const uint tileIndex = coverageReleaseTileIndexFromNdc(centerNdc);
    const uint duplicateProxy = max(tileDuplicateProxy, 1u);
    if (released)
    {
        atomicAdd(s_GeneralGaussianSplatEcsptCounters.counters[RELEASE_COST_RELEASED_DUPLICATE_COUNTER],
                  duplicateProxy);
        atomicAdd(s_GeneralGaussianSplatEcsptCounters.counters[releaseCostReleasedDuplicateCounterIndex(tileIndex)],
                  duplicateProxy);
        const bool largeFootprint = radiusPx >= coverageReleaseLargeFootprintPx();
        if (largeFootprint)
        {
            atomicAdd(s_GeneralGaussianSplatEcsptCounters.counters[RELEASE_COST_LARGE_RELEASED_COUNTER],
                      1u);
            atomicAdd(
                s_GeneralGaussianSplatEcsptCounters.counters[RELEASE_COST_LARGE_RELEASED_DUPLICATE_COUNTER],
                duplicateProxy);
        }
        const uint scaledRadius = releaseCostScaledRadius(radiusPx);
        atomicAdd(s_GeneralGaussianSplatEcsptCounters.counters[RELEASE_COST_RELEASED_RADIUS_SUM_COUNTER],
                  scaledRadius);
        atomicMax(s_GeneralGaussianSplatEcsptCounters.counters[RELEASE_COST_RELEASED_RADIUS_MAX_COUNTER],
                  scaledRadius);
        const uint radiusBin = releaseCostRadiusBin(radiusPx);
        atomicAdd(s_GeneralGaussianSplatEcsptCounters.counters[
                      RELEASE_COST_RELEASED_RADIUS_HIST_OFFSET + radiusBin],
                  1u);
    }
    else
    {
        atomicAdd(s_GeneralGaussianSplatEcsptCounters.counters[RELEASE_COST_BASE_DUPLICATE_COUNTER],
                  duplicateProxy);
        atomicAdd(s_GeneralGaussianSplatEcsptCounters.counters[releaseCostBaseDuplicateCounterIndex(tileIndex)],
                  duplicateProxy);
    }
}

float coverageReleaseMinimumKeepProbability()
{
    return max(
        shaderAntiPopCurveKeepProbability(clamp(u_PC.foveatedLevelsAndTransition.z, 0.0, 1.0)),
        1e-4);
}

float coverageReleaseDAlphaSafeThreshold()
{
    const float rMin = coverageReleaseMinimumKeepProbability();
    const float dMin = max(u_PC.coverageReleaseFloats.y, 0.0);
    return dMin > 0.0 ? dMin / rMin : 0.0;
}

float coverageReleaseNEffSafeThreshold()
{
    const float rMin = coverageReleaseMinimumKeepProbability();
    const float sigmaMax = max(u_PC.coverageReleaseFloats.z, 0.0);
    return sigmaMax > 1e-6 ? (1.0 / rMin - 1.0) / (sigmaMax * sigmaMax) : 0.0;
}

bool coverageReleaseAlphaMassProvablySafe(const float dAlpha)
{
    const float threshold = max(coverageReleaseDAlphaSafeThreshold(),
                                coverageReleaseNEffSafeThreshold());
    return dAlpha >= threshold;
}

bool coverageReleaseProjectCandidate(const GeneralGaussianSplatPackedSource src,
                                     const GeneralGaussianSplatDrawRecord draw,
                                     const vec3 worldPos,
                                     const float lodWeight,
                                     const CameraData camera,
                                     out vec2 centerNdc,
                                     out float opacity)
{
    centerNdc = vec2(2.0);
    opacity = 0.0;

    vec4 colorOpacity = decodeGeneralGaussianSplatBaseColorOpacity(src);
    colorOpacity.a *= max(draw.params0.z, 0.0);
    colorOpacity.a *= lodWeight;
    if (colorOpacity.a < MIN_VISIBLE_OPACITY)
        return false;

    const vec4 posView = camera.view * vec4(worldPos, 1.0);
    const vec4 posClip = camera.projection * posView;
    if (posClip.w <= 1e-5)
        return false;

    const vec3 ndc = posClip.xyz / posClip.w;
    const float bounds = 1.2 * posClip.w;
    if (ndc.z <= 0.0 || ndc.z >= 1.0)
        return false;
    if (posClip.x < -bounds || posClip.x > bounds || posClip.y < -bounds || posClip.y > bounds)
        return false;

    centerNdc = ndc.xy;
    opacity = clamp(colorOpacity.a, 0.0, 1.0);
    return true;
}

void coverageReleaseAccumulateCandidate(const GeneralGaussianSplatPackedSource src,
                                        const GeneralGaussianSplatDrawRecord draw,
                                        const vec3 worldPos,
                                        const float lodWeight,
                                        const CameraData camera)
{
    vec2 centerNdc;
    float opacity;
    if (!coverageReleaseProjectCandidate(src, draw, worldPos, lodWeight, camera, centerNdc, opacity))
        return;

    const uint tileIndex = coverageReleaseTileIndexFromNdc(centerNdc);
    const bool saturatedFloor = coverageReleaseSaturatedFloorEnabled();
    const uint saturatedCounterIndex = coverageReleaseSaturatedCounterIndex(tileIndex);
    if (saturatedFloor &&
        s_GeneralGaussianSplatEcsptCounters.counters[saturatedCounterIndex] != 0u)
    {
        if (coverageReleaseDiagnosticsEnabled())
            ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_SKIPPED_FLOOR_UPDATE_COUNT, 1u);
        return;
    }

    const uint alphaScaled = coverageReleaseFloatToCounter(opacity);
    const uint alphaCounterIndex = coverageReleaseAlphaCounterIndex(tileIndex);
    const uint oldAlphaScaled =
        atomicAdd(s_GeneralGaussianSplatEcsptCounters.counters[alphaCounterIndex], alphaScaled);
    const float dAlphaAfterAdd = coverageReleaseCounterToFloat(oldAlphaScaled + alphaScaled);

    if (saturatedFloor && coverageReleaseAlphaMassProvablySafe(dAlphaAfterAdd))
    {
        atomicOr(s_GeneralGaussianSplatEcsptCounters.counters[saturatedCounterIndex], 1u);
        if (coverageReleaseDiagnosticsEnabled())
            ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_SKIPPED_FLOOR_UPDATE_COUNT, 1u);
        return;
    }

    coverageReleaseCounterAddFloat(coverageReleaseAlpha2CounterIndex(tileIndex), opacity * opacity);
}

float coverageReleaseTileFloor(const vec2 centerNdc, out float dAlpha, out float nEff)
{
    dAlpha = 0.0;
    nEff = 0.0;
    if ((u_PC.coverageReleaseParams.x & COVERAGE_RELEASE_FLAG_COVERAGE_FLOOR) == 0u)
        return 0.0;

    const uint tileIndex = coverageReleaseTileIndexFromNdc(centerNdc);
    const float alphaSum =
        coverageReleaseCounterToFloat(s_GeneralGaussianSplatEcsptCounters.counters[
            coverageReleaseAlphaCounterIndex(tileIndex)]);
    if (coverageReleaseSaturatedFloorEnabled() &&
        s_GeneralGaussianSplatEcsptCounters.counters[coverageReleaseSaturatedCounterIndex(tileIndex)] != 0u)
    {
        dAlpha = max(alphaSum, 0.0);
        nEff = coverageReleaseNEffSafeThreshold();
        return 0.0;
    }
    const float alpha2Sum =
        coverageReleaseCounterToFloat(s_GeneralGaussianSplatEcsptCounters.counters[
            coverageReleaseAlpha2CounterIndex(tileIndex)]);
    dAlpha = max(alphaSum, 0.0);
    if (dAlpha <= 1e-6)
        return 0.0;
    nEff = alpha2Sum > 1e-6 ? max(1.0, (alphaSum * alphaSum) / alpha2Sum) : 1.0;

    const float dMin = max(u_PC.coverageReleaseFloats.y, 0.0);
    const float sigmaMax = max(u_PC.coverageReleaseFloats.z, 0.0);
    const float pMean = dMin > 0.0 ? min(1.0, dMin / dAlpha) : 0.0;
    const float pVar = 1.0 / (1.0 + sigmaMax * sigmaMax * max(nEff, 0.0));
    return clamp(max(pMean, pVar), 0.0, 1.0);
}

uint coverageTextureWidth()
{
    return clamp(u_PC.coverageReleaseParams.y, 1u, 1024u);
}

uint coverageTextureHeight()
{
    return clamp(u_PC.coverageReleaseParams.z, 1u, 1024u);
}

uint coverageTextureTexelCount()
{
    return max(1u, coverageTextureWidth() * coverageTextureHeight());
}

uint coverageTextureIndexFromCoord(const uint x, const uint y)
{
    return min(y * coverageTextureWidth() + x, coverageTextureTexelCount() - 1u);
}

uint coverageTextureIndexFromNdc(const vec2 centerNdc)
{
    const vec2 uv = clamp(centerNdc * 0.5 + vec2(0.5), vec2(0.0), vec2(0.999999));
    const uint width = coverageTextureWidth();
    const uint height = coverageTextureHeight();
    const uint x = min(uint(floor(uv.x * float(width))), width - 1u);
    const uint y = min(uint(floor(uv.y * float(height))), height - 1u);
    return coverageTextureIndexFromCoord(x, y);
}

uint coverageTextureHistoryAlphaIndex(const uint texel)
{
    return texel;
}

uint coverageTextureHistoryAlpha2Index(const uint texel)
{
    return coverageTextureTexelCount() + texel;
}

uint coverageTextureCurrentAlphaIndex(const uint texel)
{
    return coverageTextureTexelCount() * 2u + texel;
}

uint coverageTextureCurrentAlpha2Index(const uint texel)
{
    return coverageTextureTexelCount() * 3u + texel;
}

float coverageTextureCounterToFloat(const uint value)
{
    return float(value) / COVERAGE_RELEASE_TILE_COUNTER_SCALE;
}

uint coverageTextureFloatToCounter(const float value)
{
    uint scaled = uint(round(max(value, 0.0) * COVERAGE_RELEASE_TILE_COUNTER_SCALE));
    if (value > 0.0)
        scaled = max(scaled, 1u);
    return scaled;
}

void coverageTextureCounterAddFloat(const uint index, const float value)
{
    atomicAdd(s_GeneralGaussianSplatCoverageTexture.words[index],
              coverageTextureFloatToCounter(value));
}

void coverageTextureAccumulateProjected(const vec2 centerNdc, const float opacity)
{
    const uint texel = coverageTextureIndexFromNdc(centerNdc);
    coverageTextureCounterAddFloat(coverageTextureCurrentAlphaIndex(texel), opacity);
    coverageTextureCounterAddFloat(coverageTextureCurrentAlpha2Index(texel), opacity * opacity);
}

float coverageTextureSampleHistoryChannel(const vec2 centerNdc, const uint channel)
{
    const uint width = coverageTextureWidth();
    const uint height = coverageTextureHeight();
    const uint texelCount = coverageTextureTexelCount();
    const vec2 uv = clamp(centerNdc * 0.5 + vec2(0.5), vec2(0.0), vec2(1.0));
    const vec2 coord = uv * vec2(float(width - 1u), float(height - 1u));
    const uint x0 = min(uint(floor(coord.x)), width - 1u);
    const uint y0 = min(uint(floor(coord.y)), height - 1u);
    const uint x1 = min(x0 + 1u, width - 1u);
    const uint y1 = min(y0 + 1u, height - 1u);
    const float tx = coord.x - float(x0);
    const float ty = coord.y - float(y0);
    const uint offset = channel * texelCount;
    const float v00 = coverageTextureCounterToFloat(
        s_GeneralGaussianSplatCoverageTexture.words[offset + coverageTextureIndexFromCoord(x0, y0)]);
    const float v10 = coverageTextureCounterToFloat(
        s_GeneralGaussianSplatCoverageTexture.words[offset + coverageTextureIndexFromCoord(x1, y0)]);
    const float v01 = coverageTextureCounterToFloat(
        s_GeneralGaussianSplatCoverageTexture.words[offset + coverageTextureIndexFromCoord(x0, y1)]);
    const float v11 = coverageTextureCounterToFloat(
        s_GeneralGaussianSplatCoverageTexture.words[offset + coverageTextureIndexFromCoord(x1, y1)]);
    return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

float coverageTextureFloor(const vec2 centerNdc, out float dAlpha, out float nEff)
{
    dAlpha = 0.0;
    nEff = 0.0;
    if (!coverageTextureFloorEnabled())
        return 0.0;

    const float alphaSum = coverageTextureSampleHistoryChannel(centerNdc, 0u);
    const float alpha2Sum = coverageTextureSampleHistoryChannel(centerNdc, 1u);
    dAlpha = max(alphaSum, 0.0);
    if (dAlpha <= 1e-6)
        return 0.0;
    nEff = alpha2Sum > 1e-6 ? max(1.0, (alphaSum * alphaSum) / alpha2Sum) : 1.0;

    const float dMin = max(u_PC.coverageReleaseFloats.y, 0.0);
    const float sigmaMax = max(u_PC.coverageReleaseFloats.z, 0.0);
    const float strength = clamp(u_PC.coverageTextureFloats.x, 0.0, 1.0);
    const float pMean = dMin > 0.0 ? min(1.0, dMin / dAlpha) : 0.0;
    const float pVar = 1.0 / (1.0 + sigmaMax * sigmaMax * max(nEff, 0.0));
    return strength * clamp(max(pMean, pVar), 0.0, 1.0);
}

void coverageTextureAccumulateCandidate(const GeneralGaussianSplatPackedSource src,
                                        const GeneralGaussianSplatDrawRecord draw,
                                        const vec3 worldPos,
                                        const float lodWeight,
                                        const CameraData camera)
{
    vec2 centerNdc;
    float opacity;
    if (!coverageReleaseProjectCandidate(src, draw, worldPos, lodWeight, camera, centerNdc, opacity))
        return;

    coverageTextureAccumulateProjected(centerNdc, opacity);
}

void coverageTextureResolveTexel(const uint texel)
{
    const uint texelCount = coverageTextureTexelCount();
    if (texel >= texelCount)
        return;

    const uint currentAlphaIndex = coverageTextureCurrentAlphaIndex(texel);
    const uint currentAlpha2Index = coverageTextureCurrentAlpha2Index(texel);
    const uint historyAlphaIndex = coverageTextureHistoryAlphaIndex(texel);
    const uint historyAlpha2Index = coverageTextureHistoryAlpha2Index(texel);
    const float currentAlpha =
        coverageTextureCounterToFloat(s_GeneralGaussianSplatCoverageTexture.words[currentAlphaIndex]);
    const float currentAlpha2 =
        coverageTextureCounterToFloat(s_GeneralGaussianSplatCoverageTexture.words[currentAlpha2Index]);
    const float historyAlpha =
        coverageTextureCounterToFloat(s_GeneralGaussianSplatCoverageTexture.words[historyAlphaIndex]);
    const float historyAlpha2 =
        coverageTextureCounterToFloat(s_GeneralGaussianSplatCoverageTexture.words[historyAlpha2Index]);
    const float beta = clamp(u_PC.coverageTextureFloats.y, 0.0, 1.0);
    s_GeneralGaussianSplatCoverageTexture.words[historyAlphaIndex] =
        coverageTextureFloatToCounter(mix(currentAlpha, historyAlpha, beta));
    s_GeneralGaussianSplatCoverageTexture.words[historyAlpha2Index] =
        coverageTextureFloatToCounter(mix(currentAlpha2, historyAlpha2, beta));

    if (coverageReleaseDiagnosticsEnabled())
    {
        if (texel == 0u)
        {
            atomicMax(s_GeneralGaussianSplatEcsptCounters.counters[ECSPT_COUNTER_COVERAGE_TEXTURE_WIDTH],
                      coverageTextureWidth());
            atomicMax(s_GeneralGaussianSplatEcsptCounters.counters[ECSPT_COUNTER_COVERAGE_TEXTURE_HEIGHT],
                      coverageTextureHeight());
        }
        if (currentAlpha > 1e-6)
        {
            ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_TEXTURE_NONZERO_TEXEL_COUNT, 1u);
            ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_TEXTURE_ALPHA_SUM,
                            uint(round(clamp(currentAlpha, 0.0, 4096.0) * COVERAGE_RELEASE_COUNTER_SCALE)));
            ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_TEXTURE_ALPHA2_SUM,
                            uint(round(clamp(currentAlpha2, 0.0, 4096.0) * COVERAGE_RELEASE_COUNTER_SCALE)));
        }
    }
}

uint coverageReleasePackHistory(const float p, const uint frameIndex16)
{
    const uint q = uint(round(clamp(p, 0.0, 1.0) * 65535.0));
    return ((frameIndex16 & 0xffffu) << 16u) | (q & 0xffffu);
}

float coverageReleaseUnpackHistoryP(const uint state)
{
    return float(state & 0xffffu) / 65535.0;
}

uint coverageReleaseUnpackHistoryFrame(const uint state)
{
    return (state >> 16u) & 0xffffu;
}

EccStochasticEval shaderCoverageStableLogpolarReleaseEval(const uint sourceIndex,
                                                          const uint persistentId,
                                                          const vec2 centerNdc,
                                                          const CameraData camera,
                                                          const float opacity,
                                                          const float footprintProxyPx)
{
    EccStochasticEval eval;
    eval.weight = 1.0;
    eval.opacity = clamp(opacity, 0.0, 1.0);
    eval.contributionProxy = 0.0;
    eval.pOld = 0.0;
    eval.pNew = 0.0;
    eval.pDelta = 0.0;
    eval.oldEccentricityDegrees = 0.0;
    eval.newEccentricityDegrees = 0.0;
    eval.flags = ECSPT_EVAL_REACHED;

    const float eccentricity = computeFoveatedEccentricityDegrees(centerNdc, camera);
    const float pLp =
        shaderAntiPopCurveKeepProbability(clamp(foveatedBaseClodLevelForEccentricity(eccentricity), 0.0, 1.0));
    float dAlpha = 0.0;
    float nEff = 0.0;
    const float pFloor =
        coverageTextureFloorEnabled() ?
            coverageTextureFloor(centerNdc, dAlpha, nEff) :
            coverageReleaseTileFloor(centerNdc, dAlpha, nEff);
    const bool guideModifierRequested = coverageGuideModifierEnabled();
    const bool guideModifierActive =
        guideModifierRequested && (!coverageTextureFloorEnabled() || coverageGuideHistoryValid());
    const float guideTarget = max(u_PC.coverageReleaseFloats.y * 96.0, 1e-4);
    const float guideCoverage = clamp(dAlpha / guideTarget, 0.0, 1.0);
    const float lowCoverageBoost = 1.0 - smoothstep(0.20, 0.55, guideCoverage);
    const float highCoverageReduce = smoothstep(0.65, 0.95, guideCoverage);
    const float guideStrength = clamp(u_PC.coverageTextureFloats.x, 0.0, 1.0);
    const float globalScale = guideModifierActive ? max(u_PC.coverageTextureFloats.w, 0.0) : 1.0;
    const float guideModifier =
        guideModifierActive ?
            clamp(1.0 + guideStrength * 0.85 * lowCoverageBoost -
                  guideStrength * 0.45 * highCoverageReduce,
                  0.20,
                  2.50) :
            1.0;
    const bool guideFoveaFullKeep =
        guideModifierRequested && eccentricity <= max(u_PC.foveatedGazeAndRings.z, 0.0);
    const float pGuided = clamp(pLp * guideModifier * globalScale, 0.0, 1.0);
    const float pStatic =
        guideFoveaFullKeep ? 1.0 :
        guideModifierActive ? pGuided :
        max(pLp, pFloor);
    const bool floorActive = guideModifierActive ? dAlpha > 1e-6 : pFloor > 0.0;
    const bool floorRaised = pStatic > pLp + 1e-6;
    const bool guideReduced = guideModifierActive && highCoverageReduce > 0.01 && guideModifier < 1.0 - 1e-6;
    const bool guideBoosted = guideModifierActive && lowCoverageBoost > 0.01 && guideModifier > 1.0 + 1e-6;

    const uint oldState = s_GeneralGaussianSplatTemporalStates.states[sourceIndex].state;
    const uint frameIndex = coverageReleaseFrameIndex16();
    const uint oldFrame = coverageReleaseUnpackHistoryFrame(oldState);
    const uint elapsedFrames = oldState == 0u ? 0u : ((frameIndex + 65536u - oldFrame) & 0xffffu);
    const bool historyReset = oldState == 0u;
    const bool reentryReset = oldState != 0u && elapsedFrames > 1u;
    const float lambda = clamp(u_PC.coverageReleaseFloats.x, 0.0, 1.0);
    const bool releaseEnabled =
        (u_PC.coverageReleaseParams.x & COVERAGE_RELEASE_FLAG_STAGGERED_RELEASE) != 0u;
    const bool releaseStateActive = releaseEnabled && oldState != 0u;
    const float pHistory = releaseStateActive ? coverageReleaseUnpackHistoryP(oldState) : 0.0;
    const uint releasePolicy = coverageReleasePolicy();
    const bool lowCoverageCell = guideModifierActive && guideCoverage < 0.55;
    const bool alphaProxyHigh = eval.opacity >= 0.32;
    const bool smallFootprintLowCoverage =
        lowCoverageCell && footprintProxyPx > 0.0 && footprintProxyPx <= coverageReleaseLargeFootprintPx() * 0.35;
    const float lowCoverageRisk = lowCoverageCell ? (1.0 - smoothstep(0.20, 0.55, guideCoverage)) : 0.0;
    float pReleaseRaw = 0.0;
    bool footprintFastDecay = false;
    if (releaseStateActive)
    {
        float effectiveLambda = lambda;
        if (releasePolicy == COVERAGE_RELEASE_POLICY_FOOTPRINT_DECAY &&
            footprintProxyPx >= coverageReleaseLargeFootprintPx())
        {
            effectiveLambda = clamp(lambda * coverageReleaseFootprintDecayScale(), 0.0, 1.0);
            footprintFastDecay = true;
        }
        const float lambdaDt =
            elapsedFrames <= 1u ? effectiveLambda : pow(effectiveLambda, float(min(elapsedFrames, 1024u)));
        pReleaseRaw = pHistory * lambdaDt;
    }
    const float releaseEpsilon = max(u_PC.coverageReleaseFloats.w, 0.0);
    const float rawReleaseDelta = max(pReleaseRaw - pStatic, 0.0);
    const float dropRisk = clamp(rawReleaseDelta * 4.0, 0.0, 1.0);
    const float alphaRisk = clamp(eval.opacity * 2.5, 0.0, 1.0);
    const float footprintRisk = smallFootprintLowCoverage ? 0.85 : 0.0;
    const float releaseRiskScore =
        clamp(max(max(lowCoverageRisk, dropRisk), max(alphaRisk, footprintRisk)), 0.0, 1.0);
    const bool riskProtected =
        rawReleaseDelta > 1e-6 &&
        (lowCoverageCell ||
         rawReleaseDelta >= 0.12 ||
         alphaProxyHigh ||
         smallFootprintLowCoverage ||
         releaseRiskScore >= coverageReleaseRiskThreshold());
    bool releasePolicyAllowed = true;
    bool releaseCapHit = false;
    if (rawReleaseDelta > 1e-6 && releasePolicy == COVERAGE_RELEASE_POLICY_RISK_ONLY)
    {
        releasePolicyAllowed = riskProtected;
    }
    if (rawReleaseDelta > 1e-6 && releasePolicy == COVERAGE_RELEASE_POLICY_CAPPED)
    {
        const float capRatio = coverageReleaseCapRatio();
        const float expectedBase = max(pStatic, 1e-4);
        const float capAcceptance = clamp(capRatio * expectedBase / max(rawReleaseDelta, 1e-4), 0.0, 1.0);
        const float priorityAcceptance = clamp(capAcceptance * mix(0.35, 1.65, releaseRiskScore), 0.0, 1.0);
        const float releaseGateHash = shaderAntiPopHash01Salted(persistentId, 0x68bc21ebu);
        releasePolicyAllowed = releaseGateHash <= priorityAcceptance;
        releaseCapHit = !releasePolicyAllowed;
    }
    if (rawReleaseDelta > 1e-6 && releasePolicy == COVERAGE_RELEASE_POLICY_TILE_LOCAL_CAPPED)
    {
        const uint tileIndex = coverageReleaseTileIndexFromNdc(centerNdc);
        const float capRatio = coverageReleaseCapRatio();
        const float expectedBase = max(pStatic, 1e-4);
        const float baseAcceptance =
            clamp(capRatio * expectedBase / max(rawReleaseDelta, 1e-4), 0.0, 1.0);
        const float denseTilePressure =
            guideModifierActive ? smoothstep(0.55, 0.95, guideCoverage) : 0.0;
        const float footprintPressure =
            footprintProxyPx > 0.0 ? smoothstep(24.0, 96.0, footprintProxyPx) : 0.0;
        const float tileBudgetScale = mix(0.95, 0.28, denseTilePressure);
        const float footprintBudgetScale = mix(1.0, 0.50, footprintPressure);
        const float riskPriority = mix(0.35, 1.0, releaseRiskScore);
        const float tileJitter =
            mix(0.95, 1.05, shaderAntiPopHash01Salted(tileIndex, 0x4d2c6b1du));
        const float tileAcceptance =
            min(baseAcceptance,
                clamp(baseAcceptance *
                      tileBudgetScale *
                      footprintBudgetScale *
                      riskPriority *
                      tileJitter,
                      0.0,
                      1.0));
        const float releaseGateHash =
            shaderAntiPopHash01Salted(persistentId ^ (tileIndex * 0x9e3779b9u), 0x8f3a51c7u);
        releasePolicyAllowed = releaseGateHash <= tileAcceptance;
        releaseCapHit = !releasePolicyAllowed;
    }
    const bool pHistoryNonzero = releaseStateActive && pHistory > 0.0;
    const bool pTargetLessThanHistory = rawReleaseDelta > 1e-6;
    const bool releaseCutoff = rawReleaseDelta > 0.0 && rawReleaseDelta < releaseEpsilon;
    const float pRelease = (releaseCutoff || !releasePolicyAllowed) ? pStatic : pReleaseRaw;
    const float pFinal = clamp(max(pStatic, pRelease), 0.0, 1.0);
    s_GeneralGaussianSplatTemporalStates.states[sourceIndex].state =
        coverageReleasePackHistory(pFinal, frameIndex);

    const bool hashEnabled =
        (u_PC.coverageReleaseParams.x & COVERAGE_RELEASE_FLAG_STABLE_HASH) != 0u;
    const float threshold = hashEnabled ? shaderAntiPopHash01(persistentId) : 0.5;
    const bool oldKeep = threshold <= pRelease;
    const bool staticKeep = threshold <= pStatic;
    const bool finalKeep = threshold <= pFinal;
    const float releaseDelta = max(pRelease - pStatic, 0.0);
    const bool rawReleaseHeldEffective = rawReleaseDelta > 1e-6 && !staticKeep && threshold <= clamp(max(pStatic, pReleaseRaw), 0.0, 1.0);
    const bool releaseCutoffEffective = releaseCutoff && rawReleaseHeldEffective;
    const bool releaseHeld = releaseDelta > 1e-6;
    const bool releaseHeldEffective = releaseHeld && !staticKeep && finalKeep;
    const bool oldOnlyRelease = releaseHeldEffective && oldKeep;

    eval.pOld = pRelease;
    eval.pNew = pStatic;
    eval.pDelta = releaseDelta;
    eval.oldEccentricityDegrees = eccentricity;
    eval.newEccentricityDegrees = eccentricity;
    eval.contributionProxy = clamp(eval.opacity * max(pFinal, 0.0), 0.0, 1.0);
    if (floorActive)
        eval.flags |= ECSPT_EVAL_COVERAGE_FLOOR_ACTIVE;
    if (floorRaised)
        eval.flags |= ECSPT_EVAL_COVERAGE_FLOOR_RAISED;
    if (staticKeep && oldKeep)
        eval.flags |= ECSPT_EVAL_SHARED;
    else if (staticKeep)
        eval.flags |= ECSPT_EVAL_UPGRADE;
    else if (oldKeep)
        eval.flags |= ECSPT_EVAL_DOWNGRADE;
    if (releaseHeldEffective)
        eval.flags |= ECSPT_EVAL_PROTECTED_DOWNGRADE | ECSPT_EVAL_COVERAGE_RELEASE_HELD;
    if (oldOnlyRelease)
        eval.flags |= ECSPT_EVAL_COVERAGE_OLD_ONLY_RELEASE;
    if (finalKeep)
    {
        eval.flags |= ECSPT_EVAL_NEW_SELECTED;
    }
    else
    {
        eval.weight = 0.0;
        eval.flags |= ECSPT_EVAL_ZERO_WEIGHT | ECSPT_EVAL_DROPPED_DOWNGRADE;
    }

    if (coverageReleaseDiagnosticsEnabled())
    {
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_CANDIDATE_ALPHA_SUM,
                        ecsptScaledCounterValue(eval.opacity));
        if (finalKeep)
            ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_RETAINED_ALPHA_SUM,
                            ecsptScaledCounterValue(eval.opacity));
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_FLOOR_ACTIVE_COUNT, floorActive ? 1u : 0u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_FLOOR_RAISED_COUNT, floorRaised ? 1u : 0u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_RELEASE_HELD_COUNT, releaseHeld ? 1u : 0u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_FINAL_DROPPED_COUNT, finalKeep ? 0u : 1u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_OLD_ONLY_RELEASE_COUNT, oldOnlyRelease ? 1u : 0u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_P_FLOOR_SUM, uint(round(pFloor * COVERAGE_RELEASE_COUNTER_SCALE)));
        atomicMax(s_GeneralGaussianSplatEcsptCounters.counters[ECSPT_COUNTER_COVERAGE_P_FLOOR_MAX],
                  uint(round(pFloor * COVERAGE_RELEASE_COUNTER_SCALE)));
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_D_ALPHA_SUM, uint(round(clamp(dAlpha, 0.0, 1024.0) * COVERAGE_RELEASE_COUNTER_SCALE)));
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_N_EFF_SUM, uint(round(clamp(nEff, 0.0, 1024.0) * COVERAGE_RELEASE_COUNTER_SCALE)));
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_FLOOR_SAMPLE_COUNT, 1u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_P_LP_SUM, uint(round(pLp * COVERAGE_RELEASE_COUNTER_SCALE)));
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_P_STATIC_SUM, uint(round(pStatic * COVERAGE_RELEASE_COUNTER_SCALE)));
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_RELEASE_DELTA_SUM,
                        uint(round(clamp(releaseDelta, 0.0, 1024.0) * COVERAGE_RELEASE_COUNTER_SCALE)));
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_HISTORY_RESET_COUNT, historyReset ? 1u : 0u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_REENTRY_RESET_COUNT, reentryReset ? 1u : 0u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_RELEASE_HELD_EFFECTIVE_COUNT, releaseHeldEffective ? 1u : 0u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_RELEASE_EPSILON_CUTOFF_COUNT, releaseCutoffEffective ? 1u : 0u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_P_FINAL_SUM,
                        uint(round(pFinal * COVERAGE_RELEASE_COUNTER_SCALE)));
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_STABLE_HASH_KEPT_COUNT, finalKeep ? 1u : 0u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_GUIDE_REDUCED_COUNT, guideReduced ? 1u : 0u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_GUIDE_BOOSTED_COUNT, guideBoosted ? 1u : 0u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_P_HISTORY_NONZERO_COUNT, pHistoryNonzero ? 1u : 0u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_P_TARGET_LESS_THAN_HISTORY_COUNT, pTargetLessThanHistory ? 1u : 0u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_RELEASED_EFFECTIVE_VISIBLE_COUNT,
                        releaseHeldEffective ? 1u : 0u);
        if (releaseHeldEffective)
        {
            ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_RELEASED_ALPHA_PROXY_SUM,
                            uint(round(clamp(eval.opacity, 0.0, 1024.0) * COVERAGE_RELEASE_COUNTER_SCALE)));
        }
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_RELEASE_CAP_HIT_COUNT, releaseCapHit ? 1u : 0u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_RISK_PROTECTED_COUNT,
                        releaseHeldEffective && riskProtected ? 1u : 0u);
        ecsptCounterAdd(ECSPT_COUNTER_COVERAGE_FOOTPRINT_FAST_DECAY_COUNT,
                        footprintFastDecay ? 1u : 0u);
    }
    return eval;
}

uint ecsptEccentricityBin(const EccStochasticEval eval)
{
    const float foveaDegrees = max(u_PC.foveatedGazeAndRings.z, 0.0);
    const float midDegrees = max(u_PC.foveatedGazeAndRings.w, foveaDegrees + 1e-3);
    const float boundaryBandDegrees = max(u_PC.gazeAnchorFloats.w, 0.0);
    const float oldEccentricity = eval.oldEccentricityDegrees;
    const float newEccentricity = eval.newEccentricityDegrees;
    const float minEccentricity = min(oldEccentricity, newEccentricity);
    const float oldBoundaryDistance =
        min(abs(oldEccentricity - foveaDegrees), abs(oldEccentricity - midDegrees));
    const float newBoundaryDistance =
        min(abs(newEccentricity - foveaDegrees), abs(newEccentricity - midDegrees));

    if (newEccentricity <= foveaDegrees)
        return 0u;
    if (oldEccentricity <= foveaDegrees && newEccentricity > foveaDegrees)
        return 1u;
    if (min(oldBoundaryDistance, newBoundaryDistance) <= max(boundaryBandDegrees, 1.0))
        return 2u;
    if (minEccentricity <= midDegrees + max(boundaryBandDegrees, 1.0) * 2.0)
        return 3u;
    return 4u;
}

void ecsptLogEventBinsForEye(const EyePreprocessResult eye)
{
    const uint flags = eye.eccStochasticFlags;
    if ((flags & ECSPT_EVAL_REACHED) == 0u)
        return;

    EccStochasticEval eval;
    eval.weight = eye.eccStochasticWeight;
    eval.opacity = eye.eccStochasticOpacity;
    eval.contributionProxy = eye.eccStochasticContributionProxy;
    eval.pOld = eye.eccStochasticPOld;
    eval.pNew = eye.eccStochasticPNew;
    eval.pDelta = eye.eccStochasticPDelta;
    eval.oldEccentricityDegrees = eye.eccStochasticOldEccentricityDegrees;
    eval.newEccentricityDegrees = eye.eccStochasticNewEccentricityDegrees;
    eval.flags = flags;

    const uint bin = ecsptEccentricityBin(eval);
    const bool instantMode = u_PC.gazeAnchorParams.y == 0u;
    const bool sharedEvent = (flags & ECSPT_EVAL_SHARED) != 0u;
    const bool upgrade = (flags & ECSPT_EVAL_UPGRADE) != 0u;
    const bool downgrade = (flags & ECSPT_EVAL_DOWNGRADE) != 0u;
    const bool protectedDowngrade = (flags & ECSPT_EVAL_PROTECTED_DOWNGRADE) != 0u;
    const bool droppedDowngrade = (flags & ECSPT_EVAL_DROPPED_DOWNGRADE) != 0u;
    const bool immediateNewFovea = (flags & ECSPT_EVAL_IMMEDIATE_NEW_FOVEA) != 0u;

    if (sharedEvent)
    {
        ecsptEventBinAdd(ECSPT_EVENT_SHARED,
                         bin,
                         eval.opacity,
                         eval.contributionProxy,
                         eval.pDelta,
                         eval.pOld,
                         eval.pNew);
    }
    if (upgrade)
    {
        ecsptEventBinAdd(instantMode ? ECSPT_EVENT_INSTANT_HASH_FLIP : ECSPT_EVENT_UPGRADE,
                         bin,
                         eval.opacity,
                         eval.contributionProxy,
                         eval.pDelta,
                         eval.pOld,
                         eval.pNew);
    }
    if (downgrade)
    {
        const uint eventType =
            instantMode ? ECSPT_EVENT_INSTANT_HASH_FLIP :
            protectedDowngrade ? ECSPT_EVENT_DOWNGRADE_PROTECTED :
            ECSPT_EVENT_DOWNGRADE_DROPPED;
        ecsptEventBinAdd(eventType,
                         bin,
                         eval.opacity,
                         eval.contributionProxy,
                         eval.pDelta,
                         eval.pOld,
                         eval.pNew);
    }
    if (immediateNewFovea)
    {
        ecsptEventBinAdd(ECSPT_EVENT_IMMEDIATE_NEW_FOVEA,
                         bin,
                         eval.opacity,
                         eval.contributionProxy,
                         eval.pDelta,
                         eval.pOld,
                         eval.pNew);
    }
}

float foveatedSelectionWeight(const uint sourceIndex,
                              const uint rank,
                              const float level,
                              const float eccentricityDegrees)
{
    if ((u_PC.foveatedClodEnabled & FOVEATED_CLOD_ENABLED_FLAG) == 0u && !shaderAntiPopActive())
        return 1.0;
    if ((u_PC.foveatedClodEnabled & FOVEATED_CLOD_SCORE_SELECTED_FLAG) != 0u)
        return 1.0;

    const float clampedLevel = clamp(level, 0.0, 1.0);
    const float rankRatio = foveatedRankRatio(rank);
    if (u_PC.shaderAntiPopParams.x == SHADER_ANTIPOP_COVERAGE_STABLE_LOGPOLAR_RELEASE)
        return 1.0;
    if (shaderAntiPopActive())
        return 1.0;

    const float peripheralScale = max(u_PC.foveatedContinuousParams.w, 0.0);
    const float peripheralFactor = foveatedTemporalPeripheralFactor(eccentricityDegrees);
    const float stabilityScale = 1.0 + peripheralScale * peripheralFactor;
    const float hysteresisRatio = clamp(max(u_PC.foveatedTemporalParams.x, 0.0) * stabilityScale, 0.0, 1.0);
    const float offLevel = clamp(clampedLevel * (1.0 + hysteresisRatio), clampedLevel, 1.0);
    const bool insideOn = rankRatio <= clampedLevel;
    const bool insideOff = rankRatio <= offLevel;

    if (!foveatedTemporalHysteresisEnabled())
    {
        if (!insideOn && !(foveatedBoundarySmoothingEnabled() && insideOff))
            return 0.0;
        return foveatedBoundaryOpacityWeight(rankRatio, clampedLevel, offLevel);
    }

    const uint oldState = s_GeneralGaussianSplatTemporalStates.states[sourceIndex].state;
    const uint oldPhase = oldState & FOVEATED_TEMPORAL_PHASE_MASK;
    const uint oldResidency =
        (oldState >> FOVEATED_TEMPORAL_RESIDENCY_SHIFT) & FOVEATED_TEMPORAL_RESIDENCY_MASK;
    const uint minResidency = min(uint(ceil(max(u_PC.foveatedTemporalParams.z, 0.0) * stabilityScale)),
                                  FOVEATED_TEMPORAL_RESIDENCY_MASK);

    bool pass = false;
    uint newPhase = FOVEATED_TEMPORAL_INACTIVE;
    uint newResidency = 0u;

    if (insideOn)
    {
        pass = true;
        newPhase = FOVEATED_TEMPORAL_ACTIVE;
        newResidency = minResidency;
    }
    else if (oldPhase == FOVEATED_TEMPORAL_ACTIVE && (insideOff || oldResidency > 0u))
    {
        pass = true;
        newPhase = FOVEATED_TEMPORAL_ACTIVE;
        newResidency = oldResidency > 0u ? oldResidency - 1u : 0u;
    }
    else if (insideOff)
    {
        newPhase = FOVEATED_TEMPORAL_CANDIDATE;
    }
    else if (oldPhase == FOVEATED_TEMPORAL_ACTIVE || oldPhase == FOVEATED_TEMPORAL_CANDIDATE)
    {
        newPhase = FOVEATED_TEMPORAL_COOLDOWN;
    }

    s_GeneralGaussianSplatTemporalStates.states[sourceIndex].state =
        packFoveatedTemporalState(newPhase, newResidency);

    if (!pass)
        return 0.0;

    return foveatedBoundaryOpacityWeight(rankRatio, clampedLevel, offLevel);
}

bool isInsideFoveatedLayerEccentricity(const uint layer, const float eccentricityDegrees)
{
    return gaussianFoveatedLayerContains(layer,
                                         eccentricityDegrees,
                                         u_PC.foveatedGazeAndRings.zw,
                                         u_PC.foveatedLevelsAndTransition.w);
}

bool foveatedCoverageCompensationEnabled()
{
    return (u_PC.foveatedClodEnabled & FOVEATED_CLOD_COVERAGE_COMPENSATION_FLAG) != 0u;
}

float foveatedCoverageAlphaBoost(const float level)
{
    if (!foveatedCoverageCompensationEnabled())
        return 1.0;

    const float safeLevel = max(clamp(level, 0.0, 1.0), FOVEATED_COVERAGE_MIN_LEVEL);
    return clamp(sqrt(1.0 / safeLevel), 1.0, FOVEATED_COVERAGE_MAX_ALPHA_BOOST);
}

float foveatedCoverageRadiusScale(const float level)
{
    const float boost = foveatedCoverageAlphaBoost(level);
    return clamp(sqrt(boost), 1.0, FOVEATED_COVERAGE_MAX_RADIUS_SCALE);
}

float foveatedCoverageCompensatedOpacity(const float opacity, const float level)
{
    const float alpha = clamp(opacity, 0.0, 0.999);
    const float boost = foveatedCoverageAlphaBoost(level);
    return 1.0 - pow(1.0 - alpha, boost);
}

mat3 buildJacobian(const vec3 camspace, const vec2 focal)
{
    float z = camspace.z;
    if (abs(z) < 1e-4)
        z = z < 0.0 ? -1e-4 : 1e-4;
    return mat3(vec3(focal.x / z, 0.0, -(focal.x * camspace.x) / (z * z)),
                vec3(0.0, -focal.y / z, (focal.y * camspace.y) / (z * z)),
                vec3(0.0));
}

bool foveatedShSmoothSuppressionEnabled()
{
    return (u_PC.shLodParams.x & FOVEATED_SH_SMOOTH_SUPPRESSION_FLAG) != 0u;
}

vec3 foveatedShSmoothBandWeights(const float eccentricityDegrees)
{
    const float foveaDegrees = max(u_PC.foveatedGazeAndRings.z, 0.0);
    const float midDegrees = max(u_PC.foveatedGazeAndRings.w, foveaDegrees + 1.0);
    const float transitionDegrees = max(u_PC.foveatedLevelsAndTransition.w, 1.0);
    const float halfDegrees = foveaDegrees + 0.5 * (midDegrees - foveaDegrees);
    const float beta3 = 1.0 - smoothstep(foveaDegrees, max(halfDegrees, foveaDegrees + 1.0), eccentricityDegrees);
    const float beta2 = 1.0 - smoothstep(halfDegrees, midDegrees, eccentricityDegrees);
    const float beta1 = 1.0 - smoothstep(midDegrees, midDegrees + transitionDegrees, eccentricityDegrees);
    return clamp(vec3(beta1, beta2, beta3), vec3(0.0), vec3(1.0));
}

vec3 evaluateGeneralGaussianSplatColorWeighted(const vec3 dir,
                                               const GeneralGaussianSplatPackedSource src,
                                               const uint shDegree,
                                               const vec3 bandWeights)
{
    vec3 result = decodeGeneralGaussianSplatBaseColorOpacity(src).rgb;
    if (shDegree == 0u)
        return max(result, vec3(0.0));

    const float x = dir.x;
    const float y = dir.y;
    const float z = dir.z;

    result += bandWeights.x *
              (-SH_C1 * y * decodeGeneralGaussianSplatShCoeffFromSource(src, 0u) +
               SH_C1 * z * decodeGeneralGaussianSplatShCoeffFromSource(src, 1u) -
               SH_C1 * x * decodeGeneralGaussianSplatShCoeffFromSource(src, 2u));

    if (shDegree > 1u)
    {
        const float xx = x * x;
        const float yy = y * y;
        const float zz = z * z;
        const float xy = x * y;
        const float yz = y * z;
        const float xz = x * z;

        result += bandWeights.y *
                  (SH_C2[0] * xy * decodeGeneralGaussianSplatShCoeffFromSource(src, 3u) +
                   SH_C2[1] * yz * decodeGeneralGaussianSplatShCoeffFromSource(src, 4u) +
                   SH_C2[2] * (2.0 * zz - xx - yy) * decodeGeneralGaussianSplatShCoeffFromSource(src, 5u) +
                   SH_C2[3] * xz * decodeGeneralGaussianSplatShCoeffFromSource(src, 6u) +
                   SH_C2[4] * (xx - yy) * decodeGeneralGaussianSplatShCoeffFromSource(src, 7u));
    }

    if (shDegree > 2u)
    {
        const float xx = x * x;
        const float yy = y * y;
        const float zz = z * z;
        const float xy = x * y;
        const float yz = y * z;
        const float xz = x * z;

        result += bandWeights.z *
                  (SH_C3[0] * y * (3.0 * xx - yy) * decodeGeneralGaussianSplatShCoeffFromSource(src, 8u) +
                   SH_C3[1] * xy * z * decodeGeneralGaussianSplatShCoeffFromSource(src, 9u) +
                   SH_C3[2] * y * (4.0 * zz - xx - yy) * decodeGeneralGaussianSplatShCoeffFromSource(src, 10u) +
                   SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) *
                       decodeGeneralGaussianSplatShCoeffFromSource(src, 11u) +
                   SH_C3[4] * x * (4.0 * zz - xx - yy) * decodeGeneralGaussianSplatShCoeffFromSource(src, 12u) +
                   SH_C3[5] * z * (xx - yy) * decodeGeneralGaussianSplatShCoeffFromSource(src, 13u) +
                   SH_C3[6] * x * (xx - 3.0 * yy) * decodeGeneralGaussianSplatShCoeffFromSource(src, 14u));
    }

    return max(result, vec3(0.0));
}

vec3 evaluateGeneralGaussianSplatColor(const vec3 dir, const GeneralGaussianSplatPackedSource src, const uint shDegree)
{
    return evaluateGeneralGaussianSplatColorWeighted(dir, src, shDegree, vec3(1.0));
}

uint foveatedShLodDegreeForEccentricity(const float eccentricityDegrees, const uint sourceShDegree)
{
    const uint sourceDegree = min(sourceShDegree, 3u);
    if ((u_PC.shLodParams.x & FOVEATED_SH_LOD_ENABLED_FLAG) == 0u)
        return sourceDegree;

    const float foveaDegrees = max(u_PC.foveatedGazeAndRings.z, 0.0);
    const float midDegrees = max(u_PC.foveatedGazeAndRings.w, foveaDegrees);
    uint requestedDegree = min(u_PC.shLodParams.w, 3u);
    if (eccentricityDegrees <= foveaDegrees)
        requestedDegree = min(u_PC.shLodParams.y, 3u);
    else if (eccentricityDegrees <= midDegrees)
        requestedDegree = min(u_PC.shLodParams.z, 3u);

    return min(requestedDegree, sourceDegree);
}

float foveatedShLodGazeWeight(const float eccentricityDegrees)
{
    const float midDegrees = max(max(u_PC.foveatedGazeAndRings.w, u_PC.foveatedGazeAndRings.z), 1.0);
    return clamp(1.0 / (1.0 + max(eccentricityDegrees, 0.0) / midDegrees), 0.05, 1.0);
}

float foveatedShLodHighEnergyAfterDegree(const uint sourceIndex, const uint baseDegree)
{
#if USE_SH_LOD_ENERGY_GUARD
    const vec4 energy = s_GeneralGaussianSplatShEnergyMetadata.energy[sourceIndex];
    if (baseDegree == 0u)
        return max(energy.x, 0.0);
    if (baseDegree == 1u)
        return max(energy.y, 0.0);
    if (baseDegree == 2u)
        return max(energy.z, 0.0);
#endif
    return 0.0;
}

uint guardedFoveatedShLodDegree(const float eccentricityDegrees,
                                const uint sourceShDegree,
                                const uint sourceIndex,
                                const float projectedAreaPx,
                                const float opacity)
{
    const uint baseDegree = foveatedShLodDegreeForEccentricity(eccentricityDegrees, sourceShDegree);
    const uint sourceDegree = min(sourceShDegree, 3u);
    const uint guardMode = uint(round(clamp(u_PC.shGuardParams.z, 0.0, 3.0)));
    if (guardMode == FOVEATED_SH_LOD_GUARD_OFF || baseDegree >= sourceDegree)
        return baseDegree;

    const bool usesEnergy =
        guardMode == FOVEATED_SH_LOD_GUARD_ENERGY ||
        guardMode == FOVEATED_SH_LOD_GUARD_ENERGY_PROJECTED_COST;
    const bool usesProjectedCost =
        guardMode == FOVEATED_SH_LOD_GUARD_PROJECTED_COST ||
        guardMode == FOVEATED_SH_LOD_GUARD_ENERGY_PROJECTED_COST;
    const float energyTerm = usesEnergy ? foveatedShLodHighEnergyAfterDegree(sourceIndex, baseDegree) : 1.0;
    const float projectedTerm = usesProjectedCost ? max(projectedAreaPx, 1e-6) : 1.0;
    const float risk = foveatedShLodGazeWeight(eccentricityDegrees) *
                       energyTerm *
                       projectedTerm *
                       clamp(opacity, 0.0, 1.0);
    const float thresholdMid = max(u_PC.shGuardParams.x, 0.0);
    const float thresholdHigh = max(u_PC.shGuardParams.y, thresholdMid);

    if (risk > thresholdHigh)
        return sourceDegree;
    if (risk > thresholdMid)
        return min(max(baseDegree, 2u), sourceDegree);
    return baseDegree;
}

float computeGeneralGaussianSplatSortDepth(const vec4 posClip,
                                           const vec4 posView,
                                           const mat3 sigmaWorld,
                                           const mat3 viewLinear,
                                           const float zFar,
                                           const uint sortOrder)
{
    if (sortOrder == SORT_ORDER_DISTANCE)
        return max(zFar - length(posView.xyz), 0.0);

    const float viewDepth = max(-posView.z, 0.0);
    if (sortOrder == SORT_ORDER_VIEW_DEPTH)
        return max(zFar - viewDepth, 0.0);

    if (sortOrder == SORT_ORDER_CONSERVATIVE_DEPTH)
    {
        const mat3 sigmaView = viewLinear * sigmaWorld * transpose(viewLinear);
        const float depthRadius = 3.0 * sqrt(max(sigmaView[2][2], 0.0));
        return max(zFar - max(viewDepth - depthRadius, 0.0), 0.0);
    }

    return max(zFar - posClip.z, 0.0);
}

EyePreprocessResult preprocessEye(const GeneralGaussianSplatPackedSource src,
                                  const GeneralGaussianSplatDrawRecord draw,
                                  const vec3 localPos,
                                  const vec3 worldPos,
                                  const mat3 modelLinear,
                                  const vec3 modelTranslation,
                                  const float lodWeight,
                                  const float temporalWeight,
                                  const uint sourceIndex,
                                  const uint rank,
                                  const CameraData camera,
                                  const bool accumulateCoverageTextureCurrent)
{
    EyePreprocessResult result;
    result.v1 = vec2(0.0);
    result.v2 = vec2(0.0);
    result.centerNdc = vec2(2.0);
    result.eccentricityDegrees = 180.0;
    result.depth = 1.0;
    result.colorOpacity = vec4(0.0);
    result.sortDepth = 0.0;
    result.eccStochasticWeight = 1.0;
    result.eccStochasticOpacity = 0.0;
    result.eccStochasticContributionProxy = 0.0;
    result.eccStochasticPOld = 0.0;
    result.eccStochasticPNew = 0.0;
    result.eccStochasticPDelta = 0.0;
    result.eccStochasticOldEccentricityDegrees = 0.0;
    result.eccStochasticNewEccentricityDegrees = 0.0;
    result.eccStochasticFlags = 0u;
    result.projectedRadiusPx = 0.0;
    result.tileDuplicateProxy = 1u;
    result.visible = false;

    vec4 colorOpacity = decodeGeneralGaussianSplatBaseColorOpacity(src);
    colorOpacity.a *= max(draw.params0.z, 0.0);
    // Ordered CLOD encodes transition fade as an opacity multiplier. Geometry,
    // covariance and SH evaluation stay identical to the raw splat path.
    colorOpacity.a *= lodWeight;
    colorOpacity.a *= temporalWeight;
    if (colorOpacity.a < MIN_VISIBLE_OPACITY)
        return result;

    const vec4 posView = camera.view * vec4(worldPos, 1.0);
    const vec4 posClip = camera.projection * posView;
    if (posClip.w <= 1e-5)
        return result;

    const vec3 centerNdc = posClip.xyz / posClip.w;
    const float bounds = 1.2 * posClip.w;
    if (centerNdc.z <= 0.0 || centerNdc.z >= 1.0)
        return result;
    if (posClip.x < -bounds || posClip.x > bounds || posClip.y < -bounds || posClip.y > bounds)
        return result;
    if (accumulateCoverageTextureCurrent && coverageTextureFloorEnabled())
        coverageTextureAccumulateProjected(centerNdc.xy, clamp(colorOpacity.a, 0.0, 1.0));
    const vec2 viewport = camera.resolution.xy;
    const vec2 focal =
        0.5 * vec2(abs(camera.projection[0][0]) * viewport.x, abs(camera.projection[1][1]) * viewport.y);
    const float eccentricityDegrees = computeFoveatedEccentricityDegrees(centerNdc.xy, camera);
#if !USE_FOVEATED_LAYER_OUTPUT
    const float foveatedClodLevel = foveatedClodLevelForEccentricity(eccentricityDegrees);
    if (shaderAntiPopActive() &&
        u_PC.shaderAntiPopParams.x == SHADER_ANTIPOP_COVERAGE_STABLE_LOGPOLAR_RELEASE)
    {
        const uint persistentId = src.aux0.x != 0u ? src.aux0.x : sourceIndex;
        float releaseFootprintProxyPx = 0.0;
        const uint releasePolicy = coverageReleasePolicy();
        if (releasePolicy == COVERAGE_RELEASE_POLICY_FOOTPRINT_DECAY ||
            releasePolicy == COVERAGE_RELEASE_POLICY_TILE_LOCAL_CAPPED)
        {
            const mat3 releaseSigmaLocal = decodeGeneralGaussianSplatCovariance(src);
            const float releaseSigmaTrace =
                max(releaseSigmaLocal[0][0] + releaseSigmaLocal[1][1] + releaseSigmaLocal[2][2], 0.0);
            const float modelScale = max(max(length(modelLinear[0]), length(modelLinear[1])), length(modelLinear[2]));
            const float focalMean = max((focal.x + focal.y) * 0.5, 1.0);
            releaseFootprintProxyPx =
                sqrt(max(releaseSigmaTrace, 0.0)) * max(modelScale, 1e-4) *
                focalMean / max(abs(posView.z), 1e-4) * max(draw.params0.y, 1e-3);
        }
        const EccStochasticEval coverageRelease =
            shaderCoverageStableLogpolarReleaseEval(sourceIndex,
                                                    persistentId,
                                                    centerNdc.xy,
                                                    camera,
                                                    colorOpacity.a,
                                                    releaseFootprintProxyPx);
        result.eccStochasticWeight = coverageRelease.weight;
        result.eccStochasticOpacity = coverageRelease.opacity;
        result.eccStochasticContributionProxy = coverageRelease.contributionProxy;
        result.eccStochasticPOld = coverageRelease.pOld;
        result.eccStochasticPNew = coverageRelease.pNew;
        result.eccStochasticPDelta = coverageRelease.pDelta;
        result.eccStochasticOldEccentricityDegrees = coverageRelease.oldEccentricityDegrees;
        result.eccStochasticNewEccentricityDegrees = coverageRelease.newEccentricityDegrees;
        result.eccStochasticFlags = coverageRelease.flags;
        if (coverageRelease.weight <= 0.0)
            return result;
    }
    const float foveatedTemporalWeight =
        foveatedSelectionWeight(sourceIndex, rank, foveatedClodLevel, eccentricityDegrees);
    if (foveatedTemporalWeight <= 0.0)
        return result;
    colorOpacity.a *= foveatedTemporalWeight;
    result.eccentricityDegrees = eccentricityDegrees;
#else
    if (u_PC.foveatedLayerParams.x != 0u &&
        !isInsideFoveatedLayerEccentricity(u_PC.foveatedLayerParams.x - 1u, eccentricityDegrees))
    {
        return result;
    }
    result.eccentricityDegrees = eccentricityDegrees;
#endif
    const float foveatedCoverageLevel =
        ((u_PC.foveatedClodEnabled & FOVEATED_CLOD_ENABLED_FLAG) != 0u) ?
            foveatedClodLevelForEccentricity(eccentricityDegrees) :
            1.0;

    const mat3 sigmaLocal = decodeGeneralGaussianSplatCovariance(src);
    const mat3 sigmaWorld = modelLinear * sigmaLocal * transpose(modelLinear);
    const mat3 viewLinear = mat3(camera.view);
    const mat3 J          = buildJacobian(posView.xyz, focal);
    const mat3 W          = transpose(mat3(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));
    const mat3 T          = W * J;
    const mat3 cov        = transpose(T) * sigmaWorld * T;

    const float kernelSize = max(draw.params0.x, 1e-4);
    const float det0 = max(1e-6, cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]);
    const float det1 =
        max(1e-6, (cov[0][0] + kernelSize) * (cov[1][1] + kernelSize) - cov[0][1] * cov[0][1]);
    if (det0 <= 1e-6 || det1 <= 1e-6)
        return result;

    colorOpacity.a *= sqrt(det0 / (det1 + 1e-6) + 1e-6);
    colorOpacity.a = foveatedCoverageCompensatedOpacity(colorOpacity.a, foveatedCoverageLevel);
    if (colorOpacity.a < MIN_VISIBLE_OPACITY)
        return result;

    const float diagonal1 = cov[0][0] + kernelSize;
    const float offDiagonal = cov[0][1];
    const float diagonal2 = cov[1][1] + kernelSize;
    const float mid = 0.5 * (diagonal1 + diagonal2);
    const float radius = length(vec2((diagonal1 - diagonal2) * 0.5, offDiagonal));
    const float lambda1 = max(mid + radius, 1e-4);
    const float lambda2 = max(mid - radius, 0.1);

    vec2 diagonalVector = vec2(offDiagonal, lambda1 - diagonal1);
    if (length(diagonalVector) < 1e-6)
        diagonalVector = vec2(1.0, 0.0);
    else
        diagonalVector = normalize(diagonalVector);

    const float cutoffScale = max(draw.params0.y, 1e-3);
    const float coverageScale = foveatedCoverageRadiusScale(foveatedCoverageLevel);
    result.v1 = sqrt(2.0 * lambda1) * diagonalVector * cutoffScale * coverageScale;
    result.v2 = sqrt(2.0 * lambda2) * vec2(diagonalVector.y, -diagonalVector.x) * cutoffScale * coverageScale;
    const vec2 projectedHalfExtentPx = abs(result.v1) + abs(result.v2);
    result.projectedRadiusPx = max(projectedHalfExtentPx.x, projectedHalfExtentPx.y);
    result.tileDuplicateProxy =
        estimateProjectedTileDuplicateProxy(centerNdc.xy, result.v1, result.v2, viewport);

    const vec3 cameraWorld = camera.inverseView[3].xyz;
    vec3 dirLocal;
    const float detLinear = determinant(modelLinear);
    if (abs(detLinear) < 1e-8)
    {
        dirLocal = worldPos - cameraWorld;
    }
    else
    {
        const vec3 cameraLocal = inverse(modelLinear) * (cameraWorld - modelTranslation);
        dirLocal = localPos - cameraLocal;
    }
    if (dot(dirLocal, dirLocal) < 1e-10)
        dirLocal = vec3(0.0, 0.0, 1.0);
    else
        dirLocal = normalize(dirLocal);

    result.centerNdc = centerNdc.xy;
    result.depth = centerNdc.z;
    result.colorOpacity = colorOpacity;
    const float projectedAreaPx =
        3.141592653589793 * abs(result.v1.x * result.v2.y - result.v1.y * result.v2.x);
    const uint effectiveShDegree = guardedFoveatedShLodDegree(eccentricityDegrees,
                                                              min(draw.shDegree, 3u),
                                                              sourceIndex,
                                                              projectedAreaPx,
                                                              result.colorOpacity.a);
    if (foveatedShSmoothSuppressionEnabled())
    {
        result.colorOpacity.rgb =
            evaluateGeneralGaussianSplatColorWeighted(dirLocal,
                                                      src,
                                                      min(draw.shDegree, 3u),
                                                      foveatedShSmoothBandWeights(eccentricityDegrees));
    }
    else
    {
        result.colorOpacity.rgb = evaluateGeneralGaussianSplatColor(dirLocal, src, effectiveShDegree);
    }
    result.sortDepth = computeGeneralGaussianSplatSortDepth(posClip,
                                                            posView,
                                                            sigmaWorld,
                                                            viewLinear,
                                                            camera.zFar,
                                                            uint(round(clamp(draw.params0.w, 0.0, 3.0))));
    result.visible = true;
    return result;
}

void packEyeResult(const EyePreprocessResult eye,
                   const uint sourceIndex,
                   const uint drawIndex,
                   out uvec4 packed0,
                   out uvec4 packed1)
{
    if (!eye.visible)
    {
        packed0 = uvec4(packHalf2x16(vec2(0.0)),
                        packHalf2x16(vec2(0.0)),
                        packHalf2x16(vec2(2.0)),
                        floatBitsToUint(1.0));
        packed1 = uvec4(0u, 0u, sourceIndex, drawIndex);
        return;
    }

    packed0 = uvec4(packHalf2x16(eye.v1 / u_Camera.resolution.xy),
                    packHalf2x16(eye.v2 / u_Camera.resolution.xy),
                    packHalf2x16(eye.centerNdc),
                    floatBitsToUint(eye.depth));
    packed1 = uvec4(packHalf2x16(eye.colorOpacity.rg),
                    packHalf2x16(eye.colorOpacity.ba),
                    sourceIndex,
                    drawIndex);
}

float combinedFoveatedEccentricity(const EyePreprocessResult eye0, const EyePreprocessResult eye1)
{
    if (eye0.visible && eye1.visible)
        return min(eye0.eccentricityDegrees, eye1.eccentricityDegrees);
    if (eye0.visible)
        return eye0.eccentricityDegrees;
    return eye1.eccentricityDegrees;
}

void writeVisibleSplat(const EyePreprocessResult eye0,
                       const EyePreprocessResult eye1,
                       const uint sourceIndex,
                       const uint drawIndex,
                       const uint visibleIndex,
                       const float sortDepth,
                       inout GeneralGaussianSplatVisibleSplat visibleSplat,
                       inout uint sortKey,
                       inout uint sortIndex)
{
    packEyeResult(eye0,
                  sourceIndex,
                  drawIndex,
                  visibleSplat.packedEye0_0,
                  visibleSplat.packedEye0_1);
    packEyeResult(eye1,
                  sourceIndex,
                  drawIndex,
                  visibleSplat.packedEye1_0,
                  visibleSplat.packedEye1_1);
#if USE_DETERMINISTIC_SOURCE_ORDER_SORT
    sortKey = sourceIndex;
#else
    sortKey = floatBitsToUint(sortDepth);
#endif
    sortIndex = visibleIndex;
}

#if USE_FOVEATED_LAYER_OUTPUT
bool writeFoveatedLayerSplat(const uint layer,
                             const EyePreprocessResult eye0,
                             const EyePreprocessResult eye1,
                             const uint sourceIndex,
                             const uint drawIndex,
                             const float selectionWeight,
                             const float sortDepth)
{
    if (selectionWeight <= 0.0)
        return false;

    EyePreprocessResult weightedEye0 = eye0;
    EyePreprocessResult weightedEye1 = eye1;
    weightedEye0.colorOpacity.a *= selectionWeight;
    weightedEye1.colorOpacity.a *= selectionWeight;

    uint visibleIndex = 0u;
    if (layer == 0u)
        visibleIndex = atomicAdd(s_GeneralGaussianSplatFoveatedFoveaVisibleCount.visibleCount, 1u);
    else if (layer == 1u)
        visibleIndex = atomicAdd(s_GeneralGaussianSplatFoveatedMidVisibleCount.visibleCount, 1u);
    else
        visibleIndex = atomicAdd(s_GeneralGaussianSplatFoveatedOuterVisibleCount.visibleCount, 1u);

    if (visibleIndex >= u_PC.maxVisibleSplats)
    {
        if (layer == 0u)
            atomicMin(s_GeneralGaussianSplatFoveatedFoveaVisibleCount.visibleCount, u_PC.maxVisibleSplats);
        else if (layer == 1u)
            atomicMin(s_GeneralGaussianSplatFoveatedMidVisibleCount.visibleCount, u_PC.maxVisibleSplats);
        else
            atomicMin(s_GeneralGaussianSplatFoveatedOuterVisibleCount.visibleCount, u_PC.maxVisibleSplats);
        return false;
    }

    if (layer == 0u)
    {
        writeVisibleSplat(weightedEye0,
                          weightedEye1,
                          sourceIndex,
                          drawIndex,
                          visibleIndex,
                          sortDepth,
                          s_GeneralGaussianSplatFoveatedFoveaVisibleSplats.splats[visibleIndex],
                          s_GeneralGaussianSplatFoveatedFoveaSortKeys.keys[visibleIndex],
                          s_GeneralGaussianSplatFoveatedFoveaSortIndices.indices[visibleIndex]);
    }
    else if (layer == 1u)
    {
        writeVisibleSplat(weightedEye0,
                          weightedEye1,
                          sourceIndex,
                          drawIndex,
                          visibleIndex,
                          sortDepth,
                          s_GeneralGaussianSplatFoveatedMidVisibleSplats.splats[visibleIndex],
                          s_GeneralGaussianSplatFoveatedMidSortKeys.keys[visibleIndex],
                          s_GeneralGaussianSplatFoveatedMidSortIndices.indices[visibleIndex]);
    }
    else
    {
        writeVisibleSplat(weightedEye0,
                          weightedEye1,
                          sourceIndex,
                          drawIndex,
                          visibleIndex,
                          sortDepth,
                          s_GeneralGaussianSplatFoveatedOuterVisibleSplats.splats[visibleIndex],
                          s_GeneralGaussianSplatFoveatedOuterSortKeys.keys[visibleIndex],
                          s_GeneralGaussianSplatFoveatedOuterSortIndices.indices[visibleIndex]);
    }
    return true;
}
#endif

void main()
{
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= u_PC.pointCount)
        return;

    if (coverageReleaseResolvePhase())
    {
        coverageTextureResolveTexel(idx);
        return;
    }

    // Single-asset Ordered CLOD can directly consume the physically sorted
    // packed-source prefix. Baseline and multi-asset fallback use the selected
    // source table to preserve draw/source indirection.
#if USE_DIRECT_PREFIX
    const uint sourceIndex = idx;
    const uint drawIndex = 0u;
    const float lodWeight = 1.0;
#else
    const GeneralGaussianSplatSelectedSource selection = s_GeneralGaussianSplatSelectedSources.sources[idx];
    if ((selection.flags & GENERAL_GAUSSIAN_SPLAT_SELECTED_FLAG_INVALID) != 0u)
        return;
    const uint sourceIndex = selection.sourceIndex;
    const uint drawIndex = selection.drawIndex;
    // Zero defaults to full opacity so non-CLOD/baseline entries can use the
    // same packed structure without needing an extra initialization path.
    const float lodWeight = selection.packedWeight == 0u ? 1.0 : clamp(uintBitsToFloat(selection.packedWeight), 0.0, 1.0);
#endif
    const bool coverageAccumulation = coverageReleaseAccumulationPhase();
    const bool ecsptDiagnostics = ecsptCounterDiagnosticsEnabled();
    if (!coverageAccumulation && ecsptDiagnostics)
        ecsptCounterAdd(ECSPT_COUNTER_TOTAL_CANDIDATES_SEEN, 1u);

    {
        const GeneralGaussianSplatPackedSource src = s_GeneralGaussianSplatPackedSources.points[sourceIndex];
        const GeneralGaussianSplatDrawRecord draw  = s_GeneralGaussianSplatDraws.draws[drawIndex];
        const vec3 localPos                        = decodeGeneralGaussianSplatPosition(src);
        const mat4 model                           = draw.model;
        const mat3 modelLinear                     = mat3(model);
        const vec3 modelTranslation                = model[3].xyz;
        const vec3 worldPos                        = (model * vec4(localPos, 1.0)).xyz;
        if (coverageAccumulation)
        {
            if (coverageTextureFloorEnabled())
                coverageTextureAccumulateCandidate(src, draw, worldPos, lodWeight, u_Camera);
            else
                coverageReleaseAccumulateCandidate(src, draw, worldPos, lodWeight, u_Camera);
            return;
        }
        const bool accumulateCoverageTextureCurrent =
            coverageTextureFloorEnabled() && coverageTextureUpdateThisFrame();
        const EyePreprocessResult eye0 =
            preprocessEye(src,
                          draw,
                          localPos,
                          worldPos,
                          modelLinear,
                          modelTranslation,
                          lodWeight,
                          1.0,
                          sourceIndex,
                          idx,
                          u_Camera,
                          accumulateCoverageTextureCurrent);
#if USE_MULTIVIEW
        const EyePreprocessResult eye1 =
            preprocessEye(src,
                          draw,
                          localPos,
                          worldPos,
                          modelLinear,
                          modelTranslation,
                          lodWeight,
                          1.0,
                          sourceIndex,
                          idx,
                          u_StereoCameraBlock.cameras[1],
                          false);
#else
        const EyePreprocessResult eye1 = eye0;
#endif
        const bool visible = eye0.visible || eye1.visible;
        if (!visible)
            return;
        if (ecsptDiagnostics)
            ecsptCounterAdd(ECSPT_COUNTER_EFFECTIVE_VISIBLE_AFTER_ECSPT_COUNT, 1u);

        float sortDepth = eye0.visible ? eye0.sortDepth : 0.0;
#if USE_MULTIVIEW
        if (eye1.visible)
            sortDepth = max(sortDepth, eye1.sortDepth);
#endif

#if USE_FOVEATED_LAYER_OUTPUT
        const uint activeLayer = u_PC.foveatedLayerParams.x;
        const float eccentricityDegrees = combinedFoveatedEccentricity(eye0, eye1);
        const float foveatedClodLevel = foveatedClodLevelForEccentricity(eccentricityDegrees);
        const float layerSelectionWeight =
            foveatedSelectionWeight(sourceIndex, idx, foveatedClodLevel, eccentricityDegrees);
        if (layerSelectionWeight <= 0.0)
            return;

        uint emittedTileInstances = 0u;
        if (activeLayer >= 1u && activeLayer <= 3u)
        {
            emittedTileInstances += writeFoveatedLayerSplat(activeLayer - 1u,
                                                            eye0,
                                                            eye1,
                                                            sourceIndex,
                                                            drawIndex,
                                                            layerSelectionWeight,
                                                            sortDepth) ? 1u : 0u;
        }
        else
        {
            if (isInsideFoveatedLayerEccentricity(0u, eccentricityDegrees))
                emittedTileInstances += writeFoveatedLayerSplat(0u,
                                                                eye0,
                                                                eye1,
                                                                sourceIndex,
                                                                drawIndex,
                                                                layerSelectionWeight,
                                                                sortDepth) ? 1u : 0u;
            if (isInsideFoveatedLayerEccentricity(1u, eccentricityDegrees))
                emittedTileInstances += writeFoveatedLayerSplat(1u,
                                                                eye0,
                                                                eye1,
                                                                sourceIndex,
                                                                drawIndex,
                                                                layerSelectionWeight,
                                                                sortDepth) ? 1u : 0u;
            if (isInsideFoveatedLayerEccentricity(2u, eccentricityDegrees))
                emittedTileInstances += writeFoveatedLayerSplat(2u,
                                                                eye0,
                                                                eye1,
                                                                sourceIndex,
                                                                drawIndex,
                                                                layerSelectionWeight,
                                                                sortDepth) ? 1u : 0u;
        }
        if (ecsptDiagnostics && emittedTileInstances > 0u)
        {
            ecsptCounterAdd(ECSPT_COUNTER_VISIBLE_INSTANT_COUNT, 1u);
            ecsptCounterAdd(ECSPT_COUNTER_TILE_INSTANCE_COUNT, emittedTileInstances);
        }
#else
        const uint visibleIndex = atomicAdd(s_GeneralGaussianSplatVisibleCount.visibleCount, 1u);
        if (visibleIndex >= u_PC.maxVisibleSplats)
        {
            atomicMin(s_GeneralGaussianSplatVisibleCount.visibleCount, u_PC.maxVisibleSplats);
            return;
        }
        if (ecsptDiagnostics)
        {
            const uint combinedEcsptFlags = eye0.eccStochasticFlags | eye1.eccStochasticFlags;
            const bool released =
                (combinedEcsptFlags & ECSPT_EVAL_COVERAGE_RELEASE_HELD) != 0u;
            const bool useEye1Cost =
                eye1.visible && (!eye0.visible || eye1.tileDuplicateProxy > eye0.tileDuplicateProxy);
            const vec2 costCenterNdc = useEye1Cost ? eye1.centerNdc : eye0.centerNdc;
            const uint costTileDuplicateProxy =
                useEye1Cost ? eye1.tileDuplicateProxy : eye0.tileDuplicateProxy;
            const float costProjectedRadiusPx =
                useEye1Cost ? eye1.projectedRadiusPx : eye0.projectedRadiusPx;
            recordReleaseCostDiagnostics(costCenterNdc,
                                         released,
                                         costTileDuplicateProxy,
                                         costProjectedRadiusPx);
        }

        writeVisibleSplat(eye0,
                          eye1,
                          sourceIndex,
                          drawIndex,
                          visibleIndex,
                          sortDepth,
                          s_GeneralGaussianSplatVisibleSplats.splats[visibleIndex],
                          s_GeneralGaussianSplatSortKeys.keys[visibleIndex],
                          s_GeneralGaussianSplatSortIndices.indices[visibleIndex]);
        if (ecsptDiagnostics)
        {
            ecsptCounterAdd(ECSPT_COUNTER_VISIBLE_INSTANT_COUNT, 1u);
            ecsptCounterAdd(ECSPT_COUNTER_TILE_INSTANCE_COUNT, 1u);
        }
#endif

    }
}

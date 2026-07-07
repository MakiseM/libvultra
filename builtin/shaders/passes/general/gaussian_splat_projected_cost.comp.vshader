[vshader]
language = glsl
version = 460

[keywords]
USE_DIRECT_PREFIX : bool permute

[comp]
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_DRAW_BUFFER
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_PACKED_SOURCE_BUFFER
#if !USE_DIRECT_PREFIX
#define VULTRA_DECLARE_GENERAL_GAUSSIAN_SPLAT_SELECTED_SOURCE_BUFFER
#endif
#include "include/common/gpu_scene.glsl"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

const float A4_TILE_SIZE_PX = 16.0;

struct A4ProjectedCostCamera
{
    mat4 view;
    mat4 projection;
    vec4 resolution;
    vec4 params;
};

layout(set = 0, binding = 0, std140) uniform A4ProjectedCostCameraBlock
{
    A4ProjectedCostCamera cameras[2];
} u_A4Cameras;

struct A4ProjectedCostChunkRecord
{
    uvec4 costAndTiles;
    uvec4 rangeAndArea;
};

layout(set = 0, binding = 47, std430) buffer A4ProjectedCostChunkBuffer
{
    A4ProjectedCostChunkRecord chunks[];
} s_A4Chunks;

layout(push_constant) uniform A4ProjectedCostPushConstants
{
    uint sourceCapacity;
    uint chunkSize;
    uint chunkCount;
    uint cameraCount;
    vec4 gazeAndRings;
} u_PC;

mat3 buildA4Jacobian(const vec3 camspace, const vec2 focal)
{
    float z = camspace.z;
    if (abs(z) < 1e-4)
        z = z < 0.0 ? -1e-4 : 1e-4;
    return mat3(vec3(focal.x / z, 0.0, -(focal.x * camspace.x) / (z * z)),
                vec3(0.0, -focal.y / z, (focal.y * camspace.y) / (z * z)),
                vec3(0.0));
}

float a4EccentricityDegrees(const vec2 centerNdc, const A4ProjectedCostCamera camera)
{
    const vec2 tanHalfFov = vec2(1.0 / max(abs(camera.projection[0][0]), 1e-5),
                                 1.0 / max(abs(camera.projection[1][1]), 1e-5));
    const float projectionYSign = camera.projection[1][1] < 0.0 ? -1.0 : 1.0;
    const vec2 gazeNdc = vec2(u_PC.gazeAndRings.x * 2.0 - 1.0,
                              projectionYSign * (1.0 - u_PC.gazeAndRings.y * 2.0));
    const vec2 deltaTan = (centerNdc - gazeNdc) * max(tanHalfFov, vec2(1e-5));
    return degrees(atan(length(deltaTan)));
}

uint a4CoverageSector(const float eccentricityDegrees)
{
    const float foveaDegrees = max(u_PC.gazeAndRings.z, 0.0);
    const float midDegrees = max(u_PC.gazeAndRings.w, foveaDegrees);
    if (eccentricityDegrees <= foveaDegrees)
        return 0u;
    if (eccentricityDegrees <= midDegrees)
        return 1u;
    return 2u;
}

uint a4TileIntersectionsForCamera(const GeneralGaussianSplatPackedSource src,
                                  const GeneralGaussianSplatDrawRecord draw,
                                  const vec3 localPos,
                                  const vec3 worldPos,
                                  const mat3 sigmaWorld,
                                  const A4ProjectedCostCamera camera,
                                  inout float projectedAreaPx)
{
    const vec4 posView = camera.view * vec4(worldPos, 1.0);
    const vec4 posClip = camera.projection * posView;
    if (posClip.w <= 1e-5)
        return 0u;

    const vec3 centerNdc = posClip.xyz / posClip.w;
    const float bounds = 1.2 * posClip.w;
    if (centerNdc.z <= 0.0 || centerNdc.z >= 1.0)
        return 0u;
    if (posClip.x < -bounds || posClip.x > bounds || posClip.y < -bounds || posClip.y > bounds)
        return 0u;

    const vec2 viewport = camera.resolution.xy;
    const vec2 focal = 0.5 * vec2(abs(camera.projection[0][0]) * viewport.x,
                                  abs(camera.projection[1][1]) * viewport.y);
    const mat3 jacobian = buildA4Jacobian(posView.xyz, focal);
    const mat3 viewLinear = mat3(camera.view);
    const mat3 worldToView = transpose(mat3(viewLinear[0], viewLinear[1], viewLinear[2]));
    const mat3 transform = worldToView * jacobian;
    const mat3 cov = transpose(transform) * sigmaWorld * transform;

    const float kernelSize = max(draw.params0.x, 1e-4);
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
    const vec2 v1 = sqrt(2.0 * lambda1) * diagonalVector * cutoffScale;
    const vec2 v2 = sqrt(2.0 * lambda2) * vec2(diagonalVector.y, -diagonalVector.x) * cutoffScale;
    const vec2 halfExtent = abs(v1) + abs(v2);
    if (!(halfExtent.x >= 0.0) || !(halfExtent.y >= 0.0) ||
        halfExtent.x > 1.0e20 || halfExtent.y > 1.0e20)
        return 0u;

    const vec2 centerPx = vec2((centerNdc.x * 0.5 + 0.5) * viewport.x,
                               (centerNdc.y * 0.5 + 0.5) * viewport.y);
    const float left = clamp(centerPx.x - halfExtent.x, 0.0, viewport.x);
    const float right = clamp(centerPx.x + halfExtent.x, 0.0, viewport.x);
    const float top = clamp(centerPx.y - halfExtent.y, 0.0, viewport.y);
    const float bottom = clamp(centerPx.y + halfExtent.y, 0.0, viewport.y);
    if (right <= left || bottom <= top)
        return 0u;

    const uint tileColumns = max(1u, uint(ceil(viewport.x / A4_TILE_SIZE_PX)));
    const uint tileRows = max(1u, uint(ceil(viewport.y / A4_TILE_SIZE_PX)));
    const uint tileX0 = min(tileColumns, uint(floor(left / A4_TILE_SIZE_PX)));
    const uint tileY0 = min(tileRows, uint(floor(top / A4_TILE_SIZE_PX)));
    const uint tileX1 = min(tileColumns, uint(ceil(right / A4_TILE_SIZE_PX)));
    const uint tileY1 = min(tileRows, uint(ceil(bottom / A4_TILE_SIZE_PX)));
    if (tileX1 <= tileX0 || tileY1 <= tileY0)
        return 0u;

    projectedAreaPx += max((right - left) * (bottom - top), 0.0);
    return (tileX1 - tileX0) * (tileY1 - tileY0);
}

void main()
{
    const uint chunkIndex = gl_GlobalInvocationID.x;
    if (chunkIndex >= u_PC.chunkCount)
        return;

    const uint beginRank = chunkIndex * u_PC.chunkSize;
    const uint endRank = min(u_PC.sourceCapacity, beginRank + u_PC.chunkSize);
    uint chunkCost = 0u;
    uint chunkTiles = 0u;
    float chunkArea = 0.0;

    for (uint rank = beginRank; rank < endRank; ++rank)
    {
#if USE_DIRECT_PREFIX
        const uint sourceIndex = rank;
        const uint drawIndex = 0u;
#else
        const GeneralGaussianSplatSelectedSource selection = s_GeneralGaussianSplatSelectedSources.sources[rank];
        if ((selection.flags & GENERAL_GAUSSIAN_SPLAT_SELECTED_FLAG_INVALID) != 0u)
            continue;
        const uint sourceIndex = selection.sourceIndex;
        const uint drawIndex = selection.drawIndex;
#endif
        const GeneralGaussianSplatPackedSource src = s_GeneralGaussianSplatPackedSources.points[sourceIndex];
        const GeneralGaussianSplatDrawRecord draw = s_GeneralGaussianSplatDraws.draws[drawIndex];
        const vec3 localPos = decodeGeneralGaussianSplatPosition(src);
        const mat4 model = draw.model;
        const mat3 modelLinear = mat3(model);
        const vec3 worldPos = (model * vec4(localPos, 1.0)).xyz;
        const mat3 sigmaLocal = decodeGeneralGaussianSplatCovariance(src);
        const mat3 sigmaWorld = modelLinear * sigmaLocal * transpose(modelLinear);

        uint sampleTiles = 0u;
        float sampleArea = 0.0;
        const uint cameraCount = min(u_PC.cameraCount, 2u);
        for (uint cameraIndex = 0u; cameraIndex < cameraCount; ++cameraIndex)
        {
            sampleTiles += a4TileIntersectionsForCamera(src,
                                                        draw,
                                                        localPos,
                                                        worldPos,
                                                        sigmaWorld,
                                                        u_A4Cameras.cameras[cameraIndex],
                                                        sampleArea);
        }
        chunkTiles += sampleTiles;
        chunkCost += max(1u, sampleTiles);
        chunkArea += sampleArea;
    }

    s_A4Chunks.chunks[chunkIndex].costAndTiles = uvec4(chunkCost, 0u, chunkTiles, 0u);
    s_A4Chunks.chunks[chunkIndex].rangeAndArea = uvec4(beginRank, endRank, floatBitsToUint(chunkArea), 0u);
}

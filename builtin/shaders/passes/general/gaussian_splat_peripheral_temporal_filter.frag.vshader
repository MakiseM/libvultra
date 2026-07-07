[vshader]
language = glsl
version = 460

[keywords]
USE_MULTIVIEW : bool permute

[frag]
#include "include/common/gaussian_splat_foveated.glsl"

#if USE_MULTIVIEW && !PLATFORM_WEBGPU
#extension GL_EXT_multiview : require
#endif

layout (location = 0) in vec2 v_TexCoord;
layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 HistoryColor;

#if USE_MULTIVIEW && !PLATFORM_WEBGPU
layout (set = 3, binding = 0) uniform sampler2DArray t_Source;
layout (set = 3, binding = 1) uniform sampler2DArray t_History;
#else
layout (set = 3, binding = 0) uniform sampler2D t_Source;
layout (set = 3, binding = 1) uniform sampler2D t_History;
#endif

layout(push_constant) uniform TemporalFilterPushConstants
{
    vec4 targetSize; // xy: output size in pixels
    vec4 temporalGazeAndRings; // xy: gaze uv, z: history start deg, w: full history deg
    vec4 temporalParams; // xy: tan half fov, z: projection y sign, w: rejection threshold
    vec4 temporalParams2; // x: clamp radius, y: lambda scale, z: history valid, w: unused
} u_PC;

float luminance(vec3 color)
{
    return dot(max(color, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
}

vec4 sampleSource(vec2 uv)
{
#if USE_MULTIVIEW && !PLATFORM_WEBGPU
    return texture(t_Source, vec3(uv, float(gl_ViewIndex)));
#else
    return texture(t_Source, uv);
#endif
}

vec4 sampleHistory(vec2 uv)
{
#if USE_MULTIVIEW && !PLATFORM_WEBGPU
    return texture(t_History, vec3(uv, float(gl_ViewIndex)));
#else
    return texture(t_History, uv);
#endif
}

void main()
{
    vec2 sampleUv = v_TexCoord;
#if PLATFORM_WEBGPU
    sampleUv.y = 1.0 - sampleUv.y;
#endif

    const vec4 source = sampleSource(sampleUv);
    const vec2 targetSize = max(u_PC.targetSize.xy, vec2(1.0));
    const vec2 viewportUv = (gl_FragCoord.xy - vec2(0.5)) / targetSize;
    const float eccentricityDegrees =
        gaussianFoveatedEccentricityDegreesFromUv(
            viewportUv,
            u_PC.temporalGazeAndRings.xy,
            u_PC.temporalParams.xy,
            u_PC.temporalParams.z == 0.0 ? 1.0 : u_PC.temporalParams.z);

    float lambda =
        smoothstep(u_PC.temporalGazeAndRings.z,
                   max(u_PC.temporalGazeAndRings.w, u_PC.temporalGazeAndRings.z + 1.0),
                   eccentricityDegrees) *
        clamp(u_PC.temporalParams2.y, 0.0, 1.0);
    if (u_PC.temporalParams2.z <= 0.5)
        lambda = 0.0;

    const vec4 history = sampleHistory(sampleUv);
    const float eps = 1e-4;
    const float currentA = clamp(source.a, 0.0, 1.0);
    const float historyA = clamp(history.a, 0.0, 1.0);
    const vec3 currentO = currentA > eps ? source.rgb / max(currentA, eps) : source.rgb;
    const vec3 historyO = historyA > eps ? history.rgb / max(historyA, eps) : history.rgb;

    const float rejectThreshold = max(u_PC.temporalParams.w, 1e-4);
    const float historyDifference = abs(luminance(currentO) - luminance(historyO));
    const float rejectWeight = 1.0 - smoothstep(rejectThreshold, rejectThreshold * 2.0, historyDifference);
    lambda *= rejectWeight;

    const float clampRadius = max(u_PC.temporalParams2.x, 0.0);
    const vec3 clampedHistoryO = clamp(historyO, currentO - vec3(clampRadius), currentO + vec3(clampRadius));
    const vec3 clampedHistoryP = clampedHistoryO * historyA;

    const vec3 premultiplied = mix(source.rgb, clampedHistoryP, lambda);
    const float alpha = mix(currentA, historyA, lambda);
    const vec3 normalized = alpha > eps ? premultiplied / max(alpha, eps) : premultiplied;

    HistoryColor = vec4(premultiplied, alpha);
    FragColor = vec4(mix(source.rgb, normalized, lambda), source.a);
}

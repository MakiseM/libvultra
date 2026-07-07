vec2 gaussianFoveatedUvToNdc(const vec2 uv, const float projectionYSign)
{
    return vec2(uv.x * 2.0 - 1.0, projectionYSign * (1.0 - uv.y * 2.0));
}

float gaussianFoveatedEccentricityDegreesFromNdc(const vec2 centerNdc,
                                                 const vec2 gazeUv,
                                                 const vec2 tanHalfFov,
                                                 const float projectionYSign)
{
    const vec2 deltaTan = (centerNdc - gaussianFoveatedUvToNdc(gazeUv, projectionYSign)) * max(tanHalfFov, vec2(1e-5));
    return degrees(atan(length(deltaTan)));
}

float gaussianFoveatedEccentricityDegreesFromUv(const vec2 viewportUv,
                                                const vec2 gazeUv,
                                                const vec2 tanHalfFov,
                                                const float projectionYSign)
{
    return gaussianFoveatedEccentricityDegreesFromNdc(
        gaussianFoveatedUvToNdc(viewportUv, projectionYSign),
        gazeUv,
        tanHalfFov,
        projectionYSign);
}

vec2 gaussianFoveatedRingBlend(const float eccentricityDegrees,
                               const vec2 ringDegrees,
                               const float transitionDegrees)
{
    const float foveaDegrees = max(ringDegrees.x, 0.0);
    const float midDegrees = max(ringDegrees.y, foveaDegrees);
    const float halfTransition = max(transitionDegrees, 0.0) * 0.5;
    return vec2(smoothstep(max(foveaDegrees - halfTransition, 0.0),
                           foveaDegrees + halfTransition,
                           eccentricityDegrees),
                smoothstep(max(midDegrees - halfTransition, 0.0),
                           midDegrees + halfTransition,
                           eccentricityDegrees));
}

float gaussianFoveatedSafeMidRatio(const vec3 levels)
{
    const float denominator = max(levels.x - levels.z, 1e-5);
    return clamp((levels.y - levels.z) / denominator, 1e-3, 0.999);
}

float gaussianFoveatedContinuousWeight(const float eccentricityDegrees,
                                       const vec2 ringDegrees,
                                       const vec3 levels,
                                       const uint distribution)
{
    const float foveaDegrees = max(ringDegrees.x, 0.0);
    const float midDegrees = max(ringDegrees.y, foveaDegrees + 1e-3);
    const float offset = max(eccentricityDegrees - foveaDegrees, 0.0);
    const float midOffset = max(midDegrees - foveaDegrees, 1e-3);
    const float ratio = gaussianFoveatedSafeMidRatio(levels);

    if (distribution == 2u)
    {
        const float sigma = midOffset / sqrt(max(-2.0 * log(ratio), 1e-4));
        const float x = offset / max(sigma, 1e-4);
        return exp(-0.5 * x * x);
    }

    if (distribution == 3u)
    {
        const float lambda = -log(ratio) / midOffset;
        return exp(-lambda * offset);
    }

    if (distribution == 4u)
    {
        const float power = 2.0;
        const float scale = midOffset / pow(max(1.0 / ratio - 1.0, 1e-4), 1.0 / power);
        const float x = offset / max(scale, 1e-4);
        return 1.0 / (1.0 + pow(x, power));
    }

    if (distribution == 5u)
    {
        const float scale = midOffset / max(exp(1.0 / ratio - 1.0) - 1.0, 1e-4);
        return 1.0 / (1.0 + log(1.0 + offset / max(scale, 1e-4)));
    }

    if (distribution == 8u)
    {
        const float e2 = ratio * midOffset / max(1.0 - ratio, 1e-4);
        return e2 / max(offset + e2, 1e-4);
    }

    if (distribution == 9u)
    {
        const float densityFloor = 0.04;
        const float conePower = 2.0;
        const float beta = 1.0;
        const float densityMid = clamp(pow(ratio, 1.0 / beta), densityFloor + 1e-4, 1.0);
        const float thetaC =
            midOffset / pow(max((1.0 - densityFloor) / (densityMid - densityFloor) - 1.0, 1e-4),
                            1.0 / conePower);
        const float density =
            densityFloor +
            (1.0 - densityFloor) /
                (1.0 + pow(offset / max(thetaC, 1e-4), conePower));
        return pow(clamp(density, 0.0, 1.0), beta);
    }

    return 0.0;
}

float gaussianFoveatedContinuousSchedulerLevel(const float eccentricityDegrees,
                                               const float budgetLevel,
                                               const vec4 continuousParams)
{
    const float theta0 = max(continuousParams.x, 1e-4);
    const float alpha = max(continuousParams.y, 0.0);
    const float minLevel = clamp(continuousParams.z, 0.0, clamp(budgetLevel, 0.0, 1.0));
    const float budget = max(clamp(budgetLevel, 0.0, 1.0), minLevel);
    const float falloff = pow(1.0 + max(eccentricityDegrees, 0.0) / theta0, alpha);
    return minLevel + (budget - minLevel) / max(falloff, 1e-5);
}

float gaussianFoveatedFoveaProtectedContinuousLevel(const float eccentricityDegrees,
                                                    const vec2 ringDegrees,
                                                    const vec3 levels,
                                                    const vec4 continuousParams)
{
    const float protectedDegrees = max(ringDegrees.x, 0.0);
    const float theta0 = max(continuousParams.x, 1e-4);
    const float alpha = max(continuousParams.y, 0.0);
    const float foveaLevel = clamp(max(levels.x, 0.95), 0.0, 1.0);
    const float peripheryLevel = clamp(continuousParams.z, 0.0, foveaLevel);
    const float offset = max(eccentricityDegrees - protectedDegrees, 0.0);
    const float falloff = 1.0 / max(pow(1.0 + offset / theta0, alpha), 1e-5);
    return peripheryLevel + (foveaLevel - peripheryLevel) * clamp(falloff, 0.0, 1.0);
}

float gaussianFoveatedClodLevel(const float eccentricityDegrees,
                                const vec2 ringDegrees,
                                const vec3 ringLevels,
                                const float transitionDegrees,
                                const uint distribution,
                                const vec4 continuousParams)
{
    const float foveaDegrees = max(ringDegrees.x, 0.0);
    const float midDegrees = max(ringDegrees.y, foveaDegrees);
    const vec3 levels = clamp(ringLevels, vec3(0.0), vec3(1.0));

    if (distribution == 0u || (distribution != 6u && distribution != 7u && transitionDegrees <= 1e-4))
    {
        if (eccentricityDegrees <= foveaDegrees)
            return levels.x;
        if (eccentricityDegrees <= midDegrees)
            return levels.y;
        return levels.z;
    }

    if (distribution == 6u)
        return gaussianFoveatedContinuousSchedulerLevel(eccentricityDegrees, levels.x, continuousParams);

    if (distribution == 7u)
        return gaussianFoveatedFoveaProtectedContinuousLevel(
            eccentricityDegrees,
            ringDegrees,
            levels,
            continuousParams);

    if (distribution >= 2u)
    {
        const float weight = gaussianFoveatedContinuousWeight(eccentricityDegrees, ringDegrees, levels, distribution);
        return levels.z + (levels.x - levels.z) * clamp(weight, 0.0, 1.0);
    }

    const vec2 blend = gaussianFoveatedRingBlend(eccentricityDegrees, ringDegrees, transitionDegrees);
    return mix(mix(levels.x, levels.y, blend.x), levels.z, blend.y);
}

float gaussianFoveatedClodLevel(const float eccentricityDegrees,
                                const vec2 ringDegrees,
                                const vec3 ringLevels,
                                const float transitionDegrees,
                                const uint distribution)
{
    return gaussianFoveatedClodLevel(eccentricityDegrees,
                                     ringDegrees,
                                     ringLevels,
                                     transitionDegrees,
                                     distribution,
                                     vec4(max(ringDegrees.y, 1.0), 2.0, ringLevels.z, 0.0));
}

bool gaussianFoveatedLayerContains(const uint layer,
                                   const float eccentricityDegrees,
                                   const vec2 ringDegrees,
                                   const float transitionDegrees)
{
    const float foveaDegrees = max(ringDegrees.x, 0.0);
    const float midDegrees = max(ringDegrees.y, foveaDegrees);
    const float halfTransition = max(transitionDegrees, 0.0) * 0.5;

    if (layer == 0u)
        return eccentricityDegrees <= foveaDegrees + halfTransition;
    if (layer == 1u)
        return eccentricityDegrees >= max(foveaDegrees - halfTransition, 0.0) &&
               eccentricityDegrees <= midDegrees + halfTransition;
    return eccentricityDegrees >= max(midDegrees - halfTransition, 0.0);
}

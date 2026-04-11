static const float c_MaxIndirectRadiance = 10;
/*
[IndirectLightingMode::Brdf/ReStirGI]
*/
[shader("raygeneration")]
void RayGen()
{
    uint2 GlobalIndex = DispatchRaysIndex().xy;
    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, g_Const.runtimeParams.activeCheckerboardField);
    if (any(pixelPosition > int2(g_Const.view.viewportSize)))
        return;

    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(GlobalIndex, g_Const.runtimeParams.frameIndex, RTXDI_SECONDARY_DI_GENERATE_INITIAL_SAMPLES_RANDOM_SEED);
    RTXDI_RandomSamplerState tileRng = RTXDI_InitRandomSampler(GlobalIndex / RTXDI_TILE_SIZE_IN_PIXELS, g_Const.runtimeParams.frameIndex, RTXDI_SECONDARY_DI_GENERATE_INITIAL_SAMPLES_RANDOM_SEED);

    const uint gbufferIndex = RTXDI_ReservoirPositionToPointer(g_Const.restirDI.reservoirBufferParams, GlobalIndex, 0);

    RAB_Surface primarySurface = RAB_GetGBufferSurface(pixelPosition, false);

    SecondaryGBufferData secondaryGBufferData = u_SecondaryGBuffer[gbufferIndex];

    const float3 throughput = Unpack_R16G16B16A16_FLOAT(secondaryGBufferData.throughputAndFlags).rgb;
    const uint secondaryFlags = secondaryGBufferData.throughputAndFlags.y >> 16;
    const bool isValidSecondarySurface = any(throughput != 0);
    const bool isSpecularRay = (secondaryFlags & kSecondaryGBuffer_IsSpecularRay) != 0;
    const bool isDeltaSurface = (secondaryFlags & kSecondaryGBuffer_IsDeltaSurface) != 0;
    const bool isEnvironmentMap = (secondaryFlags & kSecondaryGBuffer_IsEnvironmentMap) != 0;

    RAB_Surface secondarySurface = RAB_EmptySurface();

    // Unpack the G-buffer data
    secondarySurface.worldPos = secondaryGBufferData.worldPos;
    secondarySurface.viewDepth = 1.0; // doesn't matter
    secondarySurface.normal = octToNdirUnorm32(secondaryGBufferData.normal);
    secondarySurface.geoNormal = secondarySurface.normal;
    secondarySurface.material.diffuseAlbedo = Unpack_R11G11B10_UFLOAT(secondaryGBufferData.diffuseAlbedo);
    float4 specularRough = Unpack_R8G8B8A8_Gamma_UFLOAT(secondaryGBufferData.specularAndRoughness);
    secondarySurface.material.specularF0 = specularRough.rgb;
    secondarySurface.material.roughness = specularRough.a;
    secondarySurface.diffuseProbability = getSurfaceDiffuseProbability(secondarySurface);
    secondarySurface.viewDir = normalize(primarySurface.worldPos - secondarySurface.worldPos);

    //-----------------------------------------------------------------------------------------------//
    //-----------------------------------------------------------------------------------------------//
    //-----------------------------------------------------------------------------------------------//
    
    float3 radiance = secondaryGBufferData.emission;

    // Shade the secondary surface.
    if (isValidSecondarySurface && !isEnvironmentMap)
    {
        RAB_LightSample lightSample;
        RTXDI_DIReservoir reservoir = RTXDI_SampleLightsForSurface(
            rng, 
            tileRng, 
            secondarySurface,
            g_Const.brdfPT.secondarySurfaceReSTIRDIParams.initialSamplingParams, 
            g_Const.lightBufferParams,
            /*out*/lightSample);

        float3 indirectDiffuse = 0;
        float3 indirectSpecular = 0;
        float lightDistance = 0;
        ShadeSurfaceWithLightSample(
            /*inout*/reservoir, 
            secondarySurface, 
            g_Const.restirDI.shadingParams, 
            lightSample, 
            /* previousFrameTLAS = */ false,
            /* enableVisibilityReuse = */ false, 
            /* enableVisibilityShortcut */ false, 
            /*out*/indirectDiffuse, 
            /*out*/indirectSpecular, 
            /*out*/lightDistance);

        radiance += indirectDiffuse * secondarySurface.material.diffuseAlbedo + indirectSpecular;
        
        // Firefly suppression
        float indirectLuminance = calcLuminance(radiance);
        if (indirectLuminance > c_MaxIndirectRadiance)
            radiance *= c_MaxIndirectRadiance / indirectLuminance;
    }

    //-----------------------------------------------------------------------------------------------//
    //-----------------------------------------------------------------------------------------------//
    //-----------------------------------------------------------------------------------------------//

    bool outputShadingResult = true;
    if (g_Const.brdfPT.enableReSTIRGI)
    {
        RTXDI_GIReservoir reservoir = RTXDI_EmptyGIReservoir();

        // For delta reflection rays, just output the shading result in this shader
        // and don't include it into ReSTIR GI reservoirs.
        outputShadingResult = isSpecularRay && isDeltaSurface;

        if (isValidSecondarySurface && !outputShadingResult)
        {
            // This pixel has a valid indirect sample so it stores information as an initial GI reservoir.
            reservoir = RTXDI_GIReservoir{
                .position = secondarySurface.worldPos,
                .normal = secondarySurface.normal,
                .radiance = radiance,
                .weightSum = secondaryGBufferData.pdf > 0.0 ? 
                            1.0 / secondaryGBufferData.pdf : 0.0,
                .M = 1;
                .age = 0;
            };
        }

        uint2 reservoirPosition = RTXDI_PixelPosToReservoirPos(
            pixelPosition, 
            g_Const.runtimeParams.activeCheckerboardField);

        uint pointer = RTXDI_ReservoirPositionToPointer(
            g_Const.restirGI.reservoirBufferParams, 
            reservoirPosition, 
            g_Const.restirGI.bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex);

        u_GIReservoirs[pointer] = RTXDI_PackGIReservoir(reservoir);

        // Save the initial sample radiance for MIS in the final shading pass
        secondaryGBufferData.emission = !outputShadingResult ? radiance : 0;
        u_SecondaryGBuffer[gbufferIndex] = secondaryGBufferData;
    }

    if (outputShadingResult) // !g_Const.brdfPT.enableReSTIRGI || (isSpecularRay && isDeltaSurface)
    {
        float3 diffuse = 0;
        float3 specular = 0;
        if (!isDeltaSurface)
        {
            diffuse = !isSpecularRay ? radiance * throughput.rgb : 0.0; // throughput.rgb == payload.throughput * BRDF_over_PDF
            specular = isSpecularRay ? radiance * throughput.rgb : 0.0;
        }

        u_DiffuseLighting[pixelPosition].rgb += diffuse;
        u_SpecularLighting[pixelPosition].rgb += specular;
    }
}



bool ShadeSurfaceWithLightSample(
    inout RTXDI_DIReservoir reservoir,
    RAB_Surface surface,
    RTXDI_ShadingParameters shadingParams,
    RAB_LightSample lightSample,
    bool previousFrameTLAS,
    // bool enableVisibilityReuse,
    bool enableVisibilityShortcut,
    out float3 diffuse,
    out float3 specular,
    out float lightDistance)
{
    diffuse = float3(0.0f, 0.0f, 0.0f);
    specular = float3(0.0f, 0.0f, 0.0f);
    lightDistance = 0.0f;

    if (lightSample.solidAnglePdf <= 0)
        return false;

    bool needToStore = false;
    // TODO: This final visibility check always seems to destroy the reservoir.
    if (shadingParams.enableFinalVisibility)
    {
        float3 visibility = GetFinalVisibility(
            SceneBVH, 
            surface, 
            lightSample.position);
        
        RTXDI_StoreVisibilityInDIReservoir(reservoir, visibility, enableVisibilityShortcut);
        needToStore = true;

        lightSample.radiance *= visibility;
    }

    lightSample.radiance *= RTXDI_GetDIReservoirInvPdf(reservoir) / lightSample.solidAnglePdf;

    if (any(lightSample.radiance > 0))
    {
        SplitBrdf brdf = EvaluateBrdf(surface, lightSample.position);

        diffuse = brdf.demodulatedDiffuse * lightSample.radiance;
        specular = brdf.specular * lightSample.radiance;

        lightDistance = length(lightSample.position - surface.worldPos);
    }

    return needToStore;
}


bool ShadeSurfaceWithLightSample(
    inout RTXDI_DIReservoir reservoir,
    RAB_Surface surface,
    RTXDI_ShadingParameters shadingParams,
    RAB_LightSample lightSample,
    bool previousFrameTLAS,
    bool enableVisibilityReuse,
    bool enableVisibilityShortcut,
    out float3 diffuse,
    out float3 specular,
    out float lightDistance)
{
    diffuse = float3(0.0f, 0.0f, 0.0f);
    specular = float3(0.0f, 0.0f, 0.0f);
    lightDistance = 0.0f;

    if (lightSample.solidAnglePdf <= 0)
        return false;

    bool needToStore = false;
    // TODO: This final visibility check always seems to destroy the reservoir.
    if (shadingParams.enableFinalVisibility)
    {
        float3 visibility = float3(0.0f, 0.0f, 0.0f);
        bool visibilityReused = false;

        if (shadingParams.reuseFinalVisibility && enableVisibilityReuse)
        {
            RTXDI_VisibilityReuseParameters rparams;
            rparams.maxAge = shadingParams.finalVisibilityMaxAge;
            rparams.maxDistance = shadingParams.finalVisibilityMaxDistance;

            visibilityReused = RTXDI_GetDIReservoirVisibility(reservoir, rparams, visibility);
        }

        if (!visibilityReused)
        {
            if (previousFrameTLAS && g_Const.enablePreviousTLAS)
                visibility = GetFinalVisibility(PrevSceneBVH, surface, lightSample.position);
            else
                visibility = GetFinalVisibility(SceneBVH, surface, lightSample.position);
            RTXDI_StoreVisibilityInDIReservoir(reservoir, visibility, enableVisibilityShortcut);
            needToStore = true;
        }

        lightSample.radiance *= visibility;
    }

    lightSample.radiance *= RTXDI_GetDIReservoirInvPdf(reservoir) / lightSample.solidAnglePdf;

    if (any(lightSample.radiance > 0))
    {
        SplitBrdf brdf = EvaluateBrdf(surface, lightSample.position);

        diffuse = brdf.demodulatedDiffuse * lightSample.radiance;
        specular = brdf.specular * lightSample.radiance;

        lightDistance = length(lightSample.position - surface.worldPos);
    }

    return needToStore;
}



























/*
Two questions:
- targetPdf : lightEmit * brdf * G * area <-- why area?
- In RTXDI_CombineDIReservoirs, 
    float risWeight = .targetPdf * .weightSum(==W) * .M(==1);
                    = .targetPdf * (.weightSum(==w_sum) / (.targetPdf * .numMisSamples)) * .M(==1)
                    = .weightSum(==w_sum) / .numMisSamples  <--- ???
*/

// Samples a polymorphic light relative to the given receiver surface.
// For most light types, the "uv" parameter is just a pair of uniform random numbers, originally
// produced by the RTXDI_GetNextRandom function and then stored in light reservoirs.
// For importance sampled environment lights, the "uv" parameter has the texture coordinates
// in the PDF texture, normalized to the (0..1) range.
RAB_LightSample RAB_SamplePolymorphicLight(RAB_LightInfo lightInfo, RAB_Surface surface, float2 uv)
{
    PolymorphicLightSample pls = PolymorphicLight::calcSample(lightInfo, uv, surface.worldPos);

    RAB_LightSample lightSample;
    lightSample.position = pls.position;
    lightSample.normal = pls.normal;
    lightSample.radiance = pls.radiance;
    lightSample.solidAnglePdf = pls.solidAnglePdf;
    lightSample.lightType = getLightType(lightInfo);
    return lightSample;
}

// Computes the multi importance sampling pdf for brdf and light sample.
// For light and BRDF PDFs wrt solid angle, blend between the two.
//      lightSelectionPdf is a dimensionless selection pdf
float RTXDI_LightBrdfMisWeight(
    RAB_Surface surface, 
    RAB_LightSample lightSample,
    float lightSelectionPdf, 
    float lightMisWeight, 
    bool isEnvironmentMap,
    float brdfMisWeight, 
    float brdfCutoff)
{
    float lightSolidAnglePdf = RAB_LightSampleSolidAnglePdf(lightSample);
    if (brdfMisWeight == 0 || RAB_IsAnalyticLightSample(lightSample) ||
        lightSolidAnglePdf <= 0 || isinf(lightSolidAnglePdf) || isnan(lightSolidAnglePdf))
    {
        // BRDF samples disabled or we can't trace BRDF rays MIS with analytical lights
        return lightMisWeight * lightSelectionPdf;
    }

    float3 lightDir;
    float lightDistance;
    RAB_GetLightDirDistance(surface, lightSample, lightDir, lightDistance);

    // Compensate for ray shortening due to brdf cutoff, does not apply to environment map sampling
    float brdfPdf = RAB_SurfaceEvaluateBrdfPdf(surface, lightDir);
    float maxDistance = RTXDI_BrdfMaxDistanceFromPdf(brdfCutoff, brdfPdf);
    if (!isEnvironmentMap && lightDistance > maxDistance)
        brdfPdf = 0.f;

    // Convert light selection pdf (unitless) to a solid angle measurement
    float sourcePdfWrtSolidAngle = lightSelectionPdf * lightSolidAnglePdf;

    // MIS blending against solid angle pdfs.
    float blendedPdfWrtSolidangle = lightMisWeight * sourcePdfWrtSolidAngle + brdfMisWeight * brdfPdf;

    // Convert back, RTXDI divides shading again by this term later
    return blendedPdfWrtSolidangle / lightSolidAnglePdf;
}


// Computes the weight of the given light samples when the given surface is
// shaded using that light sample. Exact or approximate BRDF evaluation can be
// used to compute the weight. ReSTIR will converge to a correct lighting result
// even if all samples have a fixed weight of 1.0, but that will be very noisy.
// Scaling of the weights can be arbitrary, as long as it's consistent
// between all lights and surfaces.
float RAB_GetLightSampleTargetPdfForSurface(RAB_LightSample lightSample, RAB_Surface surface)
{
    if (lightSample.solidAnglePdf <= 0)
        return 0;
    
    float3 brdfTimesNoL = float3(0.0);
    {
        float3 N = surface.normal;
        float3 V = surface.viewDir;
        float3 L = normalize(lightSample.position - surface.worldPos);

        if (dot(L, surface.geoNormal) > 0)
        {
            brdfTimesNoL = max(0, dot(N, L)) / M_PI * surface.material.diffuseAlbedo + 
                (surface.material.roughness < kMinRoughness) ? float3(0.0)
                : GGX_times_NdotL(V, L, N, max(surface.material.roughness, kMinRoughness), surface.material.specularF0);
        }
    }

    return (1.0/lightSample.solidAnglePdf) * RTXDI_Luminance(lightSample.radiance * brdfTimesNoL);
}

// Adds a new, non-reservoir light sample into the reservoir, returns true if this sample was selected.
// Algorithm (3) from the ReSTIR paper, Streaming RIS using weighted reservoir sampling.
bool RTXDI_StreamSample(
    inout RTXDI_DIReservoir reservoir,
    uint lightIndex,
    float2 uv,
    float random,
    float targetPdf,
    float invSourcePdf)
{
    float risWeight = targetPdf * invSourcePdf;
    reservoir.M += 1;
    reservoir.weightSum += risWeight;

    // New samples don't have visibility or age information, we can skip that.
    if (random * reservoir.weightSum < risWeight)
    {
        reservoir.lightData = lightIndex | RTXDI_DIReservoir_LightValidBit;
        reservoir.uvData = uint(saturate(uv.x) * 0xffff) | (uint(saturate(uv.y) * 0xffff) << 16);
        reservoir.targetPdf = targetPdf;
        return true;
    }
    return false;
}


RTXDI_DIReservoir RTXDI_SampleBrdf(
    inout RTXDI_RandomSamplerState rng,
    RAB_Surface surface,
    uint numBrdfSamples,
    float brdfCutoff,
    float brdfRayMinT,
    RTXDI_InitialSamplingMisData misData,
    inout RTXDI_RandomSamplerState coherentRng,
    RTXDI_LightBufferParameters lightBufferParams,
    out RAB_LightSample o_selectedSample)
{
    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();
    
    for (uint i = 0; i < numBrdfSamples; ++i)
    {
        float lightSourcePdf = 0;
        float3 sampleDir;
        uint lightIndex = RTXDI_InvalidLightIndex;
        float2 randXY = float2(0, 0);
        RAB_LightSample candidateSample = RAB_EmptyLightSample();

        if (RAB_SurfaceImportanceSampleBrdf(surface, rng, sampleDir))
        {
            float brdfPdf = RAB_SurfaceEvaluateBrdfPdf(surface, sampleDir);
            float maxDistance = RTXDI_BrdfMaxDistanceFromPdf(brdfCutoff, brdfPdf);
            
            bool hitAnything = RAB_TraceRayForLocalLight(
                RAB_GetSurfaceWorldPos(surface), 
                sampleDir,
                brdfRayMinT, 
                maxDistance, 
                lightIndex, 
                randXY);

            if (lightIndex != RTXDI_InvalidLightIndex)
            {
                RAB_LightInfo lightInfo = RAB_LoadLightInfo(lightIndex, false);
                candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, randXY);
                    
                if (brdfCutoff > 0.f)
                {
                    // If Mis cutoff is used, we need to evaluate the sample and make sure it actually could have been
                    // generated by the area sampling technique. This is due to numerical precision.
                    float3 lightDir;
                    float lightDistance;
                    RAB_GetLightDirDistance(surface, candidateSample, lightDir, lightDistance);

                    float brdfPdf = RAB_SurfaceEvaluateBrdfPdf(surface, lightDir);
                    float maxDistance = RTXDI_BrdfMaxDistanceFromPdf(brdfCutoff, brdfPdf);
                    if (lightDistance > maxDistance)
                        lightIndex = RTXDI_InvalidLightIndex;
                }

                if (lightIndex != RTXDI_InvalidLightIndex)
                {
                    lightSourcePdf = RAB_EvaluateLocalLightSourcePdf(lightIndex);
                }
            }
            else if (!hitAnything && (lightBufferParams.environmentLightParams.lightPresent != 0))
            {
                // sample environment light
                lightIndex = lightBufferParams.environmentLightParams.lightIndex;
                RAB_LightInfo lightInfo = RAB_LoadLightInfo(lightIndex, false);
                randXY = RAB_GetEnvironmentMapRandXYFromDir(sampleDir);
                candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, randXY);
                lightSourcePdf = RAB_EvaluateEnvironmentMapSamplingPdf(sampleDir);
            }
        }

        if (lightSourcePdf == 0)
        {
            // Did not hit a visible light
            continue;
        }

        bool isEnvMapSample = lightIndex == lightBufferParams.environmentLightParams.lightIndex;
        float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
        float blendedSourcePdf = RTXDI_LightBrdfMisWeight(surface, candidateSample, lightSourcePdf,
            isEnvMapSample ? misData.environmentMapMisWeight : misData.localLightMisWeight, 
            isEnvMapSample,
            misData.brdfMisWeight, brdfCutoff);
        float risRnd = RTXDI_GetNextRandom(rng);

        bool selected = RTXDI_StreamSample(state, lightIndex, randXY, risRnd, targetPdf, 1.0f / blendedSourcePdf);
        if (selected) 
            o_selectedSample = candidateSample;
    }

    RTXDI_FinalizeResampling(state, 1.0, misData.numMisSamples);
    state.M = 1;

    return state;
}

RTXDI_DIReservoir RTXDI_SampleLocalLights(
    inout RTXDI_RandomSamplerState rng,
    // inout RTXDI_RandomSamplerState coherentRng,
    RAB_Surface surface,
    RTXDI_DIInitialSamplingParameters sampleParams,
    RTXDI_InitialSamplingMisData misData,
    ReSTIRDI_LocalLightSamplingMode localLightSamplingMode,
    RTXDI_LightBufferRegion region,
    out RAB_LightSample o_selectedSample)
{
    o_selectedSample = RAB_EmptyLightSample();

    if (region.numLights == 0 || sampleParams.numLocalLightSamples == 0)
        return RTXDI_EmptyDIReservoir();

    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();

    for (uint i = 0; i < sampleParams.numLocalLightSamples; i++)
    {
        // STRATIFY SAMPLING : rnd in [i/N, (i+1)/N)
        float rnd = (RTXDI_GetNextRandom(rng) + i) / sampleParams.numLocalLightSamples; 
        float2 uv = float2(RTXDI_GetNextRandom(rng), RTXDI_GetNextRandom(rng));
        float risRnd = RTXDI_GetNextRandom(rng);

        float invSourcePdf = float(region.numLights);
        uint lightIndex = region.firstLightIndex + min(uint(floor(rnd * region.numLights)), region.numLights - 1);
        RAB_LightInfo lightInfo = t_LightDataBuffer[lightIndex];

        RAB_LightSample candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, uv);
        float blendedSourcePdf = RTXDI_LightBrdfMisWeight(
            surface, 
            candidateSample, 
            1.0 / invSourcePdf,
            misData.localLightMisWeight, 
            false, 
            misData.brdfMisWeight, 
            sampleParams.brdfCutoff);

        if (blendedSourcePdf == 0)
            continue;

        float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);

        bool selected = RTXDI_StreamSample(
            state, 
            lightIndex, 
            uv, 
            risRnd, 
            targetPdf, 
            1.0 / blendedSourcePdf);

        if (selected) 
            o_selectedSample = candidateSample;
    }

    float denominator = reservoir.targetPdf * misData.numMisSamples;
    state.weightSum = (denominator == 0.0) ? 0.0 : reservoir.weightSum / denominator;
    state.M = 1;

    return state;
}

// Adds `newReservoir` into `reservoir`, returns true if the new reservoir's sample was selected.
// Algorithm (4) from the ReSTIR paper, Combining the streams of multiple reservoirs.
// Normalization - Equation (6) - is postponed until all reservoirs are combined.
bool RTXDI_CombineDIReservoirs(
    inout RTXDI_DIReservoir reservoir,
    const RTXDI_DIReservoir newReservoir,
    float random)
{
    float risWeight = newReservoir.targetPdf * newReservoir.weightSum * newReservoir.M;
    reservoir.M += newReservoir.M;
    reservoir.weightSum += risWeight;

    // If we did select this sample, update the relevant data
    if (random * reservoir.weightSum < risWeight)
    {
        reservoir.lightData = newReservoir.lightData;
        reservoir.uvData = newReservoir.uvData;
        reservoir.targetPdf = newReservoir.targetPdf;
        reservoir.packedVisibility = newReservoir.packedVisibility;
        reservoir.spatialDistance = newReservoir.spatialDistance;
        reservoir.age = newReservoir.age;
        return true;
    }

    return false;
}

// Samples the local, infinite, and environment lights for a given surface
RTXDI_DIReservoir RTXDI_SampleLightsForSurface(
    inout RTXDI_RandomSamplerState rng,
    // inout RTXDI_RandomSamplerState coherentRng,
    RAB_Surface surface,
    RTXDI_DIInitialSamplingParameters sampleParams,
    RTXDI_LightBufferParameters lightBufferParams,
    out RAB_LightSample o_lightSample)
{
    RTXDI_InitialSamplingMisData misData = RTXDI_ComputeInitialSamplingMisData(sampleParams);

    RAB_LightSample localSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir localReservoir = RTXDI_SampleLocalLights(
        rng, 
        // coherentRng, 
        surface,
        sampleParams, 
        misData, 
        sampleParams.localLightSamplingMode, 
        lightBufferParams.localLightBufferRegion,
        localSample);

    RAB_LightSample infiniteSample = RAB_EmptyLightSample();  
    RTXDI_DIReservoir infiniteReservoir = RTXDI_SampleInfiniteLights(
        rng, 
        surface,
        sampleParams.numInfiniteLightSamples, 
        lightBufferParams.infiniteLightBufferRegion, 
        infiniteSample);

    RAB_LightSample brdfSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir brdfReservoir = RTXDI_SampleBrdf(
        rng, 
        surface, 
        sampleParams.numBrdfSamples, 
        sampleParams.brdfCutoff, 
        sampleParams.brdfRayMinT, 
        misData, 
        // coherentRng, 
        lightBufferParams, 
        brdfSample);

    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();
    RTXDI_CombineDIReservoirs(state, localReservoir, 0.5);
    bool selectInfinite = RTXDI_CombineDIReservoirs(state, infiniteReservoir, RTXDI_GetNextRandom(rng));
    bool selectBrdf = RTXDI_CombineDIReservoirs(state, brdfReservoir, RTXDI_GetNextRandom(rng));
    
    RTXDI_FinalizeResampling(state, 1.0, 1.0);
    state.M = 1;

    if (selectBrdf)
        o_lightSample = brdfSample;
    else if (selectInfinite)
        o_lightSample = infiniteSample;
    else
        o_lightSample = localSample;

    if(sampleParams.enableInitialVisibility 
        && RTXDI_IsValidDIReservoir(state)
        && !RAB_GetConservativeVisibility(surface, o_lightSample))
    {
       RTXDI_StoreVisibilityInDIReservoir(state, 0, true);
    }

    return state;
}

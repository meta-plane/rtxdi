
// Very similar to RTXDI_TemporalResamplingParameters but it has an extra field
// It's also not the same algo, and we don't want the two to be coupled
struct RTXDI_GITemporalResamplingParameters //ReSTIRGI_TemporalResamplingParameters
{
    // Surface depth similarity threshold for temporal reuse.
    // If the previous frame surface's depth is within this threshold from the current frame surface's depth,
    // the surfaces are considered similar. The threshold is relative, i.e. 0.1 means 10% of the current depth.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    /*OK*/float depthThreshold;

    // Surface normal similarity threshold for temporal reuse.
    // If the dot product of two surfaces' normals is higher than this threshold, the surfaces are considered similar.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    /*OK*/float normalThreshold;

    // Maximum history length for reuse, measured in frames.
    // Higher values result in more stable and high quality sampling, at the cost of slow reaction to changes.
    uint32_t maxHistoryLength;

    // Enables resampling from a location around the current pixel instead of what the motion vector points at,
    // in case no surface near the motion vector matches the current surface (e.g. disocclusion).
    // This behavoir makes disocclusion areas less noisy but locally biased, usually darker.
    /*OK*/uint32_t enableFallbackSampling;

    // Controls the bias correction math for temporal reuse. Depending on the setting, it can add
    // some shader cost and one approximate shadow ray per pixel (or per two pixels if checkerboard sampling is enabled).
    // Ideally, these rays should be traced through the previous frame's BVH to get fully unbiased results.
    RTXDI_GIBiasCorrectionMode biasCorrectionMode;

    // Discard the reservoir if its age exceeds this value.
    /*OK*/uint32_t maxReservoirAge;

    // Enables permuting the pixels sampled from the previous frame in order to add temporal
    // variation to the output signal and make it more denoiser friendly.
    /*OK*/uint32_t enablePermutationSampling;

    // Random number for permutation sampling that is the same for all pixels in the frame
    /*OK*/uint32_t uniformRandomNumber;
};

struct RTXDI_GIParameters
{
    RTXDI_ReservoirBufferParameters reservoirBufferParams;
    RTXDI_GIBufferIndices bufferIndices;
    RTXDI_GITemporalResamplingParameters temporalResamplingParams = {
        .depthThreshold = 0.1f;
        .enableFallbackSampling = true;
        .enablePermutationSampling = false;
        .maxHistoryLength = 8;
        .maxReservoirAge = 30;
        .normalThreshold = 0.6f;
        .biasCorrectionMode = RTXDI_GIBiasCorrectionMode::Basic;
    };
    RTXDI_BoilingFilterParameters boilingFilterParams;
    RTXDI_GISpatialResamplingParameters spatialResamplingParams;
    RTXDI_GISpatioTemporalResamplingParameters spatioTemporalResamplingParams;
    RTXDI_GIFinalShadingParameters finalShadingParams;
};

// A surface with enough information to evaluate BRDFs
struct RAB_Surface
{
    float3 worldPos;
    float3 viewDir;
    float viewDepth;
    float3 normal;
    float3 geoNormal;
    float diffuseProbability;

    struct RAB_Material
    {
        float3 diffuseAlbedo;
        float3 specularF0;
        float roughness;
        float3 emissiveColor;
    } material;
};

// This structure represents a indirect lighting reservoir that stores the radiance and weight
// as well as its the position where the radiane come from.
struct RTXDI_GIReservoir
{
    float3 position;    // postion of the 2nd bounce surface.
    float3 normal;      // normal vector of the 2nd bounce surface.
    float3 radiance;    // incoming radiance from the 2nd bounce surface.
    float weightSum;    // Overloaded: represents RIS weight sum during streaming, then reservoir weight (W, i.e., inverse PDF) after FinalizeResampling
    uint M;             // Number of samples considered for this reservoir
    uint age;           // Number of frames the chosen sample has survived.
};

// Creates a GI reservoir from a raw light sample.
// Note: the original sample PDF can be embedded into sampleRadiance, in which case the samplePdf parameter should be set to 1.0.
RTXDI_GIReservoir RTXDI_MakeGIReservoir(
    const float3 samplePos,
    const float3 sampleNormal,
    const float3 sampleRadiance,
    const float samplePdf)
{
    RTXDI_GIReservoir reservoir;
    reservoir.position = samplePos;
    reservoir.normal = sampleNormal;
    reservoir.radiance = sampleRadiance;
    reservoir.weightSum = samplePdf > 0.0 ? 1.0 / samplePdf : 0.0;
    reservoir.M = 1;
    reservoir.age = 0;
    return reservoir;
}

// Adds `newReservoir` into `reservoir`, returns true if the new reservoir's sample was selected.
// This function assumes the newReservoir has been normalized, so its weightSum means "1/g * 1/M * \sum{g/p}"
// and the targetPdf is a conversion factor from the newReservoir's space to the reservoir's space (integrand).
bool RTXDI_CombineGIReservoirs(
    inout RTXDI_GIReservoir reservoir,
    const RTXDI_GIReservoir newReservoir,
    float random,
    float targetPdf)
{
    // What's the current weight (times any prior-step RIS normalization factor)
    const float risWeight = targetPdf * newReservoir.weightSum * newReservoir.M;

    reservoir.M += newReservoir.M;
    reservoir.weightSum += risWeight;

    if (random * reservoir.weightSum <= risWeight)
    {
        reservoir.position = newReservoir.position;
        reservoir.normal = newReservoir.normal;
        reservoir.radiance = newReservoir.radiance;
        reservoir.age = newReservoir.age;
        return true;
    }

    return false;
}

void RTXDI_StoreGIReservoir(
    const RTXDI_GIReservoir reservoir,
    RTXDI_ReservoirBufferParameters reservoirParams,
    uint2 reservoirPosition,
    uint reservoirArrayIndex)
{
    uint pointer = RTXDI_ReservoirPositionToPointer(reservoirParams, reservoirPosition, reservoirArrayIndex);
    u_GIReservoirs[pointer] = RTXDI_PackGIReservoir(reservoir);
}

RTXDI_GIReservoir RTXDI_LoadGIReservoir(
    RTXDI_ReservoirBufferParameters reservoirParams,
    uint2 reservoirPosition,
    uint reservoirArrayIndex)
{
    uint pointer = RTXDI_ReservoirPositionToPointer(reservoirParams, reservoirPosition, reservoirArrayIndex);
    return RTXDI_UnpackGIReservoir(u_GIReservoirs[pointer]);
}

bool RTXDI_IsValidGIReservoir(const RTXDI_GIReservoir reservoir)
{
    return reservoir.M != 0;
}

void RTXDI_FinalizeGIResampling(
    inout RTXDI_GIReservoir reservoir,
    float normalizationNumerator,
    float normalizationDenominator)
{
    reservoir.weightSum = (normalizationDenominator == 0.0) ? 0.0 
        : (reservoir.weightSum * normalizationNumerator) / normalizationDenominator;
}


[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 GlobalIndex : SV_DispatchThreadID, uint2 LocalIndex : SV_GroupThreadID)
{
    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, g_Const.runtimeParams.activeCheckerboardField);

    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(GlobalIndex, g_Const.runtimeParams.frameIndex, RTXDI_GI_TEMPORAL_RESAMPLING_RANDOM_SEED);
    
    const RAB_Surface primarySurface = RAB_GetGBufferSurface(pixelPosition, false);
    
    const uint2 reservoirPosition = RTXDI_PixelPosToReservoirPos(pixelPosition, g_Const.runtimeParams.activeCheckerboardField);
    
    RTXDI_GIReservoir reservoir = RTXDI_LoadGIReservoir(
        g_Const.restirGI.reservoirBufferParams, 
        reservoirPosition, 
        g_Const.restirGI.bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex);

    /*
        - isValidSecondarySurface: infinite surface(environment map) 포함
        - isSpecularRay && isDeltaSurface: state of primary surface and sampling ray(to secondary surface) 
        - radiance: radiance from secondary surface to primary surface
        - secondaryGBufferData.pdf: the pdf of sampling ray(to secondary surface)

        if (isValidSecondarySurface && !(isSpecularRay && isDeltaSurface))
        {
            reservoir == RTXDI_GIReservoir{
                .position = secondarySurface.worldPos,
                .normal = secondarySurface.normal,
                .radiance = radiance,
                .weightSum = secondaryGBufferData.pdf > 0.0 ? 1.0 / secondaryGBufferData.pdf : 0.0,
                .M = 1;
                .age = 0;
            };
        }
        else
        {
            reservoir == {};
        }
    */

    float3 motionVector = t_MotionVectors[pixelPosition].xyz;
    motionVector = convertMotionVectorToPixelSpace(g_Const.view, g_Const.prevView, pixelPosition, motionVector);

    if (RAB_IsSurfaceValid(primarySurface)) 
    {
        RTXDI_GITemporalResamplingParameters tParams = g_Const.restirGI.temporalResamplingParams;

        // Age threshold should vary.
        // This is to avoid to die a bunch of GI reservoirs at once at a disoccluded area.
        tParams.maxReservoirAge *= (0.5 + RTXDI_GetNextRandom(rng) * 0.5);

        reservoir = RTXDI_GITemporalResampling(
            pixelPosition, 
            primarySurface, 
            motionVector, 
            g_Const.restirGI.bufferIndices.temporalResamplingInputBufferIndex, 
            reservoir, 
            rng, 
            g_Const.runtimeParams, 
            g_Const.restirGI.reservoirBufferParams, 
            tParams);
    }

    RTXDI_StoreGIReservoir(
        reservoir, 
        g_Const.restirGI.reservoirBufferParams, 
        reservoirPosition, 
        g_Const.restirGI.bufferIndices.temporalResamplingOutputBufferIndex);
}


// Temporal resampling for GI reservoir pass.
RTXDI_GIReservoir RTXDI_GITemporalResampling(
    const uint2 pixelPosition,                              // pixelPosition
    const RAB_Surface surface,                              // primarySurface
    float3 screenSpaceMotion,                               // motionVector
    uint sourceBufferIndex,                                 // g_Const.restirGI.bufferIndices.temporalResamplingInputBufferIndex
    const RTXDI_GIReservoir inputReservoir,                 // reservoir
    inout RTXDI_RandomSamplerState rng,                     // rng
    const RTXDI_RuntimeParameters params,                   // g_Const.runtimeParams
    const RTXDI_ReservoirBufferParameters reservoirParams,  // g_Const.restirGI.reservoirBufferParams
    const RTXDI_GITemporalResamplingParameters tparams)     // g_Const.restirGI.temporalResamplingParams, .maxReservoirAge *= (0.5 + RTXDI_GetNextRandom(rng) * 0.5)
{
    // Backproject this pixel to last frame
    int2 prevPos = int2(round(float2(pixelPosition) + screenSpaceMotion.xy));

    // Try to find a matching surface in the neighborhood of the reprojected pixel
    const int temporalSampleCount = 5;
    const int sampleCount = temporalSampleCount + (tparams.enableFallbackSampling ? 1 : 0);
    const int temporalSampleStartIdx = int(RTXDI_GetNextRandom(rng) * 8);

    RTXDI_GIReservoir temporalReservoir;
    bool foundTemporalReservoir = false;
    RAB_Surface temporalSurface = RAB_EmptySurface();
    
    for (int i = 0; i < sampleCount; i++)
    {
        const bool isFirstSample = i == 0;
        const bool isFallbackSample = i == temporalSampleCount;

        int2 idx = prevPos;
        if (isFallbackSample)
        {
            // Last sample is a fallback for disocclusion areas: use zero motion vector.
            idx = int2(pixelPosition);
        }
        else if (!isFirstSample)
        {
            int sampleIdx = (temporalSampleStartIdx + i) & 7;
            const int radius = (params.activeCheckerboardField == 0) ? 1 : 2;
            int2 offset;
            {
                // Generates a pattern of offsets for looking closely around a given pixel.
                // The pattern places 'sampleIdx' at the following locations in screen space around pixel (x):
                //   0 4 3
                //   6 x 7
                //   2 5 1
                int mask2 = sampleIdx >> 1 & 0x01;       // 0, 0, 1, 1, 0, 0, 1, 1
                int mask4 = 1 - (sampleIdx >> 2 & 0x01); // 1, 1, 1, 1, 0, 0, 0, 0
                int tmp0 = -1 + 2 * (sampleIdx & 0x01);  // -1, 1,....
                int tmp1 = 1 - 2 * mask2;                // 1, 1,-1,-1, 1, 1,-1,-1
                int tmp2 = mask4 | mask2;                // 1, 1, 1, 1, 0, 0, 1, 1
                int tmp3 = mask4 | (1 - mask2);          // 1, 1, 1, 1, 1, 1, 0, 0
                offset = int2(tmp0, tmp0 * tmp1) * int2(tmp2, tmp3) * radius;
            }

            idx += offset;
        }

        if ((tparams.enablePermutationSampling && isFirstSample) || isFallbackSample)
        {
            // Apply permutation sampling for the first (non-jittered) sample,
            // also for the last (fallback) sample to prevent visible repeating patterns in disocclusions.
            RTXDI_ApplyPermutationSampling(idx, tparams.uniformRandomNumber);
        }
        
        // Grab shading / g-buffer data from last frame
        temporalSurface = RAB_GetGBufferSurface(idx, true);

        if (!RAB_IsSurfaceValid(temporalSurface))
        {
            continue;
        }

        // Test surface similarity, discard the sample if the surface is too different.
        // Skip this test for the last (fallback) sample.
        if (!isFallbackSample && !RTXDI_IsValidNeighbor(
            RAB_GetSurfaceNormal(surface), 
            RAB_GetSurfaceNormal(temporalSurface),
            RAB_GetSurfaceLinearDepth(surface) + screenSpaceMotion.z, 
            RAB_GetSurfaceLinearDepth(temporalSurface),
            tparams.normalThreshold, 
            tparams.depthThreshold))
        {
            continue;
        }

        // Test material similarity and perform any other app-specific tests.
        if (!RAB_AreMaterialsSimilar(RAB_GetMaterial(surface), RAB_GetMaterial(temporalSurface)))
        {
            continue;
        }

        // Read temporal reservoir.
        uint2 prevReservoirPos = RTXDI_PixelPosToReservoirPos(idx, params.activeCheckerboardField);
        temporalReservoir = RTXDI_LoadGIReservoir(reservoirParams, prevReservoirPos, sourceBufferIndex);

        // Check if the reservoir is a valid one.
        if (!RTXDI_IsValidGIReservoir(temporalReservoir))
        {
            continue;
        }

        foundTemporalReservoir = true;
        break;
    }
    
    if (foundTemporalReservoir)
    {
        float jacobian = RTXDI_CalculateJacobian(
            RAB_GetSurfaceWorldPos(surface), 
            RAB_GetSurfaceWorldPos(temporalSurface), 
            temporalReservoir.position, 
            temporalReservoir.normal);

        if (!RAB_ValidateGISampleWithJacobian(jacobian))
            foundTemporalReservoir = false;

        temporalReservoir.weightSum *= jacobian;
        
        // Clamp history length
        temporalReservoir.M = min(temporalReservoir.M, tparams.maxHistoryLength);

        // Make the sample older
        ++temporalReservoir.age;

        if (temporalReservoir.age > tparams.maxReservoirAge)
            foundTemporalReservoir = false;
    }

    RTXDI_GIReservoir curReservoir = RTXDI_EmptyGIReservoir();

    float selectedTargetPdf = 0;
    if (RTXDI_IsValidGIReservoir(inputReservoir))
    {
        selectedTargetPdf = RTXDI_Luminance(RAB_GetReflectedBrdfRadianceForSurface( // brdf * raadiance * cos
            inputReservoir.position, 
            inputReservoir.radiance, 
            surface));

        RTXDI_CombineGIReservoirs(curReservoir, inputReservoir, 0.5, selectedTargetPdf);
    }

    bool selectedPreviousSample = false;
    if (foundTemporalReservoir)
    {
        float targetPdf = RTXDI_Luminance(RAB_GetReflectedBrdfRadianceForSurface(
            temporalReservoir.position, 
            temporalReservoir.radiance, 
            surface));

        selectedPreviousSample = RTXDI_CombineGIReservoirs(curReservoir, temporalReservoir, RTXDI_GetNextRandom(rng), targetPdf);
        if (selectedPreviousSample)
        {
            selectedTargetPdf = targetPdf;
        }
    }

    if (tparams.biasCorrectionMode >= RTXDI_BIAS_CORRECTION_BASIC)
    {
        float pi = selectedTargetPdf;
        float piSum = selectedTargetPdf * inputReservoir.M;

        if (RTXDI_IsValidGIReservoir(curReservoir) && foundTemporalReservoir)
        {
            float temporalP = RAB_GetGISampleTargetPdfForSurface(curReservoir.position, curReservoir.radiance, temporalSurface);

            if (tparams.biasCorrectionMode == RTXDI_BIAS_CORRECTION_RAY_TRACED && temporalP > 0)
            {
                if (!RAB_GetTemporalConservativeVisibility(surface, temporalSurface, curReservoir.position))
                {
                    temporalP = 0;
                }
            }

            pi = selectedPreviousSample ? temporalP : pi;
            piSum += temporalP * temporalReservoir.M;
        }

        // Normalizing
        float normalizationDenominator = piSum * selectedTargetPdf;
        RTXDI_FinalizeGIResampling(curReservoir, pi, normalizationDenominator);
    }
    else
    {
        float normalizationDenominator = selectedTargetPdf * curReservoir.M;
        RTXDI_FinalizeGIResampling(curReservoir, 1.0, normalizationDenominator);
    }

    return curReservoir;
}



// Bias correction modes for temporal and spatial resampling:
// Use (1/M) normalization, which is very biased but also very fast.
#define RTXDI_BIAS_CORRECTION_OFF 0
// Use MIS-like normalization but assume that every sample is visible.
#define RTXDI_BIAS_CORRECTION_BASIC 1
// Use pairwise MIS normalization (assuming every sample is visible).  Better perf & specular quality
#define RTXDI_BIAS_CORRECTION_PAIRWISE 2
// Use MIS-like normalization with visibility rays. Unbiased.
#define RTXDI_BIAS_CORRECTION_RAY_TRACED 3


/*
Difference:
- tparams.biasCorrectionMode
------------------------------------
- age, history 테스트
- RTXDI 타겟함수변경
- RTXDI temporalSampleCount변경
- 모션벡터 차이 확인 (.z성분)
- boiling 필터
*/
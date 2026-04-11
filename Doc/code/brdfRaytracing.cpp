static const float c_MaxIndirectRadiance = 10;

/*
[DirectLightingMode::Brdf or IndirectLightingMode::Brdf/ReStirGI]
->
DirectLightingMode::None   + IndirectLightingMode::Brdf         (?)
DirectLightingMode::None   + IndirectLightingMode::ReStirGI     (**)
DirectLightingMode::Brdf   + IndirectLightingMode::None         (?)
DirectLightingMode::Brdf   + IndirectLightingMode::Brdf         (?) (x)
DirectLightingMode::Brdf   + IndirectLightingMode::ReStirGI     (**)(x)
DirectLightingMode::ReStir + IndirectLightingMode::Brdf         (*)
DirectLightingMode::ReStir + IndirectLightingMode::ReStirGI     (*)
(X)
DirectLightingMode::None   + IndirectLightingMode::None
DirectLightingMode::ReStir + IndirectLightingMode::None
*/

[shader("raygeneration")]
void RayGen()
{
    uint2 GlobalIndex = DispatchRaysIndex().xy;
    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, g_Const.runtimeParams.activeCheckerboardField);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);
    if (!RAB_IsSurfaceValid(surface))
        return;

    const RayDesc ray = {
        .Origin = surface.worldPos,
        .TMin = 0.001f * max(1, 0.1 * length(surface.worldPos - g_Const.view.cameraDirectionOrPosition.xyz)),
        .TMax = 1000.0,
    };

    const bool isDeltaSurface = surface.material.roughness < kMinRoughness;

    //-----------------------------------------------------------------------------------------------//
    //-----------------------------------------------------------------------------------------------//
    //-----------------------------------------------------------------------------------------------//

    bool isSpecularRay = false;
    float3 BRDF_over_PDF;
    float overall_PDF;

    {
        RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(
            GlobalIndex, g_Const.runtimeParams.frameIndex, RTXDI_GI_GENERATE_INITIAL_SAMPLES_RANDOM_SEED);
        float2 Rand = {RTXDI_GetNextRandom(rng), RTXDI_GetNextRandom(rng)};

        float3 tangent, bitangent;
        branchlessONB(surface.normal, tangent, bitangent);

        const float3 V = normalize(g_Const.view.cameraDirectionOrPosition.xyz - surface.worldPos);

        float3 specularDirection;
        float3 specular_BRDF_over_PDF;
        {
            float3 Ve = float3(dot(V, tangent), dot(V, bitangent), dot(V, surface.normal));
            float3 He = sampleGGX_VNDF(Ve, surface.material.roughness, Rand);
            float3 H = isDeltaSurface ? surface.normal : normalize(He.x * tangent + He.y * bitangent + He.z * surface.normal);
            specularDirection = reflect(-V, H);

            float HoV = saturate(dot(H, V));
            float NoV = saturate(dot(surface.normal, V));
            float3 F = Schlick_Fresnel(surface.material.specularF0, HoV);
            // float G1 = isDeltaSurface ? 1.0 : (NoV > 0) ? G1_Smith(surface.material.roughness, NoV) : 0; // error??
            float G1 = isDeltaSurface ? 1.0 : (NoV > 0) ? G1_Smith(surface.material.roughness, dot(specularDirection, surface.normal)) : 0;
            specular_BRDF_over_PDF = F * G1;
        }

        float3 diffuseDirection;
        float diffuse_BRDF_over_PDF;
        {
            float solidAnglePdf;
            float3 localDirection = sampleCosHemisphere(Rand, solidAnglePdf);
            diffuseDirection = tangent * localDirection.x + bitangent * localDirection.y + surface.normal * localDirection.z;
            diffuse_BRDF_over_PDF = 1.0;
        }

        // Ignores PDF of specular or diffuse
        // Chooses PDF based on relative luminance
        float specular_PDF = saturate(calcLuminance(specular_BRDF_over_PDF) /
            calcLuminance(specular_BRDF_over_PDF + diffuse_BRDF_over_PDF * surface.material.diffuseAlbedo));
        isSpecularRay = RTXDI_GetNextRandom(rng) < specular_PDF;

        if (isSpecularRay)
        {
            ray.Direction = specularDirection;
            BRDF_over_PDF = specular_BRDF_over_PDF / specular_PDF;
        }
        else
        {
            ray.Direction = diffuseDirection;
            BRDF_over_PDF = diffuse_BRDF_over_PDF / (1.0 - specular_PDF);
        }

        if (dot(surface.geoNormal, ray.Direction) <= 0.0)
        {
            BRDF_over_PDF = 0.0;
            ray.TMax = 0;
        }

        // Calculates PDF of individual respective lobes.
        const float specularLobe_PDF = ImportanceSampleGGX_VNDF_PDF(surface.material.roughness, surface.normal, V, ray.Direction);
        const float diffuseLobe_PDF = saturate(dot(ray.Direction, surface.normal)) / c_pi;

        // For delta surfaces, we only pass the diffuse lobe to ReSTIR GI, and this pdf is for that.
        overall_PDF = isDeltaSurface ? diffuseLobe_PDF : lerp(diffuseLobe_PDF, specularLobe_PDF, specular_PDF);
    }

    //-----------------------------------------------------------------------------------------------//
    //-----------------------------------------------------------------------------------------------//
    //-----------------------------------------------------------------------------------------------//

    RAB_RayPayload payload = {
        .instanceID = ~0u,
        .throughput = 1.0
    };

    {
        uint instanceMask = INSTANCE_MASK_OPAQUE
            | g_Const.sceneConstants.enableAlphaTestedGeometry ? INSTANCE_MASK_ALPHA_TESTED : 0
            | g_Const.sceneConstants.enableTransparentGeometry ? INSTANCE_MASK_TRANSPARENT : 0;

        TraceRay(SceneBVH, RAY_FLAG_NONE, instanceMask, 0, 0, 0, ray, payload);
    }


    //-----------------------------------------------------------------------------------------------//
    //-----------------------------------------------------------------------------------------------//
    //-----------------------------------------------------------------------------------------------//
    
    float3 radiance = 0;

    struct
    {
        float3 position;
        float3 normal;
        float3 diffuseAlbedo;
        float3 specularF0;
        float roughness;
        bool isEnvironmentMap;
    } secondarySurface;

    {
        // Include the emissive component of surfaces seen [with BRDF rays if requested (i.e. when Direct Lighting mode
        // is set to BRDF)] or [on delta reflection rays] because those bypass ReSTIR GI and direct specular lighting,
        // and we need to see reflections of lamps and the sky in mirrors.
        const bool includeEmissiveComponent = 
            g_Const.brdfPT.enableIndirectEmissiveSurfaces // <- DirectLightingMode::Brdf
            || (isSpecularRay && isDeltaSurface);

        if (payload.instanceID != ~0u)
        {
            GeometrySample gs = getGeometryFromHit(
                payload.instanceID,
                payload.geometryIndex,
                payload.primitiveIndex,
                payload.barycentrics,
                GeomAttr_Normal | GeomAttr_TexCoord | GeomAttr_Position,
                t_InstanceData, t_GeometryData, t_MaterialConstants);

            MaterialSample ms = sampleGeometryMaterial(
                gs, 0, 0, 0,
                MatAttr_BaseColor | MatAttr_Emissive | MatAttr_MetalRough, s_MaterialSampler);

            ms.shadingNormal = getBentNormal(gs.flatNormal, ms.shadingNormal, ray.Direction);
            ms.roughness = max(ms.roughness, g_Const.brdfPT.materialOverrideParams.minSecondaryRoughness);

            if (includeEmissiveComponent)
                radiance += ms.emissiveColor;

            secondarySurface.position = ray.Origin + ray.Direction * payload.committedRayT;
            secondarySurface.normal = (dot(gs.geometryNormal, ray.Direction) < 0) ? gs.geometryNormal : -gs.geometryNormal;
            secondarySurface.diffuseAlbedo = ms.diffuseAlbedo;
            secondarySurface.specularF0 = ms.specularF0;
            secondarySurface.roughness = ms.roughness;
            secondarySurface.isEnvironmentMap = false;
        }
        else
        {
            if (g_Const.sceneConstants.enableEnvironmentMap && includeEmissiveComponent)
            {
                float3 environmentRadiance = GetEnvironmentRadiance(ray.Direction);
                radiance += environmentRadiance;
            }

            secondarySurface.position = ray.Origin + ray.Direction * DISTANT_LIGHT_DISTANCE;
            secondarySurface.normal = -ray.Direction;
            secondarySurface.diffuseAlbedo = 0;
            secondarySurface.specularF0 = 0;
            secondarySurface.roughness = 0;
            secondarySurface.isEnvironmentMap = true;
        }
    }

    //-----------------------------------------------------------------------------------------------//
    //-----------------------------------------------------------------------------------------------//
    //-----------------------------------------------------------------------------------------------//

    if (g_Const.enableBrdfIndirect) // IndirectLightingMode::Brdf/ReStirGI
    {
        SecondaryGBufferData secondaryGBufferData = (SecondaryGBufferData)0;
        secondaryGBufferData.worldPos = secondarySurface.position;
        secondaryGBufferData.normal = ndirToOctUnorm32(secondarySurface.normal);
        secondaryGBufferData.throughputAndFlags = Pack_R16G16B16A16_FLOAT(float4(payload.throughput * BRDF_over_PDF, 0));
        secondaryGBufferData.diffuseAlbedo = Pack_R11G11B10_UFLOAT(secondarySurface.diffuseAlbedo);
        secondaryGBufferData.specularAndRoughness = Pack_R8G8B8A8_Gamma_UFLOAT(float4(secondarySurface.specularF0, secondarySurface.roughness));

        if (g_Const.brdfPT.enableReSTIRGI) // IndirectLightingMode::ReStirGI
        {
            if (isSpecularRay && isDeltaSurface)
            {
                // Special case for specular rays on delta surfaces: they bypass ReSTIR GI and are shaded
                // entirely in the ShadeSecondarySurfaces pass, so they need the right throughput here.
            }
            else
            {
                // BRDF_over_PDF will be multiplied after resampling GI reservoirs.
                secondaryGBufferData.throughputAndFlags = Pack_R16G16B16A16_FLOAT(float4(payload.throughput, 0));
            }

            // The emission from the secondary surface needs to be added when creating the initial
            // GI reservoir sample in ShadeSecondarySurface.hlsl. It need to be stored separately.
            secondaryGBufferData.emission = radiance;
            radiance = 0;

            secondaryGBufferData.pdf = overall_PDF;
        }

        uint flags = 0;
        if (isSpecularRay) flags |= kSecondaryGBuffer_IsSpecularRay;
        if (isDeltaSurface) flags |= kSecondaryGBuffer_IsDeltaSurface;
        if (secondarySurface.isEnvironmentMap) flags |= kSecondaryGBuffer_IsEnvironmentMap;
        secondaryGBufferData.throughputAndFlags.y |= flags << 16;

        uint gbufferIndex = RTXDI_ReservoirPositionToPointer(
            g_Const.restirGI.reservoirBufferParams, GlobalIndex, 0);

        u_SecondaryGBuffer[gbufferIndex] = secondaryGBufferData;
    }

    //-----------------------------------------------------------------------------------------------//
    //-----------------------------------------------------------------------------------------------//
    //-----------------------------------------------------------------------------------------------//

    float3 diffuse = 0;
    float3 specular = 0;
    if (!isDeltaSurface)
    {
        diffuse = !isSpecularRay ? radiance * payload.throughput * BRDF_over_PDF : 0.0;
        specular = isSpecularRay ? radiance * payload.throughput * BRDF_over_PDF : 0.0;
    }
    
    if (g_Const.enableBrdfAdditiveBlend) // DirectLightingMode::ReStir
    {
        u_DiffuseLighting[pixelPosition].rgb += diffuse;
        u_SpecularLighting[pixelPosition].rgb += specular;
    }
    else
    {
        u_DiffuseLighting[pixelPosition].rgb = diffuse;
        u_SpecularLighting[pixelPosition].rgb = specular;
    }

    /*
    if ((any(radiance > 0) || !g_Const.enableBrdfAdditiveBlend)) // !DirectLightingMode::ReStir
    {
        // DirectLightingMode::None   + IndirectLightingMode::Brdf
        // DirectLightingMode::None   + IndirectLightingMode::ReStirGI
        // DirectLightingMode::Brdf   + IndirectLightingMode::None
        // DirectLightingMode::Brdf   + IndirectLightingMode::Brdf
        // DirectLightingMode::Brdf   + IndirectLightingMode::ReStirGI  <-- now, captured
        // DirectLightingMode::ReStir + IndirectLightingMode::Brdf     && radiance > 0
        
        radiance *= payload.throughput;

        float3 diffuse = 0;
        float3 specular = 0;
        if (!isDeltaSurface)
        {
            diffuse = !isSpecularRay ? radiance * BRDF_over_PDF : 0.0;
            specular = isSpecularRay ? radiance * BRDF_over_PDF : 0.0;
            payload.committedRayT = 0;
        }

        StoreShadingOutput(
            GlobalIndex, 
            pixelPosition, 
            // surface.viewDepth, 
            // surface.material.roughness, 
            diffuse, 
            specular, 
            payload.committedRayT, 
            !g_Const.enableBrdfAdditiveBlend,   // !DirectLightingMode::ReStir
            !g_Const.enableBrdfIndirect);       // IndirectLightingMode::None
    }
    else
    {
        // DirectLightingMode::ReStir + IndirectLightingMode::ReStirGI 
        // DirectLightingMode::ReStir + IndirectLightingMode::Brdf     && radiance == 0
    }
    */
}

/*
output:
- u_SecondaryGBuffer[...]
    .worldPos
    .normal
    .diffuseAlbedo
    .specularAndRoughness
    --------------------
    .throughputAndFlags
    .pdf
    .emission
- u_DiffuseLighting[...]
- u_SpecularLighting[...]
*/


void StoreShadingOutput(
    uint2 reservoirPosition,
    uint2 pixelPosition,
    // float viewDepth,
    // float roughness,
    float3 diffuse,
    float3 specular,
    float lightDistance,
    bool isFirstPass,
    bool isLastPass)
{
    uint2 lightingTexturePos = (g_Const.denoiserMode == DENOISER_MODE_OFF)
        ? pixelPosition : reservoirPosition;

    float diffuseHitT = lightDistance;
    float specularHitT = lightDistance;

    if (!isFirstPass) // DirectLightingMode::ReStir
    {
        float4 priorDiffuse = u_DiffuseLighting[lightingTexturePos];
        float4 priorSpecular = u_SpecularLighting[lightingTexturePos];

        if (calcLuminance(diffuse) > calcLuminance(priorDiffuse.rgb) || lightDistance == 0)
            diffuseHitT = priorDiffuse.w;

        if (calcLuminance(specular) > calcLuminance(priorSpecular.rgb) || lightDistance == 0)
            specularHitT = priorSpecular.w;

        diffuse += priorDiffuse.rgb;
        specular += priorSpecular.rgb;
    }

    u_DiffuseLighting[lightingTexturePos] = float4(diffuse, diffuseHitT);
    u_SpecularLighting[lightingTexturePos] = float4(specular, specularHitT);
}




[shader("miss")]
void Miss(inout RAB_RayPayload payload : SV_RayPayload)
{
}



[shader("closesthit")]
void ClosestHit(inout RAB_RayPayload payload : SV_RayPayload, in RayAttributes attrib : SV_IntersectionAttributes)
{
    payload.committedRayT = RayTCurrent();
    payload.instanceID = InstanceID();
    payload.geometryIndex = GeometryIndex();
    payload.primitiveIndex = PrimitiveIndex();
    payload.frontFace = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE;
    payload.barycentrics = attrib.uv;
}

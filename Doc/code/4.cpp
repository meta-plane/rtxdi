
m_ui.indirectLightingMode == IndirectLightingMode::ReStirGI
- m_ui.lightingSettings.directLightingMode == DirectLightingMode::None
- m_ui.lightingSettings.directLightingMode == DirectLightingMode::Brdf
- m_ui.lightingSettings.directLightingMode == DirectLightingMode::ReStir




void SceneRenderer::RenderScene(nvrhi::IFramebuffer* framebuffer)
{
    bool enableDirectReStirPass = m_ui.lightingSettings.directLightingMode == DirectLightingMode::ReStir;
    bool enableBrdfAndIndirectPass = m_ui.lightingSettings.directLightingMode == DirectLightingMode::Brdf || m_ui.indirectLightingMode != IndirectLightingMode::None;
    bool enableIndirect = m_ui.indirectLightingMode != IndirectLightingMode::None;

    LightingPasses::RenderSettings lightingSettings = GetFullyProcessedLightingSettings(
        denoiserMode, enableDirectReStirPass);

    RenderDirectLighting(enableDirectReStirPass, checkerboard, lightingSettings);
    RenderIndirectLighting(enableBrdfAndIndirectPass, enableDirectReStirPass, lightingSettings);

}


LightingPasses::RenderSettings SceneRenderer::GetFullyProcessedLightingSettings(
    uint32_t denoiserMode, bool enableDirectReStirPass)
{
    LightingPasses::RenderSettings lightingSettings = m_ui.lightingSettings;
    lightingSettings.enablePreviousTLAS &= m_ui.enableAnimations;
    lightingSettings.denoiserMode = DENOISER_MODE_OFF;
    lightingSettings.enableGradients = false;

    if (!enableDirectReStirPass) // !DirectLightingMode::ReStir
    {
        // Secondary resampling can only be done as a post-process of ReSTIR direct lighting
        lightingSettings.brdfptParams.enableSecondaryResampling = false;

        // Gradients are only produced by the direct ReSTIR pass
        lightingSettings.enableGradients = false;
    }

    return lightingSettings;
}


void SceneRenderer::RenderIndirectLighting(bool enableBrdfAndIndirectPass, bool enableDirectReStirPass, const LightingPasses::RenderSettings& lightingSettings)
{
    if (enableBrdfAndIndirectPass)
    {
        m_lightingPasses->RenderIndirectLighting(
            m_commandList,
            *m_isContext,
            m_view, m_viewPrevious, m_viewPreviousPrevious,
            lightingSettings,
            m_ui.gbufferSettings,
            *m_environmentLight,
            m_ui.indirectLightingMode, ///* enableIndirect = */ enableIndirect,
            /* enableAdditiveBlend = */ enableDirectReStirPass,
            /* enableEmissiveSurfaces = */ m_ui.lightingSettings.directLightingMode == DirectLightingMode::Brdf,
            /* enableAccumulation = */ m_ui.aaMode == AntiAliasingMode::Accumulation
        );
    }
}




void LightingPasses::RenderIndirectLighting(
    nvrhi::ICommandList* commandList, 
    rtxdi::ImportanceSamplingContext& isContext,
    const donut::engine::IView& view,
    const donut::engine::IView& previousView,
    const donut::engine::IView& previousPreviousView,
    const RenderSettings& localSettings,
    const GBufferSettings& gbufferSettings,
    const EnvironmentLight& environmentLight,
    IndirectLightingMode indirectLightingMode,
    bool enableAdditiveBlend,
    bool enableEmissiveSurfaces,
    bool enableAccumulation
    )
{
    ResamplingConstants constants = {};
    view.FillPlanarViewConstants(constants.view);
    previousView.FillPlanarViewConstants(constants.prevView);
    previousPreviousView.FillPlanarViewConstants(constants.prevPrevView);

    rtxdi::ReSTIRDIContext& restirDIContext = isContext.GetReSTIRDIContext();
    rtxdi::ReSTIRGIContext& restirGIContext = isContext.GetReSTIRGIContext();
    rtxdi::ReSTIRPTContext& restirPTContext = isContext.GetReSTIRPTContext();

    constants.denoiserMode = localSettings.denoiserMode;
    constants.enableBrdfIndirect = indirectLightingMode != IndirectLightingMode::None;
    constants.enableBrdfAdditiveBlend = enableAdditiveBlend;
    constants.enableAccumulation = enableAccumulation;
    constants.sceneConstants.enableEnvironmentMap = (environmentLight.textureIndex >= 0);
    constants.sceneConstants.environmentMapTextureIndex = (environmentLight.textureIndex >= 0) ? environmentLight.textureIndex : 0;
    constants.sceneConstants.environmentScale = environmentLight.radianceScale.x;
    constants.sceneConstants.environmentRotation = environmentLight.rotation;
    FillResamplingConstants(constants, localSettings, isContext);
    FillBRDFPTConstants(constants.brdfPT, gbufferSettings, localSettings, isContext.GetLightBufferParameters());
    constants.brdfPT.enableIndirectEmissiveSurfaces = enableEmissiveSurfaces;
    constants.brdfPT.enableReSTIRGI = indirectLightingMode == IndirectLightingMode::ReStirGI;
    constants.pt = localSettings.ptParameters;

    RTXDI_GIBufferIndices restirGIBufferIndices = restirGIContext.GetBufferIndices();
    m_currentFrameGIOutputReservoir = restirGIBufferIndices.finalShadingInputBufferIndex;

    commandList->writeBuffer(m_constantBuffer, &constants, sizeof(constants));

    dm::int2 dispatchSize = {
        view.GetViewExtent().width(),
        view.GetViewExtent().height()
    };

    if (restirDIContext.GetStaticParameters().CheckerboardSamplingMode != rtxdi::CheckerboardMode::Off)
        dispatchSize.x /= 2;
    bool brdfPassExecuted = false;
    if (localSettings.directLightingMode == DirectLightingMode::Brdf)
    {
        ExecuteRayTracingPass(commandList, m_brdfRayTracingPass, localSettings.enableRayCounts, "BrdfRayTracingPass", dispatchSize, ProfilerSection::BrdfRays);
        brdfPassExecuted = true;
    }
    if (indirectLightingMode != IndirectLightingMode::None)
    {
        // Place an explicit UAV barrier between the passes. See the note on barriers in RenderDirectLighting(...)
        nvrhi::utils::BufferUavBarrier(commandList, m_secondarySurfaceBuffer);

        if (indirectLightingMode == IndirectLightingMode::Brdf)
        {
            if (!brdfPassExecuted)
                ExecuteRayTracingPass(commandList, m_brdfRayTracingPass, localSettings.enableRayCounts, "BrdfRayTracingPass", dispatchSize, ProfilerSection::BrdfRays);
            ExecuteRayTracingPass(commandList, m_shadeSecondarySurfacesPass, localSettings.enableRayCounts, "ShadeSecondarySurfaces", dispatchSize, ProfilerSection::ShadeSecondary, nullptr);
        }
        else if (indirectLightingMode == IndirectLightingMode::ReStirGI)
        {
            if (!brdfPassExecuted)
                ExecuteRayTracingPass(commandList, m_brdfRayTracingPass, localSettings.enableRayCounts, "BrdfRayTracingPass", dispatchSize, ProfilerSection::BrdfRays);
            ExecuteRayTracingPass(commandList, m_shadeSecondarySurfacesPass, localSettings.enableRayCounts, "ShadeSecondarySurfaces", dispatchSize, ProfilerSection::ShadeSecondary, nullptr);

            m_restirGIPasses.EnableRayCounts(localSettings.enableRayCounts);
            m_restirGIPasses.SetDescriptorTable(m_scene->GetDescriptorTable());
            m_restirGIPasses.SetBindingSet(m_bindingSet);
            m_restirGIPasses.Render(commandList, view, restirGIContext, m_GIReservoirBuffer);
        }
        else if (indirectLightingMode == IndirectLightingMode::ReStirPT)
        {
            m_restirPTPasses.EnableRayCounts(localSettings.enableRayCounts);
            m_restirPTPasses.SetDescriptorTable(m_scene->GetDescriptorTable());
            m_restirPTPasses.SetBindingSet(m_bindingSet);
            m_restirPTPasses.Render(commandList, view, restirPTContext, m_PTReservoirBuffer);
        }
    }
}




void SceneRenderer::RenderScene(nvrhi::IFramebuffer* framebuffer)
{
    if (ProcessFrameStepMode(framebuffer))
        return;

    const engine::PerspectiveCamera* activeCamera = nullptr;
    uint effectiveFrameIndex = m_renderFrameIndex;

    RunBenchmarkAnimation(activeCamera, effectiveFrameIndex);
    EnforceFpsLimit();

    if (m_ui.resetISContext)
    {
        GetDevice()->waitForIdle();

        m_isContext = nullptr;
        m_rtxdiResources = nullptr;
        m_ui.resetISContext = false;
    }

    UpdateEnvironmentMap();

    m_scene->RefreshSceneGraph(GetFrameIndex());

    const auto& fbinfo = framebuffer->getFramebufferInfo();
    uint32_t renderWidth = fbinfo.width;
    uint32_t renderHeight = fbinfo.height;
    if (m_args.renderWidth > 0 && m_args.renderHeight > 0)
    {
        renderWidth = m_args.renderWidth;
        renderHeight = m_args.renderHeight;
    }
    SetupView(renderWidth, renderHeight, activeCamera);
    bool exposureResetRequired = false;
    SetupRenderPasses(renderWidth, renderHeight, exposureResetRequired);
    UpdateReSTIRDIContextFromUI();
    UpdateReGIRContextFromUI();
    UpdateReSTIRGIContextFromUI();
    UpdateReSTIRPTContextFromUI();
    m_shouldRunShaderDebugPrint = m_ui.debugOutputSettings.shaderDebugPrintOnClickOrAlways ? m_shouldRunShaderDebugPrint : true;
    m_shaderDebugPrintPass->Enable(m_ui.debugOutputSettings.enableShaderDebugPrint && m_shouldRunShaderDebugPrint);
    m_shouldRunShaderDebugPrint = m_ui.debugOutputSettings.shaderDebugPrintOnClickOrAlways ? false : true;

    AdvanceFrameForRenderPasses();

    bool cameraIsStatic = m_previousViewValid && m_view.GetViewMatrix() == m_viewPrevious.GetViewMatrix();
    ProcessAccumulationModeLogic(cameraIsStatic);

    float accumulationWeight = 1.f / (float)m_ui.numAccumulatedFrames;

    m_profiler->ResolvePreviousFrame();

    UpdateSelectedMaterialInUI();
    UpdateEnvironmentLightAndSunLight();

    m_ui.enableDenoiser = false;
    uint32_t denoiserMode = DENOISER_MODE_OFF;

    m_commandList->open();
    m_profiler->BeginFrame(m_commandList);
    m_shaderDebugPrintPass->InitializeFrame(m_commandList);

    AssignIesProfiles(m_commandList);
    m_scene->RefreshBuffers(m_commandList, GetFrameIndex());
    m_rtxdiResources->InitializeNeighborOffsets(m_commandList, m_isContext->GetNeighborOffsetCount()); // TODO Move to be next to other RTXDI passes?
    UpdateAccelerationStructure();
    EnvironmentMap();
    nvrhi::utils::ClearColorAttachment(m_commandList, framebuffer, 0, nvrhi::Color(0.f));
    GBuffer();

    // The light indexing members of frameParameters are written by PrepareLightsPass below
    rtxdi::ReSTIRDIContext& restirDIContext = m_isContext->GetReSTIRDIContext();

    UpdateRtxdiFrameIndex(effectiveFrameIndex);
    PrepareLights();
    ProcessLocalLightPdfMipMap();
    ApplyNrdCheckerboardSettings();
    CopyPSRBuffers();

    const bool checkerboard = restirDIContext.GetStaticParameters().CheckerboardSamplingMode != rtxdi::CheckerboardMode::Off;

    bool enableDirectReStirPass = m_ui.lightingSettings.directLightingMode == DirectLightingMode::ReStir;
    bool enableBrdfAndIndirectPass = m_ui.lightingSettings.directLightingMode == DirectLightingMode::Brdf || m_ui.indirectLightingMode != IndirectLightingMode::None;
    bool enableIndirect = m_ui.indirectLightingMode != IndirectLightingMode::None;

    // When indirect lighting is enabled, we don't want ReSTIR to be the NRD front-end,
    // it should just write out the raw color data.
    RTXDI_ShadingParameters restirDIShadingParams = m_isContext->GetReSTIRDIContext().GetShadingParameters();
    restirDIShadingParams.enableDenoiserInputPacking = !enableIndirect;
    m_isContext->GetReSTIRDIContext().SetShadingParameters(restirDIShadingParams);

    LightingPasses::RenderSettings lightingSettings = GetFullyProcessedLightingSettings(denoiserMode, enableDirectReStirPass);
    PresampleLights(enableDirectReStirPass, enableIndirect, lightingSettings);
    RenderDirectLighting(enableDirectReStirPass, checkerboard, lightingSettings);
    RenderIndirectLighting(enableBrdfAndIndirectPass, enableDirectReStirPass, lightingSettings);
    HandleNoLightingCase(enableDirectReStirPass, enableBrdfAndIndirectPass);
    Denoiser(lightingSettings);
    m_compositingPass->Render(m_commandList, m_view, m_viewPrevious, denoiserMode, checkerboard, m_ui, *m_environmentLight);
    TransparentGeometry();
    ResolveAA(m_commandList, accumulationWeight);
    Bloom();
    ReferenceImage(cameraIsStatic);
    ToneMapping(exposureResetRequired);
    DebugPathViz();
    FinalFramebufferOutput(framebuffer);

    m_shaderDebugPrintPass->ReadBackPrintData(m_commandList);

    m_profiler->EndFrame(m_commandList);

    m_commandList->close();
    GetDevice()->executeCommandList(m_commandList);

    ProcessScreenshotFrame();
    OutputShaderMessages();

    m_ui.gbufferSettings.enableMaterialReadback = false;

    if (m_ui.enableAnimations)
        m_framesSinceAnimation = 0;
    else
        m_framesSinceAnimation++;

    m_viewPreviousPrevious = m_viewPrevious;
    m_viewPrevious = m_view;
    m_previousViewValid = true;
    m_ui.resetAccumulation = false;
    ++m_renderFrameIndex;
}
# RTXDI FullSample - 1 Frame Resource Flow Diagram

## Pass Dispatch Type

| Pass | Shader | Type | Ray Trace |
|------|--------|------|-----------|
| PASS 0 | RaytracedGBuffer.hlsl | Compute 또는 RayGen (설정에 따름) | Yes (primary ray) |
| PASS 0 | RasterizedGBuffer.hlsl | vkCmdDraw (VS/PS Rasterization) | No |
| PASS 1a | PresampleLights.hlsl | Compute (항상) | No |
| PASS 1b | PresampleEnvironmentMap.hlsl | Compute (항상) | No |
| PASS 1c | PresampleReGIR.hlsl | Compute (항상) | No |
| PASS 2 | DI/GenerateInitialSamples.hlsl | Compute 또는 RayGen (설정에 따름) | No |
| PASS 3 | DI/TemporalResampling.hlsl | Compute 또는 RayGen (설정에 따름) | No |
| PASS 4 | DI/SpatialResampling.hlsl | Compute 또는 RayGen (설정에 따름) | No |
| PASS 5 | DI/ShadeSamples.hlsl | RayGen (항상) | Yes (shadow ray) |
| PASS 6 | BrdfRayTracing.hlsl | Compute 또는 RayGen (설정에 따름) | Yes (BRDF ray) |
| PASS 7 | ShadeSecondarySurfaces.hlsl | Compute 또는 RayGen (설정에 따름) | Yes (shadow ray) |
| PASS 8 | GI/TemporalResampling.hlsl | Compute 또는 RayGen (설정에 따름) | No |
| PASS 9 | GI/SpatialResampling.hlsl | Compute 또는 RayGen (설정에 따름) | No |
| PASS 10 | GI/FinalShading.hlsl | RayGen (항상) | Yes (visibility ray) |

## Data Flow

```
                    PASS 0: G-Buffer (RaytracedGBuffer.hlsl)
                    ┌──────────────────────────────────────────────┐
                    │ Depth, Normal, GeoNormal,                    │
                    │ DiffuseAlbedo, SpecularRough                 │
                    │ + Emissive, MotionVectors, NormalRoughness   │
                    └────┬───────────────┬───────────────┬─────────┘
                         │               │               │
          ┌──────────────┘               │               └────────────────┐
          ▼                              ▼                                ▼
   ┌──────────────┐              ┌──────────────┐                 ┌──────────────┐
   │ PASS 1a:     │              │ PASS 1b:     │                 │ PASS 1c:     │
   │ Presample    │              │ Presample    │                 │ Presample    │
   │ Lights       │              │ EnvMap       │                 │ ReGIR        │
   │ (Presample   │              │ (Presample   │                 │ (Presample   │
   │ Lights.hlsl) │              │ Environment  │                 │ ReGIR.hlsl)  │
   │              │              │ Map.hlsl)    │                 │              │
   └──────┬───────┘              └──────┬───────┘                 └──────┬───────┘
          │ RisBuffer                   │ RisBuffer                      │ RisBuffer
          │ RisLightDataBuffer          │                                │
          └──────────────┬──────────────┴────────────────────────────────┘
                         ▼
             PASS 2: DI InitialSamples (DI/GenerateInitialSamples.hlsl)
                                               DI: 3-slot, L=lastFrameOutputReservoir
                    │ DIReservoirBuffer [(L+1)%3] WRITE
                    ▼
             PASS 3: DI TemporalResampling (DI/TemporalResampling.hlsl)
                    │ DIReservoirBuffer [(L+1)%3] READ (current) + WRITE (in-place)
                    │ DIReservoirBuffer [L]       READ (previous frame)
                    ▼
             PASS 4: DI SpatialResampling (DI/SpatialResampling.hlsl)
                    │ DIReservoirBuffer [(L+1)%3] READ
                    │ DIReservoirBuffer [(L+2)%3] WRITE
                    ▼
             PASS 5: DI ShadeSamples (DI/ShadeSamples.hlsl)
                    │ DIReservoirBuffer [(L+2)%3] READ + visibility update
                    ├──→ DiffuseLighting  ← DI direct (first write)
                    ├──→ SpecularLighting ← DI direct (first write)
                    └──→ RestirLuminance
                              │
                    ▼─────────┘
             PASS 6: BrdfRayTracing (BrdfRayTracing.hlsl)
                    ├──→ SecondaryGBuffer WRITE
                    └──→ PSR Textures WRITE
                              │
                    ▼─────────┘
             PASS 7: ShadeSecondarySurfaces (ShadeSecondarySurfaces.hlsl)
                                               GI: 2-slot fixed (TemporalAndSpatial)
                    │ SecondaryGBuffer READ (2nd surface) → WRITE (.emission)
                    │ DIReservoirBuffer READ (secondary resampling)
                    ├──→ GIReservoirBuffer [0] WRITE (initial GI reservoir)
                    └──→ Diffuse/SpecularLighting (delta surface only)
                              │
                    ▼─────────┘
             PASS 8: GI TemporalResampling (GI/TemporalResampling.hlsl)
                    │ GIReservoirBuffer [0] READ (current) + WRITE (in-place)
                    │ GIReservoirBuffer [1] READ (previous frame)
                    ▼
             PASS 9: GI SpatialResampling (GI/SpatialResampling.hlsl)
                    │ GIReservoirBuffer [0] READ
                    │ GIReservoirBuffer [1] WRITE
                    ▼
             PASS 10: GI FinalShading (GI/FinalShading.hlsl)
                    │ GIReservoirBuffer [1] READ
                    │ SecondaryGBuffer READ (MIS weight)
                    ├──→ DiffuseLighting  (+= GI indirect, additive)
                    └──→ SpecularLighting (+= GI indirect, additive)
                              │
                    ▼─────────┘
             PASS 11: NRD Denoiser
                    │ DiffuseLighting, SpecularLighting READ
                    │ Depth, NormalRoughness, MotionVectors READ
                    ├──→ DenoisedDiffuseLighting WRITE
                    └──→ DenoisedSpecularLighting WRITE
                              │
                    ▼─────────┘
             PASS 12: Compositing (CompositingPass.hlsl)
                    │ GBuffer + Denoised Lighting + PSR READ
                    └──→ HdrColor WRITE
                              │
                    ▼─────────┘
             PASS 13~15: Glass → TAA → ToneMapping (GlassPass.hlsl)
                    └──→ LdrColor → Swapchain Present
```

## Reservoir Slot Rotation

### DI (3-slot, L = lastFrameOutputReservoir)

DI는 Gradient 패스가 이전/현재 프레임 reservoir을 동시에 읽어야 하므로 3슬롯 필요.

```cpp
// ReSTIRDI.cpp:272-285 (TemporalAndSpatial mode, L = m_lastFrameOutputReservoir)
m_bufferIndices.initialSamplingOutputBufferIndex   = (L+1) % 3;
m_bufferIndices.temporalResamplingInputBufferIndex = L;
m_bufferIndices.temporalResamplingOutputBufferIndex= (L+1) % 3;  // in-place with initial
m_bufferIndices.spatialResamplingInputBufferIndex  = (L+1) % 3;
m_bufferIndices.spatialResamplingOutputBufferIndex = (L+2) % 3;
m_bufferIndices.shadingInputBufferIndex            = (L+2) % 3;
m_currentFrameOutputReservoir                      = (L+2) % 3;
```

```
L: 0 → 2 → 1 → 0 → 2 → 1 → ...

Frame f  :  prev=0  init/temp=1  spatial/shade=2
Frame f+1:  prev=2  init/temp=0  spatial/shade=1
Frame f+2:  prev=1  init/temp=2  spatial/shade=0
```

### GI TemporalAndSpatial (2-slot, 고정)

GI에는 Gradient 패스가 없으므로 spatial이 prev 슬롯을 덮어써도 됨. 2슬롯으로 충분.

```cpp
// ReSTIRGI.cpp:207-214
case rtxdi::ReSTIRGI_ResamplingMode::TemporalAndSpatial:
    m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex = 0;
    m_bufferIndices.temporalResamplingInputBufferIndex = 1;
    m_bufferIndices.temporalResamplingOutputBufferIndex = 0;
    m_bufferIndices.spatialResamplingInputBufferIndex = 0;
    m_bufferIndices.spatialResamplingOutputBufferIndex = 1;
    m_bufferIndices.finalShadingInputBufferIndex = 1;
    break;
```

```
Frame f  :  prev=1  init/temp=0  spatial/shade=1
Frame f+1:  prev=1  init/temp=0  spatial/shade=1
Frame f+2:  prev=1  init/temp=0  spatial/shade=1
```

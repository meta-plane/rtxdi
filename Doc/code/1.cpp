
struct ResamplingConstants
{
    PlanarViewConstants view;
    PlanarViewConstants prevView;
    PlanarViewConstants prevPrevView;
    RTXDI_RuntimeParameters runtimeParams;
    
    float4 reblurHitDistParams;

    uint pad3;
    uint enablePreviousTLAS;
    uint denoiserMode;
    uint discountNaiveSamples;
    
    uint enableBrdfIndirect;
    uint enableBrdfAdditiveBlend;    
    uint enableAccumulation; // StoreShadingOutput
    DirectLightingMode directLightingMode;

    SceneConstants sceneConstants;

    // Common buffer params
    RTXDI_LightBufferParameters lightBufferParams;
    RTXDI_RISBufferSegmentParameters localLightsRISBufferSegmentParams;
    RTXDI_RISBufferSegmentParameters environmentLightRISBufferSegmentParams;

    // Algo-specific params
    RTXDI_Parameters restirDI;
    ReGIR_Parameters regir;
    RTXDI_GIParameters restirGI;
    RTXDI_PTParameters restirPT;
    PTParameters pt;
    BRDFPathTracing_Parameters brdfPT;

    uint visualizeRegirCells;
    uint enableDenoiserPSR;
    uint usePSRMvecForResampling;
    uint updatePSRwithResampling;
    
    uint2 environmentPdfTextureSize;
    uint2 localLightPdfTextureSize;

    ReSTIRShaderDebugParameters debug;
};

struct RTXDI_LightBufferRegion
{
    uint32_t firstLightIndex;
    uint32_t numLights;
};

enum class PolymorphicLightType
{
    kSphere = 0,
    kCylinder,
    kDisk,
    kRect,
    kTriangle,
    kDirectional,
    kEnvironment,
    kPoint
};

// Stores shared light information (type) and specific light information
// See PolymorphicLight.hlsli for encoding format
typedef struct PolymorphicLightInfo
{
    // uint4[0]
    float3 center;
    uint colorTypeAndFlags; // RGB8 + uint8 (see the kPolymorphicLight... constants above)

    // uint4[1]
    uint direction1; // oct-encoded
    uint direction2; // oct-encoded
    uint scalars; // 2x float16
    uint logRadiance; // uint16

    // uint4[2] -- optional, contains only shaping data
    uint iesProfileIndex;
    uint primaryAxis; // oct-encoded
    uint cosConeAngleAndSoftness; // 2x float16
    uint padding;
} RAB_LightInfo;


// Structure that groups the parameters for RTXDI_GetReservoirVisibility(...)
// Reusing final visibility reduces the number of high-quality shadow rays needed to shade
// the scene, at the cost of somewhat softer or laggier shadows.
struct RTXDI_VisibilityReuseParameters
{
    // Controls the maximum age of the final visibility term, measured in frames, that can be reused from the
    // previous frame(s). Higher values result in better performance.
    uint maxAge;

    // Controls the maximum distance in screen space between the current pixel and the pixel that has
    // produced the final visibility term. The distance does not include the motion vectors.
    // Higher values result in better performance and softer shadows.
    float maxDistance;
};


// This structure represents a single light reservoir that stores the weights, the sample ref,
// sample count (M), and visibility for reuse. It can be serialized into RTXDI_PackedDIReservoir for storage.
struct RTXDI_DIReservoir
{
    // Light index (bits 0..30) and validity bit (31)
    uint lightData;

    // Sample UV encoded in 16-bit fixed point format
    uint uvData;

    // Overloaded: represents RIS weight sum during streaming,
    // then reservoir weight (inverse PDF) after FinalizeResampling
    float weightSum;

    // Target PDF of the selected sample
    float targetPdf;

    // Number of samples considered for this reservoir (pairwise MIS makes this a float)
    float M;

    // Visibility information stored in the reservoir for reuse
    uint packedVisibility;

    // Screen-space distance between the current location of the reservoir
    // and the location where the visibility information was generated,
    // minus the motion vectors applied in temporal resampling
    int2 spatialDistance;

    // How many frames ago the visibility information was generated
    uint age;

    // Cannonical weight when using pairwise MIS (ignored except during pairwise MIS computations)
    float canonicalWeight;
};

// This structure represents a indirect lighting reservoir that stores the radiance and weight
// as well as its the position where the radiane come from.
struct RTXDI_GIReservoir
{
    // postion of the 2nd bounce surface.
    float3 position;

    // normal vector of the 2nd bounce surface.
    float3 normal;

    // incoming radiance from the 2nd bounce surface.
    float3 radiance;

    // Overloaded: represents RIS weight sum during streaming,
    // then reservoir weight (inverse PDF) after FinalizeResampling
    float weightSum;

    // Number of samples considered for this reservoir
    uint M;

    // Number of frames the chosen sample has survived.
    uint age;
};

struct RTXDI_PackedGIReservoir
{
#ifdef __cplusplus
    using float3 = float[3];
#endif

    float3      position;
    uint32_t    packed_miscData_age_M; // See Reservoir.hlsli about the detail of the bit field.

    uint32_t    packed_radiance;    // Stored as 32bit LogLUV format.
    float       weight;
    uint32_t    packed_normal;      // Stored as 2x 16-bit snorms in the octahedral mapping
    float       unused;
};

RTXDI_DIReservoir RTXDI_EmptyDIReservoir()
{
    RTXDI_DIReservoir s;
    s.lightData = 0;
    s.uvData = 0;
    s.weightSum = 0;
    s.targetPdf = 0;
    s.M = 0;
    s.packedVisibility = 0;
    s.spatialDistance = int2(0, 0);
    s.age = 0;
    s.canonicalWeight = 0;
    return s;
}

struct RAB_LightSample
{
    float3 position;
    float3 normal;
    float3 radiance;
    float solidAnglePdf;
    PolymorphicLightType lightType;
};

struct RTXDI_DIInitialSamplingParameters
{
    uint32_t numLocalLightSamples;
    uint32_t numInfiniteLightSamples;
    uint32_t numEnvironmentSamples;
    uint32_t numBrdfSamples;

    float brdfCutoff;
    float brdfRayMinT;
    ReSTIRDI_LocalLightSamplingMode localLightSamplingMode;
    uint32_t enableInitialVisibility;

    uint32_t environmentMapImportanceSampling; // Only used in InitialSamplingFunctions.hlsli via RAB_EvaluateEnvironmentMapSamplingPdf
};

struct SecondaryGBufferData
{
    float3 worldPos;
    uint normal;

    uint2 throughputAndFlags;   // .x = throughput.rg as float16, .y = throughput.b as float16, flags << 16
    uint diffuseAlbedo;         // R11G11B10_UFLOAT
    uint specularAndRoughness;  // R8G8B8A8_Gamma_UFLOAT
    
    float3 emission;
    float pdf;
};

struct RAB_Surface
{
    float3 worldPos;
    float3 viewDir;
    float3 normal;
    float3 geoNormal;
    float viewDepth;
    float diffuseProbability;
    RAB_Material material;
};
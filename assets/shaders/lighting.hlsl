Texture2D gBuffer0 : register(t0);
Texture2D gBuffer1 : register(t1);

SamplerState linearSampler : register(s0);

cbuffer Light : register(b0)
{
    float3 lightDir;
    float pad0;
    float3 lightColor;
    float pad1;
};

struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

VSOutput VSMain(uint id : SV_VertexID)
{
    const float2 triangleVertices[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};

    VSOutput output;

    output.position = float4(triangleVertices[id], float2(0.0f, 1.0f));
    output.uv = float2(output.position.x, -output.position.y) * 0.5f + 0.5f;

    return output;
}

float4 PSMain(VSOutput input) : SV_Target
{
    float4 albedoMetalness = gBuffer0.Sample(linearSampler, input.uv);
    float4 normalRoughness = gBuffer1.Sample(linearSampler, input.uv);

    float3 albedo = pow(albedoMetalness.rgb, 2.2f);  // convert from srgb to linear
    float metalness = albedoMetalness.a;
    float3 normal = normalize(normalRoughness.rgb * 2 - 1);
    float roughness = normalRoughness.a;

    float3 normalizedReflectedLightDir = normalize(-lightDir);
    float brightness = saturate(dot(normal, normalizedReflectedLightDir));

    float3 ambient = 0.3f;

    float3 color = albedo * (ambient + lightColor * brightness);
    color = pow(color, 1.0 / 2.2);

    return float4(color, 1.0f);
}
cbuffer ConstantBuffer : register(b0)
{
    float4x4 model;
    float4x4 view;
    float4x4 projection;
};

struct VSInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
};

struct VSOutput
{
    float4 position : SV_Position;
    float3 normal : NORMAL;
};

struct PSOut
{
    float4 g0 : SV_Target0;  // albedo.rgb + metalness.a
    float4 g1 : SV_Target1;  // normal.rgb + roughness.a

};

VSOutput VSMain(VSInput input)
{
    VSOutput output;
    
    // transform position via MVP
    float4 worldPosition = mul(float4(input.position, 1.0f), model);
    float4 viewPosition = mul(worldPosition, view);
    output.position = mul(viewPosition, projection);
    
    // transform normal to world space
    output.normal = mul(float4(input.normal, 0.0f), model).xyz;

    return output;
}

PSOut PSMain(VSOutput input)
{
    PSOut output;

    float3 albedo = float3(0.8, 0.2, 0.2);
    float metalness = 0.0;
    float3 worldNormal = normalize(input.normal);
    float roughness = 0.5;

    output.g0 = float4(albedo, metalness);
    output.g1 = float4(worldNormal * 0.5 + 0.5, roughness);

    return output;
}
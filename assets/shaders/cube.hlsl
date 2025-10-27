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

VSOutput VSMain(VSInput input)
{
    VSOutput output;

    // transform position via MVP
    float4 worldPos = mul(float4(input.position, 1.0f), model);
    float4 viewPos = mul(worldPos, view);
    output.position = mul(viewPos, projection);
    
    // transform normal to world space
    output.normal = mul(float4(input.normal, 0.0f), model).xyz;
    
    return output;
}

float4 PSMain(VSOutput input) : SV_Target
{
    float3 normal = normalize(input.normal);
    float lighting = max(dot(normal, float3(0, 0, -1)), 0.3f);

    return float4(lighting, lighting, lighting, 1.0f);
}
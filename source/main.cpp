#include <iostream>
#include <exception>
#include <stdexcept>
#include <string>
#include <map>
#include <utility>
#include <vector>

#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>

#include <DirectXMath.h>

#include <wrl/client.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

struct Vertex
{
    DirectX::XMFLOAT3 position;
    DirectX::XMFLOAT3 normal;
};

struct Transform
{
    DirectX::XMFLOAT3 position;
    DirectX::XMFLOAT3 orientation;
    DirectX::XMFLOAT3 scale;
};

struct Mesh
{
    std::vector<Vertex> vertices;
    std::vector<UINT> indices;
};

struct RenderableMeshDesc
{
    UINT vertexBufferOffset;
    UINT numVertices;
    
    UINT indexBufferOffset;
    UINT numIndices;
    
    UINT constantBufferOffset;

    Transform transform;
};

struct GeometryConstantBuffer
{
    DirectX::XMFLOAT4X4 modelMatrix;
    DirectX::XMFLOAT4X4 viewMatrix;
    DirectX::XMFLOAT4X4 projectionMatrix;
};

struct LightingConstantBuffer
{
    DirectX::XMFLOAT3 lightDirection;
    float pad0;
    DirectX::XMFLOAT3 lightColor;
    float pad1;
};

static constexpr UINT windowWidth = 1280;
static constexpr UINT windowHeight = 720;
static bool windowIsVisible = true;
static bool windowIsWindowedMode = false;
static GLFWwindow* window = nullptr;

static constexpr std::size_t numBackBuffers = 3;
static constexpr std::size_t numGBufferComponents = 2;
static constexpr std::size_t numRtvDescriptors = 255;
static constexpr std::size_t numSrvDescriptors = 255;
static constexpr std::size_t numDsvDesciptors = 1;

static constexpr D3D12_VIEWPORT viewport{0.0f, 0.0f, static_cast<float>(windowWidth), static_cast<float>(windowHeight)};
static constexpr D3D12_RECT scissorRect{0, 0, windowWidth, windowHeight};
static Microsoft::WRL::ComPtr<ID3D12Device> device;
static Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue;
static Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocators[numBackBuffers] = {};
static Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList1> commandLists[numBackBuffers] = {};
static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtvDescriptorHeap;
static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> dsvDescriptorHeap;
static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srvDescriptorHeap;
static Microsoft::WRL::ComPtr<IDXGISwapChain3> swapChain;
static Microsoft::WRL::ComPtr<ID3D12Resource> backBuffers[numBackBuffers] = {};
static D3D12_CPU_DESCRIPTOR_HANDLE backBuffersRtvCpuDescriptorHandles[numBackBuffers] = {};
static Microsoft::WRL::ComPtr<ID3D12Resource> gBuffer[numGBufferComponents] = {};
static D3D12_CPU_DESCRIPTOR_HANDLE gBufferRtvCpuDescriptorHandles[numGBufferComponents] = {};
static D3D12_CPU_DESCRIPTOR_HANDLE gBufferSrvCpuDescriptorHandles[numGBufferComponents] = {};
static D3D12_GPU_DESCRIPTOR_HANDLE gBufferSrvGpuDescriptorHandles[numGBufferComponents] = {};
static Microsoft::WRL::ComPtr<ID3D12Resource> depthBuffer;
static D3D12_CPU_DESCRIPTOR_HANDLE depthBufferDsvCpuDescriptorHandle{};
static Microsoft::WRL::ComPtr<ID3D12Fence> fence;
static UINT64 frameIndex = 0;
static HANDLE fenceEvent;
static UINT64 fenceValues[numBackBuffers] = {};

static Microsoft::WRL::ComPtr<ID3D12RootSignature> geometryRootSignature;
static Microsoft::WRL::ComPtr<ID3D12RootSignature> lightingRootSignature;
static Microsoft::WRL::ComPtr<ID3D12PipelineState> geometryPipelineState;
static Microsoft::WRL::ComPtr<ID3D12PipelineState> lightingPipelineState;

static std::vector<Mesh> meshes;
static std::vector<RenderableMeshDesc> renderableMeshDescs;
static Microsoft::WRL::ComPtr<ID3D12Resource> unifiedVertexBuffer;
static Microsoft::WRL::ComPtr<ID3D12Resource> unifiedIndexBuffer;
static D3D12_VERTEX_BUFFER_VIEW unifiedVertexBufferView{};
static D3D12_INDEX_BUFFER_VIEW unifiedIndexBufferView{};

static Microsoft::WRL::ComPtr<ID3D12Resource> geometryConstantBuffers[numBackBuffers] = {};
static Microsoft::WRL::ComPtr<ID3D12Resource> lightingConstantBuffers[numBackBuffers] = {};
static void* geometryConstantBuffersMappedMemory[numBackBuffers] = {};
static void* lightingConstantBuffersMappedMemory[numBackBuffers] = {};

inline static void ThrowIfFailed(HRESULT hResult);

static DirectX::XMMATRIX CalculateModelMatrix(const Transform& transform);

// todo:
static void LoadPipeline(HWND hWnd);
static void LoadAssets();
static void LoadObjFile(const std::string& filename);

int main()
{
    try
    {
        if (!glfwInit())
        {
            throw std::runtime_error{"failed to initialize GLFW"};
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(windowWidth, windowHeight, "D3D12 Renderer", nullptr, nullptr);
        if (window == nullptr)
        {
            throw std::runtime_error{"failed to create window"};
        }

        HWND hWnd = glfwGetWin32Window(window);
        LoadPipeline(hWnd);
        LoadAssets();

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();

            const UINT currentBackBufferIndex = swapChain->GetCurrentBackBufferIndex();

            if (fence->GetCompletedValue() < fenceValues[currentBackBufferIndex])
            {
                ThrowIfFailed(fence->SetEventOnCompletion(fenceValues[currentBackBufferIndex], fenceEvent));
                WaitForSingleObject(fenceEvent, INFINITE);
            }

            ThrowIfFailed(commandAllocators[currentBackBufferIndex]->Reset());
            ThrowIfFailed(commandLists[currentBackBufferIndex]->Reset(commandAllocators[currentBackBufferIndex].Get(), nullptr));

            ID3D12Resource* backBuffer = backBuffers[currentBackBufferIndex].Get();

            // set-up geometry pipeline to draw g-buffer
            commandLists[currentBackBufferIndex]->RSSetViewports(1, &viewport);
            commandLists[currentBackBufferIndex]->RSSetScissorRects(1, &scissorRect);
            commandLists[currentBackBufferIndex]->OMSetRenderTargets(2, gBufferRtvCpuDescriptorHandles, FALSE, &depthBufferDsvCpuDescriptorHandle);

            // bind unified vertex buffer
            commandLists[currentBackBufferIndex]->IASetVertexBuffers(0, 1, &unifiedVertexBufferView);

            // bind unified index buffer
            commandLists[currentBackBufferIndex]->IASetIndexBuffer(&unifiedIndexBufferView);

            const float gBuffer0ClearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const float gBuffer1ClearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            commandLists[currentBackBufferIndex]->ClearRenderTargetView(gBufferRtvCpuDescriptorHandles[0], gBuffer0ClearColor, 0, nullptr);
            commandLists[currentBackBufferIndex]->ClearRenderTargetView(gBufferRtvCpuDescriptorHandles[1], gBuffer1ClearColor, 0, nullptr);
            commandLists[currentBackBufferIndex]->ClearDepthStencilView(depthBufferDsvCpuDescriptorHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
            commandLists[currentBackBufferIndex]->SetPipelineState(geometryPipelineState.Get());
            commandLists[currentBackBufferIndex]->SetGraphicsRootSignature(geometryRootSignature.Get());
            commandLists[currentBackBufferIndex]->SetGraphicsRootConstantBufferView(0, geometryConstantBuffers[currentBackBufferIndex]->GetGPUVirtualAddress());
            commandLists[currentBackBufferIndex]->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            // calculate orientation for each renderable mesh
            static float time = 0.0f;
            time += 0.01f;
            for (RenderableMeshDesc& renderableMeshDesc : renderableMeshDescs)
            {
                renderableMeshDesc.transform.orientation.y = time;
            }

            // render each renderable mesh
            for (const RenderableMeshDesc& renderableMeshDesc : renderableMeshDescs)
            {
                // calculate MVP matrices
                DirectX::XMMATRIX model = CalculateModelMatrix(renderableMeshDesc.transform);
                DirectX::XMMATRIX view = DirectX::XMMatrixLookAtLH(
                    DirectX::XMVectorSet(0.0f, 1.0f, -8.0f, 1.0f),
                    DirectX::XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f),
                    DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f)
                );
                DirectX::XMMATRIX projection = DirectX::XMMatrixPerspectiveFovLH(
                    DirectX::XM_PIDIV4,
                    static_cast<float>(windowWidth) / static_cast<float>(windowHeight),
                    0.1f,
                    100.0f
                );

                model = DirectX::XMMatrixTranspose(model);
                view = DirectX::XMMatrixTranspose(view);
                projection = DirectX::XMMatrixTranspose(projection);

                GeometryConstantBuffer geometryConstantBuffer{};
                DirectX::XMStoreFloat4x4(&(geometryConstantBuffer.modelMatrix), model);
                DirectX::XMStoreFloat4x4(&(geometryConstantBuffer.viewMatrix), view);
                DirectX::XMStoreFloat4x4(&(geometryConstantBuffer.projectionMatrix), projection);

                // upload MVP matrices
                std::memcpy(static_cast<char*>(geometryConstantBuffersMappedMemory[currentBackBufferIndex]) + renderableMeshDesc.constantBufferOffset, &geometryConstantBuffer, sizeof(GeometryConstantBuffer));

                commandLists[currentBackBufferIndex]->SetGraphicsRootConstantBufferView(0, geometryConstantBuffers[currentBackBufferIndex]->GetGPUVirtualAddress() + renderableMeshDesc.constantBufferOffset);

                commandLists[currentBackBufferIndex]->DrawIndexedInstanced(renderableMeshDesc.numIndices, 1, renderableMeshDesc.indexBufferOffset, renderableMeshDesc.vertexBufferOffset, 0);
            }

            // transition barriers to prepare for lighting pass

            // gbuffer0 : from render target to pixel shader resource
            D3D12_RESOURCE_BARRIER gBuffer0ResourceBarrier{};
            gBuffer0ResourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            gBuffer0ResourceBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            gBuffer0ResourceBarrier.Transition.pResource = gBuffer[0].Get();
            gBuffer0ResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            gBuffer0ResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            gBuffer0ResourceBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            
            // gbuffer1 : from render target to pixel shader resource
            D3D12_RESOURCE_BARRIER gBuffer1ResourceBarrier{};
            gBuffer1ResourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            gBuffer1ResourceBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            gBuffer1ResourceBarrier.Transition.pResource = gBuffer[1].Get();
            gBuffer1ResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            gBuffer1ResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            gBuffer1ResourceBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            
            // depth buffer : from write to read
            D3D12_RESOURCE_BARRIER depthBufferResourceBarrier{};
            depthBufferResourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            depthBufferResourceBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            depthBufferResourceBarrier.Transition.pResource = depthBuffer.Get();
            depthBufferResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_WRITE;
            depthBufferResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_DEPTH_READ;
            depthBufferResourceBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            
            // current back buffer : from present to render target
            D3D12_RESOURCE_BARRIER backBufferResourceBarrier{};
            backBufferResourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            backBufferResourceBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            backBufferResourceBarrier.Transition.pResource = backBuffer;
            backBufferResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
            backBufferResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;

            const D3D12_RESOURCE_BARRIER geometryExitTransitionBarriers[] = {gBuffer0ResourceBarrier, gBuffer1ResourceBarrier, depthBufferResourceBarrier, backBufferResourceBarrier};

            commandLists[currentBackBufferIndex]->ResourceBarrier(4, geometryExitTransitionBarriers);

            // set-up lighting pipeline to draw final image
            const float clearColor[] = {0.39f, 0.58f, 0.93f, 1.0f};

            commandLists[currentBackBufferIndex]->OMSetRenderTargets(1, &(backBuffersRtvCpuDescriptorHandles[currentBackBufferIndex]), FALSE, nullptr);
            commandLists[currentBackBufferIndex]->ClearRenderTargetView(backBuffersRtvCpuDescriptorHandles[currentBackBufferIndex], clearColor, 0, nullptr);

            ID3D12DescriptorHeap* descriptorHeaps[] = {srvDescriptorHeap.Get()};

            commandLists[currentBackBufferIndex]->SetDescriptorHeaps(1, descriptorHeaps);
            commandLists[currentBackBufferIndex]->SetPipelineState(lightingPipelineState.Get());
            commandLists[currentBackBufferIndex]->SetGraphicsRootSignature(lightingRootSignature.Get());
            commandLists[currentBackBufferIndex]->SetGraphicsRootDescriptorTable(0, gBufferSrvGpuDescriptorHandles[0]);
            commandLists[currentBackBufferIndex]->SetGraphicsRootConstantBufferView(1, lightingConstantBuffers[currentBackBufferIndex]->GetGPUVirtualAddress());
            commandLists[currentBackBufferIndex]->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            commandLists[currentBackBufferIndex]->DrawInstanced(3, 1, 0, 0);

            // transitions to prepare for next frame's geometry pass
            
            // gbuffer0 : from pixel shader resource to render target
            gBuffer0ResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            gBuffer0ResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            
            // gbuffer1 : from pixel shader resource to render target
            gBuffer1ResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            gBuffer1ResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            
            // depth buffer : from read to write
            depthBufferResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_READ;
            depthBufferResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_DEPTH_WRITE;
            
            // current back buffer : from render target to present
            backBufferResourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            backBufferResourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;

            const D3D12_RESOURCE_BARRIER lightingExitTranitionBarriers[] = {gBuffer0ResourceBarrier, gBuffer1ResourceBarrier, depthBufferResourceBarrier, backBufferResourceBarrier};

            commandLists[currentBackBufferIndex]->ResourceBarrier(4, lightingExitTranitionBarriers);

            ThrowIfFailed(commandLists[currentBackBufferIndex]->Close());

            ID3D12CommandList* commandList[] = {commandLists[currentBackBufferIndex].Get()};

            commandQueue->ExecuteCommandLists(1, commandList);

            ThrowIfFailed(swapChain->Present(1, 0));

            ++frameIndex;
            ThrowIfFailed(commandQueue->Signal(fence.Get(), frameIndex));
            fenceValues[currentBackBufferIndex] = frameIndex;
        }

        CloseHandle(fenceEvent);

        // todo:
        glfwDestroyWindow(window);
        
        glfwTerminate();

        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return -1;
    }
}

inline static void ThrowIfFailed(HRESULT hResult)
{
    if (FAILED(hResult))
    {
        throw std::exception{"Win32 Function Failed"};
    }
}

static DirectX::XMMATRIX CalculateModelMatrix(const Transform& transform)
{
    DirectX::XMMATRIX scaleMatrix = DirectX::XMMatrixScaling((transform.scale).x, (transform.scale).y, (transform.scale).z);
    DirectX::XMMATRIX rotationMatrix = DirectX::XMMatrixRotationRollPitchYaw((transform.orientation).x, (transform.orientation).y, (transform.orientation).z);
    DirectX::XMMATRIX translationMatrix = DirectX::XMMatrixTranslation((transform.position).x, (transform.position).y, (transform.position).z);

    return (scaleMatrix * rotationMatrix * translationMatrix);
}

static void LoadPipeline(HWND hWnd)
{
#if defined(_DEBUG)
{
    Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
    ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(debugController.GetAddressOf())));

    Microsoft::WRL::ComPtr<ID3D12Debug1> debug1Controller;
    ThrowIfFailed(debugController.As(&debug1Controller));

    debug1Controller->SetEnableGPUBasedValidation(TRUE);
    debugController->EnableDebugLayer();
}
#endif

    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf())));

    Microsoft::WRL::ComPtr<IDXGIAdapter1> hardwardAdapter;
    // todo:

    // create device
    ThrowIfFailed(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.GetAddressOf())));

    // create command queue
    D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
    commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    ThrowIfFailed(device->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(commandQueue.GetAddressOf())));

    // create command allocators and command lists
    for (UINT i = 0; i < numBackBuffers; ++i)
    {
        ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(commandAllocators[i].GetAddressOf())));

        ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocators[i].Get(), nullptr, IID_PPV_ARGS(commandLists[i].GetAddressOf())));
        ThrowIfFailed(commandLists[i]->Close());
    }

    // create RTV descriptor heap
    D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc{};
    rtvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDescriptorHeapDesc.NumDescriptors = numRtvDescriptors;

    ThrowIfFailed(device->CreateDescriptorHeap(&rtvDescriptorHeapDesc, IID_PPV_ARGS(rtvDescriptorHeap.GetAddressOf())));

    // create DSV descriptor heap
    D3D12_DESCRIPTOR_HEAP_DESC dsvDescriptorHeapDesc{};
    dsvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvDescriptorHeapDesc.NumDescriptors = numDsvDesciptors;

    ThrowIfFailed(device->CreateDescriptorHeap(&dsvDescriptorHeapDesc, IID_PPV_ARGS(dsvDescriptorHeap.GetAddressOf())));

    // create SRV descriptor heap
    D3D12_DESCRIPTOR_HEAP_DESC srvDescriptorHeapDesc{};
    srvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    srvDescriptorHeapDesc.NumDescriptors = numSrvDescriptors;
    srvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

    ThrowIfFailed(device->CreateDescriptorHeap(&srvDescriptorHeapDesc, IID_PPV_ARGS(srvDescriptorHeap.GetAddressOf())));

    // create swap chain
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc1{};
    swapChainDesc1.BufferCount = numBackBuffers;
    swapChainDesc1.Width = windowWidth;
    swapChainDesc1.Height = windowHeight;
    swapChainDesc1.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc1.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc1.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc1.SampleDesc = {1, 0};

    Microsoft::WRL::ComPtr<IDXGISwapChain1> swapChain1;
    ThrowIfFailed(factory->CreateSwapChainForHwnd(commandQueue.Get(), hWnd, &swapChainDesc1, nullptr, nullptr, swapChain1.GetAddressOf()));

    ThrowIfFailed(swapChain1.As(&swapChain));

    // create swap chain RTVs
    const UINT rtvDescriptorHandleIncrementSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    const D3D12_CPU_DESCRIPTOR_HANDLE rtvDescriptorStartHandle = rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < numBackBuffers; ++i)
    {
        ThrowIfFailed(swapChain->GetBuffer(i, IID_PPV_ARGS(backBuffers[i].GetAddressOf())));

        backBuffersRtvCpuDescriptorHandles[i] = {rtvDescriptorStartHandle.ptr + SIZE_T(i) * SIZE_T(rtvDescriptorHandleIncrementSize)};

        device->CreateRenderTargetView(backBuffers[i].Get(), nullptr, backBuffersRtvCpuDescriptorHandles[i]);
    }

    // create g-buffer
    D3D12_HEAP_PROPERTIES gBufferHeapProperties{};
    gBufferHeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC gBufferResourceDesc{};
    gBufferResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    gBufferResourceDesc.Width = windowWidth;
    gBufferResourceDesc.Height = windowHeight;
    gBufferResourceDesc.DepthOrArraySize = 1;
    gBufferResourceDesc.MipLevels = 1;
    gBufferResourceDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    gBufferResourceDesc.SampleDesc = {1, 0};
    gBufferResourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    D3D12_CLEAR_VALUE gBufferClearValue{};
    gBufferClearValue.Format = gBufferResourceDesc.Format;

    ThrowIfFailed(device->CreateCommittedResource(&gBufferHeapProperties, D3D12_HEAP_FLAG_NONE, &gBufferResourceDesc, D3D12_RESOURCE_STATE_RENDER_TARGET, &gBufferClearValue, IID_PPV_ARGS(gBuffer[0].GetAddressOf())));

    gBufferResourceDesc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
    gBufferClearValue.Format = gBufferResourceDesc.Format;

    ThrowIfFailed(device->CreateCommittedResource(&gBufferHeapProperties, D3D12_HEAP_FLAG_NONE, &gBufferResourceDesc, D3D12_RESOURCE_STATE_RENDER_TARGET, &gBufferClearValue, IID_PPV_ARGS(gBuffer[1].GetAddressOf())));

    // create g-buffer RTVs
    const D3D12_CPU_DESCRIPTOR_HANDLE freeRtvDescriptorHandle = {rtvDescriptorStartHandle.ptr + SIZE_T(numBackBuffers) * SIZE_T(rtvDescriptorHandleIncrementSize)};
    for (UINT i = 0; i < numGBufferComponents; ++i)
    {
        gBufferRtvCpuDescriptorHandles[i] = {freeRtvDescriptorHandle.ptr + SIZE_T(i) * SIZE_T(rtvDescriptorHandleIncrementSize)};

        device->CreateRenderTargetView(gBuffer[i].Get(), nullptr, gBufferRtvCpuDescriptorHandles[i]);
    }

    // create depth buffer
    D3D12_HEAP_PROPERTIES depthBufferHeapProperties{};
    depthBufferHeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC depthBufferResourceDesc{};
    depthBufferResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    depthBufferResourceDesc.Width = windowWidth;
    depthBufferResourceDesc.Height = windowHeight;
    depthBufferResourceDesc.DepthOrArraySize = 1;
    depthBufferResourceDesc.MipLevels = 1;
    depthBufferResourceDesc.Format = DXGI_FORMAT_D32_FLOAT;
    depthBufferResourceDesc.SampleDesc = {1, 0};
    depthBufferResourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    D3D12_CLEAR_VALUE depthBufferClearValue{};
    depthBufferClearValue.Format = depthBufferResourceDesc.Format;
    depthBufferClearValue.DepthStencil.Depth = 1.0f;

    ThrowIfFailed(device->CreateCommittedResource(&depthBufferHeapProperties, D3D12_HEAP_FLAG_NONE, &depthBufferResourceDesc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &depthBufferClearValue, IID_PPV_ARGS(depthBuffer.GetAddressOf())));

    // create depth buffer RTV
    depthBufferDsvCpuDescriptorHandle = dsvDescriptorHeap->GetCPUDescriptorHandleForHeapStart();

    device->CreateDepthStencilView(depthBuffer.Get(), nullptr, depthBufferDsvCpuDescriptorHandle);

    // create g-buffer SRVs
    const UINT srvDescriptorHandleIncrementSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const D3D12_CPU_DESCRIPTOR_HANDLE srvCpuDescriptorHeapStartHandle = srvDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
    const D3D12_GPU_DESCRIPTOR_HANDLE srvGpuDescriptorHeapStartHandle = srvDescriptorHeap->GetGPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < numGBufferComponents; ++i)
    {
        gBufferSrvCpuDescriptorHandles[i] = {srvCpuDescriptorHeapStartHandle.ptr + SIZE_T(i) * SIZE_T(srvDescriptorHandleIncrementSize)};
        gBufferSrvGpuDescriptorHandles[i] = {srvGpuDescriptorHeapStartHandle.ptr + SIZE_T(i) * SIZE_T(srvDescriptorHandleIncrementSize)};
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC gBufferSrvDesc{};
    gBufferSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    gBufferSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    gBufferSrvDesc.Texture2D.MipLevels = 1;
    gBufferSrvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;

    device->CreateShaderResourceView(gBuffer[0].Get(), &gBufferSrvDesc, gBufferSrvCpuDescriptorHandles[0]);

    gBufferSrvDesc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;

    device->CreateShaderResourceView(gBuffer[1].Get(), &gBufferSrvDesc, gBufferSrvCpuDescriptorHandles[1]);

    // set-up a single fence
    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())));
    frameIndex = swapChain->GetCurrentBackBufferIndex();
    fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
}

static void LoadAssets()
{
    // create per-frame geometry constant buffers (persistently mapped)
    D3D12_HEAP_PROPERTIES geometryConstantBuffersHeapProperties{};
    geometryConstantBuffersHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC geometryConstantBuffersResourceDesc{};
    geometryConstantBuffersResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    geometryConstantBuffersResourceDesc.Width = ((sizeof(GeometryConstantBuffer) + 255u) & ~255u) * 100;  // todo:
    geometryConstantBuffersResourceDesc.Height = 1;
    geometryConstantBuffersResourceDesc.DepthOrArraySize = 1;
    geometryConstantBuffersResourceDesc.MipLevels = 1;
    geometryConstantBuffersResourceDesc.SampleDesc = {1, 0};
    geometryConstantBuffersResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    for (UINT i = 0; i < numBackBuffers; ++i)
    {
        ThrowIfFailed(device->CreateCommittedResource(&geometryConstantBuffersHeapProperties, D3D12_HEAP_FLAG_NONE, &geometryConstantBuffersResourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(geometryConstantBuffers[i].GetAddressOf())));

        ThrowIfFailed(geometryConstantBuffers[i]->Map(0, nullptr, &(geometryConstantBuffersMappedMemory[i])));
    }

    // create per-frame lighting constant buffers (persistently mapped)
    D3D12_HEAP_PROPERTIES lightingConstantBuffersHeapProperties{};
    lightingConstantBuffersHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC lightingConstantBuffersResourceDesc{};
    lightingConstantBuffersResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    lightingConstantBuffersResourceDesc.Width = (sizeof(LightingConstantBuffer) + 255u) & ~255u;
    lightingConstantBuffersResourceDesc.Height = 1;
    lightingConstantBuffersResourceDesc.DepthOrArraySize = 1;
    lightingConstantBuffersResourceDesc.MipLevels = 1;
    lightingConstantBuffersResourceDesc.SampleDesc = {1, 0};
    lightingConstantBuffersResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    for (UINT i = 0; i < numBackBuffers; ++i)
    {
        ThrowIfFailed(device->CreateCommittedResource(&lightingConstantBuffersHeapProperties, D3D12_HEAP_FLAG_NONE, &lightingConstantBuffersResourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(lightingConstantBuffers[i].GetAddressOf())));

        ThrowIfFailed(lightingConstantBuffers[i]->Map(0, nullptr, &(lightingConstantBuffersMappedMemory[i])));
    }

    // create geometry root signature
    D3D12_ROOT_PARAMETER geometryCbvRootParamter{};
    geometryCbvRootParamter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    geometryCbvRootParamter.Descriptor.ShaderRegister = 0;
    geometryCbvRootParamter.Descriptor.RegisterSpace = 0;
    geometryCbvRootParamter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC geometryRootSignatureDesc{};
    geometryRootSignatureDesc.NumParameters = 1;
    geometryRootSignatureDesc.pParameters = &geometryCbvRootParamter;
    geometryRootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    Microsoft::WRL::ComPtr<ID3DBlob> errorBlob;
    Microsoft::WRL::ComPtr<ID3DBlob> geometryRootSignatureBlob;

    ThrowIfFailed(D3D12SerializeRootSignature(&geometryRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, geometryRootSignatureBlob.GetAddressOf(), errorBlob.GetAddressOf()));

    ThrowIfFailed(device->CreateRootSignature(0, geometryRootSignatureBlob->GetBufferPointer(), geometryRootSignatureBlob->GetBufferSize(), IID_PPV_ARGS(geometryRootSignature.GetAddressOf())));

    errorBlob.Reset();

    // create lighting root signature
    D3D12_DESCRIPTOR_RANGE gBufferSrvDescriptorRange{};
    gBufferSrvDescriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    gBufferSrvDescriptorRange.NumDescriptors = numGBufferComponents;
    gBufferSrvDescriptorRange.BaseShaderRegister = 0;

    D3D12_ROOT_PARAMETER lightingDescriptorTableRootParameter{};
    lightingDescriptorTableRootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    lightingDescriptorTableRootParameter.DescriptorTable.NumDescriptorRanges = 1;
    lightingDescriptorTableRootParameter.DescriptorTable.pDescriptorRanges = &gBufferSrvDescriptorRange;
    lightingDescriptorTableRootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_PARAMETER lightingCbvRootParameter{};
    lightingCbvRootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    lightingCbvRootParameter.Descriptor.ShaderRegister = 0;
    lightingCbvRootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_STATIC_SAMPLER_DESC lightingLinearStaticSampler{};
    lightingLinearStaticSampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    lightingLinearStaticSampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    lightingLinearStaticSampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    lightingLinearStaticSampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    lightingLinearStaticSampler.ShaderRegister = 0;
    lightingLinearStaticSampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    const D3D12_ROOT_PARAMETER lightingRootParameters[] = {lightingDescriptorTableRootParameter, lightingCbvRootParameter};

    D3D12_ROOT_SIGNATURE_DESC lightingRootSignatureDesc{};
    lightingRootSignatureDesc.NumParameters = 2;
    lightingRootSignatureDesc.pParameters = lightingRootParameters;
    lightingRootSignatureDesc.NumStaticSamplers = 1;
    lightingRootSignatureDesc.pStaticSamplers = &lightingLinearStaticSampler;

    Microsoft::WRL::ComPtr<ID3DBlob> lightingRootSignatureBlob;
    
    ThrowIfFailed(D3D12SerializeRootSignature(&lightingRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, lightingRootSignatureBlob.GetAddressOf(), errorBlob.GetAddressOf()));

    ThrowIfFailed(device->CreateRootSignature(0, lightingRootSignatureBlob->GetBufferPointer(), lightingRootSignatureBlob->GetBufferSize(), IID_PPV_ARGS(lightingRootSignature.GetAddressOf())));

    errorBlob.Reset();

    // create geometry shaders
    static Microsoft::WRL::ComPtr<ID3DBlob> geometryVertexShaderBlob;
    static Microsoft::WRL::ComPtr<ID3DBlob> geometryPixelShaderBlob;
    
    // compile geometry vertex shader
    if (FAILED(D3DCompileFromFile(L"../assets/shaders/geometry.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION, 0, geometryVertexShaderBlob.GetAddressOf(), errorBlob.GetAddressOf())))
    {
        if (errorBlob != nullptr)
        {
            std::cerr << (char*)(errorBlob->GetBufferPointer()) << std::endl;
        }
        throw std::runtime_error{"failed to compile vertex shader"};
    }
    errorBlob.Reset();
    
    // compile geometry pixel shader
    if (FAILED(D3DCompileFromFile(L"../assets/shaders/geometry.hlsl", nullptr, nullptr, "PSMain", "ps_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION, 0, geometryPixelShaderBlob.GetAddressOf(), errorBlob.GetAddressOf())))
    {
        if (errorBlob != nullptr)
        {
            std::cerr << (char*)(errorBlob->GetBufferPointer()) << std::endl;
        }
        throw std::runtime_error{"failed to compile pixel shader"};
    }
    errorBlob.Reset();

    // create lighting shaders
    static Microsoft::WRL::ComPtr<ID3DBlob> lightingVertexShaderBlob;
    static Microsoft::WRL::ComPtr<ID3DBlob> lightingPixelShaderBlob;
    
    // compile lighting vertex shader
    if (FAILED(D3DCompileFromFile(L"../assets/shaders/lighting.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION, 0, lightingVertexShaderBlob.GetAddressOf(), errorBlob.GetAddressOf())))
    {
        if (errorBlob != nullptr)
        {
            std::cerr << (char*)(errorBlob->GetBufferPointer()) << std::endl;
        }
        throw std::runtime_error{"failed to compile vertex shader"};
    }
    errorBlob.Reset();
    
    // compile lighting pixel shader
    if (FAILED(D3DCompileFromFile(L"../assets/shaders/lighting.hlsl", nullptr, nullptr, "PSMain", "ps_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION, 0, lightingPixelShaderBlob.GetAddressOf(), errorBlob.GetAddressOf())))
    {
        if (errorBlob != nullptr)
        {
            std::cerr << (char*)(errorBlob->GetBufferPointer()) << std::endl;
        }
        throw std::runtime_error{"failed to compile pixel shader"};
    }
    errorBlob.Reset();

    // create gometry pipeline state
    const D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC geometryPipelineStateDesc{};
    geometryPipelineStateDesc.pRootSignature = geometryRootSignature.Get();
    geometryPipelineStateDesc.VS = {geometryVertexShaderBlob->GetBufferPointer(), geometryVertexShaderBlob->GetBufferSize()};
    geometryPipelineStateDesc.PS = {geometryPixelShaderBlob->GetBufferPointer(), geometryPixelShaderBlob->GetBufferSize()};
    geometryPipelineStateDesc.InputLayout = {inputElementDescs, 2};
    geometryPipelineStateDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    geometryPipelineStateDesc.NumRenderTargets = numGBufferComponents;
    geometryPipelineStateDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    geometryPipelineStateDesc.RTVFormats[1] = DXGI_FORMAT_R10G10B10A2_UNORM;
    geometryPipelineStateDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;

    D3D12_RASTERIZER_DESC geometryRasterizerDesc{};
    geometryRasterizerDesc.FillMode = D3D12_FILL_MODE_SOLID;
    geometryRasterizerDesc.CullMode = D3D12_CULL_MODE_BACK;
    geometryRasterizerDesc.FrontCounterClockwise = FALSE;
    geometryRasterizerDesc.DepthBias = 0;
    geometryRasterizerDesc.DepthBiasClamp = 0.0f;
    geometryRasterizerDesc.MultisampleEnable = FALSE;
    geometryRasterizerDesc.AntialiasedLineEnable = FALSE;
    geometryRasterizerDesc.ForcedSampleCount = 0;
    geometryRasterizerDesc.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    geometryRasterizerDesc.DepthClipEnable = TRUE;

    geometryPipelineStateDesc.RasterizerState = geometryRasterizerDesc;

    D3D12_BLEND_DESC geometryBlendDesc{};
    geometryBlendDesc.AlphaToCoverageEnable = FALSE;
    geometryBlendDesc.IndependentBlendEnable = FALSE;
    for (int i = 0; i < 8; ++i)
    {
        auto& renderTarget = geometryBlendDesc.RenderTarget[i];
        renderTarget.BlendEnable = FALSE;
        renderTarget.LogicOpEnable = FALSE;
        renderTarget.SrcBlend = D3D12_BLEND_ONE;
        renderTarget.DestBlend = D3D12_BLEND_ZERO;
        renderTarget.BlendOp = D3D12_BLEND_OP_ADD;
        renderTarget.SrcBlendAlpha = D3D12_BLEND_ONE;
        renderTarget.DestBlendAlpha = D3D12_BLEND_ZERO;
        renderTarget.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        renderTarget.LogicOp = D3D12_LOGIC_OP_NOOP;
        renderTarget.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    }

    geometryPipelineStateDesc.BlendState = geometryBlendDesc;
    geometryPipelineStateDesc.SampleMask = UINT_MAX;
    
    D3D12_DEPTH_STENCIL_DESC geometryDepthStencilDesc{};
    geometryDepthStencilDesc.DepthEnable = TRUE;
    geometryDepthStencilDesc.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    geometryDepthStencilDesc.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    geometryDepthStencilDesc.StencilEnable = FALSE;

    geometryPipelineStateDesc.DepthStencilState = geometryDepthStencilDesc;
    geometryPipelineStateDesc.SampleDesc = {1, 0};

    ThrowIfFailed(device->CreateGraphicsPipelineState(&geometryPipelineStateDesc, IID_PPV_ARGS(geometryPipelineState.GetAddressOf())));

    // create lighting pipeline state
    D3D12_GRAPHICS_PIPELINE_STATE_DESC lightingPipelineStateDesc{};
    lightingPipelineStateDesc.pRootSignature = lightingRootSignature.Get();
    lightingPipelineStateDesc.VS = {lightingVertexShaderBlob->GetBufferPointer(), lightingVertexShaderBlob->GetBufferSize()};
    lightingPipelineStateDesc.PS = {lightingPixelShaderBlob->GetBufferPointer(), lightingPixelShaderBlob->GetBufferSize()};
    lightingPipelineStateDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    lightingPipelineStateDesc.NumRenderTargets  = 1;
    lightingPipelineStateDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;  // swapChain format

    D3D12_RASTERIZER_DESC lightingRasterizerDesc{};
    lightingRasterizerDesc.FillMode = D3D12_FILL_MODE_SOLID;
    lightingRasterizerDesc.CullMode = D3D12_CULL_MODE_NONE;
    lightingRasterizerDesc.FrontCounterClockwise = FALSE;
    lightingRasterizerDesc.DepthBias = 0;
    lightingRasterizerDesc.DepthBiasClamp = 0.0f;
    lightingRasterizerDesc.MultisampleEnable = FALSE;
    lightingRasterizerDesc.AntialiasedLineEnable = FALSE;
    lightingRasterizerDesc.ForcedSampleCount = 0;
    lightingRasterizerDesc.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    lightingRasterizerDesc.DepthClipEnable = TRUE;

    lightingPipelineStateDesc.RasterizerState = lightingRasterizerDesc;

    D3D12_BLEND_DESC lightingBlendDesc{};
    lightingBlendDesc.AlphaToCoverageEnable = FALSE;
    lightingBlendDesc.IndependentBlendEnable = FALSE;
    for (int i = 0; i < 8; ++i)
    {
        auto& renderTarget = lightingBlendDesc.RenderTarget[i];
        renderTarget.BlendEnable = FALSE;
        renderTarget.LogicOpEnable = FALSE;
        renderTarget.SrcBlend = D3D12_BLEND_ONE;
        renderTarget.DestBlend = D3D12_BLEND_ZERO;
        renderTarget.BlendOp = D3D12_BLEND_OP_ADD;
        renderTarget.SrcBlendAlpha = D3D12_BLEND_ONE;
        renderTarget.DestBlendAlpha = D3D12_BLEND_ZERO;
        renderTarget.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        renderTarget.LogicOp = D3D12_LOGIC_OP_NOOP;
        renderTarget.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    }

    lightingPipelineStateDesc.BlendState = lightingBlendDesc;
    lightingPipelineStateDesc.SampleMask = UINT_MAX;

    D3D12_DEPTH_STENCIL_DESC lightingDepthStencilDesc{};
    lightingDepthStencilDesc.DepthEnable = FALSE;

    lightingPipelineStateDesc.DepthStencilState = lightingDepthStencilDesc;
    lightingPipelineStateDesc.SampleDesc = {1, 0};


    ThrowIfFailed(device->CreateGraphicsPipelineState(&lightingPipelineStateDesc, IID_PPV_ARGS(lightingPipelineState.GetAddressOf())));

    LoadObjFile("../assets/models/cube.obj");
    LoadObjFile("../assets/models/pyramid.obj");
    LoadObjFile("../assets/models/cube.obj");

    UINT numVertices = 0;
    UINT numIndices = 0;
    for (const Mesh& mesh : meshes)
    {
        numVertices += static_cast<UINT>((mesh.vertices).size());
        numIndices += static_cast<UINT>((mesh.indices).size());
    }

    // create unified vertex buffer
    D3D12_HEAP_PROPERTIES unifiedVertexBufferHeapProperties{};
    unifiedVertexBufferHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC unifiedVertexBufferResourceDesc{};
    unifiedVertexBufferResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    unifiedVertexBufferResourceDesc.Width = numVertices * sizeof(Vertex);
    unifiedVertexBufferResourceDesc.Height = 1;
    unifiedVertexBufferResourceDesc.DepthOrArraySize = 1;
    unifiedVertexBufferResourceDesc.MipLevels = 1;
    unifiedVertexBufferResourceDesc.SampleDesc = {1, 0};
    unifiedVertexBufferResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ThrowIfFailed(device->CreateCommittedResource(&unifiedVertexBufferHeapProperties, D3D12_HEAP_FLAG_NONE, &unifiedVertexBufferResourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(unifiedVertexBuffer.GetAddressOf())));

    // create unified index buffer
    D3D12_HEAP_PROPERTIES unifiedIndexBufferHeapProperties{};
    unifiedIndexBufferHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC unifiedIndexBufferResourceDesc{};
    unifiedIndexBufferResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    unifiedIndexBufferResourceDesc.Width = numIndices * sizeof(UINT);
    unifiedIndexBufferResourceDesc.Height = 1;
    unifiedIndexBufferResourceDesc.DepthOrArraySize = 1;
    unifiedIndexBufferResourceDesc.MipLevels = 1;
    unifiedIndexBufferResourceDesc.SampleDesc = {1, 0};
    unifiedIndexBufferResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ThrowIfFailed(device->CreateCommittedResource(&unifiedIndexBufferHeapProperties, D3D12_HEAP_FLAG_NONE, &unifiedIndexBufferResourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(unifiedIndexBuffer.GetAddressOf())));

    void* unifiedVertexBufferMappedMemory;
    ThrowIfFailed(unifiedVertexBuffer->Map(0, nullptr, &unifiedVertexBufferMappedMemory));

    void* unifiedIndexBufferMappedMemory;
    ThrowIfFailed(unifiedIndexBuffer->Map(0, nullptr, &unifiedIndexBufferMappedMemory));

    float i = 0;
    UINT vertexBufferOffset = 0;
    UINT indexBufferOffset = 0;
    UINT constantBufferOffset = 0;
    for (const Mesh& mesh : meshes)
    {
        std::memcpy(static_cast<char*>(unifiedVertexBufferMappedMemory) + vertexBufferOffset * sizeof(Vertex), (mesh.vertices).data(), (mesh.vertices).size() * sizeof(Vertex));

        std::memcpy(static_cast<char*>(unifiedIndexBufferMappedMemory) + indexBufferOffset * sizeof(UINT), (mesh.indices).data(), (mesh.indices).size() * sizeof(UINT));

        RenderableMeshDesc renderableMeshDesc{};
        renderableMeshDesc.vertexBufferOffset = vertexBufferOffset;
        renderableMeshDesc.numVertices = static_cast<UINT>((mesh.vertices).size());
        renderableMeshDesc.indexBufferOffset = indexBufferOffset;
        renderableMeshDesc.numIndices = static_cast<UINT>((mesh.indices).size());
        renderableMeshDesc.constantBufferOffset = constantBufferOffset;
        (renderableMeshDesc.transform).position = DirectX::XMFLOAT3{-2.0f + i * 2.0f, 0.0f, 0.0f};
        (renderableMeshDesc.transform).orientation = DirectX::XMFLOAT3{0.0f, 0.0f, 0.0f};
        (renderableMeshDesc.transform).scale = DirectX::XMFLOAT3{1.0f, 1.0f, 1.0f};

        renderableMeshDescs.push_back(renderableMeshDesc);

        vertexBufferOffset += static_cast<UINT>((mesh.vertices).size());
        indexBufferOffset += static_cast<UINT>((mesh.indices).size());
        constantBufferOffset += (sizeof(GeometryConstantBuffer) + 255u) & ~255u;

        ++i;
    }

    unifiedVertexBuffer->Unmap(0, nullptr);
    
    unifiedIndexBuffer->Unmap(0, nullptr);

    // initialize unified vertex buffer view
    unifiedVertexBufferView.BufferLocation = unifiedVertexBuffer->GetGPUVirtualAddress();
    unifiedVertexBufferView.StrideInBytes = sizeof(Vertex);
    unifiedVertexBufferView.SizeInBytes = numVertices * sizeof(Vertex);

    // initialize unified index buffer view
    unifiedIndexBufferView.BufferLocation = unifiedIndexBuffer->GetGPUVirtualAddress();
    unifiedIndexBufferView.SizeInBytes = numIndices * sizeof(UINT);
    unifiedIndexBufferView.Format = DXGI_FORMAT_R32_UINT;
}

static void LoadObjFile(const std::string& filename)
{
    tinyobj::ObjReaderConfig objReaderConfig{};
    objReaderConfig.triangulate = true;
    tinyobj::ObjReader objReader{};
    if (!objReader.ParseFromFile(filename, objReaderConfig))
    {
        if (!(objReader.Error()).empty())
        {
            std::cerr << objReader.Error();
        }
        throw std::runtime_error{"failed to load OBJ file"};
    }

    if (!(objReader.Warning()).empty())
    {
        std::cerr << objReader.Warning();
    }

    tinyobj::attrib_t meshAttributes = objReader.GetAttrib();
    std::vector<tinyobj::shape_t> meshShapes = objReader.GetShapes();

    std::map<std::pair<std::size_t, std::size_t>, std::size_t> objVertexToCustomVertexIndex;

    Mesh mesh;

    for (const tinyobj::shape_t& shape : meshShapes)
    {
        std::size_t offset = 0;
        for (std::size_t faceIndex = 0; faceIndex < (shape.mesh.num_face_vertices).size(); ++faceIndex)
        {
            for (std::size_t vertexIndex = 0; vertexIndex < static_cast<std::size_t>((shape.mesh.num_face_vertices)[faceIndex]); ++vertexIndex)
            {
                tinyobj::index_t index = (shape.mesh).indices[offset + vertexIndex];

                std::pair<std::size_t, std::size_t> indexPair{};
                indexPair.first = index.vertex_index;
                indexPair.second = index.normal_index;
                const auto& itr = objVertexToCustomVertexIndex.find(indexPair);
                if (itr != objVertexToCustomVertexIndex.end())
                {
                    (mesh.indices).push_back(static_cast<UINT>(itr->second));
                    continue;
                }

                Vertex vertex{};

                tinyobj::real_t x = meshAttributes.vertices[3 * static_cast<std::size_t>(index.vertex_index) + 0];
                tinyobj::real_t y = meshAttributes.vertices[3 * static_cast<std::size_t>(index.vertex_index) + 1];
                tinyobj::real_t z = meshAttributes.vertices[3 * static_cast<std::size_t>(index.vertex_index) + 2];

                vertex.position = DirectX::XMFLOAT3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};

                if (index.normal_index >= 0)
                {
                    tinyobj::real_t nx = meshAttributes.normals[3 * static_cast<std::size_t>(index.normal_index) + 0];
                    tinyobj::real_t ny = meshAttributes.normals[3 * static_cast<std::size_t>(index.normal_index) + 1];
                    tinyobj::real_t nz = meshAttributes.normals[3 * static_cast<std::size_t>(index.normal_index) + 2];

                    vertex.normal = DirectX::XMFLOAT3{static_cast<float>(nx), static_cast<float>(ny), static_cast<float>(nz)};
                }
                if (index.texcoord_index >= 0)
                {
                    tinyobj::real_t u = meshAttributes.texcoords[2 * static_cast<std::size_t>(index.texcoord_index) + 0];
                    tinyobj::real_t v = meshAttributes.texcoords[2 * static_cast<std::size_t>(index.texcoord_index) + 1];

                    // todo: 
                }

                objVertexToCustomVertexIndex[indexPair] = (mesh.vertices).size();

                (mesh.indices).push_back(static_cast<UINT>((mesh.vertices).size()));
                (mesh.vertices).push_back(vertex);
            }
            offset += static_cast<std::size_t>((shape.mesh.num_face_vertices)[faceIndex]);
        }
    }
    meshes.push_back(mesh);
}
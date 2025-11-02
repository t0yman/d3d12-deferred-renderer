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

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <DirectXMath.h>

#include <wrl/client.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

using namespace DirectX;

struct Vertex
{
    XMFLOAT3 position;
    XMFLOAT3 normal;
};

struct Mesh
{
    std::vector<Vertex> vertices;
    std::vector<std::uint16_t> indices;
};

struct MvpInfo
{
    XMFLOAT4X4 model;
    XMFLOAT4X4 view;
    XMFLOAT4X4 projection;
};

struct LightInfo
{
    XMFLOAT3 lightDir;
    float pad0;
    XMFLOAT3 lightColor;
    float pad1;
};

static constexpr int windowWidth = 1280;
static constexpr int windowHeight = 720;
static GLFWwindow* window = nullptr;

static constexpr UINT numBackBuffers = 3;
static constexpr UINT numGBufferComponents = 2;
static constexpr UINT numRtvDescriptors = 255;
static constexpr UINT numSrvDescriptors = 255;
static constexpr UINT numDsvDescriptors = 1;

static constexpr D3D12_VIEWPORT viewport{0.0f, 0.0f, static_cast<float>(windowWidth), static_cast<float>(windowHeight), 0.0f, 1.0f};
static constexpr RECT scissorRect{0, 0, windowWidth, windowHeight};
static Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
static Microsoft::WRL::ComPtr<ID3D12Device> device;
static Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue;
static Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocators[numBackBuffers] = {};
static Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandLists[numBackBuffers] = {};
static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtvDescriptorHeap;
static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> dsvDescriptorHeap;
static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srvDescriptorHeap;
static Microsoft::WRL::ComPtr<IDXGISwapChain3> swapChain;
static Microsoft::WRL::ComPtr<ID3D12Resource> swapChainBuffers[numBackBuffers];
static D3D12_CPU_DESCRIPTOR_HANDLE swapChainRtvCpuDescriptorHandles[numBackBuffers] = {};
static Microsoft::WRL::ComPtr<ID3D12Resource> gBuffer[numGBufferComponents];
static D3D12_CPU_DESCRIPTOR_HANDLE gBufferRtvCpuDescriptorHandles[numGBufferComponents] = {};
static D3D12_CPU_DESCRIPTOR_HANDLE gBufferSrvCpuDescriptorHandles[numGBufferComponents] = {};
static D3D12_GPU_DESCRIPTOR_HANDLE gBufferSrvGpuDescriptorHandles[numGBufferComponents] = {};
static Microsoft::WRL::ComPtr<ID3D12Resource> depthBuffer;
static D3D12_CPU_DESCRIPTOR_HANDLE depthBufferDsvCpuDescriptorHandle{};
static Microsoft::WRL::ComPtr<ID3DBlob> geometryVertexShaderBlob;
static Microsoft::WRL::ComPtr<ID3DBlob> geometryPixelShaderBlob;
static Microsoft::WRL::ComPtr<ID3DBlob> lightingVertexShaderBlob;
static Microsoft::WRL::ComPtr<ID3DBlob> lightingPixelShaderBlob;
static Microsoft::WRL::ComPtr<ID3D12RootSignature> geometryRootSignature;
static Microsoft::WRL::ComPtr<ID3D12RootSignature> lightingRootSignature;
static Microsoft::WRL::ComPtr<ID3D12PipelineState> geometryPipelineState;
static Microsoft::WRL::ComPtr<ID3D12PipelineState> lightingPipelineState;

static Microsoft::WRL::ComPtr<ID3D12Fence> fence;
static UINT frameIndex = 1;
static HANDLE fenceEvent;
static UINT64 fenceValues[numBackBuffers] = {};

static Mesh mesh{};
static Microsoft::WRL::ComPtr<ID3D12Resource> vertexBuffer;
static Microsoft::WRL::ComPtr<ID3D12Resource> indexBuffer;
static D3D12_VERTEX_BUFFER_VIEW vertexBufferView{};
static D3D12_INDEX_BUFFER_VIEW indexBufferView{};

static Microsoft::WRL::ComPtr<ID3D12Resource> mvpInfoBuffers[numBackBuffers];
static void* mvpInfoBuffersMappedMemory[numBackBuffers] = {};
static Microsoft::WRL::ComPtr<ID3D12Resource> lightInfoBuffers[numBackBuffers];
static void* lightInfoBuffersMappedMemory[numBackBuffers] = {};

inline static void ThrowIfFailed(HRESULT hResult);

static void LoadPipeline(HWND hWnd);
static void LoadAssets();
static void LoadObjFile(const std::string& fllename);

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
            glfwTerminate();

            throw std::runtime_error{"failed to create window"};
        }

        HWND hWnd = glfwGetWin32Window(window);
        LoadPipeline(hWnd);
        LoadAssets();

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();

            const UINT currentFrameIndex = swapChain->GetCurrentBackBufferIndex();

            if (fence->GetCompletedValue() < fenceValues[currentFrameIndex])  // think fence->GetCompletedValue() == lastFrameNumberSuccessfulyRenderer
            {
                // being here means the last frame the GPU has finished rendering was not the current frame
                // i.e. the current frame is still in the process of being renderered
                ThrowIfFailed(fence->SetEventOnCompletion(fenceValues[currentFrameIndex], fenceEvent));
                WaitForSingleObject(fenceEvent, INFINITE);
            }

            ThrowIfFailed(commandAllocators[currentFrameIndex]->Reset());
            ThrowIfFailed(commandLists[currentFrameIndex]->Reset(commandAllocators[currentFrameIndex].Get(), nullptr));

            ID3D12Resource* backBuffer = swapChainBuffers[currentFrameIndex].Get();

            static float rotation = 0.0f;
            rotation += 0.01f;

            // calculate mvp
            XMMATRIX model = XMMatrixRotationY(rotation) * XMMatrixTranslation(0.0f, 0.0f, 4.0f);
            XMMATRIX view = XMMatrixLookAtLH(XMVectorSet(0.0f, 1.0f, -3.0f, 1.0f), XMVectorSet(0.0f, 0.0f, 4.0f, 1.0f), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f));
            XMMATRIX projection = XMMatrixPerspectiveFovLH(XM_PIDIV4, static_cast<float>(windowWidth) / static_cast<float>(windowHeight), 0.1f, 100.0f);

            model = XMMatrixTranspose(model);
            view = XMMatrixTranspose(view);
            projection = XMMatrixTranspose(projection);

            // upload mvpInfo to geometry shader
            MvpInfo mvpInfo;
            XMStoreFloat4x4(&(mvpInfo.model), model);
            XMStoreFloat4x4(&(mvpInfo.view), view);
            XMStoreFloat4x4(&(mvpInfo.projection), projection);

            std::memcpy(mvpInfoBuffersMappedMemory[currentFrameIndex], &mvpInfo, sizeof(mvpInfo));

            // upload lightInfo to lighting shader
            LightInfo lightInfo;
            lightInfo.lightDir = XMFLOAT3{-0.3f, -0.2f, -1.0f};
            lightInfo.pad0 = 0.0f;
            lightInfo.lightColor = XMFLOAT3{1.0f, 1.0f, 1.0f};
            lightInfo.pad1 = 0.0f;

            std::memcpy(lightInfoBuffersMappedMemory[currentFrameIndex], &lightInfo, sizeof(lightInfo));

            // set-up geometry pipeline to draw to G-Buffer
            commandLists[currentFrameIndex]->RSSetViewports(1, &viewport);
            commandLists[currentFrameIndex]->RSSetScissorRects(1, &scissorRect);
            commandLists[currentFrameIndex]->OMSetRenderTargets(2, gBufferRtvCpuDescriptorHandles, FALSE, &depthBufferDsvCpuDescriptorHandle);
            commandLists[currentFrameIndex]->IASetVertexBuffers(0, 1, &vertexBufferView);
            commandLists[currentFrameIndex]->IASetIndexBuffer(&indexBufferView);

            const float gBuffer0ClearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const float gBuffer1ClearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            commandLists[currentFrameIndex]->ClearRenderTargetView(gBufferRtvCpuDescriptorHandles[0], gBuffer0ClearColor, 0, nullptr);
            commandLists[currentFrameIndex]->ClearRenderTargetView(gBufferRtvCpuDescriptorHandles[1], gBuffer1ClearColor, 0, nullptr);
            commandLists[currentFrameIndex]->ClearDepthStencilView(depthBufferDsvCpuDescriptorHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
            commandLists[currentFrameIndex]->SetPipelineState(geometryPipelineState.Get());
            commandLists[currentFrameIndex]->SetGraphicsRootSignature(geometryRootSignature.Get());
            commandLists[currentFrameIndex]->SetGraphicsRootConstantBufferView(0, mvpInfoBuffers[currentFrameIndex]->GetGPUVirtualAddress());
            commandLists[currentFrameIndex]->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            commandLists[currentFrameIndex]->DrawIndexedInstanced((mesh.indices).size(), 1, 0, 0, 0);


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

            commandLists[currentFrameIndex]->ResourceBarrier(4, geometryExitTransitionBarriers);

            // set-up lighting pipeline to draw final image
            const float clearColor[] = {0.39f, 0.58f, 0.93f, 1.0f};

            commandLists[currentFrameIndex]->ClearRenderTargetView(swapChainRtvCpuDescriptorHandles[currentFrameIndex], clearColor, 0, nullptr);
            commandLists[currentFrameIndex]->OMSetRenderTargets(1, &(swapChainRtvCpuDescriptorHandles[currentFrameIndex]), FALSE, nullptr);
            
            ID3D12DescriptorHeap* descriptorHeaps[] = {srvDescriptorHeap.Get()};

            commandLists[currentFrameIndex]->SetDescriptorHeaps(1, descriptorHeaps);
            commandLists[currentFrameIndex]->SetPipelineState(lightingPipelineState.Get());
            commandLists[currentFrameIndex]->SetGraphicsRootSignature(lightingRootSignature.Get());
            commandLists[currentFrameIndex]->SetGraphicsRootDescriptorTable(0, gBufferSrvGpuDescriptorHandles[0]);
            commandLists[currentFrameIndex]->SetGraphicsRootConstantBufferView(1, lightInfoBuffers[currentFrameIndex]->GetGPUVirtualAddress());
            commandLists[currentFrameIndex]->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            commandLists[currentFrameIndex]->DrawInstanced(3, 1, 0, 0);

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

            commandLists[currentFrameIndex]->ResourceBarrier(4, lightingExitTranitionBarriers);

            ThrowIfFailed(commandLists[currentFrameIndex]->Close());

            ID3D12CommandList* commandList[] = {commandLists[currentFrameIndex].Get()};

            commandQueue->ExecuteCommandLists(1, commandList);

            ThrowIfFailed(swapChain->Present(1, 0));

            ++frameIndex;
            ThrowIfFailed(commandQueue->Signal(fence.Get(), frameIndex));
            fenceValues[currentFrameIndex] = frameIndex;
        }

        CloseHandle(fenceEvent);

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

static void LoadPipeline(HWND hWnd)
{
#if defined(_DEBUG)
{
    Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
    ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(debugController.GetAddressOf())));
    Microsoft::WRL::ComPtr<ID3D12Debug1> debugController1;
    ThrowIfFailed(debugController.As(&debugController1));
    debugController1->SetEnableGPUBasedValidation(TRUE);
    debugController->EnableDebugLayer();
}
#endif

    ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf())));

    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    // todo:

    // create device
    ThrowIfFailed(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.GetAddressOf())));

    // create command queue
    D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
    commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    ThrowIfFailed(device->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(commandQueue.GetAddressOf())));

    // create command allocators and command lists
    for (UINT i = 0; i < numBackBuffers; ++i)
    {
        ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(commandAllocators[i].GetAddressOf())));

        ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocators[i].Get(), nullptr, IID_PPV_ARGS(commandLists[i].GetAddressOf())));
        ThrowIfFailed(commandLists[i]->Close());
    }

    // create rtv descriptor heap
    D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc{};
    rtvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDescriptorHeapDesc.NumDescriptors = numRtvDescriptors;

    ThrowIfFailed(device->CreateDescriptorHeap(&rtvDescriptorHeapDesc, IID_PPV_ARGS(rtvDescriptorHeap.GetAddressOf())));

    // create dsv descriptor heap
    D3D12_DESCRIPTOR_HEAP_DESC dsvDescriptorHeapDesc{};
    dsvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvDescriptorHeapDesc.NumDescriptors = numDsvDescriptors;

    ThrowIfFailed(device->CreateDescriptorHeap(&dsvDescriptorHeapDesc, IID_PPV_ARGS(dsvDescriptorHeap.GetAddressOf())));

    // create srv descriptor heap
    D3D12_DESCRIPTOR_HEAP_DESC srvDescriptorHeapDesc{};
    srvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvDescriptorHeapDesc.NumDescriptors = numSrvDescriptors;
    srvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    ThrowIfFailed(device->CreateDescriptorHeap(&srvDescriptorHeapDesc, IID_PPV_ARGS(srvDescriptorHeap.GetAddressOf())));

    // create swap chain
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc{};
    swapChainDesc.BufferCount = numBackBuffers;
    swapChainDesc.Width = windowWidth;
    swapChainDesc.Height = windowHeight;
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.SampleDesc = {1, 0};

    Microsoft::WRL::ComPtr<IDXGISwapChain1> swapChain1;
    ThrowIfFailed(factory->CreateSwapChainForHwnd(commandQueue.Get(), hWnd, &swapChainDesc, nullptr, nullptr, swapChain1.GetAddressOf()));

    ThrowIfFailed(swapChain1.As(&swapChain));

    // create swap chain back buffer rtvs
    const UINT rtvDescriptorHandleIncrementSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    const D3D12_CPU_DESCRIPTOR_HANDLE rtvDescriptorHeapStartHandle = rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < numBackBuffers; ++i)
    {
        ThrowIfFailed(swapChain->GetBuffer(i, IID_PPV_ARGS(swapChainBuffers[i].GetAddressOf())));

        swapChainRtvCpuDescriptorHandles[i] = {rtvDescriptorHeapStartHandle.ptr + SIZE_T(i) * SIZE_T(rtvDescriptorHandleIncrementSize)};

        device->CreateRenderTargetView(swapChainBuffers[i].Get(), nullptr, swapChainRtvCpuDescriptorHandles[i]);
    }

    // create G-Buffer
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

    // create G-Buffer rtvs
    const D3D12_CPU_DESCRIPTOR_HANDLE nextFreeRtvDescriptorHandle = {rtvDescriptorHeapStartHandle.ptr + SIZE_T(numBackBuffers) * SIZE_T(rtvDescriptorHandleIncrementSize)};
    for (UINT i = 0; i < numGBufferComponents; ++i)
    {
        gBufferRtvCpuDescriptorHandles[i] = {nextFreeRtvDescriptorHandle.ptr + SIZE_T(i) * SIZE_T(rtvDescriptorHandleIncrementSize)};

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

    // create depth buffer rtv
    depthBufferDsvCpuDescriptorHandle = dsvDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
    
    device->CreateDepthStencilView(depthBuffer.Get(), nullptr, depthBufferDsvCpuDescriptorHandle);

    // create G-Buffer srvs
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

    // create geometry shader root signature
    D3D12_ROOT_PARAMETER geometryCbvRootParameter{};
    geometryCbvRootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    geometryCbvRootParameter.Descriptor.ShaderRegister = 0;
    geometryCbvRootParameter.Descriptor.RegisterSpace = 0;
    geometryCbvRootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC geometryRootSignatureDesc{};
    geometryRootSignatureDesc.NumParameters = 1;
    geometryRootSignatureDesc.pParameters = &geometryCbvRootParameter;
    geometryRootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    Microsoft::WRL::ComPtr<ID3DBlob> errorBlob;
    Microsoft::WRL::ComPtr<ID3DBlob> geometryRootSignatureBlob;

    ThrowIfFailed(D3D12SerializeRootSignature(&geometryRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, geometryRootSignatureBlob.GetAddressOf(), errorBlob.GetAddressOf()));

    ThrowIfFailed(device->CreateRootSignature(0, geometryRootSignatureBlob->GetBufferPointer(), geometryRootSignatureBlob->GetBufferSize(), IID_PPV_ARGS(geometryRootSignature.GetAddressOf())));

    errorBlob.Reset();

    // create lighting shader root signature
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

    // create geometry pipeline state
    D3D12_INPUT_ELEMENT_DESC geometryInputElementDescs[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC geometryPipelineStateDesc{};
    geometryPipelineStateDesc.pRootSignature = geometryRootSignature.Get();
    geometryPipelineStateDesc.VS = {geometryVertexShaderBlob->GetBufferPointer(), geometryVertexShaderBlob->GetBufferSize()};
    geometryPipelineStateDesc.PS = {geometryPixelShaderBlob->GetBufferPointer(), geometryPixelShaderBlob->GetBufferSize()};
    geometryPipelineStateDesc.InputLayout = {geometryInputElementDescs, 2};
    geometryPipelineStateDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    geometryPipelineStateDesc.NumRenderTargets = 2;
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
    lightingPipelineStateDesc.NumRenderTargets = 1;
    lightingPipelineStateDesc.RTVFormats[0] = swapChainDesc.Format;

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

    // set-up a single fence
    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())));
    frameIndex = 1;
    fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // set-up per frame mvpInfo constant buffers (persistently mapped)
    D3D12_HEAP_PROPERTIES mvpInfoHeapProperties{};
    mvpInfoHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC mvpInfoResourceDesc{};
    mvpInfoResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    mvpInfoResourceDesc.Width = (sizeof(MvpInfo) + 255u) & ~255u;
    mvpInfoResourceDesc.Height = 1;
    mvpInfoResourceDesc.DepthOrArraySize = 1;
    mvpInfoResourceDesc.MipLevels = 1;
    mvpInfoResourceDesc.SampleDesc = {1, 0};
    mvpInfoResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    for (UINT i = 0; i < numBackBuffers; ++i)
    {
        ThrowIfFailed(device->CreateCommittedResource(&mvpInfoHeapProperties, D3D12_HEAP_FLAG_NONE, &mvpInfoResourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(mvpInfoBuffers[i].GetAddressOf())));

        ThrowIfFailed(mvpInfoBuffers[i]->Map(0, nullptr, &(mvpInfoBuffersMappedMemory[i])));
    }

    // set-up per frame lightInfo constant buffers (also persistently mapped)
    D3D12_HEAP_PROPERTIES lightInfoHeapProperties{};
    lightInfoHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC lightInfoResourceDesc{};
    lightInfoResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    lightInfoResourceDesc.Width = (sizeof(LightInfo) + 255u) & ~255u;
    lightInfoResourceDesc.Height = 1;
    lightInfoResourceDesc.DepthOrArraySize = 1;
    lightInfoResourceDesc.MipLevels = 1;
    lightInfoResourceDesc.SampleDesc = {1, 0};
    lightInfoResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    for (UINT i = 0; i < numBackBuffers; ++i)
    {
        ThrowIfFailed(device->CreateCommittedResource(&lightInfoHeapProperties, D3D12_HEAP_FLAG_NONE, &lightInfoResourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(lightInfoBuffers[i].GetAddressOf())));

        ThrowIfFailed(lightInfoBuffers[i]->Map(0, nullptr, &(lightInfoBuffersMappedMemory[i])));
    }
}

static void LoadAssets()
{
    // load OBJ mesh
    LoadObjFile("../assets/models/pyramid.obj");

    // create vertex buffer
    D3D12_HEAP_PROPERTIES vertexBufferHeapProperties{};
    vertexBufferHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC vertexBufferResourceDesc{};
    vertexBufferResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    vertexBufferResourceDesc.Width = (mesh.vertices).size() * sizeof(Vertex);
    vertexBufferResourceDesc.Height = 1;
    vertexBufferResourceDesc.DepthOrArraySize = 1;
    vertexBufferResourceDesc.MipLevels = 1;
    vertexBufferResourceDesc.SampleDesc = {1, 0};
    vertexBufferResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ThrowIfFailed(device->CreateCommittedResource(&vertexBufferHeapProperties, D3D12_HEAP_FLAG_NONE, &vertexBufferResourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(vertexBuffer.GetAddressOf())));
    
    void* vertexBufferMappedMemory;
    ThrowIfFailed(vertexBuffer->Map(0, nullptr, &vertexBufferMappedMemory));
    std::memcpy(vertexBufferMappedMemory, (mesh.vertices).data(), (mesh.vertices).size() * sizeof(Vertex));
    
    vertexBuffer->Unmap(0, nullptr);

    vertexBufferView.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
    vertexBufferView.StrideInBytes = sizeof(Vertex);
    vertexBufferView.SizeInBytes = (mesh.vertices).size() * sizeof(Vertex);

    // create index buffer
    D3D12_HEAP_PROPERTIES indexBufferHeapProperties{};
    indexBufferHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC indexBufferResourceDesc{};
    indexBufferResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    indexBufferResourceDesc.Width = (mesh.indices).size() * sizeof(std::uint16_t);
    indexBufferResourceDesc.Height = 1;
    indexBufferResourceDesc.DepthOrArraySize = 1;
    indexBufferResourceDesc.MipLevels = 1;
    indexBufferResourceDesc.SampleDesc = {1, 0};
    indexBufferResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ThrowIfFailed(device->CreateCommittedResource(&indexBufferHeapProperties, D3D12_HEAP_FLAG_NONE, &indexBufferResourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(indexBuffer.GetAddressOf())));
    
    void* indexBufferMappedMemory;
    ThrowIfFailed(indexBuffer->Map(0, nullptr, &indexBufferMappedMemory));
    std::memcpy(indexBufferMappedMemory, (mesh.indices).data(), (mesh.indices).size() * sizeof(std::uint16_t));

    indexBuffer->Unmap(0, nullptr);

    indexBufferView.BufferLocation = indexBuffer->GetGPUVirtualAddress();
    indexBufferView.SizeInBytes = (mesh.indices).size() * sizeof(std::uint16_t);
    indexBufferView.Format = DXGI_FORMAT_R16_UINT;
}

static void LoadObjFile(const std::string& filename)
{
    tinyobj::ObjReaderConfig objReaderConfig{};
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
                    (mesh.indices).push_back(static_cast<std::uint16_t>(itr->second));
                    continue;
                }

                Vertex vertex{};

                tinyobj::real_t x = meshAttributes.vertices[3 * static_cast<std::size_t>(index.vertex_index) + 0];
                tinyobj::real_t y = meshAttributes.vertices[3 * static_cast<std::size_t>(index.vertex_index) + 1];
                tinyobj::real_t z = meshAttributes.vertices[3 * static_cast<std::size_t>(index.vertex_index) + 2];

                vertex.position = XMFLOAT3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};

                if (index.normal_index >= 0)
                {
                    tinyobj::real_t nx = meshAttributes.normals[3 * static_cast<std::size_t>(index.normal_index) + 0];
                    tinyobj::real_t ny = meshAttributes.normals[3 * static_cast<std::size_t>(index.normal_index) + 1];
                    tinyobj::real_t nz = meshAttributes.normals[3 * static_cast<std::size_t>(index.normal_index) + 2];

                    vertex.normal = XMFLOAT3{static_cast<float>(nx), static_cast<float>(ny), static_cast<float>(nz)};
                }
                if (index.texcoord_index >= 0)
                {
                    tinyobj::real_t u = meshAttributes.texcoords[2 * static_cast<std::size_t>(index.texcoord_index) + 0];
                    tinyobj::real_t v = meshAttributes.texcoords[2 * static_cast<std::size_t>(index.texcoord_index) + 1];

                    // todo: 
                }

                objVertexToCustomVertexIndex[indexPair] = (mesh.vertices).size();

                (mesh.indices).push_back(static_cast<std::uint16_t>((mesh.vertices).size()));
                (mesh.vertices).push_back(vertex);
            }
            offset += static_cast<std::size_t>((shape.mesh.num_face_vertices)[faceIndex]);
        }
    }


}
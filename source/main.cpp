#include <iostream>
#include <exception>
#include <stdexcept>

#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <DirectXMath.h>

#include <wrl/client.h>

using namespace DirectX;

struct Vertex
{
    XMFLOAT3 position;
    XMFLOAT3 normal;
};

struct ConstantBuffer
{
    XMFLOAT4X4 model;
    XMFLOAT4X4 view;
    XMFLOAT4X4 projection;
};

struct LightConstantBuffer
{
    XMFLOAT3 lightDir;
    float pad0;
    XMFLOAT3 lightColor;
    float pad1;
};

static constexpr int windowWidth = 1280;
static constexpr int windowHeight = 720;
static GLFWwindow* window = nullptr;

static constexpr UINT numFrames = 3;

static constexpr D3D12_VIEWPORT viewport{0.0f, 0.0f, static_cast<float>(windowWidth), static_cast<float>(windowHeight), 0.0f, 1.0f};
static constexpr RECT scissorRect{0, 0, windowWidth, windowHeight};
static Microsoft::WRL::ComPtr<ID3D12Device> device;
static Microsoft::WRL::ComPtr<IDXGISwapChain3> swapChain;
static Microsoft::WRL::ComPtr<ID3D12Resource> swapChainRenderTargets[numFrames];
static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> swapChainRtvHeap;
static D3D12_CPU_DESCRIPTOR_HANDLE swapChainRtvHandles[numFrames] = {};
static Microsoft::WRL::ComPtr<ID3D12Resource> gBuffer0;
static Microsoft::WRL::ComPtr<ID3D12Resource> gBuffer1;
static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> gBufferRtvHeap;
static D3D12_CPU_DESCRIPTOR_HANDLE gBufferRtvHandles[2] = {};
static Microsoft::WRL::ComPtr<ID3D12Resource> depthBuffer;
static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> dsvHeap;
static D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;
static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srvHeap;
static D3D12_CPU_DESCRIPTOR_HANDLE srvRtvHandles[2] = {};
static D3D12_GPU_DESCRIPTOR_HANDLE srvGpuHandles[2] = {};
static Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocators[numFrames];
static Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandLists[numFrames];
static Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue;
static Microsoft::WRL::ComPtr<ID3D12RootSignature> geometryRootSignature;
static Microsoft::WRL::ComPtr<ID3D12PipelineState> geometryPipelineState;
static Microsoft::WRL::ComPtr<ID3D12RootSignature> lightingRootSignature;
static Microsoft::WRL::ComPtr<ID3D12PipelineState> lightingPipelineState;

static Microsoft::WRL::ComPtr<ID3D12Resource> constantBuffers[numFrames];
static void* constantBuffersMappedMemory[numFrames] = {};
static Microsoft::WRL::ComPtr<ID3D12Resource> lightConstantBuffers[numFrames];
static void* lightConstantBufferMappedMemory[numFrames] = {};
static Microsoft::WRL::ComPtr<ID3D12Resource> vertexBuffer;
static Microsoft::WRL::ComPtr<ID3D12Resource> indexBuffer;

static UINT frameIndex = 1;
static HANDLE fenceEvent;
static Microsoft::WRL::ComPtr<ID3D12Fence> fence;
static UINT64 fenceValues[numFrames] = {};

inline void ThrowIfFailed(HRESULT hr);

void LoadPipeline(HWND hWnd);

int main()
{

    try
    {
        if (glfwInit() == false)
        {
            throw std::runtime_error{"Failed to initialize GLFW"};
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(windowWidth, windowHeight, "D3D12 Renderer", nullptr, nullptr);
        if (window == nullptr)
        {
            glfwTerminate();
            
            throw std::runtime_error{"Failed to create window"};
        }

        HWND hWnd = glfwGetWin32Window(window);
        LoadPipeline(hWnd);

        Vertex vertices[] = {
            // Front face
            {{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
            {{ 0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
            {{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
            {{-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
            // Back face
            {{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
            {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
            {{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
            {{ 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
            // Top face
            {{-0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}},
            {{ 0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}},
            {{ 0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
            {{-0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
            // Bottom face
            {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}},
            {{ 0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}},
            {{ 0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}},
            {{-0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}},
            // Right face
            {{ 0.5f, -0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}},
            {{ 0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
            {{ 0.5f,  0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
            {{ 0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}},
            // Left face
            {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}},
            {{-0.5f, -0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}},
            {{-0.5f,  0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}},
            {{-0.5f,  0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}},
        };
        std::uint16_t indices[] = {
            0,1,2, 0,2,3,       // Front
            4,5,6, 4,6,7,       // Back
            8,9,10, 8,10,11,    // Top
            12,13,14, 12,14,15, // Bottom
            16,17,18, 16,18,19, // Right
            20,21,22, 20,22,23  // Left
        };

        // create vertex buffer
        D3D12_HEAP_PROPERTIES vertexBufferHeapProperties{};
        vertexBufferHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC vertexBufferDesc{};
        vertexBufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        vertexBufferDesc.Width = sizeof(vertices);
        vertexBufferDesc.Height = 1;
        vertexBufferDesc.DepthOrArraySize = 1;
        vertexBufferDesc.MipLevels = 1;
        vertexBufferDesc.SampleDesc.Count = 1;
        vertexBufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ThrowIfFailed(device->CreateCommittedResource(&vertexBufferHeapProperties, D3D12_HEAP_FLAG_NONE, &vertexBufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(vertexBuffer.GetAddressOf())));
        void* vertexBufferMemory;
        ThrowIfFailed(vertexBuffer->Map(0, nullptr, &vertexBufferMemory));
        std::memcpy(vertexBufferMemory, vertices, sizeof(vertices));
        vertexBuffer->Unmap(0, nullptr);
        D3D12_VERTEX_BUFFER_VIEW vertexBufferView{};
        vertexBufferView.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
        vertexBufferView.StrideInBytes = sizeof(Vertex);
        vertexBufferView.SizeInBytes = sizeof(vertices);

        // create index buffer
        D3D12_HEAP_PROPERTIES indexBufferHeapProperties{};
        indexBufferHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC indexBufferDesc{};
        indexBufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        indexBufferDesc.Width = sizeof(indices);
        indexBufferDesc.Height = 1;
        indexBufferDesc.DepthOrArraySize = 1;
        indexBufferDesc.MipLevels = 1;
        indexBufferDesc. SampleDesc.Count = 1;
        indexBufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ThrowIfFailed(device->CreateCommittedResource(&indexBufferHeapProperties, D3D12_HEAP_FLAG_NONE, &indexBufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(indexBuffer.GetAddressOf())));
        void* indexBufferMemory;
        ThrowIfFailed(indexBuffer->Map(0, nullptr, &indexBufferMemory));
        std::memcpy(indexBufferMemory, indices, sizeof(indices));
        indexBuffer->Unmap(0, nullptr);
        D3D12_INDEX_BUFFER_VIEW indexBufferView{};
        indexBufferView.BufferLocation = indexBuffer->GetGPUVirtualAddress();
        indexBufferView.SizeInBytes = sizeof(indices);
        indexBufferView.Format = DXGI_FORMAT_R16_UINT;

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

            // reset command allocator and command list
            ThrowIfFailed(commandAllocators[currentFrameIndex]->Reset());
            ThrowIfFailed(commandLists[currentFrameIndex]->Reset(commandAllocators[currentFrameIndex].Get(), nullptr));
            
            ID3D12Resource* backBuffer = swapChainRenderTargets[currentFrameIndex].Get();

            // create geometry transition barriers
            // GBuffer0 : from pixel shader resource to render target
            D3D12_RESOURCE_BARRIER gBuffer0Barrier{};
            gBuffer0Barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            gBuffer0Barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            gBuffer0Barrier.Transition.pResource = gBuffer0.Get();
            gBuffer0Barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            gBuffer0Barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            gBuffer0Barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            // GBuffer1 : from pixel shader resource to render target
            D3D12_RESOURCE_BARRIER gBuffer1Barrier{};
            gBuffer1Barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            gBuffer1Barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            gBuffer1Barrier.Transition.pResource = gBuffer1.Get();
            gBuffer1Barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            gBuffer1Barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            gBuffer1Barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            // DepthBuffer : from read to write
            D3D12_RESOURCE_BARRIER depthBufferBarrier{};
            depthBufferBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            depthBufferBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            depthBufferBarrier.Transition.pResource = depthBuffer.Get();
            depthBufferBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_READ;
            depthBufferBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_DEPTH_WRITE;
            const D3D12_RESOURCE_BARRIER geometryTransitionBarriers[] = {gBuffer0Barrier, gBuffer1Barrier, depthBufferBarrier};
            commandLists[currentFrameIndex]->ResourceBarrier(3, geometryTransitionBarriers);

            // calculate MVP matrices
            static float rotation = 0.0f;
            rotation += 0.01f;
            XMMATRIX model = XMMatrixRotationY(rotation) * XMMatrixTranslation(0.0f, 0.0f, 4.0f);
            XMMATRIX view = XMMatrixLookAtLH(
                XMVectorSet(0.0f, 1.0f, -3.0f, 1.0f),
                XMVectorSet(0.0f, 0.0f, 4.0f, 1.0f),
                XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f)
            );
            XMMATRIX projection = XMMatrixPerspectiveFovLH(
                XM_PIDIV4,
                static_cast<float>(windowWidth) / static_cast<float>(windowHeight),
                0.1f,
                100.0f
            );

            model = XMMatrixTranspose(model);
            view = XMMatrixTranspose(view);
            projection = XMMatrixTranspose(projection);

            // update frame's constant buffer with transposed matrices
            // HLSL expects matrices in column-major ordering
            ConstantBuffer currentFrameConstantBuffer;
            XMStoreFloat4x4(&(currentFrameConstantBuffer.model), model);
            XMStoreFloat4x4(&(currentFrameConstantBuffer.view), view);
            XMStoreFloat4x4(&(currentFrameConstantBuffer.projection), projection);
            std::memcpy(constantBuffersMappedMemory[currentFrameIndex], &currentFrameConstantBuffer, sizeof(currentFrameConstantBuffer));

            // calculate light info
            XMFLOAT3 lightDir = XMFLOAT3{-0.3f, -0.2f, -1.0f};
            XMFLOAT3 lightColor = XMFLOAT3{1.0f, 1.0f, 1.0f};

            LightConstantBuffer currentFrameLightConstantBuffer;
            currentFrameLightConstantBuffer.lightDir = lightDir;
            currentFrameLightConstantBuffer.pad0 = 0.0f;
            currentFrameLightConstantBuffer.lightColor = lightColor;
            currentFrameLightConstantBuffer.pad1 = 0.0f;
            std::memcpy(lightConstantBufferMappedMemory[currentFrameIndex], &currentFrameLightConstantBuffer, sizeof(currentFrameLightConstantBuffer));

            // set geometry pipeline
            commandLists[currentFrameIndex]->RSSetViewports(1, &viewport);
            commandLists[currentFrameIndex]->RSSetScissorRects(1, &scissorRect);
            commandLists[currentFrameIndex]->OMSetRenderTargets(2, gBufferRtvHandles, FALSE, &dsvHandle);
            commandLists[currentFrameIndex]->IASetVertexBuffers(0, 1, &vertexBufferView);
            commandLists[currentFrameIndex]->IASetIndexBuffer(&indexBufferView);
            const float gBuffer0ClearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const float gBuffer1ClearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            commandLists[currentFrameIndex]->ClearRenderTargetView(gBufferRtvHandles[0], gBuffer0ClearColor, 0, nullptr);
            commandLists[currentFrameIndex]->ClearRenderTargetView(gBufferRtvHandles[1], gBuffer1ClearColor, 0, nullptr);
            commandLists[currentFrameIndex]->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
            commandLists[currentFrameIndex]->SetPipelineState(geometryPipelineState.Get());
            commandLists[currentFrameIndex]->SetGraphicsRootSignature(geometryRootSignature.Get());
            commandLists[currentFrameIndex]->SetGraphicsRootConstantBufferView(0, constantBuffers[currentFrameIndex]->GetGPUVirtualAddress());
            commandLists[currentFrameIndex]->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            commandLists[currentFrameIndex]->DrawIndexedInstanced(36, 1, 0, 0, 0);

            // create lighting transition barriers
            // GBuffer0 : from render target to pixel shader resource
            gBuffer0Barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            gBuffer0Barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            gBuffer0Barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            // GBuffer1 : from render target to pixel shader resource
            gBuffer1Barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            gBuffer1Barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            gBuffer1Barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            // DepthBuffer : from write to read
            depthBufferBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_WRITE;
            depthBufferBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_DEPTH_READ;
            depthBufferBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            // BackBuffer : from present to render target
            D3D12_RESOURCE_BARRIER backBufferBarrier{};
            backBufferBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            backBufferBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            backBufferBarrier.Transition.pResource = backBuffer;
            backBufferBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
            backBufferBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            const D3D12_RESOURCE_BARRIER lightingTransitionBarriers[] = {gBuffer0Barrier, gBuffer1Barrier, depthBufferBarrier, backBufferBarrier};
            commandLists[currentFrameIndex]->ResourceBarrier(4, lightingTransitionBarriers);

            const float clearColor[] = {0.39f, 0.58f, 0.93f, 1.0f};
            commandLists[currentFrameIndex]->ClearRenderTargetView(swapChainRtvHandles[currentFrameIndex], clearColor, 0, nullptr);
            commandLists[currentFrameIndex]->OMSetRenderTargets(1, &(swapChainRtvHandles[currentFrameIndex]), FALSE, nullptr);
            ID3D12DescriptorHeap* heaps[] = {srvHeap.Get()};
            commandLists[currentFrameIndex]->SetDescriptorHeaps(1, heaps);
            commandLists[currentFrameIndex]->SetPipelineState(lightingPipelineState.Get());
            commandLists[currentFrameIndex]->SetGraphicsRootSignature(lightingRootSignature.Get());
            commandLists[currentFrameIndex]->SetGraphicsRootDescriptorTable(0, srvGpuHandles[0]);
            commandLists[currentFrameIndex]->SetGraphicsRootConstantBufferView(1, lightConstantBuffers[currentFrameIndex]->GetGPUVirtualAddress());
            commandLists[currentFrameIndex]->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            commandLists[currentFrameIndex]->DrawInstanced(3, 1, 0, 0);

            // BackBuffer : from render target to  present
            backBufferBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            backBufferBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
            backBufferBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            commandLists[currentFrameIndex]->ResourceBarrier(1, &backBufferBarrier);

            ThrowIfFailed(commandLists[currentFrameIndex]->Close());

            // execute
            ID3D12CommandList* currentFrameCommandList[] = {commandLists[currentFrameIndex].Get()};
            commandQueue->ExecuteCommandLists(1, currentFrameCommandList);

            // present
            ThrowIfFailed(swapChain->Present(1, 0));

            ++frameIndex;
            ThrowIfFailed(commandQueue->Signal(fence.Get(), frameIndex));
            fenceValues[currentFrameIndex] = frameIndex;
        }

        glfwDestroyWindow(window);
        glfwTerminate();

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}

inline void ThrowIfFailed(HRESULT hr)
{
    if (FAILED(hr))
    {
        throw std::exception{"Win32 Function Failed"};
    }
}

void LoadPipeline(HWND hWnd)
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

    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf())));
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    // todo: create adapter

    Microsoft::WRL::ComPtr<ID3DBlob> errorBlob;

    // create device
    ThrowIfFailed(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.GetAddressOf())));

    // create command queue
    D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
    commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    ThrowIfFailed(device->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(commandQueue.GetAddressOf())));

    // create swap chain
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc{};
    swapChainDesc.BufferCount = numFrames;
    swapChainDesc.Width = windowWidth;
    swapChainDesc.Height = windowHeight;
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.SampleDesc.Count = 1;
    Microsoft::WRL::ComPtr<IDXGISwapChain1> swapChain1;
    ThrowIfFailed(factory->CreateSwapChainForHwnd(commandQueue.Get(), hWnd, &swapChainDesc, nullptr, nullptr, swapChain1.GetAddressOf()));
    ThrowIfFailed(swapChain1.As(&swapChain));

    // create depth buffer
    D3D12_HEAP_PROPERTIES depthBufferHeapProperties{};
    depthBufferHeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC depthBufferDesc{};
    depthBufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    depthBufferDesc.Width = windowWidth;
    depthBufferDesc.Height = windowHeight;
    depthBufferDesc.DepthOrArraySize = 1;
    depthBufferDesc.MipLevels = 1;
    depthBufferDesc.Format = DXGI_FORMAT_D32_FLOAT;
    depthBufferDesc.SampleDesc = {1, 0};
    depthBufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE depthBufferClearValue{};
    depthBufferClearValue.Format = depthBufferDesc.Format;
    depthBufferClearValue.DepthStencil.Depth = 1.0f;  // far plane distance
    ThrowIfFailed(device->CreateCommittedResource(&depthBufferHeapProperties, D3D12_HEAP_FLAG_NONE, &depthBufferDesc, D3D12_RESOURCE_STATE_DEPTH_READ, &depthBufferClearValue, IID_PPV_ARGS(depthBuffer.GetAddressOf())));
    
    // create RTVs for swap chain buffers
    D3D12_DESCRIPTOR_HEAP_DESC swapChainRtvHeapDesc{};
    swapChainRtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    swapChainRtvHeapDesc.NumDescriptors = numFrames;
    ThrowIfFailed(device->CreateDescriptorHeap(&swapChainRtvHeapDesc, IID_PPV_ARGS(swapChainRtvHeap.GetAddressOf())));
    const UINT swapChainRtvDescSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    const D3D12_CPU_DESCRIPTOR_HANDLE swapChainRtvHandleStart = swapChainRtvHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < numFrames; ++i)
    {
        ThrowIfFailed(swapChain->GetBuffer(i, IID_PPV_ARGS(swapChainRenderTargets[i].GetAddressOf())));
        swapChainRtvHandles[i] = swapChainRtvHandleStart;
        swapChainRtvHandles[i].ptr += SIZE_T(i) * SIZE_T(swapChainRtvDescSize);
        device->CreateRenderTargetView(swapChainRenderTargets[i].Get(), nullptr, swapChainRtvHandles[i]);
    }

    // create RTVs for G-buffer
    D3D12_HEAP_PROPERTIES gBuffer0HeapProperties{};
    gBuffer0HeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC gBuffer0Desc{};
    gBuffer0Desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    gBuffer0Desc.Width = windowWidth;
    gBuffer0Desc.Height = windowHeight;
    gBuffer0Desc.DepthOrArraySize = 1;
    gBuffer0Desc.MipLevels = 1;
    gBuffer0Desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    gBuffer0Desc.SampleDesc = {1, 0};
    gBuffer0Desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE gBuffer0ClearValue{};
    gBuffer0ClearValue.Format = gBuffer0Desc.Format;
    ThrowIfFailed(device->CreateCommittedResource(&gBuffer0HeapProperties, D3D12_HEAP_FLAG_NONE, &gBuffer0Desc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &gBuffer0ClearValue, IID_PPV_ARGS(gBuffer0.GetAddressOf())));
    D3D12_HEAP_PROPERTIES gBuffer1HeapProperties{};
    gBuffer1HeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC gBuffer1Desc{};
    gBuffer1Desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    gBuffer1Desc.Width = windowWidth;
    gBuffer1Desc.Height = windowHeight;
    gBuffer1Desc.DepthOrArraySize = 1;
    gBuffer1Desc.MipLevels = 1;
    gBuffer1Desc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
    gBuffer1Desc.SampleDesc = {1, 0};
    gBuffer1Desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE gBuffer1ClearValue{};
    gBuffer1ClearValue.Format = gBuffer1Desc.Format;
    ThrowIfFailed(device->CreateCommittedResource(&gBuffer1HeapProperties, D3D12_HEAP_FLAG_NONE, &gBuffer1Desc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &gBuffer1ClearValue, IID_PPV_ARGS(gBuffer1.GetAddressOf())));
    D3D12_DESCRIPTOR_HEAP_DESC gBufferRtvHeapDesc{};
    gBufferRtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    gBufferRtvHeapDesc.NumDescriptors = 2;
    ThrowIfFailed(device->CreateDescriptorHeap(&gBufferRtvHeapDesc, IID_PPV_ARGS(gBufferRtvHeap.GetAddressOf())));
    const UINT gBufferRtvDescSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    const D3D12_CPU_DESCRIPTOR_HANDLE gBufferRtvHandleStart = gBufferRtvHeap->GetCPUDescriptorHandleForHeapStart();
    gBufferRtvHandles[0] = gBufferRtvHandleStart;
    gBufferRtvHandles[1] = {gBufferRtvHandles[0].ptr + gBufferRtvDescSize};
    device->CreateRenderTargetView(gBuffer0.Get(), nullptr, gBufferRtvHandles[0]);
    device->CreateRenderTargetView(gBuffer1.Get(), nullptr, gBufferRtvHandles[1]);

    // create RTV for depth buffer (DSV)
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc{};
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.NumDescriptors = 1;
    ThrowIfFailed(device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(dsvHeap.GetAddressOf())));
    dsvHandle = dsvHeap->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(depthBuffer.Get(), nullptr, dsvHandle);

    // create SRV
    D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc{};
    srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    srvHeapDesc.NumDescriptors = 2;
    ThrowIfFailed(device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(srvHeap.GetAddressOf())));
    const UINT srvRtvDescSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    srvRtvHandles[0] = srvHeap->GetCPUDescriptorHandleForHeapStart();
    srvRtvHandles[1] = {srvRtvHandles[0].ptr + srvRtvDescSize};
    D3D12_SHADER_RESOURCE_VIEW_DESC srv0Desc{};
    srv0Desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv0Desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv0Desc.Texture2D.MipLevels = 1;
    srv0Desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    device->CreateShaderResourceView(gBuffer0.Get(), &srv0Desc, srvRtvHandles[0]);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv1Desc{};
    srv1Desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv1Desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv1Desc.Texture2D.MipLevels = 1;
    srv1Desc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
    device->CreateShaderResourceView(gBuffer1.Get(), &srv1Desc, srvRtvHandles[1]);
    srvGpuHandles[0] = srvHeap->GetGPUDescriptorHandleForHeapStart();
    srvGpuHandles[1] = {srvGpuHandles[0].ptr + srvRtvDescSize};

    // setup per-frame resources
    // maintain one set of GPU recording state per swapchain buffer
    // create a command allocator/list for each swap chain buffer
    for (UINT i = 0; i < numFrames; ++i)
    {
        ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(commandAllocators[i].GetAddressOf())));
        ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocators[i].Get(), nullptr, IID_PPV_ARGS(commandLists[i].GetAddressOf())));
        ThrowIfFailed(commandLists[i]->Close());
    }
    // setup a single fence
    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())));
    frameIndex = 1;
    fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    // setup per-frame constant buffers (persistently mapped)
    D3D12_HEAP_PROPERTIES constantBuffersHeapProperties{};
    constantBuffersHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC constantBuffersResourceDesc{};
    constantBuffersResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    constantBuffersResourceDesc.Width = (sizeof(ConstantBuffer) + 255) & ~255;
    constantBuffersResourceDesc.Height = 1;
    constantBuffersResourceDesc.DepthOrArraySize = 1;
    constantBuffersResourceDesc.MipLevels = 1;
    constantBuffersResourceDesc.SampleDesc = {1, 0};
    constantBuffersResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    for (UINT i = 0; i < numFrames; ++i)
    {
        ThrowIfFailed(device->CreateCommittedResource(&constantBuffersHeapProperties, D3D12_HEAP_FLAG_NONE, &constantBuffersResourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(constantBuffers[i].GetAddressOf())));
        ThrowIfFailed(constantBuffers[i]->Map(0, nullptr, &(constantBuffersMappedMemory[i])));
    }
    // setup per-frame light constant buffers (persistently mapped)
    D3D12_HEAP_PROPERTIES lightConstantBuffersHeapProperties{};
    lightConstantBuffersHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC lightConstantBuffersResourceDesc{};
    lightConstantBuffersResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    lightConstantBuffersResourceDesc.Width = (sizeof(LightConstantBuffer) + 255) & ~255;
    lightConstantBuffersResourceDesc.Height = 1;
    lightConstantBuffersResourceDesc.DepthOrArraySize = 1;
    lightConstantBuffersResourceDesc.MipLevels = 1;
    lightConstantBuffersResourceDesc.SampleDesc = {1, 0};
    lightConstantBuffersResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    for (UINT i = 0; i < numFrames; ++i)
    {
        ThrowIfFailed(device->CreateCommittedResource(&lightConstantBuffersHeapProperties, D3D12_HEAP_FLAG_NONE, &lightConstantBuffersResourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(lightConstantBuffers[i].GetAddressOf())));
        ThrowIfFailed(lightConstantBuffers[i]->Map(0, nullptr, &(lightConstantBufferMappedMemory[i])));
    }

    // compile vertex shader
    Microsoft::WRL::ComPtr<ID3DBlob> vertexShaderBlob;
    if (FAILED(D3DCompileFromFile(L"../assets/shaders/geometry.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION, 0, vertexShaderBlob.GetAddressOf(), errorBlob.GetAddressOf())))
    {
        if (errorBlob != nullptr)
        {
            std::cerr << (char*)(errorBlob->GetBufferPointer()) << std::endl;
        }
        throw std::runtime_error{"Failed to compile vertex shader"};
    }
    errorBlob.Reset();
    // compile pixel shader
    Microsoft::WRL::ComPtr<ID3DBlob> pixelShaderBlob;
    if (FAILED(D3DCompileFromFile(L"../assets/shaders/geometry.hlsl", nullptr, nullptr, "PSMain", "ps_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION, 0, pixelShaderBlob.GetAddressOf(), errorBlob.GetAddressOf())))
    {
        if (errorBlob != nullptr)
        {
            std::cerr << (char*)(errorBlob->GetBufferPointer()) << std::endl;
        }
        throw std::runtime_error{"Failed to compile pixel shader"};
    }
    errorBlob.Reset();

    // create geometry root signature
    D3D12_ROOT_PARAMETER geometryRootParameter{};
    geometryRootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    geometryRootParameter.Descriptor.ShaderRegister = 0;  // b0
    geometryRootParameter.Descriptor.RegisterSpace = 0;
    geometryRootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC geometryRootSignatureDesc{};
    geometryRootSignatureDesc.NumParameters = 1;
    geometryRootSignatureDesc.pParameters = &geometryRootParameter;
    geometryRootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    Microsoft::WRL::ComPtr<ID3DBlob> geometryRootSignatureBlob;
    ThrowIfFailed(D3D12SerializeRootSignature(&geometryRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, geometryRootSignatureBlob.GetAddressOf(), errorBlob.GetAddressOf()));
    ThrowIfFailed(device->CreateRootSignature(0, geometryRootSignatureBlob->GetBufferPointer(), geometryRootSignatureBlob->GetBufferSize(), IID_PPV_ARGS(geometryRootSignature.GetAddressOf())));
    errorBlob.Reset();

    // create geometry PSO
    D3D12_INPUT_ELEMENT_DESC geometryInputElementDescs[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };
    D3D12_GRAPHICS_PIPELINE_STATE_DESC geometryPsoDesc{};
    geometryPsoDesc.pRootSignature = geometryRootSignature.Get();
    geometryPsoDesc.VS = {vertexShaderBlob->GetBufferPointer(), vertexShaderBlob->GetBufferSize()};
    geometryPsoDesc.PS = {pixelShaderBlob->GetBufferPointer(), pixelShaderBlob->GetBufferSize()};
    geometryPsoDesc.InputLayout = {geometryInputElementDescs, 2};
    geometryPsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    geometryPsoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    geometryPsoDesc.RTVFormats[1] = DXGI_FORMAT_R10G10B10A2_UNORM;
    geometryPsoDesc.NumRenderTargets = 2;
    geometryPsoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    D3D12_RASTERIZER_DESC rasterizerDesc{};
    rasterizerDesc.FillMode = D3D12_FILL_MODE_SOLID;
    rasterizerDesc.CullMode = D3D12_CULL_MODE_BACK;
    rasterizerDesc.FrontCounterClockwise = FALSE;
    rasterizerDesc.DepthBias = 0;
    rasterizerDesc.DepthBiasClamp = 0.0f;
    rasterizerDesc.MultisampleEnable = FALSE;
    rasterizerDesc.AntialiasedLineEnable = FALSE;
    rasterizerDesc.ForcedSampleCount = 0;
    rasterizerDesc.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    rasterizerDesc.DepthClipEnable = TRUE;
    geometryPsoDesc.RasterizerState = rasterizerDesc;
    D3D12_BLEND_DESC blendDesc{};
    blendDesc.AlphaToCoverageEnable = FALSE;
    blendDesc.IndependentBlendEnable = FALSE;
    for (int i = 0; i < 8; ++i)
    {
        auto& renderTarget = blendDesc.RenderTarget[i];
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
    geometryPsoDesc.BlendState = blendDesc;
    geometryPsoDesc.SampleMask = UINT_MAX;
    D3D12_DEPTH_STENCIL_DESC dsDesc{};
    dsDesc.DepthEnable = TRUE;
    dsDesc.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    dsDesc.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    dsDesc.StencilEnable = FALSE;
    geometryPsoDesc.DepthStencilState = dsDesc;
    geometryPsoDesc.SampleDesc = {1, 0};
    ThrowIfFailed(device->CreateGraphicsPipelineState(&geometryPsoDesc, IID_PPV_ARGS(geometryPipelineState.GetAddressOf())));

    // compile vertex shader
    vertexShaderBlob.Reset();
    if (FAILED(D3DCompileFromFile(L"../assets/shaders/lighting.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION, 0, vertexShaderBlob.GetAddressOf(), errorBlob.GetAddressOf())))
    {
        if (errorBlob != nullptr)
        {
            std::cerr << (char*)(errorBlob->GetBufferPointer()) << std::endl;
        }
        throw std::runtime_error{"Failed to compile vertex shader"};
    }
    errorBlob.Reset();
    // compile pixel shader
    pixelShaderBlob.Reset();
    if (FAILED(D3DCompileFromFile(L"../assets/shaders/lighting.hlsl", nullptr, nullptr, "PSMain", "ps_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION, 0, pixelShaderBlob.GetAddressOf(), errorBlob.GetAddressOf())))
    {
        if (errorBlob != nullptr)
        {
            std::cerr << (char*)(errorBlob->GetBufferPointer()) << std::endl;
        }
        throw std::runtime_error{"Failed to compile pixel shader"};
    }
    errorBlob.Reset();

    // create lighting root signature
    D3D12_DESCRIPTOR_RANGE srvDescRange = {};
    srvDescRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvDescRange.NumDescriptors = 2;
    srvDescRange.BaseShaderRegister = 0;
    D3D12_ROOT_PARAMETER lightingRootParameterTable = {};
    lightingRootParameterTable.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    lightingRootParameterTable.DescriptorTable.NumDescriptorRanges = 1;
    lightingRootParameterTable.DescriptorTable.pDescriptorRanges = &srvDescRange;
    lightingRootParameterTable.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_PARAMETER lightingRootParameterCbv = {};
    lightingRootParameterCbv.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    lightingRootParameterCbv.Descriptor.ShaderRegister = 0;
    lightingRootParameterCbv.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_STATIC_SAMPLER_DESC lightingSamplerDesc = {};
    lightingSamplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    lightingSamplerDesc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    lightingSamplerDesc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    lightingSamplerDesc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    lightingSamplerDesc.ShaderRegister = 0;
    lightingSamplerDesc.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    const D3D12_ROOT_PARAMETER parameters[2] = {lightingRootParameterTable, lightingRootParameterCbv};
    D3D12_ROOT_SIGNATURE_DESC lightingRootSignatureDesc = {};
    lightingRootSignatureDesc.NumParameters = 2;
    lightingRootSignatureDesc.pParameters = parameters;
    lightingRootSignatureDesc.NumStaticSamplers = 1;
    lightingRootSignatureDesc.pStaticSamplers = &lightingSamplerDesc;
    Microsoft::WRL::ComPtr<ID3DBlob> lightingRootSignatureBlob;
    ThrowIfFailed(D3D12SerializeRootSignature(&lightingRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, lightingRootSignatureBlob.GetAddressOf(), errorBlob.GetAddressOf()));
    ThrowIfFailed(device->CreateRootSignature(0, lightingRootSignatureBlob->GetBufferPointer(), lightingRootSignatureBlob->GetBufferSize(), IID_PPV_ARGS(lightingRootSignature.GetAddressOf())));
    errorBlob.Reset();

    // create lighting pso
    D3D12_GRAPHICS_PIPELINE_STATE_DESC lightingPsoDesc{};
    lightingPsoDesc.pRootSignature = lightingRootSignature.Get();
    lightingPsoDesc.VS = {vertexShaderBlob->GetBufferPointer(), vertexShaderBlob->GetBufferSize()};
    lightingPsoDesc.PS = {pixelShaderBlob->GetBufferPointer(), pixelShaderBlob->GetBufferSize()};
    lightingPsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    lightingPsoDesc.NumRenderTargets = 1;
    lightingPsoDesc.RTVFormats[0] = swapChainDesc.Format;
    D3D12_RASTERIZER_DESC lightingRasterizerDesc = rasterizerDesc;
    lightingRasterizerDesc.CullMode = D3D12_CULL_MODE_NONE;
    lightingPsoDesc.RasterizerState = lightingRasterizerDesc;
    lightingPsoDesc.BlendState = blendDesc;
    D3D12_DEPTH_STENCIL_DESC lightDsDesc{};
    lightDsDesc.DepthEnable = FALSE;
    lightingPsoDesc.DepthStencilState = lightDsDesc;
    lightingPsoDesc.SampleDesc = {1, 0};
    lightingPsoDesc.SampleMask = UINT_MAX;
    ThrowIfFailed(device->CreateGraphicsPipelineState(&lightingPsoDesc, IID_PPV_ARGS(lightingPipelineState.GetAddressOf())));
}
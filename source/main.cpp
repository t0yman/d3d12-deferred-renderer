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

static constexpr UINT numFrames = 3;
static constexpr UINT numRtvDescriptors = 255;
static constexpr UINT numDsvDescriptors = 1;
static constexpr UINT numSrvDescriptors = 2;

static constexpr D3D12_VIEWPORT viewport{0.0f, 0.0f, static_cast<float>(windowWidth), static_cast<float>(windowHeight), 0.0f, 1.0f};
static constexpr RECT scissorRect{0, 0, windowWidth, windowHeight};
static Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
static Microsoft::WRL::ComPtr<ID3D12Device> device;
static Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue;
static Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocators[numFrames];
static Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandLists[numFrames];
static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtvDescriptorHeap;
static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> dsvDescriptorHeap;
static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srvDescriptorHeap;
static Microsoft::WRL::ComPtr<IDXGISwapChain3> swapChain;
static Microsoft::WRL::ComPtr<ID3D12Resource> swapChainBackBuffers[numFrames];
static D3D12_CPU_DESCRIPTOR_HANDLE swapChainRtvDescriptorHandles[numFrames] = {};
static Microsoft::WRL::ComPtr<ID3D12Resource> gBuffer[2];
static D3D12_CPU_DESCRIPTOR_HANDLE gBufferRtvDescriptorHandles[2] = {};
static Microsoft::WRL::ComPtr<ID3D12Resource> depthBuffer;
static D3D12_CPU_DESCRIPTOR_HANDLE depthBufferDsvDescriptorHandle;
static D3D12_CPU_DESCRIPTOR_HANDLE srvCpuDescriptorHandles[numSrvDescriptors] = {};
static D3D12_GPU_DESCRIPTOR_HANDLE srvGpuDescriptorHandles[numSrvDescriptors] = {};
static Microsoft::WRL::ComPtr<ID3DBlob> geometryVertexShaderBlob;
static Microsoft::WRL::ComPtr<ID3DBlob> geometryPixelShaderBlob;
static Microsoft::WRL::ComPtr<ID3DBlob> lightingVertexShaderBlob;
static Microsoft::WRL::ComPtr<ID3DBlob> lightingPixelShaderBlob;
static Microsoft::WRL::ComPtr<ID3D12RootSignature> geometryShaderRootSignagture;
static Microsoft::WRL::ComPtr<ID3D12RootSignature> lightingShaderRootSignature;

static Microsoft::WRL::ComPtr<ID3D12PipelineState> geometryShaderPipelineState;
static Microsoft::WRL::ComPtr<ID3D12PipelineState> lightingShaderPipelineState;

static Microsoft::WRL::ComPtr<ID3D12Fence> fence;
static UINT frameIndex = 1;
static HANDLE fenceEvent;
static UINT64 fenceValues[numFrames] = {};

static Microsoft::WRL::ComPtr<ID3D12Resource> vertexBuffer;
static Microsoft::WRL::ComPtr<ID3D12Resource> indexBuffer;

static Microsoft::WRL::ComPtr<ID3D12Resource> mvpInfoBuffers[numFrames];
static void* mvpInfoBuffersMappedMemory[numFrames] = {};
static Microsoft::WRL::ComPtr<ID3D12Resource> lightInfoBuffers[numFrames];
static void* lightInfoBuffersMappedMemory[numFrames] = {};

inline static void ThrowIfFailed(HRESULT hResult);

static void LoadPipeline(HWND hWnd);

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

        Vertex vertices[] = {
            // front face
            {{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
            {{ 0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
            {{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
            {{-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
            // back face
            {{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
            {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
            {{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
            {{ 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
            // top face
            {{-0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}},
            {{ 0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}},
            {{ 0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
            {{-0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
            // bottom face
            {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}},
            {{ 0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}},
            {{ 0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}},
            {{-0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}},
            // right face
            {{ 0.5f, -0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}},
            {{ 0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
            {{ 0.5f,  0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
            {{ 0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}},
            // left face
            {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}},
            {{-0.5f, -0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}},
            {{-0.5f,  0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}},
            {{-0.5f,  0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}},
        };
        std::uint16_t indices[] = {
            0,1,2, 0,2,3,       // front
            4,5,6, 4,6,7,       // back
            8,9,10, 8,10,11,    // top
            12,13,14, 12,14,15, // bottom
            16,17,18, 16,18,19, // right
            20,21,22, 20,22,23  // left
        };

        // create vertex buffer
        D3D12_HEAP_PROPERTIES vertexBufferHeapProperties{};
        vertexBufferHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC vertexBufferResourceDesc{};
        vertexBufferResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        vertexBufferResourceDesc.Width = sizeof(vertices);
        vertexBufferResourceDesc.Height = 1;
        vertexBufferResourceDesc.DepthOrArraySize = 1;
        vertexBufferResourceDesc.MipLevels = 1;
        vertexBufferResourceDesc.SampleDesc = {1, 0};
        vertexBufferResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        // todo:
        ThrowIfFailed(device->CreateCommittedResource(&vertexBufferHeapProperties, D3D12_HEAP_FLAG_NONE, &vertexBufferResourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(vertexBuffer.GetAddressOf())));
        void* vertexBufferMappedMemory;
        ThrowIfFailed(vertexBuffer->Map(0, nullptr, &vertexBufferMappedMemory));
        std::memcpy(vertexBufferMappedMemory, vertices, sizeof(vertices));
        vertexBuffer->Unmap(0, nullptr);
        D3D12_VERTEX_BUFFER_VIEW vertexBufferView{};
        vertexBufferView.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
        vertexBufferView.StrideInBytes = sizeof(Vertex);
        vertexBufferView.SizeInBytes = sizeof(vertices);

        // create index buffer
        D3D12_HEAP_PROPERTIES indexBufferHeapProperties{};
        indexBufferHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC indexBufferResourceDesc{};
        indexBufferResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        indexBufferResourceDesc.Width = sizeof(indices);
        indexBufferResourceDesc.Height = 1;
        indexBufferResourceDesc.DepthOrArraySize = 1;
        indexBufferResourceDesc.MipLevels = 1;
        indexBufferResourceDesc.SampleDesc = {1, 0};
        indexBufferResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ThrowIfFailed(device->CreateCommittedResource(&indexBufferHeapProperties, D3D12_HEAP_FLAG_NONE, &indexBufferResourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(indexBuffer.GetAddressOf())));
        void* indexBufferMappedMemory;
        ThrowIfFailed(indexBuffer->Map(0, nullptr, &indexBufferMappedMemory));
        std::memcpy(indexBufferMappedMemory, indices, sizeof(indices));
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

            ThrowIfFailed(commandAllocators[currentFrameIndex]->Reset());
            ThrowIfFailed(commandLists[currentFrameIndex]->Reset(commandAllocators[currentFrameIndex].Get(), nullptr));

            ID3D12Resource* backBuffer = swapChainBackBuffers[currentFrameIndex].Get();

            static float rotation = 0.0f;
            rotation += 0.01f;

            // calculate MVP
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

            // upload mvp to geometry shader
            MvpInfo currentFrameMvpInfo;
            XMStoreFloat4x4(&(currentFrameMvpInfo.model), model);
            XMStoreFloat4x4(&(currentFrameMvpInfo.view), view);
            XMStoreFloat4x4(&(currentFrameMvpInfo.projection), projection);
            std::memcpy(mvpInfoBuffersMappedMemory[currentFrameIndex], &currentFrameMvpInfo, sizeof(currentFrameMvpInfo));

            // upload light info to lighting shader
            LightInfo currentFrameLightInfo;
            currentFrameLightInfo.lightDir = XMFLOAT3{-0.3f, -0.2f, -1.0f};
            currentFrameLightInfo.pad0 = 0.0f;
            currentFrameLightInfo.lightColor = XMFLOAT3{1.0f, 1.0f, 1.0f};;
            currentFrameLightInfo.pad1 = 0.0f;
            std::memcpy(lightInfoBuffersMappedMemory[currentFrameIndex], &currentFrameLightInfo, sizeof(currentFrameLightInfo));

            // set-up geometry pipeline to draw to G-Buffer
            commandLists[currentFrameIndex]->RSSetViewports(1, &viewport);
            commandLists[currentFrameIndex]->RSSetScissorRects(1, &scissorRect);
            commandLists[currentFrameIndex]->OMSetRenderTargets(2, gBufferRtvDescriptorHandles, FALSE, &depthBufferDsvDescriptorHandle);
            commandLists[currentFrameIndex]->IASetVertexBuffers(0, 1, &vertexBufferView);
            commandLists[currentFrameIndex]->IASetIndexBuffer(&indexBufferView);
            const float gBuffer0ClearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const float gBuffer1ClearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            commandLists[currentFrameIndex]->ClearRenderTargetView(gBufferRtvDescriptorHandles[0], gBuffer0ClearColor, 0, nullptr);
            commandLists[currentFrameIndex]->ClearRenderTargetView(gBufferRtvDescriptorHandles[1], gBuffer1ClearColor, 0, nullptr);
            commandLists[currentFrameIndex]->ClearDepthStencilView(depthBufferDsvDescriptorHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
            commandLists[currentFrameIndex]->SetPipelineState(geometryShaderPipelineState.Get());
            commandLists[currentFrameIndex]->SetGraphicsRootSignature(geometryShaderRootSignagture.Get());
            commandLists[currentFrameIndex]->SetGraphicsRootConstantBufferView(0, mvpInfoBuffers[currentFrameIndex]->GetGPUVirtualAddress());
            commandLists[currentFrameIndex]->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            commandLists[currentFrameIndex]->DrawIndexedInstanced(36, 1, 0, 0, 0);

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

            // set-up lighting pipeline to draw finial image
            const float clearColor[] = {0.39f, 0.58f, 0.93f, 1.0f};
            commandLists[currentFrameIndex]->ClearRenderTargetView(swapChainRtvDescriptorHandles[currentFrameIndex], clearColor, 0, nullptr);
            commandLists[currentFrameIndex]->OMSetRenderTargets(1, &(swapChainRtvDescriptorHandles[currentFrameIndex]), FALSE, nullptr);
            ID3D12DescriptorHeap* heaps[] = {srvDescriptorHeap.Get()};
            commandLists[currentFrameIndex]->SetDescriptorHeaps(1, heaps);
            commandLists[currentFrameIndex]->SetPipelineState(lightingShaderPipelineState.Get());
            commandLists[currentFrameIndex]->SetGraphicsRootSignature(lightingShaderRootSignature.Get());
            commandLists[currentFrameIndex]->SetGraphicsRootDescriptorTable(0, srvGpuDescriptorHandles[0]);
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

            ID3D12CommandList* currentFrameCommandList[] = {commandLists[currentFrameIndex].Get()};
            commandQueue->ExecuteCommandLists(1, currentFrameCommandList);

            ThrowIfFailed(swapChain->Present(1, 0));

            ++frameIndex;
            ThrowIfFailed(commandQueue->Signal(fence.Get(), frameIndex));
            fenceValues[currentFrameIndex] = frameIndex;
        }

        CloseHandle(fenceEvent);

        glfwDestroyWindow(window);
        glfwTerminate();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
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
    // todo: create adapter

    // create device
    ThrowIfFailed(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.GetAddressOf())));

    // create command queue
    D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
    commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    ThrowIfFailed(device->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(commandQueue.GetAddressOf())));

    // create command allocators and command lists
    for (UINT i = 0; i < numFrames; ++i)
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
    dsvDescriptorHeapDesc.NumDescriptors = numDsvDescriptors;
    ThrowIfFailed(device->CreateDescriptorHeap(&dsvDescriptorHeapDesc, IID_PPV_ARGS(dsvDescriptorHeap.GetAddressOf())));

    // create SRV desciptor heap
    D3D12_DESCRIPTOR_HEAP_DESC srvDescriptorHeapDesc{};
    srvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvDescriptorHeapDesc.NumDescriptors = numSrvDescriptors;
    srvDescriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(device->CreateDescriptorHeap(&srvDescriptorHeapDesc, IID_PPV_ARGS(srvDescriptorHeap.GetAddressOf())));

    // create swap chain
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc{};
    swapChainDesc.BufferCount = numFrames;
    swapChainDesc.Width = windowWidth;
    swapChainDesc.Height = windowHeight;
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.SampleDesc = {1, 0};
    Microsoft::WRL::ComPtr<IDXGISwapChain1> swapChain1;
    ThrowIfFailed(factory->CreateSwapChainForHwnd(commandQueue.Get(), hWnd, &swapChainDesc, nullptr, nullptr, swapChain1.GetAddressOf()));
    ThrowIfFailed(swapChain1.As(&swapChain));

    // create swap chain back buffer RTVs
    const UINT rtvDescriptorHandleIncrementSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    const D3D12_CPU_DESCRIPTOR_HANDLE rtvDescriptorHeapStartHandle = rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < numFrames; ++i)
    {
        ThrowIfFailed(swapChain->GetBuffer(i, IID_PPV_ARGS(swapChainBackBuffers[i].GetAddressOf())));

        swapChainRtvDescriptorHandles[i] = rtvDescriptorHeapStartHandle;
        swapChainRtvDescriptorHandles[i].ptr += SIZE_T(i) * SIZE_T(rtvDescriptorHandleIncrementSize);
        device->CreateRenderTargetView(swapChainBackBuffers[i].Get(), nullptr, swapChainRtvDescriptorHandles[i]);
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

    // create G-Buffer RTVs
    const D3D12_CPU_DESCRIPTOR_HANDLE nextFreeRtvDescriptorHandle = {rtvDescriptorHeapStartHandle.ptr + SIZE_T(numFrames) * SIZE_T(rtvDescriptorHandleIncrementSize)};
    for (UINT i = 0; i < 2; ++i)
    {
        gBufferRtvDescriptorHandles[i]= {nextFreeRtvDescriptorHandle.ptr + SIZE_T(i) * SIZE_T(rtvDescriptorHandleIncrementSize)};
        device->CreateRenderTargetView(gBuffer[i].Get(), nullptr, gBufferRtvDescriptorHandles[i]);
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
    depthBufferDsvDescriptorHandle = dsvDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(depthBuffer.Get(), nullptr, depthBufferDsvDescriptorHandle);

    // create SRVs
    const UINT srvDescriptorHandleIncrementSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const D3D12_CPU_DESCRIPTOR_HANDLE srvDescriptorHeapStartHandle = srvDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < 2; ++i)
    {
        srvCpuDescriptorHandles[i] = srvDescriptorHeapStartHandle;
        srvCpuDescriptorHandles[i].ptr += SIZE_T(i) * SIZE_T(srvDescriptorHandleIncrementSize);
    }
    
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    device->CreateShaderResourceView(gBuffer[0].Get(), &srvDesc, srvCpuDescriptorHandles[0]);
    
    srvDesc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
    device->CreateShaderResourceView(gBuffer[1].Get(), &srvDesc, srvCpuDescriptorHandles[1]);

    srvGpuDescriptorHandles[0] = srvDescriptorHeap->GetGPUDescriptorHandleForHeapStart();
    srvGpuDescriptorHandles[1] = {srvGpuDescriptorHandles[0].ptr + srvDescriptorHandleIncrementSize};

    // create gemoetry shader root signature
    D3D12_ROOT_PARAMETER geometryShaderRootParameter{};
    geometryShaderRootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    geometryShaderRootParameter.Descriptor.ShaderRegister = 0;
    geometryShaderRootParameter.Descriptor.RegisterSpace = 0;
    geometryShaderRootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC geometryShaderRootSignatureDesc{};
    geometryShaderRootSignatureDesc.NumParameters = 1;
    geometryShaderRootSignatureDesc.pParameters = &geometryShaderRootParameter;
    geometryShaderRootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    Microsoft::WRL::ComPtr<ID3DBlob> errorBlob;
    Microsoft::WRL::ComPtr<ID3DBlob> geometryShaderRootSignagtureBlob;
    ThrowIfFailed(D3D12SerializeRootSignature(&geometryShaderRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, geometryShaderRootSignagtureBlob.GetAddressOf(), errorBlob.GetAddressOf()));
    ThrowIfFailed(device->CreateRootSignature(0, geometryShaderRootSignagtureBlob->GetBufferPointer(), geometryShaderRootSignagtureBlob->GetBufferSize(), IID_PPV_ARGS(geometryShaderRootSignagture.GetAddressOf())));
    errorBlob.Reset();

    // create lighting shader root signature
    D3D12_DESCRIPTOR_RANGE srvDescriptorRange{};
    srvDescriptorRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvDescriptorRange.NumDescriptors = 2;
    srvDescriptorRange.BaseShaderRegister = 0;
    D3D12_ROOT_PARAMETER lightingShaderDescriptorTableRootParameter{};
    lightingShaderDescriptorTableRootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    lightingShaderDescriptorTableRootParameter.DescriptorTable.NumDescriptorRanges = 1;
    lightingShaderDescriptorTableRootParameter.DescriptorTable.pDescriptorRanges = &srvDescriptorRange;
    lightingShaderDescriptorTableRootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_PARAMETER lightingShaderCbvRootParameter{};
    lightingShaderCbvRootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    lightingShaderCbvRootParameter.Descriptor.ShaderRegister = 0;
    lightingShaderCbvRootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_STATIC_SAMPLER_DESC lightingShaderLinearStaticSampler{};
    lightingShaderLinearStaticSampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    lightingShaderLinearStaticSampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    lightingShaderLinearStaticSampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    lightingShaderLinearStaticSampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    lightingShaderLinearStaticSampler.ShaderRegister = 0;
    lightingShaderLinearStaticSampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    const D3D12_ROOT_PARAMETER lightingShaderRootParameters[]  = {lightingShaderDescriptorTableRootParameter, lightingShaderCbvRootParameter};
    D3D12_ROOT_SIGNATURE_DESC lightingShaderRootSignatureDesc{};
    lightingShaderRootSignatureDesc.NumParameters = 2;
    lightingShaderRootSignatureDesc.pParameters = lightingShaderRootParameters;
    lightingShaderRootSignatureDesc.NumStaticSamplers = 1;
    lightingShaderRootSignatureDesc.pStaticSamplers = &lightingShaderLinearStaticSampler;
    Microsoft::WRL::ComPtr<ID3DBlob> lightingShaderRootSignatureBlob;
    ThrowIfFailed(D3D12SerializeRootSignature(&lightingShaderRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, lightingShaderRootSignatureBlob.GetAddressOf(), errorBlob.GetAddressOf()));
    ThrowIfFailed(device->CreateRootSignature(0, lightingShaderRootSignatureBlob->GetBufferPointer(), lightingShaderRootSignatureBlob->GetBufferSize(), IID_PPV_ARGS(lightingShaderRootSignature.GetAddressOf())));
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
    geometryPipelineStateDesc.pRootSignature = geometryShaderRootSignagture.Get();
    geometryPipelineStateDesc.VS = {geometryVertexShaderBlob->GetBufferPointer(), geometryVertexShaderBlob->GetBufferSize()};
    geometryPipelineStateDesc.PS = {geometryPixelShaderBlob->GetBufferPointer(), geometryPixelShaderBlob->GetBufferSize()};
    geometryPipelineStateDesc.InputLayout = {geometryInputElementDescs, 2};
    geometryPipelineStateDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    geometryPipelineStateDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    geometryPipelineStateDesc.RTVFormats[1] = DXGI_FORMAT_R10G10B10A2_UNORM;
    geometryPipelineStateDesc.NumRenderTargets = 2;
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
    ThrowIfFailed(device->CreateGraphicsPipelineState(&geometryPipelineStateDesc, IID_PPV_ARGS(geometryShaderPipelineState.GetAddressOf())));

    // create lighting pipeline state
    D3D12_GRAPHICS_PIPELINE_STATE_DESC lightingPipelineStateDesc{};
    lightingPipelineStateDesc.pRootSignature = lightingShaderRootSignature.Get();
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
    ThrowIfFailed(device->CreateGraphicsPipelineState(&lightingPipelineStateDesc, IID_PPV_ARGS(lightingShaderPipelineState.GetAddressOf())));

    // set-up a single fence
    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())));
    frameIndex = 1;
    fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // set-up per-frame mvpInfo constant buffers (persistently mapped)
    D3D12_HEAP_PROPERTIES mvpInfoHeapProperties{};
    mvpInfoHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC mvpInfoResouceDesc{};
    mvpInfoResouceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    mvpInfoResouceDesc.Width = (sizeof(MvpInfo) + 255u) & ~255u;
    mvpInfoResouceDesc.Height = 1;
    mvpInfoResouceDesc.DepthOrArraySize = 1;
    mvpInfoResouceDesc.MipLevels = 1;
    mvpInfoResouceDesc.SampleDesc = {1, 0};
    mvpInfoResouceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    for (UINT i = 0; i < numFrames; ++i)
    {
        ThrowIfFailed(device->CreateCommittedResource(&mvpInfoHeapProperties, D3D12_HEAP_FLAG_NONE, &mvpInfoResouceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(mvpInfoBuffers[i].GetAddressOf())));
        ThrowIfFailed(mvpInfoBuffers[i]->Map(0, nullptr, &(mvpInfoBuffersMappedMemory[i])));
    }

    // set-up per-frame lightInfo constant buffers (persistently mapped)
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
    for (UINT i = 0; i < numFrames; ++i)
    {
        ThrowIfFailed(device->CreateCommittedResource(&lightInfoHeapProperties, D3D12_HEAP_FLAG_NONE, &lightInfoResourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(lightInfoBuffers[i].GetAddressOf())));
        ThrowIfFailed(lightInfoBuffers[i]->Map(0, nullptr, &(lightInfoBuffersMappedMemory[i])));
    }
}
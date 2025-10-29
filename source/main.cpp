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
    float position[3];
    float normal[3];
};

struct ConstantBuffer
{
    XMMATRIX model;
    XMMATRIX view;
    XMMATRIX projection;
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
static Microsoft::WRL::ComPtr<ID3D12Resource> depthBuffer;
static Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> dsvHeap;
static D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;
static Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocators[numFrames];
static Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandLists[numFrames];
static Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue;
static Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
static Microsoft::WRL::ComPtr<ID3D12PipelineState> pipelineState;

static Microsoft::WRL::ComPtr<ID3D12Resource> constantBuffers[numFrames];
static void* constantBuffersMappedMemory[numFrames] = {};
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

            const UINT currentBackBufferIndex = swapChain->GetCurrentBackBufferIndex();
            if (fence->GetCompletedValue() < fenceValues[currentBackBufferIndex])  // think fence->GetCompletedValue() == lastFrameNumberSuccessfulyRenderer
            {
                // being here means the last frame the GPU has finished rendering was not the current frame
                // i.e. the current frame is still in the process of being renderered
                ThrowIfFailed(fence->SetEventOnCompletion(fenceValues[currentBackBufferIndex], fenceEvent));
                WaitForSingleObject(fenceEvent, INFINITE);
            }

            // reset command allocator and command list
            ThrowIfFailed(commandAllocators[currentBackBufferIndex]->Reset());
            ThrowIfFailed(commandLists[currentBackBufferIndex]->Reset(commandAllocators[currentBackBufferIndex].Get(), nullptr));

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

            // update frame's constant buffer with transposed matrices
            // HLSL expects matrices in column-major ordering
            ConstantBuffer currentFrameConstantBuffer;
            currentFrameConstantBuffer.model = XMMatrixTranspose(model);
            currentFrameConstantBuffer.view = XMMatrixTranspose(view);
            currentFrameConstantBuffer.projection = XMMatrixTranspose(projection);
            std::memcpy(constantBuffersMappedMemory[currentBackBufferIndex], &currentFrameConstantBuffer, sizeof(currentFrameConstantBuffer));

            // transition to render target
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = swapChainRenderTargets[currentBackBufferIndex].Get();
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            commandLists[currentBackBufferIndex]->ResourceBarrier(1, &barrier);

            // clear to cornflower blue
            float clearColor[] = {0.39f, 0.58f, 0.93f, 1.0f};
            commandLists[currentBackBufferIndex]->ClearRenderTargetView(swapChainRtvHandles[currentBackBufferIndex], clearColor, 0, nullptr);
            commandLists[currentBackBufferIndex]->ClearDepthStencilView(dsvHeap->GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

            // set pipeline and draw
            commandLists[currentBackBufferIndex]->SetPipelineState(pipelineState.Get());
            commandLists[currentBackBufferIndex]->SetGraphicsRootSignature(rootSignature.Get());
            commandLists[currentBackBufferIndex]->SetGraphicsRootConstantBufferView(0, constantBuffers[currentBackBufferIndex]->GetGPUVirtualAddress());
            commandLists[currentBackBufferIndex]->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            commandLists[currentBackBufferIndex]->RSSetViewports(1, &viewport);
            commandLists[currentBackBufferIndex]->RSSetScissorRects(1, &scissorRect);
            commandLists[currentBackBufferIndex]->OMSetRenderTargets(1, &(swapChainRtvHandles[currentBackBufferIndex]), FALSE, &dsvHandle);
            commandLists[currentBackBufferIndex]->IASetVertexBuffers(0, 1, &vertexBufferView);
            commandLists[currentBackBufferIndex]->IASetIndexBuffer(&indexBufferView);
            commandLists[currentBackBufferIndex]->DrawIndexedInstanced(36, 1, 0, 0, 0);

            // transition to present buffer
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
            commandLists[currentBackBufferIndex]->ResourceBarrier(1, &barrier);

            ThrowIfFailed(commandLists[currentBackBufferIndex]->Close());

            // execute
            ID3D12CommandList* currentFrameCommandList[] = {commandLists[currentBackBufferIndex].Get()};
            commandQueue->ExecuteCommandLists(1, currentFrameCommandList);

            // present
            ThrowIfFailed(swapChain->Present(1, 0));

            ++frameIndex;
            ThrowIfFailed(commandQueue->Signal(fence.Get(), frameIndex));
            fenceValues[currentBackBufferIndex] = frameIndex;
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
    D3D12_RESOURCE_DESC depthBufferDesc{};
    depthBufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    depthBufferDesc.Width = windowWidth;
    depthBufferDesc.Height = windowHeight;
    depthBufferDesc.DepthOrArraySize = 1;
    depthBufferDesc.MipLevels = 1;
    depthBufferDesc.Format = DXGI_FORMAT_D32_FLOAT;
    depthBufferDesc.SampleDesc.Count = 1;
    depthBufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_HEAP_PROPERTIES depthBufferHeapProperties{};
    depthBufferHeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_CLEAR_VALUE depthBufferClearValue{};
    depthBufferClearValue.Format = DXGI_FORMAT_D32_FLOAT;
    depthBufferClearValue.DepthStencil.Depth = 1.0f;  // far plane distance
    ThrowIfFailed(device->CreateCommittedResource(&depthBufferHeapProperties, D3D12_HEAP_FLAG_NONE, &depthBufferDesc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &depthBufferClearValue, IID_PPV_ARGS(depthBuffer.GetAddressOf())));
    
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

    // create RTV for depth buffer (DSV)
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc{};
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.NumDescriptors = 1;
    ThrowIfFailed(device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(dsvHeap.GetAddressOf())));
    dsvHandle = dsvHeap->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(depthBuffer.Get(), nullptr, dsvHandle);

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
    constantBuffersResourceDesc.SampleDesc.Count = 1;
    constantBuffersResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    for (UINT i = 0; i < numFrames; ++i)
    {
        ThrowIfFailed(device->CreateCommittedResource(&constantBuffersHeapProperties, D3D12_HEAP_FLAG_NONE, &constantBuffersResourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(constantBuffers[i].GetAddressOf())));
        ThrowIfFailed(constantBuffers[i]->Map(0, nullptr, &(constantBuffersMappedMemory[i])));
    }

    // compile vertex shader
    Microsoft::WRL::ComPtr<ID3DBlob> vertexShaderBlob;
    if (FAILED(D3DCompileFromFile(L"../assets/shaders/cube.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION, 0, vertexShaderBlob.GetAddressOf(), errorBlob.GetAddressOf())))
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
    if (FAILED(D3DCompileFromFile(L"../assets/shaders/cube.hlsl", nullptr, nullptr, "PSMain", "ps_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION, 0, pixelShaderBlob.GetAddressOf(), errorBlob.GetAddressOf())))
    {
        if (errorBlob != nullptr)
        {
            std::cerr << (char*)(errorBlob->GetBufferPointer()) << std::endl;
        }
        throw std::runtime_error{"Failed to compile pixel shader"};
    }
    errorBlob.Reset();

    // create root signature
    D3D12_ROOT_PARAMETER rootParameter{};
    rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParameter.Descriptor.ShaderRegister = 0;  // b0
    rootParameter.Descriptor.RegisterSpace = 0;
    rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc{};
    rootSignatureDesc.NumParameters = 1;
    rootSignatureDesc.pParameters = &rootParameter;
    rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    Microsoft::WRL::ComPtr<ID3DBlob> rootSignatureBlob;
    ThrowIfFailed(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, rootSignatureBlob.GetAddressOf(), errorBlob.GetAddressOf()));
    ThrowIfFailed(device->CreateRootSignature(0, rootSignatureBlob->GetBufferPointer(), rootSignatureBlob->GetBufferSize(), IID_PPV_ARGS(rootSignature.GetAddressOf())));
    errorBlob.Reset();

    // create PSO
    D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rootSignature.Get();
    psoDesc.VS = {vertexShaderBlob->GetBufferPointer(), vertexShaderBlob->GetBufferSize()};
    psoDesc.PS = {pixelShaderBlob->GetBufferPointer(), pixelShaderBlob->GetBufferSize()};
    psoDesc.InputLayout = {inputElementDescs, 2};
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    psoDesc.RasterizerState.DepthClipEnable = TRUE;
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.DepthStencilState.DepthEnable = TRUE;
    psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    psoDesc.DepthStencilState.StencilEnable = FALSE;
    psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc.Count = 1;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    ThrowIfFailed(device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(pipelineState.GetAddressOf())));
}
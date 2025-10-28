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

inline void ThrowIfFailed(HRESULT hr)
{
    if (FAILED(hr))
    {
        throw std::exception{};
    }
}

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

        const int windowWidth = 1280;
        const int windowHeight = 720;

        GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "D3D12 Renderer", nullptr, nullptr);
        if (window == nullptr)
        {
            glfwTerminate();
            
            throw std::runtime_error{"Failed to create window"};
        }

        HWND hWnd = glfwGetWin32Window(window);

        // enable debug layer
        // catches mistakes with detailed error messages
#if defined(_DEBUG)
        Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
        HRESULT hResult = D3D12GetDebugInterface(IID_PPV_ARGS(debugController.GetAddressOf()));
        ThrowIfFailed(hResult);
        if (SUCCEEDED(hResult))
        {
            Microsoft::WRL::ComPtr<ID3D12Debug1> debugController1;
            if (SUCCEEDED(debugController.As(&debugController1)))
            {
                debugController1->SetEnableGPUBasedValidation(TRUE);
            }
            debugController->EnableDebugLayer();
        }
#endif

        // create device
        // represents GPU
        Microsoft::WRL::ComPtr<ID3D12Device> device;
        hResult = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.GetAddressOf()));
        ThrowIfFailed(hResult);

        // compile vertex shader
        Microsoft::WRL::ComPtr<ID3DBlob> vertexShaderBlob;
        Microsoft::WRL::ComPtr<ID3DBlob> errorBlob;
        hResult = D3DCompileFromFile(L"../assets/shaders/cube.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION, 0, vertexShaderBlob.GetAddressOf(), errorBlob.GetAddressOf());
        if (FAILED(hResult))
        {
            if (errorBlob != nullptr)
            {
                std::cerr << (char*)(errorBlob->GetBufferPointer()) << std::endl;
            }
            throw std::runtime_error{"Failed to compile vertex shader"};
        }

        // compile pixel shader
        Microsoft::WRL::ComPtr<ID3DBlob> pixelShaderBlob;
        errorBlob.Reset();
        hResult = D3DCompileFromFile(L"../assets/shaders/cube.hlsl", nullptr, nullptr, "PSMain", "ps_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION, 0, pixelShaderBlob.GetAddressOf(), errorBlob.GetAddressOf());
        if (FAILED(hResult))
        {
            if (errorBlob != nullptr)
            {
                std::cerr << (char*)(errorBlob->GetBufferPointer()) << std::endl;
            }
            throw std::runtime_error{"Failed to compile pixel shader"};
        }

        // create root signature
        // configure root signature with constant buffer parameter
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
        hResult = D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, rootSignatureBlob.GetAddressOf(), errorBlob.GetAddressOf());
        ThrowIfFailed(hResult);
        Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
        hResult = device->CreateRootSignature(0, rootSignatureBlob->GetBufferPointer(), rootSignatureBlob->GetBufferSize(), IID_PPV_ARGS(rootSignature.GetAddressOf()));
        ThrowIfFailed(hResult);

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
        Microsoft::WRL::ComPtr<ID3D12PipelineState> pipelineState;
        hResult = device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(pipelineState.GetAddressOf()));
        ThrowIfFailed(hResult);

        // create command queue
        // represents the GPU's work queue. Commands are submitted here, and the GPU executes them FIFO order.
        D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
        commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue;
        hResult = device->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(commandQueue.GetAddressOf()));
        ThrowIfFailed(hResult);

        // create swap chain
        // represents back buffers. manages the buffers you render to.
        DXGI_SWAP_CHAIN_DESC1 swapChainDesc{};
        swapChainDesc.BufferCount = 3;  // triple buffering
        swapChainDesc.Width = windowWidth;
        swapChainDesc.Height = windowHeight;
        swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        swapChainDesc.SampleDesc.Count = 1;
        Microsoft::WRL::ComPtr<IDXGIFactory2> factory;
        hResult = CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf()));
        ThrowIfFailed(hResult);
        Microsoft::WRL::ComPtr<IDXGISwapChain1> swapChain1;
        hResult = factory->CreateSwapChainForHwnd(commandQueue.Get(), hWnd, &swapChainDesc, nullptr, nullptr, swapChain1.GetAddressOf());
        ThrowIfFailed(hResult);
        Microsoft::WRL::ComPtr<IDXGISwapChain3> swapChain;
        hResult = swapChain1.As(&swapChain);
        ThrowIfFailed(hResult);

        // setup per-frame resources
        // maintain one set of GPU recording state per swapchain buffer so the CPU can keep perparing frame N+1 while the GPU finishes frame N
        // 3 frames in flight for smoother frame pacing
        static constexpr UINT kNumFrames = 3;
        // create command allocator / command list per frame
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocators[kNumFrames];
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandLists[kNumFrames];
        for (UINT i = 0; i < kNumFrames; ++i)
        {
            hResult = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(commandAllocators[i].GetAddressOf()));
            ThrowIfFailed(hResult);
            hResult = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocators[i].Get(), nullptr, IID_PPV_ARGS(commandLists[i].GetAddressOf()));
            ThrowIfFailed(hResult);
            hResult = commandLists[i]->Close();
            ThrowIfFailed(hResult);
        }
        // maintain a single fence + per-frame fence values
        Microsoft::WRL::ComPtr<ID3D12Fence> fence;
        UINT64 fenceValues[kNumFrames] = {};
        UINT64 currentFrameFenceValue = 0;
        HANDLE currentFrameFenceEvent = nullptr;
        hResult = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf()));
        ThrowIfFailed(hResult);
        currentFrameFenceValue = 1;
        currentFrameFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        // maintain per-frame constant buffers (persistently mapped)
        // think of constant buffers as uniform variable that we plan on uploading to the shader
        Microsoft::WRL::ComPtr<ID3D12Resource> constantBuffers[kNumFrames];
        void* constantBuffersMappedMemory[kNumFrames] = {};
        D3D12_HEAP_PROPERTIES constantBufferHeapProps{};
        constantBufferHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC constantBufferResourceDesc{};
        constantBufferResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        constantBufferResourceDesc.Width = (sizeof(ConstantBuffer) + 255) & ~255;
        constantBufferResourceDesc.Height = 1;
        constantBufferResourceDesc.DepthOrArraySize = 1;
        constantBufferResourceDesc.MipLevels = 1;
        constantBufferResourceDesc.SampleDesc.Count = 1;
        constantBufferResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        for (UINT i = 0; i < kNumFrames; ++i)
        {
            hResult = device->CreateCommittedResource(&constantBufferHeapProps, D3D12_HEAP_FLAG_NONE, &constantBufferResourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(constantBuffers[i].GetAddressOf()));
            ThrowIfFailed(hResult);
            hResult = constantBuffers[i]->Map(0, nullptr, &(constantBuffersMappedMemory[i]));
            ThrowIfFailed(hResult);
        }
        // create render target views (RTV)
        // a desciptor is simply metadata that tells the GPU how to interpret a resource
        // a resource is simply raw GPU memory
        // first we create the descriptor heap (an array of descriptors) to hold descriptors to resources
        D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc{};
        rtvHeapDesc.NumDescriptors = kNumFrames;
        rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtvHeap;
        hResult = device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(rtvHeap.GetAddressOf()));
        ThrowIfFailed(hResult);
        // next create and maintain a descriptor for each back buffer
        UINT rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandles[kNumFrames];
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandleStart = rtvHeap->GetCPUDescriptorHandleForHeapStart();
        Microsoft::WRL::ComPtr<ID3D12Resource> renderTargets[kNumFrames];
        for (UINT i = 0; i < kNumFrames; ++i)
        {
            hResult = swapChain->GetBuffer(i, IID_PPV_ARGS(renderTargets[i].GetAddressOf()));
            ThrowIfFailed(hResult);
            rtvHandles[i] = rtvHandleStart;
            rtvHandles[i].ptr += SIZE_T(i) * SIZE_T(rtvDescriptorSize);
            device->CreateRenderTargetView(renderTargets[i].Get(), nullptr, rtvHandles[i]);
        }

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
        Microsoft::WRL::ComPtr<ID3D12Resource> depthBuffer;
        hResult = device->CreateCommittedResource(&depthBufferHeapProperties, D3D12_HEAP_FLAG_NONE, &depthBufferDesc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &depthBufferClearValue, IID_PPV_ARGS(depthBuffer.GetAddressOf()));
        ThrowIfFailed(hResult);
        // create depth stencil view (DSV)
        D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc{};
        dsvHeapDesc.NumDescriptors = 1;
        dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> dsvHeap;
        hResult = device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(dsvHeap.GetAddressOf()));
        ThrowIfFailed(hResult);
        D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = dsvHeap->GetCPUDescriptorHandleForHeapStart();
        device->CreateDepthStencilView(depthBuffer.Get(), nullptr, dsvHandle);

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
        D3D12_HEAP_PROPERTIES heapProps{};
        heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC resourceDesc{};
        resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resourceDesc.Width = sizeof(vertices);
        resourceDesc.Height = 1;
        resourceDesc.DepthOrArraySize = 1;
        resourceDesc.MipLevels = 1;
        resourceDesc.SampleDesc.Count = 1;
        resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Microsoft::WRL::ComPtr<ID3D12Resource> vertexBuffer;
        hResult = device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(vertexBuffer.GetAddressOf()));
        ThrowIfFailed(hResult);
        void* vertexData;
        hResult = vertexBuffer->Map(0, nullptr, &vertexData);
        ThrowIfFailed(hResult);
        std::memcpy(vertexData, vertices, sizeof(vertices));
        vertexBuffer->Unmap(0, nullptr);
        D3D12_VERTEX_BUFFER_VIEW vertexBufferView{};
        vertexBufferView.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
        vertexBufferView.StrideInBytes = sizeof(Vertex);
        vertexBufferView.SizeInBytes = sizeof(vertices);

        // create index buffer
        resourceDesc.Width = sizeof(indices);
        Microsoft::WRL::ComPtr<ID3D12Resource> indexBuffer;
        hResult = device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(indexBuffer.GetAddressOf()));
        ThrowIfFailed(hResult);
        void* indexData;
        hResult = indexBuffer->Map(0, nullptr, &indexData);
        ThrowIfFailed(hResult);
        std::memcpy(indexData, indices, sizeof(indices));
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
                hResult = fence->SetEventOnCompletion(fenceValues[currentBackBufferIndex], currentFrameFenceEvent);
                ThrowIfFailed(hResult);
                WaitForSingleObject(currentFrameFenceEvent, INFINITE);
            }

            // reset command allocator and command list
            hResult = commandAllocators[currentBackBufferIndex]->Reset();
            ThrowIfFailed(hResult);
            hResult = commandLists[currentBackBufferIndex]->Reset(commandAllocators[currentBackBufferIndex].Get(), nullptr);
            ThrowIfFailed(hResult);

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
            barrier.Transition.pResource = renderTargets[currentBackBufferIndex].Get();
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            commandLists[currentBackBufferIndex]->ResourceBarrier(1, &barrier);

            // clear to cornflower blue
            float clearColor[] = {0.39f, 0.58f, 0.93f, 1.0f};
            commandLists[currentBackBufferIndex]->ClearRenderTargetView(rtvHandles[currentBackBufferIndex], clearColor, 0, nullptr);
            commandLists[currentBackBufferIndex]->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

            // set pipeline and draw
            commandLists[currentBackBufferIndex]->SetPipelineState(pipelineState.Get());
            commandLists[currentBackBufferIndex]->SetGraphicsRootSignature(rootSignature.Get());
            commandLists[currentBackBufferIndex]->SetGraphicsRootConstantBufferView(0, constantBuffers[currentBackBufferIndex]->GetGPUVirtualAddress());
            commandLists[currentBackBufferIndex]->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            D3D12_VIEWPORT viewport{0.0f, 0.0f, static_cast<float>(windowWidth), static_cast<float>(windowHeight), 0.0f, 1.0f};
            RECT scissorRect{0, 0, windowWidth, windowHeight};
            commandLists[currentBackBufferIndex]->RSSetViewports(1, &viewport);
            commandLists[currentBackBufferIndex]->RSSetScissorRects(1, &scissorRect);
            commandLists[currentBackBufferIndex]->OMSetRenderTargets(1, &(rtvHandles[currentBackBufferIndex]), FALSE, &dsvHandle);
            commandLists[currentBackBufferIndex]->IASetVertexBuffers(0, 1, &vertexBufferView);
            commandLists[currentBackBufferIndex]->IASetIndexBuffer(&indexBufferView);
            commandLists[currentBackBufferIndex]->DrawIndexedInstanced(36, 1, 0, 0, 0);

            // transition to present buffer
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
            commandLists[currentBackBufferIndex]->ResourceBarrier(1, &barrier);

            hResult = commandLists[currentBackBufferIndex]->Close();
            ThrowIfFailed(hResult);

            // execute
            ID3D12CommandList* currentFrameCommandList[] = {commandLists[currentBackBufferIndex].Get()};
            commandQueue->ExecuteCommandLists(1, currentFrameCommandList);

            // present
            hResult = swapChain->Present(1, 0);
            ThrowIfFailed(hResult);

            // wait for previous frame to finish rendering
            const UINT64 fenceValue = currentFrameFenceValue++;  // think fenceValue = currentFrameNumber
            hResult = commandQueue->Signal(fence.Get(), fenceValue);  // tell GPU to set fence = currentFrameNumber when the GPU is finished rendering the currently rendered frame
            ThrowIfFailed(hResult);
            fenceValues[currentBackBufferIndex] = fenceValue;
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
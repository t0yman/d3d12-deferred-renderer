#include <iostream>
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
        ID3D12Debug* debugController;
        D3D12GetDebugInterface(IID_PPV_ARGS(&debugController));
        debugController->EnableDebugLayer();

        // create device
        // represents GPU
        ID3D12Device* device;
        D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));

        // compile vertex shader
        ID3DBlob* vertexShaderBlob;
        ID3DBlob* errorBlob;
        HRESULT hResult = D3DCompileFromFile(L"../assets/shaders/cube.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION, 0, &vertexShaderBlob, &errorBlob);
        if (FAILED(hResult))
        {
            if (errorBlob != nullptr)
            {
                std::cerr << (char*)(errorBlob->GetBufferPointer()) << std::endl;
            }
            throw std::runtime_error{"Failed to compile vertex shader"};
        }

        // compile pixel shader
        ID3D10Blob* pixelShaderBlob;
        hResult = D3DCompileFromFile(L"../assets/shaders/cube.hlsl", nullptr, nullptr, "PSMain", "ps_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION, 0, &pixelShaderBlob, &errorBlob);
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
        ID3DBlob* rootSignatureBlob;
        D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rootSignatureBlob, &errorBlob);
        ID3D12RootSignature* rootSignature;
        device->CreateRootSignature(0, rootSignatureBlob->GetBufferPointer(), rootSignatureBlob->GetBufferSize(), IID_PPV_ARGS(&rootSignature));

        // create PSO
        D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        };
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
        psoDesc.pRootSignature = rootSignature;
        psoDesc.VS = {vertexShaderBlob->GetBufferPointer(), vertexShaderBlob->GetBufferSize()};
        psoDesc.PS = {pixelShaderBlob->GetBufferPointer(), pixelShaderBlob->GetBufferSize()};
        psoDesc.InputLayout = {inputElementDescs, 2};
        psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
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
        ID3D12PipelineState* pipelineState;
        device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pipelineState));

        // create command queue
        // represents the GPU's work queue. Commands are submitted here, and the GPU executes them FIFO order.
        D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
        commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        ID3D12CommandQueue* commandQueue;
        device->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(&commandQueue));

        // create swap chain
        // represents back buffers. manages the buffers you render to.
        DXGI_SWAP_CHAIN_DESC1 swapChainDesc{};
        swapChainDesc.BufferCount = 2;  // double buffering
        swapChainDesc.Width = windowWidth;
        swapChainDesc.Height = windowHeight;
        swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        swapChainDesc.SampleDesc.Count = 1;
        IDXGIFactory2* factory;
        CreateDXGIFactory1(IID_PPV_ARGS(&factory));
        Microsoft::WRL::ComPtr<IDXGISwapChain1> swapChain1;
        factory->CreateSwapChainForHwnd(commandQueue, hWnd, &swapChainDesc, nullptr, nullptr, &swapChain1);
        Microsoft::WRL::ComPtr<IDXGISwapChain3> swapChain;
        swapChain1.As(&swapChain);

        // create render target views (RTVs)
        // a descriptor is metadata that tells the GPU how to interpret its resources
        // the resource is raw memory
        // first create descriptor heap (array of descriptors) to hold RTVs
        D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc{};
        rtvHeapDesc.NumDescriptors = 2;  // one per back buffer
        rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        ID3D12DescriptorHeap* rtvHeap;
        device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap));
        // next create an RTV for each back buffer
        UINT rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap->GetCPUDescriptorHandleForHeapStart();
        ID3D12Resource* renderTargets[2];
        for (UINT i = 0; i < 2; ++i)
        {
            swapChain->GetBuffer(i, IID_PPV_ARGS(&renderTargets[i]));
            device->CreateRenderTargetView(renderTargets[i], nullptr, rtvHandle);
            rtvHandle.ptr += rtvDescriptorSize;
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
        ID3D12Resource* depthBuffer;
        device->CreateCommittedResource(&depthBufferHeapProperties, D3D12_HEAP_FLAG_NONE, &depthBufferDesc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &depthBufferClearValue, IID_PPV_ARGS(&depthBuffer));
        // create depth stencil view (DSV)
        D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc{};
        dsvHeapDesc.NumDescriptors = 1;
        dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        ID3D12DescriptorHeap* dsvHeap;
        device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&dsvHeap));
        D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = dsvHeap->GetCPUDescriptorHandleForHeapStart();
        device->CreateDepthStencilView(depthBuffer, nullptr, dsvHandle);

        // create command allocator
        // represents memory pool for command lists
        ID3D12CommandAllocator* commandAllocator;
        device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator));

        // create command list
        // represents the queue you record commands to
        ID3D12GraphicsCommandList* commandList;
        device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator, nullptr, IID_PPV_ARGS(&commandList));
        commandList->Close();  // ensure commandList created in recording state by closing it for now

        // create fence
        // represents the number of the current frame being renderered
        // used for CPU/GPU synchronization
        ID3D12Fence* fence;
        device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        UINT64 fenceValue = 0;
        HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

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
        ID3D12Resource* vertexBuffer;
        device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&vertexBuffer));
        void* vertexData;
        vertexBuffer->Map(0, nullptr, &vertexData);
        std::memcpy(vertexData, vertices, sizeof(vertices));
        vertexBuffer->Unmap(0, nullptr);
        D3D12_VERTEX_BUFFER_VIEW vertexBufferView{};
        vertexBufferView.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
        vertexBufferView.StrideInBytes = sizeof(Vertex);
        vertexBufferView.SizeInBytes = sizeof(vertices);

        // create index buffer
        resourceDesc.Width = sizeof(indices);
        ID3D12Resource* indexBuffer;
        device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&indexBuffer));
        void* indexData;
        indexBuffer->Map(0, nullptr, &indexData);
        std::memcpy(indexData, &indices, sizeof(indices));
        indexBuffer->Unmap(0, nullptr);
        D3D12_INDEX_BUFFER_VIEW indexBufferView{};
        indexBufferView.BufferLocation = indexBuffer->GetGPUVirtualAddress();
        indexBufferView.SizeInBytes = sizeof(indices);
        indexBufferView.Format = DXGI_FORMAT_R16_UINT;

        // create constant buffer (think of it as uniform variable we are planning on uploading to the shader)
        D3D12_HEAP_PROPERTIES constantBufferProps{};
        constantBufferProps.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC constantBufferResourceDesc{};
        constantBufferResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        constantBufferResourceDesc.Width = (sizeof(ConstantBuffer) + 255) & ~255;
        constantBufferResourceDesc.Height = 1;
        constantBufferResourceDesc.DepthOrArraySize = 1;
        constantBufferResourceDesc.MipLevels = 1;
        constantBufferResourceDesc.SampleDesc.Count = 1;
        constantBufferResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ID3D12Resource* constantBuffer;
        device->CreateCommittedResource(&constantBufferProps, D3D12_HEAP_FLAG_NONE, &constantBufferResourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&constantBuffer));

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();

            // resent allocator and command list
            commandAllocator->Reset();
            commandList->Reset(commandAllocator, nullptr);

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

            // update constant buffer with transposed matrices
            // HLSL expects matrices in column-major ordering
            ConstantBuffer currentFrameConstantBuffer;
            currentFrameConstantBuffer.model = XMMatrixTranspose(model);
            currentFrameConstantBuffer.view = XMMatrixTranspose(view);
            currentFrameConstantBuffer.projection = XMMatrixTranspose(projection);
            void* constantBufferData;
            constantBuffer->Map(0, nullptr, &constantBufferData);
            std::memcpy(constantBufferData, &currentFrameConstantBuffer, sizeof(currentFrameConstantBuffer));
            constantBuffer->Unmap(0, nullptr);

            // get current back buffer
            UINT frameIndex = swapChain->GetCurrentBackBufferIndex();
            ID3D12Resource* backBuffer = renderTargets[frameIndex];

            // transition to render target
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = backBuffer;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            commandList->ResourceBarrier(1, &barrier);

            // clear to cornflower blue
            D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap->GetCPUDescriptorHandleForHeapStart();
            rtvHandle.ptr += frameIndex * rtvDescriptorSize;
            float clearColor[] = {0.39f, 0.58f, 0.93f, 1.0f};
            commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
            commandList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

            // set pipeline and draw
            commandList->SetPipelineState(pipelineState);
            commandList->SetGraphicsRootSignature(rootSignature);
            commandList->SetGraphicsRootConstantBufferView(0, constantBuffer->GetGPUVirtualAddress());
            commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            D3D12_VIEWPORT viewport{0.0f, 0.0f, static_cast<float>(windowWidth), static_cast<float>(windowHeight), 0.0f, 1.0f};
            RECT scissorRect{0, 0, windowWidth, windowHeight};
            commandList->RSSetViewports(1, &viewport);
            commandList->RSSetScissorRects(1, &scissorRect);
            commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
            commandList->IASetVertexBuffers(0, 1, &vertexBufferView);
            commandList->IASetIndexBuffer(&indexBufferView);
            commandList->DrawIndexedInstanced(36, 1, 0, 0, 0);

            // transition to present buffer
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
            commandList->ResourceBarrier(1, &barrier);

            commandList->Close();

            // execute
            ID3D12CommandList* commandLists[] = {commandList};
            commandQueue->ExecuteCommandLists(1, commandLists);

            // present
            swapChain->Present(1, 0);

            // wait for previous frame to finish rendering
            const UINT64 currentFenceValue = fenceValue;
            commandQueue->Signal(fence, currentFenceValue);  // tell GPU to set a fence value to currentFenceValue upon completion of rendering current frame
            ++fenceValue;  // increment CPU's fenceValue to indicate rendering of next frame
            if (fence->GetCompletedValue() < fenceValue)  // is the GPU still drawing the previous frame?
            {
                fence->SetEventOnCompletion(currentFenceValue, fenceEvent);  // wait for fence to reach currentFenceValue
                WaitForSingleObject(fenceEvent, INFINITE);
            }
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
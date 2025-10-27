#include <iostream>
#include <stdexcept>

#include <d3d12.h>
#include <dxgi1_6.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <DirectXMath.h>

#include <wrl/client.h>

using namespace DirectX;

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

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();

            // reset allocator and command list
            commandAllocator->Reset();
            commandList->Reset(commandAllocator, nullptr);

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
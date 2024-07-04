const init = async () => {
    const canvas = document.getElementById("canvas-container");
    const resizeCanvas = () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    };

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    //  make sure we can initialize WebGPU
    if (!navigator.gpu) {
        console.error("WebGPU cannot be initialized - navigator.gpu not found");
        return null;
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        console.error("WebGPU cannot be initialized - Adapter not found");
        return null;
    }
    const device = await adapter.requestDevice();
    device.lost.then(() => {
        console.error("WebGPU cannot be initialized - Device has been lost");
        return null;
    });

    const context = canvas.getContext("webgpu");
    if (!context) {
        console.error("WebGPU cannot be initialized - Canvas does not support WebGPU");
        return null;
    }

    // configure swap chain
    const devicePixelRatio = window.devicePixelRatio || 1;
    const presentationSize = [
        canvas.clientWidth * devicePixelRatio,
        canvas.clientHeight * devicePixelRatio,
    ];
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    context.configure({
        device,
        format: presentationFormat,
        size: presentationSize,
    });

    // setup vertices
    const vertices = new Float32Array([
        // first triangle
        -1.0, 0.0, -1.0, 1, 0.5, 0.5, 0.5, 1,
         1.0, 0.0, -1.0, 1, 0.5, 0.5, 0.5, 1,

         1.0, 0.0, -1.0, 1, 0.5, 0.5, 0.5, 1,
        -1.0, 0.0, 1.0, 1, 0.5, 0.5, 0.5, 1,

        -1.0, 0.0,  1.0, 1, 0.5, 0.5, 0.5, 1,
        -1.0, 0.0, -1.0, 1, 0.5, 0.5, 0.5, 1,
        // second triangle
        -1.0, 0.0,  1.0, 1, 0.5, 0.5, 0.5, 1,
         1.0, 0.0, -1.0, 1, 0.5, 0.5, 0.5, 1,

         1.0, 0.0, -1.0, 1, 0.5, 0.5, 0.5, 1,
         1.0, 0.0, 1.0, 1, 0.5, 0.5, 0.5, 1,

         1.0, 0.0, 1.0, 1, 0.5, 0.5, 0.5, 1,
        -1.0, 0.0, 1.0, 1, 0.5, 0.5, 0.5, 1,
    ]);

    const vertexBuffer = device.createBuffer({
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(vertexBuffer.getMappedRange()).set(vertices);
    vertexBuffer.unmap();

    const vertexBuffersDescriptors = [
        {
        attributes: [
            {
            shaderLocation: 0,
            offset: 0,
            format: "float32x4",
            },
            {
            shaderLocation: 1,
            offset: 16,
            format: "float32x4",
            },
        ],
        arrayStride: 32,
        stepMode: "vertex",
        },
    ];

    // load shaders
    const shaderModule = device.createShaderModule({
        code: `
          struct Uniforms {
            modelViewProjection : mat4x4<f32>,
          };

          struct VertexOut {
              @builtin(position) position : vec4<f32>,
              @location(0) color : vec4<f32>,
          };

          @group(0) @binding(0) var<uniform> uniforms : Uniforms;

          @vertex
          fn vertex_main(@location(0) position: vec4<f32>,
                         @location(1) color: vec4<f32>) -> VertexOut {
              var output : VertexOut;
              output.position = uniforms.modelViewProjection * position;
              output.color = color;
              return output;
          } 

          @fragment
          fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32> {
              return fragData.color;
          } 
        `,
    });

    // create pipeline layout
    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.VERTEX,
            buffer: {
                type: 'uniform',
            },
            },],
        })],
    });

    // create render pipeline
    const pipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
        module: shaderModule,
        entryPoint: "vertex_main",
        buffers: vertexBuffersDescriptors,
        },
        fragment: {
        module: shaderModule,
        entryPoint: "fragment_main",
        targets: [
            {
            format: presentationFormat,
            },
        ],
        },
        primitive: {
        topology: "line-list",
        },
    });

    const modelMatrix = glMatrix.mat4.create();
    const viewMatrix = glMatrix.mat4.create();
    const projectionMatrix = glMatrix.mat4.create();
    const modelViewProjectionMatrix = glMatrix.mat4.create();

    glMatrix.mat4.lookAt(viewMatrix, [4, 4, 4], [0, 0, 0], [0, 1, 0]);
    glMatrix.mat4.perspective(projectionMatrix, Math.PI / 4, canvas.clientWidth / canvas.clientHeight, 0.1, 100.0);

    const uniformBuffer = device.createBuffer({
        size: modelViewProjectionMatrix.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const uniformBindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{
        binding: 0,
        resource: {
            buffer: uniformBuffer,
        },
        },],
    });

    // create render pass descriptor
    const renderPassDescriptor = {
        colorAttachments: [
        {
            view: undefined, // Assigned later
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            loadOp: "clear",
            storeOp: "store",
        },
        ],
    };

    // define render loop
    function frame() {
        glMatrix.mat4.multiply(modelViewProjectionMatrix, viewMatrix, modelMatrix);
        glMatrix.mat4.multiply(modelViewProjectionMatrix, projectionMatrix, modelViewProjectionMatrix);
        
        device.queue.writeBuffer(uniformBuffer, 0, modelViewProjectionMatrix);

        renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture()
        .createView();
        
        const commandEncoder = device.createCommandEncoder();
        const passEncoder =
        commandEncoder.beginRenderPass(renderPassDescriptor);

        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, uniformBindGroup);
        passEncoder.setVertexBuffer(0, vertexBuffer);
        passEncoder.draw(vertices.length / 8);
        passEncoder.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
};

init();

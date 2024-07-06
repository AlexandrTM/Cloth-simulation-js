function generateVertices(x, y, cellSize, vertices) {
    for (let i = 0; i < x; i++) {
        for (let j = 0; j < y; j++) {
            const posX = i * cellSize;
            const posY = j * cellSize;

            // First triangle
            vertices.push(
                posX           , 0.0, posY           , 1, 0.5, 0.5, 0.5, 1,
                posX + cellSize, 0.0, posY           , 1, 0.5, 0.5, 0.5, 1,
                posX + cellSize, 0.0, posY + cellSize, 1, 0.5, 0.5, 0.5, 1,
            );

            // Second triangle
            vertices.push(
                posX           , 0.0, posY           , 1, 0.5, 0.5, 0.5, 1,
                posX + cellSize, 0.0, posY + cellSize, 1, 0.5, 0.5, 0.5, 1,
                posX           , 0.0, posY + cellSize, 1, 0.5, 0.5, 0.5, 1,
            );
        }
    }
}

function simulateCloth(vertices, clothWidth, clothHeight, gravity, time) {
    for (let i = 0; i <= clothWidth; i++) {
        for (let j = 0; j <= clothHeight; j++) {
            const index = (i * (clothHeight + 1) + j) * 8;

            // fixed cornerns
            if(
            (i === 0 && j === 0) || 
            (i === 0 && j === clothHeight - 1) || 
            (i === clothWidth - 1 && j === 0) || 
            (i === clothWidth - 1 && j === clothHeight - 1)) {
                continue;
            }

            // center vertex sine wave movement
            if (i === Math.floor(clothWidth / 2) && j === Math.floor(clothHeight / 2)) {
                vertices[index + 1] = Math.sin(time) * 0.5;
                continue;
            }

            //vertices[index + 1] += gravity * 0.016; // Apply gravity to the y-coordinate
        }
    }
}

const init = async () => {
    // canvas resizing
    // #region
    const canvas = document.getElementById("canvas-container");
    const resizeCanvas = () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    };

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    // #endregion

    //  make sure we can initialize WebGPU
    // #region
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
    // #endregion

    // configure swap chain
    // #region
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
    // #endregion

    // setup vertices
    const vertices = [];
    const clothWidth = 20;
    const clothHeight = 20;
    const clothCellSize = 0.25;
    // #region

    generateVertices(clothWidth, clothHeight, clothCellSize, vertices);
    const vertexBufferSize = vertices.length * Float32Array.BYTES_PER_ELEMENT;
    //console.log(vertices.length / 8);
    //console.log(vertices.byteLength);

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
    // #endregion

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
        topology: "triangle-list",
        },
    });

    // setup MVP matrix and uniform buffer
    // #region
    const modelMatrix = glMatrix.mat4.create();
    const viewMatrix = glMatrix.mat4.create();
    const projectionMatrix = glMatrix.mat4.create();
    const modelViewProjectionMatrix = glMatrix.mat4.create();

    glMatrix.mat4.lookAt(viewMatrix, 
        [clothHeight * clothCellSize * 2.15, clothHeight * clothCellSize * 1.9, clothHeight * clothCellSize * 2.15], 
        [0, 0, 0], 
        [0, 1, 0]
    );
    glMatrix.mat4.perspective(projectionMatrix, Math.PI / 4, canvas.clientWidth / canvas.clientHeight, 0.1, 10000);

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
    // #endregion

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

    let time = 0;
    const gravity = -9.8;

    // define render loop
    function frame() {
        time += 0.016;

        simulateCloth(vertices, clothWidth, clothHeight, gravity, time);

        glMatrix.mat4.multiply(modelViewProjectionMatrix, viewMatrix, modelMatrix);
        glMatrix.mat4.multiply(modelViewProjectionMatrix, projectionMatrix, modelViewProjectionMatrix);
        
        device.queue.writeBuffer(uniformBuffer, 0, modelViewProjectionMatrix);

        const vertexBuffer = device.createBuffer({
            size: vertexBufferSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(vertexBuffer.getMappedRange()).set(vertices);
            vertexBuffer.unmap();

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

function generateVertices(width, height, cellSize, vertices, indices) {
    for (let i = 0; i < width; i++) {
        for (let j = 0; j < height; j++) {
            const posX = i * cellSize;
            const posZ = j * cellSize;

            vertices.push(posX, 0.0, posZ, 1, 0.5, 0.5, 0.5, 1);

            const bottomLeft  = i * (height) + j;
            const bottomRight = bottomLeft + 1;
            const topLeft     = bottomLeft + (height);
            const topRight    = topLeft + 1;

            // first triangle
            indices.push(bottomLeft, bottomRight, topRight);
            // second triangle
            indices.push(bottomLeft, topRight, topLeft);
        }
    }
}

function simulateCloth(vertices, clothWidth, clothHeight, gravity, time) {
    for (let i = 0; i < clothWidth; i++) {
        for (let j = 0; j < clothHeight; j++) {
            const index = (i * (clothHeight) + j) * 8;

            // fixed cornerns
            if(
            (i === 0              && j === 0              ) || 
            (i === 0              && j === clothHeight - 1) || 
            (i === clothWidth - 1 && j === 0              ) || 
            (i === clothWidth - 1 && j === clothHeight - 1)) {
                continue;
            }

            // center vertex sine wave movement
            if (i === Math.floor(clothWidth / 2) && j === Math.floor(clothHeight / 2)) {
                vertices[index + 1] = Math.sin(time) * 1.0;
                continue;
            }

            //vertices[index + 1] += gravity * 0.016; // Apply gravity to the y-coordinate
        }
    }
}

const init = async () => { 
    const vertices = [];
    const indices = [];
    const clothWidth = 5;
    const clothHeight = 5;
    const clothCellSize = 0.25;

    let time = 0;
    const gravity = -9.8;

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

    // make sure we can initialize WebGPU
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

    // create vertices
    // #region
    generateVertices(clothWidth, clothHeight, clothCellSize, vertices, indices);
    const vertexBufferSize = vertices.length * Float32Array.BYTES_PER_ELEMENT;
    const indexBufferSize = indices.length * Uint32Array.BYTES_PER_ELEMENT;
    //console.log(vertices.length / 8);
    //console.log(indices.length);
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

    const indexBuffer = device.createBuffer({
        size: indexBufferSize,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(indexBuffer.getMappedRange()).set(indices);
    indexBuffer.unmap();
    const vertexBuffer = device.createBuffer({
        size: vertexBufferSize,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    new Float32Array(vertexBuffer.getMappedRange()).set(vertices);
    vertexBuffer.unmap();
    // #endregion 

    // load shaders
    const shaderModule = device.createShaderModule({
        code: `
struct Uniforms {
    modelViewProjection : mat4x4<f32>,
};

struct WireframeSettings {
    width: f32,
    color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>,
};

@group(0) @binding(0) 
var<uniform> uniforms : Uniforms;

@group(0) @binding(1)
var<uniform> wireframeSettings : WireframeSettings;

@vertex
fn vertex_main(@location(0) position: vec4<f32>,
                @location(1) color: vec4<f32>) -> VertexOut {
    var output : VertexOut;
    output.position = uniforms.modelViewProjection * position;
    output.color = color;
    return output;
} 

fn edgeFactor(bary: vec3f) -> f32 {
  let d = fwidth(bary);
  let a3 = smoothstep(vec3f(0.0), d * wireframeSettings.width, bary);
  return min(min(a3.x, a3.y), a3.z);
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32> {
    // Calculate partial derivatives of position
    let ddx = dpdx(fragData.position.xy);
    let ddy = dpdy(fragData.position.xy);
    
    // Calculate the determinant to find edge presence
    let edgeFactor = abs(ddx.x * ddy.y - ddx.y * ddy.x);

    // Edge threshold to determine wireframe
    let edgeThreshold = 1.0; // Adjust this threshold as needed

    if (edgeFactor < edgeThreshold) {
        return wireframeSettings.color;  // Color the wireframe
    } else {
        return fragData.color;  // Color the triangle interior
    }
}
        `,
    });

    // create MVP matrix and uniform buffers
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

    const wireframeSettings = {
        width: 1.0,
        color: [0.0, 0.0, 0.0, 1.0]
    };

    const MVP_uniform_buffer = device.createBuffer({
        size: modelViewProjectionMatrix.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const wireframe_uniform_buffer = device.createBuffer({
        size: Math.ceil(Float32Array.BYTES_PER_ELEMENT * (1 + wireframeSettings.color.length) / 16) * 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // #endregion

    // create pipeline layout
    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: {
                    type: 'uniform',
                },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: {
                    type: 'uniform',
                },
            },
            { 
                binding: 2, 
                visibility: GPUShaderStage.VERTEX, 
                buffer: { 
                    type: 'read-only-storage' 
                } 
            },
            { 
                binding: 3, 
                visibility: GPUShaderStage.FRAGMENT, 
                buffer: { 
                    type: 'read-only-storage' 
                } 
            },
        ],
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

    // create bind group
    const uniformBindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
        {
            binding: 0,
            resource: { buffer: MVP_uniform_buffer },
        },
        {
            binding: 1,
            resource: { buffer: wireframe_uniform_buffer },
        },
        { 
            binding: 2, 
            resource: { buffer: vertexBuffer },
        },
        { 
            binding: 3, 
            resource: { buffer: vertexBuffer },
        },
        ],
    });

    // write wireframeSettings into uniform buffer
    // #region
    // Convert data to typed arrays
    const widthArray = new Float32Array([wireframeSettings.width]);
    const colorArray = new Float32Array(wireframeSettings.color);

    // Combine into a single typed array
    const wireframeData = new Float32Array(Math.ceil(Float32Array.BYTES_PER_ELEMENT * (1 + wireframeSettings.color.length) / 16) * 4);
    wireframeData.set(widthArray, 0);
    wireframeData.set(colorArray, 4);
    
    //console.log(wireframeData.byteLength);
    //console.log(wireframe_uniform_buffer.size);
    device.queue.writeBuffer(wireframe_uniform_buffer, 0, wireframeData);
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

    // define render loop
    function frame() {
        time += 0.016;

        simulateCloth(vertices, clothWidth, clothHeight, gravity, time);

        glMatrix.mat4.multiply(modelViewProjectionMatrix, viewMatrix, modelMatrix);
        glMatrix.mat4.multiply(modelViewProjectionMatrix, projectionMatrix, modelViewProjectionMatrix);
        
        device.queue.writeBuffer(MVP_uniform_buffer, 0, modelViewProjectionMatrix);
        
        // update vertex buffer
        // #region
        const vertexBuffer = device.createBuffer({
            size: vertexBufferSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });
        new Float32Array(vertexBuffer.getMappedRange()).set(vertices);
        vertexBuffer.unmap();
        // #endregion

        renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture().createView();
        
        const commandEncoder = device.createCommandEncoder();
        const passEncoder =
        commandEncoder.beginRenderPass(renderPassDescriptor);

        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, uniformBindGroup);
        passEncoder.setVertexBuffer(0, vertexBuffer);
        passEncoder.setIndexBuffer(indexBuffer, "uint32");
        passEncoder.drawIndexed(indices.length, 1, 0, 0, 0);
        passEncoder.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
};

init();

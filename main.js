function generateVertices(width, length, cellSize, vertices, indices) {
    // vertices
    for (let i = 0; i < width; i++) {
        for (let j = 0; j < length; j++) {
            const posX = i * cellSize;
            const posZ = j * cellSize;

            vertices.push(posX, 0.0, posZ, 1, 0.5, 0.5, 0.5, 1);
        }
    }
    // indices
    for (let i = 0; i < width - 1; i++) {
        for (let j = 0; j < length - 1; j++) {
            const topLeft     = i * length + j;
            const topRight    = topLeft + 1;
            const bottomLeft  = topLeft + length;
            const bottomRight = bottomLeft + 1;

            // First triangle
            indices.push(topLeft, bottomLeft, topRight);

            // Second triangle
            indices.push(topRight, bottomLeft, bottomRight);
        }
    }
    //console.log(Math.max( ...indices ));
}

function generateDistanceConstraints(width, length) {
    const constraints = [];
    const restLength = 1.0; // adjust based on the spacing between vertices

    for (let i = 0; i < width; i++) {
        for (let j = 0; j < length; j++) {
            const index = i * length + j;
            if (i < width - 1) {
                constraints.push({ vertex1: index, vertex2: index + length, restLength });
            }
            if (j < length - 1) {
                constraints.push({ vertex1: index, vertex2: index + 1, restLength });
            }
        }
    }

    return constraints;
}

// function simulateCloth(vertices, clothWidth, clothLength, gravity, time) {
//     for (let i = 0; i < clothWidth; i++) {
//         for (let j = 0; j < clothLength; j++) {
//             const index = (i * (clothLength) + j) * 8;

//             // fixed cornerns
//             if(
//             (i === 0              && j === 0              ) || 
//             (i === 0              && j === clothLength - 1) || 
//             (i === clothWidth - 1 && j === 0              ) || 
//             (i === clothWidth - 1 && j === clothLength - 1)) {
//                 continue;
//             }

//             // center vertex sine wave movement
//             if (i === Math.floor(clothWidth / 2) && j === Math.floor(clothLength / 2)) {
//                 vertices[index + 1] = Math.sin(time * 2) * 8.0;
//                 continue;
//             }

//             //vertices[index + 1] += gravity * 0.016; // Apply gravity to the y-coordinate
//         }
//     }
// }

const init = async () => { 
    const vertices = [];
    const indices = [];
    const clothWidth = 10;
    const clothLength = 10;
    const clothCellSize = 1.0;

    const constraints = generateDistanceConstraints(clothWidth, clothLength);
    let distanceConstraints = [];

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
    generateVertices(clothWidth, clothLength, clothCellSize, vertices, indices);
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
    // #endregion 

    // create vertex and index buffers
    // #region
    const indexBuffer = device.createBuffer({
        size: indexBufferSize,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
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
    @location(1) barycentric: vec3<f32>,
};

@group(0) @binding(0) 
var<uniform> uniforms : Uniforms;

@group(0) @binding(1)
var<uniform> wireframeSettings : WireframeSettings;
@group(0) @binding(2)
var<storage, read> vertexBuffer : array<f32>;
@group(0) @binding(3) 
var<storage, read> indexBuffer: array<u32>;

@vertex
fn vertex_main(@location(0) position: vec4<f32>,
                @location(1) color: vec4<f32>,
                @builtin(vertex_index) vertexIdx: u32) -> VertexOut {
    var output : VertexOut;

    let index = indexBuffer[vertexIdx];
    //let vertexPos = vec4<f32>(vertexBuffer[index * 8u], vertexBuffer[index * 8u + 1u], vertexBuffer[index * 8u + 2u], 1.0);
    output.position = uniforms.modelViewProjection * position;

    // Assign barycentric coordinates based on the vertex index in the triangle
    let vertNdx = vertexIdx % 3u;
    output.barycentric = vec3<f32>(0.0);
    output.barycentric[vertNdx] = 1.0;
    
    output.color = color;
    //output.color = vec4<f32>(output.barycentric, 1.0);

    return output;
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32> {
    //let bary = fragData.barycentric;

    // use smoothstep for anti-aliasing the edges
    //let edgeThreshold = wireframeSettings.width * fwidth(bary);
    //let edgeSmoothFactor = smoothstep(vec3<f32>(0.0), edgeThreshold, bary);

    //let edgeFactor = min(min(edgeSmoothFactor.x, edgeSmoothFactor.y), edgeSmoothFactor.z);

    //if (edgeFactor < 0.1) {
    //    return wireframeSettings.color;  // color the wireframe
    //} else {
    //return vec4<f32>(edgeSmoothFactor, 1.0);
    return fragData.color;  // color the triangle interior
    //}
}
    `
    });

    // create MVP matrix, wireframe and uniform buffers
    // #region
    const modelMatrix = glMatrix.mat4.create();
    const viewMatrix = glMatrix.mat4.create();
    const projectionMatrix = glMatrix.mat4.create();
    const modelViewProjectionMatrix = glMatrix.mat4.create();

    glMatrix.mat4.lookAt(viewMatrix, 
        [clothLength * clothCellSize * 2.15, clothLength * clothCellSize * 1.9, clothLength * clothCellSize * 2.15], 
        [0, 0, 0], 
        [0, 1, 0]
    );
    glMatrix.mat4.perspective(projectionMatrix, Math.PI / 4, canvas.clientWidth / canvas.clientHeight, 0.1, 10000);

    const wireframeSettings = {
        width: 1.0,
        color: [1.0, 0.0, 0.0, 1.0]
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
                buffer: { type: 'uniform' },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: { type: 'uniform' },
            },
            { 
                binding: 2, 
                visibility: GPUShaderStage.VERTEX, 
                buffer: { type: 'read-only-storage' }, 
            },
            { 
                binding: 3, 
                visibility: GPUShaderStage.VERTEX, 
                buffer: { type: 'read-only-storage' },
            },
        ]})]
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
            { format: presentationFormat }, 
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
            resource: { buffer: indexBuffer },
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

    // compute pipeline
    // #region
    // initialize distance constraints
    for (let i = 0; i < constraints.length; i++) {
        const constraint = constraints[i];
        const baseIndex = i * 3;
        distanceConstraints[baseIndex] = constraint.vertex1;
        distanceConstraints[baseIndex + 1] = constraint.vertex2;
        distanceConstraints[baseIndex + 2] = constraint.restLength;
    }

    let distanceConstraintsBuffer = device.createBuffer({
        size: distanceConstraints.length * Float32Array.BYTES_PER_ELEMENT * 3, // each distance constraint has 3 floats (vertex1, vertex2, restLength)
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const computeShaderModule = device.createShaderModule({
        code: `
struct Vertex {
    position : vec3<f32>,
    invMass : f32,
    predictedPosition : vec3<f32>,
};

struct DistanceConstraint {
    vertex1 : u32,
    vertex2 : u32,
    restLength : f32,
};

struct VertexBuffer {
    vertices : array<Vertex>,
};

struct DistanceConstraintsBuffer {
    constraints : array<DistanceConstraint>,
};

@group(0) @binding(0)
var<storage, read_write> vertexBuffer : VertexBuffer;

@group(0) @binding(1)
var<storage, read> distanceConstraintsBuffer : DistanceConstraintsBuffer;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let vertexIndex = global_id.x;

    // Simulate cloth dynamics using PBD

    // 1. Initialize variables
    let vertex = &vertexBuffer.vertices[vertexIndex];
    var newPosition : vec3<f32> = vertex.position;

    // 2. Apply distance constraints
    for (var i = 0u; i < arrayLength(&distanceConstraintsBuffer.constraints); i = i + 1u) {
        let constraint = distanceConstraintsBuffer.constraints[i];
        
        if (vertexIndex == constraint.vertex1 || vertexIndex == constraint.vertex2) {
            // Resolve indices
            var otherIndex = constraint.vertex2;
            if (vertexIndex == constraint.vertex2) {
                otherIndex = constraint.vertex1;
            }

            // Calculate correction based on current positions
            let delta = vertexBuffer.vertices[otherIndex].position - vertex.position;
            let currentDistance = length(delta);
            let correction = delta * (1.0 - constraint.restLength / currentDistance) * 0.5;
            
            newPosition += correction * vertex.invMass;
        }
    }

    // 3. Update predicted position
    vertexBuffer.vertices[vertexIndex].predictedPosition = newPosition;
}
        `
    });

    // Create compute pipeline layout
    const computePipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [ device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage', },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage', },
            },
        ]})]
    });

    const computePipeline = device.createComputePipeline({
        layout: computePipelineLayout,
        compute: {
            module: computeShaderModule,
            entryPoint: 'main',
        },
    });

    const computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: { buffer: vertexBuffer },
            },
            {
                binding: 1,
                resource: { buffer: distanceConstraintsBuffer },
            },
        ],
    });
    // #endregion

    // define render loop
    function frame() {
        time += 0.016;

        //simulateCloth(vertices, clothWidth, clothLength, gravity, time);
        const dispatchSize = Math.ceil(vertices.length / 64); // Adjust workgroup size as needed
        const computeCommandEncoder = device.createCommandEncoder();
        const computePassEncoder = computeCommandEncoder.beginComputePass();
        computePassEncoder.setPipeline(computePipeline);
        computePassEncoder.setBindGroup(0, computeBindGroup);
        computePassEncoder.dispatchWorkgroups(dispatchSize);
        computePassEncoder.end();
        device.queue.submit([computeCommandEncoder.finish()]);

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

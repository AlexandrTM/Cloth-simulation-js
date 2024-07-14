function generateVertices(width, length, cellSize, vertices, indices) {
    // vertices
    for (let i = 0; i < width; i++) {
        for (let j = 0; j < length; j++) {
            const posX = i * cellSize;
            const posZ = j * cellSize;

            // Calculate mass: corner vertices have zero mass
            const isCorner = (
             i === 0         && j === 0         ) || 
            (i === 0         && j === length - 1) ||
            (i === width - 1 && j === 0         ) || 
            (i === width - 1 && j === length - 1);
            const mass = isCorner ? 1.0 : 1.0;

            vertices.push(
                posX, 0.0, posZ,                 // postion
                0.0,                             // *padding*
                i / width / 1.5,                 // r *color*
                0.2,                             // g
                j / length / 1.5,                // b
                1.0,                             // a
                mass,                            // mass
                0.0, 0.0, 0.0,                   // *padding*
                0.0, 0.0, 0.0,                   // force
                0.0,                             // *padding*
                0.0, 0.0, 0.0,                   // velocity
                0.0,                             // *padding*
            );
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
    //console.log(vertices.length);
    //console.log(vertices.length / 20);
}

function generateDistanceConstraints(width, length, vertices) {
    const constraints = [];
    const distanceConstraints = [];

    for (let i = 0; i < width; i++) {
        for (let j = 0; j < length; j++) {
            const index = i * length + j;
            if (i < width - 1) {
                const restLength = calculateDistance(index, index + length, vertices);
                constraints.push({ vertex1: index, vertex2: index + length, restLength });
                //console.log(index, index + length);
                //console.log(restLength);
                //console.log(
                //     vertices[index * 20    ], 
                //     vertices[index * 20 + 1], 
                //     vertices[index * 20 + 2],
                //     vertices[(index + length) * 20    ],
                //     vertices[(index + length) * 20 + 1],
                //     vertices[(index + length) * 20 + 2],);
            }
            if (j < length - 1) {
                const restLength = calculateDistance(index, index + 1, vertices);
                constraints.push({ vertex1: index, vertex2: index + 1, restLength });
                //console.log(index, index + 1);
                //console.log(restLength);
            }

            // Diagonal constraint (bottom-right)
            if (i < width - 1 && j < length - 1) {
                const restLength = calculateDistance(index, index + length + 1, vertices);
                constraints.push({ vertex1: index, vertex2: index + length + 1, restLength });
            }

            // Diagonal constraint (bottom-left)
            if (i < width - 1 && j > 0) {
                const restLength = calculateDistance(index, index + length - 1, vertices);
                constraints.push({ vertex1: index, vertex2: index + length - 1, restLength });
            }
        }
    }
    //console.log(constraints.length);
    
    for (let i = 0; i < constraints.length; i++) {
        const constraint = constraints[i];
        distanceConstraints.push(constraint.vertex1);
        distanceConstraints.push(constraint.vertex2);
        distanceConstraints.push(constraint.restLength);
    }
    //console.log(distanceConstraints.length);

    return distanceConstraints;
}

function calculateDistance(v1, v2, vertices) {
    return Math.sqrt(
        (vertices[v1 * 20    ] - vertices[v2 * 20    ]) ** 2 + 
        (vertices[v1 * 20 + 1] - vertices[v2 * 20 + 1]) ** 2 + 
        (vertices[v1 * 20 + 2] - vertices[v2 * 20 + 2]) ** 2);
}

function simulateClothOnHost(vertices, clothWidth, clothLength, gravity, time) {
    for (let i = 0; i < clothWidth; i++) {
        for (let j = 0; j < clothLength; j++) {
            const index = (i * (clothLength) + j) * 20;

            // fixed cornerns
            if(
            (i === 0              && j === 0              ) || 
            (i === 0              && j === clothLength - 1) || 
            (i === clothWidth - 1 && j === 0              ) || 
            (i === clothWidth - 1 && j === clothLength - 1)) {
                continue;
            }

            // center vertex sine wave movement
            if (i === Math.floor(clothWidth / 2) && j === Math.floor(clothLength / 2)) {
                vertices[index + 1] = Math.sin(time * 2) * 40.0;
                continue;
            }

            //vertices[index + 1] += gravity * time; // Apply gravity to the y-coordinate
        }
    }
}

const init = async () => { 
    const vertices = [];
    const indices = [];
    const clothWidth = 23;
    const clothLength = 23;
    const clothCellSize = 1.0;

    generateVertices(clothWidth, clothLength, clothCellSize, vertices, indices);
    let distanceConstraints = generateDistanceConstraints(clothWidth, clothLength, vertices);
    
    const gravitySettings = {
        gravityEnabled: true,
        gravity: [0.0, -9.8, 0.0],
    };

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

    // create vertex attribute descriptors
    // #region
    const vertexBuffersDescriptors = [
        {
        attributes: [
            { shaderLocation: 0, offset: 0 , format: "float32x3" }, // position
            { shaderLocation: 1, offset: 16, format: "float32x4" }, // color
            { shaderLocation: 2, offset: 32, format: "float32"   }, // mass
            { shaderLocation: 3, offset: 48, format: "float32x3" }, // force
            { shaderLocation: 4, offset: 64, format: "float32x3" }, // velocity
        ],
        arrayStride: 80,
        stepMode: "vertex",
        },
    ];
    // #endregion 

    // create vertex and index buffers
    // #region
    const vertexBufferSize = vertices.length * Float32Array.BYTES_PER_ELEMENT;
    const indexBufferSize = indices.length * Uint32Array.BYTES_PER_ELEMENT;

    const indexBuffer = device.createBuffer({
        size: indexBufferSize,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    new Uint32Array(indexBuffer.getMappedRange()).set(indices);
    indexBuffer.unmap();

    const vertexBuffer = device.createBuffer({
        size: vertexBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(vertexBuffer.getMappedRange()).set(vertices);
    vertexBuffer.unmap();

    // only for simulation on host
    function updateVertexBuffer(device) {
        const vertexBuffer = device.createBuffer({
            size: vertexBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(vertexBuffer.getMappedRange()).set(vertices);
        vertexBuffer.unmap();

        return vertexBuffer;
    }
    // #endregion

    // create MVP matrix, wireframe and uniform buffers
    // #region
    const modelMatrix = glMatrix.mat4.create();
    const viewMatrix = glMatrix.mat4.create();
    const projectionMatrix = glMatrix.mat4.create();
    const modelViewProjectionMatrix = glMatrix.mat4.create();

    glMatrix.mat4.lookAt(viewMatrix, 
        [clothLength * clothCellSize * 2.15, clothLength * clothCellSize * 1.9, clothLength * clothCellSize * 2.15], 
        //[clothLength * clothCellSize * 2.15, 0.0, clothLength * clothCellSize * 2.15], 
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

    function updateMVPMatrix(device) {
        glMatrix.mat4.multiply(modelViewProjectionMatrix, viewMatrix, modelMatrix);
        glMatrix.mat4.multiply(modelViewProjectionMatrix, projectionMatrix, modelViewProjectionMatrix);
        device.queue.writeBuffer(MVP_uniform_buffer, 0, modelViewProjectionMatrix);
    }
    // #endregion
 
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

    // gravity settings and time buffer
    // #region
    const gravitySettingsBuffer = device.createBuffer({
        size: 32, // u32(bool) + 12 padding + vec3<f32> + 4 padding
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(gravitySettingsBuffer, 0, 
        new Uint32Array([gravitySettings.gravityEnabled]));
    device.queue.writeBuffer(gravitySettingsBuffer, 16, 
        new Float32Array(gravitySettings.gravity));

    function updateGravitySettingsBuffer(device) {
        gravitySettings.gravityEnabled = document.getElementById('gravityCheckbox').checked;

        device.queue.writeBuffer(gravitySettingsBuffer, 0, 
            new Uint32Array([gravitySettings.gravityEnabled]));
        device.queue.writeBuffer(gravitySettingsBuffer, 16, 
            new Float32Array(gravitySettings.gravity));
    }

    const timeSinceLaunchBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    function updateTimeBuffer(timeSinceLaunch) {
        device.queue.writeBuffer(timeSinceLaunchBuffer, 0, 
            new Float32Array([timeSinceLaunch]));
    }
    // #endregion

    // initial position
    // #region
    const initialPositions = new Float32Array(clothWidth * clothLength * 3); // assuming 3 floats per position (x, y, z)
    for (let i = 0; i < clothWidth * clothLength; i++) {
        initialPositions[i * 3    ] = vertices[i * 20    ];
        initialPositions[i * 3 + 1] = vertices[i * 20 + 1];
        initialPositions[i * 3 + 2] = vertices[i * 20 + 2];
    }
    //console.log(initialPositions.length);

    const initialPositionsBuffer = device.createBuffer({
        size: initialPositions.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(initialPositionsBuffer, 0, initialPositions);
    //console.log(initialPositionsBuffer.size);
    // #endregion

    // render pipeline
    // #region
    // load shader
    const shaderModule = device.createShaderModule({
        code: `
struct Uniforms {
    modelViewProjection : mat4x4<f32>,
};

struct Vertex {
    position : vec3<f32>,
    color : vec4<f32>,
    mass : f32,
    force : vec3<f32>,
    velocity : vec3<f32>,
};

struct VertexBuffer {
    vertices : array<Vertex>,
};

struct WireframeSettings {
    width : f32,
    color : vec4<f32>,
};

struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) barycentric : vec3<f32>,
};

@group(0) @binding(0) 
var<uniform> uniforms : Uniforms;
@group(0) @binding(1)
var<uniform> wireframeSettings : WireframeSettings;
@group(0) @binding(2)
var<storage, read> vertexBuffer : VertexBuffer;
@group(0) @binding(3) 
var<storage, read> indexBuffer: array<u32>;

@vertex
fn vertex_main( @location(0) position: vec3<f32>,
                @location(1) color: vec4<f32>,
                @location(2) mass: f32,
                @location(3) force : vec3<f32>,
                @location(4) velocity: vec3<f32>,
                @builtin(vertex_index) vertexIdx: u32) -> VertexOut {
    var output : VertexOut;

    let index = indexBuffer[vertexIdx];
    //let vertexPos = vec4<f32>(vertexBuffer[index * 8u], vertexBuffer[index * 8u + 1u], vertexBuffer[index * 8u + 2u], 1.0);
    output.position = uniforms.modelViewProjection * vec4<f32>(position, 1.0);

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
    let bary = fragData.barycentric;

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
            targets: [ { format: presentationFormat } ],
        },
        primitive: {
            topology: "triangle-list",
        },
    });

    // create bind group
    const uniformBindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: { buffer: MVP_uniform_buffer } },
        { binding: 1, resource: { buffer: wireframe_uniform_buffer } },
        { binding: 2, resource: { buffer: vertexBuffer } },
        { binding: 3, resource: { buffer: indexBuffer } },
        ],
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
    // #endregion

    // compute pipeline
    // #region
    // distanceConstraints
    // #region
    const distanceConstraintsBufferSize = distanceConstraints.length * Float32Array.BYTES_PER_ELEMENT;
    const distanceConstraintsBuffer = device.createBuffer({
        size: distanceConstraintsBufferSize, // each distance constraint has 3 floats (vertex1, vertex2, restLength)
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const buffer = new ArrayBuffer(distanceConstraintsBufferSize);
    const dataView = new DataView(buffer);

    let offset = 0;
    //console.log(distanceConstraints.length / 3);
    for (let i = 0; i < distanceConstraints.length / 3; i++) {
        const v1         = distanceConstraints[i * 3    ];
        const v2         = distanceConstraints[i * 3 + 1];
        const restLength = distanceConstraints[i * 3 + 2];

        //console.log(v1);
        //console.log(v2);
        //console.log(restLength);

        dataView.setUint32(offset, v1, true);
        offset += 4;
        dataView.setUint32(offset, v2, true);
        offset += 4;
        dataView.setFloat32(offset, restLength, true);
        offset += 4;
    }
    // Write data to the GPU buffer
    device.queue.writeBuffer(distanceConstraintsBuffer, 0, new Uint8Array(buffer));
    // #endregion

    clothSizeBuffer = device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(clothSizeBuffer, 0, new Uint32Array([clothWidth, clothLength]));

    const computeShaderModule = device.createShaderModule({
        code: `
struct DistanceConstraint {
    @align(4) v1 : u32,
    @align(4) v2 : u32,
    @align(4) restLength : f32,
};

struct InitialPosition {
    x : f32,
    y : f32,
    z : f32,
}

struct GravitySettings { // alignment 16
    @align(16) gravityEnabled: u32,
    @align(16) gravity: vec3<f32>,
};

struct Vertex {
    @align(16) position : vec3<f32>,
    @align(16) color : vec4<f32>,
    @align(16) mass : f32,
    @align(16) force : vec3<f32>,
    @align(16) velocity : vec3<f32>,
};

@group(0) @binding(0)
var<storage, read_write> vertices : array<Vertex>;
@group(0) @binding(1)
var<storage, read> distanceConstraints : array<DistanceConstraint>;
@group(0) @binding(2)
var<uniform> gravitySettings : GravitySettings;
@group(0) @binding(3)
var<uniform> timeSinceLaunch : f32;
@group(0) @binding(4)
var<storage, read> initialPositions : array<InitialPosition>;
@group(0) @binding(5)
var<storage, read> clothSize : vec2<u32>;

fn is_corner_vertex(vertex_index : u32) -> bool {
    return (
        vertex_index == 0u                                 || 
        vertex_index == (clothSize.x - 1u)                 || 
        vertex_index == (clothSize.x) * (clothSize.y - 1u) || 
        vertex_index == (clothSize.x) * (clothSize.y - 1u) + (clothSize.y - 1u));
}

fn is_center_vertex(vertex_index : u32) -> bool {
    let center_vertex_index = clothSize.x * 
                            (u32(ceil(f32(clothSize.y) / 2.0)) - 1u) + 
                            (u32(ceil(f32(clothSize.y) / 2.0)) - 1u);
    return (vertex_index == center_vertex_index);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    var i = global_id.x;
    let numVertices = arrayLength(&vertices);
    let numConstraints = arrayLength(&distanceConstraints);

    let time_step = 0.01;
    let stiffness = 0.8;
    let damping = 0.98;
    let elasticity = 0.0;

    let wind: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

    let amplitude = 5.0;
    let frequency = 3.0;

    if (i < numVertices) {
    //for (var i = 0u; i < numVertices; i = i + 1u) {
        // Retrieve the vertex to update
        var vertex = vertices[i];
        vertex.force = vec3<f32>(0.0, 0.0, 0.0);

        let initialPosition : vec3<f32> = vec3<f32>(
            initialPositions[i].x, 
            initialPositions[i].y, 
            initialPositions[i].z);

        // Apply sinusoidal movement to the center vertex
        if (is_corner_vertex(i)) {}
        else if (is_center_vertex(i)) {
            vertex.position.y = initialPosition.y + amplitude * sin(timeSinceLaunch * frequency);
        }
        else {
            if (gravitySettings.gravityEnabled == 1u) {
                vertex.force += gravitySettings.gravity * vertex.mass;
                vertex.force += wind;
            }
            vertex.velocity = vertex.velocity + (vertex.force / vertex.mass) * time_step;
            vertex.velocity = vertex.velocity * damping;
            vertex.position = vertex.position + vertex.velocity * time_step;
        }
        // Apply wave motion to all non-corner vertices
        // let amplitude = 0.5;
        // let frequency = 1.0;
        // let waveMotionX = amplitude * sin(frequency * timeSinceLaunch + initialPosition.x);
        // let waveMotionY = amplitude * cos(frequency * timeSinceLaunch + initialPosition.y);
        // vertex.position.z = initialPosition.z + waveMotionX + waveMotionY;

        vertices[i] = vertex;
    }
    
    for (var j = 0u; j < 10u; j = j + 1u) {
        for (var i = 0u; i < numConstraints; i = i + 1u) {
            let v1_indx = distanceConstraints[i].v1;
            let v2_indx = distanceConstraints[i].v2;
            // let v1_in_pos : vec3<f32> = vec3<f32>(
            //     initialPositions[v1_indx].x, 
            //     initialPositions[v1_indx].y, 
            //     initialPositions[v1_indx].z);
            // let v2_in_pos : vec3<f32> = vec3<f32>(
            //     initialPositions[v2_indx].x, 
            //     initialPositions[v2_indx].y, 
            //     initialPositions[v2_indx].z);

            let v1 : Vertex = vertices[v1_indx];
            let v2 : Vertex = vertices[v2_indx]; 
            let restLength : f32 = distanceConstraints[i].restLength;

            let currentLength : f32 = distance(v1.position, v2.position);
            let deltaLength : f32 = currentLength - restLength;
            let correction : f32 = (deltaLength / restLength) * 0.5 * stiffness;
            let direction : vec3<f32> = normalize(v2.position - v1.position);
            
            let correctionVector : vec3<f32> = correction * direction;

            // Apply the elasticity force
            let elasticityForce : vec3<f32> = elasticity * correctionVector;
            
            if (!is_corner_vertex(v1_indx) && !is_center_vertex(v1_indx)) {
                vertices[v1_indx].position = v1.position + correctionVector - elasticityForce;
            }
            if (!is_corner_vertex(v2_indx) && !is_center_vertex(v2_indx)) {
                vertices[v2_indx].position = v2.position - correctionVector + elasticityForce;
            }
        }
    }
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
                buffer: { type: 'storage' }
            },
            { 
                binding: 1, 
                visibility: GPUShaderStage.COMPUTE, 
                buffer: { type: 'read-only-storage' }
            },
            { 
                binding: 2, 
                visibility: GPUShaderStage.COMPUTE, 
                buffer: { type: 'uniform' }
            },
            { 
                binding: 3, 
                visibility: GPUShaderStage.COMPUTE, 
                buffer: { type: 'uniform' }
            },
            { 
                binding: 4, 
                visibility: GPUShaderStage.COMPUTE, 
                buffer: { type: 'read-only-storage' }
            },
            { 
                binding: 5, 
                visibility: GPUShaderStage.COMPUTE, 
                buffer: { type: 'read-only-storage' }
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
            { binding: 0, resource: { buffer: vertexBuffer }},
            { binding: 1, resource: { buffer: distanceConstraintsBuffer }},
            { binding: 2, resource: { buffer: gravitySettingsBuffer }},
            { binding: 3, resource: { buffer: timeSinceLaunchBuffer }},
            { binding: 4, resource: { buffer: initialPositionsBuffer }},
            { binding: 5, resource: { buffer: clothSizeBuffer }},
        ],
    });
    // #endregion
    //console.log(vertexBuffer.size);
    //console.log(vertices.length / 20);
    // define render loop
    let currentTime;
    let startTime = performance.now();
    let timeSinceLaunch;

    function frame() {
        currentTime = performance.now();
        timeSinceLaunch = (currentTime - startTime) / 1000.0; // Convert to seconds
        //console.log(gravitySettings.gravityEnabled);

        updateTimeBuffer(timeSinceLaunch);
        updateGravitySettingsBuffer(device);
        updateMVPMatrix(device);
        // only when simulating on host
        //const vertexBuffer = updateVertexBuffer(device);
        //simulateClothOnHost(vertices, clothWidth, clothLength, -9.8, time)

        const vertexCount = vertices.length / 20;
        const dispatchSize = Math.ceil(vertexCount / 64);
        //console.log(vertexCount);
        //console.log(dispatchSize);
        const computeCommandEncoder = device.createCommandEncoder();
        const computePassEncoder = computeCommandEncoder.beginComputePass();
        computePassEncoder.setPipeline(computePipeline);
        computePassEncoder.setBindGroup(0, computeBindGroup);
        computePassEncoder.dispatchWorkgroups(dispatchSize);
        computePassEncoder.end();

        renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture().createView();

        const renderCommandEncoder = device.createCommandEncoder();
        const passEncoder =
        renderCommandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, uniformBindGroup);
        passEncoder.setVertexBuffer(0, vertexBuffer);
        passEncoder.setIndexBuffer(indexBuffer, "uint32");
        passEncoder.drawIndexed(indices.length, 1, 0, 0, 0);
        passEncoder.end();

        const commandBuffers = [
            computeCommandEncoder.finish(),
            renderCommandEncoder.finish(),
        ];
        device.queue.submit(commandBuffers);
        
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
};

init();

struct DistanceConstraint {
    vertex1 : u32,
    vertex2 : u32,
    restLength : f32,
};

struct GravitySettings { // alignment 16
    gravityEnabled: u32,
    gravity: vec3<f32>,
};

struct Vertex {
    position : vec4<f32>,
    color : vec4<f32>,
    invMass : f32,
    predictedPosition : vec4<f32>,
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
@group(0) @binding(2)
var<uniform> gravitySettings : GravitySettings;
@group(0) @binding(3)
var<uniform> timeSinceLaunch : f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let vertexIndex = global_id.x;

    // Simulate cloth dynamics using PBD

    // vertexBuffer.vertices[vertexIndex].position = vec4<f32>(1.0, 1.0, 1.0, 1.0);

    // Apply sinusoidal movement to the center vertex
    if (vertexIndex == 50u) {
        let amplitude = 0.5;
        let frequency = 1.0;

        let displacement = vec3<f32>(
            amplitude * sin(timeSinceLaunch * frequency),
            0.0,
            0.0,
        );

        vertexBuffer.vertices[vertexIndex].position.x += displacement.x;
        // vertexBuffer.vertices[vertexIndex].position.y += displacement.y;
        // vertexBuffer.vertices[vertexIndex].position.z += displacement.z;
    }

    let mu = 0.98;  // damping coefficient
    let d = 1.0;    // DistanceConstraint, rest length between two vertices
    let t = 0.0167; // time step
    let c = 10;     // wave speed

    let f1 = c * c * t * t / (d * d);
    let f2 = 1.0 / (mu * t + 2.0);
    let k1 = (4.0 - 8.0 * f1) * f2;
    let k2 = (mu * t - 2.0) * f2;
    let k3 = 2.0 * f1 * f2;

    // "renderBuffer"     current positions of the vertices
    // "1 - renderBuffer" previous positions of the vertices
    // "vertexBuffer.vertices[vertexIndex].position"        previous positions of the vertices
    // "vertexBuffer.vertices[vertexIndex].predictedPostion" current positions of the vertices

    let renderBuffer = 0;
    let currentPos  : vec4<f32>;
    let previousPos : vec4<f32>;

    // if (renderBuffer == 0) {
    //     currentPos = vertexBuffer.vertices[vertexIndex].predictedPostion;
    //     previousPos = vertexBuffer.vertices[vertexIndex].position;
    // } 
    // if (renderBuffer == 1) {
    //     currentPos = vertexBuffer.vertices[vertexIndex].position;
    //     previousPos = vertexBuffer.vertices[vertexIndex].predictedPostion;
    // }

    for (var j = 1u; j < 9u; j++) {
        for (var i = 1u; i < 9u; i++) {
            vertexBuffer.vertices[10u * j + i].predictedPostion.y = k1 * vertexBuffer.vertices[10u * j + i].predictedPostion.y + k2 * vertexBuffer.vertices[10u * j + i].position.y + k3 * (
                vertexBuffer.vertices[10u * j + i + 1u].predictedPostion.y + 
                vertexBuffer.vertices[10u * j + i - 1u].predictedPostion.y +
                vertexBuffer.vertices[10u * j + i + 10u].predictedPostion.y + 
                vertexBuffer.vertices[10u * j + i - 10u].predictedPostion.y
            );
        }
    }

    // Swap buffers.
    renderBuffer = 1 - renderBuffer;

    // Update predicted position
    vertexBuffer.vertices[vertexIndex].predictedPosition = newPosition;
}


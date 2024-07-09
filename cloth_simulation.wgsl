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
    // Initialize variables
    let vertex = vertexBuffer.vertices[vertexIndex];
    var newPosition : vec4<f32> = vertex.position;

    // Apply sinusoidal movement to the center vertex
    if (vertexIndex == 50u) {
        let amplitude = 0.5;
        let frequency = 1.0;

        let displacement = vec3<f32>(
            amplitude * sin(timeSinceLaunch * frequency),
            0.0,
            0.0,
        );

        newPosition.x += displacement.x;
        // newPosition.y += displacement.y;
        // newPosition.z += displacement.z;
    }

    let mu = 0.98; // damping coefficient
    let d = 1.0; // grid spacing
    let t = 0.0167; // time step
    let c = 1; // wave speed

    let f1 = c * c * t * t / (d * d);
    let f2 = 1.0 / (mu * t + 2.0);
    let k1 = (4.0 - 8.0 * f1) * f2;
    let k2 = (mu * t - 2.0) * f2;
    let k3 = 2.0 * f1 * f2;

    // Swap buffers.
    let renderBuffer = 1 - renderBuffer;

    // "renderBuffer" current position of the vertices
    // "1 - renderBuffer" previous position of the vertices

    for (var j = 1u; j < 9; j++) {
        let crnt = &buffer[renderBuffer][j * 10..];
        let prev = &mut buffer[1 - renderBuffer][j * 10..];

        for (var i = 1u; i < 9; i++) {
            prev[i].z = k1 * crnt[i].z + k2 * prev[i].z + k3 * (
                crnt[i + 1].z + crnt[i - 1].z +
                crnt[i + 10].z + crnt[i - 10].z
            );
        }
    }

    // Update predicted position
    vertexBuffer.vertices[vertexIndex].predictedPosition = newPosition;
}


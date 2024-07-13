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
    mass : f32,
    force : vec3<f32>,
    velocity : vec3<f32>,
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

@compute @workgroup_size(128)
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
}


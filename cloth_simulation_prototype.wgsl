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
    @align(16) position : vec4<f32>,
    @align(16) color : vec4<f32>,
    @align(16)  mass : f32,
    @align(16) force : vec3<f32>,
    @align(16) velocity : vec3<f32>,
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

    // Each workgroup processes multiple vertices
    let numVertices = arrayLength(&vertexBuffer.vertices);

    for (var i = 0u; i < numVertices; i = i + 1u) {
        // Retrieve the vertex to update
        var vertex = vertexBuffer.vertices[i];
        var newPosition : vec4<f32> = vertex.position;

        // Apply sinusoidal movement to the center vertex (example for vertexIndex 50u)
        if (i == 50u) {
            let amplitude = 150.5;
            let frequency = 1.0;

            let displacement = vec3<f32>(
                amplitude * sin(timeSinceLaunch * frequency),
                0.0,
                0.0,
            );

            newPosition.x += displacement.x;
        }

        vertexBuffer.vertices[i].position = newPosition;
    }
}
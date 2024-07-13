struct DistanceConstraint {
    vertex1 : u32,
    vertex2 : u32,
    restLength : f32,
};

struct GravitySettings { // alignment 16
    @align(16) gravityEnabled: u32,
    @align(16) gravity: vec3<f32>,
};

struct Vertex {
    @align(16) position : vec4<f32>,
    @align(16) color : vec4<f32>,
    @align(16) mass : f32,
    @align(16) force : vec3<f32>,
    @align(16) velocity : vec3<f32>,
};

struct DistanceConstraintsBuffer {
    constraints : array<DistanceConstraint>,
};

@group(0) @binding(0)
var<storage, read_write> vertexBuffer : array<Vertex>;
@group(0) @binding(1)
var<storage, read> distanceConstraintsBuffer : DistanceConstraintsBuffer;
@group(0) @binding(2)
var<uniform> gravitySettings : GravitySettings;
@group(0) @binding(3)
var<uniform> timeSinceLaunch : f32;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let numVertices = arrayLength(&vertexBuffer);

    for (var i = 0u; i < numVertices; i = i + 1u) {
        // Retrieve the vertex to update
        var vertex = vertexBuffer[i];
        var newPosition : vec4<f32> = vertex.position;
        // 0 9 90 99 corner vertices
        // Apply sinusoidal movement to the center vertex
        if (i == 0u || i == 9u || i == 90u || i == 99u) {}
        else if (i == 44u) {
            let amplitude = 5.0;
            let frequency = 1.0;

            newPosition.y = amplitude * sin(timeSinceLaunch * frequency);
        }
        else {
        
        }

        vertexBuffer[i].position = newPosition;
    }
}
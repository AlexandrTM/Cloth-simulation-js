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

    // Initialize variables
    let vertex = vertexBuffer.vertices[vertexIndex];
    var newPosition : vec4<f32> = vertex.position;

    // Apply gravity if enabled
    if (gravitySettings.gravityEnabled == 1u && vertex.invMass > 0.0) {
        newPosition += vec4<f32>(gravitySettings.gravity * vertex.invMass, 1.0);
    }

    // Apply distance constraints
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

    // Update predicted position
    vertexBuffer.vertices[vertexIndex].predictedPosition = newPosition;
}
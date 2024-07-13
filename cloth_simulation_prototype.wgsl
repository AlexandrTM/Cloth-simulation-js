struct DistanceConstraint {
    @align(4) v1 : u32,
    @align(4) v2 : u32,
    @align(4) restLength : f32,
};

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
var<storage, read_write> vertexBuffer : array<Vertex>;
@group(0) @binding(1)
var<storage, read> distanceConstraintsBuffer : array<DistanceConstraint>;
@group(0) @binding(2)
var<uniform> gravitySettings : GravitySettings;
@group(0) @binding(3)
var<uniform> timeSinceLaunch : f32;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let numVertices = arrayLength(&vertexBuffer);
    let numConstraints = arrayLength(&distanceConstraintsBuffer);
    let time_step = 0.01;

    for (var i = 0u; i < numVertices; i = i + 1u) {
        // Retrieve the vertex to update
        var vertex = vertexBuffer[i];
        var newPosition : vec3<f32> = vertex.position;
        // 0 9 90 99 corner vertices
        // Apply sinusoidal movement to the center vertex
        if (i == 0u || i == 9u || i == 90u || i == 99u) {}
        else if (i == 44u) {
            let amplitude = 5.0;
            let frequency = 1.0;

            vertex.position.y = amplitude * sin(timeSinceLaunch * frequency);
        }
        else {
            if (gravitySettings.gravityEnabled == 1u) {
                vertex.force = vertex.force + gravitySettings.gravity * vertex.mass;
                vertex.velocity = vertex.velocity + (vertex.force / vertex.mass) * time_step;
                vertex.position = vertex.position + vertex.velocity * time_step;
            }
        }

        vertexBuffer[i] = vertex;
    }

    for (var i = 0u; i < numConstraints; i = i + 1u) {
        let v1 : Vertex = vertexBuffer[distanceConstraintsBuffer[i].v1];
        let v2 : Vertex = vertexBuffer[distanceConstraintsBuffer[i].v2]; 
        let restLength : f32 = distanceConstraintsBuffer[i].restLength;

        let currentLength : f32 = distance(v1.position, v2.position);
        if (currentLength > 1.0){
            vertexBuffer[distanceConstraintsBuffer[i].v1].color = vec4<f32>(1.0);
        }
        let deltaLength : f32 = currentLength - restLength;
        let correction : f32 = (deltaLength / currentLength) * 0.5;
        let direction : vec3<f32> = normalize(v2.position - v1.position);
        
        let correctionVector : vec3<f32> = correction * direction;
        vertexBuffer[distanceConstraintsBuffer[i].v1].position = v1.position - correctionVector;
        vertexBuffer[distanceConstraintsBuffer[i].v2].position = v2.position + correctionVector;
    }
}
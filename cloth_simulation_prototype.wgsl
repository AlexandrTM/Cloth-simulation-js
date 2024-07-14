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
var<storage, read_write> vertexBuffer : array<Vertex>;
@group(0) @binding(1)
var<storage, read> distanceConstraintsBuffer : array<DistanceConstraint>;
@group(0) @binding(2)
var<uniform> gravitySettings : GravitySettings;
@group(0) @binding(3)
var<uniform> timeSinceLaunch : f32;
@group(0) @binding(4)
var<storage, read> initialPositionsBuffer : array<InitialPosition>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let numVertices = arrayLength(&vertexBuffer);
    let numConstraints = arrayLength(&distanceConstraintsBuffer);

    let time_step = 0.001;
    let stiffness = 1.0;
    let damping = 0.01;

    for (var j = 0u; j < 5u; j = j + 1u) {
        for (var i = 0u; i < numVertices; i = i + 1u) {
            // Retrieve the vertex to update
            var vertex = vertexBuffer[i];
            let initialPosition : vec3<f32> = vec3<f32>(
                initialPositionsBuffer[i].x, 
                initialPositionsBuffer[i].y, 
                initialPositionsBuffer[i].z);
            // 0 9 90 99 corner vertices
            // Apply sinusoidal movement to the center vertex
            if (i == 0u || i == 9u || i == 90u || i == 99u) {}
            else if (i == 44u) {
                let amplitude = 4.0;
                let frequency = 3.0;

                vertex.position.y = initialPosition.y + amplitude * sin(timeSinceLaunch * frequency);
            }
            else {
                if (gravitySettings.gravityEnabled == 1u) {
                    vertex.force = vertex.force + gravitySettings.gravity * vertex.mass;
                    vertex.velocity = vertex.velocity + (vertex.force / vertex.mass) * time_step;
                    vertex.position = initialPosition + vertex.velocity * time_step;
                }

                // Apply damping
                vertex.velocity = vertex.velocity * (1.0 - damping);
            }

            vertexBuffer[i] = vertex;
        }

        for (var i = 0u; i < numConstraints; i = i + 1u) {
            let v1_indx = distanceConstraintsBuffer[i].v1;
            let v2_indx = distanceConstraintsBuffer[i].v2;
            let v1_is_corner = (v1_indx == 0u || v1_indx == 9u || v1_indx == 90u || v1_indx == 99u);
            let v2_is_corner = (v2_indx == 0u || v2_indx == 9u || v2_indx == 90u || v2_indx == 99u);
            let v1_in_pos : vec3<f32> = vec3<f32>(
                initialPositionsBuffer[v1_indx].x, 
                initialPositionsBuffer[v1_indx].y, 
                initialPositionsBuffer[v1_indx].z);
            let v2_in_pos : vec3<f32> = vec3<f32>(
                initialPositionsBuffer[v2_indx].x, 
                initialPositionsBuffer[v2_indx].y, 
                initialPositionsBuffer[v2_indx].z);

            let v1 : Vertex = vertexBuffer[v1_indx];
            let v2 : Vertex = vertexBuffer[v2_indx]; 
            let restLength : f32 = distanceConstraintsBuffer[i].restLength;

            let currentLength : f32 = distance(v1.position, v2.position);
            let deltaLength : f32 = currentLength - restLength;
            let correction : f32 = (deltaLength / currentLength) * 0.5 * stiffness;
            let direction : vec3<f32> = normalize(v2.position - v1.position);
            
            let correctionVector : vec3<f32> = correction * direction;
            
            if (!v1_is_corner && (v1_indx != 44u)) {
                vertexBuffer[v1_indx].position = v1.position + correctionVector;
            }
            if (!v2_is_corner && (v2_indx != 44u)) {
                vertexBuffer[v2_indx].position = v2.position - correctionVector;
            }
        }
    }
}
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

fn calculateSpringForce(v1 : Vertex, v2 : Vertex, restLength : f32, stiffness : f32) -> vec3<f32> {
    let currentLength = distance(v1.position, v2.position);
    let deltaLength = currentLength - restLength;
    let direction : vec3<f32> = normalize(v2.position - v1.position);
    let correction : f32 = (deltaLength / currentLength) * 0.5 * stiffness * stiffness;
    let force : vec3<f32> = correction * direction;
    return force;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    var idx = global_id.x;
    let numVertices = arrayLength(&vertices);
    let numConstraints = arrayLength(&distanceConstraints);

    let time_step = 0.01;
    let stiffness = 40.0;
    let damping = 0.98;
    let elasticity = -1.0;

    let wind: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

    let amplitude = 5.0;
    let frequency = 3.0;

    if (idx < numVertices) {
        var vertex = vertices[idx];
        var totalForce : vec3<f32> = vec3<f32>(0.0);

        let initialPosition : vec3<f32> = vec3<f32>(
            initialPositions[idx].x, 
            initialPositions[idx].y, 
            initialPositions[idx].z);

        // Accumulate external forces
        if (is_corner_vertex(idx)) {}
        else if (is_center_vertex(idx)) {
            vertex.position.y = initialPosition.y + amplitude * sin(timeSinceLaunch * frequency);
        }
        else {
            if (gravitySettings.gravityEnabled == 1u) {
                totalForce += gravitySettings.gravity * vertex.mass;
            }
            totalForce += wind;
        }

        // Apply spring forces from distance constraints
        for (var i = 0u; i < numConstraints; i = i + 1u) {
            if (distanceConstraints[i].v1 == idx) {
                let v2 : Vertex = vertices[distanceConstraints[i].v2];
                let springForce : vec3<f32> = calculateSpringForce(vertex, v2, distanceConstraints[i].restLength, stiffness);
                totalForce += springForce;
            } else if (distanceConstraints[i].v2 == idx) {
                let v1 : Vertex = vertices[distanceConstraints[i].v1];
                let springForce : vec3<f32> = calculateSpringForce(vertex, v1, distanceConstraints[i].restLength, stiffness);
                totalForce += springForce;
            }
        }

        // Update velocity and position using Verlet integration
        vertex.velocity += (totalForce / vertex.mass) * time_step;
        vertex.velocity *= damping;
        if (!is_corner_vertex(idx) && !is_center_vertex(idx)) {
            vertex.position += vertex.velocity * time_step;
        }

        vertices[idx] = vertex;
    }
}
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
    let stiffness = 2.0;
    let damping = 0.1;
    let wind: vec3<f32> = vec3<f32>(3.0, 9.5, 0.0);

    let amplitude = 4.0;
    let frequency = 3.0;

    //for (var j = 0u; j < 5u; j = j + 1u) {
        if (i < numVertices) {
        //for (var i = 0u; i < numVertices; i = i + 1u) {
            // Retrieve the vertex to update
            var vertex = vertices[i];
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
                    vertex.force = vertex.force + gravitySettings.gravity * vertex.mass;
                    //vertex.force = vertex.force + wind;
                    vertex.velocity = vertex.velocity + (vertex.force / vertex.mass) * time_step;
                    vertex.position = vertex.position + vertex.velocity * time_step;
                }
                // Apply damping
                //vertex.velocity = vertex.velocity * (1.0 - damping);
            }

            vertices[i] = vertex;
        }

        for (var i = 0u; i < numConstraints; i = i + 1u) {
            let v1_indx = distanceConstraints[i].v1;
            let v2_indx = distanceConstraints[i].v2;
            let v1_in_pos : vec3<f32> = vec3<f32>(
                initialPositions[v1_indx].x, 
                initialPositions[v1_indx].y, 
                initialPositions[v1_indx].z);
            let v2_in_pos : vec3<f32> = vec3<f32>(
                initialPositions[v2_indx].x, 
                initialPositions[v2_indx].y, 
                initialPositions[v2_indx].z);

            let v1 : Vertex = vertices[v1_indx];
            let v2 : Vertex = vertices[v2_indx]; 
            let restLength : f32 = distanceConstraints[i].restLength;

            let currentLength : f32 = distance(v1.position, v2.position);
            let deltaLength : f32 = currentLength - restLength;
            let correction : f32 = (deltaLength / currentLength) * 0.5 * stiffness;
            let direction : vec3<f32> = normalize(v2.position - v1.position);
            
            let correctionVector : vec3<f32> = correction * direction;
            
            if (!is_corner_vertex(v1_indx) && !is_center_vertex(v1_indx)) {
                vertices[v1_indx].position = v1.position + correctionVector;
            }
            if (!is_corner_vertex(v2_indx) && !is_center_vertex(v2_indx)) {
                vertices[v2_indx].position = v2.position - correctionVector;
            }
        }
    //}
}
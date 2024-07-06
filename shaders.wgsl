struct Uniforms {
    modelViewProjection : mat4x4<f32>,
};

struct WireframeSettings {
    width: f32,
    color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>,
};

@group(0) @binding(0) 
var<uniform> uniforms : Uniforms;

@group(0) @binding(1)
var<uniform> wireframeSettings : WireframeSettings;

@vertex
fn vertex_main(@location(0) position: vec4<f32>,
                @location(1) color: vec4<f32>) -> VertexOut {
    var output : VertexOut;
    output.position = uniforms.modelViewProjection * position;
    output.color = color;
    return output;
} 

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32> {
    // Calculate partial derivatives of position
    let ddx = vec3(
        fragData.position.x + 1.0 - fragData.position.x,
        fragData.position.y + 1.0 - fragData.position.y,
        fragData.position.z + 1.0 - fragData.position.z
    );
    
    let ddy = vec3(
        fragData.position.x + 1.0 - fragData.position.x,
        fragData.position.y + 1.0 - fragData.position.y,
        fragData.position.z + 1.0 - fragData.position.z
    );
    // Calculate the length of the gradient to detect edges
    let edgeFactor = length(vec3(ddx.y * ddy.z - ddx.z * ddy.y,
                                 ddx.z * ddy.x - ddx.x * ddy.z,
                                 ddx.x * ddy.y - ddx.y * ddy.x));

    // Edge threshold to determine wireframe
    let edgeThreshold = 0.1; // Adjust this threshold as needed

    if (edgeFactor < edgeThreshold) {
        return wireframeSettings.color;  // Color the wireframe
    } else {
        return fragData.color + wireframeSettings.color;  // Color the triangle interior
    }
}
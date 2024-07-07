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

fn edgeFactor(bary: vec3f) -> f32 {
  let d = fwidth(bary);
  let a3 = smoothstep(vec3f(0.0), d * wireframeSettings.width, bary);
  return min(min(a3.x, a3.y), a3.z);
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32> {
    // Calculate partial derivatives of position
    let ddx = dpdx(fragData.position);   
    let ddy = dpdy(fragData.position);
    // Calculate the length of the gradient to detect edges
    let edgeFactor = length(cross(ddx, ddy));

    // Edge threshold to determine wireframe
    let edgeThreshold = 0.1; // Adjust this threshold as needed

    if (edgeFactor > edgeThreshold) {
        return wireframeSettings.color;  // Color the wireframe
    } else {
        return fragData.color;  // Color the triangle interior
    }
}
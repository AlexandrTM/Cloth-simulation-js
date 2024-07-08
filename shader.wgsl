struct Uniforms {
    modelViewProjection : mat4x4<f32>,
};

struct WireframeSettings {
    width : f32,
    color : vec4<f32>,
};

struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) barycentric : vec3<f32>,
};

@group(0) @binding(0) 
var<uniform> uniforms : Uniforms;

@group(0) @binding(1)
var<uniform> wireframeSettings : WireframeSettings;
@group(0) @binding(2)
var<storage, read> vertexBuffer : array<f32>;
@group(0) @binding(3) 
var<storage, read> indexBuffer: array<u32>;

@vertex
fn vertex_main(@location(0) position: vec4<f32>,
                @location(1) color: vec4<f32>,
                @location(2) invMass: f32,
                @location(3) predictedPosition: vec4<f32>,
                @builtin(vertex_index) vertexIdx: u32) -> VertexOut {
    var output : VertexOut;

    let index = indexBuffer[vertexIdx];
    //let vertexPos = vec4<f32>(vertexBuffer[index * 8u], vertexBuffer[index * 8u + 1u], vertexBuffer[index * 8u + 2u], 1.0);
    output.position = uniforms.modelViewProjection * position;

    // Assign barycentric coordinates based on the vertex index in the triangle
    let vertNdx = vertexIdx % 3u;
    output.barycentric = vec3<f32>(0.0);
    output.barycentric[vertNdx] = 1.0;
    
    output.color = color;
    //output.color = vec4<f32>(output.barycentric, 1.0);

    return output;
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32> {
    let bary = fragData.barycentric;

    // use smoothstep for anti-aliasing the edges
    //let edgeThreshold = wireframeSettings.width * fwidth(bary);
    //let edgeSmoothFactor = smoothstep(vec3<f32>(0.0), edgeThreshold, bary);

    //let edgeFactor = min(min(edgeSmoothFactor.x, edgeSmoothFactor.y), edgeSmoothFactor.z);

    //if (edgeFactor < 0.1) {
    //    return wireframeSettings.color;  // color the wireframe
    //} else {
    //return vec4<f32>(edgeSmoothFactor, 1.0);
    return fragData.color;  // color the triangle interior
    //}
}
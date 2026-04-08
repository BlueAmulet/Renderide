// Linear blend skinning (compute). Bind buffers expected to match layout produced by mesh preprocess.
// Bone palette entries are world_bone * unity_bindpose (inverse bind matrix per bone), built on CPU each frame.
// Positions use M; normals use transpose(inverse(mat3(M))) per bone (inverse-transpose / cotangent rule).

@group(0) @binding(0) var<storage, read> bone_matrices: array<mat4x4<f32>>;
@group(0) @binding(1) var<storage, read> src_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bone_idx: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bone_weights: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> dst_pos: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> src_n: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> dst_n: array<vec4<f32>>;

fn mat3_linear(m: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
}

/// Full `mat3x3` inverse using `dot`/`cross` (WGSL here has no `inverse()` builtin for matrices).
fn mat3_inverse(m: mat3x3<f32>) -> mat3x3<f32> {
    let c0 = m[0];
    let c1 = m[1];
    let c2 = m[2];
    let det = dot(c0, cross(c1, c2));
    if (abs(det) < 1e-12) {
        return mat3x3<f32>(
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(0.0, 0.0, 1.0),
        );
    }
    let inv_det = 1.0 / det;
    let i0 = cross(c1, c2) * inv_det;
    let i1 = cross(c2, c0) * inv_det;
    let i2 = cross(c0, c1) * inv_det;
    return mat3x3<f32>(i0, i1, i2);
}

/// Upper 3×3 inverse transpose for transforming bind-pose normals to world (handles non-uniform scale).
fn normal_matrix(m: mat4x4<f32>) -> mat3x3<f32> {
    let m3 = mat3_linear(m);
    return transpose(mat3_inverse(m3));
}

@compute @workgroup_size(64)
fn skin_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&src_pos);
    if (i >= n) {
        return;
    }
    let p = src_pos[i];
    let idx = bone_idx[i];
    let w = bone_weights[i];
    let p4 = vec4<f32>(p.xyz, 1.0);
    var acc = vec4<f32>(0.0);
    acc += w.x * (bone_matrices[idx.x] * p4);
    acc += w.y * (bone_matrices[idx.y] * p4);
    acc += w.z * (bone_matrices[idx.z] * p4);
    acc += w.w * (bone_matrices[idx.w] * p4);
    let ws = w.x + w.y + w.z + w.w;

    let nb = src_n[i];
    let n_bind = vec3<f32>(nb.xyz);
    var acc_n = vec3<f32>(0.0);
    acc_n += w.x * (normal_matrix(bone_matrices[idx.x]) * n_bind);
    acc_n += w.y * (normal_matrix(bone_matrices[idx.y]) * n_bind);
    acc_n += w.z * (normal_matrix(bone_matrices[idx.z]) * n_bind);
    acc_n += w.w * (normal_matrix(bone_matrices[idx.w]) * n_bind);

    if (ws > 1e-6) {
        dst_pos[i] = vec4<f32>((acc / ws).xyz, p.w);
        let nn = normalize(acc_n / ws);
        dst_n[i] = vec4<f32>(nn, nb.w);
    } else {
        dst_pos[i] = p;
        dst_n[i] = vec4<f32>(normalize(n_bind), nb.w);
    }
}

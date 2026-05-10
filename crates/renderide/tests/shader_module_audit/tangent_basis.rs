//! Shader source audits for mesh tangent-basis invariants.

use super::*;

#[test]
fn mesh_world_tangent_applies_model_transform_parity() -> io::Result<()> {
    let src = fs::read_to_string(manifest_dir().join("shaders/modules/mesh/vertex.wgsl"))?;

    for required in [
        "fn model_handedness(draw: pd::PerDrawUniforms) -> f32",
        "pd::position_stream_is_world_space(draw)",
        "dot(draw.model[0].xyz, cross(draw.model[1].xyz, draw.model[2].xyz))",
        "let tangent_sign = select(1.0, -1.0, t.w < 0.0);",
        "tangent_sign * model_handedness(draw)",
    ] {
        assert!(
            src.contains(required),
            "mesh/vertex.wgsl must contain `{required}`"
        );
    }

    assert!(
        !src.contains("preserved verbatim"),
        "world_tangent docs must not describe raw tangent handedness preservation"
    );

    Ok(())
}

#[test]
fn pbs_tbn_reorthonormalizes_interpolated_frame() -> io::Result<()> {
    let src = fs::read_to_string(manifest_dir().join("shaders/modules/pbs_normal.wgsl"))?;

    for required in [
        "let n = world_n * inverseSqrt(n_len_sq);",
        "let t_ortho = world_t.xyz - n * dot(world_t.xyz, n);",
        "let t = t_ortho * inverseSqrt(t_len_sq);",
        "rmath::safe_normalize(cross(n, t)",
    ] {
        assert!(
            src.contains(required),
            "pbs_normal.wgsl must contain `{required}`"
        );
    }

    for forbidden in [
        "let n = world_n;",
        "let t = world_t.xyz;",
        "let b = cross(n, t) * sign;",
    ] {
        assert!(
            !src.contains(forbidden),
            "pbs_normal.wgsl must not contain `{forbidden}`"
        );
    }

    Ok(())
}

#[test]
fn custom_mesh_tbn_shaders_route_through_shared_parity() -> io::Result<()> {
    let xiexe =
        fs::read_to_string(manifest_dir().join("shaders/modules/xiexe_toon2_surface.wgsl"))?;
    assert!(
        xiexe.contains("let world_tangent = mv::world_tangent(d, tangent);"),
        "Xiexe Toon 2 must use the shared parity-aware world tangent helper"
    );
    assert!(
        !xiexe.contains("tangent.w,"),
        "Xiexe Toon 2 must not preserve raw tangent.w directly"
    );

    let matcap = material_source("matcap.wgsl")?;
    assert!(
        matcap.contains("pnorm::orthonormal_tbn(world_n, mv::world_tangent(d, tangent))"),
        "Matcap must build its mesh TBN through shared parity-aware helpers"
    );
    for forbidden in [
        "let tangent_sign = select",
        "tangent.w < 0.0",
        "cross(world_n, world_t)",
    ] {
        assert!(
            !matcap.contains(forbidden),
            "Matcap must not contain local mesh TBN fragment `{forbidden}`"
        );
    }

    Ok(())
}

#[test]
fn mesh_tangent_handedness_is_not_recomputed_in_material_roots() -> io::Result<()> {
    let allowed = [
        "shaders/modules/mesh/vertex.wgsl",
        "shaders/modules/pbs_normal.wgsl",
        "shaders/passes/compute/mesh_skinning.wgsl",
    ];
    let mut offenders = Vec::new();

    for relative_dir in ["shaders/materials", "shaders/modules", "shaders/passes"] {
        for path in wgsl_files_recursive(relative_dir)? {
            let label = file_label(&path);
            if allowed.contains(&label.as_str()) {
                continue;
            }
            let src = fs::read_to_string(&path)?;
            for forbidden in [
                "let tangent_sign = select",
                "tangent.w < 0.0",
                "t.w < 0.0",
                "cross(world_n, world_t) *",
                "cross(n, t) * sign",
            ] {
                if src.contains(forbidden) {
                    offenders.push(format!("{label} still contains `{forbidden}`"));
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "mesh tangent-basis handedness must stay centralized:\n  {}",
        offenders.join("\n  ")
    );

    Ok(())
}

#[test]
fn skinning_derives_tangent_handedness_from_deformed_bitangent() -> io::Result<()> {
    let src = fs::read_to_string(manifest_dir().join("shaders/passes/compute/mesh_skinning.wgsl"))?;

    for required in [
        "b_bind = cross(n_bind, t_bind) * bind_sign;",
        "acc_b += w.x * (mat3_linear(bone_matrices[bx]) * b_bind);",
        "let bb = safe_normalize(acc_b / ws, b_bind);",
        "dot(cross(nn, tt), bb) < 0.0",
    ] {
        assert!(
            src.contains(required),
            "mesh_skinning.wgsl must contain `{required}`"
        );
    }

    Ok(())
}

#[test]
fn procedural_tangent_frames_are_explicitly_exempt_from_mesh_parity() -> io::Result<()> {
    let ggx = module_source("ggx_prefilter.wgsl")?;
    assert!(
        ggx.contains("fn tangent_to_world(local_dir: vec3<f32>, n: vec3<f32>) -> vec3<f32>")
            && ggx.contains("let tangent = normalize(cross(up, n));")
            && !ggx.contains("renderide::mesh::vertex"),
        "GGX prefilter builds a procedural sampling basis and must not import mesh tangent parity"
    );

    let gtao = fs::read_to_string(manifest_dir().join("shaders/passes/post/gtao_main.wgsl"))?;
    assert!(
        gtao.contains("accepted.x * cross(l, t)")
            && gtao.contains("accepted.y * cross(t, r)")
            && !gtao.contains("renderide::mesh::vertex"),
        "GTAO builds a screen-space procedural basis and must not import mesh tangent parity"
    );

    Ok(())
}

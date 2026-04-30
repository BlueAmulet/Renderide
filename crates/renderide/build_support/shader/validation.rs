//! Naga validation and Renderide shader contract checks.

use naga::back::wgsl::WriterFlags;
use naga::valid::{Capabilities, ValidationFlags, Validator};

use super::directives::BuildPassDirective;
use super::error::BuildError;

/// Checks that `module` declares the entry points required by `passes`.
pub(super) fn validate_entry_points(
    module: &naga::Module,
    label: &str,
    passes: &[BuildPassDirective],
) -> Result<(), BuildError> {
    if passes.is_empty() {
        let has_compute = module
            .entry_points
            .iter()
            .any(|e| e.stage == naga::ShaderStage::Compute);
        if has_compute {
            return Ok(());
        }
        let has_vs = module
            .entry_points
            .iter()
            .any(|e| e.stage == naga::ShaderStage::Vertex && e.name == "vs_main");
        let has_any_fs = module
            .entry_points
            .iter()
            .any(|e| e.stage == naga::ShaderStage::Fragment);
        if !has_vs || !has_any_fs {
            return Err(BuildError::Message(format!(
                "{label}: expected a vs_main vertex entry point and at least one @fragment \
                 entry point (vertex={has_vs} fragment={has_any_fs})",
            )));
        }
        return Ok(());
    }
    for pass in passes {
        let has_vs = module
            .entry_points
            .iter()
            .any(|e| e.stage == naga::ShaderStage::Vertex && e.name == pass.vertex_entry.as_str());
        let has_fs = module.entry_points.iter().any(|e| {
            e.stage == naga::ShaderStage::Fragment && e.name == pass.fragment_entry.as_str()
        });
        if !has_vs || !has_fs {
            return Err(BuildError::Message(format!(
                "{label}: pass `{:?}` expected entry points {} and {} (vertex={has_vs} fragment={has_fs})",
                pass.kind, pass.vertex_entry, pass.fragment_entry
            )));
        }
    }
    Ok(())
}

/// Canonical Unity pipeline-state property names that must never appear in material uniforms.
const PIPELINE_STATE_PROPERTY_NAMES: &[&str] = &[
    "_SrcBlend",
    "_SrcBlendBase",
    "_SrcBlendAdd",
    "_DstBlend",
    "_DstBlendBase",
    "_DstBlendAdd",
    "_ZWrite",
    "_ZTest",
    "_Cull",
    "_Stencil",
    "_StencilComp",
    "_StencilOp",
    "_StencilFail",
    "_StencilZFail",
    "_StencilReadMask",
    "_StencilWriteMask",
    "_ColorMask",
    "_OffsetFactor",
    "_OffsetUnits",
];

/// Rejects any material whose `@group(1) @binding(0)` uniform contains pipeline-state fields.
pub(super) fn validate_no_pipeline_state_uniform_fields(
    module: &naga::Module,
    label: &str,
) -> Result<(), BuildError> {
    for (_, var) in module.global_variables.iter() {
        let Some(binding) = &var.binding else {
            continue;
        };
        if binding.group != 1 || binding.binding != 0 {
            continue;
        }
        if !matches!(var.space, naga::AddressSpace::Uniform) {
            continue;
        }
        let ty = &module.types[var.ty];
        let naga::TypeInner::Struct { ref members, .. } = ty.inner else {
            continue;
        };
        for member in members {
            let Some(name) = member.name.as_deref() else {
                continue;
            };
            if PIPELINE_STATE_PROPERTY_NAMES.contains(&name) {
                let struct_name = ty.name.as_deref().unwrap_or("<unnamed>");
                return Err(BuildError::Message(format!(
                    "{label}: material uniform struct `{struct_name}` declares pipeline-state \
                     field `{name}` at @group(1) @binding(0). Pipeline-state properties \
                     flow through MaterialBlendMode + MaterialRenderState and are baked into \
                     MaterialPipelineCacheKey; remove the field from the WGSL struct."
                )));
            }
        }
    }
    Ok(())
}

/// Validates a naga module and flattens it back to WGSL.
pub(super) fn module_to_wgsl(module: &naga::Module, label: &str) -> Result<String, BuildError> {
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator
        .validate(module)
        .map_err(|e| BuildError::Message(format!("validate {label}: {e}")))?;
    naga::back::wgsl::write_string(module, &info, WriterFlags::EXPLICIT_TYPES)
        .map_err(|e| BuildError::Message(format!("wgsl out {label}: {e}")))
}

//! Naga validation and Renderide shader contract checks.

use std::collections::BTreeMap;

use naga::back::wgsl::WriterFlags;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{
    Binding, EntryPoint, FunctionArgument, FunctionResult, Handle, Interpolation, Sampling,
    ShaderStage, Type, TypeInner,
};

use super::directives::BuildPassDirective;
use super::error::BuildError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EntryIoSlot {
    ty: Handle<Type>,
    interpolation: Option<Interpolation>,
    sampling: Option<Sampling>,
    blend_src: Option<u32>,
    per_primitive: bool,
}

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
            .any(|e| e.stage == ShaderStage::Compute);
        if has_compute {
            return Ok(());
        }
        let has_vs = module
            .entry_points
            .iter()
            .any(|e| e.stage == ShaderStage::Vertex && e.name == "vs_main");
        let has_any_fs = module
            .entry_points
            .iter()
            .any(|e| e.stage == ShaderStage::Fragment);
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
            .any(|e| e.stage == ShaderStage::Vertex && e.name == pass.vertex_entry.as_str());
        let has_fs = module
            .entry_points
            .iter()
            .any(|e| e.stage == ShaderStage::Fragment && e.name == pass.fragment_entry.as_str());
        if !has_vs || !has_fs {
            return Err(BuildError::Message(format!(
                "{label}: pass `{:?}` expected entry points {} and {} (vertex={has_vs} fragment={has_fs})",
                pass.kind, pass.vertex_entry, pass.fragment_entry
            )));
        }
    }
    Ok(())
}

/// Checks that every declared pass has compatible vertex output and fragment input locations.
pub(super) fn validate_pass_interfaces(
    module: &naga::Module,
    label: &str,
    passes: &[BuildPassDirective],
) -> Result<(), BuildError> {
    for pass in passes {
        let Some(vertex) = find_entry_point(module, ShaderStage::Vertex, &pass.vertex_entry) else {
            return Err(BuildError::Message(format!(
                "{label}: pass `{:?}` missing vertex entry point {}",
                pass.kind, pass.vertex_entry
            )));
        };
        let Some(fragment) = find_entry_point(module, ShaderStage::Fragment, &pass.fragment_entry)
        else {
            return Err(BuildError::Message(format!(
                "{label}: pass `{:?}` missing fragment entry point {}",
                pass.kind, pass.fragment_entry
            )));
        };
        validate_pass_interface_pair(module, label, pass, vertex, fragment)?;
    }
    Ok(())
}

fn find_entry_point<'a>(
    module: &'a naga::Module,
    stage: ShaderStage,
    name: &str,
) -> Option<&'a EntryPoint> {
    module
        .entry_points
        .iter()
        .find(|entry| entry.stage == stage && entry.name == name)
}

fn validate_pass_interface_pair(
    module: &naga::Module,
    label: &str,
    pass: &BuildPassDirective,
    vertex: &EntryPoint,
    fragment: &EntryPoint,
) -> Result<(), BuildError> {
    let vertex_outputs = collect_entry_output_locations(module, vertex, label)?;
    let fragment_inputs = collect_entry_input_locations(module, fragment, label)?;
    for (location, fragment_slot) in fragment_inputs {
        let Some(vertex_slot) = vertex_outputs.get(&location) else {
            return Err(BuildError::Message(format!(
                "{label}: pass `{:?}` fragment entry {} reads @location({location}), \
                 but vertex entry {} does not write it",
                pass.kind, fragment.name, vertex.name
            )));
        };
        if module.types[vertex_slot.ty].inner != module.types[fragment_slot.ty].inner {
            return Err(BuildError::Message(format!(
                "{label}: pass `{:?}` @location({location}) type mismatch between vertex {} ({}) \
                 and fragment {} ({})",
                pass.kind,
                vertex.name,
                type_label(module, vertex_slot.ty),
                fragment.name,
                type_label(module, fragment_slot.ty)
            )));
        }
        if vertex_slot.interpolation != fragment_slot.interpolation
            || vertex_slot.sampling != fragment_slot.sampling
            || vertex_slot.blend_src != fragment_slot.blend_src
            || vertex_slot.per_primitive != fragment_slot.per_primitive
        {
            return Err(BuildError::Message(format!(
                "{label}: pass `{:?}` @location({location}) interpolation mismatch between \
                 vertex {} ({}) and fragment {} ({})",
                pass.kind,
                vertex.name,
                io_slot_label(*vertex_slot),
                fragment.name,
                io_slot_label(fragment_slot)
            )));
        }
    }
    Ok(())
}

fn collect_entry_input_locations(
    module: &naga::Module,
    entry: &EntryPoint,
    label: &str,
) -> Result<BTreeMap<u32, EntryIoSlot>, BuildError> {
    let mut slots = BTreeMap::new();
    let owner = format!("{label}: fragment entry {} input", entry.name);
    for arg in &entry.function.arguments {
        collect_argument_locations(module, arg, &owner, &mut slots)?;
    }
    Ok(slots)
}

fn collect_entry_output_locations(
    module: &naga::Module,
    entry: &EntryPoint,
    label: &str,
) -> Result<BTreeMap<u32, EntryIoSlot>, BuildError> {
    let mut slots = BTreeMap::new();
    let owner = format!("{label}: vertex entry {} output", entry.name);
    if let Some(result) = entry.function.result.as_ref() {
        collect_result_locations(module, result, &owner, &mut slots)?;
    }
    Ok(slots)
}

fn collect_argument_locations(
    module: &naga::Module,
    arg: &FunctionArgument,
    owner: &str,
    slots: &mut BTreeMap<u32, EntryIoSlot>,
) -> Result<(), BuildError> {
    collect_locations(module, arg.ty, arg.binding.as_ref(), owner, slots)
}

fn collect_result_locations(
    module: &naga::Module,
    result: &FunctionResult,
    owner: &str,
    slots: &mut BTreeMap<u32, EntryIoSlot>,
) -> Result<(), BuildError> {
    collect_locations(module, result.ty, result.binding.as_ref(), owner, slots)
}

fn collect_locations(
    module: &naga::Module,
    ty: Handle<Type>,
    binding: Option<&Binding>,
    owner: &str,
    slots: &mut BTreeMap<u32, EntryIoSlot>,
) -> Result<(), BuildError> {
    if let Some(binding) = binding {
        insert_location_slot(owner, ty, binding, slots)?;
        return Ok(());
    }
    if let TypeInner::Struct { members, .. } = &module.types[ty].inner {
        for member in members {
            collect_locations(module, member.ty, member.binding.as_ref(), owner, slots)?;
        }
    }
    Ok(())
}

fn insert_location_slot(
    owner: &str,
    ty: Handle<Type>,
    binding: &Binding,
    slots: &mut BTreeMap<u32, EntryIoSlot>,
) -> Result<(), BuildError> {
    let Binding::Location {
        location,
        interpolation,
        sampling,
        blend_src,
        per_primitive,
    } = binding
    else {
        return Ok(());
    };
    let slot = EntryIoSlot {
        ty,
        interpolation: *interpolation,
        sampling: *sampling,
        blend_src: *blend_src,
        per_primitive: *per_primitive,
    };
    if slots.insert(*location, slot).is_some() {
        return Err(BuildError::Message(format!(
            "{owner} declares duplicate @location({location})"
        )));
    }
    Ok(())
}

fn type_label(module: &naga::Module, ty: Handle<Type>) -> String {
    let ty = &module.types[ty];
    ty.name.clone().unwrap_or_else(|| format!("{:?}", ty.inner))
}

fn io_slot_label(slot: EntryIoSlot) -> String {
    format!(
        "interpolation={:?} sampling={:?} blend_src={:?} per_primitive={}",
        slot.interpolation, slot.sampling, slot.blend_src, slot.per_primitive
    )
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
        let TypeInner::Struct { ref members, .. } = ty.inner else {
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

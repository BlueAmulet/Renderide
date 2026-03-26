//! Blend mode selection for native UI WGSL pipelines (`UI_Unlit`, `UI_TextUnlit`).
//!
//! Unity exposes `_SrcBlend` / `_DstBlend` as floats (see Resonite `UI_Unlit.shader`). FrooxEngine
//! updates these from its `BlendMode` enum via the material writer.

use wgpu::{BlendComponent, BlendFactor, BlendOperation, BlendState};

use super::material_properties::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};
use super::ui_material_contract::{UiTextUnlitPropertyIds, UiUnlitPropertyIds};

/// Surface blend preset for native UI rasterization (wgpu fixed-function blend).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub enum NativeUiSurfaceBlend {
    /// `SrcAlpha` × fragment + `OneMinusSrcAlpha` × destination (standard UI transparency).
    #[default]
    Alpha,
    /// `One` × fragment + `One` × destination (additive glow / bright UI).
    Additive,
}

impl NativeUiSurfaceBlend {
    /// Parses INI / env tokens: `alpha`, `additive`.
    pub fn parse_ini(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "alpha" | "premultiplied" => Some(Self::Alpha),
            "additive" | "add" => Some(Self::Additive),
            _ => None,
        }
    }

    /// Fixed-function blend state for the swapchain color target.
    pub fn to_wgpu_blend_state(self) -> BlendState {
        match self {
            Self::Alpha => BlendState::ALPHA_BLENDING,
            Self::Additive => BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
            },
        }
    }
}

/// Unity `BlendMode`-style source/destination factors as serialized floats (common values).
fn blend_from_src_dst(src: f32, dst: f32) -> Option<NativeUiSurfaceBlend> {
    const EPS: f32 = 0.05;
    let near = |a: f32, b: f32| (a - b).abs() < EPS;
    // SrcAlpha = 5, OneMinusSrcAlpha = 10 (UnityEngine.Rendering.BlendMode).
    if near(src, 5.0) && near(dst, 10.0) {
        return Some(NativeUiSurfaceBlend::Alpha);
    }
    // One = 1 additive.
    if near(src, 1.0) && near(dst, 1.0) {
        return Some(NativeUiSurfaceBlend::Additive);
    }
    None
}

/// Resolves blend for `UI_Unlit` from material-only property lookup (no mesh property block).
pub fn resolve_native_ui_surface_blend_unlit(
    store: &MaterialPropertyStore,
    material_asset_id: i32,
    ids: &UiUnlitPropertyIds,
    default_blend: NativeUiSurfaceBlend,
) -> NativeUiSurfaceBlend {
    let lookup = MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: None,
    };
    let src = read_blend_float(store, lookup, ids.src_blend);
    let dst = read_blend_float(store, lookup, ids.dst_blend);
    match (src, dst) {
        (Some(s), Some(d)) => blend_from_src_dst(s, d).unwrap_or(default_blend),
        _ => default_blend,
    }
}

/// Resolves blend for `UI_TextUnlit` from material-only property lookup.
pub fn resolve_native_ui_surface_blend_text(
    store: &MaterialPropertyStore,
    material_asset_id: i32,
    ids: &UiTextUnlitPropertyIds,
    default_blend: NativeUiSurfaceBlend,
) -> NativeUiSurfaceBlend {
    let lookup = MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: None,
    };
    let src = read_blend_float(store, lookup, ids.src_blend);
    let dst = read_blend_float(store, lookup, ids.dst_blend);
    match (src, dst) {
        (Some(s), Some(d)) => blend_from_src_dst(s, d).unwrap_or(default_blend),
        _ => default_blend,
    }
}

fn read_blend_float(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    pid: i32,
) -> Option<f32> {
    if pid < 0 {
        return None;
    }
    match store.get_merged(lookup, pid) {
        Some(MaterialPropertyValue::Float(f)) => Some(*f),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{NativeUiSurfaceBlend, blend_from_src_dst};

    #[test]
    fn maps_unity_alpha_blend_pair() {
        assert_eq!(
            blend_from_src_dst(5.0, 10.0),
            Some(NativeUiSurfaceBlend::Alpha)
        );
    }

    #[test]
    fn maps_additive_one_one() {
        assert_eq!(
            blend_from_src_dst(1.0, 1.0),
            Some(NativeUiSurfaceBlend::Additive)
        );
    }
}

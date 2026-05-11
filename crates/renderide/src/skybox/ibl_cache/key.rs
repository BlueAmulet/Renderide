//! IBL bake keys and scalar helper math.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::gpu::GpuLimits;
use crate::skybox::specular::SkyboxIblSource;

/// Compute workgroup edge used by every mip-0 producer and the GGX convolve.
const IBL_WORKGROUP_EDGE: u32 = 8;
/// Base GGX importance sample count for mip 1; doubles per mip up to [`IBL_MAX_SAMPLES`].
const IBL_BASE_SAMPLE_COUNT: u32 = 64;
/// Cap on GGX importance sample count for the highest-roughness mips.
const IBL_MAX_SAMPLES: u32 = 1024;

/// Clamps the configured cube face size against the device texture limit.
pub(crate) fn clamp_face_size(face_size: u32, limits: &GpuLimits) -> u32 {
    face_size.min(limits.max_texture_dimension_2d()).max(1)
}

/// Identity for one IBL bake.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) enum SkyboxIblKey {
    /// Analytic procedural / gradient skybox material identity.
    Analytic {
        /// Active skybox material asset id.
        material_asset_id: i32,
        /// Material property generation; invalidates when host edits material props.
        material_generation: u64,
        /// Stable hash of the shader route stem.
        route_hash: u64,
        /// Destination cube face edge (clamped to device limits).
        face_size: u32,
    },
    /// Host-uploaded cubemap material identity.
    Cubemap {
        /// Skybox material asset id when this source came from a material, or `-1` for direct probe sources.
        material_asset_id: i32,
        /// Material property generation when this source came from a material.
        material_generation: u64,
        /// Stable hash of the shader route stem when this source came from a material.
        route_hash: u64,
        /// Source cubemap asset id.
        asset_id: i32,
        /// Source GPU allocation generation.
        allocation_generation: u64,
        /// Source resident mip count; growth re-bakes once more mips arrive.
        mip_levels_resident: u32,
        /// Source content generation; re-uploading the same mips re-bakes.
        content_generation: u64,
        /// Storage V-flip flag for the source cube.
        storage_v_inverted: bool,
        /// Destination cube face edge.
        face_size: u32,
    },
    /// Host-uploaded equirect Texture2D material identity.
    Equirect {
        /// Skybox material asset id when this source came from a material.
        material_asset_id: i32,
        /// Material property generation when this source came from a material.
        material_generation: u64,
        /// Stable hash of the shader route stem when this source came from a material.
        route_hash: u64,
        /// Source Texture2D asset id.
        asset_id: i32,
        /// Source GPU allocation generation.
        allocation_generation: u64,
        /// Source resident mip count.
        mip_levels_resident: u32,
        /// Source content generation; re-uploading the same mips re-bakes.
        content_generation: u64,
        /// Storage V-flip flag for the source texture.
        storage_v_inverted: bool,
        /// Bit-stable hash of `_FOV` material parameters.
        fov_hash: u64,
        /// Bit-stable hash of `_MainTex_ST` material parameters.
        st_hash: u64,
        /// Destination cube face edge.
        face_size: u32,
    },
    /// Analytic constant-color identity.
    SolidColor {
        /// Renderer-side identity for this color source.
        identity: u64,
        /// Linear RGBA color bit hash.
        color_hash: u64,
        /// Destination cube face edge.
        face_size: u32,
    },
    /// Renderer-captured OnChanges reflection-probe cubemap identity.
    RuntimeCubemap {
        /// Render space that owns the captured probe.
        render_space_id: i32,
        /// Dense reflection-probe renderable index.
        renderable_index: i32,
        /// Monotonic renderer-side capture generation.
        generation: u64,
        /// Source mip count resident on the captured cubemap.
        mip_levels: u32,
        /// Destination cube face edge.
        face_size: u32,
    },
}

impl SkyboxIblKey {
    /// Returns the destination face size for this bake.
    pub(super) fn face_size(&self) -> u32 {
        match *self {
            Self::Analytic { face_size, .. }
            | Self::Cubemap { face_size, .. }
            | Self::Equirect { face_size, .. }
            | Self::SolidColor { face_size, .. }
            | Self::RuntimeCubemap { face_size, .. } => face_size,
        }
    }

    /// Returns a stable renderer-side identity hash for the frame-global binding key.
    #[cfg(test)]
    pub(super) fn source_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

/// Builds a cache key for an active source using an already-clamped destination face size.
pub(crate) fn build_key(source: &SkyboxIblSource, face_size: u32) -> SkyboxIblKey {
    match source {
        SkyboxIblSource::Analytic(src) => SkyboxIblKey::Analytic {
            material_asset_id: src.material_asset_id,
            material_generation: src.material_generation,
            route_hash: src.route_hash,
            face_size,
        },
        SkyboxIblSource::Cubemap(src) => SkyboxIblKey::Cubemap {
            material_asset_id: src.material_asset_id,
            material_generation: src.material_generation,
            route_hash: src.route_hash,
            asset_id: src.asset_id,
            allocation_generation: src.allocation_generation,
            mip_levels_resident: src.mip_levels_resident,
            content_generation: src.content_generation,
            storage_v_inverted: src.storage_v_inverted,
            face_size,
        },
        SkyboxIblSource::Equirect(src) => SkyboxIblKey::Equirect {
            material_asset_id: src.material_asset_id,
            material_generation: src.material_generation,
            route_hash: src.route_hash,
            asset_id: src.asset_id,
            allocation_generation: src.allocation_generation,
            mip_levels_resident: src.mip_levels_resident,
            content_generation: src.content_generation,
            storage_v_inverted: src.storage_v_inverted,
            fov_hash: hash_float4(&src.equirect_fov),
            st_hash: hash_float4(&src.equirect_st),
            face_size,
        },
        SkyboxIblSource::SolidColor(src) => SkyboxIblKey::SolidColor {
            identity: src.identity,
            color_hash: hash_float4(&src.color),
            face_size,
        },
        SkyboxIblSource::RuntimeCubemap(src) => SkyboxIblKey::RuntimeCubemap {
            render_space_id: src.render_space_id,
            renderable_index: src.renderable_index,
            generation: src.generation,
            mip_levels: src.mip_levels,
            face_size,
        },
    }
}

/// Hashes four `f32`s by their bit patterns.
fn hash_float4(values: &[f32; 4]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for v in values {
        v.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Returns the full mip count for a cube face edge.
pub(crate) fn mip_levels_for_edge(edge: u32) -> u32 {
    u32::BITS - edge.max(1).leading_zeros()
}

/// Returns the dispatch group count along one 8x8 compute dimension.
pub(super) fn dispatch_groups(size: u32) -> u32 {
    size.max(1).div_ceil(IBL_WORKGROUP_EDGE)
}

/// Returns a mip edge clamped to one texel.
pub(crate) fn mip_extent(base: u32, mip: u32) -> u32 {
    (base >> mip).max(1)
}

/// Returns the highest source mip LOD available to filtered importance sampling.
pub(super) fn source_max_lod(mip_levels: u32) -> f32 {
    mip_levels.saturating_sub(1) as f32
}

/// Returns the GGX importance sample count for the given convolve mip.
pub(super) fn convolve_sample_count(mip_index: u32) -> u32 {
    if mip_index == 0 {
        return 1;
    }
    let exponent = (mip_index - 1).min(4);
    (IBL_BASE_SAMPLE_COUNT << exponent).min(IBL_MAX_SAMPLES)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: applying the runtime parabolic LOD then the inverse returns the input.
    #[test]
    fn roughness_lod_round_trip() {
        for i in 0..=20u32 {
            let r = i as f32 / 20.0;
            let lod = r * (2.0 - r);
            let r_back = 1.0 - (1.0 - lod).max(0.0).sqrt();
            assert!((r - r_back).abs() < 1e-6, "r={r} r_back={r_back}");
        }
    }

    /// Mip count includes mip 0 through the one-texel mip.
    #[test]
    fn mip_levels_for_edge_includes_tail_mip() {
        assert_eq!(mip_levels_for_edge(1), 1);
        assert_eq!(mip_levels_for_edge(2), 2);
        assert_eq!(mip_levels_for_edge(128), 8);
        assert_eq!(mip_levels_for_edge(256), 9);
    }

    /// Source-LOD clamping exposes every generated source mip to filtered importance sampling.
    #[test]
    fn source_max_lod_tracks_last_generated_mip() {
        assert_eq!(source_max_lod(0), 0.0);
        assert_eq!(source_max_lod(1), 0.0);
        assert_eq!(source_max_lod(8), 7.0);
    }

    /// Per-mip sample count clamps to the documented base/cap envelope.
    #[test]
    fn convolve_sample_count_envelope() {
        assert_eq!(convolve_sample_count(0), 1);
        assert_eq!(convolve_sample_count(1), 64);
        assert_eq!(convolve_sample_count(2), 128);
        assert_eq!(convolve_sample_count(3), 256);
        assert_eq!(convolve_sample_count(4), 512);
        assert_eq!(convolve_sample_count(5), 1024);
        assert_eq!(convolve_sample_count(8), 1024);
    }

    /// Analytic key invariants: identity bits change the source hash.
    #[test]
    fn analytic_key_hash_changes_with_identity_fields() {
        let a = SkyboxIblKey::Analytic {
            material_asset_id: 1,
            material_generation: 2,
            route_hash: 3,
            face_size: 256,
        };
        let b = SkyboxIblKey::Analytic {
            material_asset_id: 1,
            material_generation: 2,
            route_hash: 3,
            face_size: 128,
        };
        let c = SkyboxIblKey::Analytic {
            material_asset_id: 1,
            material_generation: 9,
            route_hash: 3,
            face_size: 256,
        };
        assert_ne!(a.source_hash(), b.source_hash());
        assert_ne!(a.source_hash(), c.source_hash());
    }

    /// Cubemap key invariants: residency growth and face size resize both invalidate.
    #[test]
    fn cubemap_key_invalidates_on_residency_or_face_change() {
        let a = cubemap_key(1, 1, 0, 1, 256);
        let b = cubemap_key(1, 1, 0, 4, 256);
        let c = cubemap_key(1, 1, 0, 1, 128);
        assert_ne!(a, b);
        assert_ne!(a, c);
        let d = cubemap_key(1, 2, 0, 1, 256);
        assert_ne!(a, d);
    }

    /// Cubemap allocation and material identity invalidate same-id sources.
    #[test]
    fn cubemap_key_invalidates_on_allocation_or_material_change() {
        let base = cubemap_key(1, 1, 5, 1, 256);
        let reallocated_same_upload_generation = cubemap_key(2, 1, 5, 1, 256);
        let material_changed = cubemap_key(1, 1, 6, 1, 256);

        assert_ne!(base, reallocated_same_upload_generation);
        assert_ne!(base, material_changed);
    }

    /// Equirect key invariants: FOV / ST hash inputs invalidate the bake.
    #[test]
    fn equirect_key_invalidates_on_param_changes() {
        let base = equirect_key(1, 3, 1, 5, [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]);
        let altered_fov = equirect_key(1, 3, 1, 5, [2.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]);
        let altered_st = equirect_key(1, 3, 1, 5, [1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]);
        assert_ne!(base, altered_fov);
        assert_ne!(base, altered_st);
        let altered_content = equirect_key(1, 3, 2, 5, [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]);
        assert_ne!(base, altered_content);
    }

    /// Equirect allocation and material identity invalidate same-id sources.
    #[test]
    fn equirect_key_invalidates_on_allocation_or_material_change() {
        let base = equirect_key(1, 3, 1, 5, [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]);
        let reallocated_same_upload_generation =
            equirect_key(2, 3, 1, 5, [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]);
        let material_changed = equirect_key(1, 3, 1, 6, [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]);

        assert_ne!(base, reallocated_same_upload_generation);
        assert_ne!(base, material_changed);
    }

    fn cubemap_key(
        allocation_generation: u64,
        content_generation: u64,
        material_generation: u64,
        mip_levels_resident: u32,
        face_size: u32,
    ) -> SkyboxIblKey {
        SkyboxIblKey::Cubemap {
            material_asset_id: 21,
            material_generation,
            route_hash: 99,
            asset_id: 7,
            allocation_generation,
            mip_levels_resident,
            content_generation,
            storage_v_inverted: false,
            face_size,
        }
    }

    fn equirect_key(
        allocation_generation: u64,
        mip_levels_resident: u32,
        content_generation: u64,
        material_generation: u64,
        fov: [f32; 4],
        st: [f32; 4],
    ) -> SkyboxIblKey {
        SkyboxIblKey::Equirect {
            material_asset_id: 21,
            material_generation,
            route_hash: 99,
            asset_id: 9,
            allocation_generation,
            mip_levels_resident,
            content_generation,
            storage_v_inverted: false,
            fov_hash: hash_float4(&fov),
            st_hash: hash_float4(&st),
            face_size: 256,
        }
    }
}

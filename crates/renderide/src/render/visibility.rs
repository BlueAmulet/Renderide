//! CPU frustum visibility for rigid meshes using local [`RenderBoundingBox`] and the same
//! view–projection as [`super::pass::mesh_draw::collect_mesh_draws`].
//!
//! Skinned draws are not culled here: bind-pose bounds do not bound deformed vertices, and the
//! skinned MVP path differs from `view_proj * model_matrix`.

use glam::{Mat4, Vec3, Vec4};
use nalgebra::{Matrix4, Vector3};

use crate::scene::math::matrix_na_to_glam;
use crate::scene::render_transform_to_matrix;
use crate::shared::RenderBoundingBox;

use super::batch::SpaceDrawBatch;
use super::view::ViewParams;

/// Epsilon for homogeneous clip comparisons and behind-camera checks.
const CLIP_EPS: f32 = 1e-5;

/// Maximum absolute half-extent below which uploaded mesh bounds are treated as **untrusted** for
/// frustum culling.
///
/// Hosts may send zero extents when metadata is invalid (see FrooxEngine `Mesh` invalid-bounds
/// handling), which collapses the culled volume to a single point at local origin and causes false
/// negatives when vertices extend away from that pivot.
pub(crate) const DEGENERATE_MESH_BOUNDS_EXTENT_EPS: f32 = 1e-8;

/// Below this max half-extent (world-upload units), a successful frustum cull is logged at trace
/// level as potentially suspicious metadata.
pub(crate) const SUSPICIOUS_MESH_BOUNDS_MAX_EXTENT: f32 = 1e-3;

/// Clamps scale components to avoid degenerate view matrices.
///
/// Matches [`super::pass::mesh_draw`] batch view setup.
fn filter_scale(scale: Vector3<f32>) -> Vector3<f32> {
    const MIN_SCALE: f32 = 1e-8;
    if scale.x.abs() < MIN_SCALE || scale.y.abs() < MIN_SCALE || scale.z.abs() < MIN_SCALE {
        Vector3::new(1.0, 1.0, 1.0)
    } else {
        scale
    }
}

/// Applies handedness fix to the view matrix for coordinate system alignment.
fn apply_view_handedness_fix(view: Mat4) -> Mat4 {
    let z_flip = Mat4::from_scale(Vec3::new(1.0, 1.0, -1.0));
    z_flip * view
}

/// View–projection matrix (`glam`) for a [`SpaceDrawBatch`], matching mesh pass MVP setup.
///
/// Uses the batch’s [`SpaceDrawBatch::view_transform`], optional orthographic overlay override,
/// and the primary `proj` matrix for non-overlay or when no override is present.
pub fn view_proj_glam_for_batch(
    batch: &SpaceDrawBatch,
    proj: &Matrix4<f32>,
    overlay_projection_override: Option<&ViewParams>,
) -> Mat4 {
    let mut vt = batch.view_transform;
    vt.scale = filter_scale(vt.scale);
    let view_mat = apply_view_handedness_fix(render_transform_to_matrix(&vt).inverse());
    let proj_na = batch
        .is_overlay
        .then_some(overlay_projection_override)
        .flatten()
        .map(|v| v.to_projection_matrix())
        .unwrap_or(*proj);
    matrix_na_to_glam(&proj_na) * view_mat
}

/// World-space axis-aligned bounds from a local center/extents box and model matrix.
fn world_aabb_from_local_bounds(
    bounds: &RenderBoundingBox,
    model_matrix: Mat4,
) -> Option<(Vec3, Vec3)> {
    let c = bounds.center;
    let e = bounds.extents;
    if !(c.x.is_finite()
        && c.y.is_finite()
        && c.z.is_finite()
        && e.x.is_finite()
        && e.y.is_finite()
        && e.z.is_finite())
    {
        return None;
    }
    let ex = e.x.abs();
    let ey = e.y.abs();
    let ez = e.z.abs();
    let min_l = Vec3::new(c.x - ex, c.y - ey, c.z - ez);
    let max_l = Vec3::new(c.x + ex, c.y + ey, c.z + ez);

    let mut wmin = Vec3::splat(f32::INFINITY);
    let mut wmax = Vec3::splat(f32::NEG_INFINITY);
    for x in [min_l.x, max_l.x] {
        for y in [min_l.y, max_l.y] {
            for z in [min_l.z, max_l.z] {
                let p = model_matrix.transform_point3(Vec3::new(x, y, z));
                if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
                    return None;
                }
                wmin = wmin.min(p);
                wmax = wmax.max(p);
            }
        }
    }
    Some((wmin, wmax))
}

/// Returns `true` if the world AABB may intersect the view frustum (homogeneous clip volume).
///
/// Uses WebGPU / Vulkan clip rules before perspective divide: `|x| ≤ w`, `|y| ≤ w`, `0 ≤ z ≤ w`.
/// The AABB is culled only when it lies entirely outside one of those half-spaces (tested on all
/// eight corners). Matches reverse-Z projection: visible depth still lies in `[0, w]` in clip space.
fn world_aabb_visible_in_homogeneous_clip(
    view_proj: Mat4,
    world_min: Vec3,
    world_max: Vec3,
) -> bool {
    let xs = [world_min.x, world_max.x];
    let ys = [world_min.y, world_max.y];
    let zs = [world_min.z, world_max.z];

    // Behind: all corners have non-positive w (entire box behind or on eye plane).
    let mut all_w_nonpositive = true;
    for &x in &xs {
        for &y in &ys {
            for &z in &zs {
                let clip = view_proj * Vec4::new(x, y, z, 1.0);
                if clip.w > CLIP_EPS {
                    all_w_nonpositive = false;
                    break;
                }
            }
            if !all_w_nonpositive {
                break;
            }
        }
        if !all_w_nonpositive {
            break;
        }
    }
    if all_w_nonpositive {
        return false;
    }

    // Left: inside iff x + w >= 0
    if all_corners_satisfy(&xs, &ys, &zs, view_proj, |p| p.x + p.w < -CLIP_EPS) {
        return false;
    }
    // Right: inside iff w - x >= 0
    if all_corners_satisfy(&xs, &ys, &zs, view_proj, |p| p.w - p.x < -CLIP_EPS) {
        return false;
    }
    // Bottom: y + w >= 0
    if all_corners_satisfy(&xs, &ys, &zs, view_proj, |p| p.y + p.w < -CLIP_EPS) {
        return false;
    }
    // Top: w - y >= 0
    if all_corners_satisfy(&xs, &ys, &zs, view_proj, |p| p.w - p.y < -CLIP_EPS) {
        return false;
    }
    // Near (Vulkan depth): z >= 0
    if all_corners_satisfy(&xs, &ys, &zs, view_proj, |p| p.z < -CLIP_EPS) {
        return false;
    }
    // Far: z <= w
    if all_corners_satisfy(&xs, &ys, &zs, view_proj, |p| p.z - p.w > CLIP_EPS) {
        return false;
    }

    true
}

/// Returns true if `predicate` holds for every world-space corner of the AABB after `view_proj`.
fn all_corners_satisfy(
    xs: &[f32; 2],
    ys: &[f32; 2],
    zs: &[f32; 2],
    view_proj: Mat4,
    predicate: impl Fn(Vec4) -> bool,
) -> bool {
    for &x in xs {
        for &y in ys {
            for &z in zs {
                let clip = view_proj * Vec4::new(x, y, z, 1.0);
                if !predicate(clip) {
                    return false;
                }
            }
        }
    }
    true
}

/// Returns `true` when uploaded center/extents are non-finite or all half-extents are below
/// [`DEGENERATE_MESH_BOUNDS_EXTENT_EPS`]. In those cases frustum culling must not run: the volume
/// collapses to a point (or is undefined) and is not a reliable proxy for triangle coverage.
pub(crate) fn mesh_bounds_degenerate_for_cull(bounds: &RenderBoundingBox) -> bool {
    let e = bounds.extents;
    if !(e.x.is_finite() && e.y.is_finite() && e.z.is_finite()) {
        return true;
    }
    let m = e.x.abs().max(e.y.abs()).max(e.z.abs());
    m < DEGENERATE_MESH_BOUNDS_EXTENT_EPS
}

/// Largest absolute half-extent along any axis (finite components only); `0` if extents are bad.
pub(crate) fn mesh_bounds_max_half_extent(bounds: &RenderBoundingBox) -> f32 {
    let e = bounds.extents;
    if !(e.x.is_finite() && e.y.is_finite() && e.z.is_finite()) {
        return 0.0;
    }
    e.x.abs().max(e.y.abs()).max(e.z.abs())
}

/// Whether a rigid mesh draw should be submitted: local bounds transformed by `model_matrix`,
/// tested against `view_proj`.
///
/// Returns `true` if the draw should be kept (visible or indeterminate). Returns `true` when bounds
/// are **non-finite** in world space, or when local half-extents are degenerate per
/// [`mesh_bounds_degenerate_for_cull`] (conservative: do not cull).
pub fn rigid_mesh_potentially_visible(
    bounds: &RenderBoundingBox,
    model_matrix: Mat4,
    view_proj: Mat4,
) -> bool {
    if mesh_bounds_degenerate_for_cull(bounds) {
        return true;
    }
    let Some((wmin, wmax)) = world_aabb_from_local_bounds(bounds, model_matrix) else {
        return true;
    };
    world_aabb_visible_in_homogeneous_clip(view_proj, wmin, wmax)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::batch::SpaceDrawBatch;
    use crate::render::pass::reverse_z_projection;
    use crate::scene::render_transform_to_matrix;
    use crate::shared::RenderTransform;
    use nalgebra::{Matrix4 as NaMatrix4, Point3, Quaternion, Vector3 as NaVector3};

    fn perspective_proj(aspect: f32) -> Matrix4<f32> {
        reverse_z_projection(aspect, 60f32.to_radians(), 0.1, 100.0)
    }

    fn look_vp_naive() -> (Matrix4<f32>, Mat4) {
        let view = NaMatrix4::look_at_rh(
            &Point3::new(0.0, 0.0, 5.0),
            &Point3::new(0.0, 0.0, 0.0),
            &NaVector3::new(0.0, 1.0, 0.0),
        );
        let proj = perspective_proj(1.0);
        let vp = matrix_na_to_glam(&(proj * view));
        (proj, vp)
    }

    #[test]
    fn box_in_front_of_camera_visible() {
        let (_proj, vp) = look_vp_naive();
        let bounds = RenderBoundingBox {
            center: nalgebra::Vector3::new(0.0, 0.0, 0.0),
            extents: nalgebra::Vector3::new(0.5, 0.5, 0.5),
        };
        let model = Mat4::IDENTITY;
        assert!(rigid_mesh_potentially_visible(&bounds, model, vp));
    }

    #[test]
    fn box_behind_camera_culled() {
        let (_proj, vp) = look_vp_naive();
        let bounds = RenderBoundingBox {
            center: nalgebra::Vector3::new(0.0, 0.0, 20.0),
            extents: nalgebra::Vector3::new(0.5, 0.5, 0.5),
        };
        let model = Mat4::IDENTITY;
        assert!(!rigid_mesh_potentially_visible(&bounds, model, vp));
    }

    #[test]
    fn box_far_left_outside_frustum_culled() {
        let (_proj, vp) = look_vp_naive();
        let bounds = RenderBoundingBox {
            center: nalgebra::Vector3::new(50.0, 0.0, 0.0),
            extents: nalgebra::Vector3::new(0.5, 0.5, 0.5),
        };
        let model = Mat4::IDENTITY;
        assert!(!rigid_mesh_potentially_visible(&bounds, model, vp));
    }

    #[test]
    fn mesh_bounds_degenerate_for_cull_detects_zero_extents() {
        let b = RenderBoundingBox {
            center: NaVector3::zeros(),
            extents: NaVector3::zeros(),
        };
        assert!(mesh_bounds_degenerate_for_cull(&b));
        let b2 = RenderBoundingBox {
            center: NaVector3::zeros(),
            extents: NaVector3::new(1.0, 1.0, 1.0),
        };
        assert!(!mesh_bounds_degenerate_for_cull(&b2));
    }

    /// Regression: host invalid bounds → zero extents at local origin would collapse to one world
    /// point; a tight box at the same center is culled, but degenerate bounds must stay visible.
    #[test]
    fn degenerate_zero_extents_conservative_not_culled() {
        let (_proj, vp) = look_vp_naive();
        let tight = RenderBoundingBox {
            center: NaVector3::new(50.0, 0.0, 0.0),
            extents: NaVector3::new(0.5, 0.5, 0.5),
        };
        assert!(!rigid_mesh_potentially_visible(&tight, Mat4::IDENTITY, vp));
        let degenerate = RenderBoundingBox {
            center: NaVector3::new(50.0, 0.0, 0.0),
            extents: NaVector3::zeros(),
        };
        assert!(rigid_mesh_potentially_visible(
            &degenerate,
            Mat4::IDENTITY,
            vp
        ));
    }

    /// [`view_proj_glam_for_batch`] must match `P * z_flip * V` for the same camera pose.
    #[test]
    fn view_proj_glam_for_batch_matches_z_flipped_look_at() {
        let view_na = NaMatrix4::look_at_rh(
            &Point3::new(0.0, 0.0, 5.0),
            &Point3::new(0.0, 0.0, 0.0),
            &NaVector3::new(0.0, 1.0, 0.0),
        );
        let proj = perspective_proj(1.0);
        let v_glam = matrix_na_to_glam(&view_na);
        let cam_glam = v_glam.inverse();
        let (_scale, rotation, translation) = cam_glam.to_scale_rotation_translation();
        let view_transform = RenderTransform {
            position: NaVector3::new(translation.x, translation.y, translation.z),
            scale: NaVector3::new(1.0, 1.0, 1.0),
            rotation: Quaternion::new(rotation.w, rotation.x, rotation.y, rotation.z),
        };
        let cam_from_rt = render_transform_to_matrix(&view_transform);
        let diff_cam = (cam_glam - cam_from_rt).to_cols_array();
        let max_cam = diff_cam.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_cam < 1e-4,
            "RenderTransform round-trip from look-at camera max abs {}",
            max_cam
        );

        let batch = SpaceDrawBatch {
            space_id: 0,
            is_overlay: false,
            view_transform,
            draws: vec![],
        };
        let vp_batch = view_proj_glam_for_batch(&batch, &proj, None);
        let z_flip = Mat4::from_scale(Vec3::new(1.0, 1.0, -1.0));
        let vp_ref = matrix_na_to_glam(&proj) * z_flip * v_glam;
        let diff = (vp_batch - vp_ref).to_cols_array();
        let max_abs = diff.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-3,
            "view_proj batch vs P*z_flip*V max abs diff {}",
            max_abs
        );
    }
}

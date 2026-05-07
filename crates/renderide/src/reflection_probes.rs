//! Nonblocking GPU SH2 projection for reflection-probe host tasks.

mod projection_pipeline;
mod readback_jobs;
mod sh2_math;
mod sh2_system;
mod source_resolution;
pub(crate) mod specular;
mod task_rows;

use sh2_math::constant_color_sh2;
use sh2_system::{
    DEFAULT_SAMPLE_SIZE, GpuSh2Source, MAX_PENDING_JOB_AGE_FRAMES, Projection360EquirectKey,
    SH2_OUTPUT_BYTES, Sh2ProjectParams, Sh2SourceKey,
};

pub(crate) use sh2_system::ReflectionProbeSh2System;

#[cfg(test)]
use crate::shared::{ComputeResult, ReflectionProbeSH2Task, RenderSH2};
#[cfg(test)]
use crate::skybox::params::{DEFAULT_MAIN_TEX_ST, PROJECTION360_DEFAULT_FOV};
#[cfg(test)]
use glam::Vec3;
#[cfg(test)]
use sh2_math::*;
#[cfg(test)]
use sh2_system::SkyParamMode;
#[cfg(test)]
use task_rows::{
    TaskAnswer, debug_assert_no_scheduled_rows, read_i32_le, task_stride, write_task_answer,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_color_evaluates_back_to_color() {
        let color = Vec3::new(0.25, 0.5, 1.0);
        let sh = constant_color_sh2(color);
        let evaluated = evaluate_sh2(&sh, Vec3::Y);
        assert!((evaluated - color).length() < 1e-5);
    }

    #[test]
    fn basis_constants_match_unity_values() {
        assert!((SH_C0 - 0.282_094_8).abs() < 1e-7);
        assert!((SH_C1 - 0.488_602_52).abs() < 1e-7);
        assert!((SH_C2 - 1.092_548_5).abs() < 1e-7);
        assert!((SH_C3 - 0.315_391_57).abs() < 1e-7);
        assert!((SH_C4 - 0.546_274_24).abs() < 1e-7);
    }

    #[test]
    fn projection360_equirect_view_sampling_uses_opposite_world_direction() {
        let mut params = Sh2ProjectParams::empty(SkyParamMode::Procedural);
        params.color0 = PROJECTION360_DEFAULT_FOV;
        params.color1 = DEFAULT_MAIN_TEX_ST;
        params.scalars = [1.0, 0.0, 0.0, 0.0];

        let world_dir = Vec3::X;
        let visible_uv = projection360_equirect_uv_for_world_dir(world_dir, &params);
        let opposite_uv = raw_equirect_uv_for_dir(-world_dir);
        let direct_uv = raw_equirect_uv_for_dir(world_dir);

        assert!((visible_uv[0] - opposite_uv[0]).abs() < 1e-6);
        assert!((visible_uv[1] - opposite_uv[1]).abs() < 1e-6);
        assert!((visible_uv[0] - direct_uv[0]).abs() > 0.25);
    }

    #[test]
    fn projection360_fov_st_and_storage_affect_equirect_source_key() {
        let mut base = Sh2ProjectParams::empty(SkyParamMode::Procedural);
        base.color0 = PROJECTION360_DEFAULT_FOV;
        base.color1 = DEFAULT_MAIN_TEX_ST;
        base.scalars = [0.0, 0.0, 0.0, 0.0];
        let base_key = Projection360EquirectKey::from_params(&base);

        let mut fov = base;
        fov.color0[2] = 0.125;
        let mut st = base;
        st.color1[2] = 0.25;
        let mut storage = base;
        storage.scalars[0] = 1.0;

        assert_ne!(base_key, Projection360EquirectKey::from_params(&fov));
        assert_ne!(base_key, Projection360EquirectKey::from_params(&st));
        assert_ne!(base_key, Projection360EquirectKey::from_params(&storage));
    }

    #[test]
    fn projection360_cubemap_path_keeps_world_direction() {
        let world_dir = Vec3::new(0.25, 0.5, -1.0).normalize();
        let sample_dir = projection360_cubemap_sample_dir_for_world_dir(world_dir);
        assert!((sample_dir - world_dir).length() < 1e-6);
    }

    #[test]
    fn gradient_sky_sampling_matches_visible_axes() {
        let mut params = Sh2ProjectParams::empty(SkyParamMode::Gradient);
        params.color0 = [0.0, 0.0, 0.0, 1.0];
        params.gradient_count = 1;
        params.dirs_spread[0] = [1.0, 0.0, 0.0, 1.0];
        params.gradient_color0[0] = [1.0, 0.0, 0.0, 1.0];
        params.gradient_color1[0] = [0.0, 0.0, 1.0, 1.0];
        params.gradient_params[0] = [0.0, 1.0, 0.0, 1.0];

        let plus_x = gradient_sky_visible_color_for_dir(Vec3::X, &params);
        let minus_x = gradient_sky_visible_color_for_dir(-Vec3::X, &params);
        let plus_y = gradient_sky_visible_color_for_dir(Vec3::Y, &params);
        let plus_z = gradient_sky_visible_color_for_dir(Vec3::Z, &params);

        assert!((plus_x - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6);
        assert!((minus_x - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-6);
        assert!((plus_y - Vec3::new(0.5, 0.0, 0.5)).length() < 1e-6);
        assert!((plus_z - Vec3::new(0.5, 0.0, 0.5)).length() < 1e-6);
    }

    #[test]
    fn gradient_sky_sampling_keeps_raw_unity_direction_magnitude() {
        let mut params = Sh2ProjectParams::empty(SkyParamMode::Gradient);
        params.color0 = [0.0, 0.0, 0.0, 1.0];
        params.gradient_count = 1;
        params.dirs_spread[0] = [0.5, 0.0, 0.0, 1.0];
        params.gradient_color0[0] = [1.0, 0.0, 0.0, 1.0];
        params.gradient_color1[0] = [0.0, 0.0, 1.0, 1.0];
        params.gradient_params[0] = [0.0, 1.0, 0.0, 1.0];

        let plus_x = gradient_sky_visible_color_for_dir(Vec3::X, &params);

        assert!((plus_x - Vec3::new(0.75, 0.0, 0.25)).length() < 1e-6);
    }

    /// Verifies procedural sky params preserve visible-shader sun and exposure semantics.
    #[test]
    fn procedural_sky_sampling_uses_packed_sun_and_exposure() {
        let mut params = Sh2ProjectParams::empty(SkyParamMode::Procedural);
        params.color0 = [0.4, 0.5, 0.6, 1.0];
        params.color1 = [0.1, 0.1, 0.1, 1.0];
        params.direction = [0.0, 1.0, 0.0, 0.0];
        params.scalars = [2.0, 0.5, 1.0, 1.0];
        params.gradient_color0[0] = [1.0, 0.9, 0.8, 1.0];

        let with_sun = procedural_sky_visible_color_for_dir(Vec3::Y, &params);
        params.scalars[3] = 0.0;
        let without_sun = procedural_sky_visible_color_for_dir(Vec3::Y, &params);
        params.scalars[0] = 1.0;
        let half_exposure = procedural_sky_visible_color_for_dir(Vec3::Y, &params);

        assert!(with_sun.x > without_sun.x);
        assert!((without_sun - half_exposure * 2.0).length() < 1e-5);
    }

    #[test]
    fn projection360_equirect_lobe_evaluates_strongest_in_visible_world_direction() {
        let sh = project_projection360_equirect_lobe(24, -Vec3::X);
        let visible_direction = evaluate_sh2(&sh, Vec3::X).x;
        let opposite_direction = evaluate_sh2(&sh, -Vec3::X).x;
        assert!(visible_direction > opposite_direction);
    }

    #[test]
    fn task_answer_postpone_leaves_no_scheduled_row() {
        const RESULT_OFFSET: usize = std::mem::offset_of!(ReflectionProbeSH2Task, result);
        let mut row = vec![0u8; task_stride()];
        row[0..4].copy_from_slice(&0i32.to_le_bytes());
        row[4..8].copy_from_slice(&0i32.to_le_bytes());
        row[RESULT_OFFSET..RESULT_OFFSET + 4]
            .copy_from_slice(&(ComputeResult::Scheduled as i32).to_le_bytes());

        write_task_answer(&mut row, 0, TaskAnswer::status(ComputeResult::Postpone));
        debug_assert_no_scheduled_rows(&row);

        let result = read_i32_le(&row[RESULT_OFFSET..RESULT_OFFSET + 4]);
        assert_eq!(result, Some(ComputeResult::Postpone as i32));
    }

    /// Per-fragment object-space view direction parity with Unity's `ObjSpaceViewDir(i.pos)`.
    ///
    /// The mesh `Projection360` shader passes the object-space view direction through the
    /// vertex stage *un-normalized* -- perspective-correct interpolation of a function that
    /// is linear in the vertex world position yields the per-fragment direction after
    /// `normalize`. Normalizing per vertex would distort the interpolated direction (the
    /// angular error scales with the triangle's angular extent and breaks narrow-FOV
    /// projections used by video players). This test pins the parity for an orthonormal
    /// model matrix (rotation + translation), which is the practical case.
    #[test]
    fn projection360_object_view_dir_interpolates_to_per_fragment_unity_value() {
        use glam::{Mat3, Mat4, Quat};

        let rotation = Quat::from_axis_angle(Vec3::new(0.3, 0.7, 0.5).normalize(), 1.1);
        let translation = Vec3::new(1.5, -0.5, 2.25);
        let model = Mat4::from_rotation_translation(rotation, translation);
        let model3 = Mat3::from_quat(rotation);

        let v_obj = [
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.5),
        ];
        let cam_world = Vec3::new(0.4, 1.2, -3.5);

        // Vertex-shader output (un-normalized).
        let per_vertex: [Vec3; 3] = std::array::from_fn(|i| {
            let world = model.transform_point3(v_obj[i]);
            model3.transpose() * (cam_world - world)
        });

        // Barycentric center of the triangle.
        let bary = [1.0 / 3.0; 3];
        let interpolated =
            per_vertex[0] * bary[0] + per_vertex[1] * bary[1] + per_vertex[2] * bary[2];
        let frag_dir = interpolated.normalize();

        // Unity reference: `ObjSpaceViewDir(i.pos)` evaluated at the interpolated object-space
        // position, normalized. `inverse(model)` of an orthonormal matrix is its transpose.
        let model_inv = model.inverse();
        let cam_obj = model_inv.transform_point3(cam_world);
        let frag_obj = v_obj[0] * bary[0] + v_obj[1] * bary[1] + v_obj[2] * bary[2];
        let unity_dir = (cam_obj - frag_obj).normalize();

        assert!(
            (frag_dir - unity_dir).length() < 1e-5,
            "interpolated obj_view_dir = {frag_dir:?}, unity reference = {unity_dir:?}",
        );
    }

    #[test]
    fn computed_task_answer_writes_data_before_result_slot() {
        const RESULT_OFFSET: usize = std::mem::offset_of!(ReflectionProbeSH2Task, result);
        const DATA_OFFSET: usize = std::mem::offset_of!(ReflectionProbeSH2Task, result_data);
        let mut row = vec![0u8; task_stride()];
        row[0..4].copy_from_slice(&0i32.to_le_bytes());
        row[4..8].copy_from_slice(&0i32.to_le_bytes());
        let sh = RenderSH2 {
            sh0: Vec3::new(1.0, 2.0, 3.0),
            ..RenderSH2::default()
        };

        write_task_answer(&mut row, 0, TaskAnswer::computed(sh));
        debug_assert_no_scheduled_rows(&row);

        let result = read_i32_le(&row[RESULT_OFFSET..RESULT_OFFSET + 4]);
        let first_component = f32::from_le_bytes(
            row[DATA_OFFSET..DATA_OFFSET + 4]
                .try_into()
                .expect("four-byte f32"),
        );
        assert_eq!(result, Some(ComputeResult::Computed as i32));
        assert_eq!(first_component, 1.0);
    }
}

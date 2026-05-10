//! Camera render-task readback and packing tests.

use glam::IVec2;

use super::*;

#[test]
fn readback_layout_removes_row_padding_contract() {
    let layout = compute_readback_layout(
        wgpu::Extent3d {
            width: 17,
            height: 3,
            depth_or_array_layers: 1,
        },
        4096,
    )
    .expect("layout");

    assert_eq!(layout.bytes_per_row_tight, 68);
    assert_eq!(
        layout.bytes_per_row_padded,
        wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
    );
    assert_eq!(
        layout.buffer_size,
        u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * 3
    );
}

#[test]
fn copy_padded_rows_to_tight_strips_padding() {
    let layout = ReadbackLayout {
        width: 2,
        height: 2,
        bytes_per_row_tight: 8,
        bytes_per_row_padded: 12,
        buffer_size: 24,
    };
    let padded = [
        1, 2, 3, 4, 5, 6, 7, 8, 99, 99, 99, 99, 10, 11, 12, 13, 14, 15, 16, 17, 88, 88, 88, 88,
    ];

    let tight = copy_padded_rows_to_tight(&padded, &layout).expect("copy rows");

    assert_eq!(
        tight,
        vec![1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]
    );
}

#[test]
fn pack_rgba8_preserves_rows_and_converts_formats() {
    let extent = CameraTaskExtent {
        width: 2,
        height: 2,
    };
    let rgba = [
        10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43,
    ];

    let mut argb = vec![0; 16];
    pack_rgba8_to_host_buffer(&rgba, extent, CameraTaskOutputFormat::Argb32, &mut argb)
        .expect("argb pack");
    assert_eq!(
        argb,
        vec![
            13, 10, 11, 12, 23, 20, 21, 22, 33, 30, 31, 32, 43, 40, 41, 42
        ]
    );

    let mut rgba_out = vec![0; 16];
    pack_rgba8_to_host_buffer(&rgba, extent, CameraTaskOutputFormat::Rgba32, &mut rgba_out)
        .expect("rgba pack");
    assert_eq!(
        rgba_out,
        vec![
            10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43
        ]
    );

    let mut bgra = vec![0; 16];
    pack_rgba8_to_host_buffer(&rgba, extent, CameraTaskOutputFormat::Bgra32, &mut bgra)
        .expect("bgra pack");
    assert_eq!(
        bgra,
        vec![
            12, 11, 10, 13, 22, 21, 20, 23, 32, 31, 30, 33, 42, 41, 40, 43
        ]
    );

    let mut rgb = vec![0; 12];
    pack_rgba8_to_host_buffer(&rgba, extent, CameraTaskOutputFormat::Rgb24, &mut rgb)
        .expect("rgb pack");
    assert_eq!(rgb, vec![10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42]);
}

#[test]
fn pack_rgba8_rejects_small_destination_without_writing_past_end() {
    let extent = CameraTaskExtent {
        width: 2,
        height: 1,
    };
    let rgba = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut dst = [9u8; 7];

    let error = pack_rgba8_to_host_buffer(&rgba, extent, CameraTaskOutputFormat::Rgba32, &mut dst)
        .expect_err("small dst must fail");

    assert!(matches!(
        error,
        CameraReadbackError::ResultDescriptorTooSmall {
            required: 8,
            actual: 7
        }
    ));
    assert_eq!(dst, [9u8; 7]);
}

#[test]
fn task_extent_rejects_invalid_dimensions() {
    let parameters = CameraRenderParameters {
        resolution: IVec2::new(-1, 4),
        ..Default::default()
    };

    assert!(matches!(
        CameraTaskExtent::from_parameters(&parameters),
        Err(CameraReadbackError::InvalidExtent {
            width: -1,
            height: 4
        })
    ));
}

#[test]
fn draw_filter_prefers_only_render_list_over_excludes() {
    let task = CameraRenderTask {
        only_render_list: vec![1, 2],
        exclude_render_list: vec![3],
        ..Default::default()
    };

    let filter = draw_filter_from_camera_render_task(&task);

    assert!(filter.only.as_ref().is_some_and(|only| only.contains(&1)));
    assert!(filter.exclude.is_empty());
}

#[test]
fn output_format_accepts_initial_cpu_formats() {
    assert_eq!(
        CameraTaskOutputFormat::from_texture_format(TextureFormat::ARGB32),
        Some(CameraTaskOutputFormat::Argb32)
    );
    assert_eq!(
        CameraTaskOutputFormat::from_texture_format(TextureFormat::RGBA32),
        Some(CameraTaskOutputFormat::Rgba32)
    );
    assert_eq!(
        CameraTaskOutputFormat::from_texture_format(TextureFormat::BGRA32),
        Some(CameraTaskOutputFormat::Bgra32)
    );
    assert_eq!(
        CameraTaskOutputFormat::from_texture_format(TextureFormat::RGB24),
        Some(CameraTaskOutputFormat::Rgb24)
    );
    assert_eq!(
        CameraTaskOutputFormat::from_texture_format(TextureFormat::RGBAHalf),
        None
    );
}

#[test]
fn alpha_coverage_repair_only_runs_for_alpha_outputs() {
    assert!(CameraTaskOutputFormat::Argb32.needs_alpha_coverage_repair());
    assert!(CameraTaskOutputFormat::Rgba32.needs_alpha_coverage_repair());
    assert!(CameraTaskOutputFormat::Bgra32.needs_alpha_coverage_repair());
    assert!(!CameraTaskOutputFormat::Rgb24.needs_alpha_coverage_repair());
}

#[test]
fn alpha_coverage_uses_reverse_z_clear_contract() {
    assert!(!alpha_coverage::depth_marks_coverage(
        crate::gpu::MAIN_FORWARD_DEPTH_CLEAR
    ));
    assert!(!alpha_coverage::depth_marks_coverage(f32::NAN));
    assert!(alpha_coverage::depth_marks_coverage(
        crate::gpu::MAIN_FORWARD_DEPTH_CLEAR + f32::EPSILON
    ));
    assert!(alpha_coverage::depth_marks_coverage(1.0));
}

#[test]
fn camera_render_task_post_processing_policy_matches_host_parameters() {
    let disabled = CameraRenderParameters {
        post_processing: false,
        screen_space_reflections: true,
        ..Default::default()
    };
    let disabled_policy = camera_render_task_post_processing(&disabled);

    assert_eq!(disabled_policy, ViewPostProcessing::disabled());

    let enabled_without_ssr = CameraRenderParameters {
        post_processing: true,
        screen_space_reflections: false,
        ..Default::default()
    };
    let enabled_without_ssr_policy = camera_render_task_post_processing(&enabled_without_ssr);

    assert_eq!(
        enabled_without_ssr_policy,
        ViewPostProcessing::new(true, false, false)
    );

    let enabled_with_ssr = CameraRenderParameters {
        post_processing: true,
        screen_space_reflections: true,
        ..Default::default()
    };
    let enabled_with_ssr_policy = camera_render_task_post_processing(&enabled_with_ssr);

    assert_eq!(
        enabled_with_ssr_policy,
        ViewPostProcessing::new(true, true, false)
    );
}

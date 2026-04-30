//! Adapter feature negotiation.
//!
//! Picks the subset of [`wgpu::Features`] Renderide can use against a given adapter. Pure
//! data: no instance, surface, or device side effects.

/// Intersects [`wgpu::Adapter::features`] with the feature bits Renderide requires for rendering.
///
/// When the `tracy` Cargo feature is active, also requests the subset of
/// `TIMESTAMP_QUERY | TIMESTAMP_QUERY_INSIDE_ENCODERS` that the adapter supports. Pass-level
/// queries only need `TIMESTAMP_QUERY`; `TIMESTAMP_QUERY_INSIDE_ENCODERS` additionally enables
/// encoder-level queries. Either feature being absent is gracefully tolerated:
/// [`crate::profiling::GpuProfilerHandle::try_new`] returns [`None`] only when
/// `TIMESTAMP_QUERY` itself is missing.
pub(crate) fn adapter_render_features_intersection(adapter: &wgpu::Adapter) -> wgpu::Features {
    let compression = wgpu::Features::TEXTURE_COMPRESSION_BC
        | wgpu::Features::TEXTURE_COMPRESSION_ETC2
        | wgpu::Features::TEXTURE_COMPRESSION_ASTC;
    let optional_float32_filterable = wgpu::Features::FLOAT32_FILTERABLE;
    let adapter_format_features = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    let optional_depth32_stencil8 = wgpu::Features::DEPTH32FLOAT_STENCIL8;
    let multisample_array = wgpu::Features::MULTISAMPLE_ARRAY;
    let timestamp = crate::profiling::timestamp_query_features_if_supported(adapter);
    adapter.features()
        & (compression
            | optional_float32_filterable
            | adapter_format_features
            | optional_depth32_stencil8
            | multisample_array)
        | timestamp
}

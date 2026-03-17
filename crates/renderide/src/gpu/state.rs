//! GPU state: surface, device, queue, and mesh buffer cache.
//!
//! Extension point for frustum culling.
//! Stub: use nalgebra::Aabb3 to test mesh AABB against view frustum planes.
//! Types: Aabb3<f32>, Point3<f32>, Vector3<f32>, Matrix4<f32>.
//! fn frustum_cull(aabb: &Aabb3<f32>, view_proj: &Matrix4<f32>) -> bool { ... }

use winit::window::Window;

use super::accel::{AccelCache, RayTracingState};
use super::mesh::GpuMeshBuffers;

/// wgpu state for rendering.
pub struct GpuState {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub mesh_buffer_cache: std::collections::HashMap<i32, GpuMeshBuffers>,
    pub depth_texture: Option<wgpu::Texture>,
    /// Dimensions of the current depth texture. Used to avoid recreation on resize when unchanged.
    pub depth_size: (u32, u32),
    /// Whether EXPERIMENTAL_RAY_QUERY was successfully requested and enabled at device creation.
    /// Used for future RTAO (Ray-Traced Ambient Occlusion) support.
    pub ray_tracing_available: bool,
    /// BLAS cache for non-skinned meshes. `Some` only when [`ray_tracing_available`](Self::ray_tracing_available).
    pub accel_cache: Option<AccelCache>,
    /// Ray tracing state holding the current frame's TLAS. Rebuilt each frame when ray tracing available.
    pub ray_tracing_state: Option<RayTracingState>,
}

/// Initializes wgpu surface, device, queue, and mesh pipeline.
///
/// Prefers Vulkan backend when available (ray tracing often better supported).
/// Uses [`wgpu::PowerPreference::HighPerformance`] to prefer discrete GPUs (NVIDIA/AMD)
/// over integrated (Intel), since integrated GPUs often report ray query support but
/// have `max_blas_geometry_count=0` (no actual acceleration structure support).
pub async fn init_gpu(
    window: &Window,
) -> Result<GpuState, Box<dyn std::error::Error + Send + Sync>> {
    let enabled_backends = wgpu::Instance::enabled_backend_features();
    let use_vulkan_only = enabled_backends.contains(wgpu::Backends::VULKAN);

    logger::info!(
        "GPU init: backends={:?} use_vulkan_only={} (Vulkan preferred for ray tracing)",
        enabled_backends,
        use_vulkan_only
    );

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: if use_vulkan_only {
            wgpu::Backends::VULKAN
        } else {
            enabled_backends
        },
        ..Default::default()
    });

    let surface = instance
        .create_surface(window)
        .map_err(|e| format!("create_surface: {:?}", e))?;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| format!("request_adapter: {:?}", e))?;

    let adapter_info = adapter.get_info();
    let ray_query_supported =
        adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);

    let required_features = if ray_query_supported {
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::EXPERIMENTAL_RAY_QUERY
    } else {
        wgpu::Features::TIMESTAMP_QUERY
    };

    let experimental_features = if ray_query_supported {
        unsafe { wgpu::ExperimentalFeatures::enabled() }
    } else {
        wgpu::ExperimentalFeatures::disabled()
    };

    let required_limits = if ray_query_supported {
        wgpu::Limits::default().using_acceleration_structure_values(adapter.limits())
    } else {
        wgpu::Limits::default()
    };

    let (device, queue, ray_tracing_available) = match adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features,
            required_limits,
            experimental_features,
            ..Default::default()
        })
        .await
    {
        Ok((device, queue)) => {
            let max_blas = device.limits().max_blas_geometry_count;
            let ray_tracing_available = ray_query_supported && max_blas > 0;

            logger::info!(
                "GPU init: adapter={} backend={:?} ray_query_supported={} max_blas_geometry_count={} ray_tracing_available={}",
                adapter_info.name,
                adapter_info.backend,
                ray_query_supported,
                max_blas,
                ray_tracing_available
            );
            if !ray_tracing_available && ray_query_supported {
                logger::warn!(
                    "Ray tracing disabled: {} reports ray query but max_blas_geometry_count={} (Intel integrated/software rasterizer). Use a discrete GPU (NVIDIA RTX, AMD RX 6000+) for RTAO.",
                    adapter_info.name,
                    max_blas
                );
            } else if !ray_query_supported {
                logger::info!(
                    "Ray tracing unavailable: adapter does not support EXPERIMENTAL_RAY_QUERY (Vulkan/DX12 ray tracing)"
                );
            }

            (device, queue, ray_tracing_available)
        }
        Err(e) if ray_query_supported => {
            logger::warn!(
                "GPU init: request_device with ray query failed: {:?}, falling back to non-ray-tracing device",
                e
            );
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::TIMESTAMP_QUERY,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    ..Default::default()
                })
                .await
                .map_err(|e2| format!("request_device fallback: {:?}", e2))?;

            logger::info!(
                "GPU init: adapter={} backend={:?} ray_tracing_available=false (fallback device)",
                adapter_info.name,
                adapter_info.backend
            );

            (device, queue, false)
        }
        Err(e) => return Err(format!("request_device: {:?}", e).into()),
    };
    let size = window.inner_size();
    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    config.present_mode = wgpu::PresentMode::Fifo;
    surface.configure(&device, &config);
    let depth_texture = create_depth_texture(&device, &config);
    let depth_size = (config.width, config.height);

    Ok(GpuState {
        surface: unsafe { std::mem::transmute(surface) },
        device,
        queue,
        config,
        mesh_buffer_cache: std::collections::HashMap::new(),
        depth_texture: Some(depth_texture),
        depth_size,
        ray_tracing_available,
        accel_cache: if ray_tracing_available {
            Some(AccelCache::new())
        } else {
            None
        },
        ray_tracing_state: if ray_tracing_available {
            Some(RayTracingState::new())
        } else {
            None
        },
    })
}

/// Creates a depth-stencil texture for the given surface configuration.
///
/// Uses [`Depth24PlusStencil8`] to support GraphicsChunk masking (scroll rects, clipping)
/// in the overlay pass. Stencil is cleared at the start of the mesh pass.
pub fn create_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth-stencil texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

/// Ensures depth texture matches the given config. Reuses existing if dimensions match.
/// Returns `Some(new_texture)` when recreation is needed, `None` when current can be reused.
pub fn ensure_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    depth_size: (u32, u32),
) -> Option<wgpu::Texture> {
    if depth_size.0 == config.width && depth_size.1 == config.height {
        None
    } else {
        Some(create_depth_texture(device, config))
    }
}

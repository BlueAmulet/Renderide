//! Cubemap storage-orientation helpers.
//!
//! Host cubemap face data uses Unity's per-face V orientation. These helpers map canonical sample
//! directions to that storage convention while keeping callers' world directions unchanged.

#define_import_path renderide::cubemap_storage

/// Returns the source texture direction for a canonical cubemap sample direction.
fn sample_dir(dir: vec3<f32>, storage_v_inverted: f32) -> vec3<f32> {
    if (storage_v_inverted <= 0.5) {
        return dir;
    }
    let a = abs(dir);
    if (a.y >= a.x && a.y >= a.z) {
        return vec3<f32>(dir.x, dir.y, -dir.z);
    }
    return vec3<f32>(dir.x, -dir.y, dir.z);
}

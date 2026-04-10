//! Runtime discovery of the Khronos OpenXR **loader** shared library (`openxr_loader.dll` on Windows,
//! `libopenxr_loader.so` on Linux, `libopenxr_loader.dylib` on macOS).
//!
//! # Override path
//!
//! Set [`RENDERIDE_OPENXR_LOADER`] to a filesystem path that is either:
//! - the loader library file itself (e.g. `C:\Path\to\openxr_loader.dll`), or
//! - a directory containing that file (the per-OS filename below is appended).
//!
//! This is checked after the executable’s directory and before optional standard install locations
//! (Windows only) and the platform default search used by [`openxr::Entry::load`].
//!
//! # Shipping builds
//!
//! Copy the loader from the [Khronos OpenXR SDK](https://github.com/KhronosGroup/OpenXR-SDK) next to
//! the renderer executable so discovery succeeds without changing `PATH`.

use std::path::PathBuf;

use openxr as xr;

use super::XrBootstrapError;

/// Environment variable: path to the OpenXR loader library file, or a directory that contains it.
pub const RENDERIDE_OPENXR_LOADER: &str = "RENDERIDE_OPENXR_LOADER";

/// Basename of the Khronos OpenXR loader for the current target OS (matches openxr-rs `Entry::load`).
pub fn openxr_loader_library_filename() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "openxr_loader.dll"
    }
    #[cfg(target_os = "macos")]
    {
        "libopenxr_loader.dylib"
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        "libopenxr_loader.so"
    }
}

fn push_unique(out: &mut Vec<PathBuf>, path: PathBuf) {
    if !out.iter().any(|p| p == &path) {
        out.push(path);
    }
}

/// Resolves `RENDERIDE_OPENXR_LOADER` into a single candidate path to the loader library.
fn path_from_renderide_openxr_loader_env(name: &str) -> Option<PathBuf> {
    let raw = std::env::var_os(RENDERIDE_OPENXR_LOADER)?;
    let p = PathBuf::from(raw);
    Some(if p.is_dir() {
        p.join(name)
    } else {
        p
    })
}

#[cfg(target_os = "windows")]
fn windows_openxr_sdk_bin_candidates(name: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for key in ["ProgramFiles", "ProgramFiles(x86)"] {
        if let Ok(root) = std::env::var(key) {
            let base = PathBuf::from(root).join("OpenXR");
            for rel in ["bin", "bin/win64"] {
                push_unique(&mut out, base.join(rel).join(name));
            }
        }
    }
    out
}

/// Ordered candidate paths for [`openxr::Entry::load_from`]: executable directory, env override,
/// optional OS-specific defaults, then implicit search via [`openxr::Entry::load`].
pub fn openxr_loader_candidate_paths() -> Vec<PathBuf> {
    let name = openxr_loader_library_filename();
    let mut out = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            push_unique(&mut out, parent.join(name));
        }
    }

    if let Some(p) = path_from_renderide_openxr_loader_env(name) {
        push_unique(&mut out, p);
    }

    #[cfg(target_os = "windows")]
    {
        for p in windows_openxr_sdk_bin_candidates(name) {
            push_unique(&mut out, p);
        }
    }

    out
}

/// Loads the OpenXR entry by trying [`openxr_loader_candidate_paths`] with [`xr::Entry::load_from`],
/// then falling back to [`xr::Entry::load`].
pub fn load_openxr_entry() -> Result<xr::Entry, XrBootstrapError> {
    let paths = openxr_loader_candidate_paths();
    for path in &paths {
        match unsafe { xr::Entry::load_from(path.as_path()) } {
            Ok(entry) => {
                logger::debug!("OpenXR loader loaded from {}", path.display());
                return Ok(entry);
            }
            Err(e) => {
                logger::trace!(
                    "OpenXR loader not loaded from {}: {e}",
                    path.display()
                );
            }
        }
    }

    match unsafe { xr::Entry::load() } {
        Ok(entry) => {
            logger::debug!("OpenXR loader loaded via default library search");
            Ok(entry)
        }
        Err(e) => Err(XrBootstrapError::Message(format!(
            "OpenXR loader not found: {e}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_filename_matches_openxr_crate() {
        #[cfg(target_os = "windows")]
        assert_eq!(openxr_loader_library_filename(), "openxr_loader.dll");
        #[cfg(target_os = "macos")]
        assert_eq!(openxr_loader_library_filename(), "libopenxr_loader.dylib");
        #[cfg(all(
            not(target_os = "windows"),
            not(target_os = "macos")
        ))]
        assert_eq!(openxr_loader_library_filename(), "libopenxr_loader.so");
    }

    #[test]
    fn candidate_paths_include_exe_parent_joined_name() {
        let paths = openxr_loader_candidate_paths();
        let name = openxr_loader_library_filename();
        let exe_parent = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()));
        if let Some(parent) = exe_parent {
            assert!(
                paths.contains(&parent.join(name)),
                "expected {:?} in candidates {:?}",
                parent.join(name),
                paths
            );
        }
    }
}

//! Windows: optionally copy `openxr_loader.dll` next to the build output so dynamic OpenXR loading
//! can resolve the loader when running tests or the binary from `target/<profile>/`.

fn main() {
    #[cfg(windows)]
    copy_openxr_loader();
}

#[cfg(windows)]
fn copy_openxr_loader() {
    use std::path::PathBuf;

    let Some(out_dir) = resolve_cargo_target_dir() else {
        println!("cargo:rerun-if-changed=build.rs");
        return;
    };

    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".into());
    let dest_dir = out_dir.join(&profile);

    // Common SteamVR locations for openxr_loader.dll
    let candidates = [
        r"C:\Program Files (x86)\Steam\steamapps\common\SteamVR\bin\win64\openxr_loader.dll",
        r"C:\Program Files\Steam\steamapps\common\SteamVR\bin\win64\openxr_loader.dll",
    ];

    let src = candidates
        .iter()
        .map(std::path::Path::new)
        .find(|p| p.exists());

    if let Some(src) = src {
        let dest = dest_dir.join("openxr_loader.dll");
        if !dest.exists() {
            match std::fs::copy(src, &dest) {
                Ok(_) => {
                    println!("cargo:warning=Copied openxr_loader.dll from SteamVR to {dest:?}");
                }
                Err(e) => {
                    println!(
                        "cargo:warning=Could not copy openxr_loader.dll from {} to {}: {e}",
                        src.display(),
                        dest.display()
                    );
                }
            }
        }
    } else {
        println!("cargo:warning=openxr_loader.dll not found in SteamVR — VR may fall back to desktop. Install SteamVR or copy openxr_loader.dll manually to target/{profile}/");
    }

    println!("cargo:rerun-if-changed=build.rs");
}

/// Resolves the Cargo `target/` directory used for artifact output, or `None` if unavailable.
#[cfg(windows)]
fn resolve_cargo_target_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;

    if let Ok(dir) = std::env::var("CARGO_TARGET_DIR") {
        return Some(PathBuf::from(dir));
    }

    let Some(manifest_os) = std::env::var_os("CARGO_MANIFEST_DIR") else {
        println!("cargo:warning=copy_openxr_loader: CARGO_MANIFEST_DIR unset; skipping OpenXR loader copy");
        return None;
    };

    let manifest = PathBuf::from(manifest_os);
    let found = manifest.ancestors().find_map(|p| {
        let t = p.join("target");
        if t.is_dir() {
            Some(t)
        } else {
            None
        }
    });

    if found.is_none() {
        println!("cargo:warning=copy_openxr_loader: could not find target/ directory; skipping OpenXR loader copy");
    }

    found
}

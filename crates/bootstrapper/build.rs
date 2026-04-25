//! Build script that embeds a Windows side-by-side manifest declaring a
//! dependency on Common Controls v6, so `rfd`'s `TaskDialogIndirect` import
//! resolves at process load time instead of failing with "Entry Point Not
//! Found in comctl32.dll".
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_CFG_WINDOWS").is_some() {
        use embed_manifest::{embed_manifest, new_manifest};
        embed_manifest(new_manifest("Renderide.Bootstrapper"))?;
    }
    Ok(())
}

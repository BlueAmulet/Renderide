//! Source audits for WGSL module factoring invariants.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Returns the renderide crate directory.
fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Recursively returns all WGSL files below `relative_dir`.
fn wgsl_files_recursive(relative_dir: &str) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    collect_wgsl_files(&manifest_dir().join(relative_dir), &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_wgsl_files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_wgsl_files(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "wgsl") {
            out.push(path);
        }
    }
    Ok(())
}

fn file_label(path: &Path) -> String {
    normalize_file_label(
        path.strip_prefix(manifest_dir())
            .unwrap_or(path)
            .display()
            .to_string(),
    )
}

fn normalize_file_label(label: impl AsRef<str>) -> String {
    label.as_ref().replace('\\', "/")
}

fn define_import_path(src: &str) -> Option<&str> {
    src.lines().find_map(|line| {
        line.trim_start()
            .strip_prefix("#define_import_path")
            .map(str::trim)
            .filter(|path| !path.is_empty())
    })
}

fn material_source(file_name: &str) -> io::Result<String> {
    fs::read_to_string(manifest_dir().join("shaders/materials").join(file_name))
}

fn module_source(file_name: &str) -> io::Result<String> {
    fs::read_to_string(manifest_dir().join("shaders/modules").join(file_name))
}

fn declares_f32_field(src: &str, field_name: &str) -> bool {
    src.lines().any(|line| {
        let trimmed = line.trim();
        let Some((name, ty)) = trimmed.split_once(':') else {
            return false;
        };
        name.trim() == field_name && ty.trim_start().starts_with("f32")
    })
}

fn all_texture_samples_guarded_by_keyword(src: &str, texture_name: &str, keyword: &str) -> bool {
    let sample = format!("textureSample({texture_name},");
    let guard = format!("uvu::kw_enabled(mat.{keyword})");
    let mut saw_sample = false;

    for (sample_pos, _) in src.match_indices(&sample) {
        saw_sample = true;
        let before_sample = &src[..sample_pos];
        let Some(guard_pos) = before_sample.rfind(&guard) else {
            return false;
        };
        if before_sample[guard_pos..].contains('}') {
            return false;
        }
    }

    saw_sample
}

fn normal_sampling_guarded_by_keyword(src: &str) -> bool {
    let Some(call_pos) = src.find("sample_optional_world_normal(") else {
        return false;
    };
    let call = &src[call_pos..];
    let Some(call_end) = call.find(");") else {
        return false;
    };
    call[..call_end].contains("uvu::kw_enabled(mat._NORMALMAP)")
}

fn count_font_atlas_lod_bias_samples(src: &str) -> usize {
    src.match_indices("ts::sample_tex_2d(")
        .filter(|(sample_pos, _)| {
            let call = &src[*sample_pos..];
            let call_end = call.find(");").unwrap_or(call.len());
            call[..call_end].contains("_FontAtlas")
        })
        .count()
}

#[path = "shader_module_audit/hygiene.rs"]
mod hygiene;
#[path = "shader_module_audit/pbs.rs"]
mod pbs;
#[path = "shader_module_audit/text.rs"]
mod text;
#[path = "shader_module_audit/xiexe_and_probes.rs"]
mod xiexe_and_probes;

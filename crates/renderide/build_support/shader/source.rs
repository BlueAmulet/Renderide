//! Shader source discovery and source-alias loading.

use std::fs;
use std::path::{Path, PathBuf};

use super::directives::parse_source_alias;
use super::error::BuildError;
use super::model::{ShaderJob, ShaderSourceClass};

/// Lists every `.wgsl` file directly under `dir`, sorted lexicographically.
pub(super) fn list_wgsl_files(dir: &Path) -> Result<Vec<PathBuf>, BuildError> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| BuildError::Message(format!("read {}: {e}", dir.display())))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "wgsl"))
        .collect();
    paths.sort();
    Ok(paths)
}

/// Discovers all source shaders that must be compiled, in deterministic order.
pub(super) fn discover_shader_jobs(shader_root: &Path) -> Result<Vec<ShaderJob>, BuildError> {
    let mut jobs = Vec::new();
    for source_class in ShaderSourceClass::ALL {
        let dir = shader_root.join(source_class.source_subdir());
        if !dir.is_dir() {
            continue;
        }
        for source_path in list_wgsl_files(&dir)? {
            jobs.push(ShaderJob {
                compile_order: jobs.len(),
                source_class,
                source_path,
                validation: source_class.validation(),
            });
        }
    }
    Ok(jobs)
}

/// Loads the WGSL source used for composition, following `//#source_alias` when present.
pub(super) fn shader_source_for_compile(
    source_path: &Path,
) -> Result<(String, String), BuildError> {
    let wrapper_source = fs::read_to_string(source_path)
        .map_err(|e| BuildError::Message(format!("read {}: {e}", source_path.display())))?;
    let wrapper_file_path = source_path.to_str().ok_or_else(|| {
        BuildError::Message(format!(
            "shader path must be UTF-8: {}",
            source_path.display()
        ))
    })?;
    let Some(alias) = parse_source_alias(&wrapper_source, wrapper_file_path)? else {
        return Ok((wrapper_source, wrapper_file_path.to_string()));
    };
    let alias_path = source_path.with_file_name(format!("{alias}.wgsl"));
    if alias_path == source_path {
        return Err(BuildError::Message(format!(
            "{wrapper_file_path}: `//#source_alias` cannot point at itself"
        )));
    }
    let alias_source = fs::read_to_string(&alias_path)
        .map_err(|e| BuildError::Message(format!("read {}: {e}", alias_path.display())))?;
    let alias_file_path = alias_path.to_str().ok_or_else(|| {
        BuildError::Message(format!(
            "shader alias path must be UTF-8: {}",
            alias_path.display()
        ))
    })?;
    Ok((alias_source, alias_file_path.to_string()))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    /// Source discovery follows the algebraic class order and new shader layout.
    #[test]
    fn discovers_jobs_in_class_order() -> Result<(), BuildError> {
        let root = tempfile::tempdir()?;
        for subdir in ["materials", "passes/post", "passes/compute"] {
            fs::create_dir_all(root.path().join(subdir))?;
        }
        fs::write(root.path().join("passes/compute/compute_b.wgsl"), "")?;
        fs::write(root.path().join("materials/mat_b.wgsl"), "")?;
        fs::write(root.path().join("materials/mat_a.wgsl"), "")?;
        fs::write(root.path().join("passes/post/post_a.wgsl"), "")?;

        let jobs = discover_shader_jobs(root.path())?;
        let names = jobs
            .iter()
            .map(|job| {
                job.source_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("")
            })
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            ["mat_a.wgsl", "mat_b.wgsl", "post_a.wgsl", "compute_b.wgsl"]
        );
        assert_eq!(jobs[0].source_class, ShaderSourceClass::Material);
        assert_eq!(jobs[2].source_class, ShaderSourceClass::Post);
        assert_eq!(jobs[3].source_class, ShaderSourceClass::Compute);
        Ok(())
    }
}

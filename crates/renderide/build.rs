//! Composes `shaders/source/modules/*.wgsl` with [`naga_oil`] (`#import`), validates with naga, and
//! writes flat `shaders/target/*.wgsl` plus `OUT_DIR/embedded_shaders.rs`.
//!
//! Material sources under `shaders/source/materials/*.wgsl`: the **file stem** must match
//! `normalize_unity_shader_lookup_key` in the renderide crate (Unity shader **asset** name, e.g.
//! `UI_TextUnlit` → `ui_textunlit` → `ui_textunlit_default` / `ui_textunlit_multiview`).
//! Build **always** emits composed targets `{stem}_default` and `{stem}_multiview` for every material file;
//! each source must handle multiview (typically `#ifdef MULTIVIEW` around `@builtin(view_index)` and
//! view-projection selection in a single `vs_main`, or separate `vs_main` blocks) as documented below.
//!
//! ## Multiview variants (`*_default` / `*_multiview`)
//!
//! Sources use `#ifdef MULTIVIEW` … `#else` … `#endif`. In naga-oil, `#ifdef NAME` is true when
//! `NAME` exists in the compose [`ShaderDefValue`] map **regardless of value** — so
//! `MULTIVIEW` → `ShaderDefValue::Bool(false)` still
//! enables the `#ifdef` branch. For the non-multiview target, **omit** `MULTIVIEW` from `shader_defs`
//! entirely; for the multiview target, set `MULTIVIEW` to [`ShaderDefValue::Bool(true)`].
//!
//! Alternatively, WGSL could use `#if MULTIVIEW == true` with both `Bool(true)` and `Bool(false)`
//! in the map; the omit-key approach matches existing `#ifdef` in source materials without edits.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use naga::back::wgsl::WriterFlags;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue, ShaderLanguage,
    ShaderType,
};

/// Shader defs for material sources that use `#ifdef MULTIVIEW`.
///
/// When `enable` is false, returns an empty map so `#ifdef MULTIVIEW` is not taken. When true,
/// inserts `MULTIVIEW` as [`ShaderDefValue::Bool(true)`].
fn multiview_shader_defs(enable: bool) -> HashMap<String, ShaderDefValue> {
    let mut defs = HashMap::new();
    if enable {
        defs.insert("MULTIVIEW".to_string(), ShaderDefValue::Bool(true));
    }
    defs
}

fn validate_and_write_wgsl(
    module: &naga::Module,
    label: &str,
    out_path: &std::path::Path,
    expect_view_index: Option<bool>,
) {
    let has_vs = module
        .entry_points
        .iter()
        .any(|e| e.stage == naga::ShaderStage::Vertex && e.name == "vs_main");
    let has_fs = module
        .entry_points
        .iter()
        .any(|e| e.stage == naga::ShaderStage::Fragment && e.name == "fs_main");
    if !has_vs || !has_fs {
        panic!(
            "{label}: expected entry points vs_main and fs_main (vertex={has_vs} fragment={has_fs})",
        );
    }

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator
        .validate(module)
        .unwrap_or_else(|e| panic!("validate {label}: {e}"));
    let wgsl = naga::back::wgsl::write_string(module, &info, WriterFlags::EXPLICIT_TYPES)
        .unwrap_or_else(|e| panic!("wgsl out {label}: {e}"));
    if let Some(want) = expect_view_index {
        let has = wgsl.contains("@builtin(view_index)");
        if want != has {
            panic!(
                "{label}: expected @builtin(view_index) {} in output (multiview shader_defs contract)",
                if want { "present" } else { "absent" }
            );
        }
    }
    fs::write(out_path, &wgsl).unwrap_or_else(|e| panic!("write {}: {e}", out_path.display()));
}

fn add_globals_module(composer: &mut Composer, globals_source: &str) {
    composer
        .add_composable_module(ComposableModuleDescriptor {
            source: globals_source,
            file_path: "shaders/source/modules/globals.wgsl",
            language: ShaderLanguage::Wgsl,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("add globals module: {e}"));
}

fn add_per_draw_module(composer: &mut Composer, per_draw_source: &str) {
    composer
        .add_composable_module(ComposableModuleDescriptor {
            source: per_draw_source,
            file_path: "shaders/source/modules/per_draw.wgsl",
            language: ShaderLanguage::Wgsl,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("add per_draw module: {e}"));
}

fn add_pbs_brdf_module(composer: &mut Composer, source: &str) {
    composer
        .add_composable_module(ComposableModuleDescriptor {
            source,
            file_path: "shaders/source/modules/pbs_brdf.wgsl",
            language: ShaderLanguage::Wgsl,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("add pbs_brdf module: {e}"));
}

fn add_pbs_cluster_module(composer: &mut Composer, source: &str) {
    composer
        .add_composable_module(ComposableModuleDescriptor {
            source,
            file_path: "shaders/source/modules/pbs_cluster.wgsl",
            language: ShaderLanguage::Wgsl,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("add pbs_cluster module: {e}"));
}

fn add_alpha_clip_sample_module(composer: &mut Composer, source: &str) {
    composer
        .add_composable_module(ComposableModuleDescriptor {
            source,
            file_path: "shaders/source/modules/alpha_clip_sample.wgsl",
            language: ShaderLanguage::Wgsl,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("add alpha_clip_sample module: {e}"));
}

/// Composes one material with all shared naga-oil modules (globals, PBS, alpha-clip helpers).
#[allow(clippy::too_many_arguments)] // One call site; splitting would not improve readability.
fn compose_material(
    globals_source: &str,
    per_draw_source: &str,
    pbs_brdf_source: &str,
    pbs_cluster_source: &str,
    alpha_clip_sample_source: &str,
    material_source: &str,
    material_file_path: &str,
    shader_defs: HashMap<String, ShaderDefValue>,
) -> naga::Module {
    let mut composer = Composer::default().with_capabilities(Capabilities::all());
    add_globals_module(&mut composer, globals_source);
    add_per_draw_module(&mut composer, per_draw_source);
    add_pbs_brdf_module(&mut composer, pbs_brdf_source);
    add_pbs_cluster_module(&mut composer, pbs_cluster_source);
    add_alpha_clip_sample_module(&mut composer, alpha_clip_sample_source);
    composer
        .make_naga_module(NagaModuleDescriptor {
            source: material_source,
            file_path: material_file_path,
            shader_type: ShaderType::Wgsl,
            shader_defs,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("compose {material_file_path}: {e}"))
}

fn main() {
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let globals_path = manifest_dir.join("shaders/source/modules/globals.wgsl");
    let per_draw_path = manifest_dir.join("shaders/source/modules/per_draw.wgsl");
    let pbs_brdf_path = manifest_dir.join("shaders/source/modules/pbs_brdf.wgsl");
    let pbs_cluster_path = manifest_dir.join("shaders/source/modules/pbs_cluster.wgsl");
    let alpha_clip_sample_path = manifest_dir.join("shaders/source/modules/alpha_clip_sample.wgsl");
    let materials_dir = manifest_dir.join("shaders/source/materials");
    let target_dir = manifest_dir.join("shaders/target");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));

    println!("cargo:rerun-if-changed=shaders/source");
    println!("cargo:rerun-if-changed=build.rs");

    fs::create_dir_all(materials_dir.parent().expect("materials parent")).expect("mkdir");
    fs::create_dir_all(&target_dir).expect("mkdir target");

    let globals_source = fs::read_to_string(&globals_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", globals_path.display()));
    let per_draw_source = fs::read_to_string(&per_draw_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", per_draw_path.display()));
    let pbs_brdf_source = fs::read_to_string(&pbs_brdf_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", pbs_brdf_path.display()));
    let pbs_cluster_source = fs::read_to_string(&pbs_cluster_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", pbs_cluster_path.display()));
    let alpha_clip_sample_source = fs::read_to_string(&alpha_clip_sample_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", alpha_clip_sample_path.display()));

    let mut embedded_arms = String::new();
    let mut output_stems: Vec<String> = Vec::new();

    let mut material_paths: Vec<PathBuf> = fs::read_dir(&materials_dir)
        .expect("read materials")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "wgsl").unwrap_or(false))
        .collect();
    material_paths.sort();

    for path in &material_paths {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("material stem");

        let material_source =
            fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

        let variants = [
            (format!("{stem}_default"), false),
            (format!("{stem}_multiview"), true),
        ];
        for (target_stem, multiview) in variants {
            let defs = multiview_shader_defs(multiview);
            let module = compose_material(
                &globals_source,
                &per_draw_source,
                &pbs_brdf_source,
                &pbs_cluster_source,
                &alpha_clip_sample_source,
                &material_source,
                path.to_str().expect("utf8 path"),
                defs,
            );
            let label = format!("{target_stem} (MULTIVIEW={multiview})");
            let out_path = target_dir.join(format!("{target_stem}.wgsl"));
            validate_and_write_wgsl(&module, &label, &out_path, Some(multiview));

            embedded_arms.push_str(&format!(
                "        \"{target_stem}\" => Some(include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/shaders/target/{target_stem}.wgsl\"))),\n"
            ));
            output_stems.push(target_stem);
        }
    }

    let stems_list = output_stems
        .iter()
        .map(|s| format!("    \"{s}\","))
        .collect::<Vec<_>>()
        .join("\n");

    let embedded_rs = format!(
        r#"// Generated by `build.rs` — do not edit.

/// Flattened WGSL for `stem` (see `shaders/target/{{stem}}.wgsl`).
pub fn embedded_target_wgsl(stem: &str) -> Option<&'static str> {{
    match stem {{
{embedded_arms}        _ => None,
    }}
}}

/// Stems under `shaders/target/*.wgsl` present at build time.
pub const COMPILED_MATERIAL_STEMS: &[&str] = &[
{stems}
];
"#,
        embedded_arms = embedded_arms,
        stems = stems_list
    );

    let gen_path = out_dir.join("embedded_shaders.rs");
    fs::write(&gen_path, embedded_rs).expect("write embedded_shaders.rs");
}

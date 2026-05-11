//! Resolve shader asset names from on-disk **Unity AssetBundle** files using `unity-asset`.
//!
//! [`crate::shared::ShaderUpload::file`] is typically an **extensionless path** (or any path) whose bytes
//! parse as UnityFS / AssetBundle data--not a Unity `.asset` YAML file. Route selection still prefers
//! [`unity_asset::environment::Environment::bundle_container_entries`]: `AssetBundle.m_Container`
//! asset paths matched to embedded Shader objects, then stemmed (e.g. `.../ui_unlit.shader` -> `ui_unlit`).
//!
//! Serialized shader objects are also read for the internal Shader name so Froox variant suffixes
//! (`{shader_name}_{variant_bits:08X}`) can be stripped and carried as metadata.

use std::fmt::Display;
use std::path::Path;

use unity_asset::AssetBundle;
use unity_asset::SerializedFile;
use unity_asset::UnityValue;
use unity_asset::class_ids::SHADER;
use unity_asset::environment::BinarySource;
use unity_asset::environment::Environment;
use unity_asset::load_bundle_from_memory;

/// Maximum file size to read when probing a bundle.
const MAX_READ_BYTES: usize = 32 * 1024 * 1024;

/// Maximum regular files examined under a directory hint (dev / loose layouts).
const MAX_DIR_FILES: usize = 256;

/// Maximum characters from parse errors included in logs.
const MAX_ERR_LOG_CHARS: usize = 240;

/// Hex prefix length for short probe lines.
const PROBE_HEX_SHORT: usize = 8;

/// Shader asset route metadata resolved from an uploaded Unity shader AssetBundle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedUnityShaderAsset {
    /// Shader asset filename or stem used for route selection.
    pub shader_asset_name: String,
    /// Froox shader variant bitmask parsed from the internal Shader name suffix, when present.
    pub shader_variant_bits: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InternalShaderName {
    full_name: String,
    shader_asset_name: String,
    shader_variant_bits: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InternalNameSource {
    MNamePeek,
    UnityObjectName,
    UnityObjectNameField,
    UnityObjectParsedForm,
    UnityObjectScript,
    ShaderLabBytes,
    ShaderObjectBytes,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InternalShaderNameCandidate {
    path_id: i64,
    class_id: i32,
    source: InternalNameSource,
    name: InternalShaderName,
}

/// Shader asset filename or stem plus optional Froox variant bitmask from a filesystem path.
pub(crate) fn try_resolve_shader_asset_name_from_path(
    path: &Path,
) -> Option<ResolvedUnityShaderAsset> {
    let meta = std::fs::metadata(path).ok()?;
    let resolved = if meta.is_file() {
        try_from_file(path)
    } else if meta.is_dir() {
        try_from_directory(path)
    } else {
        None
    };
    if let Some(parsed) = &resolved {
        logger::info!(
            "shader_unity_asset: resolved shader_asset_name={:?} shader_variant_bits={:?} from path {}",
            parsed.shader_asset_name,
            parsed.shader_variant_bits,
            path.display()
        );
    }
    resolved
}

fn try_from_file(path: &Path) -> Option<ResolvedUnityShaderAsset> {
    try_from_file_inner(path, true).0
}

/// When `log_failure` is `false` (directory scan), probe data is returned without per-file [`logger::warn!`].
fn try_from_file_inner(
    path: &Path,
    log_failure: bool,
) -> (Option<ResolvedUnityShaderAsset>, Option<FileBinaryProbe>) {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            logger::warn!(
                "shader_unity_asset: cannot read {:?} for binary probe: {}",
                path.display(),
                e
            );
            return (None, None);
        }
    };

    let mut probe = FileBinaryProbe::new(&bytes);
    if bytes.is_empty() {
        if log_failure {
            probe.warn_short(path, "empty file");
        }
        return (None, Some(probe));
    }
    if bytes.len() > MAX_READ_BYTES {
        if log_failure {
            probe.warn_short(path, "file too large");
        }
        return (None, Some(probe));
    }

    let mut env = Environment::new();
    let _ = env.load_file(path);
    let source = BinarySource::path(path);

    let mut memory_bundle: Option<AssetBundle> = None;
    if env.bundles().get(&source).is_none() {
        match load_bundle_from_memory(bytes) {
            Ok(b) => memory_bundle = Some(b),
            Err(e) => {
                probe.bundle_err = Some(truncate_display(&e, MAX_ERR_LOG_CHARS));
                logger::debug!(
                    "shader_unity_asset: {:?} not an AssetBundle: {}",
                    path.display(),
                    probe.bundle_err.as_deref().unwrap_or("")
                );
            }
        }
    }
    let bundle_ref: Option<&AssetBundle> = env.bundles().get(&source).or(memory_bundle.as_ref());

    if let Some(bundle) = bundle_ref {
        probe.bundle_parse_ok = true;
        probe.bundle_assets = bundle.assets.len();
        log_bundle_parse_debug(path, bundle);
        if let Some(resolved) = shader_resolution_from_bundle(path, bundle) {
            return (Some(resolved), None);
        }
        if log_failure {
            probe.warn_short(path, "AssetBundle: no shader name");
            probe.log_debug_detail();
        }
        return (None, Some(probe));
    }

    if log_failure {
        probe.warn_short(path, "not an AssetBundle");
        probe.log_debug_detail();
    }
    (None, Some(probe))
}

fn log_bundle_parse_debug(path: &Path, bundle: &AssetBundle) {
    logger::debug!(
        "shader_unity_asset: parsed AssetBundle {:?}: {} SerializedFile(s)",
        path.display(),
        bundle.assets.len()
    );
}

fn log_container_resolution(path_id: i64, name: &str, container_asset_path: &str) {
    logger::debug!(
        "shader_unity_asset: Shader path_id={} source=m_Container asset_path={:?} name={:?}",
        path_id,
        container_asset_path,
        name
    );
}

fn log_internal_name_resolution(
    path_id: i64,
    class_id: i32,
    source: InternalNameSource,
    name: &InternalShaderName,
) {
    let source = match source {
        InternalNameSource::MNamePeek => "m_Name_peek",
        InternalNameSource::UnityObjectName => "typetree_m_Name",
        InternalNameSource::UnityObjectNameField => "typetree_name",
        InternalNameSource::UnityObjectParsedForm => "typetree_m_ParsedForm",
        InternalNameSource::UnityObjectScript => "typetree_m_Script",
        InternalNameSource::ShaderLabBytes => "ShaderLab_bytes",
        InternalNameSource::ShaderObjectBytes => "Shader_object_bytes",
    };
    logger::info!(
        "shader_unity_asset: Shader path_id={} class_id={} source={} full_name={:?} stem={:?} variant_bits={:?}",
        path_id,
        class_id,
        source,
        name.full_name,
        name.shader_asset_name,
        name.shader_variant_bits
    );
}

/// Per-file binary probe state for structured failure logs.
struct FileBinaryProbe {
    bytes_len: usize,
    prefix_hex: String,
    prefix_ascii: String,
    bundle_parse_ok: bool,
    bundle_assets: usize,
    bundle_err: Option<String>,
}

impl FileBinaryProbe {
    fn new(bytes: &[u8]) -> Self {
        Self {
            bytes_len: bytes.len(),
            prefix_hex: format_hex_prefix(bytes, 24),
            prefix_ascii: ascii_prefix_hint(bytes, 40),
            bundle_parse_ok: false,
            bundle_assets: 0,
            bundle_err: None,
        }
    }

    /// One short [`logger::warn!`] line; full fields via [`Self::log_debug_detail`].
    fn warn_short(&self, path: &Path, reason: &str) {
        logger::warn!(
            "shader_unity_asset: {:?} -- {} | bytes={} hex8={} | bundle_ok={} | err {:?}",
            path.display(),
            reason,
            self.bytes_len,
            short_hex_prefix(&self.prefix_hex, PROBE_HEX_SHORT),
            self.bundle_parse_ok,
            self.bundle_err.as_deref().unwrap_or("")
        );
    }

    fn log_debug_detail(&self) {
        logger::debug!(
            "shader_unity_asset: probe detail bytes={} prefix_hex={} prefix_ascii={:?} bundle_ok={} bundle_assets={} bundle_err={:?}",
            self.bytes_len,
            self.prefix_hex,
            self.prefix_ascii,
            self.bundle_parse_ok,
            self.bundle_assets,
            self.bundle_err
        );
    }
}

fn short_hex_prefix(space_separated_hex: &str, max_bytes: usize) -> String {
    space_separated_hex
        .split_whitespace()
        .take(max_bytes)
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_hex_prefix(bytes: &[u8], max: usize) -> String {
    bytes
        .iter()
        .take(max)
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn ascii_prefix_hint(bytes: &[u8], max: usize) -> String {
    let take = bytes.iter().copied().take(max).collect::<Vec<u8>>();
    if take.is_empty() {
        return String::new();
    }
    if take
        .iter()
        .all(|b| b.is_ascii_graphic() || matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
    {
        String::from_utf8_lossy(&take).chars().take(40).collect()
    } else {
        String::new()
    }
}

fn truncate_display(err: impl Display, max: usize) -> String {
    let s = err.to_string();
    if s.len() <= max {
        return s;
    }
    format!("{}...", &s[..max.saturating_sub(1)])
}

fn try_from_directory(dir: &Path) -> Option<ResolvedUnityShaderAsset> {
    let read_dir = match std::fs::read_dir(dir) {
        Ok(d) => d,
        Err(e) => {
            logger::warn!(
                "shader_unity_asset: cannot read directory {:?}: {}",
                dir.display(),
                e
            );
            return None;
        }
    };

    let mut paths: Vec<std::path::PathBuf> = read_dir
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();

    let files_total = paths.len();
    if files_total == 0 {
        logger::warn!(
            "shader_unity_asset: directory {:?} contains no regular files (only subdirs or empty); cannot probe Unity binaries here",
            dir.display()
        );
        return None;
    }
    paths.sort_unstable();
    // Prefer typical Unity extensions when scanning a loose directory (extensionless bundles sort last).
    paths.sort_by_key(|p| {
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        match ext.as_str() {
            "asset" | "unity" | "shader" => 0,
            _ => 1,
        }
    });

    let mut examined = 0usize;
    let mut bundle_parse_hits = 0usize;
    let mut first_probe: Option<FileBinaryProbe> = None;

    for (idx, p) in paths.into_iter().enumerate() {
        if idx >= MAX_DIR_FILES {
            break;
        }
        examined += 1;
        logger::debug!(
            "shader_unity_asset: directory {:?} examining [{}/{}] {:?}",
            dir.display(),
            examined,
            files_total.min(MAX_DIR_FILES),
            p.display()
        );
        let (name, probe) = try_from_file_inner(&p, false);
        if let Some(name) = name {
            return Some(name);
        }
        if let Some(probe) = probe {
            if probe.bundle_parse_ok {
                bundle_parse_hits += 1;
            }
            if first_probe.is_none() {
                first_probe = Some(probe);
            }
        }
    }

    logger::warn!(
        "shader_unity_asset: directory {:?} -- no shader name (files_total={} examined={} cap={} bundle_hits={})",
        dir.display(),
        files_total,
        examined,
        MAX_DIR_FILES,
        bundle_parse_hits
    );
    if let Some(ref fp) = first_probe {
        logger::debug!("shader_unity_asset: first failed file probe sample");
        fp.log_debug_detail();
    }

    None
}

fn shader_resolution_from_bundle(
    bundle_path: &Path,
    bundle: &AssetBundle,
) -> Option<ResolvedUnityShaderAsset> {
    let container_name = shader_name_from_bundle_container(bundle_path, bundle);
    let internal_name = shader_internal_name_from_bundle(bundle);
    let shader_asset_name = container_name.or_else(|| {
        internal_name
            .as_ref()
            .map(|name| name.shader_asset_name.clone())
    })?;
    Some(ResolvedUnityShaderAsset {
        shader_asset_name,
        shader_variant_bits: internal_name.and_then(|name| name.shader_variant_bits),
    })
}

fn shader_internal_name_from_bundle(bundle: &AssetBundle) -> Option<InternalShaderName> {
    let mut best: Option<InternalShaderNameCandidate> = None;
    for asset in &bundle.assets {
        if let Some(candidate) = shader_internal_name_from_serialized_file(asset)
            && update_best_internal_name_candidate(&mut best, candidate)
        {
            break;
        }
    }

    let candidate = best?;
    log_internal_name_resolution(
        candidate.path_id,
        candidate.class_id,
        candidate.source,
        &candidate.name,
    );
    Some(candidate.name)
}

fn shader_internal_name_from_serialized_file(
    sf: &SerializedFile,
) -> Option<InternalShaderNameCandidate> {
    let mut best: Option<InternalShaderNameCandidate> = None;
    for handle in sf.object_handles() {
        if handle.class_id() != SHADER {
            continue;
        }
        let path_id = handle.path_id();
        let class_id = handle.class_id();
        match handle.peek_name() {
            Ok(Some(name)) if !name.trim().is_empty() => {
                if let Some(parsed) = parse_internal_shader_name(&name) {
                    update_best_internal_name_candidate(
                        &mut best,
                        InternalShaderNameCandidate {
                            path_id,
                            class_id,
                            source: InternalNameSource::MNamePeek,
                            name: parsed,
                        },
                    );
                    if best_internal_name_has_variant(best.as_ref()) {
                        break;
                    }
                }
            }
            Ok(Some(_)) => {}
            Ok(None) => {
                logger::debug!(
                    "shader_unity_asset: Shader path_id={} peek_name None; typetree read",
                    path_id
                );
            }
            Err(e) => {
                logger::debug!(
                    "shader_unity_asset: Shader path_id={} peek_name err {}; typetree read",
                    path_id,
                    e
                );
            }
        }

        match handle.read() {
            Ok(obj) => {
                let candidates = shader_internal_name_candidates_from_loaded_unity_object(
                    path_id, class_id, &obj,
                );
                for candidate in candidates {
                    if update_best_internal_name_candidate(&mut best, candidate) {
                        break;
                    }
                }
                if best_internal_name_has_variant(best.as_ref()) {
                    break;
                }
                logger::debug!(
                    "shader_unity_asset: Shader path_id={} typetree ok; keys_sample={:?}",
                    path_id,
                    obj.property_names().iter().take(24).collect::<Vec<_>>()
                );
            }
            Err(e) => {
                logger::debug!(
                    "shader_unity_asset: Shader path_id={} ObjectHandle::read failed: {}",
                    path_id,
                    e
                );
            }
        }

        let bytes = match handle.raw_data() {
            Ok(bytes) => bytes,
            Err(e) => {
                logger::debug!(
                    "shader_unity_asset: Shader path_id={} raw_data failed: {}",
                    path_id,
                    e
                );
                continue;
            }
        };
        if let Some((source, parsed)) = find_internal_shader_name_in_bytes(bytes) {
            update_best_internal_name_candidate(
                &mut best,
                InternalShaderNameCandidate {
                    path_id,
                    class_id,
                    source,
                    name: parsed,
                },
            );
            if best_internal_name_has_variant(best.as_ref()) {
                break;
            }
        }
    }
    best
}

fn update_best_internal_name_candidate(
    best: &mut Option<InternalShaderNameCandidate>,
    candidate: InternalShaderNameCandidate,
) -> bool {
    if best.as_ref().is_none_or(|current| {
        internal_name_candidate_rank(&candidate) > internal_name_candidate_rank(current)
    }) {
        *best = Some(candidate);
    }
    best_internal_name_has_variant(best.as_ref())
}

fn best_internal_name_has_variant(candidate: Option<&InternalShaderNameCandidate>) -> bool {
    candidate.is_some_and(|candidate| candidate.name.shader_variant_bits.is_some())
}

fn internal_name_candidate_rank(candidate: &InternalShaderNameCandidate) -> (u8, u8) {
    (
        u8::from(candidate.name.shader_variant_bits.is_some()),
        internal_name_source_rank(candidate.source),
    )
}

fn internal_name_source_rank(source: InternalNameSource) -> u8 {
    match source {
        InternalNameSource::UnityObjectParsedForm
        | InternalNameSource::UnityObjectScript
        | InternalNameSource::ShaderLabBytes
        | InternalNameSource::ShaderObjectBytes => 3,
        InternalNameSource::UnityObjectName | InternalNameSource::UnityObjectNameField => 2,
        InternalNameSource::MNamePeek => 1,
    }
}

fn shader_internal_name_candidates_from_loaded_unity_object(
    path_id: i64,
    class_id: i32,
    obj: &unity_asset_binary::object::UnityObject,
) -> Vec<InternalShaderNameCandidate> {
    let mut candidates = Vec::new();

    for (key, source) in [
        ("m_ParsedForm", InternalNameSource::UnityObjectParsedForm),
        ("m_Script", InternalNameSource::UnityObjectScript),
    ] {
        if let Some(value) = obj.get(key) {
            if source == InternalNameSource::UnityObjectParsedForm
                && let Some(parsed) = parsed_form_internal_shader_name(value)
            {
                candidates.push(InternalShaderNameCandidate {
                    path_id,
                    class_id,
                    source,
                    name: parsed,
                });
            }
            let text = unity_value_searchable_text(value);
            if let Some(parsed) = find_internal_shader_name_in_text(&text) {
                candidates.push(InternalShaderNameCandidate {
                    path_id,
                    class_id,
                    source,
                    name: parsed,
                });
            }
        }
    }

    if let Some(parsed) = obj
        .name()
        .filter(|name| !name.trim().is_empty())
        .and_then(|name| parse_internal_shader_name(&name))
    {
        candidates.push(InternalShaderNameCandidate {
            path_id,
            class_id,
            source: InternalNameSource::UnityObjectName,
            name: parsed,
        });
    }

    if let Some(parsed) = obj
        .get("name")
        .and_then(UnityValue::as_str)
        .filter(|name| !name.trim().is_empty())
        .and_then(parse_internal_shader_name)
    {
        candidates.push(InternalShaderNameCandidate {
            path_id,
            class_id,
            source: InternalNameSource::UnityObjectNameField,
            name: parsed,
        });
    }

    candidates
}

fn parsed_form_internal_shader_name(value: &UnityValue) -> Option<InternalShaderName> {
    let UnityValue::Object(fields) = value else {
        return None;
    };
    ["m_Name", "name"].into_iter().find_map(|key| {
        fields
            .get(key)
            .and_then(UnityValue::as_str)
            .filter(|name| !name.trim().is_empty())
            .and_then(parse_internal_shader_name)
    })
}

fn unity_value_searchable_text(value: &UnityValue) -> String {
    match value {
        UnityValue::Null => String::new(),
        UnityValue::Bool(value) => value.to_string(),
        UnityValue::Integer(value) => value.to_string(),
        UnityValue::Float(value) => value.to_string(),
        UnityValue::String(value) => value.clone(),
        UnityValue::Array(values) => values
            .iter()
            .map(unity_value_searchable_text)
            .collect::<Vec<_>>()
            .join(" "),
        UnityValue::Bytes(bytes) => String::from_utf8_lossy(bytes).into_owned(),
        UnityValue::Object(values) => values
            .values()
            .map(unity_value_searchable_text)
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn find_internal_shader_name_in_bytes(
    data: &[u8],
) -> Option<(InternalNameSource, InternalShaderName)> {
    let text = String::from_utf8_lossy(data);
    if let Some(name) = find_internal_shader_name_in_text(&text) {
        return Some((InternalNameSource::ShaderLabBytes, name));
    }
    find_variant_internal_shader_name_in_text_tokens(&text)
        .map(|name| (InternalNameSource::ShaderObjectBytes, name))
}

fn find_internal_shader_name_in_text(text: &str) -> Option<InternalShaderName> {
    let mut cursor = 0;
    let mut best = None;

    while let Some(relative_index) = text.get(cursor..)?.find("Shader") {
        let index = cursor + relative_index;
        let tail = text.get(index..)?;
        let window: String = tail.chars().take(4096).collect();
        if let Some(parsed) =
            parse_shader_lab_quoted_name(&window).and_then(|name| parse_internal_shader_name(&name))
        {
            if parsed.shader_variant_bits.is_some() {
                return Some(parsed);
            }
            if best.is_none() {
                best = Some(parsed);
            }
        }
        cursor = index + "Shader".len();
    }

    best
}

fn find_variant_internal_shader_name_in_text_tokens(text: &str) -> Option<InternalShaderName> {
    let mut token = String::new();
    for ch in text.chars().chain(std::iter::once('\0')) {
        if is_shader_name_token_char(ch) {
            token.push(ch);
            continue;
        }
        if let Some(parsed) = parse_internal_shader_name(&token)
            && parsed.shader_variant_bits.is_some()
        {
            return Some(parsed);
        }
        token.clear();
    }
    None
}

fn is_shader_name_token_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '/' | '\\' | '-' | '.' | ' ')
}

fn parse_shader_lab_quoted_name(text: &str) -> Option<String> {
    let shader_index = text.find("Shader")?;
    let mut chars = text
        .get(shader_index + "Shader".len()..)?
        .chars()
        .peekable();
    while chars.peek().is_some_and(|c| c.is_whitespace()) {
        chars.next();
    }
    let escaped_outer_quotes = if chars.peek() == Some(&'\\') {
        chars.next();
        if chars.next()? != '"' {
            return None;
        }
        true
    } else if chars.next()? == '"' {
        false
    } else {
        return None;
    };
    let mut name = String::new();
    let mut escaped = false;
    for ch in chars {
        if escaped {
            if escaped_outer_quotes && ch == '"' {
                return (!name.trim().is_empty()).then_some(name);
            }
            name.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return (!name.trim().is_empty()).then_some(name);
        } else {
            name.push(ch);
        }
    }
    None
}

fn parse_internal_shader_name(name: &str) -> Option<InternalShaderName> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return None;
    }
    let (stem, shader_variant_bits) =
        split_variant_suffix(trimmed).map_or((trimmed, None), |(stem, bits)| (stem, Some(bits)));
    let shader_asset_name = shader_asset_stem_from_internal_name(stem)?;
    Some(InternalShaderName {
        full_name: trimmed.to_string(),
        shader_asset_name,
        shader_variant_bits,
    })
}

fn split_variant_suffix(name: &str) -> Option<(&str, u32)> {
    let (stem, suffix) = name.rsplit_once('_')?;
    if stem.trim().is_empty() || suffix.len() != 8 || !suffix.chars().all(|c| c.is_ascii_hexdigit())
    {
        return None;
    }
    u32::from_str_radix(suffix, 16)
        .ok()
        .map(|bits| (stem, bits))
}

fn shader_asset_stem_from_internal_name(name: &str) -> Option<String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return None;
    }
    let base = trimmed
        .rsplit('/')
        .next()
        .and_then(|segment| segment.rsplit('\\').next())
        .unwrap_or(trimmed)
        .trim();
    if base.is_empty() {
        return None;
    }
    Some(base.to_string())
}

/// Shader stem from [`Environment::bundle_container_entries`] by matching Shader `path_id` to
/// `AssetBundle.m_Container`.
fn shader_name_from_bundle_container(bundle_path: &Path, bundle: &AssetBundle) -> Option<String> {
    let mut env = Environment::new();
    let _ = env.load_file(bundle_path);
    let source = BinarySource::path(bundle_path);
    if env.bundles().get(&source).is_none() {
        logger::debug!(
            "shader_unity_asset: Environment has no bundle for {:?} (m_Container unavailable)",
            bundle_path.display()
        );
        return None;
    }
    let entries = env.bundle_container_entries(bundle_path).ok()?;
    if entries.is_empty() {
        logger::debug!(
            "shader_unity_asset: no m_Container entries for {:?}",
            bundle_path.display()
        );
        return None;
    }

    let shader_path_ids: Vec<i64> = bundle
        .assets
        .iter()
        .flat_map(|sf| {
            sf.object_handles()
                .filter(|h| h.class_id() == SHADER)
                .map(|h| h.path_id())
        })
        .collect();

    for pid in shader_path_ids {
        if let Some(entry) = entries.iter().find(|e| e.path_id == pid)
            && let Some(name) = shader_asset_name_from_container_asset_path(&entry.asset_path)
        {
            log_container_resolution(pid, &name, &entry.asset_path);
            return Some(name);
        }
    }
    None
}

/// Derives a shader asset name from a Unity `m_Container` asset path (e.g. `.../ui_unlit.shader` -> `ui_unlit`).
fn shader_asset_name_from_container_asset_path(asset_path: &str) -> Option<String> {
    let p = asset_path.replace('\\', "/");
    let seg = p.rsplit('/').next()?.trim();
    if seg.is_empty() {
        return None;
    }
    let base = seg
        .strip_suffix(".shader")
        .unwrap_or(seg)
        .rsplit('/')
        .next()
        .unwrap_or(seg)
        .trim();
    if base.is_empty() {
        return None;
    }
    let lower = base.to_ascii_lowercase();
    if lower.starts_with("cab-") {
        return None;
    }
    Some(base.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_asset_path_strips_shader_suffix() {
        assert_eq!(
            shader_asset_name_from_container_asset_path("assets/foo/my_shader.shader").as_deref(),
            Some("my_shader")
        );
        assert_eq!(
            shader_asset_name_from_container_asset_path("archive:/CAB-deadbeef").as_deref(),
            None
        );
    }

    #[test]
    fn container_asset_path_handles_backslashes_whitespace_and_plain_stems() {
        assert_eq!(
            shader_asset_name_from_container_asset_path("Assets\\Shaders\\UI Text Unlit.shader")
                .as_deref(),
            Some("UI Text Unlit")
        );
        assert_eq!(
            shader_asset_name_from_container_asset_path("  assets/foo/ToonLit.shader  ").as_deref(),
            Some("ToonLit")
        );
        assert_eq!(
            shader_asset_name_from_container_asset_path("assets/foo/AlreadyStem").as_deref(),
            Some("AlreadyStem")
        );
        assert_eq!(
            shader_asset_name_from_container_asset_path("").as_deref(),
            None
        );
        assert_eq!(
            shader_asset_name_from_container_asset_path("assets/foo/   ").as_deref(),
            None
        );
    }

    #[test]
    fn prefix_formatters_are_stable_for_empty_short_and_truncated_inputs() {
        assert_eq!(format_hex_prefix(&[], 8), "");
        assert_eq!(format_hex_prefix(&[0, 1, 0xab, 0xff], 8), "00 01 ab ff");
        assert_eq!(format_hex_prefix(&[0, 1, 2, 3], 2), "00 01");
        assert_eq!(short_hex_prefix("00 01 02 03", 2), "00 01");
        assert_eq!(short_hex_prefix("00 01", 8), "00 01");
    }

    #[test]
    fn ascii_prefix_hint_only_returns_printable_prefixes() {
        assert_eq!(ascii_prefix_hint(b"", 8), "");
        assert_eq!(ascii_prefix_hint(b"UnityFS\nBundle", 32), "UnityFS\nBundle");
        assert_eq!(ascii_prefix_hint(&[0xff, b'A', b'B'], 32), "");
        assert_eq!(ascii_prefix_hint(b"abcdef", 3), "abc");
    }

    #[test]
    fn truncate_display_preserves_short_errors_and_truncates_long_errors() {
        assert_eq!(truncate_display("short", 16), "short");
        let truncated = truncate_display("abcdefghijklmnopqrstuvwxyz", 8);
        assert_eq!(truncated, "abcdefg...");
    }

    #[test]
    fn internal_shader_name_strips_variant_suffix() {
        assert_eq!(
            parse_internal_shader_name("Unlit_00002202"),
            Some(InternalShaderName {
                full_name: "Unlit_00002202".to_string(),
                shader_asset_name: "Unlit".to_string(),
                shader_variant_bits: Some(0x2202),
            })
        );
        assert_eq!(
            parse_internal_shader_name("Custom/With_Underscore_00000080"),
            Some(InternalShaderName {
                full_name: "Custom/With_Underscore_00000080".to_string(),
                shader_asset_name: "With_Underscore".to_string(),
                shader_variant_bits: Some(0x80),
            })
        );
        assert_eq!(
            parse_internal_shader_name("Unlit_nothex123"),
            Some(InternalShaderName {
                full_name: "Unlit_nothex123".to_string(),
                shader_asset_name: "Unlit_nothex123".to_string(),
                shader_variant_bits: None,
            })
        );
    }

    #[test]
    fn shaderlab_name_parser_finds_variant_stem() {
        assert_eq!(
            find_internal_shader_name_in_text(r#"Shader "Unlit_00000200" { }"#),
            Some(InternalShaderName {
                full_name: "Unlit_00000200".to_string(),
                shader_asset_name: "Unlit".to_string(),
                shader_variant_bits: Some(0x200),
            })
        );
        assert_eq!(
            find_internal_shader_name_in_text(r#"Shader \"Unlit_00000200\" { }"#),
            Some(InternalShaderName {
                full_name: "Unlit_00000200".to_string(),
                shader_asset_name: "Unlit".to_string(),
                shader_variant_bits: Some(0x200),
            })
        );
    }

    #[test]
    fn shaderlab_name_parser_skips_non_declarations() {
        assert_eq!(
            find_internal_shader_name_in_text(r#"SubShader { } Shader "Unlit_00000200" { }"#),
            Some(InternalShaderName {
                full_name: "Unlit_00000200".to_string(),
                shader_asset_name: "Unlit".to_string(),
                shader_variant_bits: Some(0x200),
            })
        );
    }

    #[test]
    fn parsed_form_name_field_is_internal_shader_name() {
        let parsed_form = UnityValue::Object(
            [(
                "m_Name".to_string(),
                UnityValue::String("Unlit_00000200".to_string()),
            )]
            .into_iter()
            .collect(),
        );

        assert_eq!(
            parsed_form_internal_shader_name(&parsed_form),
            Some(InternalShaderName {
                full_name: "Unlit_00000200".to_string(),
                shader_asset_name: "Unlit".to_string(),
                shader_variant_bits: Some(0x200),
            })
        );
    }

    #[test]
    fn raw_shader_bytes_can_fallback_to_variant_name_token() {
        assert_eq!(
            find_internal_shader_name_in_bytes(b"\x0e\0\0\0Unlit_00000200\0"),
            Some((
                InternalNameSource::ShaderObjectBytes,
                InternalShaderName {
                    full_name: "Unlit_00000200".to_string(),
                    shader_asset_name: "Unlit".to_string(),
                    shader_variant_bits: Some(0x200),
                }
            ))
        );
    }

    #[test]
    fn variant_candidate_wins_over_plain_object_name() {
        let mut best = None;
        update_best_internal_name_candidate(
            &mut best,
            InternalShaderNameCandidate {
                path_id: 1,
                class_id: SHADER,
                source: InternalNameSource::MNamePeek,
                name: parse_internal_shader_name("Unlit").expect("plain shader name"),
            },
        );
        update_best_internal_name_candidate(
            &mut best,
            InternalShaderNameCandidate {
                path_id: 1,
                class_id: SHADER,
                source: InternalNameSource::UnityObjectScript,
                name: parse_internal_shader_name("Unlit_00000200").expect("variant shader name"),
            },
        );

        let selected = best.expect("selected shader name");
        assert_eq!(selected.source, InternalNameSource::UnityObjectScript);
        assert_eq!(selected.name.full_name, "Unlit_00000200");
        assert_eq!(selected.name.shader_asset_name, "Unlit");
        assert_eq!(selected.name.shader_variant_bits, Some(0x200));
    }

    #[test]
    fn file_binary_probe_records_prefixes_without_parsing() {
        let probe = FileBinaryProbe::new(b"UnityFS\0binary");
        assert_eq!(probe.bytes_len, 14);
        assert!(probe.prefix_hex.starts_with("55 6e 69 74 79 46 53 00"));
        assert_eq!(probe.prefix_ascii, "");
        assert!(!probe.bundle_parse_ok);
        assert_eq!(probe.bundle_assets, 0);
        assert_eq!(probe.bundle_err, None);
    }

    #[test]
    fn path_hint_rejects_missing_paths_and_empty_directories() {
        let temp = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            try_resolve_shader_asset_name_from_path(&temp.path().join("missing")),
            None
        );
        assert_eq!(try_resolve_shader_asset_name_from_path(temp.path()), None);
    }
}

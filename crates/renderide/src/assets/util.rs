//! Small helpers shared across asset ingestion (shader name normalization, etc.).

/// Normalizes a Unity `Shader "…"` label or path for stable dictionary lookup (whitespace, `/` → `_`, lowercased).
///
/// Shared by shader routing and embedded `{key}_default` stem resolution so lookups stay
/// consistent without import cycles between `assets::shader::route` and materials.
pub fn normalize_unity_shader_lookup_key(name: &str) -> String {
    let token = name.split_whitespace().next().unwrap_or(name).trim();
    token
        .chars()
        .map(|c| {
            if c.is_whitespace() || c == '/' {
                '_'
            } else {
                c.to_ascii_lowercase()
            }
        })
        .collect()
}

/// Normalizes a shader token for comparison: keeps only ASCII alphanumeric characters and folds to lowercase.
///
/// Used when mapping Unity shader names and path hints to compact comparison keys.
pub fn compact_alnum_lower(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

#[cfg(test)]
mod compact_alnum_lower_tests {
    use super::compact_alnum_lower;

    #[test]
    fn strips_non_alnum_and_lowercases() {
        assert_eq!(compact_alnum_lower("Foo/Bar-Baz_1"), "foobarbaz1");
    }

    #[test]
    fn empty_when_no_alnum() {
        assert_eq!(compact_alnum_lower(" /.-_"), "");
    }
}

#[cfg(test)]
mod normalize_unity_shader_lookup_key_tests {
    use super::normalize_unity_shader_lookup_key;

    #[test]
    fn folds_path_separators_and_lowercases_first_token() {
        assert_eq!(
            normalize_unity_shader_lookup_key("Custom/UI/TextUnlit kw1"),
            "custom_ui_textunlit"
        );
    }

    #[test]
    fn empty_input_yields_empty() {
        assert_eq!(normalize_unity_shader_lookup_key(""), "");
        assert_eq!(normalize_unity_shader_lookup_key("   "), "");
    }

    #[test]
    fn preserves_inner_ascii_other_chars() {
        assert_eq!(normalize_unity_shader_lookup_key("Foo-Bar_1"), "foo-bar_1");
    }
}

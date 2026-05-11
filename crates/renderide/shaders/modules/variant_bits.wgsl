//! Helpers for renderer-reserved shader variant bitfields.

#define_import_path renderide::variant_bits

fn enabled(bits: u32, mask: u32) -> bool {
    return (bits & mask) != 0u;
}

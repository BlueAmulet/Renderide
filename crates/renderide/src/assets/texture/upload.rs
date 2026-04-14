//! Applies [`SetTexture2DData`] into an existing [`wgpu::Texture`] using [`wgpu::Queue::write_texture`]
//! ([`wgpu::TexelCopyTextureInfo`] / [`wgpu::TexelCopyBufferLayout`]).
//!
//! When [`crate::shared::TextureUploadHint::has_region`] is set with a non-empty rectangle, mip0 may be
//! uploaded as a sub-rect only (uncompressed RGBA8-family GPU storage, single mip, no `flip_y`).
//! Other cases fall back to the full mip chain path.
//!
//! The [`wgpu::TextureFormat`] must match the texture’s creation format (see [`resolve_texture2d_wgpu_format`]).

use crate::shared::{ColorProfile, SetTexture2DData, SetTexture2DFormat, TextureUploadHint};

use super::decode::{decode_mip_to_rgba8, flip_mip_rows, needs_rgba8_decode_before_upload};
use super::format::pick_wgpu_storage_format;
use super::layout::{
    host_format_is_compressed, mip_byte_len, mip_dimensions_at_level, mip_tight_bytes_per_texel,
};

/// Decides GPU storage format for a new 2D texture from host [`SetTexture2DFormat`].
///
/// Uses native compressed/uncompressed `wgpu` formats when supported; falls back to RGBA8 when
/// compression features are missing or the host layout needs swizzle ([`needs_rgba8_decode_before_upload`]).
pub fn resolve_texture2d_wgpu_format(
    device: &wgpu::Device,
    fmt: &SetTexture2DFormat,
) -> wgpu::TextureFormat {
    if needs_rgba8_decode_before_upload(fmt.format) {
        return rgba8_fallback_format(fmt.profile);
    }
    if let Some(f) = pick_wgpu_storage_format(device, fmt.format, fmt.profile) {
        return f;
    }
    rgba8_fallback_format(fmt.profile)
}

fn rgba8_fallback_format(profile: ColorProfile) -> wgpu::TextureFormat {
    match profile {
        ColorProfile::SRGB | ColorProfile::SRGBAlpha => wgpu::TextureFormat::Rgba8UnormSrgb,
        ColorProfile::Linear => wgpu::TextureFormat::Rgba8Unorm,
    }
}

/// Describes a sub-rectangle within a full mip for tight row-major extraction (uncompressed).
struct MipSubrectCopy {
    /// Full mip width in texels.
    full_width: u32,
    /// Full mip height in texels.
    full_height: u32,
    /// Bytes per texel in the host mip slice.
    bpp: usize,
    /// Sub-rectangle min X in texels.
    x: u32,
    /// Sub-rectangle min Y in texels.
    y: u32,
    /// Sub-rectangle width in texels.
    w: u32,
    /// Sub-rectangle height in texels.
    h: u32,
}

/// Matches `TextureUploadHint.IsEmptyRegion` in `Renderite.Shared`.
fn hint_region_is_empty(hint: &TextureUploadHint) -> bool {
    if hint.has_region == 0 {
        return false;
    }
    if hint.region_data.width != 0 {
        return hint.region_data.height == 0;
    }
    true
}

/// Packs a tight row-major buffer for `write_texture` from a rectangular sub-region of a full mip.
fn pack_subrect_tight(full: &[u8], r: &MipSubrectCopy) -> Result<Vec<u8>, String> {
    let row_stride = (r.full_width as usize)
        .checked_mul(r.bpp)
        .ok_or_else(|| "row_stride overflow".to_string())?;
    let row_len = (r.w as usize)
        .checked_mul(r.bpp)
        .ok_or_else(|| "row_len overflow".to_string())?;
    let total = row_len
        .checked_mul(r.h as usize)
        .ok_or_else(|| "subrect total bytes overflow".to_string())?;
    let mut out = Vec::new();
    out.try_reserve(total).map_err(|e| e.to_string())?;
    for row in 0..r.h {
        let y = r.y + row;
        if y >= r.full_height {
            return Err("subrect row out of bounds".into());
        }
        let row_start = (y as usize)
            .checked_mul(row_stride)
            .and_then(|b| b.checked_add((r.x as usize).checked_mul(r.bpp)?))
            .ok_or_else(|| "row_start overflow".to_string())?;
        let end = row_start
            .checked_add(row_len)
            .ok_or_else(|| "row_end overflow".to_string())?;
        if end > full.len() {
            return Err("subrect row extends past mip buffer".into());
        }
        out.extend_from_slice(&full[row_start..end]);
    }
    Ok(out)
}

/// Parameters for [`write_texture_subregion`] (partial [`wgpu::Queue::write_texture`]).
struct TextureWriteSubregion<'a> {
    /// Submission queue for the copy.
    queue: &'a wgpu::Queue,
    /// Destination texture.
    texture: &'a wgpu::Texture,
    /// Mip level to write.
    mip_level: u32,
    /// Destination origin X in texels.
    origin_x: u32,
    /// Destination origin Y in texels.
    origin_y: u32,
    /// Copy width in texels.
    width: u32,
    /// Copy height in texels.
    height: u32,
    /// Storage format of `texture`.
    format: wgpu::TextureFormat,
    /// Tightly packed texel bytes for the sub-rectangle.
    bytes: &'a [u8],
}

/// Writes a tight sub-rectangle of texels into `texture` at the given mip and origin.
fn write_texture_subregion(w: TextureWriteSubregion<'_>) -> Result<(), String> {
    let queue = w.queue;
    let texture = w.texture;
    let mip_level = w.mip_level;
    let origin_x = w.origin_x;
    let origin_y = w.origin_y;
    let width = w.width;
    let height = w.height;
    let format = w.format;
    let bytes = w.bytes;
    let (bw, bh) = format.block_dimensions();
    let copy_width = if bw > 1 {
        width.div_ceil(bw) * bw
    } else {
        width
    };
    let copy_height = if bh > 1 {
        height.div_ceil(bh) * bh
    } else {
        height
    };
    let size = wgpu::Extent3d {
        width: copy_width,
        height: copy_height,
        depth_or_array_layers: 1,
    };
    let (layout, expected_len) = copy_layout_for_mip(format, width, height)?;
    if bytes.len() != expected_len {
        return Err(format!(
            "subregion mip data len {} != expected {} ({}x{} {:?})",
            bytes.len(),
            expected_len,
            width,
            height,
            format
        ));
    }

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: wgpu::Origin3d {
                x: origin_x,
                y: origin_y,
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        layout,
        size,
    );
    Ok(())
}

/// Sub-rect upload for mip0 when the host sets [`TextureUploadHint::has_region`].
///
/// Returns [`None`] when the fast path does not apply (caller uses the full mip chain path).
fn try_write_texture2d_subregion(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    fmt: &SetTexture2DFormat,
    wgpu_format: wgpu::TextureFormat,
    upload: &SetTexture2DData,
    raw: &[u8],
) -> Option<Result<u32, String>> {
    if upload.flip_y {
        return None;
    }
    if upload.start_mip_level != 0 {
        return None;
    }
    if upload.mip_map_sizes.len() != 1 || upload.mip_starts.len() != 1 {
        return None;
    }
    if host_format_is_compressed(fmt.format) || needs_rgba8_decode_before_upload(fmt.format) {
        return None;
    }
    if !is_rgba8_family(wgpu_format) {
        return None;
    }

    let want = upload.data.length.max(0) as usize;
    if raw.len() < want {
        return Some(Err(format!(
            "raw shorter than descriptor (need {want}, got {})",
            raw.len()
        )));
    }
    let payload = &raw[..want];

    let w = upload.mip_map_sizes[0].x.max(0) as u32;
    let h = upload.mip_map_sizes[0].y.max(0) as u32;
    if w == 0 || h == 0 {
        return Some(Err("non-positive mip dimensions".into()));
    }

    let tex_extent = texture.size();
    if tex_extent.width != w || tex_extent.height != h {
        return None;
    }

    let (start_bias, _) = match choose_mip_start_bias(fmt.format, upload, payload.len()) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };

    let start_raw = upload.mip_starts[0];
    if start_raw < 0 {
        return Some(Err("negative mip_starts".into()));
    }
    let start_abs = start_raw as usize;
    if start_abs < start_bias {
        return Some(Err(format!(
            "mip 0 start {} is before descriptor offset {}",
            start_abs, start_bias
        )));
    }
    let start = start_abs - start_bias;
    let host_len = match mip_byte_len(fmt.format, w, h) {
        Some(l) => l as usize,
        None => {
            return Some(Err(format!(
                "mip byte size unsupported for {:?}",
                fmt.format
            )));
        }
    };
    let mip_src = match payload.get(start..start + host_len) {
        Some(s) => s,
        None => return Some(Err("mip 0 slice out of range".into())),
    };

    let bpp = mip_tight_bytes_per_texel(mip_src.len(), w, h)?;
    if bpp != 4 {
        return None;
    }

    let rx = upload.hint.region_data.x.max(0) as u32;
    let ry = upload.hint.region_data.y.max(0) as u32;
    let rw = upload.hint.region_data.width.max(0) as u32;
    let rh = upload.hint.region_data.height.max(0) as u32;
    if rw == 0 || rh == 0 {
        return Some(Err("region width/height must be positive".into()));
    }
    if rx.saturating_add(rw) > w || ry.saturating_add(rh) > h {
        return Some(Err(format!(
            "region {rw}x{rh} at ({rx}, {ry}) out of bounds for mip {w}x{h}",
        )));
    }

    let packed = match pack_subrect_tight(
        mip_src,
        &MipSubrectCopy {
            full_width: w,
            full_height: h,
            bpp,
            x: rx,
            y: ry,
            w: rw,
            h: rh,
        },
    ) {
        Ok(p) => p,
        Err(e) => return Some(Err(e)),
    };
    match write_texture_subregion(TextureWriteSubregion {
        queue,
        texture,
        mip_level: 0,
        origin_x: rx,
        origin_y: ry,
        width: rw,
        height: rh,
        format: wgpu_format,
        bytes: &packed,
    }) {
        Ok(()) => Some(Ok(1)),
        Err(e) => Some(Err(e)),
    }
}

/// Uploads mips from `raw` (exact [`SharedMemoryBufferDescriptor`] window) into `texture` using `wgpu_format`.
pub fn write_texture2d_mips(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    fmt: &SetTexture2DFormat,
    wgpu_format: wgpu::TextureFormat,
    upload: &SetTexture2DData,
    raw: &[u8],
) -> Result<u32, String> {
    if upload.hint.has_region != 0 {
        if hint_region_is_empty(&upload.hint) {
            logger::trace!(
                "texture {}: TextureUploadHint.has_region set but region empty; skipping upload",
                upload.asset_id
            );
            return Ok(0);
        }
        match try_write_texture2d_subregion(queue, texture, fmt, wgpu_format, upload, raw) {
            Some(Ok(n)) => {
                logger::trace!(
                    "texture {}: sub-region texture upload ({} mips equivalent)",
                    upload.asset_id,
                    n
                );
                return Ok(n);
            }
            Some(Err(e)) => return Err(e),
            None => {
                logger::trace!(
                    "texture {}: TextureUploadHint.has_region set; using full mip upload path",
                    upload.asset_id
                );
            }
        }
    }
    let want = upload.data.length.max(0) as usize;
    if raw.len() < want {
        return Err(format!(
            "raw shorter than descriptor (need {want}, got {})",
            raw.len()
        ));
    }
    let payload = &raw[..want];

    let start_base = upload.start_mip_level.max(0) as u32;
    let mipmap_count = fmt.mipmap_count.max(1) as u32;
    if start_base >= mipmap_count {
        return Err(format!(
            "start_mip_level {start_base} >= mipmap_count {mipmap_count}"
        ));
    }

    let flip = upload.flip_y;
    if flip && host_format_is_compressed(fmt.format) && !is_rgba8_family(wgpu_format) {
        logger::warn!(
            "texture {}: flip_y unsupported for compressed GPU texture {:?}; mips may look upside-down",
            upload.asset_id,
            wgpu_format
        );
    }

    let tex_extent = texture.size();
    let fmt_w = fmt.width.max(0) as u32;
    let fmt_h = fmt.height.max(0) as u32;
    if tex_extent.width != fmt_w || tex_extent.height != fmt_h {
        return Err(format!(
            "GPU texture {}x{} does not match SetTexture2DFormat {}x{} for asset {}",
            tex_extent.width, tex_extent.height, fmt_w, fmt_h, upload.asset_id
        ));
    }

    if upload.mip_map_sizes.len() != upload.mip_starts.len() {
        return Err("mip_map_sizes and mip_starts length mismatch".into());
    }
    if upload.mip_map_sizes.is_empty() {
        return Err("no mips in upload".into());
    }

    let (start_bias, valid_prefix_mips) = choose_mip_start_bias(fmt.format, upload, payload.len())?;
    if start_bias != 0 {
        logger::debug!(
            "texture {}: rebasing mip_starts by descriptor offset {}",
            upload.asset_id,
            start_bias
        );
    }

    let mut uploaded_mips = 0u32;
    for (i, sz) in upload.mip_map_sizes.iter().enumerate() {
        let w = sz.x.max(0) as u32;
        let h = sz.y.max(0) as u32;
        let mip_level = start_base + i as u32;
        if mip_level >= mipmap_count {
            return Err(format!(
                "upload mip {mip_level} exceeds texture mips {mipmap_count}"
            ));
        }

        let (gw, gh) = mip_dimensions_at_level(tex_extent.width, tex_extent.height, mip_level);
        if w != gw || h != gh {
            return Err(format!(
                "texture {} mip {mip_level}: upload says {w}x{h} but GPU mip is {gw}x{gh} (base {}x{} from format); fix host SetTexture2DFormat vs SetTexture2DData",
                upload.asset_id,
                tex_extent.width,
                tex_extent.height
            ));
        }

        let start_raw = upload.mip_starts[i];
        if start_raw < 0 {
            if uploaded_mips == 0 {
                return Err("negative mip_starts".into());
            }
            logger::warn!(
                "texture {}: uploaded {uploaded_mips}/{} mips; stopping at mip {} because mip_starts is negative",
                upload.asset_id,
                upload.mip_map_sizes.len(),
                i
            );
            break;
        }
        let start_abs = start_raw as usize;
        if start_abs < start_bias {
            if uploaded_mips == 0 {
                return Err(format!(
                    "mip 0 start {} is before descriptor offset {}",
                    start_abs, start_bias
                ));
            }
            logger::warn!(
                "texture {}: uploaded {uploaded_mips}/{} mips; stopping at mip {} because start {} is before descriptor offset {}",
                upload.asset_id,
                upload.mip_map_sizes.len(),
                i,
                start_abs,
                start_bias
            );
            break;
        }
        let start = start_abs - start_bias;
        let host_len = mip_byte_len(fmt.format, w, h)
            .ok_or_else(|| format!("mip byte size unsupported for {:?}", fmt.format))?
            as usize;
        let Some(mip_src) = payload.get(start..start + host_len) else {
            if uploaded_mips == 0 {
                return Err(format!(
                    "mip 0 slice out of range after rebasing by {start_bias} (payload_len={}, valid_prefix_mips={valid_prefix_mips})",
                    payload.len()
                ));
            }
            logger::warn!(
                "texture {}: uploaded {uploaded_mips}/{} mips; stopping at mip {} because payload_len={} does not cover start={} len={} after rebasing by {}",
                upload.asset_id,
                upload.mip_map_sizes.len(),
                i,
                payload.len(),
                start,
                host_len,
                start_bias
            );
            break;
        };

        let pixels: std::borrow::Cow<'_, [u8]> = if is_rgba8_family(wgpu_format) {
            if needs_rgba8_decode_before_upload(fmt.format) || host_format_is_compressed(fmt.format)
            {
                std::borrow::Cow::Owned(
                    decode_mip_to_rgba8(fmt.format, w, h, flip, mip_src).ok_or_else(|| {
                        format!("RGBA decode failed for mip {i} ({:?})", fmt.format)
                    })?,
                )
            } else if flip {
                let mut v = mip_src.to_vec();
                let bpp = mip_tight_bytes_per_texel(v.len(), w, h).ok_or_else(|| {
                    format!(
                        "mip {i}: RGBA8 upload len {} not divisible by {}×{} texels",
                        v.len(),
                        w,
                        h
                    )
                })?;
                if bpp != 4 {
                    return Err(format!(
                        "mip {i}: RGBA8 family expects 4 bytes per texel, got {bpp}"
                    ));
                }
                flip_mip_rows(&mut v, w, h, bpp);
                std::borrow::Cow::Owned(v)
            } else {
                std::borrow::Cow::Borrowed(mip_src)
            }
        } else {
            if needs_rgba8_decode_before_upload(fmt.format) {
                return Err(format!(
                    "host {:?} must use RGBA decode but GPU format is {:?}",
                    fmt.format, wgpu_format
                ));
            }
            if flip && !host_format_is_compressed(fmt.format) {
                let mut v = mip_src.to_vec();
                let bpp_host = mip_tight_bytes_per_texel(v.len(), w, h).ok_or_else(|| {
                    format!(
                        "mip {i}: len {} not divisible by {}×{} texels (cannot infer row stride for flip_y)",
                        v.len(),
                        w,
                        h
                    )
                })?;
                if let Ok(bpp_gpu) = uncompressed_row_bytes(wgpu_format) {
                    if bpp_host != bpp_gpu {
                        logger::warn!(
                            "texture {} mip {i}: host texel stride {} B != GPU {:?} stride {} B; flip_y uses host packing",
                            upload.asset_id,
                            bpp_host,
                            wgpu_format,
                            bpp_gpu
                        );
                    }
                }
                flip_mip_rows(&mut v, w, h, bpp_host);
                std::borrow::Cow::Owned(v)
            } else {
                if flip && host_format_is_compressed(fmt.format) {
                    logger::warn!(
                        "texture {} mip {i}: flip_y skipped for compressed {:?} GPU upload",
                        upload.asset_id,
                        wgpu_format
                    );
                }
                std::borrow::Cow::Borrowed(mip_src)
            }
        };

        write_one_mip(
            queue,
            texture,
            mip_level,
            w,
            h,
            wgpu_format,
            pixels.as_ref(),
        )?;
        uploaded_mips += 1;
    }

    if uploaded_mips == 0 {
        return Err("no mip levels uploaded".into());
    }
    Ok(uploaded_mips)
}

fn choose_mip_start_bias(
    format: crate::shared::TextureFormat,
    upload: &SetTexture2DData,
    payload_len: usize,
) -> Result<(usize, usize), String> {
    let offset_bias = upload.data.offset.max(0) as usize;
    let candidates = if offset_bias > 0 {
        [0usize, offset_bias]
    } else {
        [0usize, 0usize]
    };
    let mut best_bias = 0usize;
    let mut best_prefix = 0usize;
    for bias in candidates {
        let prefix = valid_mip_prefix_len(format, upload, payload_len, bias)?;
        if prefix > best_prefix {
            best_prefix = prefix;
            best_bias = bias;
        }
    }
    if best_prefix == 0 {
        return Err(format!(
            "mip region exceeds shared memory descriptor (payload_len={}, descriptor_offset={})",
            payload_len, offset_bias
        ));
    }
    Ok((best_bias, best_prefix))
}

fn valid_mip_prefix_len(
    format: crate::shared::TextureFormat,
    upload: &SetTexture2DData,
    payload_len: usize,
    bias: usize,
) -> Result<usize, String> {
    let mut count = 0usize;
    for (i, sz) in upload.mip_map_sizes.iter().enumerate() {
        if sz.x <= 0 || sz.y <= 0 {
            return Err("non-positive mip dimensions".into());
        }
        let w = sz.x as u32;
        let h = sz.y as u32;
        let host_len = mip_byte_len(format, w, h)
            .ok_or_else(|| format!("mip byte size unsupported for {:?}", format))?
            as usize;
        let start_raw = upload.mip_starts[i];
        if start_raw < 0 {
            return Err("negative mip_starts".into());
        }
        let start_abs = start_raw as usize;
        if start_abs < bias {
            break;
        }
        let start = start_abs - bias;
        if start
            .checked_add(host_len)
            .is_none_or(|end| end > payload_len)
        {
            break;
        }
        count += 1;
    }
    Ok(count)
}

fn is_rgba8_family(gpu: wgpu::TextureFormat) -> bool {
    matches!(
        gpu,
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
    )
}

fn uncompressed_row_bytes(f: wgpu::TextureFormat) -> Result<usize, String> {
    let (bw, bh) = f.block_dimensions();
    if bw != 1 || bh != 1 {
        return Err("internal: expected uncompressed format".into());
    }
    let bsz = f
        .block_copy_size(None)
        .ok_or_else(|| format!("wgpu format {f:?} has no block size"))?;
    Ok(bsz as usize)
}

fn write_one_mip(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip_level: u32,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    bytes: &[u8],
) -> Result<(), String> {
    // For block-compressed formats wgpu requires the copy extent to be a multiple of the
    // block dimensions (the "physical" mip size).  The data produced by copy_layout_for_mip
    // already covers the padded block grid (via div_ceil), so only the Extent3d needs aligning.
    let (bw, bh) = format.block_dimensions();
    let copy_width = if bw > 1 {
        width.div_ceil(bw) * bw
    } else {
        width
    };
    let copy_height = if bh > 1 {
        height.div_ceil(bh) * bh
    } else {
        height
    };
    let size = wgpu::Extent3d {
        width: copy_width,
        height: copy_height,
        depth_or_array_layers: 1,
    };
    let (layout, expected_len) = copy_layout_for_mip(format, width, height)?;
    if bytes.len() != expected_len {
        return Err(format!(
            "mip data len {} != expected {} ({}x{} {:?})",
            bytes.len(),
            expected_len,
            width,
            height,
            format
        ));
    }

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        layout,
        size,
    );
    Ok(())
}

fn copy_layout_for_mip(
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> Result<(wgpu::TexelCopyBufferLayout, usize), String> {
    let (bw, bh) = format.block_dimensions();
    let block_bytes = format
        .block_copy_size(None)
        .ok_or_else(|| format!("no block copy size for {:?}", format))?;
    if bw == 1 && bh == 1 {
        let bpp = block_bytes as usize;
        let bpr = bpp
            .checked_mul(width as usize)
            .ok_or("bytes_per_row overflow")?;
        let expected = bpr
            .checked_mul(height as usize)
            .ok_or("expected bytes overflow")?;
        let bpr_u32 = u32::try_from(bpr).map_err(|_| "bpr u32 overflow")?;
        return Ok((
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr_u32),
                rows_per_image: Some(height),
            },
            expected,
        ));
    }

    let blocks_x = width.div_ceil(bw);
    let blocks_y = height.div_ceil(bh);
    let row_bytes_u = blocks_x
        .checked_mul(block_bytes)
        .ok_or("row bytes overflow")?;
    let expected_u = row_bytes_u
        .checked_mul(blocks_y)
        .ok_or("expected size overflow")?;
    let expected = expected_u as usize;
    Ok((
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row_bytes_u),
            rows_per_image: Some(blocks_y),
        },
        expected,
    ))
}

#[cfg(test)]
mod tests {
    use glam::IVec2;

    use super::{
        choose_mip_start_bias, hint_region_is_empty, pack_subrect_tight, valid_mip_prefix_len,
        MipSubrectCopy,
    };
    use crate::shared::{SetTexture2DData, TextureFormat, TextureUploadHint};

    #[test]
    fn relative_mip_starts_need_no_rebase() {
        let mut upload = SetTexture2DData::default();
        upload.data.length = 80;
        upload.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
        upload.mip_starts = vec![0, 64];

        let (bias, prefix) = choose_mip_start_bias(TextureFormat::RGBA32, &upload, 80).unwrap();
        assert_eq!(bias, 0);
        assert_eq!(prefix, 2);
    }

    #[test]
    fn absolute_mip_starts_rebase_to_descriptor_offset() {
        let mut upload = SetTexture2DData::default();
        upload.data.offset = 128;
        upload.data.length = 80;
        upload.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
        upload.mip_starts = vec![128, 192];

        let (bias, prefix) = choose_mip_start_bias(TextureFormat::RGBA32, &upload, 80).unwrap();
        assert_eq!(bias, 128);
        assert_eq!(prefix, 2);
    }

    #[test]
    fn valid_prefix_len_stops_when_later_mip_exceeds_payload() {
        let mut upload = SetTexture2DData::default();
        upload.data.length = 68;
        upload.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
        upload.mip_starts = vec![0, 64];

        let prefix = valid_mip_prefix_len(TextureFormat::RGBA32, &upload, 68, 0).unwrap();
        assert_eq!(prefix, 1);
    }

    #[test]
    fn hint_region_empty_matches_shared_semantics() {
        let mut h = TextureUploadHint::default();
        assert!(!hint_region_is_empty(&h));
        h.has_region = 1;
        h.region_data.width = 0;
        h.region_data.height = 0;
        assert!(hint_region_is_empty(&h));
        h.region_data.width = 10;
        h.region_data.height = 0;
        assert!(hint_region_is_empty(&h));
        h.region_data.height = 10;
        assert!(!hint_region_is_empty(&h));
    }

    #[test]
    fn pack_subrect_tight_extracts_top_left() {
        let mut v = vec![0u8; 4 * 4 * 4];
        for y in 0..2 {
            for x in 0..2 {
                let i = (y * 4 + x) * 4;
                v[i..i + 4].fill(1);
            }
        }
        let out = pack_subrect_tight(
            &v,
            &MipSubrectCopy {
                full_width: 4,
                full_height: 4,
                bpp: 4,
                x: 0,
                y: 0,
                w: 2,
                h: 2,
            },
        )
        .unwrap();
        assert_eq!(out.len(), 16);
        assert!(out.iter().all(|&b| b == 1));
    }
}

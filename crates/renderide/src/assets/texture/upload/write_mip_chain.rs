//! Full mip chain path: decode, optional flip, [`super::mip_write_common::write_one_mip`] per level.

use crate::shared::{SetTexture2DData, SetTexture2DFormat};

use super::super::decode::{decode_mip_to_rgba8, flip_mip_rows, needs_rgba8_decode_before_upload};
use super::super::layout::{
    host_format_is_compressed, mip_byte_len, mip_dimensions_at_level, mip_tight_bytes_per_texel,
};
use super::mip_write_common::{
    choose_mip_start_bias, is_rgba8_family, uncompressed_row_bytes, write_one_mip,
};
use super::subregion::{hint_region_is_empty, try_write_texture2d_subregion};

/// Incremental full mip-chain upload: call [`Self::upload_next_mip`] until [`MipChainAdvance::Finished`].
#[derive(Debug)]
pub struct TextureMipChainUploader {
    next_i: usize,
    uploaded_mips: u32,
    start_bias: usize,
    start_base: u32,
    mipmap_count: u32,
    tex_extent: wgpu::Extent3d,
    flip: bool,
    stopped: bool,
    generating_tail: bool,
    last_rgba8_mip: Option<Rgba8Mip>,
}

/// Result of one [`TextureMipChainUploader::upload_next_mip`] step.
#[derive(Debug)]
pub enum MipChainAdvance {
    /// Uploaded or generated a single mip; call again for the next level (same `payload` slice).
    UploadedOne {
        /// Total mips successfully written in this chain.
        total_uploaded: u32,
    },
    /// Chain complete (`total_uploaded` mips in this chain).
    Finished {
        /// Total mips successfully written in this chain.
        total_uploaded: u32,
    },
}

#[derive(Clone, Debug)]
struct Rgba8Mip {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl TextureMipChainUploader {
    /// Validates `raw` / `upload` / `fmt` against `texture` and prepares chain state (no GPU work).
    pub fn new(
        texture: &wgpu::Texture,
        fmt: &SetTexture2DFormat,
        upload: &SetTexture2DData,
        raw: &[u8],
    ) -> Result<Self, String> {
        let want = upload.data.length.max(0) as usize;
        if raw.len() < want {
            return Err(format!(
                "raw shorter than descriptor (need {want}, got {})",
                raw.len()
            ));
        }

        let start_base = upload.start_mip_level.max(0) as u32;
        let mipmap_count = fmt.mipmap_count.max(1) as u32;
        if start_base >= mipmap_count {
            return Err(format!(
                "start_mip_level {start_base} >= mipmap_count {mipmap_count}"
            ));
        }

        let flip = upload.flip_y;

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

        let payload_len = want;
        let (start_bias, _valid_prefix_mips) =
            choose_mip_start_bias(fmt.format, upload, payload_len)?;
        if start_bias != 0 {
            logger::debug!(
                "texture {}: rebasing mip_starts by descriptor offset {}",
                upload.asset_id,
                start_bias
            );
        }

        Ok(Self {
            next_i: 0,
            uploaded_mips: 0,
            start_bias,
            start_base,
            mipmap_count,
            tex_extent,
            flip,
            stopped: false,
            generating_tail: false,
            last_rgba8_mip: None,
        })
    }

    /// Writes at most one mip level. `payload` must be `&raw[..upload.data.length]` for the same mapping as `new`.
    pub fn upload_next_mip(
        &mut self,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        fmt: &SetTexture2DFormat,
        wgpu_format: wgpu::TextureFormat,
        upload: &SetTexture2DData,
        payload: &[u8],
    ) -> Result<MipChainAdvance, String> {
        if self.stopped {
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }

        let flip = self.flip;
        if flip && host_format_is_compressed(fmt.format) && !is_rgba8_family(wgpu_format) {
            logger::warn!(
                "texture {}: flip_y unsupported for compressed GPU texture {:?}; mips may look upside-down",
                upload.asset_id,
                wgpu_format
            );
        }

        let tex_extent = self.tex_extent;
        let start_base = self.start_base;
        let mipmap_count = self.mipmap_count;
        let start_bias = self.start_bias;
        let (_bias_check, valid_prefix_mips) =
            choose_mip_start_bias(fmt.format, upload, payload.len())?;
        debug_assert_eq!(start_bias, _bias_check);

        let i = self.next_i;
        if i >= upload.mip_map_sizes.len() {
            if self.uploaded_mips == 0 {
                return Err("no mip levels uploaded".into());
            }
            return self.upload_generated_tail_mip(queue, texture, wgpu_format, upload);
        }

        let sz = upload.mip_map_sizes[i];
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
            if self.uploaded_mips == 0 {
                return Err("negative mip_starts".into());
            }
            if !self.generating_tail {
                logger::warn!(
                    "texture {}: uploaded {}/{} mips; synthesizing tail at mip {} because mip_starts is negative",
                    upload.asset_id,
                    self.uploaded_mips,
                    upload.mip_map_sizes.len(),
                    i
                );
                self.generating_tail = true;
            }
            return self.upload_generated_tail_mip(queue, texture, wgpu_format, upload);
        }
        let start_abs = start_raw as usize;
        if start_abs < start_bias {
            if self.uploaded_mips == 0 {
                return Err(format!(
                    "mip 0 start {} is before descriptor offset {}",
                    start_abs, start_bias
                ));
            }
            if !self.generating_tail {
                logger::warn!(
                    "texture {}: uploaded {}/{} mips; synthesizing tail at mip {} because start {} is before descriptor offset {}",
                    upload.asset_id,
                    self.uploaded_mips,
                    upload.mip_map_sizes.len(),
                    i,
                    start_abs,
                    start_bias
                );
                self.generating_tail = true;
            }
            return self.upload_generated_tail_mip(queue, texture, wgpu_format, upload);
        }
        let start = start_abs - start_bias;
        let host_len = mip_byte_len(fmt.format, w, h)
            .ok_or_else(|| format!("mip byte size unsupported for {:?}", fmt.format))?
            as usize;
        let Some(mip_src) = payload.get(start..start + host_len) else {
            if self.uploaded_mips == 0 {
                return Err(format!(
                    "mip 0 slice out of range after rebasing by {start_bias} (payload_len={}, valid_prefix_mips={valid_prefix_mips})",
                    payload.len()
                ));
            }
            if !self.generating_tail {
                logger::warn!(
                    "texture {}: uploaded {}/{} mips; synthesizing tail at mip {} because payload_len={} does not cover start={} len={} after rebasing by {}",
                    upload.asset_id,
                    self.uploaded_mips,
                    upload.mip_map_sizes.len(),
                    i,
                    payload.len(),
                    start,
                    host_len,
                    start_bias
                );
                self.generating_tail = true;
            }
            return self.upload_generated_tail_mip(queue, texture, wgpu_format, upload);
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
        if is_rgba8_family(wgpu_format) {
            self.last_rgba8_mip = Some(Rgba8Mip {
                width: w,
                height: h,
                pixels: pixels.as_ref().to_vec(),
            });
        }
        self.uploaded_mips += 1;
        self.next_i += 1;

        if self.start_base + self.next_i as u32 >= self.mipmap_count {
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }

        Ok(MipChainAdvance::UploadedOne {
            total_uploaded: self.uploaded_mips,
        })
    }

    fn upload_generated_tail_mip(
        &mut self,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        wgpu_format: wgpu::TextureFormat,
        upload: &SetTexture2DData,
    ) -> Result<MipChainAdvance, String> {
        let mip_level = self.start_base + self.next_i as u32;
        if mip_level >= self.mipmap_count {
            self.stopped = true;
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }

        if !is_rgba8_family(wgpu_format) {
            self.stopped = true;
            logger::warn!(
                "texture {}: uploaded {}/{} mips; cannot synthesize remaining tail for GPU format {:?}",
                upload.asset_id,
                self.uploaded_mips,
                self.mipmap_count.saturating_sub(self.start_base),
                wgpu_format
            );
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }

        let Some(source) = self.last_rgba8_mip.as_ref() else {
            self.stopped = true;
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        };

        if !self.generating_tail {
            logger::warn!(
                "texture {}: uploaded {}/{} host mips; synthesizing missing RGBA8 tail mips",
                upload.asset_id,
                self.uploaded_mips,
                self.mipmap_count.saturating_sub(self.start_base)
            );
            self.generating_tail = true;
        }

        let (w, h) =
            mip_dimensions_at_level(self.tex_extent.width, self.tex_extent.height, mip_level);
        let pixels = downsample_rgba8_box(&source.pixels, source.width, source.height, w, h)?;
        write_one_mip(queue, texture, mip_level, w, h, wgpu_format, &pixels)?;
        self.last_rgba8_mip = Some(Rgba8Mip {
            width: w,
            height: h,
            pixels,
        });
        self.uploaded_mips += 1;
        self.next_i += 1;

        if self.start_base + self.next_i as u32 >= self.mipmap_count {
            self.stopped = true;
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }

        Ok(MipChainAdvance::UploadedOne {
            total_uploaded: self.uploaded_mips,
        })
    }
}

fn downsample_rgba8_box(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<Vec<u8>, String> {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return Err("zero-sized RGBA8 mip".into());
    }
    let expected = (src_w as usize)
        .checked_mul(src_h as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or_else(|| "RGBA8 mip byte size overflow".to_string())?;
    if src.len() != expected {
        return Err(format!(
            "RGBA8 mip len {} != expected {} ({}x{})",
            src.len(),
            expected,
            src_w,
            src_h
        ));
    }

    let dst_len = (dst_w as usize)
        .checked_mul(dst_h as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or_else(|| "RGBA8 target mip byte size overflow".to_string())?;
    let mut out = vec![0u8; dst_len];
    let sw = src_w as usize;
    let sh = src_h as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;

    for dy in 0..dh {
        let y0 = dy * sh / dh;
        let y1 = ((dy + 1) * sh).div_ceil(dh).max(y0 + 1).min(sh);
        for dx in 0..dw {
            let x0 = dx * sw / dw;
            let x1 = ((dx + 1) * sw).div_ceil(dw).max(x0 + 1).min(sw);
            let mut sum = [0u32; 4];
            let mut count = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let si = (sy * sw + sx) * 4;
                    sum[0] += u32::from(src[si]);
                    sum[1] += u32::from(src[si + 1]);
                    sum[2] += u32::from(src[si + 2]);
                    sum[3] += u32::from(src[si + 3]);
                    count += 1;
                }
            }
            let di = (dy * dw + dx) * 4;
            out[di] = ((sum[0] + count / 2) / count) as u8;
            out[di + 1] = ((sum[1] + count / 2) / count) as u8;
            out[di + 2] = ((sum[2] + count / 2) / count) as u8;
            out[di + 3] = ((sum[3] + count / 2) / count) as u8;
        }
    }

    Ok(out)
}

/// Result of [`texture_upload_start`]: either sub-region finished in one step or a mip-chain uploader is needed.
#[derive(Debug)]
pub enum TextureDataStart {
    /// Sub-region path completed (`n` is the mip-equivalent count from the subregion helper).
    SubregionComplete(u32),
    /// Full mip chain; call [`TextureMipChainUploader::upload_next_mip`] until [`MipChainAdvance::Finished`].
    MipChain(TextureMipChainUploader),
}

/// Classifies sub-region vs full mip chain and runs the sub-region upload when applicable.
pub fn texture_upload_start(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    fmt: &SetTexture2DFormat,
    wgpu_format: wgpu::TextureFormat,
    upload: &SetTexture2DData,
    raw: &[u8],
) -> Result<TextureDataStart, String> {
    if upload.hint.has_region != 0 {
        if hint_region_is_empty(&upload.hint) {
            logger::trace!(
                "texture {}: TextureUploadHint.has_region set but region empty; skipping upload",
                upload.asset_id
            );
            return Ok(TextureDataStart::SubregionComplete(0));
        }
        match try_write_texture2d_subregion(queue, texture, fmt, wgpu_format, upload, raw) {
            Some(Ok(n)) => {
                logger::trace!(
                    "texture {}: sub-region texture upload ({} mips equivalent)",
                    upload.asset_id,
                    n
                );
                return Ok(TextureDataStart::SubregionComplete(n));
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
    Ok(TextureDataStart::MipChain(TextureMipChainUploader::new(
        texture, fmt, upload, raw,
    )?))
}

/// Uploads mips from `raw` (exact shared-memory descriptor window) into `texture` using `wgpu_format`.
pub fn write_texture2d_mips(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    fmt: &SetTexture2DFormat,
    wgpu_format: wgpu::TextureFormat,
    upload: &SetTexture2DData,
    raw: &[u8],
) -> Result<u32, String> {
    let want = upload.data.length.max(0) as usize;
    if raw.len() < want {
        return Err(format!(
            "raw shorter than descriptor (need {want}, got {})",
            raw.len()
        ));
    }
    let payload = &raw[..want];

    match texture_upload_start(queue, texture, fmt, wgpu_format, upload, raw)? {
        TextureDataStart::SubregionComplete(n) => Ok(n),
        TextureDataStart::MipChain(mut uploader) => loop {
            match uploader.upload_next_mip(queue, texture, fmt, wgpu_format, upload, payload)? {
                MipChainAdvance::UploadedOne { .. } => {}
                MipChainAdvance::Finished { total_uploaded } => {
                    return Ok(total_uploaded);
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::downsample_rgba8_box;

    #[test]
    fn rgba8_box_downsample_averages_pixels() {
        let src = vec![
            0, 0, 0, 255, 10, 20, 30, 255, 20, 40, 60, 255, 30, 60, 90, 255,
        ];
        let out = downsample_rgba8_box(&src, 2, 2, 1, 1).unwrap();
        assert_eq!(out, vec![15, 30, 45, 255]);
    }

    #[test]
    fn rgba8_box_downsample_handles_odd_dimensions() {
        let src = vec![
            0, 0, 0, 255, 10, 0, 0, 255, 20, 0, 0, 255, 30, 0, 0, 255, 40, 0, 0, 255, 50, 0, 0,
            255, 60, 0, 0, 255, 70, 0, 0, 255, 80, 0, 0, 255,
        ];
        let out = downsample_rgba8_box(&src, 3, 3, 1, 1).unwrap();
        assert_eq!(out, vec![40, 0, 0, 255]);
    }
}

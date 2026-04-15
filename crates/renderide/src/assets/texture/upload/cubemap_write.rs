//! [`SetCubemapData`](crate::shared::SetCubemapData) → cubemap array layers ([`super::mip_write_common::write_cubemap_face_mip`]).

use crate::shared::{SetCubemapData, SetCubemapFormat};

use super::super::decode::{decode_mip_to_rgba8, flip_mip_rows, needs_rgba8_decode_before_upload};
use super::super::layout::{
    host_format_is_compressed, mip_byte_len, mip_dimensions_at_level, mip_tight_bytes_per_texel,
};
use super::mip_write_common::{is_rgba8_family, uncompressed_row_bytes, write_cubemap_face_mip};
use super::write_mip_chain::MipChainAdvance;

/// Incremental cubemap upload: one face × one mip per step.
#[derive(Debug)]
pub struct CubemapMipChainUploader {
    face: u32,
    mip_i: usize,
    uploaded: u32,
    start_bias: usize,
    start_base: u32,
    mipmap_count: u32,
    face_size: u32,
    flip: bool,
}

impl CubemapMipChainUploader {
    /// Validates `raw` / `upload` / `fmt` (no GPU work).
    pub fn new(
        texture: &wgpu::Texture,
        fmt: &SetCubemapFormat,
        upload: &SetCubemapData,
        raw: &[u8],
    ) -> Result<Self, String> {
        let want = upload.data.length.max(0) as usize;
        if raw.len() < want {
            return Err(format!(
                "raw shorter than descriptor (need {want}, got {})",
                raw.len()
            ));
        }

        if upload.mip_map_sizes.is_empty() {
            return Err("cubemap: no mips in upload".into());
        }
        if upload.mip_starts.len() != 6 {
            return Err(format!(
                "cubemap: expected mip_starts len 6 (faces), got {}",
                upload.mip_starts.len()
            ));
        }
        for (fi, starts) in upload.mip_starts.iter().enumerate() {
            if starts.len() != upload.mip_map_sizes.len() {
                return Err(format!(
                    "cubemap: face {fi} mip_starts len {} != mip_map_sizes len {}",
                    starts.len(),
                    upload.mip_map_sizes.len()
                ));
            }
        }

        let start_base = upload.start_mip_level.max(0) as u32;
        let mipmap_count = fmt.mipmap_count.max(1) as u32;
        if start_base >= mipmap_count {
            return Err(format!(
                "start_mip_level {start_base} >= mipmap_count {mipmap_count}"
            ));
        }

        let tex_extent = texture.size();
        let face_size = fmt.size.max(0) as u32;
        if tex_extent.width != face_size
            || tex_extent.height != face_size
            || tex_extent.depth_or_array_layers != 6
        {
            return Err(format!(
                "GPU cubemap {}×{}×{} does not match format face {} (asset {})",
                tex_extent.width,
                tex_extent.height,
                tex_extent.depth_or_array_layers,
                face_size,
                upload.asset_id
            ));
        }

        let payload_len = want;
        let (start_bias, _prefix) = choose_mip_start_bias_cubemap(fmt.format, upload, payload_len)?;

        Ok(Self {
            face: 0,
            mip_i: 0,
            uploaded: 0,
            start_bias,
            start_base,
            mipmap_count,
            face_size,
            flip: upload.flip_y,
        })
    }

    /// Writes at most one face mip. `payload` is `&raw[..upload.data.length]`.
    pub fn upload_next_face_mip(
        &mut self,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        fmt: &SetCubemapFormat,
        wgpu_format: wgpu::TextureFormat,
        upload: &SetCubemapData,
        payload: &[u8],
    ) -> Result<MipChainAdvance, String> {
        if self.face >= 6 {
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded,
            });
        }

        let mip_i = self.mip_i;
        debug_assert!(mip_i < upload.mip_map_sizes.len());

        let sz = upload.mip_map_sizes[mip_i];
        let w = sz.x.max(0) as u32;
        let h = sz.y.max(0) as u32;
        let mip_level = self.start_base + mip_i as u32;
        if mip_level >= self.mipmap_count {
            return Err(format!(
                "cubemap mip_level {mip_level} exceeds format mipmap_count {}",
                self.mipmap_count
            ));
        }

        let (gw, gh) = mip_dimensions_at_level(self.face_size, self.face_size, mip_level);
        if w != gw || h != gh {
            return Err(format!(
                "cubemap {} mip {mip_level}: upload says {w}×{h} but GPU mip is {gw}×{gh}",
                upload.asset_id
            ));
        }

        let start_raw = upload.mip_starts[self.face as usize][mip_i];
        if start_raw < 0 {
            return Err("negative mip_starts".into());
        }
        let start_abs = start_raw as usize;
        if start_abs < self.start_bias {
            return Err(format!(
                "mip start {} is before descriptor offset {}",
                start_abs, self.start_bias
            ));
        }
        let start = start_abs - self.start_bias;

        let host_len = mip_byte_len(fmt.format, w, h)
            .ok_or_else(|| format!("cubemap mip byte size unsupported for {:?}", fmt.format))?
            as usize;

        let Some(mip_src) = payload.get(start..start + host_len) else {
            return Err(format!(
                "cubemap {} face {} mip {mip_i}: slice out of range (start {start} len {host_len}, payload {})",
                upload.asset_id,
                self.face,
                payload.len()
            ));
        };

        let flip = self.flip;
        let pixels: std::borrow::Cow<'_, [u8]> = if is_rgba8_family(wgpu_format) {
            if needs_rgba8_decode_before_upload(fmt.format) || host_format_is_compressed(fmt.format)
            {
                std::borrow::Cow::Owned(
                    decode_mip_to_rgba8(fmt.format, w, h, flip, mip_src).ok_or_else(|| {
                        format!(
                            "RGBA decode failed for cubemap face {} mip {mip_i}",
                            self.face
                        )
                    })?,
                )
            } else if flip {
                let mut v = mip_src.to_vec();
                let bpp = mip_tight_bytes_per_texel(v.len(), w, h).ok_or_else(|| {
                    format!(
                        "cubemap mip {mip_i}: RGBA8 upload len {} not divisible by {}×{} texels",
                        v.len(),
                        w,
                        h
                    )
                })?;
                if bpp != 4 {
                    return Err(format!(
                        "cubemap mip {mip_i}: RGBA8 family expects 4 bytes per texel, got {bpp}"
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
                        "cubemap mip {mip_i}: len {} not divisible by {}×{} texels (flip_y)",
                        v.len(),
                        w,
                        h
                    )
                })?;
                if let Ok(bpp_gpu) = uncompressed_row_bytes(wgpu_format) {
                    if bpp_host != bpp_gpu {
                        logger::warn!(
                            "cubemap {} face {} mip {mip_i}: host texel stride {} B != GPU {:?} stride {} B",
                            upload.asset_id,
                            self.face,
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
                        "cubemap {} mip {mip_i}: flip_y skipped for compressed {:?} GPU upload",
                        upload.asset_id,
                        wgpu_format
                    );
                }
                std::borrow::Cow::Borrowed(mip_src)
            }
        };

        write_cubemap_face_mip(
            queue,
            texture,
            mip_level,
            self.face,
            w,
            h,
            wgpu_format,
            pixels.as_ref(),
        )?;

        self.uploaded += 1;
        self.mip_i += 1;
        if self.mip_i >= upload.mip_map_sizes.len() {
            self.face += 1;
            self.mip_i = 0;
        }

        if self.face >= 6 {
            return Ok(MipChainAdvance::Finished {
                total_uploaded: self.uploaded,
            });
        }

        Ok(MipChainAdvance::UploadedOne)
    }
}

fn choose_mip_start_bias_cubemap(
    format: crate::shared::TextureFormat,
    upload: &SetCubemapData,
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
        let prefix = valid_cubemap_mip_prefix_len(format, upload, payload_len, bias)?;
        if prefix > best_prefix {
            best_prefix = prefix;
            best_bias = bias;
        }
    }
    if best_prefix == 0 {
        return Err(format!(
            "cubemap mip region exceeds shared memory descriptor (payload_len={}, descriptor_offset={})",
            payload_len, offset_bias
        ));
    }
    Ok((best_bias, best_prefix))
}

fn valid_cubemap_mip_prefix_len(
    format: crate::shared::TextureFormat,
    upload: &SetCubemapData,
    payload_len: usize,
    bias: usize,
) -> Result<usize, String> {
    let mut count = 0usize;
    'outer: for face in 0..6usize {
        for (i, sz) in upload.mip_map_sizes.iter().enumerate() {
            if sz.x <= 0 || sz.y <= 0 {
                return Err("non-positive mip dimensions".into());
            }
            let w = sz.x as u32;
            let h = sz.y as u32;
            let host_len = mip_byte_len(format, w, h)
                .ok_or_else(|| format!("mip byte size unsupported for {:?}", format))?
                as usize;
            let starts = upload
                .mip_starts
                .get(face)
                .ok_or_else(|| "cubemap mip_starts face missing".to_string())?;
            let start_raw = *starts
                .get(i)
                .ok_or_else(|| "cubemap mip_starts index".to_string())?;
            if start_raw < 0 {
                return Err("negative mip_starts".into());
            }
            let start_abs = start_raw as usize;
            if start_abs < bias {
                break 'outer;
            }
            let start = start_abs - bias;
            let end = start
                .checked_add(host_len)
                .ok_or_else(|| "mip end overflow".to_string())?;
            if end > payload_len {
                break 'outer;
            }
            count += 1;
        }
    }
    Ok(count)
}

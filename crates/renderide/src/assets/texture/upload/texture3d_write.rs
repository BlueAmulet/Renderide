//! [`SetTexture3DData`](crate::shared::SetTexture3DData) → [`wgpu::Queue::write_texture`] for [`wgpu::TextureDimension::D3`].

use crate::shared::{SetTexture3DData, SetTexture3DFormat};

use super::super::decode::{decode_mip_to_rgba8, needs_rgba8_decode_before_upload};
use super::super::layout::{host_format_is_compressed, mip_byte_len, mip_dimensions_at_level_3d};
use super::mip_write_common::{is_rgba8_family, write_texture3d_volume_mip};

/// Incremental 3D mip upload: one mip level per [`Texture3dMipChainUploader::upload_next_mip`] call.
#[derive(Debug)]
pub struct Texture3dMipChainUploader {
    next_mip: u32,
    uploaded_mips: u32,
    base_w: u32,
    base_h: u32,
    base_d: u32,
    mipmap_count: u32,
}

/// Result of one [`Texture3dMipChainUploader::upload_next_mip`] step.
#[derive(Debug)]
pub enum Texture3dMipAdvance {
    /// Uploaded a single mip; call again.
    UploadedOne,
    /// Chain complete.
    Finished {
        /// Total mips successfully written.
        total_uploaded: u32,
    },
}

impl Texture3dMipChainUploader {
    /// Validates `raw` against `fmt` and prepares chain state (no GPU work).
    pub fn new(
        texture: &wgpu::Texture,
        fmt: &SetTexture3DFormat,
        upload: &SetTexture3DData,
        raw: &[u8],
    ) -> Result<Self, String> {
        let want = upload.data.length.max(0) as usize;
        if raw.len() < want {
            return Err(format!(
                "raw shorter than descriptor (need {want}, got {})",
                raw.len()
            ));
        }

        let base_w = fmt.width.max(0) as u32;
        let base_h = fmt.height.max(0) as u32;
        let base_d = fmt.depth.max(0) as u32;
        let mipmap_count = fmt.mipmap_count.max(1) as u32;

        let tex_extent = texture.size();
        if tex_extent.width != base_w
            || tex_extent.height != base_h
            || tex_extent.depth_or_array_layers != base_d
        {
            return Err(format!(
                "GPU texture {}×{}×{} does not match SetTexture3DFormat {}×{}×{} for asset {}",
                tex_extent.width,
                tex_extent.height,
                tex_extent.depth_or_array_layers,
                base_w,
                base_h,
                base_d,
                upload.asset_id
            ));
        }

        let mut total_need = 0usize;
        for level in 0..mipmap_count {
            let (w, h, d) = mip_dimensions_at_level_3d(base_w, base_h, base_d, level);
            let slice = mip_byte_len(fmt.format, w, h).ok_or_else(|| {
                format!(
                    "texture3d {}: mip byte size unsupported for {:?}",
                    upload.asset_id, fmt.format
                )
            })? as usize;
            let vol = slice
                .checked_mul(d as usize)
                .ok_or_else(|| "texture3d mip volume byte overflow".to_string())?;
            total_need = total_need
                .checked_add(vol)
                .ok_or_else(|| "texture3d mip chain total overflow".to_string())?;
        }

        if total_need > want {
            return Err(format!(
                "texture3d {}: mip chain needs {total_need} B but descriptor length is {want}",
                upload.asset_id
            ));
        }

        Ok(Self {
            next_mip: 0,
            uploaded_mips: 0,
            base_w,
            base_h,
            base_d,
            mipmap_count,
        })
    }

    /// Writes at most one mip level. `payload` is `&raw[..upload.data.length]`.
    pub fn upload_next_mip(
        &mut self,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        fmt: &SetTexture3DFormat,
        wgpu_format: wgpu::TextureFormat,
        upload: &SetTexture3DData,
        payload: &[u8],
    ) -> Result<Texture3dMipAdvance, String> {
        let level = self.next_mip;
        if level >= self.mipmap_count {
            return Ok(Texture3dMipAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }

        let (w, h, d) = mip_dimensions_at_level_3d(self.base_w, self.base_h, self.base_d, level);

        let mut offset = 0usize;
        for l in 0..level {
            let (lw, lh, ld) = mip_dimensions_at_level_3d(self.base_w, self.base_h, self.base_d, l);
            let slice = mip_byte_len(fmt.format, lw, lh).ok_or_else(|| {
                format!(
                    "texture3d {}: mip byte size unsupported for {:?}",
                    upload.asset_id, fmt.format
                )
            })? as usize;
            let vol = slice
                .checked_mul(ld as usize)
                .ok_or_else(|| "texture3d offset overflow".to_string())?;
            offset = offset
                .checked_add(vol)
                .ok_or_else(|| "texture3d offset overflow".to_string())?;
        }

        let slice_bytes = mip_byte_len(fmt.format, w, h).ok_or_else(|| {
            format!(
                "texture3d {}: mip byte size unsupported for {:?}",
                upload.asset_id, fmt.format
            )
        })? as usize;
        let vol_bytes = slice_bytes
            .checked_mul(d as usize)
            .ok_or_else(|| "texture3d volume bytes overflow".to_string())?;

        let mip_src = payload
            .get(offset..offset + vol_bytes)
            .ok_or_else(|| {
                format!(
                    "texture3d {}: mip {level} slice out of range (offset {offset} len {vol_bytes} payload {})",
                    upload.asset_id,
                    payload.len()
                )
            })?;

        let pixels: std::borrow::Cow<'_, [u8]> = if is_rgba8_family(wgpu_format) {
            if needs_rgba8_decode_before_upload(fmt.format) || host_format_is_compressed(fmt.format)
            {
                let mut out = Vec::with_capacity(vol_bytes);
                let mut z_off = 0usize;
                for _z in 0..d {
                    let slice_raw = mip_src
                        .get(z_off..z_off + slice_bytes)
                        .ok_or_else(|| "texture3d slice bounds".to_string())?;
                    let decoded = decode_mip_to_rgba8(fmt.format, w, h, false, slice_raw)
                        .ok_or_else(|| {
                            format!(
                                "texture3d {}: RGBA decode failed mip {level}",
                                upload.asset_id
                            )
                        })?;
                    out.extend_from_slice(&decoded);
                    z_off += slice_bytes;
                }
                std::borrow::Cow::Owned(out)
            } else {
                std::borrow::Cow::Borrowed(mip_src)
            }
        } else {
            if needs_rgba8_decode_before_upload(fmt.format) {
                return Err(format!(
                    "texture3d {}: host {:?} must decode to RGBA but GPU format is {:?}",
                    upload.asset_id, fmt.format, wgpu_format
                ));
            }
            std::borrow::Cow::Borrowed(mip_src)
        };

        write_texture3d_volume_mip(queue, texture, level, w, h, d, wgpu_format, pixels.as_ref())?;

        self.uploaded_mips += 1;
        self.next_mip += 1;

        if self.next_mip >= self.mipmap_count {
            return Ok(Texture3dMipAdvance::Finished {
                total_uploaded: self.uploaded_mips,
            });
        }
        Ok(Texture3dMipAdvance::UploadedOne)
    }
}

/// Runs the full mip chain upload for 3D data (non-cooperative path).
pub fn write_texture3d_mips(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    fmt: &SetTexture3DFormat,
    wgpu_format: wgpu::TextureFormat,
    upload: &SetTexture3DData,
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
    let mut uploader = Texture3dMipChainUploader::new(texture, fmt, upload, raw)?;
    loop {
        match uploader.upload_next_mip(queue, texture, fmt, wgpu_format, upload, payload)? {
            Texture3dMipAdvance::UploadedOne => {}
            Texture3dMipAdvance::Finished { total_uploaded } => {
                return Ok(total_uploaded);
            }
        }
    }
}

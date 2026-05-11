//! Skinning buffer extraction helpers.

/// Splits the mesh tail `bone_counts` and `bone_weights` regions into GPU storage buffers for the
/// skinning shader: `array<vec4<u32>>` joint indices and `array<vec4<f32>>` weights per vertex.
///
/// The host stores a compact variable-length stream: one byte count per vertex followed by that many
/// `(f32 weight, i32 index)` tuples. The renderer preserves the four strongest finite positive
/// influences for each vertex and normalizes them for linear blend skinning.
pub fn split_bone_weights_tail_for_gpu(
    bone_counts: &[u8],
    bone_weights_tail: &[u8],
    vertex_count: usize,
) -> Option<(Vec<u8>, Vec<u8>)> {
    if vertex_count == 0 {
        return None;
    }
    if bone_counts.len() < vertex_count {
        return None;
    }

    let mut idx_bytes = vec![0u8; vertex_count * 16];
    let mut wt_bytes = vec![0u8; vertex_count * 16];
    let mut tail_offset = 0usize;

    for (v, bone_count) in bone_counts.iter().copied().enumerate().take(vertex_count) {
        let mut influences = [BoneInfluence::ZERO; 4];
        for _ in 0..usize::from(bone_count) {
            let end = tail_offset.checked_add(8)?;
            let src = bone_weights_tail.get(tail_offset..end)?;
            tail_offset = end;
            let weight = f32::from_le_bytes(src[0..4].try_into().ok()?);
            let index = i32::from_le_bytes(src[4..8].try_into().ok()?);
            if weight.is_finite() && weight > 0.0 && index >= 0 {
                insert_influence(
                    &mut influences,
                    BoneInfluence {
                        weight,
                        index: index as u32,
                    },
                );
            }
        }
        let weight_sum = influences
            .iter()
            .fold(0.0f32, |sum, influence| sum + influence.weight);
        for (k, influence) in influences.iter().enumerate() {
            let w = if weight_sum > 1.0e-6 {
                influence.weight / weight_sum
            } else {
                0.0
            };
            let wb = v * 16 + k * 4;
            wt_bytes[wb..wb + 4].copy_from_slice(&w.to_le_bytes());
            idx_bytes[wb..wb + 4].copy_from_slice(&influence.index.to_le_bytes());
        }
    }
    Some((idx_bytes, wt_bytes))
}

#[derive(Clone, Copy)]
struct BoneInfluence {
    weight: f32,
    index: u32,
}

impl BoneInfluence {
    const ZERO: Self = Self {
        weight: 0.0,
        index: 0,
    };
}

fn insert_influence(influences: &mut [BoneInfluence; 4], candidate: BoneInfluence) {
    let mut insert_at = influences.len();
    for (i, current) in influences.iter().enumerate() {
        if candidate.weight > current.weight {
            insert_at = i;
            break;
        }
    }
    if insert_at == influences.len() {
        return;
    }
    for i in (insert_at + 1..influences.len()).rev() {
        influences[i] = influences[i - 1];
    }
    influences[insert_at] = candidate;
}

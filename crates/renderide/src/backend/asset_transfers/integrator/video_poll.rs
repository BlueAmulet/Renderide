//! Video-texture event polling performed at the end of each asset-integration drain.

use crate::ipc::DualQueueIpc;

use super::super::AssetTransferQueue;

/// Polls video texture players after upload integration.
///
/// Samples each player's clock error against the host's last-applied playback request and records
/// the latest result so the runtime can flush it into the next
/// [`crate::shared::FrameStartData`].
pub(super) fn poll_video_texture_events(
    asset: &mut AssetTransferQueue,
    ipc: &mut Option<&mut DualQueueIpc>,
) {
    profiling::scope!("asset::video_texture_poll_events");
    let mut video_textures = std::mem::take(&mut asset.video.video_players);
    {
        profiling::scope!("video::sample_clock_errors");
        for player in video_textures.values_mut() {
            player.process_events(asset, ipc);
            if let Some(state) = player.sample_clock_error() {
                asset.video.record_pending_clock_error(state);
            }
        }
    }
    asset.video.video_players = video_textures;
}

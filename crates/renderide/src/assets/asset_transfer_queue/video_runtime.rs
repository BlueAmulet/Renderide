//! Video texture runtime state owned by the asset-transfer facade.

use hashbrown::HashMap;

use crate::assets::video::player::VideoPlayer;
use crate::shared::VideoTextureClockErrorState;

/// Active video players and per-frame video telemetry.
#[derive(Default)]
pub(crate) struct VideoAssetRuntime {
    /// Active GStreamer-backed video players keyed by asset id.
    pub(crate) video_players: HashMap<i32, VideoPlayer>,
    /// Per-frame accumulator of sampled video clock errors.
    pub(crate) pending_video_clock_errors: Vec<VideoTextureClockErrorState>,
}

impl VideoAssetRuntime {
    /// Drains clock-error samples for the next host begin-frame message.
    pub(crate) fn take_pending_clock_errors(&mut self) -> Vec<VideoTextureClockErrorState> {
        std::mem::take(&mut self.pending_video_clock_errors)
    }

    /// Starts cooperative shutdown for all active video players.
    pub(crate) fn begin_shutdown(&mut self) {
        for player in self.video_players.values_mut() {
            player.begin_shutdown();
        }
    }

    /// Returns `true` once all active video player workers have quiesced.
    pub(crate) fn shutdown_complete(&mut self) -> bool {
        let mut complete = true;
        for player in self.video_players.values_mut() {
            complete &= player.poll_shutdown_complete();
        }
        complete
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_pending_clock_errors_drains_accumulator() {
        let mut runtime = VideoAssetRuntime {
            pending_video_clock_errors: vec![VideoTextureClockErrorState {
                asset_id: 4,
                current_clock_error: 0.25,
            }],
            ..Default::default()
        };

        let drained = runtime.take_pending_clock_errors();

        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].asset_id, 4);
        assert!(runtime.pending_video_clock_errors.is_empty());
    }

    #[test]
    fn empty_video_runtime_shutdown_is_complete() {
        let mut runtime = VideoAssetRuntime::default();

        runtime.begin_shutdown();

        assert!(runtime.shutdown_complete());
    }
}

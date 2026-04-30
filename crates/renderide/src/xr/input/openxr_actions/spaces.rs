//! OpenXR controller pose-space creation.

use openxr as xr;

use super::action_handles::OpenxrInputActions;

pub(super) fn create_grip_and_aim_spaces(
    session: &xr::Session<xr::Vulkan>,
    actions: &OpenxrInputActions,
) -> Result<(xr::Space, xr::Space, xr::Space, xr::Space), xr::sys::Result> {
    Ok((
        actions
            .left_grip_pose
            .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?,
        actions
            .right_grip_pose
            .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?,
        actions
            .left_aim_pose
            .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?,
        actions
            .right_aim_pose
            .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?,
    ))
}

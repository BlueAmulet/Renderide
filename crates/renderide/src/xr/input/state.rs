//! Maps OpenXR action state and resolved poses into host [`crate::shared::VRControllerState`].

use glam::Vec2;

use crate::shared::{
    BodyNode, Chirality, GenericControllerState, IndexControllerState, TouchControllerModel,
    TouchControllerState, VRControllerState, ViveControllerState, WindowsMRControllerState,
};

use super::frame::ControllerFrame;
use super::profile::{device_label, ActiveControllerProfile};

pub(super) fn vec2_nonzero(v: Vec2) -> bool {
    v.length_squared() > 1e-6
}

pub(super) fn body_node_for_side(side: Chirality) -> BodyNode {
    match side {
        Chirality::Left => BodyNode::LeftController,
        Chirality::Right => BodyNode::RightController,
    }
}

/// Maps the active OpenXR profile to a host [`VRControllerState`] variant.
///
/// [`ActiveControllerProfile::Generic`] is encoded as [`VRControllerState::TouchControllerState`]
/// so the host input stack receives the same polymorphic shape as Touch controllers (Quest-class
/// paths) instead of [`VRControllerState::GenericControllerState`], which would deserialize to a
/// different controller type on the host.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_controller_state(
    profile: ActiveControllerProfile,
    side: Chirality,
    is_tracking: bool,
    frame: ControllerFrame,
    trigger: f32,
    trigger_touch: bool,
    trigger_click: bool,
    squeeze: f32,
    squeeze_click: bool,
    thumbstick: Vec2,
    thumbstick_touch: bool,
    thumbstick_click: bool,
    trackpad: Vec2,
    trackpad_touch: bool,
    trackpad_click: bool,
    trackpad_force: f32,
    primary: bool,
    secondary: bool,
    primary_touch: bool,
    secondary_touch: bool,
    menu: bool,
    thumbrest_touch: bool,
    select: bool,
) -> VRControllerState {
    let device_id = Some(match side {
        Chirality::Left => "OpenXR Left".to_string(),
        Chirality::Right => "OpenXR Right".to_string(),
    });
    let device_model = Some(device_label(profile).to_string());
    let body_node = body_node_for_side(side);
    let trigger_touch = trigger_touch || trigger > 0.01;
    let trigger_click = trigger_click || trigger > 0.75;
    let grip_touch = squeeze_click || squeeze > 0.05;
    let grip_click = squeeze_click || squeeze > 0.85;
    let joystick_touch = thumbstick_touch || vec2_nonzero(thumbstick);
    let touchpad_touch = trackpad_touch || vec2_nonzero(trackpad) || trackpad_force > 0.01;
    match profile {
        ActiveControllerProfile::Touch => {
            VRControllerState::TouchControllerState(TouchControllerState {
                model: TouchControllerModel::QuestAndRiftS,
                start: menu,
                button_yb: secondary,
                button_xa: primary,
                button_yb_touch: secondary_touch,
                button_xa_touch: primary_touch,
                thumbrest_touch,
                grip: squeeze,
                grip_click,
                joystick_raw: thumbstick,
                joystick_touch,
                joystick_click: thumbstick_click,
                trigger,
                trigger_touch,
                trigger_click,
                device_id,
                device_model,
                side,
                body_node,
                is_device_active: true,
                is_tracking,
                position: frame.position,
                rotation: frame.rotation,
                has_bound_hand: frame.has_bound_hand,
                hand_position: frame.hand_position,
                hand_rotation: frame.hand_rotation,
                battery_level: 1.0,
                battery_charging: false,
            })
        }
        ActiveControllerProfile::Index => {
            VRControllerState::IndexControllerState(IndexControllerState {
                grip: squeeze,
                grip_touch,
                grip_click,
                button_a: primary,
                button_b: secondary,
                button_atouch: primary_touch,
                button_btouch: secondary_touch,
                trigger,
                trigger_touch,
                trigger_click,
                joystick_raw: thumbstick,
                joystick_touch,
                joystick_click: thumbstick_click,
                touchpad: trackpad,
                touchpad_touch,
                touchpad_press: trackpad_click || trackpad_force > 0.3,
                touchpad_force: trackpad_force,
                device_id,
                device_model,
                side,
                body_node,
                is_device_active: true,
                is_tracking,
                position: frame.position,
                rotation: frame.rotation,
                has_bound_hand: frame.has_bound_hand,
                hand_position: frame.hand_position,
                hand_rotation: frame.hand_rotation,
                battery_level: 1.0,
                battery_charging: false,
            })
        }
        ActiveControllerProfile::Vive => {
            VRControllerState::ViveControllerState(ViveControllerState {
                grip: squeeze_click || squeeze > 0.5,
                app: menu,
                trigger_hair: trigger_touch,
                trigger_click,
                trigger,
                touchpad_touch,
                touchpad_click: trackpad_click,
                touchpad: trackpad,
                device_id,
                device_model,
                side,
                body_node,
                is_device_active: true,
                is_tracking,
                position: frame.position,
                rotation: frame.rotation,
                has_bound_hand: frame.has_bound_hand,
                hand_position: frame.hand_position,
                hand_rotation: frame.hand_rotation,
                battery_level: 1.0,
                battery_charging: false,
            })
        }
        ActiveControllerProfile::WindowsMr => {
            VRControllerState::WindowsMRControllerState(WindowsMRControllerState {
                grip: squeeze_click || squeeze > 0.5,
                app: menu,
                trigger_hair: trigger_touch,
                trigger_click,
                trigger,
                touchpad_touch,
                touchpad_click: trackpad_click,
                touchpad: trackpad,
                joystick_click: thumbstick_click,
                joystick_raw: thumbstick,
                device_id,
                device_model,
                side,
                body_node,
                is_device_active: true,
                is_tracking,
                position: frame.position,
                rotation: frame.rotation,
                has_bound_hand: frame.has_bound_hand,
                hand_position: frame.hand_position,
                hand_rotation: frame.hand_rotation,
                battery_level: 1.0,
                battery_charging: false,
            })
        }
        ActiveControllerProfile::Generic => {
            VRControllerState::TouchControllerState(TouchControllerState {
                model: TouchControllerModel::QuestAndRiftS,
                start: menu,
                button_yb: secondary,
                button_xa: primary,
                button_yb_touch: secondary_touch,
                button_xa_touch: primary_touch,
                thumbrest_touch,
                grip: squeeze,
                grip_click,
                joystick_raw: thumbstick,
                joystick_touch,
                joystick_click: thumbstick_click,
                trigger,
                trigger_touch,
                trigger_click,
                device_id,
                device_model,
                side,
                body_node,
                is_device_active: true,
                is_tracking,
                position: frame.position,
                rotation: frame.rotation,
                has_bound_hand: frame.has_bound_hand,
                hand_position: frame.hand_position,
                hand_rotation: frame.hand_rotation,
                battery_level: 1.0,
                battery_charging: false,
            })
        }
        ActiveControllerProfile::Simple => {
            VRControllerState::GenericControllerState(GenericControllerState {
                strength: if select { 1.0 } else { trigger },
                axis: Vec2::ZERO,
                touching_strength: trigger_touch || select,
                touching_axis: false,
                primary: select,
                menu,
                grab: grip_click,
                secondary: false,
                device_id,
                device_model,
                side,
                body_node,
                is_device_active: true,
                is_tracking,
                position: frame.position,
                rotation: frame.rotation,
                has_bound_hand: frame.has_bound_hand,
                hand_position: frame.hand_position,
                hand_rotation: frame.hand_rotation,
                battery_level: 1.0,
                battery_charging: false,
            })
        }
    }
}

use bevy::prelude::*;
use crate::util::{BigTransform, BigPoint, BigVector};
use nalgebra as na;

#[derive(Component)]
pub struct Camera {
    pub transform: BigTransform,
}

impl Camera {
    pub fn new(eye: BigPoint, target: BigPoint, up: BigVector) {
        Self {
            transform: BigTransform::look_at_rh(eye, target, up)
        }
    }
}
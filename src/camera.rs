use bevy::prelude::*;
use nalgebra as na;

#[derive(Component)]
pub struct Camera {
    pub position: [num::BigRational; 3],
}
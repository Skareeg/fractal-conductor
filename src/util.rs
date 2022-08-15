use bevy::reflect::Array;
use num::BigRational;
use num::traits::Zero;
use std::boxed::Box;
use ndarray as na;

type BigVector = na::Array1<BigRational>;
type BigTransform = na::Array3<BigRational>;

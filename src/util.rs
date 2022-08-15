use arrayvec::ArrayVec;
use bevy::reflect::Array;
use num::BigRational;
use num::traits::Zero;
use nalgebra as na;
use std::boxed::Box;

/// Named so because it represents heap held BigRationals
#[derive(PartialEq, Debug, Default)]
pub struct BurningHeap(Box<num::BigRational>);

impl From<num::BigRational> for BurningHeap {
    fn from(val: num::BigRational) -> Self {
        Self(Box::new(val))
    }
}

impl Clone for BurningHeap {
    fn clone(&self) -> Self {
        Self(Box::new(self.0.as_ref().clone()))
    }
}

impl std::ops::Add for BurningHeap {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        (self.0.as_ref() + rhs.0.as_ref()).into()
    }
}

impl std::ops::Sub for BurningHeap {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        (self.0.as_ref() - rhs.0.as_ref()).into()
    }
}

impl std::ops::AddAssign for BurningHeap {
    fn add_assign(&mut self, rhs: Self) {
        self.0.as_mut().add_assign(rhs.0.as_ref());
    }
}

impl std::ops::SubAssign for BurningHeap {
    fn sub_assign(&mut self, rhs: Self) {
        self.0.as_mut().sub_assign(rhs.0.as_ref());
    }
}

impl std::ops::Mul for BurningHeap {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        (self.0.as_ref() * rhs.0.as_ref()).into()
    }
}

impl std::ops::Div for BurningHeap {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        (self.0.as_ref() / rhs.0.as_ref()).into()
    }
}

impl std::ops::MulAssign for BurningHeap {
    fn mul_assign(&mut self, rhs: Self) {
        self.0.as_mut().mul_assign(rhs.0.as_ref());
    }
}

impl std::ops::DivAssign for BurningHeap {
    fn div_assign(&mut self, rhs: Self) {
        self.0.as_mut().div_assign(rhs.0.as_ref());
    }
}

impl num::traits::Zero for BurningHeap {
    fn zero() -> Self {
        BigRational::zero().into()
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
    fn set_zero(&mut self) {
        self.0.set_zero();
    }
}

// impl na::SimdValue for BurningHeap {
//     type Element = Self;
//     type SimdBool = bool;
//     fn lanes() -> usize {
//         1
//     }
//     fn splat(val: Self::Element) -> Self {
//         val
//     }
//     fn extract(&self, i: usize) -> Self::Element {
//         self.clone()
//     }
//     unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
//         self.clone()
//     }
//     fn replace(&mut self, i: usize, val: Self::Element) {
//         self.0 = val.0;
//     }
//     unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
//         self.0 = val.0;
//     }
//     fn select(self, cond: Self::SimdBool, other: Self) -> Self {
//         self.clone()
//     }
//     fn map_lanes(self, f: impl Fn(Self::Element) -> Self::Element) -> Self
//     where
//             Self: Clone, {
//         f(self)
//     }
//     fn zip_map_lanes(self, b: Self, f: impl Fn(Self::Element, Self::Element) -> Self::Element) -> Self
//     where
//             Self: Clone, {
//         f(self, b)
//     }
// }

// impl na::SimdPartialOrd for BurningHeap {
//     fn simd_gt(self, other: Self) -> Self::SimdBool {
//         self.0 > other.0
//     }
//     fn simd_lt(self, other: Self) -> Self::SimdBool {
//         self.0 < other.0
//     }
//     fn simd_ge(self, other: Self) -> Self::SimdBool {
//         self.0 >= other.0
//     }
//     fn simd_le(self, other: Self) -> Self::SimdBool {
//         self.0 <= other.0
//     }
//     fn simd_eq(self, other: Self) -> Self::SimdBool {
//         self.0 == other.0
//     }
//     fn simd_ne(self, other: Self) -> Self::SimdBool {
//         self.0 != other.0
//     }
//     fn simd_max(self, other: Self) -> Self {
//         BigRational::max(self.0, other.0).into()
//     }
//     fn simd_min(self, other: Self) -> Self {
//         BigRational::min(self.0, other.0).into()
//     }
//     fn simd_clamp(self, min: Self, max: Self) -> Self {
//         BigRational::clamp(self.0, min.0, max.0).into()
//     }
//     fn simd_horizontal_min(self) -> Self::Element {
//         self.clone()
//     }
//     fn simd_horizontal_max(self) -> Self::Element {
//         self.clone()
//     }
// }

// impl simba::simd::SimdSigned for BurningHeap {
//     fn simd_abs(&self) -> Self {
//         self.0.abs().into()
//     }
//     fn simd_abs_sub(&self, other: &Self) -> Self {
//         let a = self.0.abs();
//         if a <= other.0 {
//             Self::zero()
//         } else {
//             other - a
//         }
//     }
//     fn simd_signum(&self) -> Self {
//     }
//     fn is_simd_positive(&self) -> Self::SimdBool {
//     }
//     fn is_simd_negative(&self) -> Self::SimdBool {
//     }
// }

// impl simba::scalar::SubsetOf<Self> for BurningHeap {
// }

// impl simba::scalar::SupersetOf<f64> for BurningHeap {
// }

impl core::ops::Neg for BurningHeap {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.0.neg().into()
    }
}

// // impl simba::scalar::ClosedNeg for BurningHeap {
// // }

// impl na::Field for BurningHeap {
// }

// impl na::SimdComplexField for BurningHeap {
//     type SimdRealField = Self;
//     fn simd_horizontal_sum(self) -> Self::Element {
//     }
//     fn simd_horizontal_product(self) -> Self::Element {
//     }
// }

// impl na::SimdRealField for BurningHeap {
//     fn simd_copysign(self, sign: Self) -> Self {
//     }
//     fn simd_atan2(self, other: Self) -> Self {
//     }
//     fn simd_default_epsilon() -> Self {
//     }
//     fn simd_pi() -> Self {
//     }
//     fn simd_two_pi() -> Self {
//     }
//     fn simd_frac_pi_2() -> Self {
//     }
//     fn simd_frac_pi_3() -> Self {
//     }
//     fn simd_frac_pi_4() -> Self {
//     }
//     fn simd_frac_pi_6() -> Self {
//     }
//     fn simd_frac_pi_8() -> Self {
//     }
//     fn simd_frac_1_pi() -> Self {
//     }
//     fn simd_frac_2_pi() -> Self {
//     }
//     fn simd_frac_2_sqrt_pi() -> Self {
//     }
//     fn simd_e() -> Self {
//     }
//     fn simd_log2_e() -> Self {
//     }
//     fn simd_log10_e() -> Self {
//     }
//     fn simd_ln_2() -> Self {
//     }
//     fn simd_ln_10() -> Self {
//     }
// }

pub type BigTransform = nalgebra::Isometry3<BurningHeap>;
pub type BigPoint = nalgebra::Point3<BurningHeap>;
pub type BigVector = nalgebra::Vector3<BurningHeap>;

// type BigVector = BigMatrix<1, 3>;

// #[derive(Clone, Debug, Default)]
// pub struct BigMatrix<const R: usize, const C: usize> {
//     pub data: ArrayVec<ArrayVec<BigRational, C>, R>,
// }

// impl<const R: usize, const C: usize> std::ops::Add for BigMatrix<R, C> {
//     type Output = BigMatrix<R, C>;
//     fn add(self, rhs: Self) -> Self::Output {
//         let mut data = ArrayVec::<ArrayVec::<BigRational, C>, R>::new();
//         for c in 1..C {
//             for r in 1..R {
//                 data[c][r] = self.data[c][r] + rhs.data[c][r];
//             }
//         }
//         Self { data }
//     }
// }

// impl<const R: usize, const C: usize> std::ops::AddAssign for BigMatrix<R, C> {
//     fn add_assign(&mut self, rhs: Self) {
//         for c in 1..C {
//             for r in 1..R {
//                 self.data[c][r] += rhs.data[c][r];
//             }
//         }
//     }
// }

// impl<const R1: usize, const C: usize, const R2: usize> std::ops::Mul<BigMatrix<C, R2>> for BigMatrix<R1, C> {
//     type Output = BigMatrix<R1, R2>;
//     fn mul(self, rhs: BigMatrix<C, R2>) -> Self::Output {
//         let mut data = ArrayVec::<ArrayVec::<BigRational, R1>, R2>::new();
//         for c in 1..R1 {
//             for r in 1..R2 {
//                 let mut sum = num::BigRational::new(0.into(), 1.into());
//                 for i in 1..C {
//                     sum += self.data[i][r] * rhs.data[c][i];
//                 }
//                 data[c][r] = sum;
//             }
//         }
//         Self { data }
//     }
// }
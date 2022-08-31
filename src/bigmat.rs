use std::{f64::NAN, ops::{Add, Sub, Mul, Div}, borrow::Cow};
use std::fmt;

use num::{BigRational, BigInt, BigUint, FromPrimitive, ToPrimitive};

pub fn new_bri(v: i64) -> BigRational {
    BigRational::new_raw(v.into(), 1.into())
}
pub fn new_brf(v: f64) -> BigRational {
    BigRational::from_f64(v).expect(&format!("Could not create BigRational from {}.", v))
}

#[test]
fn test_sqrt() {
    // This should come out very near but above 277.
    let mut testnum = BigUint::from_u128(3u128).expect("Could not convert u128.");
    let mut testden = BigUint::from_u128(2u128).expect("Could not convert u128.");
    println!("Test Number: {} / {}", testnum.clone(), testden.clone());
    let testrat = BigRational::new_raw(testnum.clone().into(), testden.clone().into()).to_f64().expect("Could not map to f64.");
    println!("Test Rational: {}", testrat);
    let mut baseline: f64 = 3.0 / 2.0;
    println!("Baseline Rational: {}", baseline);

    // Need to increase the total ration values to preserve precision.
    let precision = BigUint::from_u128(1000).expect("Could not convert u128.");
    println!("Precision: {}", precision.clone());
    let mut scalednum = testnum.clone();
    let mut scaledden = testden.clone();
    let mut iterations = precision.clone();
    let zero = BigUint::new(vec![0]);
    let one = BigUint::new(vec![1]);
    let ten = BigUint::new(vec![10]);
    while iterations != zero {
        scalednum = scalednum * &ten;
        scaledden = scaledden * &ten;
        iterations = iterations - &one;
    }
    println!("Scaled Number: {}... / {}...", scalednum.clone().to_string().chars().take(10).collect::<String>(), scaledden.clone().to_string().chars().take(10).collect::<String>());
    let scaledrat = BigRational::new_raw(testnum.clone().into(), testden.clone().into()).to_f64().expect("Could not map to f64.");
    println!("Scaled Rational: {}", testrat);

    let sqrtnum = scalednum.sqrt();
    let sqrtden = scaledden.sqrt();
    println!("Test Number: {}... / {}...", sqrtnum.clone().to_string().chars().take(10).collect::<String>(), sqrtden.clone().to_string().chars().take(10).collect::<String>());
    let sqrtrat = BigRational::new_raw(sqrtnum.clone().into(), sqrtden.clone().into()).to_f64().expect("Could not map to f64.");
    println!("Test Rational: {}", sqrtrat);
    let sqrtbaseline: f64 = baseline.sqrt();
    println!("Baseline Rational: {}", sqrtbaseline);

    // let (_, digits) = testnum.to_radix_le(10);
    // println!("Digits in LE: {:?}", digits);
    // let num_pairs = (digits.len() + 1) / 2;
    // let mut pairs: Vec<u8> = Vec::new();
    // pairs.reserve(num_pairs);
    // for i in 0..num_pairs {
    //     let index = i * 2;
    //     let mut num = digits[index];
    //     if index + 1 < digits.len() {
    //         num += digits[index + 1] * 10;
    //     }
    //     pairs.push(num);
    // }
    // println!("Pairs: {:?}", pairs);
}

/// Find the sqrt of a BigRational, preserving at least <precision> number of digits *relative to the current precision*.
pub fn ratio_sqrt(n: BigRational, precision: BigUint) -> BigRational {
    let (signnum, numerator) = n.numer().clone().into_parts();
    let (signden, denominator) = n.denom().clone().into_parts();

    // Need to increase the total ration values to preserve precision.
    let mut scalednum = numerator.clone();
    let mut scaledden = denominator.clone();
    let mut iterations = precision.clone();
    let zero = BigUint::new(vec![0]);
    let one = BigUint::new(vec![1]);
    let ten = BigUint::new(vec![10]);
    while iterations != zero {
        scalednum = scalednum * &ten;
        scaledden = scaledden * &ten;
        iterations = iterations - &one;
    }
    
    let sqrtnum = scalednum.sqrt();
    let sqrtden = scaledden.sqrt();

    BigRational::new_raw(BigInt::from_biguint(signnum, sqrtnum), BigInt::from_biguint(signden, sqrtden))
}

/// An arbitrary precision matrix object for linear algebra at any dimensionality.
#[derive(Clone, Debug, Default)]
pub struct BigMatrix<const I: usize, const J: usize> {
    pub data: Vec<BigRational>,
    pub rows: usize,
    pub cols: usize,
}

/// Arbitrary matrix operations.
impl<const I: usize, const J: usize> BigMatrix<I, J> {
    pub fn zero() -> Self {
        Self::splat(new_bri(0))
    }
    pub fn one() -> Self {
        Self::splat(new_bri(1))
    }
    pub fn neg_one() -> Self {
        Self::splat(new_bri(-1))
    }
    pub fn new() -> Self {
        Self {
            data: Vec::with_capacity(I * J),
            rows: I,
            cols: J,
        }
    }
    pub fn splat(v: BigRational) -> Self {
        Self {
            data: vec![v; I * J],
            rows: I,
            cols: J,
        }
    }
    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> &BigRational {
        &self.data[i * J + j]
    }
}

/// Multidimensional vector functions.
impl<const I: usize> BigMatrix<I, 1> {
    /// Extremely heavy magnitude operation.
    /// Computed with a default precision of 1000 digits.
    pub fn length(&self) -> BigRational {
        ratio_sqrt(self.length_squared(), BigUint::new(vec![1000]))
    }
    pub fn length_squared(&self) -> BigRational {
        // &self.x * &self.x + &self.y * &self.y + &self.z * &self.z
        let mut sum = new_bri(0);
        for d in &self.data {
            sum += d * d;
        }
        sum
    }
    pub fn length_recip(&self) -> BigRational {
        self.length().recip()
    }
    pub fn distance(&self, v: &BigMatrix<I, 1>) -> BigRational {
        (self - v).length()
    }
    pub fn distance_squared(&self, v: &BigMatrix<I, 1>) -> BigRational {
        (self - v).length_squared()
    }
    pub fn dot(&self, v: &BigMatrix<I, 1>) -> BigRational {
        let mut sum = new_bri(0);
        for index in 0..I {
            sum += &self.data[index] * &v.data[index];
        }
        sum
    }
    pub fn normalize(&self) -> Self {
        if !self.data.iter().any(|d| d.numer().magnitude() != &BigUint::new(vec![0])) {
            panic!("Multidimensional arbitrary precision vector of length 0 cannot be normalized.")
        }
        self / self.length()
    }
    pub fn try_normalize(&self) -> Option<Self> {
        if !self.data.iter().any(|d| d.numer().magnitude() != &BigUint::new(vec![0])) {
            return None;
        }
        Some(self / self.length())
    }
    pub fn normalize_or_zero(&self) -> Self {
        if !self.data.iter().any(|d| d.numer().magnitude() != &BigUint::new(vec![0])) {
            return Self::splat(new_bri(0));
        }
        self / self.length()
    }
    pub fn is_normalized(&self) -> bool {
        let cutoff = BigRational::new_raw(BigInt::new(num::bigint::Sign::Plus, vec![1]), BigInt::new(num::bigint::Sign::Plus, vec![1000000]));
        let min = new_bri(1) - &cutoff;
        let max = new_bri(1) + &cutoff;
        let length = self.length();
        min <= length && length <= max
    }
    pub fn project_onto(&self, rhs: &BigMatrix<I, 1>) -> Self {
        rhs * (self.dot(rhs) / rhs.dot(rhs))
    }
    pub fn reject_from(&self, rhs: &BigMatrix<I, 1>) -> Self {
        (self - rhs) * (self.dot(rhs) / rhs.dot(rhs))
    }
    pub fn lerp(&self, rhs: &BigMatrix<I, 1>, s: &BigRational) -> Self {
        self + (&(rhs - self) * s)
    }
    pub fn clamp_length(&self, min: &BigRational, max: &BigRational) -> Self {
        if min > max {
            panic!("Min is bigger than max in clamp length.")
        }
        let length = self.length();
        if &length < min || max < &length {
            return self.normalize() * length;
        }
        self.clone()
    }
    pub fn clamp_length_max(&self, max: &BigRational) -> Self {
        let length = self.length();
        if max < &length {
            return self.normalize() * length;
        }
        self.clone()
    }
    pub fn clamp_length_min(&self, min: &BigRational) -> Self {
        let length = self.length();
        if &length < min {
            return self.normalize() * length;
        }
        self.clone()
    }
}

/// Dimension 3 vectors.
impl BigMatrix<3, 1> {
    pub fn x() -> Self {
        Self {
            data: vec![new_bri(1), new_bri(0), new_bri(0)],
            rows: 3,
            cols: 1,
        }
    }
    pub fn y() -> Self {
        Self {
            data: vec![new_bri(0), new_bri(1), new_bri(0)],
            rows: 3,
            cols: 1,
        }
    }
    pub fn z() -> Self {
        Self {
            data: vec![new_bri(0), new_bri(0), new_bri(1)],
            rows: 3,
            cols: 1,
        }
    }
    pub fn new_vec3_f64(x: f64, y: f64, z: f64) -> Self {
        Self {
            data: vec![new_brf(x), new_brf(y), new_brf(z)],
            rows: 3,
            cols: 1,
        }
    }
    pub fn cross(&self, v: &BigMatrix<3, 1>) -> Self {
        Self {
            data: vec![
                (&self.data[1] * &v.data[2]) - (&self.data[2] * &v.data[1]),
                (&self.data[2] * &v.data[0]) - (&self.data[0] * &v.data[2]),
                (&self.data[0] * &v.data[1]) - (&self.data[1] * &v.data[0])
            ],
            cols: 1,
            rows: 3,
        }
    }
}

impl<const I: usize, const J: usize> fmt::Display for BigMatrix<I, J> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ ")?;
        let mut comma = false;
        for j in 0..J {
            if comma {
                write!(f, ", ")?;
            }
            write!(f, "[")?;
            let mut inner_comma = false;
            for i in 0..I {
                if inner_comma {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.get(i, j))?;
                inner_comma = true;
            }
            write!(f, "]")?;
            comma = true;
        }
        write!(f, " }}")
    }
}

fn new_big_matrix_add<const I: usize, const J: usize>(lhs: &BigMatrix<I, J>, rhs: &BigMatrix<I, J>) -> BigMatrix<I, J> {
    let mut data = Vec::new();
    for i in 0..(I * J) {
        data.push(&lhs.data[i] + &rhs.data[i]);
    }
    BigMatrix::<I, J> {
        data,
        rows: I,
        cols: J,
    }
}
fn new_big_matrix_sub<const I: usize, const J: usize>(lhs: &BigMatrix<I, J>, rhs: &BigMatrix<I, J>) -> BigMatrix<I, J> {
    let mut data = Vec::new();
    for i in 0..(I * J) {
        data.push(&lhs.data[i] - &rhs.data[i]);
    }
    BigMatrix::<I, J> {
        data,
        rows: I,
        cols: J,
    }
}
fn new_big_matrix_mul<const I: usize, const J: usize, const K: usize>(lhs: &BigMatrix<I, J>, rhs: &BigMatrix<J, K>) -> BigMatrix<I, K> {
    let mut data: Vec<BigRational> = Vec::new();
    for out_row_index in 0..I {
        for out_col_index in 0..K {
            let mut sum = new_bri(0);
            for col_row_index in 0..J {
                sum += lhs.get(out_row_index, col_row_index) * rhs.get(col_row_index, out_col_index);
            }
        }
    }
    BigMatrix::<I, K> {
        data,
        rows: I,
        cols: K,
    }
}
fn new_big_matrix_mul_scalar<const I: usize, const J: usize>(lhs: &BigMatrix<I, J>, rhs: &BigRational) -> BigMatrix<I, J> {
    let mut data = Vec::new();
    for i in 0..(I * J) {
        data.push(&lhs.data[i] * rhs);
    }
    BigMatrix::<I, J> {
        data,
        rows: I,
        cols: J,
    }
}
fn new_big_matrix_div_scalar<const I: usize, const J: usize>(lhs: &BigMatrix<I, J>, rhs: &BigRational) -> BigMatrix<I, J> {
    let mut data = Vec::new();
    for i in 0..(I * J) {
        data.push(&lhs.data[i] / rhs);
    }
    BigMatrix::<I, J> {
        data,
        rows: I,
        cols: J,
    }
}

impl<const I: usize, const J: usize> Add<BigMatrix<I, J>> for BigMatrix<I, J> {
    type Output = BigMatrix<I, J>;
    fn add(self, rhs: BigMatrix<I, J>) -> Self::Output {
        new_big_matrix_add(&self, &rhs)
    }
}
impl<const I: usize, const J: usize> Add<&BigMatrix<I, J>> for BigMatrix<I, J> {
    type Output = BigMatrix<I, J>;
    fn add(self, rhs: &BigMatrix<I, J>) -> Self::Output {
        new_big_matrix_add(&self, &rhs)
    }
}
impl<const I: usize, const J: usize> Add<BigMatrix<I, J>> for &BigMatrix<I, J> {
    type Output = BigMatrix<I, J>;
    fn add(self, rhs: BigMatrix<I, J>) -> Self::Output {
        new_big_matrix_add(&self, &rhs)
    }
}
impl<const I: usize, const J: usize> Add<&BigMatrix<I, J>> for &BigMatrix<I, J> {
    type Output = BigMatrix<I, J>;
    fn add(self, rhs: &BigMatrix<I, J>) -> Self::Output {
        new_big_matrix_add(&self, &rhs)
    }
}

impl<const I: usize, const J: usize> Sub<BigMatrix<I, J>> for BigMatrix<I, J> {
    type Output = BigMatrix<I, J>;
    fn sub(self, rhs: BigMatrix<I, J>) -> Self::Output {
        new_big_matrix_sub(&self, &rhs)
    }
}
impl<const I: usize, const J: usize> Sub<&BigMatrix<I, J>> for BigMatrix<I, J> {
    type Output = BigMatrix<I, J>;
    fn sub(self, rhs: &BigMatrix<I, J>) -> Self::Output {
        new_big_matrix_sub(&self, &rhs)
    }
}
impl<const I: usize, const J: usize> Sub<BigMatrix<I, J>> for &BigMatrix<I, J> {
    type Output = BigMatrix<I, J>;
    fn sub(self, rhs: BigMatrix<I, J>) -> Self::Output {
        new_big_matrix_sub(&self, &rhs)
    }
}
impl<const I: usize, const J: usize> Sub<&BigMatrix<I, J>> for &BigMatrix<I, J> {
    type Output = BigMatrix<I, J>;
    fn sub(self, rhs: &BigMatrix<I, J>) -> Self::Output {
        new_big_matrix_sub(&self, &rhs)
    }
}

impl<const I: usize, const J: usize, const K: usize> Mul<BigMatrix<J, K>> for BigMatrix<I, J> {
    type Output = BigMatrix<I, K>;
    fn mul(self, rhs: BigMatrix<J, K>) -> Self::Output {
        new_big_matrix_mul(&self, &rhs)
    }
}
impl<const I: usize, const J: usize, const K: usize> Mul<&BigMatrix<J, K>> for BigMatrix<I, J> {
    type Output = BigMatrix<I, K>;
    fn mul(self, rhs: &BigMatrix<J, K>) -> Self::Output {
        new_big_matrix_mul(&self, &rhs)
    }
}
impl<const I: usize, const J: usize, const K: usize> Mul<BigMatrix<J, K>> for &BigMatrix<I, J> {
    type Output = BigMatrix<I, K>;
    fn mul(self, rhs: BigMatrix<J, K>) -> Self::Output {
        new_big_matrix_mul(&self, &rhs)
    }
}
impl<const I: usize, const J: usize, const K: usize> Mul<&BigMatrix<J, K>> for &BigMatrix<I, J> {
    type Output = BigMatrix<I, K>;
    fn mul(self, rhs: &BigMatrix<J, K>) -> Self::Output {
        new_big_matrix_mul(&self, &rhs)
    }
}

impl<const I: usize, const J: usize> Mul<BigRational> for BigMatrix<I, J> {
    type Output = BigMatrix<I, J>;
    fn mul(self, rhs: BigRational) -> Self::Output {
        new_big_matrix_mul_scalar(&self, &rhs)
    }
}
impl<const I: usize, const J: usize> Mul<&BigRational> for BigMatrix<I, J> {
    type Output = BigMatrix<I, J>;
    fn mul(self, rhs: &BigRational) -> Self::Output {
        new_big_matrix_mul_scalar(&self, &rhs)
    }
}
impl<const I: usize, const J: usize> Mul<BigRational> for &BigMatrix<I, J> {
    type Output = BigMatrix<I, J>;
    fn mul(self, rhs: BigRational) -> Self::Output {
        new_big_matrix_mul_scalar(&self, &rhs)
    }
}
impl<const I: usize, const J: usize> Mul<&BigRational> for &BigMatrix<I, J> {
    type Output = BigMatrix<I, J>;
    fn mul(self, rhs: &BigRational) -> Self::Output {
        new_big_matrix_mul_scalar(&self, &rhs)
    }
}

impl<const I: usize, const J: usize> Div<BigRational> for BigMatrix<I, J> {
    type Output = BigMatrix<I, J>;
    fn div(self, rhs: BigRational) -> Self::Output {
        new_big_matrix_div_scalar(&self, &rhs)
    }
}
impl<const I: usize, const J: usize> Div<&BigRational> for BigMatrix<I, J> {
    type Output = BigMatrix<I, J>;
    fn div(self, rhs: &BigRational) -> Self::Output {
        new_big_matrix_div_scalar(&self, &rhs)
    }
}
impl<const I: usize, const J: usize> Div<BigRational> for &BigMatrix<I, J> {
    type Output = BigMatrix<I, J>;
    fn div(self, rhs: BigRational) -> Self::Output {
        new_big_matrix_div_scalar(&self, &rhs)
    }
}
impl<const I: usize, const J: usize> Div<&BigRational> for &BigMatrix<I, J> {
    type Output = BigMatrix<I, J>;
    fn div(self, rhs: &BigRational) -> Self::Output {
        new_big_matrix_div_scalar(&self, &rhs)
    }
}

impl From<bevy::math::Vec3> for BigMatrix<3, 1> {
    fn from(v: bevy::math::Vec3) -> Self {
        Self::new_vec3_f64(v.x.into(), v.y.into(), v.z.into())
    }
}

type BigVector3 = BigMatrix<3, 1>;

#[test]
fn test_arbitrary_mat() {
	let mat1 = BigMatrix::<3, 3>::new();
	let mat2 = BigMatrix::<3, 1>::new();
}

#[test]
fn test_simple_mat() {
	let t1 = BigVector3::x();
	let t2 = BigVector3::y();
	let t3 = BigVector3::z();
	let s = new_bri(2);
	println!("t1 = {}", t1.clone());
	println!("t2 = {}", t2.clone());
	println!("t3 = {}", t3.clone());
	println!("s = {}", s.clone());
	let ta = t1 + t2;
	println!("ta = t1 + t2 = {}", ta.clone());
	let ts = ta - t3;
	println!("ts = ta - ts = {}", ts.clone());
	let tm = ts * s.clone();
	println!("tm = ts * s = {}", tm.clone());
    let td = tm / s;
	println!("td = tm / s = {}", td);
}
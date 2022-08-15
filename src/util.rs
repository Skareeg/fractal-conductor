use arrayvec::ArrayVec;
use bevy::reflect::Array;
use num::BigRational;

type BigVector = BigMatrix<1, 3>;

#[derive(Clone, Debug, Default)]
pub struct BigMatrix<const R: usize, const C: usize> {
    pub data: ArrayVec<ArrayVec<BigRational, C>, R>,
}

impl<const R: usize, const C: usize> std::ops::Add for BigMatrix<R, C> {
    type Output = BigMatrix<R, C>;
    fn add(self, rhs: Self) -> Self::Output {
        let mut data = ArrayVec::<ArrayVec::<BigRational, C>, R>::new();
        for c in 1..C {
            for r in 1..R {
                data[c][r] = self.data[c][r] + rhs.data[c][r];
            }
        }
        Self { data }
    }
}

impl<const R: usize, const C: usize> std::ops::AddAssign for BigMatrix<R, C> {
    fn add_assign(&mut self, rhs: Self) {
        for c in 1..C {
            for r in 1..R {
                self.data[c][r] += rhs.data[c][r];
            }
        }
    }
}

impl<const R1: usize, const C: usize, const R2: usize> std::ops::Mul<BigMatrix<C, R2>> for BigMatrix<R1, C> {
    type Output = BigMatrix<R1, R2>;
    fn mul(self, rhs: BigMatrix<C, R2>) -> Self::Output {
        let mut datum = ArrayVec::<ArrayVec::<BigRational, R1>, R2>::new();
        for c in 1..R1 {
            for r in 1..R2 {
                let mut sum = num::BigRational::new(0.into(), 1.into());
                for i in 1..C {
                    sum += self.data[i][r] * rhs.data[c][i];
                }
                datum[c][r] = sum;
            }
        }
        Self { data: datum }
    }
}
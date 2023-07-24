#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::octet::Octet;
use crate::octets::{add_assign, fused_addassign_mul_scalar_binary, mulassign_scalar};
use crate::octets::{fused_addassign_mul_scalar, BinaryOctetVec};
#[cfg(feature = "benchmarking")]
use std::mem::size_of;

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct DenseOctetMatrix {
    height: usize,
    width: usize,
    elements: Vec<u8>,
}

impl DenseOctetMatrix {
    pub fn new(height: usize, width: usize, _: usize) -> DenseOctetMatrix {
        let elements: Vec<u8> = vec![0; width * height];
        DenseOctetMatrix {
            height,
            width,
            elements,
        }
    }

    pub fn fma_sub_row(
        &mut self,
        row: usize,
        mut start_col: usize,
        scalar: &Octet,
        other: &BinaryOctetVec,
    ) {
        start_col += row * self.width;
        fused_addassign_mul_scalar_binary(
            &mut self.elements[start_col..(start_col + other.len())],
            other,
            scalar,
        );
    }

    pub fn set(&mut self, i: usize, j: usize, value: Octet) {
        self.elements[i * self.width + j] = value.byte();
    }

    pub fn height(&self) -> usize {
        self.height
    }

    #[cfg(all(test, feature = "std"))]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg(feature = "benchmarking")]
    pub fn size_in_bytes(&self) -> usize {
        let mut bytes = size_of::<Self>();
        bytes += size_of::<u8>() * self.height * self.width;

        bytes
    }

    pub fn mul_assign_row(&mut self, row: usize, value: &Octet) {
        mulassign_scalar(
            &mut self.elements[row * self.width..(row + 1) * self.width],
            value,
        );
    }

    pub fn get(&self, i: usize, j: usize) -> Octet {
        Octet::new(self.elements[i * self.width + j])
    }

    pub fn swap_rows(&mut self, i: usize, j: usize) {
        debug_assert!(i < self.height && j < self.height);
        if i != j {
            // SAFETY: elements size is guaranteed to be multiple of width
            unsafe {
                let base = self.elements.as_mut_ptr();
                std::ptr::swap_nonoverlapping(
                    base.add(i * self.width),
                    base.add(j * self.height),
                    self.width,
                );
            }
        }
    }

    pub fn swap_columns(&mut self, i: usize, j: usize, start_row_hint: usize) {
        debug_assert!(i < self.width && j < self.width);

        let base = self.elements.as_mut_ptr();

        for row in start_row_hint..self.height {
            let offset = row * self.width;
            // SAFETY: i and j are in elements range
            unsafe {
                std::ptr::swap(base.add(offset + i), base.add(offset + j));
            }
        }
    }

    pub fn fma_rows(&mut self, dest: usize, multiplicand: usize, scalar: &Octet) {
        assert_ne!(dest, multiplicand);
        let (dest_row, temp_row) =
            get_both_indices(&mut self.elements, self.width, dest, multiplicand);

        if *scalar == Octet::one() {
            add_assign(dest_row, temp_row);
        } else {
            fused_addassign_mul_scalar(dest_row, temp_row, scalar);
        }
    }
}

fn get_both_indices(vector: &mut [u8], width: usize, i: usize, j: usize) -> (&mut [u8], &mut [u8]) {
    debug_assert_ne!(i, j);
    debug_assert!(i * width < vector.len());
    debug_assert!(j * width < vector.len());
    if i < j {
        let (first, last) = vector.split_at_mut(j * width);
        return (&mut first[i * width..(i + 1) * width], &mut last[0..width]);
    } else {
        let (first, last) = vector.split_at_mut(i * width);
        return (&mut last[0..width], &mut first[j * width..(j + 1) * width]);
    }
}

//! A hyperplane classifier checks whether the dot product of a vector with some normal
//! is within a specified range.

use rand::{Rng, ThreadRng};
use Classifier;
use Generator;

// TODO: allow arbitrary subsets of the dimensions
// TODO: for now we'll just use a 2d normal for the image example
// TODO: allow upper and lower thresholds
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Plane {
    normal: [f64; 2],
    threshold: f64
}

impl Classifier for Plane {
    // TODO: work for more than just 2d
    fn classify(&self, sample: &Vec<f64>) -> bool {
        sample[0] * self.normal[0] + sample[1] * self.normal[1] > self.threshold
    }
}

impl Default for Plane {
    fn default() -> Plane {
        Plane {
            normal: [1f64, 0f64],
            threshold: 0f64
        }
    }
}

pub struct PlaneGenerator {
    pub rng: ThreadRng,
    pub num_dims: usize,
    pub min_thresh: f64,
    pub max_thresh: f64
}

impl Generator for PlaneGenerator {
    type Classifier = Plane;

    fn generate(&mut self, count: usize) -> Vec<Plane> {
        let mut planes = vec![Plane::default(); count];
        for i in 0..count {
            // TODO: allow more dimensions, generate angle uniformly
            let x = self.rng.gen_range(0f64, 1f64);
            let y = self.rng.gen_range(0f64, 1f64);
            let thresh = self.rng.gen_range(self.min_thresh, self.max_thresh);
            planes[i] = Plane {
                normal: [x, y],
                threshold: thresh
            }
        }
        planes
    }
}

//! A stump is a single-dimensional axis-aligned linear classifier.

use rand::{Rng, ThreadRng};
use Classifier;
use Generator;

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Stump {
    dimension: usize,
    threshold: f64
}

impl Classifier for Stump {
    fn classify(&self, sample: &Vec<f64>) -> bool {
        sample[self.dimension] > self.threshold
    }
}

impl Default for Stump {
    fn default() -> Stump {
        Stump {
            dimension: 0usize,
            threshold: 0f64
        }
    }
}

pub struct StumpGenerator {
    pub rng: ThreadRng,
    pub num_dims: usize,
    pub min_thresh: f64,
    pub max_thresh: f64
}

impl Generator for StumpGenerator {
    type Classifier = Stump;

    fn generate(&mut self, count: usize) -> Vec<Stump> {
        let mut stumps = vec![Stump::default(); count];
        for i in 0..count {
            let dim = self.rng.gen_range(0, self.num_dims);
            let thresh = self.rng.gen_range(self.min_thresh, self.max_thresh);
            stumps[i] = Stump {
                dimension: dim,
                threshold: thresh
            }
        }
        stumps
    }
}

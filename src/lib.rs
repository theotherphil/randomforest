
extern crate rand;
use std::f64;
use rand::{Rng, thread_rng, ThreadRng};

#[derive(Debug, PartialEq, Clone)]
pub struct Dataset {
    labels: Vec<usize>,
    data: Vec<Vec<f64>>
}

impl Dataset {
    fn empty() -> Dataset {
        Dataset {
            labels: Vec::new(),
            data: Vec::new()
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
struct Stump {
    dimension: usize,
    threshold: f64
}

impl Stump {
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

#[derive(Debug, Clone)]
pub struct Distribution {
    probs: Vec<f64>
}

impl Default for Distribution {
    fn default() -> Distribution {
        Distribution {
            probs: vec![]
        }
    }
}

pub struct Tree {
    depth: usize,
    nodes: Vec<Stump>,
    leaves: Vec<Distribution>
}

impl Tree {
    fn num_nodes(&self) -> usize {
        2usize.pow(self.depth as u32) - 1
    }

    pub fn classify(&self, sample: &Vec<f64>) -> Distribution {
        let mut idx = 0;

        loop {
            let decision = self.nodes[idx].classify(sample);
            idx = if decision { right_child_idx(idx) } else { left_child_idx(idx) };
            let num_nodes = self.num_nodes();
            if idx > num_nodes {
                return self.leaves[idx - num_nodes].clone();
            }
        }
    }
}

pub fn left_child_idx(idx: usize) -> usize {
    2 * idx + 1
}

pub fn right_child_idx(idx: usize) -> usize {
    2 * idx + 2
}

pub fn train_tree(depth: u32, num_classes: usize, num_candidates: usize, data: &Dataset) -> Tree {
    let mut generator = StumpGenerator {
        rng: thread_rng(),
        num_dims: data.data[0].len(),
        min_thresh: 0f64,
        max_thresh: 1f64
    };

    let num_nodes = 2usize.pow(depth) - 1;
    let num_leaves = num_nodes + 1;

    let mut nodes = vec![Stump::default(); num_nodes];
    let mut nodes_data = vec![Dataset::empty(); num_nodes];
    let mut leaves = vec![Distribution::default(); num_leaves];

    nodes_data[0] = data.clone();

    // Invariant: nodes_data[i] has already been populated, but nodes[i] has not.
    for i in 0..num_nodes {
        let candidates = generator.generate(num_candidates);

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_candidate = candidates[0];
        let mut best_split = (Dataset::empty(), Dataset::empty());

        for c in candidates {
            let (left, right) = split(&c, &nodes_data[i]);

            let gain = weighted_entropy_drop(num_classes,
                                             &nodes_data[i].labels,
                                             &left.labels,
                                             &right.labels);

            if gain > best_gain {
                best_gain = gain;
                best_candidate = c;
                best_split = (left, right);
            }
        }

        let left = left_child_idx(i);
        let right = right_child_idx(i);

        if left >= num_nodes {
            let left_leaf_idx = left - num_nodes;
            let right_leaf_idx = right - num_nodes;

            leaves[left_leaf_idx] = read_class_probabilities(num_classes, &best_split.0.labels);
            leaves[right_leaf_idx] = read_class_probabilities(num_classes, &best_split.1.labels);
        }
        else {
            nodes[i] = best_candidate;
            nodes_data[left] = best_split.0;
            nodes_data[right_child_idx(i)] = best_split.1;
        }
    }

    Tree {
        depth: depth as usize,
        nodes: nodes,
        leaves: leaves
    }
}

fn read_class_probabilities(num_classes: usize, labels: &Vec<usize>) -> Distribution {
    let mut probs = vec![0f64; num_classes];
    for l in labels {
        probs[*l] += 1.0
    }
    let count = labels.len() as f64;
    for n in 0..num_classes {
        probs[n] /= count;
    }
    Distribution {
        probs: probs
    }
}

struct StumpGenerator {
    rng: ThreadRng,
    num_dims: usize,
    min_thresh: f64,
    max_thresh: f64
}

impl StumpGenerator {
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

fn split(stump: &Stump, data: &Dataset) -> (Dataset, Dataset) {
    let mut left = Dataset::empty();
    let mut right = Dataset::empty();

    for i in 0..data.labels.len() {
        let l = data.labels[i];
        let ref d = data.data[i];

        let c = stump.classify(&d);
        if c {
            right.labels.push(l);
            right.data.push(d.clone());
        }
        else {
            left.labels.push(l);
            left.data.push(d.clone());
        }
    }

    (left, right)
}

fn weighted_entropy_drop(num_classes: usize,
                         parent: &[usize],
                         left: &[usize],
                         right: &[usize]) -> f64 {
    let weighted_left = entropy(left.iter(), num_classes) * left.len() as f64;
    let weighted_right = entropy(right.iter(), num_classes) * right.len() as f64;
    entropy(parent.iter(), num_classes) - weighted_left - weighted_right
}

// Could allow non-usize labels. but then we'd need a map from label to index
fn entropy<'a, I>(labels: I, num_classes: usize) -> f64
    where I: Iterator<Item=&'a usize>
{
    let mut class_counts = vec![0f64; num_classes];
    let mut num_labels = 0f64;
    for l in labels {
        class_counts[*l] += 1f64;
        num_labels += 1f64;
    }

    let mut entropy = 0f64;
    for c in class_counts {
        if c == 0f64 {
            continue;
        }
        let prob = c / num_labels;
        entropy -= prob * prob.log2();
    }

    entropy
}

#[cfg(test)]
mod tests {
    use super::entropy;

    #[test]
    fn test_entropy_usize() {
        assert_eq!(entropy(vec![1usize, 1].into_iter(), 2), 0f64);
        assert_eq!(entropy(vec![0usize].into_iter(), 1), 0f64);
        assert_eq!(entropy(vec![0usize, 1].into_iter(), 2), 1f64);
        assert_eq!(entropy(vec![0usize, 1, 2, 3].into_iter(), 4), 2f64);
    }

    #[test]
    fn test_weighted_entropy_drop() {
        // do something...
    }
}

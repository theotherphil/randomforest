

#![cfg_attr(test, feature(test))]

#[cfg(test)]
extern crate test;
extern crate rand;

pub mod stump;
pub mod hyperplane;

use std::f64;

#[derive(Debug, PartialEq, Clone)]
pub struct Dataset {
    pub labels: Vec<usize>,
    pub data: Vec<Vec<f64>>
}

impl Dataset {
    fn empty() -> Dataset {
        Dataset {
            labels: Vec::new(),
            data: Vec::new()
        }
    }
}

pub trait Classifier {
    fn classify(&self, sample: &Vec<f64>) -> bool;
}

pub trait Generator {
    type Classifier;
    fn generate(&mut self, count: usize) -> Vec<Self::Classifier>;
}

#[derive(Debug, Clone)]
pub struct Distribution {
    pub probs: Vec<f64>
}

impl Default for Distribution {
    fn default() -> Distribution {
        Distribution {
            probs: vec![]
        }
    }
}

pub struct Tree<C> {
    depth: usize,
    nodes: Vec<C>,
    leaves: Vec<Distribution>
}

impl<C: Classifier> Tree<C> {
    fn num_nodes(&self) -> usize {
        2usize.pow(self.depth as u32) - 1
    }

    pub fn classify(&self, sample: &Vec<f64>) -> Distribution {
        let mut idx = 0;

        loop {
            let decision = self.nodes[idx].classify(sample);
            idx = if decision { right_child_idx(idx) } else { left_child_idx(idx) };
            let num_nodes = self.num_nodes();
            if idx >= num_nodes {
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

pub struct Forest<C> {
    trees: Vec<Tree<C>>,
    num_classes: usize
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct ForestParameters {
    pub num_trees: usize,
    pub depth: usize,
    pub num_classes: usize, // TODO: really a property of dataset
    pub num_candidates: usize
}

impl<C: Classifier + Default + Clone> Forest<C> {
    pub fn train<G>(params: ForestParameters, generator: &mut G, data: &Dataset) -> Forest<C>
        where G: Generator<Classifier=C>
    {
        let trees = (0..params.num_trees)
            .map(|t| {
                println!("training tree {:?}", t);
                train_tree(params.depth, params.num_classes, params.num_candidates, generator, data)
            })
            .collect();

        Forest {
            trees: trees,
            num_classes: params.num_classes
        }
    }

    pub fn classify(&self, sample: &Vec<f64>) -> Distribution {
        let mut probs = vec![0f64; self.num_classes];

        for tree in self.trees.iter() {
            let dist = tree.classify(sample);
            for i in 0..self.num_classes {
                probs[i] += dist.probs[i];
            }
        }

        for i in 0..self.num_classes {
            probs[i] /= self.trees.len() as f64;
        }

        Distribution {
            probs: probs
        }
    }
}

fn train_tree<C, G>(depth: usize,
                    num_classes: usize,
                    num_candidates: usize,
                    generator: &mut G,
                    data: &Dataset) -> Tree<C>
    where C: Classifier + Default + Clone,
          G: Generator<Classifier=C>
 {
    let num_nodes = 2usize.pow(depth as u32) - 1;
    let num_leaves = num_nodes + 1;

    let mut nodes = vec![C::default(); num_nodes];
    let mut nodes_data = vec![Dataset::empty(); num_nodes];
    let mut leaves = vec![Distribution { probs: vec![0f64; num_classes] }; num_leaves];

    nodes_data[0] = data.clone();

    // Invariant: nodes_data[i] has already been populated, but nodes[i] has not.
    for i in 0..num_nodes {
        let candidates = generator.generate(num_candidates);

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_candidate = candidates[0].clone();
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
            nodes_data[right] = best_split.1;
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
    let count = labels.len() as f64;
    // There's no early stopping rule, so there might not be any entries
    if count == 0f64 {
        return Distribution { probs: probs };
    }
    for l in labels {
        probs[*l] += 1.0
    }
    for n in 0..num_classes {
        probs[n] /= count;
    }
    Distribution {
        probs: probs
    }
}

fn split<C: Classifier>(classifier: &C, data: &Dataset) -> (Dataset, Dataset) {
    let mut left = Dataset::empty();
    let mut right = Dataset::empty();

    for i in 0..data.labels.len() {
        let ref d = data.data[i];
        let dataset = if classifier.classify(&d) { &mut right } else { &mut left };
        dataset.labels.push(data.labels[i]);
        dataset.data.push(d.clone());
    }

    (left, right)
}

fn weighted_entropy_drop(num_classes: usize,
                         parent: &[usize],
                         left: &[usize],
                         right: &[usize]) -> f64 {
    let count = parent.len() as f64;
    let weighted_left = entropy(left.iter(), num_classes) * left.len() as f64 / count;
    let weighted_right = entropy(right.iter(), num_classes) * right.len() as f64 / count;
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
    use super::{Dataset, entropy, train_tree};
    use super::stump::{Stump, StumpGenerator};
    use test;
    use rand::{Rng, thread_rng};

    #[test]
    fn test_entropy_usize() {
        assert_eq!(entropy(vec![1usize, 1].iter(), 2), 0f64);
        assert_eq!(entropy(vec![0usize].iter(), 1), 0f64);
        assert_eq!(entropy(vec![0usize, 1].iter(), 2), 1f64);
        assert_eq!(entropy(vec![0usize, 1, 2, 3].iter(), 4), 2f64);
    }

    #[test]
    fn test_weighted_entropy_drop() {
        // do something...
    }

    #[bench]
    fn bench_train_stumps(b: &mut test::Bencher) {
        let num_samples = 100;
        let num_classes = 10;
        let num_dimensions = 5;
        let num_candidates = 10;
        let depth = 6;

        let mut rng = thread_rng();
        let mut labels = vec![];
        let mut data = vec![];

        for i in 0..num_samples {
            let l = rng.gen_range(0, num_classes);
            labels.push(l);
            let d = (0..num_dimensions)
                .map(|n| rng.gen_range(0f64, 1f64))
                .collect();
            data.push(d);
        }

        let dataset = Dataset {
            labels: labels,
            data: data
        };

        let mut generator = StumpGenerator {
            rng: rng,
            num_dims: num_dimensions,
            min_thresh: 0f64,
            max_thresh: 1f64
        };

        b.iter(|| {
            train_tree(depth, num_classes, num_candidates, &mut generator, &dataset)
        });
   }
}

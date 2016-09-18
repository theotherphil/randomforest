
// TODO:
// * lots more testing
// * more benchmarks, showing scaling with various dimensions
// * support incomplete trees
// * more stopping rules, e.g. min split size, min information gain
// * support other information gains (e.g. for Hough forests)
// * serialisation
// * bagging
// * tests on realistically sized datasets
// * parallelise training by tree
// * progress logging and performance stats
// * debugging tools
// * traits to use via Box which hide the types, e.g. Forest trait
//   which doesn't expose the classifier or stopping rule types
// * conic classifiers
// * implement Hough forest object detection paper as demo
// * draw original data points on demo coloured images, with black outline
// * visualisation for leaves - e.g. visualise histograms for Distribution leaves,
//   or distribution of offset from centre for Hough leaves
// * clone of Selection in train_tree can probably be replaced using split_at_mut
// * profiling
// * GPUs?

#![cfg_attr(test, feature(test))]

#[cfg(test)]
extern crate test;
extern crate rand;

pub mod dataset;
pub mod distribution;
pub mod stump;
pub mod hyperplane;

use std::f64;
use dataset::{Dataset, DatasetView, Selection};

pub trait Classifier {
    fn classify(&self, sample: &Vec<f64>) -> bool;
}

pub trait Generator {
    type Classifier;
    fn generate(&mut self, count: usize) -> Vec<Self::Classifier>;
}

pub trait Leaf {
    fn from_dataset(dataset: &DatasetView) -> Self;
    fn empty(num_classes: usize) -> Self;
    // This probably shouldn't be on Leaf, shouldn't require
    // an extra num_classes argument, and needn't
    // return Self in general, but it'll do for now.
    fn combine(leaves: &[Self], num_classes: usize) -> Self where Self: Sized;
}

pub struct Forest<C, L> {
    trees: Vec<Tree<C, L>>,
    num_classes: usize
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct ForestParameters {
    pub num_trees: usize,
    pub depth: usize,
    pub num_candidates: usize
}

pub struct Tree<C, L> {
    // Length of longest path from root node to a leaf.
    depth: usize,
    // Implicit binary tree of classifier nodes.
    nodes: Vec<C>,
    // Leaf nodes.
    leaves: Vec<L>
}

impl<C: Classifier + Default + Clone, L: Leaf + Clone> Forest<C, L> {
    pub fn train<G>(params: ForestParameters, generator: &mut G, data: &Dataset) -> Forest<C, L>
        where G: Generator<Classifier=C>
    {
        let trees = (0..params.num_trees)
            .map(|t| {
                println!("training tree {:?}", t);
                Tree::<C, L>::train(params.depth, params.num_candidates, generator, data)
            })
            .collect();

        Forest {
            trees: trees,
            num_classes: data.num_classes
        }
    }

    pub fn classify(&self, sample: &Vec<f64>) -> L {
        let mut leaves = Vec::with_capacity(self.trees.len());

        for tree in self.trees.iter() {
            let dist = tree.classify(sample);
            leaves.push(dist);
        }

        L::combine(&leaves, self.num_classes)
    }
}

impl<C: Classifier + Default + Clone, L: Leaf + Clone> Tree<C, L> {
    fn num_nodes(&self) -> usize {
        2usize.pow(self.depth as u32) - 1
    }

    pub fn classify(&self, sample: &Vec<f64>) -> L {
        let mut idx = 0;

        loop {
            let decision = self.nodes[idx].classify(sample);
            idx = if decision { Self::right_child_idx(idx) } else { Self::left_child_idx(idx) };
            let num_nodes = self.num_nodes();
            if idx >= num_nodes {
                return self.leaves[idx - num_nodes].clone();
            }
        }
    }

    fn train<G>(depth: usize,
                        num_candidates: usize,
                        generator: &mut G,
                        data: &Dataset) -> Tree<C, L>
        where G: Generator<Classifier=C>
     {
        let num_nodes = 2usize.pow(depth as u32) - 1;
        let num_leaves = num_nodes + 1;

        let mut nodes = vec![C::default(); num_nodes];
        let mut nodes_data = vec![Selection(vec![]); num_nodes];
        let mut leaves = vec![L::empty(data.num_classes); num_leaves];

        nodes_data[0] = Selection((0..data.labels.len()).collect());

        // Invariant: nodes_data[i] has already been populated, but nodes[i] has not.
        for i in 0..num_nodes {
            let candidates = generator.generate(num_candidates);

            let mut best_gain = f64::NEG_INFINITY;
            let mut best_candidate = candidates[0].clone();
            let mut best_split = (Selection(vec![]), Selection(vec![]));

            for c in candidates {
                let (left, right) = Self::split(&c, data, &nodes_data[i].clone()); // TODO: don't clone

                let gain = weighted_entropy_drop(data.num_classes,
                                                 data,
                                                 &nodes_data[i],
                                                 &left,
                                                 &right);

                if gain > best_gain {
                    best_gain = gain;
                    best_candidate = c;
                    best_split = (left, right);
                }
            }

            let left = Self::left_child_idx(i);
            let right = Self::right_child_idx(i);

            if left >= num_nodes {
                let left_leaf_idx = left - num_nodes;
                leaves[left_leaf_idx] = L::from_dataset(&data.select(&best_split.0));

                let right_leaf_idx = right - num_nodes;
                leaves[right_leaf_idx] = L::from_dataset(&data.select(&best_split.1));
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

    fn split(classifier: &C, data: &Dataset, selection: &Selection) -> (Selection, Selection) {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &i in &selection.0 {
            let ref d = data.data[i];
            let set = if classifier.classify(&d) { &mut right_indices } else { &mut left_indices };
            set.push(i);
        }

        (Selection(left_indices), Selection(right_indices))
    }

    fn left_child_idx(idx: usize) -> usize {
        2 * idx + 1
    }

    fn right_child_idx(idx: usize) -> usize {
        2 * idx + 2
    }
}

fn weighted_entropy_drop(num_classes: usize,
                         data: &Dataset,
                         parent: &Selection,
                         left: &Selection,
                         right: &Selection) -> f64 {
    let left_weight = left.0.len() as f64 / parent.0.len() as f64;
    let right_weight = right.0.len() as f64 / parent.0.len() as f64;

    let weighted_left = entropy(data.select_labels(left), num_classes) * left_weight;
    let weighted_right = entropy(data.select_labels(right), num_classes) * right_weight;
    entropy(data.select_labels(parent), num_classes) - weighted_left - weighted_right
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
    use super::{entropy, Tree, weighted_entropy_drop};
    use dataset::{Dataset, Selection};
    use stump::{Stump, StumpGenerator};
    use distribution::Distribution;
    use test;
    use rand::{Rng, thread_rng};

    #[test]
    fn test_entropy_usize() {
        assert_eq!(entropy(vec![1usize, 1            ].iter(), 2), 0f64);
        assert_eq!(entropy(vec![0usize               ].iter(), 1), 0f64);
        assert_eq!(entropy(vec![0usize, 1            ].iter(), 2), 1f64);
        assert_eq!(entropy(vec![0usize, 1, 2, 3      ].iter(), 4), 2f64);
    }

    #[test]
    fn test_weighted_entropy_drop() {
        assert_eq!(entropy(vec![0usize, 0, 1, 1, 2, 2].iter(), 3), 3f64.log2());
        assert_eq!(entropy(vec![0usize, 0, 1, 1      ].iter(), 3), 1f64);
        assert_eq!(entropy(vec![2usize, 2            ].iter(), 3), 0f64);

        let data = Dataset {
            num_classes: 3,
            labels: vec![0, 0, 1, 1, 2, 2],
            data: vec![]
        };

        let parent = Selection(vec![0, 1, 2, 3, 4, 5]);
        let left = Selection(vec![0, 1, 2, 3]);
        let right = Selection(vec![4, 5]);

        let drop = weighted_entropy_drop(3, &data, &parent, &left, &right);
        assert_eq!(drop, 3f64.log2() - 2f64/3f64);
    }

    // TODO: add tests for e.g. constructing distributions or splitting where the set is empty

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

        for _ in 0..num_samples {
            let l = rng.gen_range(0, num_classes);
            labels.push(l);
            let d = (0..num_dimensions)
                .map(|_| rng.gen_range(0f64, 1f64))
                .collect();
            data.push(d);
        }

        let dataset = Dataset {
            num_classes: num_classes,
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
            Tree::<Stump, Distribution>::train(depth, num_candidates, &mut generator, &dataset)
        });
   }
}


use Leaf;
use dataset::DatasetView;

impl Leaf for Distribution {
    fn from_dataset(dataset: &DatasetView) -> Distribution {
        Distribution::from_labels(dataset.iter_labels(), dataset.num_classes())
    }

    fn empty(num_classes: usize) -> Distribution {
        Distribution {
            probs: vec![0f64; num_classes]
        }
    }

    fn combine(dists: &[Distribution], num_classes: usize) -> Distribution {
        let mut probs = vec![0f64; num_classes];

        for dist in dists {
            for i in 0..num_classes {
                probs[i] += dist.probs[i];
            }
        }

        for i in 0..num_classes {
            probs[i] /= dists.len() as f64;
        }

        Distribution {
            probs: probs
        }
    }
}

#[derive(Debug, Clone)]
pub struct Distribution {
    pub probs: Vec<f64>
}

impl Distribution {
    pub fn from_labels<'a, I>(labels: I, num_classes: usize) -> Distribution
        where I: ExactSizeIterator<Item=&'a usize>
    {
        let mut probs = vec![0f64; num_classes];
        let count = labels.len() as f64;
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
}

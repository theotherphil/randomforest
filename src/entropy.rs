
use SplitCriterion;
use dataset::{Dataset, Selection};

#[derive(Debug, Clone)]
pub struct WeightedEntropyDrop;

impl SplitCriterion for WeightedEntropyDrop {
    fn score(num_classes: usize, data: &Dataset,
             parent: &Selection, left: &Selection, right: &Selection) -> f64
    {
        weighted_entropy_drop(num_classes, data, parent, left, right)
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
    use super::{entropy, weighted_entropy_drop};
    use dataset::{Dataset, Selection};

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
}


use std::slice::Iter;

#[derive(Debug, PartialEq, Clone)]
pub struct Dataset {
    pub labels: Vec<usize>,
    pub data: Vec<Vec<f64>>
}

impl Dataset {
    fn iter_labels(&self) -> Iter<usize> {
        self.labels.iter()
    }

    fn iter_data(&self) -> Iter<Vec<f64>> {
        self.data.iter()
    }
}

/// A set of indices into the labels and data vectors of a Dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection(pub Vec<usize>);

pub struct LabelIter<'a> {
    labels: &'a [usize],
    selection: &'a Selection,
    pos: usize
}

impl<'a> LabelIter<'a> {
    pub fn new(backing_labels: &'a [usize], selection: &'a Selection) -> LabelIter<'a> {
        LabelIter {
            labels: backing_labels,
            selection: selection,
            pos: 0
        }
    }
}

impl<'a> Iterator for LabelIter<'a> {
    type Item = &'a usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.selection.0.len() {
            return None;
        }
        let ref ret = self.labels[self.selection.0[self.pos]];
        self.pos += 1;
        Some(ret)
    }
}

impl<'a> ExactSizeIterator for LabelIter<'a> {
    fn len(&self) -> usize {
        self.selection.0.len()
    }
}

#[cfg(test)]
mod tests {

}

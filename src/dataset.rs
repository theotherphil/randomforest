
use std::slice::Iter;

#[derive(Debug, PartialEq, Clone)]
pub struct Dataset {
    pub labels: Vec<usize>,
    pub data: Vec<Vec<f64>>
}

#[derive(Debug, PartialEq, Clone)]
pub struct DatasetView<'a> {
    backing: &'a Dataset,
    selection: &'a Selection
}

impl Dataset {
    pub fn iter_labels(&self) -> Iter<usize> {
        self.labels.iter()
    }

    pub fn iter_data(&self) -> Iter<Vec<f64>> {
        self.data.iter()
    }

    pub fn select<'a>(&'a self, selection: &'a Selection) -> DatasetView {
        DatasetView {
            backing: self,
            selection: selection
        }
    }
}

impl<'a> DatasetView<'a> {
    pub fn iter_labels(&self) -> SelectionIter<usize> {
        SelectionIter::new(&self.backing.labels, self.selection)
    }

    pub fn iter_data(&self) -> SelectionIter<Vec<f64>> {
        SelectionIter::new(&self.backing.data, self.selection)
    }
}

/// A set of indices into the labels and data vectors of a Dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection(pub Vec<usize>);

pub struct SelectionIter<'a, T: 'a> {
    values: &'a [T],
    selection: &'a Selection,
    pos: usize
}

impl<'a, T> SelectionIter<'a, T> {
    pub fn new(values: &'a [T], selection: &'a Selection) -> SelectionIter<'a, T> {
        SelectionIter {
            values: values,
            selection: selection,
            pos: 0
        }
    }
}

impl<'a, T> Iterator for SelectionIter<'a ,T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.selection.0.len() {
            return None;
        }
        let ref ret = self.values[self.selection.0[self.pos]];
        self.pos += 1;
        Some(ret)
    }
}

impl<'a, T> ExactSizeIterator for SelectionIter<'a, T> {
    fn len(&self) -> usize {
        self.selection.0.len()
    }
}

#[cfg(test)]
mod tests {
    // dataset, iter_labels, iter_data
    // view, iter_labels, iter_data
}

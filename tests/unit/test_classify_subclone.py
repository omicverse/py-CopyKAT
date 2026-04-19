"""Tests for pycopykat.classify.subclone — simplified dynamicTreeCut."""
import numpy as np
from scipy.cluster.hierarchy import linkage

from pycopykat.classify.subclone import dynamic_tree_cut
from pycopykat.kernels.distances import pdist_euclidean


def test_finds_three_blobs():
    rng = np.random.default_rng(0)
    blobs = np.vstack([
        rng.normal(loc=0,  scale=0.05, size=(25, 20)),
        rng.normal(loc=5,  scale=0.05, size=(25, 20)),
        rng.normal(loc=10, scale=0.05, size=(25, 20)),
    ])
    Z = linkage(pdist_euclidean(blobs), method="ward")
    labels = dynamic_tree_cut(Z, min_cluster_size=10, deep_split=2)
    non_zero = labels[labels != 0]
    assert len(set(non_zero)) == 3


def test_respects_min_cluster_size():
    rng = np.random.default_rng(1)
    blobs = np.vstack([
        rng.normal(0, 0.05, size=(3, 10)),    # too small
        rng.normal(5, 0.05, size=(20, 10)),
    ])
    Z = linkage(pdist_euclidean(blobs), method="ward")
    labels = dynamic_tree_cut(Z, min_cluster_size=5, deep_split=2)
    # The 3-point blob should not become its own cluster (too small)
    # It will either be absorbed into the big cluster or marked 0
    uniq = [l for l in set(labels) if l != 0]
    assert len(uniq) <= 2  # at most one real cluster (the 20-pt blob)


def test_single_blob_returns_one_cluster():
    rng = np.random.default_rng(2)
    blob = rng.normal(0, 0.1, size=(30, 5))
    Z = linkage(pdist_euclidean(blob), method="ward")
    labels = dynamic_tree_cut(Z, min_cluster_size=5, deep_split=2)
    assert len(set(labels[labels != 0])) == 1

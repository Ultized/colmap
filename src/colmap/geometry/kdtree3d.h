// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

// Header-only 3-D KD-tree implementation.
// Builds on pure C++ / Eigen; no external KD-tree library is required.
//
// Usage:
//   KDTree3d tree(points);          // points: std::vector<Eigen::Vector3d>
//   auto [idx, dist] = tree.NearestNeighbor(query);
//   auto results     = tree.KNearestNeighbors(query, k);

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// ---------------------------------------------------------------------------
// KDTree3d – compact 3-D axis-aligned KD-tree built over external point data.
//
// Points are NOT copied; the caller must keep the source vector alive.
// Build:  O(n log n)   Query:  O(log n) average
// ---------------------------------------------------------------------------
class KDTree3d {
 public:
  struct NNResult {
    size_t index;    // index into the original points vector
    double sq_dist;  // squared Euclidean distance
  };

  // Construct from an existing point vector.
  // Leaf nodes contain at most `leaf_size` points (default 10).
  explicit KDTree3d(const std::vector<Eigen::Vector3d>& points,
                    int leaf_size = 10)
      : points_(points), leaf_size_(leaf_size) {
    if (points_.empty()) return;
    indices_.resize(points_.size());
    std::iota(indices_.begin(), indices_.end(), 0);
    nodes_.reserve(2 * points_.size() / std::max(1, leaf_size_) + 4);
    Build(0, static_cast<int>(indices_.size()), 0);
  }

  // Nearest-neighbour search.  Returns {index, sq_dist}.
  // Returns {npos, inf} if the tree is empty.
  NNResult NearestNeighbor(const Eigen::Vector3d& query) const {
    NNResult best{std::numeric_limits<size_t>::max(),
                  std::numeric_limits<double>::infinity()};
    if (nodes_.empty()) return best;
    Search(0, query, best);
    return best;
  }

  // K nearest-neighbour search (unsorted).
  std::vector<NNResult> KNearestNeighbors(const Eigen::Vector3d& query,
                                           int k) const {
    // Use a max-heap of size k.
    std::vector<NNResult> heap;
    heap.reserve(k + 1);
    if (!nodes_.empty()) KSearch(0, query, k, heap);
    return heap;
  }

  // Radius search – returns all points with sq_dist < sq_radius.
  std::vector<NNResult> RadiusSearch(const Eigen::Vector3d& query,
                                      double sq_radius) const {
    std::vector<NNResult> results;
    if (!nodes_.empty()) RSearch(0, query, sq_radius, results);
    return results;
  }

  size_t Size() const { return points_.size(); }

 private:
  // -----------------------------------------------------------------------
  // Internal node layout (flat array, root = 0)
  // -----------------------------------------------------------------------
  struct Node {
    // Split axis (0=x, 1=y, 2=z) and position.  For leaves: axis = -1.
    int axis = -1;
    double split = 0.0;
    // Range of indices_ covered by this node: [begin, end).
    int begin = 0, end = 0;
    // Children: left = node_idx*2+1, right = node_idx*2+2.
    // (Stored implicitly via recursion; each node knows its own range.)
    int left = -1, right = -1;
  };

  // -----------------------------------------------------------------------
  // Build
  // -----------------------------------------------------------------------
  int Build(int begin, int end, int depth) {
    const int node_idx = static_cast<int>(nodes_.size());
    nodes_.push_back({});
    Node& node = nodes_[node_idx];
    node.begin = begin;
    node.end = end;

    const int n = end - begin;
    if (n <= leaf_size_) {
      node.axis = -1;  // leaf
      return node_idx;
    }

    // Choose split axis: dimension with maximum spread.
    Eigen::Vector3d min_pt = points_[indices_[begin]];
    Eigen::Vector3d max_pt = min_pt;
    for (int i = begin + 1; i < end; ++i) {
      min_pt = min_pt.cwiseMin(points_[indices_[i]]);
      max_pt = max_pt.cwiseMax(points_[indices_[i]]);
    }
    Eigen::Vector3d spread = max_pt - min_pt;
    int axis = 0;
    if (spread[1] > spread[0]) axis = 1;
    if (spread[2] > spread[axis]) axis = 2;
    node.axis = axis;

    // Median split.
    int mid = begin + n / 2;
    std::nth_element(indices_.begin() + begin,
                     indices_.begin() + mid,
                     indices_.begin() + end,
                     [&](size_t a, size_t b) {
                       return points_[a][axis] < points_[b][axis];
                     });
    node.split = points_[indices_[mid]][axis];

    // Build children.  Each recursive call may push_back onto nodes_, causing
    // reallocation and invalidating any Node& reference.  Always index via
    // nodes_[node_idx] after the call instead of using the earlier reference.
    nodes_[node_idx].left = Build(begin, mid, depth + 1);
    nodes_[node_idx].right = Build(mid, end, depth + 1);

    return node_idx;
  }

  // -----------------------------------------------------------------------
  // NN search
  // -----------------------------------------------------------------------
  void Search(int node_idx,
              const Eigen::Vector3d& query,
              NNResult& best) const {
    if (node_idx < 0) return;
    const Node& node = nodes_[node_idx];

    if (node.axis == -1) {
      // Leaf: check all points.
      for (int i = node.begin; i < node.end; ++i) {
        const double d = (points_[indices_[i]] - query).squaredNorm();
        if (d < best.sq_dist) {
          best.sq_dist = d;
          best.index = indices_[i];
        }
      }
      return;
    }

    // Recurse into the closer half first.
    const double diff = query[node.axis] - node.split;
    const int near_child = (diff < 0) ? node.left : node.right;
    const int far_child = (diff < 0) ? node.right : node.left;

    Search(near_child, query, best);
    // Only recurse into far half if it could contain a closer point.
    if (diff * diff < best.sq_dist) Search(far_child, query, best);
  }

  // -----------------------------------------------------------------------
  // KNN search (max-heap)
  // -----------------------------------------------------------------------
  void KSearch(int node_idx,
               const Eigen::Vector3d& query,
               int k,
               std::vector<NNResult>& heap) const {
    if (node_idx < 0) return;
    const Node& node = nodes_[node_idx];

    // Current worst distance in heap.
    auto HeapWorst = [&]() -> double {
      return heap.empty() ? std::numeric_limits<double>::infinity()
                          : heap.front().sq_dist;
    };

    if (node.axis == -1) {
      for (int i = node.begin; i < node.end; ++i) {
        const double d = (points_[indices_[i]] - query).squaredNorm();
        if (d < HeapWorst() || static_cast<int>(heap.size()) < k) {
          heap.push_back({indices_[i], d});
          std::push_heap(heap.begin(), heap.end(),
                         [](const NNResult& a, const NNResult& b) {
                           return a.sq_dist < b.sq_dist;
                         });
          if (static_cast<int>(heap.size()) > k) {
            std::pop_heap(heap.begin(), heap.end(),
                          [](const NNResult& a, const NNResult& b) {
                            return a.sq_dist < b.sq_dist;
                          });
            heap.pop_back();
          }
        }
      }
      return;
    }

    const double diff = query[node.axis] - node.split;
    const int near_child = (diff < 0) ? node.left : node.right;
    const int far_child = (diff < 0) ? node.right : node.left;

    KSearch(near_child, query, k, heap);
    if (diff * diff < HeapWorst() || static_cast<int>(heap.size()) < k) {
      KSearch(far_child, query, k, heap);
    }
  }

  // -----------------------------------------------------------------------
  // Radius search
  // -----------------------------------------------------------------------
  void RSearch(int node_idx,
               const Eigen::Vector3d& query,
               double sq_radius,
               std::vector<NNResult>& results) const {
    if (node_idx < 0) return;
    const Node& node = nodes_[node_idx];

    if (node.axis == -1) {
      for (int i = node.begin; i < node.end; ++i) {
        const double d = (points_[indices_[i]] - query).squaredNorm();
        if (d <= sq_radius) results.push_back({indices_[i], d});
      }
      return;
    }

    const double diff = query[node.axis] - node.split;
    if (diff < 0 || diff * diff <= sq_radius)
      RSearch(node.left, query, sq_radius, results);
    if (diff >= 0 || diff * diff <= sq_radius)
      RSearch(node.right, query, sq_radius, results);
  }

  // -----------------------------------------------------------------------
  // Data
  // -----------------------------------------------------------------------
  const std::vector<Eigen::Vector3d>& points_;
  std::vector<size_t> indices_;
  std::vector<Node> nodes_;
  int leaf_size_;
};

}  // namespace colmap

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

#include "colmap/geometry/kdtree3d.h"
#include "colmap/util/types.h"

#include <filesystem>
#include <memory>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// ---------------------------------------------------------------------------
// A single LiDAR point with position and surface normal.
// The normal is required for point-to-plane constraint computation.
// If computed normals are not available, set normal = (0,0,0) and the
// LidarPointCloud will fall back to point-to-point constraints.
// ---------------------------------------------------------------------------
struct LidarPoint {
  Eigen::Vector3d xyz;     // position in the LiDAR / world coordinate frame
  Eigen::Vector3d normal;  // outward surface normal (unit vector, or zero)

  bool HasNormal() const { return normal.squaredNorm() > 0.5; }
};

// ---------------------------------------------------------------------------
// LidarPointCloud – wraps a vector of LidarPoints and a KD-tree for fast
// nearest-neighbour queries.
//
// After construction the internal KD-tree is built immediately.
// The point vector is stored by value; pass std::move() for efficiency.
// ---------------------------------------------------------------------------
class LidarPointCloud {
 public:
  LidarPointCloud() = default;

  // Build from a vector of LidarPoints.  Builds the KD-tree immediately.
  explicit LidarPointCloud(std::vector<LidarPoint> points)
      : points_(std::move(points)) {
    RebuildIndex();
  }

  // ---------------------------------------------------------------------------
  // Query: nearest-neighbour in 3-D Euclidean space.
  // Returns {point_index, squared_distance}.
  // ---------------------------------------------------------------------------
  struct NNResult {
    size_t index;
    double sq_dist;
  };

  NNResult NearestNeighbor(const Eigen::Vector3d& query) const {
    const auto r = tree_->NearestNeighbor(query);
    return {r.index, r.sq_dist};
  }

  // Finds up to k nearest neighbours (unsorted, may return fewer if cloud
  // has fewer than k points).
  std::vector<NNResult> KNearestNeighbors(const Eigen::Vector3d& query,
                                           int k) const {
    const auto raw = tree_->KNearestNeighbors(query, k);
    std::vector<NNResult> out;
    out.reserve(raw.size());
    for (const auto& r : raw) out.push_back({r.index, r.sq_dist});
    return out;
  }

  // ---------------------------------------------------------------------------
  // Accessors
  // ---------------------------------------------------------------------------
  const LidarPoint& Point(size_t idx) const { return points_[idx]; }
  size_t Size() const { return points_.size(); }
  bool Empty() const { return points_.empty(); }

  const std::vector<LidarPoint>& Points() const { return points_; }

 private:
  void RebuildIndex() {
    if (points_.empty()) return;
    // Extract xyz positions for the KD-tree.
    xyz_.clear();
    xyz_.reserve(points_.size());
    for (const auto& p : points_) xyz_.push_back(p.xyz);
    tree_ = std::make_unique<KDTree3d>(xyz_);
  }

  std::vector<LidarPoint> points_;
  std::vector<Eigen::Vector3d> xyz_;  // kept alive for the KD-tree
  std::unique_ptr<KDTree3d> tree_;
};

// ---------------------------------------------------------------------------
// A resolved LiDAR constraint: one SfM 3D point ↔ one LiDAR surface element.
// Built by LidarMatcher and consumed by the bundle adjuster.
// ---------------------------------------------------------------------------
struct LidarConstraint {
  point3D_t point3D_id;        // SfM 3D point ID in the Reconstruction
  Eigen::Vector3d xyz_lidar;   // matched LiDAR point position
  Eigen::Vector3d normal;      // LiDAR surface normal (may be zero)
  double sq_dist;              // squared distance at match time (diagnostic)
  bool use_plane;              // true → point-to-plane; false → point-to-point
};

// ---------------------------------------------------------------------------
// Load a LiDAR point cloud from a file.
// Supported formats:
//   .ply              – ASCII or binary PLY (position + optional normals)
//   .txt / .xyz / .*  – ASCII, one point per line: "x y z" or "x y z nx ny nz"
// Normals default to zero when absent.
// ---------------------------------------------------------------------------
std::shared_ptr<LidarPointCloud> LoadLidarPointCloud(
    const std::filesystem::path& path);

}  // namespace colmap

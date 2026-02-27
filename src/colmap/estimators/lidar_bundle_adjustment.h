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

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/scene/lidar_point_cloud.h"
#include "colmap/scene/reconstruction.h"

#include <memory>
#include <vector>

namespace colmap {

// ---------------------------------------------------------------------------
// LidarBundleAdjustmentOptions
// ---------------------------------------------------------------------------
struct LidarBundleAdjustmentOptions {
  // Loss function applied to LiDAR residuals.
  // Cauchy is strongly recommended because KNN can produce wrong matches.
  CeresBundleAdjustmentOptions::LossFunctionType loss_function_type =
      CeresBundleAdjustmentOptions::LossFunctionType::CAUCHY;

  // Scale parameter for the robust Cauchy loss (in metres).
  // The loss transitions from quadratic to linear at this distance.
  // Default: 5 cm.
  double loss_scale = 0.05;

  // Weight multiplier applied to every LiDAR residual block via
  // ceres::ScaledLoss.  In early iterations set < 1.0 so that stale KNN
  // matches cannot overpower reprojection constraints.
  double weight = 0.1;

  // Whether to prefer point-to-plane over point-to-point.
  // Falls back to point-to-point automatically when the matched LiDAR point
  // has a zero normal.
  bool use_point_to_plane = true;
};

// ---------------------------------------------------------------------------
// LidarMatchingOptions
//
// Controls the two-phase KNN matching strategy.
// ---------------------------------------------------------------------------
struct LidarMatchingOptions {
  // --- Phase 1 (early BA iterations) ---
  // Loose distance gate: accept any match closer than this (metres).
  double phase1_max_distance = 0.50;

  // --- Phase 2 (after GPS-stabilised BA) ---
  // Tight distance gate.
  double phase2_max_distance = 0.10;

  // Minimum normal alignment required in phase 2.
  // |dot(sfm_view_mean, lidar_normal)| must be below this value.
  // 0 = no check, cos(60°) = 0.5 is a practical value.
  double phase2_max_normal_alignment = 0.5;

  // Statistical outlier rejection:
  // Remove matches where dist > mean + stat_sigma * std_dev.
  // Applied in both phases.
  double stat_sigma = 3.0;

  // Number of KNN candidates to evaluate per point before picking the best.
  int k_candidates = 5;

  // Minimum number of camera observations (track length) a Point3D must have
  // before being eligible as a LiDAR anchor.  Points observed by fewer
  // cameras have unreliable depth and should not drive the LiDAR alignment.
  // 0 = disabled.
  int min_track_length = 3;

  // Maximum mean reprojection error (in pixels) allowed for a Point3D to be
  // used as a LiDAR anchor.  Points above this threshold are likely noise or
  // SfM outliers whose position cannot be trusted for LiDAR matching.
  // 0 = disabled.
  double max_reprojection_error = 4.0;
};

// ---------------------------------------------------------------------------
// LidarMatcher
//
// Builds a vector of LidarConstraint by matching SfM Point3Ds against a
// LidarPointCloud using the two-phase strategy described in LidarMatchingOptions.
//
// Phase 1: loose KNN + statistical filter.
//          Used in early BA iterations when poses are still drifting.
// Phase 2: tight KNN + statistical filter + normal alignment check.
//          Used after GPS BA has stabilised the pose estimates.
// ---------------------------------------------------------------------------
class LidarMatcher {
 public:
  LidarMatcher(const LidarPointCloud& cloud,
               const LidarMatchingOptions& options);

  // Build constraints from all Point3Ds in reconstruction.
  // Set phase=1 for loose matching, phase=2 for tight matching.
  std::vector<LidarConstraint> BuildConstraints(
      const Reconstruction& reconstruction, int phase = 1) const;

 private:
  // Check normal alignment between a LiDAR normal and the mean viewing
  // direction of a 3D point (approximate: just uses the first observation
  // direction).
  bool NormalAlignmentOk(const Eigen::Vector3d& lidar_normal,
                          const Reconstruction& reconstruction,
                          point3D_t point3d_id) const;

  const LidarPointCloud& cloud_;
  LidarMatchingOptions options_;
};

// ---------------------------------------------------------------------------
// Factory: CreateLidarBundleAdjuster
//
// Builds a bundle adjuster that combines:
//   - Standard reprojection-error constraints (from DefaultBundleAdjuster)
//   - LiDAR point-to-plane (or point-to-point) constraints on Point3D blocks
//
// Design notes:
//   - GPS / pose-prior normalisation is NOT done here; the caller is
//     responsible for running a PosePriorBundleAdjuster first if GPS is
//     available, then calling this for the LiDAR structural step.
//   - reconstruction_->Normalize() is intentionally NOT called so that the
//     absolute scale provided by GPS is preserved.
//   - The gauge is fixed via BundleAdjustmentGauge::THREE_POINTS (same as
//     standard BA without GPS). When GPS is used upstream the scale is
//     already constrained.
// ---------------------------------------------------------------------------
std::unique_ptr<BundleAdjuster> CreateLidarBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const LidarBundleAdjustmentOptions& lidar_options,
    const BundleAdjustmentConfig& config,
    const std::vector<LidarConstraint>& constraints,
    Reconstruction& reconstruction);

}  // namespace colmap

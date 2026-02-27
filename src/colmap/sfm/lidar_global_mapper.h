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
#include "colmap/estimators/lidar_bundle_adjustment.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/lidar_point_cloud.h"
#include "colmap/sfm/prior_global_mapper.h"

#include <memory>
#include <unordered_map>

namespace colmap {

// ---------------------------------------------------------------------------
// LidarGlobalMapperOptions
//
// Extends PriorGlobalMapperOptions with LiDAR-specific knobs.
// All GPS options are inherited unchanged.
// ---------------------------------------------------------------------------
struct LidarGlobalMapperOptions : public PriorGlobalMapperOptions {
  // -----------------------------------------------------------------------
  // LiDAR KNN matching options (two-phase).
  // Phase 1 = early iterations (loose 0.5 m), Phase 2 = tight (0.1 m).
  // -----------------------------------------------------------------------
  LidarMatchingOptions lidar_matching;

  // -----------------------------------------------------------------------
  // LiDAR bundle adjustment options.
  //
  // phase1_weight: applied during Phase 1 (early, low → wrong matches can't
  //               overpower reprojection residuals).
  // phase2_weight: applied during Phase 2 (tight KNN, more trust).
  // -----------------------------------------------------------------------
  LidarBundleAdjustmentOptions lidar_ba;

  // Override weight for Phase 1 BA (typically << 1).
  double lidar_phase1_weight = 0.05;

  // Override weight for Phase 2 BA.
  double lidar_phase2_weight = 1.0;

  // -----------------------------------------------------------------------
  // Phase control.
  // The first `num_gps_only_ba_iterations` iterations run GPS BA only
  // (Phase 1): this stabilises poses before any KNN matching occurs.
  // Remaining iterations run the combined GPS+LiDAR sequence (Phase 2).
  // -----------------------------------------------------------------------
  int num_gps_only_ba_iterations = 1;

  // ---------------------------------------------------------------------------
  // Priority design flag.
  //
  // When true (default), all camera poses are fixed during the LiDAR BA step.
  // Only 3D point positions are optimised under reprojection (from the fixed
  // cameras) + LiDAR surface residuals.
  //
  // Consequence:
  //   - GPS BA is the sole owner of camera pose updates.
  //   - LiDAR BA is the sole owner of 3D point surface alignment.
  //   - Visual consistency (reprojection) is NEVER broken by LiDAR.
  //
  // Set to false only if you want LiDAR to also refine camera poses.
  // ---------------------------------------------------------------------------
  bool fix_poses_in_lidar_ba = true;
};

// ---------------------------------------------------------------------------
// LidarGlobalMapper
//
// Extends PriorGlobalMapper to add LiDAR point-to-plane structural
// constraints (type B: SfM 3D-point ↔ LiDAR surface element).
//
// Architecture:
//   - Steps 1–3 are fully delegated to PriorGlobalMapper::Solve()
//     (run with skip_bundle_adjustment=true, skip_retriangulation=true).
//   - Steps 4–5 are re-implemented here with a two-phase GPS ← LiDAR
//     sequential loop.
//
// Two-phase strategy (Step 4):
//   Phase 1 (iterations 1..num_gps_only_ba_iterations):
//     GPS BA only → stabilise poses without LiDAR.
//   Phase 2 (remaining iterations):
//     For each iteration:
//       1. GPS BA (RunPosePriorBundleAdjustment) — re-anchor poses.
//       2. Rebuild KNN constraints (phase 1 or 2 threshold).
//       3. LiDAR BA (CreateLidarBundleAdjuster) — improve 3D points.
//       4. Outlier filtering (reprojection + tri-angle).
//
// If cloud is nullptr, the mapper behaves identically to PriorGlobalMapper.
// ---------------------------------------------------------------------------
class LidarGlobalMapper : public PriorGlobalMapper {
 public:
  explicit LidarGlobalMapper(
      std::shared_ptr<const DatabaseCache> database_cache,
      std::shared_ptr<const LidarPointCloud> lidar_cloud = nullptr);

  // Set or replace the LiDAR point cloud (may be called before Solve).
  void SetLidarCloud(std::shared_ptr<const LidarPointCloud> cloud);

  // Run the full pipeline with GPS + optional LiDAR constraints.
  bool Solve(const LidarGlobalMapperOptions& options,
             std::unordered_map<frame_t, int>& cluster_ids);

 protected:
  // -----------------------------------------------------------------------
  // Run one LiDAR bundle adjustment pass on top of DefaultCeresBundleAdjuster.
  // Assumes GPS BA has already been run in the same iteration to stabilise
  // poses.  Does NOT call Normalize().
  // -----------------------------------------------------------------------
  // When fix_poses=true all camera poses are held constant so that LiDAR
  // residuals can only move 3D points, not cameras.  This preserves visual
  // consistency — GPS BA is the sole driver of pose refinement.
  bool RunLidarBundleAdjustment(
      const BundleAdjustmentOptions& ba_options,
      const LidarBundleAdjustmentOptions& lidar_options,
      const std::vector<LidarConstraint>& constraints,
      bool fix_poses = true);

  // -----------------------------------------------------------------------
  // Two-phase iterative BA with both GPS and LiDAR constraints.
  //
  // Phase 1 (GPS-only): `num_gps_only_iters` pure GPS BA rounds.
  // Phase 2 (GPS + LiDAR): remaining rounds with KNN rebuild each time.
  //
  // skip_fixed_rotation_stage / skip_joint_optimization_stage mirror the
  // same flags in IterativePriorBundleAdjustment.
  // -----------------------------------------------------------------------
  bool IterativeLidarBundleAdjustment(
      const BundleAdjustmentOptions& ba_options,
      const PosePriorBundleAdjustmentOptions& prior_ba_opts,
      const LidarGlobalMapperOptions& lidar_opts,
      double max_normalized_reproj_error,
      double min_tri_angle_deg,
      int num_iterations,
      bool skip_fixed_rotation_stage = false,
      bool skip_joint_optimization_stage = false);

 private:
  std::shared_ptr<const LidarPointCloud> lidar_cloud_;
};

}  // namespace colmap

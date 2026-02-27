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

#include "colmap/sfm/lidar_global_mapper.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/lidar_bundle_adjustment.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

namespace colmap {

// ============================================================================
// Construction
// ============================================================================

LidarGlobalMapper::LidarGlobalMapper(
    std::shared_ptr<const DatabaseCache> database_cache,
    std::shared_ptr<const LidarPointCloud> lidar_cloud)
    : PriorGlobalMapper(std::move(database_cache)),
      lidar_cloud_(std::move(lidar_cloud)) {}

void LidarGlobalMapper::SetLidarCloud(
    std::shared_ptr<const LidarPointCloud> cloud) {
  lidar_cloud_ = std::move(cloud);
}

// ============================================================================
// RunLidarBundleAdjustment
// ============================================================================

bool LidarGlobalMapper::RunLidarBundleAdjustment(
    const BundleAdjustmentOptions& ba_options,
    const LidarBundleAdjustmentOptions& lidar_options,
    const std::vector<LidarConstraint>& constraints,
    bool fix_poses) {
  auto& recon = GetReconstruction();
  if (recon.NumImages() == 0) {
    LOG(ERROR) << "Cannot run LiDAR BA: no images";
    return false;
  }
  if (recon.NumPoints3D() == 0) {
    LOG(ERROR) << "Cannot run LiDAR BA: no 3D points";
    return false;
  }
  if (constraints.empty()) {
    LOG(INFO) << "[LidarBA] No constraints; skipping LiDAR BA step.";
    return true;
  }

  BundleAdjustmentConfig ba_config;
  for (const auto& [img_id, img] : recon.Images()) {
    if (!img.HasPose()) continue;
    ba_config.AddImage(img_id);
    if (fix_poses && img.HasFrameId()) {
      // Fix the rig-from-world pose so LiDAR residuals can only move 3D
      // points, not cameras.  Reprojection residuals from these fixed cameras
      // still constrain point positions, preserving visual consistency.
      ba_config.SetConstantRigFromWorldPose(img.FrameId());
    }
  }
  if (fix_poses) {
    LOG(INFO) << "[LidarBA] Camera poses FIXED — only 3D points are optimised "
                 "(visual-consistency-first mode).";
  }
  // When fix_poses=true the gauge is fully determined by the fixed poses.
  // When fix_poses=false, GPS prior constraints (from the preceding GPS BA
  // step) have already absorbed the gauge DOF in this iteration.

  // -----------------------------------------------------------------------
  // Optimization A: select the correct linear solver for the fixed-poses case.
  //
  // When ALL cameras are constant, the Ceres problem has NO camera parameter
  // blocks — only 3D point blocks (one 3× block per point, fully independent).
  // SPARSE_SCHUR is designed for camera–point cross-structure; its Schur
  // complement step degenerates to a 0×0 camera matrix, causing the solver
  // to stall with near-zero CPU utilisation.
  //
  // SPARSE_NORMAL_CHOLESKY factors the block-diagonal point normal equations
  // directly, exploiting all CPU threads via OpenMP/TBB inner loops.
  // With 30 warm-started iterations this is 3–10× faster than the default.
  // -----------------------------------------------------------------------
  BundleAdjustmentOptions effective_ba_options = ba_options;
  if (fix_poses) {
    if (!effective_ba_options.ceres) {
      effective_ba_options.ceres =
          std::make_shared<CeresBundleAdjustmentOptions>();
    } else {
      effective_ba_options.ceres =
          std::make_shared<CeresBundleAdjustmentOptions>(
              *effective_ba_options.ceres);
    }
    effective_ba_options.ceres->auto_select_solver_type = false;
    effective_ba_options.ceres->solver_options.linear_solver_type =
        ceres::SPARSE_NORMAL_CHOLESKY;
    // Warm-started from GPS BA result; 30 iterations is ample.
    effective_ba_options.ceres->solver_options.max_num_iterations = 30;
    LOG(INFO) << "[LidarBA] Using SPARSE_NORMAL_CHOLESKY (fix_poses=true).";
  }

  auto ba = CreateLidarBundleAdjuster(
      effective_ba_options, lidar_options, ba_config, constraints, recon);
  return ba->Solve()->IsSolutionUsable();
}

// ============================================================================
// IterativeLidarBundleAdjustment
// ============================================================================

bool LidarGlobalMapper::IterativeLidarBundleAdjustment(
    const BundleAdjustmentOptions& ba_options,
    const PosePriorBundleAdjustmentOptions& prior_ba_opts,
    const LidarGlobalMapperOptions& lidar_opts,
    double max_normalized_reproj_error,
    double min_tri_angle_deg,
    int num_iterations,
    bool skip_fixed_rotation_stage,
    bool skip_joint_optimization_stage) {
  auto& recon = GetReconstruction();

  const bool has_lidar = (lidar_cloud_ && !lidar_cloud_->Empty());
  const bool use_prior  = ShouldUsePriorPosition(lidar_opts);

  // Build matching helper once (KD-tree is built in cloud constructor).
  std::unique_ptr<LidarMatcher> matcher;
  if (has_lidar) {
    matcher = std::make_unique<LidarMatcher>(*lidar_cloud_,
                                              lidar_opts.lidar_matching);
  }

  // Helper: run GPS BA (or plain BA if no priors) in one or both stages.
  auto RunGpsBa = [&](bool fix_rotation) -> bool {
    if (use_prior) {
      BundleAdjustmentOptions opts = ba_options;
      opts.constant_rig_from_world_rotation = fix_rotation;
      return RunPosePriorBundleAdjustment(opts, prior_ba_opts);
    } else {
      // Fall back to IterativeBundleAdjustment-equivalent (single pass).
      BundleAdjustmentConfig ba_config;
      for (const auto& [img_id, img] : recon.Images()) {
        if (img.HasPose()) ba_config.AddImage(img_id);
      }
      BundleAdjustmentOptions opts = ba_options;
      opts.constant_rig_from_world_rotation = fix_rotation;
      auto ba = CreateDefaultBundleAdjuster(opts, ba_config, recon);
      return ba->Solve()->IsSolutionUsable();
    }
  };

  // Helper: filter points after each iteration.
  auto FilterAfterIteration = [&](int ite) {
    ObservationManager obs_manager(recon);
    bool saturated = true;
    while (saturated && ite < num_iterations) {
      const double scaling = static_cast<double>(std::max(3 - ite, 1));
      const size_t filtered_num =
          obs_manager.FilterPoints3DWithLargeReprojectionError(
              scaling * max_normalized_reproj_error,
              recon.Point3DIds(),
              ReprojectionErrorType::NORMALIZED);
      if (filtered_num > 1e-3 * recon.NumPoints3D()) {
        saturated = false;
      } else {
        ite++;
      }
    }
    return saturated;
  };

  // =========================================================================
  // Phase 1: GPS-only stabilisation iterations.
  // =========================================================================
  const int gps_only_iters =
      has_lidar ? lidar_opts.num_gps_only_ba_iterations : num_iterations;

  for (int ite = 0; ite < gps_only_iters && ite < num_iterations; ++ite) {
    if (!skip_fixed_rotation_stage) {
      if (!RunGpsBa(/*fix_rotation=*/true)) return false;
      LOG(INFO) << "[LidarMapper Phase1] GPS-only iteration " << ite + 1
                << " / " << gps_only_iters << ", fixed-rotation done";
    }
    if (!skip_joint_optimization_stage) {
      if (!RunGpsBa(/*fix_rotation=*/false)) return false;
      LOG(INFO) << "[LidarMapper Phase1] GPS-only iteration " << ite + 1
                << " / " << gps_only_iters << " joint done";
    }

    if (FilterAfterIteration(ite)) {
      LOG(INFO) << "Phase 1: tight-filter saturation; stopping early.";
      if (!has_lidar) return true;
      break;
    }
  }

  if (!has_lidar) {
    // No LiDAR cloud: just do final strict filter and return.
    ObservationManager obs_manager_final(recon);
    obs_manager_final.FilterPoints3DWithLargeReprojectionError(
        max_normalized_reproj_error,
        recon.Point3DIds(),
        ReprojectionErrorType::NORMALIZED);
    obs_manager_final.FilterPoints3DWithSmallTriangulationAngle(
        min_tri_angle_deg, recon.Point3DIds());
    return true;
  }

  // =========================================================================
  // Phase 2: GPS + LiDAR iterations.
  //   Each iteration: (1) GPS BA → (2) rebuild KNN → (3) LiDAR BA → filter.
  // =========================================================================
  const int phase2_start = std::min(gps_only_iters, num_iterations);

  for (int ite = phase2_start; ite < num_iterations; ++ite) {
    // --- Determine tight/loose phase for KNN. ---
    // Use phase 2 (tight) only after at least 1 GPS-only iter has run.
    const int knn_phase = (ite >= 1) ? 2 : 1;

    // --- Step (1): GPS BA with fixed rotations (warm-up) ---
    if (!skip_fixed_rotation_stage) {
      if (!RunGpsBa(/*fix_rotation=*/true)) return false;
    }
    // --- Step (1b): GPS BA joint ---
    if (!skip_joint_optimization_stage) {
      if (!RunGpsBa(/*fix_rotation=*/false)) return false;
    }

    LOG(INFO) << "[LidarMapper Phase2] Iteration " << ite + 1
              << " / " << num_iterations << ": GPS BA done.";

    // --- Step (2): Rebuild KNN constraints (on updated 3D points). ---
    std::vector<LidarConstraint> constraints =
        matcher->BuildConstraints(recon, knn_phase);

    // --- Step (3): LiDAR BA (improves 3D point positions). ---
    // With fix_poses_in_lidar_ba=true, camera poses are frozen here so that
    // LiDAR surface alignment can never corrupt visual consistency.
    LidarBundleAdjustmentOptions lidar_ba = lidar_opts.lidar_ba;
    lidar_ba.weight =
        (knn_phase == 1) ? lidar_opts.lidar_phase1_weight
                         : lidar_opts.lidar_phase2_weight;

    if (!RunLidarBundleAdjustment(
            ba_options, lidar_ba, constraints,
            lidar_opts.fix_poses_in_lidar_ba)) {
      return false;
    }

    LOG(INFO) << "[LidarMapper Phase2] Iteration " << ite + 1
              << " / " << num_iterations << ": LiDAR BA done.";

    // --- Step (4): Filter outlier points. ---
    if (FilterAfterIteration(ite)) {
      LOG(INFO) << "Phase 2: tight-filter saturation; stopping early.";
      break;
    }
  }

  // Final strict filtering.
  {
    ObservationManager obs_manager_final(recon);
    obs_manager_final.FilterPoints3DWithLargeReprojectionError(
        max_normalized_reproj_error,
        recon.Point3DIds(),
        ReprojectionErrorType::NORMALIZED);
    obs_manager_final.FilterPoints3DWithSmallTriangulationAngle(
        min_tri_angle_deg, recon.Point3DIds());
  }

  return true;
}

// ============================================================================
// Solve
// ============================================================================

bool LidarGlobalMapper::Solve(
    const LidarGlobalMapperOptions& options,
    std::unordered_map<frame_t, int>& cluster_ids) {
  // -------------------------------------------------------------------------
  // Steps 1–3: Rotation averaging, track establishment, global positioning.
  // Delegate entirely to PriorGlobalMapper.  Skip Steps 4–5 so we can run
  // our own LiDAR-aware versions below.
  // -------------------------------------------------------------------------
  LidarGlobalMapperOptions opts_steps123 = options;
  opts_steps123.skip_bundle_adjustment   = true;
  opts_steps123.skip_retriangulation     = true;

  if (!PriorGlobalMapper::Solve(opts_steps123, cluster_ids)) {
    return false;
  }

  const bool use_prior = ShouldUsePriorPosition(options);
  const bool has_lidar = (lidar_cloud_ && !lidar_cloud_->Empty());

  // Build the GPS prior BA options (mirrors logic in PriorGlobalMapper::Solve).
  PosePriorBundleAdjustmentOptions prior_ba_opts = options.pose_prior_ba;
  if (use_prior) {
    if (options.use_robust_loss_on_prior_position) {
      if (!prior_ba_opts.ceres) prior_ba_opts = PosePriorBundleAdjustmentOptions{};
      prior_ba_opts.ceres->prior_position_loss_function_type =
          CeresBundleAdjustmentOptions::LossFunctionType::CAUCHY;
    }
    prior_ba_opts.ceres->prior_position_loss_scale =
        options.prior_position_loss_scale;
    prior_ba_opts.alignment_ransac_options.random_seed = options.random_seed;

    if (has_lidar) {
      LOG(INFO) << "LidarGlobalMapper: GPS + LiDAR solving ENABLED ("
                << database_cache_->PosePriors().size() << " GPS priors, "
                << lidar_cloud_->Size() << " LiDAR points).";
    } else {
      LOG(INFO) << "LidarGlobalMapper: GPS-only solving (no LiDAR cloud).";
    }
  } else {
    if (has_lidar) {
      LOG(INFO) << "LidarGlobalMapper: LiDAR-only structural constraints.";
    } else {
      LOG(INFO) << "LidarGlobalMapper: no GPS or LiDAR; standard BA.";
    }
  }

  // -------------------------------------------------------------------------
  // Step 4: Iterative bundle adjustment with GPS + LiDAR.
  // -------------------------------------------------------------------------
  if (!options.skip_bundle_adjustment) {
    LOG_HEADING1("Running iterative bundle adjustment (LiDAR-aware)");
    Timer timer;
    timer.Start();

    if (use_prior || has_lidar) {
      if (!IterativeLidarBundleAdjustment(
              options.bundle_adjustment,
              prior_ba_opts,
              options,
              options.max_normalized_reproj_error,
              options.min_tri_angle_deg,
              options.ba_num_iterations,
              options.ba_skip_fixed_rotation_stage,
              options.ba_skip_joint_optimization_stage)) {
        return false;
      }
    } else {
      // Pure reprojection BA (no GPS, no LiDAR).
      if (!IterativeBundleAdjustment(options.bundle_adjustment,
                                     options.max_normalized_reproj_error,
                                     options.min_tri_angle_deg,
                                     options.ba_num_iterations,
                                     options.ba_skip_fixed_rotation_stage,
                                     options.ba_skip_joint_optimization_stage)) {
        return false;
      }
    }

    LOG(INFO) << "Iterative bundle adjustment done in "
              << timer.ElapsedSeconds() << " seconds";
  }

  // -------------------------------------------------------------------------
  // Step 5: Retriangulation + GPS-prior refinement.
  //   Delegate to PriorGlobalMapper's IterativePriorRetriangulateAndRefine
  //   (GPS-aware), then run a final LiDAR BA pass if a cloud is available.
  // -------------------------------------------------------------------------
  if (!options.skip_retriangulation) {
    LOG_HEADING1("Running iterative retriangulation and refinement");
    Timer timer;
    timer.Start();

    bool ok;
    if (use_prior) {
      ok = IterativePriorRetriangulateAndRefine(
          options.retriangulation,
          options.bundle_adjustment,
          options,
          options.max_normalized_reproj_error,
          options.min_tri_angle_deg);
    } else {
      ok = IterativeRetriangulateAndRefine(options.retriangulation,
                                           options.bundle_adjustment,
                                           options.max_normalized_reproj_error,
                                           options.min_tri_angle_deg);
    }
    if (!ok) return false;

    // Final LiDAR structural refinement on the re-triangulated points.
    if (has_lidar) {
      LOG(INFO) << "Running final LiDAR BA pass after retriangulation.";
      LidarMatcher matcher(*lidar_cloud_, options.lidar_matching);
      auto constraints =
          matcher.BuildConstraints(GetReconstruction(), /*phase=*/2);

      LidarBundleAdjustmentOptions lidar_ba = options.lidar_ba;
      lidar_ba.weight = options.lidar_phase2_weight;

      if (!RunLidarBundleAdjustment(
              options.bundle_adjustment, lidar_ba, constraints,
              options.fix_poses_in_lidar_ba)) {
        return false;
      }
    }

    LOG(INFO) << "Iterative retriangulation done in "
              << timer.ElapsedSeconds() << " seconds";
  }

  return true;
}

}  // namespace colmap

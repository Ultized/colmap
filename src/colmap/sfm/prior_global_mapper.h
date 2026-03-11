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
#include "colmap/geometry/pose_prior.h"
#include "colmap/sfm/global_mapper.h"
#include "colmap/sfm/incremental_triangulator.h"

#include <unordered_map>
#include <vector>

namespace colmap {

// Options that extend GlobalMapperOptions with GPS/pose-prior support.
// Keeps all parent options unchanged; adds prior-specific knobs and
// automatically enables prior-constrained solving when priors are available.
struct PriorGlobalMapperOptions : public GlobalMapperOptions {
  // Enable GPS/pose-prior constraints.
  // When true *and* the database contains at least one position prior, every
  // BA stage is replaced by PosePriorBundleAdjuster and Normalize() calls
  // that would break absolute scale are suppressed.
  bool use_prior_position = false;

    // Enable loading and use of custom 6DoF pose priors from a database table.
    // If the configured table is missing or empty, the pipeline falls back to
    // the standard mapper path automatically.
    bool use_6dof_pose_priors = true;

    // Name of the SQLite table that stores full 6DoF pose priors.
    std::string six_dof_pose_prior_table = "6dof_pose_priors";

    // Fallback rotation prior standard deviation used for 6DoF BA when the
    // custom table row does not store an explicit rotation covariance.
    double six_dof_prior_rotation_stddev_deg = 1.0;

    // Reject pose-graph edges whose relative geometry disagrees too much with
    // the 6DoF priors. This keeps rotation averaging and global positioning from
    // being driven by obviously inconsistent two-view edges.
    bool use_6dof_pose_graph_filtering = true;

    // Maximum allowed relative rotation disagreement between a pose-graph edge
    // and the corresponding 6DoF priors. Set <= 0 to disable.
    double max_6dof_pose_graph_rotation_error_deg = 5.0;

    // Maximum allowed translation-direction disagreement between a pose-graph
    // edge and the 6DoF priors. Set <= 0 to disable.
    double max_6dof_pose_graph_translation_direction_error_deg = 30.0;

    // After rotation averaging, clamp frames whose rotation drifts too far from
    // the 6DoF prior before running global positioning. Set <= 0 to disable.
    double max_6dof_rotation_prior_deviation_deg = 5.0;

    // If true, keep using 6DoF priors during the retriangulation refinement
    // stage. This defaults to enabled so that databases carrying valid 6DoF
    // priors stay on the metric BA path end-to-end unless explicitly disabled.
    bool use_6dof_retriangulation_refinement = true;

    // If true, keep a LiDAR structural alignment step active throughout the
    // 6DoF retriangulation refinement stage. This requires a non-empty
    // GlobalMapper.lidar_point_cloud_path and only affects the 6DoF mapper
    // path; the LiDAR alignment is applied after each 6DoF BA pass and stays
    // enabled until retriangulation finishes.
    bool use_lidar_point_to_plane_in_retriangulation = true;

    // Maximum allowed mean reprojection error (pixels) for a sparse Point3D to
    // participate in LiDAR retriangulation alignment. Set <= 0 to disable the
    // reprojection-quality gate.
    double lidar_retriangulation_max_reprojection_error = 3.0;

    // If true, apply LiDAR retriangulation alignment only once after the final
    // 6DoF BA pass. This is a conservative mode that avoids injecting LiDAR
    // constraints into the earlier retriangulation refinement iterations.
    bool use_lidar_point_to_plane_only_in_final_retriangulation = false;

    // During non-final 6DoF retriangulation refinement stages, require LiDAR
    // anchors to have at least this many observations before they are allowed
    // to participate in the early LiDAR alignment. Set <= 0 to disable the
    // extra stability gate and use the generic LiDAR matcher threshold only.
    int lidar_retriangulation_early_min_track_length = 4;

    // During non-final 6DoF retriangulation refinement stages, require LiDAR
    // anchors to stay below this mean pixel reprojection error before they are
    // allowed to participate in early LiDAR alignment. Set <= 0 to disable
    // the extra gate.
    double lidar_retriangulation_early_max_mean_reprojection_error = 2.0;

    // Maximum number of refinement rounds in retriangulation. Lower values
    // reduce runtime, especially when 6DoF and LiDAR constraints are both
    // active. Must be >= 1.
    int retriangulation_refinement_max_refinements = 5;

    // Solver caps for the per-refinement BA inside retriangulation. These only
    // affect the repeated local refinement loop; the main global BA and final
    // BA keep using the normal bundle-adjustment options.
    int retriangulation_refinement_ba_max_num_iterations = 50;
    int retriangulation_refinement_ba_max_linear_solver_iterations = 100;

    // After each non-final 6DoF LiDAR retriangulation alignment, run one
    // observation-level reprojection cleanup on recently modified structure to
    // suppress bad local geometry before the next completion / merge step.
    // Set <= 0 to disable this early cleanup.
    double lidar_retriangulation_early_post_alignment_max_reprojection_error_px =
      2.0;

    // If true, emit detailed per-stage track snapshots for the current HP300
    // 6DoF outlier neighborhoods while running the default pure-visual
    // retriangulation refinement path.
    bool log_6dof_retriangulation_debug_snapshots = false;

  // Maximum allowed Euclidean drift (in metres) from a position prior after
  // global positioning. Frames that exceed this threshold are snapped back to
  // their prior position before bundle adjustment. Set <= 0 to disable.
  double max_position_prior_deviation = 0.5;

  // If true, frames exceeding max_position_prior_deviation are reset to their
  // prior position after global positioning. If false, the deviation is only
  // reported in the log.
  bool clamp_positions_to_prior_after_global_positioning = true;

  // If true, frames exceeding max_position_prior_deviation are reset to their
  // prior position once more after the final BA/retriangulation stage. If
  // false, the deviation is only reported in the log.
  bool clamp_positions_to_prior_after_optimization = true;

  // If >= 0, only frames with at most this many Point3D observations are
  // eligible for the final post-optimization clamp. Set < 0 to allow the
  // clamp to act on any frame that exceeds max_position_prior_deviation.
  int clamp_positions_to_prior_after_optimization_max_observations = -1;

  // If > 0, frames whose centers still deviate more than this many metres
  // from their position prior at the final post-optimization enforcement stage
  // are removed from the reconstruction instead of being kept as-is. This is
  // intended for producing a geometrically clean final solution when a frame
  // has drifted too far to trust a hard snap back to the prior.
  double delete_frames_with_position_prior_deviation_after_optimization = 0.5;

  // If true, run one additional cleanup BA on the remaining reconstruction
  // after the final post-optimization clamp/delete enforcement. This is meant
  // to lower the true post-enforcement reprojection error without reintroducing
  // the removed frames.
  bool run_bundle_adjustment_after_pose_enforcement = false;

  // Run one explicit pixel-domain reprojection filter on the remaining
  // reconstruction after the final post-optimization clamp/delete/cleanup
  // stage. Points with observations above this threshold are pruned through
  // ObservationManager, which is intended to cut the long-tail residuals of
  // the delete-only final solution without reintroducing removed frames.
  // Set <= 0 to disable. The current clean-final default is 3 px.
  double post_enforcement_max_reprojection_error_px = 3.0;

  // If true, increase 6DoF prior weights for risky frames during the final
  // joint 6DoF+LiDAR BA instead of relying only on the post-optimization
  // clamp. Risk is estimated from the current position-prior deviation and
  // optionally from a low-observation threshold.
  bool use_risk_stratified_prior_weights_in_final_joint_ba = false;

  // Frames with at most this many Point3D observations are treated as high
  // risk during the final joint BA when risk-stratified prior weights are
  // enabled. Set < 0 to disable the observation-count trigger.
  int final_joint_ba_high_risk_max_observations = 30;

  // Position-prior deviation thresholds (metres) used to classify medium/high
  // risk frames before the final joint 6DoF+LiDAR BA. Set <= 0 to disable a
  // given tier.
  double final_joint_ba_medium_risk_position_prior_deviation = 0.1;
  double final_joint_ba_high_risk_position_prior_deviation = 0.5;

  // Multipliers applied to the prior weights of medium/high risk frames in the
  // final joint 6DoF+LiDAR BA. Values > 1 strengthen the corresponding priors.
  double final_joint_ba_medium_risk_prior_weight_multiplier = 4.0;
  double final_joint_ba_high_risk_prior_weight_multiplier = 16.0;

  // Apply a robust (Cauchy) loss on prior position residuals to tolerate
  // GPS outliers and noise spikes.
  bool use_robust_loss_on_prior_position = true;

  // Scale threshold for the robust loss (chi2 95 %, 3 DOF = 7.815).
  double prior_position_loss_scale = 7.815;

    // Treat reconstructions with a suspiciously small mean reprojection error as
    // invalid. This catches degenerate outputs such as the zero-error runaway
    // state observed on the HP300 dataset. Set <= 0 to disable.
    double min_mean_reprojection_error = 1e-6;

    // Require every registered image to keep at least this many feature-point
    // observations after optimization. Set <= 0 to disable.
    int min_observations_per_registered_image = 1;

  // Options forwarded to CreatePosePriorBundleAdjuster (alignment RANSAC,
  // fallback covariance, etc.).  The loss function type and scale fields inside
  // are overridden at runtime based on use_robust_loss_on_prior_position and
  // prior_position_loss_scale.
  PosePriorBundleAdjustmentOptions pose_prior_ba;
};

// GlobalMapper subclass that injects GPS/pose-prior constraints into:
//   1. Post-rotation-averaging translation restoration (Step 1)
//   2. GPS-seeded position initialisation for GlobalPositioner (Step 3)
//   3. PosePriorBundleAdjuster usage in BA stages (Steps 4 & 5)
//   4. Suppression of Normalize() calls that would destroy absolute scale
//
// Designed for minimal conflict with upstream updates:
//   - GlobalMapper itself is left almost unchanged (only 3 protected getters
//     were added to global_mapper.h).
//   - All prior-aware logic lives exclusively in this subclass.
class PriorGlobalMapper : public GlobalMapper {
 public:
  explicit PriorGlobalMapper(
      std::shared_ptr<const DatabaseCache> database_cache);

  // Run the full global SfM pipeline with optional pose-prior constraints.
  // Mirrors GlobalMapper::Solve() but forks into GPS-aware variants for each
  // stage when use_prior_position is true and priors exist in the database.
  bool Solve(const PriorGlobalMapperOptions& options,
             std::unordered_map<frame_t, int>& cluster_ids);

 protected:
  // -----------------------------------------------------------------------
  // Helper: check whether effective GPS-constrained solving should be used.
  // Returns true only when options.use_prior_position is true AND the
  // database contains at least one position prior that maps to a registered
  // image.
  // -----------------------------------------------------------------------
  bool ShouldUsePriorPosition(
      const PriorGlobalMapperOptions& options) const;

  // -----------------------------------------------------------------------
  // Step 1 (post-rotation-averaging):
  // For each registered image that has a GPS position prior, overwrite the
  // pose so that the camera center equals the prior position while preserving
  // the current camera rotation. This anchors translations in metric space
  // before GlobalPositioning.
  // -----------------------------------------------------------------------
  void RestoreTranslationsFromPriors(const std::vector<PosePrior>& pose_priors);

  // Clamp or report frames whose centers drift too far from the corresponding
  // position prior after global positioning. This protects the pre-BA stage
  // from running away when priors are already approximately correct.
  size_t EnforceMaxPositionPriorDeviation(
      const std::vector<PosePrior>& pose_priors,
      double max_position_prior_deviation,
      double delete_position_prior_deviation,
      int max_observations,
      bool clamp_to_prior,
      const char* stage_name);

  // -----------------------------------------------------------------------
  // Step 3 – GPS-seeded global positioning (skip Normalize at the end).
  // Identical logic to GlobalMapper::GlobalPositioning except:
  //   - generate_random_positions is forced to false so that the frame
  //     centers set by RestoreTranslationsFromPriors are used as initial
  //     values for BATA.
  //   - reconstruction_->Normalize() is NOT called so absolute scale is
  //     preserved.
  // -----------------------------------------------------------------------
  bool PriorGlobalPositioning(const GlobalPositionerOptions& options,
                              double max_position_prior_deviation,
                              bool clamp_positions_to_prior,
                              double max_angular_reproj_error_deg,
                              double max_normalized_reproj_error,
                              double min_tri_angle_deg);

  // -----------------------------------------------------------------------
  // Core helper: one BA step with GPS prior constraints.
  // Builds BundleAdjustmentConfig (without gauge fixing, since GPS constrains
  // the gauge), then delegates to CreatePosePriorBundleAdjuster.
  // Does NOT call Normalize().
  // -----------------------------------------------------------------------
  bool RunPosePriorBundleAdjustment(
      const BundleAdjustmentOptions& ba_options,
      const PosePriorBundleAdjustmentOptions& prior_options);

  // -----------------------------------------------------------------------
  // Step 4 – GPS-aware iterative bundle adjustment.
  // Mirrors GlobalMapper::IterativeBundleAdjustment() but:
  //   - Uses RunPosePriorBundleAdjustment instead of RunBundleAdjustment.
  //   - Skips reconstruction_->Normalize() after each iteration.
  // -----------------------------------------------------------------------
  bool IterativePriorBundleAdjustment(
      const BundleAdjustmentOptions& options,
      const PosePriorBundleAdjustmentOptions& prior_options,
      double max_normalized_reproj_error,
      double min_tri_angle_deg,
      int num_iterations,
      bool skip_fixed_rotation_stage = false,
      bool skip_joint_optimization_stage = false);

  // -----------------------------------------------------------------------
  // Step 5 – GPS-aware iterative retriangulation and refinement.
  // Mirrors GlobalMapper::IterativeRetriangulateAndRefine() but:
  //   - Sets mapper_options.use_prior_position = true so that
  //     IncrementalMapper::IterativeGlobalRefinement uses
  //     PosePriorBundleAdjuster internally and skips Normalize.
  //   - Replaces the final RunBundleAdjustment with
  //     RunPosePriorBundleAdjustment.
  //   - Skips reconstruction_->Normalize().
  // -----------------------------------------------------------------------
  bool IterativePriorRetriangulateAndRefine(
      const IncrementalTriangulator::Options& tri_options,
      const BundleAdjustmentOptions& ba_options,
      const PriorGlobalMapperOptions& mapper_options,
      double max_normalized_reproj_error,
      double min_tri_angle_deg);

  // The database cache is stored here as well so that PosePriors() is
  // accessible without a virtual accessor (the parent's copy is private).
  std::shared_ptr<const DatabaseCache> database_cache_;
};

}  // namespace colmap

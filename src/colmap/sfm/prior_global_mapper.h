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
#include "colmap/sfm/incremental_mapper.h"
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
  bool use_prior_position = true;

  // Apply a robust (Cauchy) loss on prior position residuals to tolerate
  // GPS outliers and noise spikes.
  bool use_robust_loss_on_prior_position = true;

  // Scale threshold for the robust loss (chi2 95 %, 3 DOF = 7.815).
  double prior_position_loss_scale = 7.815;

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
  // RigFromWorld translation so that the camera center equals the prior
  // position:  t = -R * C_gps
  // This anchors translations in metric space before GlobalPositioning.
  // -----------------------------------------------------------------------
  void RestoreTranslationsFromPriors(const std::vector<PosePrior>& pose_priors);

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

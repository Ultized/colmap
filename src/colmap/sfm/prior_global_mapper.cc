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

#include "colmap/sfm/prior_global_mapper.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/global_positioning.h"
#include "colmap/math/math.h"
#include "colmap/scene/projection.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"
#include "colmap/util/types.h"

#include <algorithm>

namespace colmap {
namespace {

void SetImageCenterFromPrior(Image& image, const Eigen::Vector3d& position) {
  Frame& frame = *image.FramePtr();
  const Rigid3d& cam_from_world = image.CamFromWorld();
  const Eigen::Vector3d translation =
      -(cam_from_world.rotation().toRotationMatrix() * position);
  frame.SetCamFromWorld(
      image.CameraId(), Rigid3d(cam_from_world.rotation(), translation));
}

// Replicate the parent's option-initialization logic (the original lives in
// an anonymous namespace in global_mapper.cc and is therefore inaccessible).
PriorGlobalMapperOptions InitializePriorOptions(
    const PriorGlobalMapperOptions& options) {
  PriorGlobalMapperOptions opts = options;
  if (opts.random_seed >= 0) {
    opts.rotation_averaging.random_seed = opts.random_seed;
    opts.global_positioning.random_seed = opts.random_seed;
    opts.global_positioning.use_parameter_block_ordering = false;
    opts.retriangulation.random_seed = opts.random_seed;
    opts.pose_prior_ba.alignment_ransac_options.random_seed = opts.random_seed;
  }
  opts.global_positioning.solver_options.num_threads = opts.num_threads;
  if (opts.bundle_adjustment.ceres) {
    opts.bundle_adjustment.ceres->solver_options.num_threads = opts.num_threads;
  }
  return opts;
}

}  // namespace

// ---------------------------------------------------------------------------
// PriorGlobalMapper
// ---------------------------------------------------------------------------

PriorGlobalMapper::PriorGlobalMapper(
    std::shared_ptr<const DatabaseCache> database_cache)
    : GlobalMapper(database_cache),
      database_cache_(std::move(THROW_CHECK_NOTNULL(database_cache))) {}

bool PriorGlobalMapper::ShouldUsePriorPosition(
    const PriorGlobalMapperOptions& options) const {
  if (!options.use_prior_position) return false;
  const auto& pose_priors = database_cache_->PosePriors();
  if (pose_priors.empty()) {
    LOG(WARNING)
        << "use_prior_position=true but no pose priors found in database. "
           "Falling back to standard global mapper without GPS constraints.";
    return false;
  }
  const bool any_valid = std::any_of(
      pose_priors.begin(), pose_priors.end(), [](const PosePrior& p) {
        return p.HasPosition();
      });
  if (!any_valid) {
    LOG(WARNING) << "Pose priors found but none have a valid position. "
                    "Falling back to standard solver.";
    return false;
  }
  return true;
}

void PriorGlobalMapper::RestoreTranslationsFromPriors(
    const std::vector<PosePrior>& pose_priors) {
  auto& recon = GetReconstruction();
  size_t restored = 0;
  for (const auto& prior : pose_priors) {
    if (!prior.HasPosition()) continue;
    if (prior.corr_data_id.sensor_id.type != SensorType::CAMERA) continue;

    const image_t image_id = static_cast<image_t>(prior.corr_data_id.id);
    if (!recon.ExistsImage(image_id)) continue;

    Image& image = recon.Image(image_id);
    if (!image.HasPose()) continue;

    // Set the full camera center from the prior while preserving the current
    // camera rotation. This also handles non-reference rig cameras correctly.
    SetImageCenterFromPrior(image, prior.position);
    ++restored;
  }
  LOG(INFO) << "Restored GPS-seeded translations for " << restored
            << " cameras after rotation averaging.";
}

size_t PriorGlobalMapper::EnforceMaxPositionPriorDeviation(
    const std::vector<PosePrior>& pose_priors,
    double max_position_prior_deviation,
    double delete_position_prior_deviation,
    int max_observations,
    bool clamp_to_prior,
    const char* stage_name) {
  if (max_position_prior_deviation <= 0 && delete_position_prior_deviation <= 0) {
    return 0;
  }

  auto& recon = GetReconstruction();
  size_t num_exceeded = 0;
  size_t num_clamped = 0;
  size_t num_deleted_frames = 0;
  size_t num_deleted_images = 0;
  double max_observed_deviation = 0.0;
  std::unordered_set<frame_t> frame_ids_to_delete;

  for (const auto& prior : pose_priors) {
    if (!prior.HasPosition()) continue;
    if (prior.corr_data_id.sensor_id.type != SensorType::CAMERA) continue;

    const image_t image_id = static_cast<image_t>(prior.corr_data_id.id);
    if (!recon.ExistsImage(image_id)) continue;

    Image& image = recon.Image(image_id);
    if (!image.HasPose()) continue;
    if (max_observations >= 0 &&
        static_cast<int>(image.NumPoints3D()) > max_observations) {
      continue;
    }

    const double deviation =
        (image.ProjectionCenter() - prior.position).norm();
    const bool exceeds_clamp_threshold =
        max_position_prior_deviation > 0 &&
        deviation > max_position_prior_deviation;
    const bool exceeds_delete_threshold =
        delete_position_prior_deviation > 0 &&
        deviation > delete_position_prior_deviation;
    if (!exceeds_clamp_threshold && !exceeds_delete_threshold) continue;

    ++num_exceeded;
    max_observed_deviation = std::max(max_observed_deviation, deviation);

    if (exceeds_delete_threshold && image.HasFrameId()) {
      frame_ids_to_delete.insert(image.FrameId());
    } else if (clamp_to_prior && exceeds_clamp_threshold) {
      SetImageCenterFromPrior(image, prior.position);
      ++num_clamped;
    }
  }

  for (const frame_t frame_id : frame_ids_to_delete) {
    if (!recon.ExistsFrame(frame_id)) {
      continue;
    }
    const Frame& frame = recon.Frame(frame_id);
    num_deleted_images +=
        static_cast<size_t>(std::distance(frame.ImageIds().begin(),
                                          frame.ImageIds().end()));
    recon.DeRegisterFrame(frame_id);
    ++num_deleted_frames;
  }

  if (num_exceeded > 0) {
    std::ostringstream message;
    message << "[PriorGlobalMapper] " << stage_name << ": found "
            << num_exceeded << " frame(s) whose center deviated more than ";
    if (max_position_prior_deviation > 0) {
      message << max_position_prior_deviation << " m";
    } else {
      message << delete_position_prior_deviation << " m";
    }
    message << " from their position prior. Max deviation was "
            << max_observed_deviation << " m.";
    if (clamp_to_prior && num_clamped > 0) {
      message << " Clamped " << num_clamped << " frame(s).";
    }
    if (num_deleted_frames > 0) {
      message << " Deleted " << num_deleted_frames << " frame(s) covering "
              << num_deleted_images << " image(s) whose deviation exceeded "
              << delete_position_prior_deviation << " m.";
    }
    if (max_observations >= 0) {
      message << " Eligibility was limited to frames with <= "
              << max_observations << " Point3D observations.";
    }
    LOG(WARNING) << message.str();
  }

  return clamp_to_prior ? (num_clamped + num_deleted_frames) : num_exceeded;
}

bool PriorGlobalMapper::PriorGlobalPositioning(
    const GlobalPositionerOptions& options,
    double max_position_prior_deviation,
    bool clamp_positions_to_prior,
    double max_angular_reproj_error_deg,
    double max_normalized_reproj_error,
    double min_tri_angle_deg) {
  // Override random initialisation: use the GPS-seeded translations set by
  // RestoreTranslationsFromPriors as starting positions for BATA.
  GlobalPositionerOptions custom_opts = options;
  custom_opts.generate_random_positions = false;

  if (!RunGlobalPositioning(custom_opts, GetPoseGraph(), GetReconstruction())) {
    return false;
  }

  EnforceMaxPositionPriorDeviation(database_cache_->PosePriors(),
                                   max_position_prior_deviation,
                                   -1.0,
                                   -1,
                                   clamp_positions_to_prior,
                                   "after global positioning");

  auto& recon = GetReconstruction();
  ObservationManager obs_manager(recon);

  const std::unordered_set<point3D_t> point3D_ids_snapshot = recon.Point3DIds();

  // First pass: relaxed angular threshold for cameras without prior focal.
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      2.0 * max_angular_reproj_error_deg,
      point3D_ids_snapshot,
      ReprojectionErrorType::ANGULAR);

  // Second pass: strict threshold for cameras with prior focal length.
  const double max_angular_error_rad = DegToRad(max_angular_reproj_error_deg);
  std::vector<std::pair<image_t, point2D_t>> obs_to_delete;
  for (const point3D_t point3D_id : point3D_ids_snapshot) {
    if (!recon.ExistsPoint3D(point3D_id)) continue;
    const auto& point3D = recon.Point3D(point3D_id);
    for (const auto& track_el : point3D.track.Elements()) {
      const auto& img = recon.Image(track_el.image_id);
      const auto& camera = *img.CameraPtr();
      if (!camera.has_prior_focal_length) continue;
      const auto& point2D = img.Point2D(track_el.point2D_idx);
      const double error = CalculateAngularReprojectionError(
          point2D.xy, point3D.xyz, img.CamFromWorld(), camera);
      if (error > max_angular_error_rad) {
        obs_to_delete.emplace_back(track_el.image_id, track_el.point2D_idx);
      }
    }
  }
  for (const auto& [img_id, point2D_idx] : obs_to_delete) {
    if (recon.Image(img_id).Point2D(point2D_idx).HasPoint3D()) {
      obs_manager.DeleteObservation(img_id, point2D_idx);
    }
  }

  obs_manager.FilterPoints3DWithSmallTriangulationAngle(min_tri_angle_deg,
                                                         recon.Point3DIds());
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      10.0 * max_normalized_reproj_error,
      recon.Point3DIds(),
      ReprojectionErrorType::NORMALIZED);

  // NOTE: reconstruction_->Normalize() is intentionally skipped here.
  // GPS priors already provide absolute scale; normalising would destroy it.
  return true;
}

bool PriorGlobalMapper::RunPosePriorBundleAdjustment(
    const BundleAdjustmentOptions& ba_options,
    const PosePriorBundleAdjustmentOptions& prior_options) {
  auto& recon = GetReconstruction();
  if (recon.NumImages() == 0) {
    LOG(ERROR) << "Cannot run pose-prior bundle adjustment: no images";
    return false;
  }
  if (recon.NumPoints3D() == 0) {
    LOG(ERROR) << "Cannot run pose-prior bundle adjustment: no 3D points";
    return false;
  }

  BundleAdjustmentConfig ba_config;
  for (const auto& [img_id, img] : recon.Images()) {
    if (img.HasPose()) ba_config.AddImage(img_id);
  }
  // No FixGauge: GPS prior constraints absorb the gauge DOF.

  auto ba = CreatePosePriorBundleAdjuster(ba_options,
                                          prior_options,
                                          ba_config,
                                          database_cache_->PosePriors(),
                                          recon);
  return ba->Solve()->IsSolutionUsable();
}

bool PriorGlobalMapper::IterativePriorBundleAdjustment(
    const BundleAdjustmentOptions& options,
    const PosePriorBundleAdjustmentOptions& prior_options,
    double max_normalized_reproj_error,
    double min_tri_angle_deg,
    int num_iterations,
    bool skip_fixed_rotation_stage,
    bool skip_joint_optimization_stage) {
  auto& recon = GetReconstruction();
  for (int ite = 0; ite < num_iterations; ite++) {
    // --- Fixed-rotation stage: optimise positions only -------------------
    if (!skip_fixed_rotation_stage) {
      BundleAdjustmentOptions opts_position_only = options;
      opts_position_only.constant_rig_from_world_rotation = true;
      if (!RunPosePriorBundleAdjustment(opts_position_only, prior_options)) {
        return false;
      }
      LOG(INFO) << "Prior BA iteration " << ite + 1 << " / " << num_iterations
                << ", fixed-rotation stage finished";
    }

    // --- Joint optimisation stage ----------------------------------------
    if (!skip_joint_optimization_stage) {
      if (!RunPosePriorBundleAdjustment(options, prior_options)) {
        return false;
      }
    }
    LOG(INFO) << "Prior BA iteration " << ite + 1 << " / " << num_iterations
              << " finished";

    // NOTE: reconstruction_->Normalize() is intentionally skipped here.
    // GPS priors already provide absolute scale.

    // Progressive track filtering (threshold tightens each round).
    ObservationManager obs_manager(recon);
    bool saturated = true;
    size_t filtered_num = 0;
    while (saturated && ite < num_iterations) {
      const double scaling = static_cast<double>(std::max(3 - ite, 1));
      filtered_num += obs_manager.FilterPoints3DWithLargeReprojectionError(
          scaling * max_normalized_reproj_error,
          recon.Point3DIds(),
          ReprojectionErrorType::NORMALIZED);
      if (filtered_num > 1e-3 * recon.NumPoints3D()) {
        saturated = false;
      } else {
        ite++;
      }
    }
    if (saturated) {
      LOG(INFO) << "Fewer than 0.1% tracks filtered; stopping BA early.";
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

bool PriorGlobalMapper::IterativePriorRetriangulateAndRefine(
    const IncrementalTriangulator::Options& tri_options,
    const BundleAdjustmentOptions& ba_options,
    const PriorGlobalMapperOptions& mapper_options,
    double max_normalized_reproj_error,
    double min_tri_angle_deg) {
  auto& recon = GetReconstruction();

  // Remove all existing 3D points and re-triangulate from scratch using the
  // now well-refined poses.
  recon.DeleteAllPoints2DAndPoints3D();

  // Triangulate using IncrementalMapper (reuses the same database_cache_).
  IncrementalMapper mapper(database_cache_);
  // Use the shared_ptr to reconstruction from the parent getter.
  mapper.BeginReconstruction(GlobalMapper::Reconstruction());

  for (const auto img_id : recon.RegImageIds()) {
    mapper.TriangulateImage(tri_options, img_id);
  }

  // Inner-loop BA options: fewer iterations, looser tolerances.
  BundleAdjustmentOptions custom_ba_options;
  custom_ba_options.print_summary = false;
  if (custom_ba_options.ceres && ba_options.ceres) {
    custom_ba_options.ceres->solver_options.num_threads =
        ba_options.ceres->solver_options.num_threads;
    custom_ba_options.ceres->solver_options.max_num_iterations = 50;
    custom_ba_options.ceres->solver_options.max_linear_solver_iterations = 100;
  }

  // Configure the incremental mapper to use PosePrior BA for global
  // refinement iterations, and to skip Normalize() (handled automatically
  // via use_prior_position).
  IncrementalMapper::Options inc_options;
  inc_options.random_seed = tri_options.random_seed;
  inc_options.use_prior_position = true;
  inc_options.use_robust_loss_on_prior_position =
      mapper_options.use_robust_loss_on_prior_position;
  inc_options.prior_position_loss_scale =
      mapper_options.prior_position_loss_scale;

  mapper.IterativeGlobalRefinement(/*max_num_refinements=*/5,
                                   /*max_refinement_change=*/0.0005,
                                   inc_options,
                                   custom_ba_options,
                                   tri_options,
                                   /*normalize_reconstruction=*/false);

  mapper.EndReconstruction(/*discard=*/false);

  // Filter before final BA.
  {
    ObservationManager obs_manager(recon);
    obs_manager.FilterPoints3DWithLargeReprojectionError(
        max_normalized_reproj_error,
        recon.Point3DIds(),
        ReprojectionErrorType::NORMALIZED);
  }

  // Final BA with GPS prior constraints.
  if (!RunPosePriorBundleAdjustment(ba_options, mapper_options.pose_prior_ba)) {
    return false;
  }

  // NOTE: reconstruction_->Normalize() is intentionally skipped.

  ObservationManager obs_manager_final(recon);
  obs_manager_final.FilterPoints3DWithLargeReprojectionError(
      max_normalized_reproj_error,
      recon.Point3DIds(),
      ReprojectionErrorType::NORMALIZED);
  obs_manager_final.FilterPoints3DWithSmallTriangulationAngle(
      min_tri_angle_deg, recon.Point3DIds());
  return true;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

bool PriorGlobalMapper::Solve(const PriorGlobalMapperOptions& options,
                              std::unordered_map<frame_t, int>& cluster_ids) {
  // This deliberately mirrors GlobalMapper::Solve() structure so that
  // upstream diffs remain easy to apply.
  if (GetPoseGraph().Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  // Propagate seeds / thread counts (mirrors InitializeOptions in parent).
  PriorGlobalMapperOptions opts = InitializePriorOptions(options);

  // Determine whether GPS constraints are actually available.
  const bool use_prior = ShouldUsePriorPosition(opts);

  // Build the pose prior BA options once (avoids repeating the override).
  PosePriorBundleAdjustmentOptions prior_ba_opts = opts.pose_prior_ba;
  if (use_prior) {
    if (opts.use_robust_loss_on_prior_position) {
      if (!prior_ba_opts.ceres) {
        prior_ba_opts = PosePriorBundleAdjustmentOptions{};
      }
      prior_ba_opts.ceres->prior_position_loss_function_type =
          CeresBundleAdjustmentOptions::LossFunctionType::CAUCHY;
    }
    prior_ba_opts.ceres->prior_position_loss_scale =
        opts.prior_position_loss_scale;
    prior_ba_opts.alignment_ransac_options.random_seed = opts.random_seed;

    LOG(INFO) << "PriorGlobalMapper: GPS-constrained solving ENABLED ("
              << database_cache_->PosePriors().size() << " priors loaded).";
  } else {
    LOG(INFO)
        << "PriorGlobalMapper: GPS constraints not available; using standard "
           "global solver.";
  }

  // -------------------------------------------------------------------------
  // Step 1: Rotation averaging (unchanged)
  // -------------------------------------------------------------------------
  if (!opts.skip_rotation_averaging) {
    LOG_HEADING1("Running rotation averaging");
    Timer timer;
    timer.Start();
    if (!RotationAveraging(opts.rotation_averaging)) return false;
    LOG(INFO) << "Rotation averaging done in " << timer.ElapsedSeconds()
              << " seconds";

    // After rotation averaging, translations may be arbitrary (often 0).
    // Overwrite them with GPS-based values so GlobalPositioner starts from
    // a metric-consistent initialisation.
    if (use_prior) {
      RestoreTranslationsFromPriors(database_cache_->PosePriors());
    }
  }

  // -------------------------------------------------------------------------
  // Step 2: Track establishment (unchanged)
  // -------------------------------------------------------------------------
  if (!opts.skip_track_establishment) {
    LOG_HEADING1("Running track establishment");
    Timer timer;
    timer.Start();
    EstablishTracks(opts);
    LOG(INFO) << "Track establishment done in " << timer.ElapsedSeconds()
              << " seconds";
  }

  // -------------------------------------------------------------------------
  // Step 3: Global positioning
  //   GPS path  – skips random init; skips Normalize()
  //   Normal path – delegates directly to parent
  // -------------------------------------------------------------------------
  if (!opts.skip_global_positioning) {
    LOG_HEADING1("Running global positioning");
    Timer timer;
    timer.Start();
    bool ok;
    if (use_prior) {
      ok = PriorGlobalPositioning(opts.global_positioning,
                                   opts.max_position_prior_deviation,
                                   opts.clamp_positions_to_prior_after_global_positioning,
                                   opts.max_angular_reproj_error_deg,
                                   opts.max_normalized_reproj_error,
                                   opts.min_tri_angle_deg);
    } else {
      ok = GlobalPositioning(opts.global_positioning,
                              opts.max_angular_reproj_error_deg,
                              opts.max_normalized_reproj_error,
                              opts.min_tri_angle_deg);
    }
    if (!ok) return false;
    LOG(INFO) << "Global positioning done in " << timer.ElapsedSeconds()
              << " seconds";
  }

  // -------------------------------------------------------------------------
  // Step 4: Bundle adjustment
  //   GPS path  – PosePriorBundleAdjuster; no Normalize()
  //   Normal path – delegates directly to parent
  // -------------------------------------------------------------------------
  if (!opts.skip_bundle_adjustment) {
    LOG_HEADING1("Running iterative bundle adjustment");
    Timer timer;
    timer.Start();
    bool ok;
    if (use_prior) {
      ok = IterativePriorBundleAdjustment(opts.bundle_adjustment,
                                          prior_ba_opts,
                                          opts.max_normalized_reproj_error,
                                          opts.min_tri_angle_deg,
                                          opts.ba_num_iterations,
                                          opts.ba_skip_fixed_rotation_stage,
                                          opts.ba_skip_joint_optimization_stage);
    } else {
      ok = IterativeBundleAdjustment(opts.bundle_adjustment,
                                     opts.max_normalized_reproj_error,
                                     opts.min_tri_angle_deg,
                                     opts.ba_num_iterations,
                                     opts.ba_skip_fixed_rotation_stage,
                                     opts.ba_skip_joint_optimization_stage);
    }
    if (!ok) return false;
    LOG(INFO) << "Iterative bundle adjustment done in " << timer.ElapsedSeconds()
              << " seconds";
  }

  // -------------------------------------------------------------------------
  // Step 5: Retriangulation
  //   GPS path  – PosePriorBundleAdjuster; no Normalize()
  //   Normal path – delegates directly to parent
  // -------------------------------------------------------------------------
  if (!opts.skip_retriangulation) {
    LOG_HEADING1("Running iterative retriangulation and refinement");
    Timer timer;
    timer.Start();
    bool ok;
    if (use_prior) {
      ok = IterativePriorRetriangulateAndRefine(opts.retriangulation,
                                                opts.bundle_adjustment,
                                                opts,
                                                opts.max_normalized_reproj_error,
                                                opts.min_tri_angle_deg);
    } else {
      ok = IterativeRetriangulateAndRefine(opts.retriangulation,
                                           opts.bundle_adjustment,
                                           opts.max_normalized_reproj_error,
                                           opts.min_tri_angle_deg);
    }
    if (!ok) return false;
    LOG(INFO) << "Iterative retriangulation done in " << timer.ElapsedSeconds()
              << " seconds";
  }

  return true;
}

}  // namespace colmap

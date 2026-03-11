// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "colmap/sfm/six_dof_prior_global_mapper.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/cost_functions/lidar.h"
#include "colmap/estimators/lidar_bundle_adjustment.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/scene/projection.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace colmap {
namespace {

struct RetriangulationDebugTarget {
  std::string image_name;
  std::vector<std::string> neighbor_names;
};

struct CompletionEdgeStats {
  std::string target_name;
  std::string neighbor_name;
  size_t candidate_correspondences = 0;
  size_t already_triangulated = 0;
  size_t reproj_accepted = 0;
  size_t reproj_rejected = 0;
};

struct MergeEdgeStats {
  std::string target_name;
  std::string neighbor_name;
  size_t candidate_merges = 0;
  size_t reproj_accepted = 0;
  size_t reproj_rejected = 0;
};

struct ExternalResidualEntry {
  std::string target_name;
  std::string neighbor_name;
  point3D_t target_point3D_id = kInvalidPoint3DId;
  point2D_t target_point2D_idx = kInvalidPoint2DIdx;
  point2D_t neighbor_point2D_idx = kInvalidPoint2DIdx;
  point3D_t neighbor_point3D_id = kInvalidPoint3DId;
  bool neighbor_has_point3D = false;
  double reproj_error_px = 0.0;
};

struct TrackedPseudoPoint {
  point3D_t point3D_id = kInvalidPoint3DId;
  std::string neighbor_name;
  double distance_to_neighbor_center = 0.0;
  double ray_angle_deg = 0.0;
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
};

PriorGlobalMapperOptions InitializeSixDofOptions(
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

bool FrameHasAllCameraImagesLoaded(const Reconstruction& reconstruction,
                                   const Frame& frame) {
  for (const auto& data_id : frame.ImageIds()) {
    if (!reconstruction.ExistsImage(static_cast<image_t>(data_id.id))) {
      return false;
    }
  }
  return true;
}

bool ImageHasFinitePose(const Image& image) {
  return image.HasPose() && image.CamFromWorld().params.allFinite();
}

bool IsFinalRetriangulationStage(const char* stage_name) {
  return stage_name != nullptr && std::string(stage_name) == "final";
}

std::unordered_set<point3D_t> CollectStableEarlyLidarAnchorPointIds(
    const Reconstruction& reconstruction,
    int min_track_length,
    double max_mean_reprojection_error) {
  std::unordered_set<point3D_t> point3D_ids;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (min_track_length > 0 &&
        static_cast<int>(point3D.track.Length()) < min_track_length) {
      continue;
    }
    if (max_mean_reprojection_error > 0.0 &&
        point3D.error > max_mean_reprojection_error) {
      continue;
    }
    point3D_ids.insert(point3D_id);
  }
  return point3D_ids;
}

size_t FilterLidarConstraintsToEligiblePointIds(
    std::vector<LidarConstraint>* constraints,
    const std::unordered_set<point3D_t>& eligible_point3D_ids) {
  if (constraints == nullptr || eligible_point3D_ids.empty()) {
    const size_t removed = constraints != nullptr ? constraints->size() : 0;
    if (constraints != nullptr) {
      constraints->clear();
    }
    return removed;
  }

  const size_t old_size = constraints->size();
  constraints->erase(
      std::remove_if(
          constraints->begin(),
          constraints->end(),
          [&](const LidarConstraint& constraint) {
            return eligible_point3D_ids.count(constraint.point3D_id) == 0;
          }),
      constraints->end());
  return old_size - constraints->size();
}

size_t CountImagePoint3DObservations(const Image& image) {
  return image.NumPoints3D();
}

double ImagePositionPriorDeviation(const Image& image,
                                  const SixDofPosePrior& prior) {
  return (image.ProjectionCenter() - Inverse(prior.cam_from_world).translation())
      .norm();
}

double RiskStratifiedPriorWeightMultiplier(
    const PriorGlobalMapperOptions& options,
    const Image& image,
  const SixDofPosePrior* pose_prior) {
  if (!options.use_risk_stratified_prior_weights_in_final_joint_ba ||
      pose_prior == nullptr || !pose_prior->HasPose()) {
    return 1.0;
  }

  const int num_observations = static_cast<int>(CountImagePoint3DObservations(image));
  const double deviation = ImagePositionPriorDeviation(image, *pose_prior);

  if ((options.final_joint_ba_high_risk_max_observations >= 0 &&
       num_observations <= options.final_joint_ba_high_risk_max_observations) ||
      (options.final_joint_ba_high_risk_position_prior_deviation > 0.0 &&
       deviation > options.final_joint_ba_high_risk_position_prior_deviation)) {
    return std::max(1.0,
                    options.final_joint_ba_high_risk_prior_weight_multiplier);
  }

  if (options.final_joint_ba_medium_risk_position_prior_deviation > 0.0 &&
      deviation > options.final_joint_ba_medium_risk_position_prior_deviation) {
    return std::max(1.0,
                    options.final_joint_ba_medium_risk_prior_weight_multiplier);
  }

  return 1.0;
}

void StrengthenAbsolutePosePrior(
    AbsolutePosePriorConstraint& absolute_pose_prior,
    const PosePriorBundleAdjustmentOptions& prior_options,
    double prior_rotation_fallback_stddev_rad,
    double weight_multiplier) {
  if (weight_multiplier <= 1.0) {
    return;
  }

  const double variance_scale =
      1.0 / (weight_multiplier * weight_multiplier);
  const double position_fallback_variance =
      prior_options.prior_position_fallback_stddev *
      prior_options.prior_position_fallback_stddev * variance_scale;
  const double rotation_fallback_variance =
      prior_rotation_fallback_stddev_rad * prior_rotation_fallback_stddev_rad *
      variance_scale;

  if (absolute_pose_prior.HasPositionCov()) {
    absolute_pose_prior.position_covariance *= variance_scale;
  } else {
    absolute_pose_prior.position_covariance =
        position_fallback_variance * Eigen::Matrix3d::Identity();
  }

  if (absolute_pose_prior.HasRotationCov()) {
    absolute_pose_prior.rotation_covariance *= variance_scale;
  } else {
    absolute_pose_prior.rotation_covariance =
        rotation_fallback_variance * Eigen::Matrix3d::Identity();
  }
}

double RotationErrorDeg(const Eigen::Quaterniond& lhs,
                        const Eigen::Quaterniond& rhs) {
  return RadToDeg(lhs.angularDistance(rhs));
}

double TranslationDirectionErrorDeg(const Eigen::Vector3d& lhs,
                                    const Eigen::Vector3d& rhs) {
  constexpr double kMinNorm = 1e-8;
  const double lhs_norm = lhs.norm();
  const double rhs_norm = rhs.norm();
  if (!lhs.allFinite() || !rhs.allFinite() || lhs_norm < kMinNorm ||
      rhs_norm < kMinNorm) {
    return 0.0;
  }
  double cos_angle = lhs.dot(rhs) / (lhs_norm * rhs_norm);
  cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
  return RadToDeg(std::acos(cos_angle));
}

std::string JoinImageIds(const std::vector<image_t>& image_ids) {
  std::ostringstream stream;
  for (size_t idx = 0; idx < image_ids.size(); ++idx) {
    if (idx > 0) {
      stream << ", ";
    }
    stream << image_ids[idx];
  }
  return stream.str();
}

const std::vector<RetriangulationDebugTarget>&
GetRetriangulationDebugTargets() {
  static const std::vector<RetriangulationDebugTarget> kTargets = {
      {"right/right_00282.jpg",
       {"right/right_00280.jpg",
        "right/right_00281.jpg",
        "right/right_00283.jpg",
        "right/right_00309.jpg"}},
      {"right/right_00283.jpg",
       {"right/right_00280.jpg",
        "right/right_00281.jpg",
        "right/right_00282.jpg",
        "right/right_00309.jpg"}},
      {"left/left_00155.jpg",
       {"left/left_00141.jpg",
        "left/left_00142.jpg",
        "left/left_00156.jpg",
        "left/left_00157.jpg",
        "left/left_00158.jpg"}},
  };
  return kTargets;
}

const std::vector<RetriangulationDebugTarget>& GetRightRetriangulationDebugTargets() {
  static const std::vector<RetriangulationDebugTarget> kTargets = {
      {"right/right_00282.jpg",
       {"right/right_00280.jpg",
        "right/right_00281.jpg",
        "right/right_00283.jpg",
        "right/right_00309.jpg"}},
      {"right/right_00283.jpg",
       {"right/right_00280.jpg",
        "right/right_00281.jpg",
        "right/right_00282.jpg",
        "right/right_00309.jpg"}},
  };
  return kTargets;
}

Eigen::Vector3d WorldRayFromPoint2D(const Image& image, const Point2D& point2D) {
  const std::optional<Eigen::Vector2d> cam_point =
      image.CameraPtr()->CamFromImg(point2D.xy);
  if (!cam_point.has_value()) {
    return Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  }

  Eigen::Vector3d cam_ray(cam_point->x(), cam_point->y(), 1.0);
  cam_ray.normalize();
  return Inverse(image.CamFromWorld()).rotation() * cam_ray;
}

bool IsTrackedPseudoStage(const std::string& stage_name) {
  static const std::unordered_set<std::string> kStages = {
      "after_initial_retriangulate",
      "refinement_1_after_ba_and_normalize",
      "refinement_1_after_filter_points",
      "after_final_visual_ba_and_normalize",
      "after_final_large_reproj_filter_post_ba",
      "after_final_small_tri_filter",
  };
  return kStages.count(stage_name) > 0;
}

std::vector<ExternalResidualEntry> ComputeExternalResidualEntries(
    const Reconstruction& reconstruction,
    const CorrespondenceGraph& correspondence_graph) {
  std::vector<ExternalResidualEntry> entries;
  std::unordered_map<std::string, image_t> image_ids;
  for (const auto& target : GetRightRetriangulationDebugTargets()) {
    const Image* target_image = reconstruction.FindImageWithName(target.image_name);
    if (target_image == nullptr) {
      continue;
    }
    image_ids[target.image_name] = target_image->ImageId();
    for (const std::string& neighbor_name : target.neighbor_names) {
      const Image* neighbor_image = reconstruction.FindImageWithName(neighbor_name);
      if (neighbor_image != nullptr) {
        image_ids[neighbor_name] = neighbor_image->ImageId();
      }
    }
  }

  for (const auto& target : GetRightRetriangulationDebugTargets()) {
    const Image* target_image = reconstruction.FindImageWithName(target.image_name);
    if (target_image == nullptr) {
      continue;
    }

    for (const std::string& neighbor_name : target.neighbor_names) {
      const auto neighbor_it = image_ids.find(neighbor_name);
      if (neighbor_it == image_ids.end()) {
        continue;
      }
      const Image& neighbor_image = reconstruction.Image(neighbor_it->second);
      if (!neighbor_image.HasPose()) {
        continue;
      }
      const Camera& neighbor_camera = *neighbor_image.CameraPtr();

      std::unordered_set<uint64_t> seen_pairs;
      for (point2D_t target_point2D_idx = 0;
           target_point2D_idx < target_image->NumPoints2D();
           ++target_point2D_idx) {
        const Point2D& target_point2D = target_image->Point2D(target_point2D_idx);
        if (!target_point2D.HasPoint3D() ||
            !reconstruction.ExistsPoint3D(target_point2D.point3D_id)) {
          continue;
        }

        const Point3D& point3D = reconstruction.Point3D(target_point2D.point3D_id);
        for (const TrackElement& track_el : point3D.track.Elements()) {
          const auto corr_range = correspondence_graph.FindCorrespondences(
              track_el.image_id, track_el.point2D_idx);
          for (const auto* corr = corr_range.beg; corr < corr_range.end; ++corr) {
            if (corr->image_id != neighbor_image.ImageId()) {
              continue;
            }

            const uint64_t pair_key =
                (static_cast<uint64_t>(target_point2D_idx) << 32) |
                static_cast<uint64_t>(corr->point2D_idx);
            if (!seen_pairs.insert(pair_key).second) {
              continue;
            }

            const Point2D& neighbor_point2D =
                neighbor_image.Point2D(corr->point2D_idx);
            ExternalResidualEntry entry;
            entry.target_name = target.image_name;
            entry.neighbor_name = neighbor_name;
            entry.target_point3D_id = target_point2D.point3D_id;
            entry.target_point2D_idx = target_point2D_idx;
            entry.neighbor_point2D_idx = corr->point2D_idx;
            entry.neighbor_has_point3D = neighbor_point2D.HasPoint3D();
            entry.neighbor_point3D_id = neighbor_point2D.point3D_id;
            entry.reproj_error_px = std::sqrt(CalculateSquaredReprojectionError(
                neighbor_point2D.xy,
                point3D.xyz,
                neighbor_image.CamFromWorld(),
                neighbor_camera));
            entries.push_back(entry);
          }
        }
      }
    }
  }

  return entries;
}

void LogExternalResidualEntries(const std::vector<ExternalResidualEntry>& entries,
                                const std::string& stage_name,
                                size_t top_k) {
  LOG(INFO) << "6DoF external correspondence residuals [" << stage_name << "]";

  for (const auto& target : GetRightRetriangulationDebugTargets()) {
    for (const std::string& neighbor_name : target.neighbor_names) {
      std::vector<ExternalResidualEntry> pair_entries;
      pair_entries.reserve(entries.size());
      for (const auto& entry : entries) {
        if (entry.target_name == target.image_name &&
            entry.neighbor_name == neighbor_name) {
          pair_entries.push_back(entry);
        }
      }
      if (pair_entries.empty()) {
        continue;
      }

      auto log_group = [&](bool neighbor_has_point3D) {
        std::vector<ExternalResidualEntry> filtered;
        for (const auto& entry : pair_entries) {
          if (entry.neighbor_has_point3D == neighbor_has_point3D) {
            filtered.push_back(entry);
          }
        }
        if (filtered.empty()) {
          return;
        }

        std::sort(filtered.begin(), filtered.end(), [](const auto& lhs, const auto& rhs) {
          return lhs.reproj_error_px > rhs.reproj_error_px;
        });
        const size_t count = std::min(top_k, filtered.size());
        LOG(INFO) << "  target=" << target.image_name
                  << " neighbor=" << neighbor_name
                  << " category="
                  << (neighbor_has_point3D ? "neighbor_already_triangulated" : "neighbor_untriangulated")
                  << " top_count=" << count;
        for (size_t idx = 0; idx < count; ++idx) {
          const auto& entry = filtered[idx];
          LOG(INFO) << "    reproj_error_px=" << entry.reproj_error_px
                    << " target_point3D_id=" << entry.target_point3D_id
                    << " target_point2D_idx=" << entry.target_point2D_idx
                    << " neighbor_point2D_idx=" << entry.neighbor_point2D_idx
                    << " neighbor_point3D_id=" << entry.neighbor_point3D_id;
        }
      };

      log_group(false);
      log_group(true);
    }
  }
}

std::vector<TrackedPseudoPoint> IdentifyLeftPseudoPoints(
    const Reconstruction& reconstruction,
    double max_distance_to_neighbor_center,
    double min_ray_angle_deg) {
  std::vector<TrackedPseudoPoint> tracked_points;
  const std::string target_name = "left/left_00155.jpg";
  const Image* target_image = reconstruction.FindImageWithName(target_name);
  if (target_image == nullptr || !target_image->HasPose()) {
    return tracked_points;
  }

  static const std::vector<std::string> kNeighbors = {
      "left/left_00156.jpg",
      "left/left_00157.jpg",
      "left/left_00158.jpg",
  };

  std::unordered_set<point3D_t> seen_ids;
  for (point2D_t target_point2D_idx = 0;
       target_point2D_idx < target_image->NumPoints2D();
       ++target_point2D_idx) {
    const Point2D& target_point2D = target_image->Point2D(target_point2D_idx);
    if (!target_point2D.HasPoint3D() ||
        !reconstruction.ExistsPoint3D(target_point2D.point3D_id) ||
        !seen_ids.insert(target_point2D.point3D_id).second) {
      continue;
    }

    const Point3D& point3D = reconstruction.Point3D(target_point2D.point3D_id);
    for (const std::string& neighbor_name : kNeighbors) {
      const Image* neighbor_image = reconstruction.FindImageWithName(neighbor_name);
      if (neighbor_image == nullptr || !neighbor_image->HasPose()) {
        continue;
      }

      bool observed_by_neighbor = false;
      point2D_t neighbor_point2D_idx = kInvalidPoint2DIdx;
      for (const TrackElement& track_el : point3D.track.Elements()) {
        if (track_el.image_id == neighbor_image->ImageId()) {
          observed_by_neighbor = true;
          neighbor_point2D_idx = track_el.point2D_idx;
          break;
        }
      }
      if (!observed_by_neighbor) {
        continue;
      }

      const double distance_to_neighbor =
          (point3D.xyz - neighbor_image->ProjectionCenter()).norm();
      const Eigen::Vector3d target_ray =
          WorldRayFromPoint2D(*target_image, target_point2D);
      const Eigen::Vector3d neighbor_ray =
          WorldRayFromPoint2D(*neighbor_image,
                              neighbor_image->Point2D(neighbor_point2D_idx));
      const double ray_angle_deg = RadToDeg(std::acos(
          std::max(-1.0, std::min(1.0, target_ray.dot(neighbor_ray)))));

      if (distance_to_neighbor <= max_distance_to_neighbor_center &&
          ray_angle_deg >= min_ray_angle_deg) {
        tracked_points.push_back({target_point2D.point3D_id,
                                  neighbor_name,
                                  distance_to_neighbor,
                                  ray_angle_deg,
                                  point3D.xyz});
        break;
      }
    }
  }

  std::sort(tracked_points.begin(), tracked_points.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.distance_to_neighbor_center != rhs.distance_to_neighbor_center) {
      return lhs.distance_to_neighbor_center < rhs.distance_to_neighbor_center;
    }
    return lhs.point3D_id < rhs.point3D_id;
  });
  return tracked_points;
}

void LogTrackedPseudoPointResiduals(const Reconstruction& reconstruction,
                                    const std::vector<TrackedPseudoPoint>& tracked_points,
                                    const std::string& stage_name) {
  if (!IsTrackedPseudoStage(stage_name) || tracked_points.empty()) {
    return;
  }

  LOG(INFO) << "6DoF left_00155 tracked pseudo points [" << stage_name << "]";
  for (const auto& tracked_point : tracked_points) {
    if (!reconstruction.ExistsPoint3D(tracked_point.point3D_id)) {
      LOG(INFO) << "  point3D_id=" << tracked_point.point3D_id
                << " status=deleted"
                << " anchor_neighbor=" << tracked_point.neighbor_name;
      continue;
    }

    const Point3D& point3D = reconstruction.Point3D(tracked_point.point3D_id);
    double max_reproj_error_px = 0.0;
    double target_reproj_error_px = -1.0;
    double anchor_neighbor_reproj_error_px = -1.0;
    size_t track_length = point3D.track.Length();
    for (const TrackElement& track_el : point3D.track.Elements()) {
      const Image& image = reconstruction.Image(track_el.image_id);
      const Camera& camera = *image.CameraPtr();
      const Point2D& point2D = image.Point2D(track_el.point2D_idx);
      const double reproj_error_px = std::sqrt(CalculateSquaredReprojectionError(
          point2D.xy, point3D.xyz, image.CamFromWorld(), camera));
      max_reproj_error_px = std::max(max_reproj_error_px, reproj_error_px);
      if (image.Name() == "left/left_00155.jpg") {
        target_reproj_error_px = reproj_error_px;
      }
      if (image.Name() == tracked_point.neighbor_name) {
        anchor_neighbor_reproj_error_px = reproj_error_px;
      }
    }

    LOG(INFO) << "  point3D_id=" << tracked_point.point3D_id
              << " status=alive"
              << " anchor_neighbor=" << tracked_point.neighbor_name
              << " dist_to_neighbor_center=" << tracked_point.distance_to_neighbor_center
              << " initial_ray_angle_deg=" << tracked_point.ray_angle_deg
              << " track_length=" << track_length
              << " target_reproj_error_px=" << target_reproj_error_px
              << " anchor_neighbor_reproj_error_px=" << anchor_neighbor_reproj_error_px
              << " max_reproj_error_px=" << max_reproj_error_px;
  }
}

std::vector<CompletionEdgeStats> ComputeCompletionDiagnostics(
    const Reconstruction& reconstruction,
    const CorrespondenceGraph& correspondence_graph,
    const IncrementalTriangulator::Options& tri_options) {
  std::vector<CompletionEdgeStats> stats;
  std::unordered_map<std::string, image_t> image_ids;
  for (const auto& target : GetRightRetriangulationDebugTargets()) {
    const Image* target_image = reconstruction.FindImageWithName(target.image_name);
    if (target_image == nullptr) {
      continue;
    }
    image_ids[target.image_name] = target_image->ImageId();
    for (const std::string& neighbor_name : target.neighbor_names) {
      const Image* neighbor_image = reconstruction.FindImageWithName(neighbor_name);
      if (neighbor_image != nullptr) {
        image_ids[neighbor_name] = neighbor_image->ImageId();
      }
    }
  }

  const double max_squared_reproj_error =
      tri_options.complete_max_reproj_error * tri_options.complete_max_reproj_error;

  for (const auto& target : GetRightRetriangulationDebugTargets()) {
    const Image* target_image = reconstruction.FindImageWithName(target.image_name);
    if (target_image == nullptr) {
      continue;
    }
    for (const std::string& neighbor_name : target.neighbor_names) {
      CompletionEdgeStats edge_stats;
      edge_stats.target_name = target.image_name;
      edge_stats.neighbor_name = neighbor_name;
      const auto neighbor_it = image_ids.find(neighbor_name);
      if (neighbor_it == image_ids.end()) {
        stats.push_back(edge_stats);
        continue;
      }
      const image_t neighbor_id = neighbor_it->second;
      const Image& neighbor_image = reconstruction.Image(neighbor_id);
      if (!neighbor_image.HasPose()) {
        stats.push_back(edge_stats);
        continue;
      }
      const Camera& neighbor_camera = *neighbor_image.CameraPtr();

      std::unordered_set<uint64_t> seen_pairs;
      for (const Point2D& point2D : target_image->Points2D()) {
        if (!point2D.HasPoint3D() ||
            !reconstruction.ExistsPoint3D(point2D.point3D_id)) {
          continue;
        }
        const Point3D& point3D = reconstruction.Point3D(point2D.point3D_id);
        for (const TrackElement& track_el : point3D.track.Elements()) {
          const auto corr_range = correspondence_graph.FindCorrespondences(
              track_el.image_id, track_el.point2D_idx);
          for (const auto* corr = corr_range.beg; corr < corr_range.end; ++corr) {
            if (corr->image_id != neighbor_id) {
              continue;
            }
            const uint64_t pair_key = (static_cast<uint64_t>(track_el.image_id) << 32) |
                                      static_cast<uint64_t>(corr->point2D_idx);
            if (!seen_pairs.insert(pair_key).second) {
              continue;
            }

            ++edge_stats.candidate_correspondences;
            const Point2D& neighbor_point2D =
                neighbor_image.Point2D(corr->point2D_idx);
            if (neighbor_point2D.HasPoint3D()) {
              ++edge_stats.already_triangulated;
              continue;
            }

            const double squared_reproj_error = CalculateSquaredReprojectionError(
                neighbor_point2D.xy,
                point3D.xyz,
                neighbor_image.CamFromWorld(),
                neighbor_camera);
            if (squared_reproj_error <= max_squared_reproj_error) {
              ++edge_stats.reproj_accepted;
            } else {
              ++edge_stats.reproj_rejected;
            }
          }
        }
      }
      stats.push_back(edge_stats);
    }
  }

  return stats;
}

std::vector<MergeEdgeStats> ComputeMergeDiagnostics(
    const Reconstruction& reconstruction,
    const CorrespondenceGraph& correspondence_graph,
    const IncrementalTriangulator::Options& tri_options) {
  std::vector<MergeEdgeStats> stats;
  std::unordered_map<std::string, image_t> image_ids;
  for (const auto& target : GetRightRetriangulationDebugTargets()) {
    const Image* target_image = reconstruction.FindImageWithName(target.image_name);
    if (target_image == nullptr) {
      continue;
    }
    image_ids[target.image_name] = target_image->ImageId();
    for (const std::string& neighbor_name : target.neighbor_names) {
      const Image* neighbor_image = reconstruction.FindImageWithName(neighbor_name);
      if (neighbor_image != nullptr) {
        image_ids[neighbor_name] = neighbor_image->ImageId();
      }
    }
  }

  const double max_squared_reproj_error =
      tri_options.merge_max_reproj_error * tri_options.merge_max_reproj_error;

  for (const auto& target : GetRightRetriangulationDebugTargets()) {
    const Image* target_image = reconstruction.FindImageWithName(target.image_name);
    if (target_image == nullptr) {
      continue;
    }
    for (const std::string& neighbor_name : target.neighbor_names) {
      MergeEdgeStats edge_stats;
      edge_stats.target_name = target.image_name;
      edge_stats.neighbor_name = neighbor_name;
      const auto neighbor_it = image_ids.find(neighbor_name);
      if (neighbor_it == image_ids.end()) {
        stats.push_back(edge_stats);
        continue;
      }
      const image_t neighbor_id = neighbor_it->second;

      std::unordered_set<uint64_t> seen_pairs;
      for (const Point2D& point2D : target_image->Points2D()) {
        if (!point2D.HasPoint3D() ||
            !reconstruction.ExistsPoint3D(point2D.point3D_id)) {
          continue;
        }
        const Point3D& point3D = reconstruction.Point3D(point2D.point3D_id);
        for (const TrackElement& track_el : point3D.track.Elements()) {
          const auto corr_range = correspondence_graph.FindCorrespondences(
              track_el.image_id, track_el.point2D_idx);
          for (const auto* corr = corr_range.beg; corr < corr_range.end; ++corr) {
            if (corr->image_id != neighbor_id) {
              continue;
            }

            const Image& corr_image = reconstruction.Image(corr->image_id);
            const Point2D& corr_point2D = corr_image.Point2D(corr->point2D_idx);
            if (!corr_point2D.HasPoint3D() ||
                corr_point2D.point3D_id == point2D.point3D_id ||
                !reconstruction.ExistsPoint3D(corr_point2D.point3D_id)) {
              continue;
            }

            const point3D_t lhs_id = std::min(point2D.point3D_id, corr_point2D.point3D_id);
            const point3D_t rhs_id = std::max(point2D.point3D_id, corr_point2D.point3D_id);
            const uint64_t pair_key = (static_cast<uint64_t>(lhs_id) << 32) |
                                      static_cast<uint64_t>(rhs_id);
            if (!seen_pairs.insert(pair_key).second) {
              continue;
            }

            ++edge_stats.candidate_merges;
            const Point3D& corr_point3D =
                reconstruction.Point3D(corr_point2D.point3D_id);
            const Eigen::Vector3d merged_xyz =
                (point3D.track.Length() * point3D.xyz +
                 corr_point3D.track.Length() * corr_point3D.xyz) /
                (point3D.track.Length() + corr_point3D.track.Length());

            bool merge_success = true;
            for (const Track* track : {&point3D.track, &corr_point3D.track}) {
              for (const TrackElement& test_track_el : track->Elements()) {
                const Image& test_image = reconstruction.Image(test_track_el.image_id);
                const Camera& test_camera = *test_image.CameraPtr();
                const Point2D& test_point2D =
                    test_image.Point2D(test_track_el.point2D_idx);
                if (CalculateSquaredReprojectionError(test_point2D.xy,
                                                      merged_xyz,
                                                      test_image.CamFromWorld(),
                                                      test_camera) >
                    max_squared_reproj_error) {
                  merge_success = false;
                  break;
                }
              }
              if (!merge_success) {
                break;
              }
            }

            if (merge_success) {
              ++edge_stats.reproj_accepted;
            } else {
              ++edge_stats.reproj_rejected;
            }
          }
        }
      }
      stats.push_back(edge_stats);
    }
  }

  return stats;
}

void LogCompletionDiagnostics(const std::vector<CompletionEdgeStats>& stats,
                              const std::string& stage_name) {
  LOG(INFO) << "6DoF complete-track diagnostics [" << stage_name << "]";
  for (const auto& stat : stats) {
    LOG(INFO) << "  target=" << stat.target_name
              << " neighbor=" << stat.neighbor_name
              << " candidates=" << stat.candidate_correspondences
              << " already_has_point3D=" << stat.already_triangulated
              << " reproj_accept=" << stat.reproj_accepted
              << " reproj_reject=" << stat.reproj_rejected;
  }
}

void LogMergeDiagnostics(const std::vector<MergeEdgeStats>& stats,
                         const std::string& stage_name) {
  LOG(INFO) << "6DoF merge-track diagnostics [" << stage_name << "]";
  for (const auto& stat : stats) {
    LOG(INFO) << "  target=" << stat.target_name
              << " neighbor=" << stat.neighbor_name
              << " candidate_merges=" << stat.candidate_merges
              << " reproj_accept=" << stat.reproj_accepted
              << " reproj_reject=" << stat.reproj_rejected;
  }
}

void LogLeftTargetTriangulationRays(const Reconstruction& reconstruction,
                                    const std::string& stage_name) {
  const std::string target_name = "left/left_00155.jpg";
  const Image* target_image = reconstruction.FindImageWithName(target_name);
  if (target_image == nullptr || !target_image->HasPose()) {
    return;
  }

  static const std::vector<std::string> kNeighbors = {
      "left/left_00141.jpg",
      "left/left_00142.jpg",
      "left/left_00156.jpg",
      "left/left_00157.jpg",
      "left/left_00158.jpg",
  };

  LOG(INFO) << "6DoF left_00155 triangulation rays [" << stage_name << "]";
  for (const std::string& neighbor_name : kNeighbors) {
    const Image* neighbor_image = reconstruction.FindImageWithName(neighbor_name);
    if (neighbor_image == nullptr || !neighbor_image->HasPose()) {
      continue;
    }

    for (point2D_t point2D_idx = 0; point2D_idx < target_image->NumPoints2D();
         ++point2D_idx) {
      const Point2D& target_point2D = target_image->Point2D(point2D_idx);
      if (!target_point2D.HasPoint3D() ||
          !reconstruction.ExistsPoint3D(target_point2D.point3D_id)) {
        continue;
      }

      const Point3D& point3D = reconstruction.Point3D(target_point2D.point3D_id);
      for (const TrackElement& track_el : point3D.track.Elements()) {
        if (track_el.image_id != neighbor_image->ImageId()) {
          continue;
        }

        const Point2D& neighbor_point2D =
            neighbor_image->Point2D(track_el.point2D_idx);
        const Eigen::Vector3d target_ray =
            WorldRayFromPoint2D(*target_image, target_point2D);
        const Eigen::Vector3d neighbor_ray =
            WorldRayFromPoint2D(*neighbor_image, neighbor_point2D);
        const double baseline_angle_deg = RadToDeg(std::acos(
            std::max(-1.0,
                     std::min(1.0, target_ray.dot(neighbor_ray)))));

        LOG(INFO) << "  neighbor=" << neighbor_name
                  << " point3D_id=" << target_point2D.point3D_id
                  << " xyz=[" << point3D.xyz.transpose() << "]"
                  << " target_point2D_idx=" << point2D_idx
                  << " target_center=[" << target_image->ProjectionCenter().transpose() << "]"
                  << " target_ray=[" << target_ray.transpose() << "]"
                  << " neighbor_point2D_idx=" << track_el.point2D_idx
                  << " neighbor_center=[" << neighbor_image->ProjectionCenter().transpose() << "]"
                  << " neighbor_ray=[" << neighbor_ray.transpose() << "]"
                  << " ray_angle_deg=" << baseline_angle_deg;
      }
    }
  }
}

bool RunVisualBundleAdjustment(const BundleAdjustmentOptions& options,
                               Reconstruction& reconstruction) {
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Cannot run bundle adjustment: no registered images";
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Cannot run bundle adjustment: no 3D points to optimize";
    return false;
  }

  BundleAdjustmentConfig ba_config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (image.HasPose()) {
      ba_config.AddImage(image_id);
    }
  }
  ba_config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  auto ba = CreateDefaultBundleAdjuster(options, ba_config, reconstruction);
  return ba->Solve()->IsSolutionUsable();
}

void LogRetriangulationDebugSnapshot(
    const Reconstruction& reconstruction,
    const std::unordered_map<image_t, const SixDofPosePrior*>& image_to_prior,
    const std::string& stage_name) {
  LOG(INFO) << "6DoF visual retriangulation snapshot [" << stage_name << "]";

  for (const auto& target : GetRetriangulationDebugTargets()) {
    const Image* target_image = reconstruction.FindImageWithName(target.image_name);
    if (target_image == nullptr) {
      LOG(INFO) << "  target=" << target.image_name << " missing_from_reconstruction";
      continue;
    }

    const bool target_has_pose = ImageHasFinitePose(*target_image);
    std::unordered_map<image_t, size_t> shared_track_counts;
    std::unordered_map<image_t, size_t> shared_observation_counts;
    size_t total_track_length = 0;
    size_t triangulated_point_count = 0;

    for (const Point2D& point2D : target_image->Points2D()) {
      if (!point2D.HasPoint3D() ||
          !reconstruction.ExistsPoint3D(point2D.point3D_id)) {
        continue;
      }

      const Point3D& point3D = reconstruction.Point3D(point2D.point3D_id);
      ++triangulated_point_count;
      total_track_length += point3D.track.Length();

      std::unordered_set<image_t> unique_neighbor_ids;
      for (const TrackElement& track_el : point3D.track.Elements()) {
        if (track_el.image_id == target_image->ImageId()) {
          continue;
        }
        ++shared_observation_counts[track_el.image_id];
        unique_neighbor_ids.insert(track_el.image_id);
      }
      for (const image_t neighbor_id : unique_neighbor_ids) {
        ++shared_track_counts[neighbor_id];
      }
    }

    const double mean_track_length =
        triangulated_point_count == 0
            ? 0.0
            : static_cast<double>(total_track_length) / triangulated_point_count;

    LOG(INFO) << "  target=" << target.image_name
              << " image_id=" << target_image->ImageId()
              << " has_pose=" << target_has_pose
              << " num_points2D=" << target_image->NumPoints2D()
              << " num_points3D=" << target_image->NumPoints3D()
              << " mean_track_length=" << mean_track_length;

    const auto target_prior_it = image_to_prior.find(target_image->ImageId());
    for (const std::string& neighbor_name : target.neighbor_names) {
      const Image* neighbor_image = reconstruction.FindImageWithName(neighbor_name);
      if (neighbor_image == nullptr) {
        LOG(INFO) << "    neighbor=" << neighbor_name
                  << " missing_from_reconstruction";
        continue;
      }

      const bool neighbor_has_pose = ImageHasFinitePose(*neighbor_image);
      double rotation_error_deg = -1.0;
      double translation_direction_error_deg = -1.0;
      const auto neighbor_prior_it = image_to_prior.find(neighbor_image->ImageId());
      if (target_has_pose && neighbor_has_pose &&
          target_prior_it != image_to_prior.end() &&
          neighbor_prior_it != image_to_prior.end()) {
        const Rigid3d current_neighbor_from_target =
            neighbor_image->CamFromWorld() * Inverse(target_image->CamFromWorld());
        const Rigid3d prior_neighbor_from_target =
            neighbor_prior_it->second->cam_from_world *
            Inverse(target_prior_it->second->cam_from_world);
        rotation_error_deg = RotationErrorDeg(
            current_neighbor_from_target.rotation(),
            prior_neighbor_from_target.rotation());
        translation_direction_error_deg = TranslationDirectionErrorDeg(
            current_neighbor_from_target.translation(),
            prior_neighbor_from_target.translation());
      }

      LOG(INFO) << "    neighbor=" << neighbor_name
                << " image_id=" << neighbor_image->ImageId()
                << " has_pose=" << neighbor_has_pose
                << " shared_tracks="
                << shared_track_counts[neighbor_image->ImageId()]
                << " shared_track_observations="
                << shared_observation_counts[neighbor_image->ImageId()]
                << " neighbor_points3D=" << neighbor_image->NumPoints3D()
                << " rot_err_deg=" << rotation_error_deg
                << " trans_dir_err_deg=" << translation_direction_error_deg;
    }
  }
}

}  // namespace

SixDofPriorGlobalMapper::SixDofPriorGlobalMapper(
    std::shared_ptr<const DatabaseCache> database_cache,
    std::vector<SixDofPosePrior> six_dof_pose_priors,
    std::shared_ptr<const LidarPointCloud> lidar_cloud,
    LidarMatchingOptions lidar_matching_options,
    LidarBundleAdjustmentOptions lidar_ba_options,
    double lidar_phase1_weight,
    double lidar_phase2_weight,
    bool fix_poses_in_lidar_ba)
    : PriorGlobalMapper(std::move(database_cache)),
      six_dof_pose_priors_(std::move(six_dof_pose_priors)),
      lidar_cloud_(std::move(lidar_cloud)),
      lidar_matching_options_(lidar_matching_options),
      lidar_ba_options_(lidar_ba_options),
      lidar_phase1_weight_(lidar_phase1_weight),
      lidar_phase2_weight_(lidar_phase2_weight),
      fix_poses_in_lidar_ba_(fix_poses_in_lidar_ba) {
  for (const auto& pose_prior : six_dof_pose_priors_) {
    if (pose_prior.corr_data_id.sensor_id.type != SensorType::CAMERA) {
      continue;
    }
    AbsolutePosePriorConstraint absolute_pose_prior;
    absolute_pose_prior.image_id = static_cast<image_t>(pose_prior.corr_data_id.id);
    absolute_pose_prior.cam_from_world = pose_prior.cam_from_world;
    absolute_pose_prior.rotation_covariance = pose_prior.rotation_covariance;
    absolute_pose_prior.position_covariance = pose_prior.position_covariance;
    absolute_pose_priors_.push_back(absolute_pose_prior);
    image_to_six_dof_prior_[static_cast<image_t>(pose_prior.corr_data_id.id)] =
        &pose_prior;
  }
}

bool SixDofPriorGlobalMapper::ShouldUseSixDofPosePriors(
    const PriorGlobalMapperOptions& options) const {
  if (!options.use_6dof_pose_priors) {
    return false;
  }
  if (six_dof_pose_priors_.empty()) {
    LOG(WARNING) << "use_6dof_pose_priors=true but no 6DoF priors were loaded. "
                    "Falling back to standard prior mapper.";
    return false;
  }
  return true;
}

bool SixDofPriorGlobalMapper::ShouldUseLidarPointToPlaneInRetriangulation(
    const PriorGlobalMapperOptions& options) const {
  return options.use_lidar_point_to_plane_in_retriangulation &&
         lidar_cloud_ != nullptr && !lidar_cloud_->Empty();
}

bool SixDofPriorGlobalMapper::RunLidarPointToPlaneRetriangulationAlignment(
    const BundleAdjustmentOptions& ba_options,
    const PriorGlobalMapperOptions& mapper_options,
    const PosePriorBundleAdjustmentOptions& prior_options,
    double prior_rotation_fallback_stddev_rad,
    const std::unordered_set<point3D_t>& recent_point3D_ids,
    const char* stage_name) {
  if (!ShouldUseLidarPointToPlaneInRetriangulation(mapper_options)) {
    return true;
  }

  const bool is_final_stage = IsFinalRetriangulationStage(stage_name);
  const int lidar_phase = is_final_stage ? 2 : 1;

  auto& recon = GetReconstruction();
  if (recon.NumImages() == 0 || recon.NumPoints3D() == 0) {
    LOG(INFO) << "6DoF retriangulation LiDAR alignment skipped at "
              << stage_name << ": reconstruction has no usable images or "
              << "points.";
    return true;
  }

  LidarMatchingOptions matching_options = lidar_matching_options_;
  matching_options.max_reprojection_error =
      mapper_options.lidar_retriangulation_max_reprojection_error;

  LidarMatcher matcher(*lidar_cloud_, matching_options);
  std::vector<LidarConstraint> constraints =
      matcher.BuildConstraints(recon, lidar_phase);

    if (!is_final_stage) {
    const std::unordered_set<point3D_t> stable_point3D_ids =
      CollectStableEarlyLidarAnchorPointIds(
        recon,
        mapper_options.lidar_retriangulation_early_min_track_length,
        mapper_options
          .lidar_retriangulation_early_max_mean_reprojection_error);
    const size_t removed =
      FilterLidarConstraintsToEligiblePointIds(&constraints, stable_point3D_ids);
    LOG(INFO) << "6DoF retriangulation LiDAR early gate at " << stage_name
          << ": kept " << constraints.size() << " stable constraints, removed "
          << removed << " unstable constraints (recent_modified_points="
          << recent_point3D_ids.size() << ").";
    }

  if (constraints.empty()) {
    LOG(INFO) << "6DoF retriangulation LiDAR alignment built no constraints at "
              << stage_name << ".";
    return true;
  }

  BundleAdjustmentConfig ba_config;
  for (const auto& [img_id, img] : recon.Images()) {
    if (!ImageHasFinitePose(img)) {
      continue;
    }
    ba_config.AddImage(img_id);
    if (fix_poses_in_lidar_ba_ && img.HasFrameId()) {
      ba_config.SetConstantRigFromWorldPose(img.FrameId());
    }
  }

  BundleAdjustmentOptions effective_ba_options = ba_options;
  if (fix_poses_in_lidar_ba_) {
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
    effective_ba_options.ceres->solver_options.max_num_iterations = 30;
  }

  LidarBundleAdjustmentOptions lidar_options = lidar_ba_options_;
  lidar_options.weight =
      (lidar_phase == 1) ? lidar_phase1_weight_ : lidar_phase2_weight_;

  std::shared_ptr<BundleAdjustmentSummary> summary;
  if (fix_poses_in_lidar_ba_) {
    auto ba = CreateLidarBundleAdjuster(effective_ba_options,
                                        lidar_options,
                                        ba_config,
                                        constraints,
                                        recon);
    summary = ba->Solve();
  } else {
    std::vector<AbsolutePosePriorConstraint> absolute_pose_priors =
        absolute_pose_priors_;
    size_t num_medium_risk_priors = 0;
    size_t num_high_risk_priors = 0;
    for (auto& absolute_pose_prior : absolute_pose_priors) {
      if (!recon.ExistsImage(absolute_pose_prior.image_id)) {
        continue;
      }
      const Image& image = recon.Image(absolute_pose_prior.image_id);
      const auto prior_it = image_to_six_dof_prior_.find(absolute_pose_prior.image_id);
        const SixDofPosePrior* pose_prior =
          prior_it != image_to_six_dof_prior_.end() ? prior_it->second : nullptr;
      const double weight_multiplier = RiskStratifiedPriorWeightMultiplier(
          mapper_options, image, pose_prior);
      if (weight_multiplier > 1.0) {
        StrengthenAbsolutePosePrior(absolute_pose_prior,
                                    prior_options,
                                    prior_rotation_fallback_stddev_rad,
                                    weight_multiplier);
        if (weight_multiplier >=
            mapper_options.final_joint_ba_high_risk_prior_weight_multiplier) {
          ++num_high_risk_priors;
        } else {
          ++num_medium_risk_priors;
        }
      }
    }

    auto ba = CreateAbsolutePosePriorBundleAdjuster(effective_ba_options,
                                                    prior_options,
                                                    prior_rotation_fallback_stddev_rad,
                                                    ba_config,
                                                    std::move(absolute_pose_priors),
                                                    recon);
    auto* ceres_ba = dynamic_cast<CeresBundleAdjuster*>(ba.get());
    THROW_CHECK(ceres_ba != nullptr)
        << "Combined 6DoF + LiDAR retriangulation alignment requires Ceres.";

    std::shared_ptr<ceres::Problem>& problem = ceres_ba->Problem();

    std::unique_ptr<ceres::LossFunction> raw_loss;
    switch (lidar_options.loss_function_type) {
      case CeresBundleAdjustmentOptions::LossFunctionType::TRIVIAL:
        break;
      case CeresBundleAdjustmentOptions::LossFunctionType::SOFT_L1:
        raw_loss = std::make_unique<ceres::SoftLOneLoss>(
            lidar_options.loss_scale);
        break;
      case CeresBundleAdjustmentOptions::LossFunctionType::CAUCHY:
        raw_loss =
            std::make_unique<ceres::CauchyLoss>(lidar_options.loss_scale);
        break;
      case CeresBundleAdjustmentOptions::LossFunctionType::HUBER:
        raw_loss = std::make_unique<ceres::HuberLoss>(lidar_options.loss_scale);
        break;
    }

    std::unique_ptr<ceres::LossFunction> scaled_loss(
        new ceres::ScaledLoss(raw_loss.release(),
                              lidar_options.weight,
                              ceres::TAKE_OWNERSHIP));

    int added = 0;
    for (const auto& constraint : constraints) {
      if (!recon.ExistsPoint3D(constraint.point3D_id)) {
        continue;
      }

      Point3D& point3D = recon.Point3D(constraint.point3D_id);
      double* xyz = point3D.xyz.data();
      if (!problem->HasParameterBlock(xyz) ||
          problem->IsParameterBlockConstant(xyz)) {
        continue;
      }

      ceres::CostFunction* cost = nullptr;
      if (constraint.use_plane && lidar_options.use_point_to_plane) {
        cost = PointToPlaneCostFunctor::Create(constraint.xyz_lidar,
                                               constraint.normal);
      } else {
        cost = PointToPointCostFunctor::Create(constraint.xyz_lidar);
      }

      problem->AddResidualBlock(cost, scaled_loss.get(), xyz);
      ++added;
    }

    LOG(INFO) << "[6DoF+Lidar final BA] Added " << added
              << " LiDAR residuals with active 6DoF pose priors.";
    if (mapper_options.use_risk_stratified_prior_weights_in_final_joint_ba) {
      LOG(INFO) << "[6DoF+Lidar final BA] Strengthened prior weights for "
                << num_medium_risk_priors << " medium-risk and "
                << num_high_risk_priors << " high-risk frame(s).";
    }
    summary = ba->Solve();
  }
  if (!summary->IsSolutionUsable()) {
    LOG(ERROR) << "6DoF retriangulation LiDAR alignment failed at "
               << stage_name << ".";
    return false;
  }

  LOG(INFO) << "6DoF retriangulation LiDAR alignment finished at "
            << stage_name << " using phase " << lidar_phase << " with "
            << constraints.size()
            << " constraints and reprojection gate "
            << matching_options.max_reprojection_error << " px.";
  return true;
}

bool SixDofPriorGlobalMapper::ApplyEarlyRetriangulationObservationCleanup(
    ObservationManager& obs_manager,
    const std::unordered_set<point3D_t>& point3D_ids,
    double max_reprojection_error_px,
    const char* stage_name) {
  if (max_reprojection_error_px <= 0.0 || point3D_ids.empty()) {
    return true;
  }

  const size_t num_filtered =
      obs_manager.FilterObservationsWithLargeReprojectionError(
          max_reprojection_error_px,
          point3D_ids,
          ReprojectionErrorType::PIXEL);
  LOG(INFO) << "6DoF retriangulation early observation cleanup at "
            << stage_name << " removed " << num_filtered
            << " observation(s) from " << point3D_ids.size()
            << " recently modified point(s) with threshold "
            << max_reprojection_error_px << " px.";
  return true;
}

size_t SixDofPriorGlobalMapper::FilterPoseGraphEdgesWithSixDofPriors(
    double max_rotation_error_deg,
    double max_translation_direction_error_deg) {
  if (max_rotation_error_deg <= 0.0 &&
      max_translation_direction_error_deg <= 0.0) {
    return 0;
  }

  size_t num_filtered = 0;
  for (auto& [pair_id, edge] : GetPoseGraph().Edges()) {
    if (!edge.valid) {
      continue;
    }

    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const auto prior_it1 = image_to_six_dof_prior_.find(image_id1);
    const auto prior_it2 = image_to_six_dof_prior_.find(image_id2);
    if (prior_it1 == image_to_six_dof_prior_.end() ||
        prior_it2 == image_to_six_dof_prior_.end()) {
      continue;
    }

    const Rigid3d prior_cam2_from_cam1 =
        prior_it2->second->cam_from_world *
        Inverse(prior_it1->second->cam_from_world);
    if (max_rotation_error_deg > 0.0 &&
        RotationErrorDeg(edge.cam2_from_cam1.rotation(),
                         prior_cam2_from_cam1.rotation()) >
            max_rotation_error_deg) {
      edge.valid = false;
      ++num_filtered;
      continue;
    }

    if (max_translation_direction_error_deg > 0.0 &&
        TranslationDirectionErrorDeg(edge.cam2_from_cam1.translation(),
                                     prior_cam2_from_cam1.translation()) >
            max_translation_direction_error_deg) {
      edge.valid = false;
      ++num_filtered;
    }
  }

  LOG(INFO) << "Filtered " << num_filtered
            << " pose-graph edges using 6DoF prior consistency checks.";
  return num_filtered;
}

size_t SixDofPriorGlobalMapper::SeedRotationsFromSixDofPriors(
    bool reset_translations) {
  auto& reconstruction = GetReconstruction();
  const double nan_value = std::numeric_limits<double>::quiet_NaN();
  const Eigen::Vector3d nan_translation =
      Eigen::Vector3d::Constant(nan_value);

  size_t seeded_frames = 0;
  std::unordered_set<frame_t> processed_frame_ids;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (!image.HasFramePtr()) {
      continue;
    }

    const frame_t frame_id = image.FrameId();
    if (!processed_frame_ids.insert(frame_id).second) {
      continue;
    }

    Frame& frame = reconstruction.Frame(frame_id);
    if (!FrameHasAllCameraImagesLoaded(reconstruction, frame)) {
      continue;
    }

    const Image* selected_image = nullptr;
    const SixDofPosePrior* selected_prior = nullptr;
    for (const auto& data_id : frame.ImageIds()) {
      const image_t frame_image_id = static_cast<image_t>(data_id.id);
      if (!reconstruction.ExistsImage(frame_image_id)) {
        continue;
      }
      const Image& frame_image = reconstruction.Image(frame_image_id);
      const auto prior_it = image_to_six_dof_prior_.find(frame_image_id);
      if (prior_it == image_to_six_dof_prior_.end()) {
        continue;
      }
      if (selected_image == nullptr || frame_image.IsRefInFrame()) {
        selected_image = &frame_image;
        selected_prior = prior_it->second;
        if (frame_image.IsRefInFrame()) {
          break;
        }
      }
    }

    if (selected_image == nullptr || selected_prior == nullptr) {
      continue;
    }

    Rigid3d seeded_pose = selected_prior->cam_from_world;
    if (reset_translations) {
      seeded_pose.translation() = nan_translation;
    }
    frame.SetCamFromWorld(selected_image->CameraId(), seeded_pose);
    reconstruction.RegisterFrame(frame_id);
    ++seeded_frames;
  }

  LOG(INFO) << "Seeded rotations from 6DoF priors for " << seeded_frames
            << " frames.";
  return seeded_frames;
}

size_t SixDofPriorGlobalMapper::EnforceMaxRotationPriorDeviation(
    double max_rotation_deviation_deg, const char* stage_name) {
  if (max_rotation_deviation_deg <= 0.0) {
    return 0;
  }

  auto& reconstruction = GetReconstruction();
  size_t num_clamped = 0;
  for (const auto& [image_id, pose_prior] : image_to_six_dof_prior_) {
    if (!reconstruction.ExistsImage(image_id)) {
      continue;
    }
    Image& image = reconstruction.Image(image_id);
    if (!ImageHasFinitePose(image)) {
      continue;
    }

    const Rigid3d current_pose = image.CamFromWorld();
    if (RotationErrorDeg(current_pose.rotation(),
                         pose_prior->cam_from_world.rotation()) <=
        max_rotation_deviation_deg) {
      continue;
    }

    Rigid3d clamped_pose = current_pose;
    clamped_pose.rotation() = pose_prior->cam_from_world.rotation();
    image.FramePtr()->SetCamFromWorld(image.CameraId(), clamped_pose);
    ++num_clamped;
  }

  LOG(INFO) << "Clamped " << num_clamped
            << " image rotations back to their 6DoF priors " << stage_name
            << ".";
  return num_clamped;
}

void SixDofPriorGlobalMapper::RestorePosesFromSixDofPriors(
    const char* stage_name) {
  auto& reconstruction = GetReconstruction();
  size_t restored = 0;
  size_t skipped_incomplete_frames = 0;
  for (const auto& [image_id, pose_prior] : image_to_six_dof_prior_) {
    if (!reconstruction.ExistsImage(image_id)) {
      continue;
    }
    Image& image = reconstruction.Image(image_id);
    if (!image.HasFramePtr()) {
      continue;
    }
    if (!FrameHasAllCameraImagesLoaded(reconstruction, *image.FramePtr())) {
      ++skipped_incomplete_frames;
      continue;
    }
    image.FramePtr()->SetCamFromWorld(image.CameraId(), pose_prior->cam_from_world);
    reconstruction.RegisterFrame(image.FrameId());
    ++restored;
  }
  LOG(INFO) << "Restored full 6DoF poses for " << restored << " images "
            << stage_name << ".";
  if (skipped_incomplete_frames > 0) {
    LOG(INFO) << "Skipped " << skipped_incomplete_frames
              << " 6DoF priors because their frames reference unloaded "
                 "camera images.";
  }
}

size_t SixDofPriorGlobalMapper::DeRegisterFramesWithTooFewObservations(
    int min_observations_per_registered_image, const char* stage_name) {
  if (min_observations_per_registered_image <= 0) {
    return 0;
  }

  auto& reconstruction = GetReconstruction();
  std::vector<frame_t> frame_ids_to_deregister;
  size_t affected_images = 0;

  for (const frame_t frame_id : reconstruction.RegFrameIds()) {
    const Frame& frame = reconstruction.Frame(frame_id);
    bool should_deregister = false;
    for (const data_t& data_id : frame.ImageIds()) {
      if (!reconstruction.ExistsImage(static_cast<image_t>(data_id.id))) {
        continue;
      }
      const Image& image = reconstruction.Image(static_cast<image_t>(data_id.id));
      if (CountImagePoint3DObservations(image) <
          static_cast<size_t>(min_observations_per_registered_image)) {
        should_deregister = true;
        ++affected_images;
      }
    }
    if (should_deregister) {
      frame_ids_to_deregister.push_back(frame_id);
    }
  }

  if (frame_ids_to_deregister.empty()) {
    return 0;
  }

  ObservationManager obs_manager(reconstruction);
  for (const frame_t frame_id : frame_ids_to_deregister) {
    obs_manager.DeRegisterFrame(frame_id);
  }

  LOG(WARNING) << "De-registered " << frame_ids_to_deregister.size()
               << " frame(s) covering " << affected_images
               << " image(s) with fewer than "
               << min_observations_per_registered_image
               << " observation(s) " << stage_name << ".";
  return frame_ids_to_deregister.size();
}

bool SixDofPriorGlobalMapper::ValidateReconstructionQuality(
    double min_mean_reprojection_error,
    int min_observations_per_registered_image,
    const char* stage_name) {
  auto& reconstruction = GetReconstruction();
  DeRegisterFramesWithTooFewObservations(min_observations_per_registered_image,
                                         stage_name);

  if (reconstruction.NumRegImages() == 0) {
    LOG(ERROR) << "No registered images remain " << stage_name << ".";
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "No 3D points remain " << stage_name << ".";
    return false;
  }

  const double mean_reprojection_error =
      reconstruction.ComputeMeanReprojectionError();
  if (!std::isfinite(mean_reprojection_error) ||
      (min_mean_reprojection_error > 0.0 &&
       mean_reprojection_error < min_mean_reprojection_error)) {
    LOG(ERROR) << "Suspicious mean reprojection error "
               << mean_reprojection_error << " px " << stage_name
               << ". Rejecting reconstruction.";
    return false;
  }

  if (min_observations_per_registered_image > 0) {
    std::vector<image_t> invalid_image_ids;
    invalid_image_ids.reserve(reconstruction.NumRegImages());
    // Collect ALL invalid images so the count is accurate; show only first 10.
    for (const image_t image_id : reconstruction.RegImageIds()) {
      const Image& image = reconstruction.Image(image_id);
      if (CountImagePoint3DObservations(image) <
          static_cast<size_t>(min_observations_per_registered_image)) {
        invalid_image_ids.push_back(image_id);
      }
    }
    if (!invalid_image_ids.empty()) {
      const size_t show_limit = std::min(invalid_image_ids.size(), size_t{10});
      std::vector<image_t> shown_ids(invalid_image_ids.begin(),
                                     invalid_image_ids.begin() + show_limit);
      LOG(ERROR) << invalid_image_ids.size()
                 << " registered image(s) have fewer than "
                 << min_observations_per_registered_image
                 << " observation(s) " << stage_name
                 << ". First " << show_limit
                 << ": " << JoinImageIds(shown_ids);
      return false;
    }
  }

  LOG(INFO) << "Validated reconstruction quality " << stage_name
            << ": mean reprojection error=" << mean_reprojection_error
            << " px, mean observations per registered image="
            << reconstruction.ComputeMeanObservationsPerRegImage();
  return true;
}

bool SixDofPriorGlobalMapper::RunPostEnforcementCleanup(
    const PriorGlobalMapperOptions& mapper_options,
    const PosePriorBundleAdjustmentOptions& prior_options,
    bool use_prior,
    bool use_6dof,
    double prior_rotation_fallback_stddev_rad) {
  auto& recon = GetReconstruction();
  if (recon.NumRegImages() == 0) {
    LOG(ERROR) << "Cannot run post-enforcement cleanup: no registered images";
    return false;
  }
  if (recon.NumPoints3D() == 0) {
    LOG(ERROR) << "Cannot run post-enforcement cleanup: no 3D points";
    return false;
  }

  LOG(INFO) << "Running post-enforcement cleanup BA on the remaining "
               "reconstruction.";

  ObservationManager obs_manager(recon);
  const size_t num_filtered_large_reproj_pre =
      obs_manager.FilterPoints3DWithLargeReprojectionError(
          mapper_options.max_normalized_reproj_error,
          recon.Point3DIds(),
          ReprojectionErrorType::NORMALIZED);
  const size_t num_filtered_small_tri_pre =
      obs_manager.FilterPoints3DWithSmallTriangulationAngle(
          mapper_options.min_tri_angle_deg, recon.Point3DIds());

  bool ok = false;
  if (use_6dof) {
    ok = RunSixDofBundleAdjustment(mapper_options.bundle_adjustment,
                                   prior_options,
                                   prior_rotation_fallback_stddev_rad);
  } else if (use_prior) {
    ok = RunPosePriorBundleAdjustment(mapper_options.bundle_adjustment,
                                      prior_options);
  } else {
    ok = RunVisualBundleAdjustment(mapper_options.bundle_adjustment, recon);
  }

  if (!ok) {
    return false;
  }

  ObservationManager obs_manager_final(recon);
  const size_t num_filtered_large_reproj_post =
      obs_manager_final.FilterPoints3DWithLargeReprojectionError(
          mapper_options.max_normalized_reproj_error,
          recon.Point3DIds(),
          ReprojectionErrorType::NORMALIZED);
  const size_t num_filtered_small_tri_post =
      obs_manager_final.FilterPoints3DWithSmallTriangulationAngle(
          mapper_options.min_tri_angle_deg, recon.Point3DIds());
  recon.UpdatePoint3DErrors();

  LOG(INFO) << "Post-enforcement cleanup finished: filtered "
            << (num_filtered_large_reproj_pre + num_filtered_small_tri_pre)
            << " point(s) before BA and "
            << (num_filtered_large_reproj_post + num_filtered_small_tri_post)
            << " point(s) after BA.";
  return true;
}

bool SixDofPriorGlobalMapper::ApplyPostEnforcementObservationResidualFilter(
    double max_reprojection_error_px) {
  auto& recon = GetReconstruction();
  if (recon.NumPoints3D() == 0) {
    LOG(ERROR) << "Cannot run post-enforcement observation residual filter: "
                  "no 3D points";
    return false;
  }

  LOG(INFO) << "Running post-enforcement observation residual filter with "
               "threshold "
            << max_reprojection_error_px << " px.";

  ObservationManager obs_manager(recon);
  const size_t num_filtered_observations =
      obs_manager.FilterObservationsWithLargeReprojectionError(
          max_reprojection_error_px,
          recon.Point3DIds(),
          ReprojectionErrorType::PIXEL);
  recon.UpdatePoint3DErrors();

  LOG(INFO) << "Post-enforcement observation residual filter removed "
            << num_filtered_observations << " observation(s).";
  return true;
}

bool SixDofPriorGlobalMapper::RunSixDofBundleAdjustment(
    const BundleAdjustmentOptions& ba_options,
    const PosePriorBundleAdjustmentOptions& prior_options,
    double prior_rotation_fallback_stddev_rad) {
  auto& recon = GetReconstruction();
  if (recon.NumImages() == 0) {
    LOG(ERROR) << "Cannot run 6DoF bundle adjustment: no images";
    return false;
  }
  if (recon.NumPoints3D() == 0) {
    LOG(ERROR) << "Cannot run 6DoF bundle adjustment: no 3D points";
    return false;
  }

  BundleAdjustmentConfig ba_config;
  for (const auto& [img_id, img] : recon.Images()) {
    if (ImageHasFinitePose(img)) {
      ba_config.AddImage(img_id);
    }
  }

  auto ba = CreateAbsolutePosePriorBundleAdjuster(ba_options,
                                                  prior_options,
                                                  prior_rotation_fallback_stddev_rad,
                                                  ba_config,
                                                  absolute_pose_priors_,
                                                  recon);
  auto summary = ba->Solve();
  if (!summary->IsSolutionUsable()) {
    LOG(ERROR) << "6DoF bundle adjustment failed (Ceres termination=FAILURE; "
                  "possible cause: zero residuals, Iterations=-2, or "
                  "degenerate configuration). Halting.";
    return false;
  }
  return true;
}

bool SixDofPriorGlobalMapper::IterativeSixDofBundleAdjustment(
    const BundleAdjustmentOptions& options,
    const PosePriorBundleAdjustmentOptions& prior_options,
    double prior_rotation_fallback_stddev_rad,
    double max_normalized_reproj_error,
    double min_tri_angle_deg,
    int num_iterations,
    bool skip_fixed_rotation_stage,
    bool skip_joint_optimization_stage) {
  auto& recon = GetReconstruction();
  for (int ite = 0; ite < num_iterations; ite++) {
    if (!skip_fixed_rotation_stage) {
      BundleAdjustmentOptions opts_position_only = options;
      opts_position_only.constant_rig_from_world_rotation = true;
      if (!RunSixDofBundleAdjustment(
              opts_position_only,
              prior_options,
              prior_rotation_fallback_stddev_rad)) {
        return false;
      }
      LOG(INFO) << "6DoF prior BA iteration " << ite + 1 << " / "
                << num_iterations << ", fixed-rotation stage finished";
    }

    if (!skip_joint_optimization_stage) {
      if (!RunSixDofBundleAdjustment(
              options, prior_options, prior_rotation_fallback_stddev_rad)) {
        return false;
      }
    }
    LOG(INFO) << "6DoF prior BA iteration " << ite + 1 << " / "
              << num_iterations << " finished";

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

  ObservationManager obs_manager_final(recon);
  obs_manager_final.FilterPoints3DWithLargeReprojectionError(
      max_normalized_reproj_error,
      recon.Point3DIds(),
      ReprojectionErrorType::NORMALIZED);
  obs_manager_final.FilterPoints3DWithSmallTriangulationAngle(
      min_tri_angle_deg, recon.Point3DIds());
  return true;
}

bool SixDofPriorGlobalMapper::IterativeRetriangulateAndRefineWithSixDofPriors(
    const IncrementalTriangulator::Options& tri_options,
    const BundleAdjustmentOptions& ba_options,
    const PriorGlobalMapperOptions& mapper_options,
    const PosePriorBundleAdjustmentOptions& prior_options,
    double prior_rotation_fallback_stddev_rad,
    double max_normalized_reproj_error,
    double min_tri_angle_deg) {
  auto& recon = GetReconstruction();

  LOG(INFO) << "6DoF retriangulation: clearing existing 3D structure.";

  recon.DeleteAllPoints2DAndPoints3D();

  IncrementalMapper mapper(database_cache_);
  mapper.BeginReconstruction(GlobalMapper::Reconstruction());

  LOG(INFO) << "6DoF retriangulation: triangulating "
            << recon.NumRegImages() << " registered images.";
  for (const auto img_id : recon.RegImageIds()) {
    mapper.TriangulateImage(tri_options, img_id);
  }

  BundleAdjustmentOptions custom_ba_options;
  custom_ba_options.print_summary = false;
  if (custom_ba_options.ceres && ba_options.ceres) {
    custom_ba_options.ceres->solver_options.num_threads =
        ba_options.ceres->solver_options.num_threads;
    custom_ba_options.ceres->solver_options.max_num_iterations =
      std::max(1,
           mapper_options
             .retriangulation_refinement_ba_max_num_iterations);
    custom_ba_options.ceres->solver_options.max_linear_solver_iterations =
      std::max(1,
           mapper_options
             .retriangulation_refinement_ba_max_linear_solver_iterations);
  }

  IncrementalMapper::Options inc_options;
  inc_options.random_seed = tri_options.random_seed;
  inc_options.use_prior_position = mapper_options.use_prior_position;
  inc_options.use_robust_loss_on_prior_position =
      mapper_options.use_robust_loss_on_prior_position;
  inc_options.prior_position_loss_scale =
      mapper_options.prior_position_loss_scale;

  const int kMaxNumRefinements =
      std::max(1, mapper_options.retriangulation_refinement_max_refinements);
  constexpr double kMaxRefinementChange = 0.0005;

  const size_t num_completed_observations =
      mapper.CompleteAndMergeTracks(tri_options);
  const size_t num_retriangulated_observations = mapper.Retriangulate(tri_options);
  LOG(INFO) << "6DoF retriangulation: completed/merged "
            << num_completed_observations << " observations and retriangulated "
            << num_retriangulated_observations << " observations before BA.";

  for (int refinement_idx = 0; refinement_idx < kMaxNumRefinements;
       ++refinement_idx) {
    const size_t num_observations = recon.ComputeNumObservations();
    const std::unordered_set<point3D_t> recent_point3D_ids =
      mapper.GetModifiedPoints3D();

    LOG(INFO) << "6DoF retriangulation: refinement iteration "
              << refinement_idx + 1 << " / " << kMaxNumRefinements
              << " running 6DoF BA.";
    if (!RunSixDofBundleAdjustment(custom_ba_options,
                                   prior_options,
                                   prior_rotation_fallback_stddev_rad)) {
      LOG(ERROR) << "6DoF retriangulation: 6DoF BA failed in refinement "
                 << "iteration " << refinement_idx + 1 << ".";
      return false;
    }
    if (!mapper_options.use_lidar_point_to_plane_only_in_final_retriangulation) {
      if (!RunLidarPointToPlaneRetriangulationAlignment(
              custom_ba_options,
              mapper_options,
              prior_options,
              prior_rotation_fallback_stddev_rad,
              recent_point3D_ids,
              StringPrintf("refinement_%d", refinement_idx + 1).c_str())) {
        return false;
      }
      if (!ApplyEarlyRetriangulationObservationCleanup(
            mapper.ObservationManager(),
              recent_point3D_ids,
              mapper_options
                  .lidar_retriangulation_early_post_alignment_max_reprojection_error_px,
              StringPrintf("refinement_%d", refinement_idx + 1).c_str())) {
        return false;
      }
    }

    size_t num_changed_observations = mapper.CompleteAndMergeTracks(tri_options);
    num_changed_observations += mapper.FilterPoints(inc_options);
    const double changed =
        num_observations == 0
            ? 0.0
            : static_cast<double>(num_changed_observations) / num_observations;

    LOG(INFO) << "6DoF retriangulation: refinement iteration "
              << refinement_idx + 1 << " changed_observations_ratio="
              << changed;
    if (changed < kMaxRefinementChange) {
      break;
    }
  }

  mapper.ClearModifiedPoints3D();

  mapper.EndReconstruction(/*discard=*/false);

  LOG(INFO) << "6DoF retriangulation: filtering points before final 6DoF BA.";

  {
    ObservationManager obs_manager(recon);
    obs_manager.FilterPoints3DWithLargeReprojectionError(
        max_normalized_reproj_error,
        recon.Point3DIds(),
        ReprojectionErrorType::NORMALIZED);
  }

        LOG(INFO) << "6DoF retriangulation: running final 6DoF BA.";
  if (!RunSixDofBundleAdjustment(
          ba_options, prior_options, prior_rotation_fallback_stddev_rad)) {
    return false;
  }
  if (!RunLidarPointToPlaneRetriangulationAlignment(
      ba_options,
      mapper_options,
      prior_options,
      prior_rotation_fallback_stddev_rad,
      {},
      "final")) {
    return false;
  }

  ObservationManager obs_manager_final(recon);
  obs_manager_final.FilterPoints3DWithLargeReprojectionError(
      max_normalized_reproj_error,
      recon.Point3DIds(),
      ReprojectionErrorType::NORMALIZED);
  obs_manager_final.FilterPoints3DWithSmallTriangulationAngle(
      min_tri_angle_deg, recon.Point3DIds());
  return true;
}

  bool SixDofPriorGlobalMapper::IterativeRetriangulateAndRefineVisualOnlyWithSnapshots(
    const IncrementalTriangulator::Options& tri_options,
    const BundleAdjustmentOptions& ba_options,
    double max_normalized_reproj_error,
    double min_tri_angle_deg) {
    auto& recon = GetReconstruction();

    LOG(INFO) << "6DoF visual retriangulation debug path: clearing existing 3D structure.";
    recon.DeleteAllPoints2DAndPoints3D();

    IncrementalMapper mapper(database_cache_);
    mapper.BeginReconstruction(GlobalMapper::Reconstruction());

    LOG(INFO) << "6DoF visual retriangulation debug path: triangulating "
        << recon.NumRegImages() << " registered images.";
    for (const auto image_id : recon.RegImageIds()) {
    mapper.TriangulateImage(tri_options, image_id);
    }
    LogRetriangulationDebugSnapshot(
      recon, image_to_six_dof_prior_, "after_triangulate_all_images");
      LogExternalResidualEntries(
        ComputeExternalResidualEntries(recon, *database_cache_->CorrespondenceGraph()),
        "after_triangulate_all_images",
        5);

    BundleAdjustmentOptions custom_ba_options;
    custom_ba_options.print_summary = false;
    if (custom_ba_options.ceres && ba_options.ceres) {
    custom_ba_options.ceres->solver_options.num_threads =
      ba_options.ceres->solver_options.num_threads;
    custom_ba_options.ceres->solver_options.max_num_iterations = 50;
    custom_ba_options.ceres->solver_options.max_linear_solver_iterations = 100;
    }

    IncrementalMapper::Options inc_options;
    inc_options.random_seed = tri_options.random_seed;

    constexpr int kMaxNumRefinements = 5;
    constexpr double kMaxRefinementChange = 0.0005;

      const auto& correspondence_graph = *database_cache_->CorrespondenceGraph();

      LogCompletionDiagnostics(
        ComputeCompletionDiagnostics(recon, correspondence_graph, tri_options),
        "before_initial_complete_tracks");
      const size_t num_completed_observations = mapper.CompleteTracks(tri_options);
    LogRetriangulationDebugSnapshot(
      recon, image_to_six_dof_prior_, "after_initial_complete_merge");
      LogCompletionDiagnostics(
        ComputeCompletionDiagnostics(recon, correspondence_graph, tri_options),
        "after_initial_complete_tracks");
      LogMergeDiagnostics(
        ComputeMergeDiagnostics(recon, correspondence_graph, tri_options),
        "before_initial_merge_tracks");
      const size_t num_merged_observations = mapper.MergeTracks(tri_options);
      LogRetriangulationDebugSnapshot(
        recon, image_to_six_dof_prior_, "after_initial_merge_tracks");
      LogMergeDiagnostics(
        ComputeMergeDiagnostics(recon, correspondence_graph, tri_options),
        "after_initial_merge_tracks");
    const size_t num_retriangulated_observations = mapper.Retriangulate(tri_options);
    LogRetriangulationDebugSnapshot(
      recon, image_to_six_dof_prior_, "after_initial_retriangulate");
      LogExternalResidualEntries(
        ComputeExternalResidualEntries(recon, correspondence_graph),
        "after_initial_retriangulate",
        5);
      LogLeftTargetTriangulationRays(recon, "after_initial_retriangulate");
      const std::vector<TrackedPseudoPoint> tracked_pseudo_points =
        IdentifyLeftPseudoPoints(recon, 0.5, 40.0);
      LogTrackedPseudoPointResiduals(
        recon, tracked_pseudo_points, "after_initial_retriangulate");
    LOG(INFO) << "6DoF visual retriangulation debug path: completed/merged "
          << (num_completed_observations + num_merged_observations)
          << " observations (complete=" << num_completed_observations
          << ", merge=" << num_merged_observations
          << ") and retriangulated "
        << num_retriangulated_observations << " observations before BA.";

    for (int refinement_idx = 0; refinement_idx < kMaxNumRefinements;
       ++refinement_idx) {
    const size_t num_observations = recon.ComputeNumObservations();

    LOG(INFO) << "6DoF visual retriangulation debug path: refinement iteration "
          << refinement_idx + 1 << " / " << kMaxNumRefinements
          << " running visual BA.";
    if (!mapper.AdjustGlobalBundle(inc_options, custom_ba_options)) {
      LOG(ERROR) << "6DoF visual retriangulation debug path: visual BA failed in refinement iteration "
           << refinement_idx + 1 << ".";
      return false;
    }
    recon.Normalize();
    LogRetriangulationDebugSnapshot(
      recon,
      image_to_six_dof_prior_,
      StringPrintf("refinement_%d_after_ba_and_normalize",
             refinement_idx + 1));
    LogExternalResidualEntries(
      ComputeExternalResidualEntries(recon, correspondence_graph),
      StringPrintf("refinement_%d_after_ba_and_normalize", refinement_idx + 1),
      5);
    LogTrackedPseudoPointResiduals(
      recon,
      tracked_pseudo_points,
      StringPrintf("refinement_%d_after_ba_and_normalize", refinement_idx + 1));

    LogCompletionDiagnostics(
      ComputeCompletionDiagnostics(recon, correspondence_graph, tri_options),
      StringPrintf("refinement_%d_before_complete_tracks", refinement_idx + 1));
    const size_t num_completed = mapper.CompleteTracks(tri_options);
    LogRetriangulationDebugSnapshot(
      recon,
      image_to_six_dof_prior_,
      StringPrintf("refinement_%d_after_complete_merge",
             refinement_idx + 1));
    LogCompletionDiagnostics(
      ComputeCompletionDiagnostics(recon, correspondence_graph, tri_options),
      StringPrintf("refinement_%d_after_complete_tracks", refinement_idx + 1));
    LogMergeDiagnostics(
      ComputeMergeDiagnostics(recon, correspondence_graph, tri_options),
      StringPrintf("refinement_%d_before_merge_tracks", refinement_idx + 1));
    const size_t num_merged = mapper.MergeTracks(tri_options);
    LogRetriangulationDebugSnapshot(
      recon,
      image_to_six_dof_prior_,
      StringPrintf("refinement_%d_after_merge_tracks",
             refinement_idx + 1));
    LogMergeDiagnostics(
      ComputeMergeDiagnostics(recon, correspondence_graph, tri_options),
      StringPrintf("refinement_%d_after_merge_tracks", refinement_idx + 1));
    const size_t num_filtered = mapper.FilterPoints(inc_options);
    LogRetriangulationDebugSnapshot(
      recon,
      image_to_six_dof_prior_,
      StringPrintf("refinement_%d_after_filter_points",
             refinement_idx + 1));
    LogExternalResidualEntries(
      ComputeExternalResidualEntries(recon, correspondence_graph),
      StringPrintf("refinement_%d_after_filter_points", refinement_idx + 1),
      5);
    LogTrackedPseudoPointResiduals(
      recon,
      tracked_pseudo_points,
      StringPrintf("refinement_%d_after_filter_points", refinement_idx + 1));
    const size_t num_changed_observations = num_completed + num_merged + num_filtered;
    const double changed =
      num_observations == 0
        ? 0.0
        : static_cast<double>(num_changed_observations) / num_observations;

    LOG(INFO) << "6DoF visual retriangulation debug path: refinement iteration "
          << refinement_idx + 1 << " changed_observations_ratio="
          << changed;
    if (changed < kMaxRefinementChange) {
      break;
    }
    }

    mapper.ClearModifiedPoints3D();
    mapper.EndReconstruction(/*discard=*/false);

    ObservationManager obs_manager(recon);
    obs_manager.FilterPoints3DWithLargeReprojectionError(
      max_normalized_reproj_error,
      recon.Point3DIds(),
      ReprojectionErrorType::NORMALIZED);
    LogRetriangulationDebugSnapshot(
      recon, image_to_six_dof_prior_, "after_final_large_reproj_filter_pre_ba");
      LogExternalResidualEntries(
        ComputeExternalResidualEntries(recon, correspondence_graph),
        "after_final_large_reproj_filter_pre_ba",
        5);

    if (!RunVisualBundleAdjustment(ba_options, recon)) {
    return false;
    }
    recon.Normalize();
    LogRetriangulationDebugSnapshot(
      recon, image_to_six_dof_prior_, "after_final_visual_ba_and_normalize");
      LogExternalResidualEntries(
        ComputeExternalResidualEntries(recon, correspondence_graph),
        "after_final_visual_ba_and_normalize",
        5);
      LogTrackedPseudoPointResiduals(
        recon, tracked_pseudo_points, "after_final_visual_ba_and_normalize");

    obs_manager.FilterPoints3DWithLargeReprojectionError(
      max_normalized_reproj_error,
      recon.Point3DIds(),
      ReprojectionErrorType::NORMALIZED);
    LogRetriangulationDebugSnapshot(
      recon, image_to_six_dof_prior_, "after_final_large_reproj_filter_post_ba");
      LogExternalResidualEntries(
        ComputeExternalResidualEntries(recon, correspondence_graph),
        "after_final_large_reproj_filter_post_ba",
        5);
      LogTrackedPseudoPointResiduals(
        recon, tracked_pseudo_points, "after_final_large_reproj_filter_post_ba");
    obs_manager.FilterPoints3DWithSmallTriangulationAngle(
      min_tri_angle_deg, recon.Point3DIds());
    LogRetriangulationDebugSnapshot(
      recon, image_to_six_dof_prior_, "after_final_small_tri_filter");
      LogTrackedPseudoPointResiduals(
        recon, tracked_pseudo_points, "after_final_small_tri_filter");
    return true;
  }

bool SixDofPriorGlobalMapper::Solve(
    const PriorGlobalMapperOptions& options,
    std::unordered_map<frame_t, int>& cluster_ids) {
  if (GetPoseGraph().Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  PriorGlobalMapperOptions opts = InitializeSixDofOptions(options);
  const bool use_prior = ShouldUsePriorPosition(opts);
  const bool use_6dof = ShouldUseSixDofPosePriors(opts);

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
  }

  const double prior_rotation_fallback_stddev_rad =
      DegToRad(opts.six_dof_prior_rotation_stddev_deg);

  if (use_6dof) {
    LOG(INFO) << "SixDofPriorGlobalMapper: full 6DoF priors ENABLED ("
              << six_dof_pose_priors_.size() << " priors loaded).";
    if (opts.use_6dof_pose_graph_filtering) {
      FilterPoseGraphEdgesWithSixDofPriors(
          opts.max_6dof_pose_graph_rotation_error_deg,
          opts.max_6dof_pose_graph_translation_direction_error_deg);
    }
    // When GPS priors drive global positioning, let GP solve translations from
    // scratch (reset_translations=true).  When only 6DoF priors are available
    // we seed translations from the prior so GlobalPositioning can refine them
    // without starting from NaN (reset_translations=false).
    const bool reset_translations_for_seed =
        (!opts.skip_global_positioning && use_prior);
    SeedRotationsFromSixDofPriors(
        /*reset_translations=*/reset_translations_for_seed);
    opts.rotation_averaging.skip_initialization = true;
  }

  if (!opts.skip_rotation_averaging) {
    LOG_HEADING1("Running rotation averaging");
    Timer timer;
    timer.Start();
    if (!RotationAveraging(opts.rotation_averaging)) {
      return false;
    }
    LOG(INFO) << "Rotation averaging done in " << timer.ElapsedSeconds()
              << " seconds";

    if (use_6dof) {
      EnforceMaxRotationPriorDeviation(
          opts.max_6dof_rotation_prior_deviation_deg,
          "after rotation averaging");
      if (use_prior) {
        RestoreTranslationsFromPriors(database_cache_->PosePriors());
      } else {
        RestorePosesFromSixDofPriors("after rotation averaging");
      }
    } else if (use_prior) {
      RestoreTranslationsFromPriors(database_cache_->PosePriors());
    }
  } else if (use_6dof) {
    RestorePosesFromSixDofPriors("without rotation averaging");
  }

  if (!opts.skip_track_establishment) {
    LOG_HEADING1("Running track establishment");
    Timer timer;
    timer.Start();
    EstablishTracks(opts);
    LOG(INFO) << "Track establishment done in " << timer.ElapsedSeconds()
              << " seconds";
  }

  if (!opts.skip_global_positioning) {
    LOG_HEADING1("Running global positioning");
    Timer timer;
    timer.Start();
    bool ok = true;
    if (use_prior) {
      ok = PriorGlobalPositioning(opts.global_positioning,
                                  opts.max_position_prior_deviation,
                                  opts.clamp_positions_to_prior_after_global_positioning,
                                  opts.max_angular_reproj_error_deg,
                                  opts.max_normalized_reproj_error,
                                  opts.min_tri_angle_deg);
      if (ok && use_6dof) {
        EnforceMaxRotationPriorDeviation(
            opts.max_6dof_rotation_prior_deviation_deg,
            "after global positioning");
      }
    } else if (use_6dof) {
      // Translations have already been seeded from 6DoF priors by
      // SeedRotationsFromSixDofPriors(..., reset_translations=false).
      // Run the standard global positioning solver with those seeds
      // (generate_random_positions=false) so that it refines them without
      // destroying metric scale.
      GlobalPositionerOptions gp_opts_6dof = opts.global_positioning;
      gp_opts_6dof.generate_random_positions = false;
      ok = GlobalPositioning(gp_opts_6dof,
                             opts.max_angular_reproj_error_deg,
                             opts.max_normalized_reproj_error,
                             opts.min_tri_angle_deg);
      if (ok) {
        EnforceMaxRotationPriorDeviation(
            opts.max_6dof_rotation_prior_deviation_deg,
            "after global positioning");
      }
    } else {
      ok = GlobalPositioning(opts.global_positioning,
                             opts.max_angular_reproj_error_deg,
                             opts.max_normalized_reproj_error,
                             opts.min_tri_angle_deg);
    }
    if (!ok) {
      return false;
    }
    LOG(INFO) << "Global positioning done in " << timer.ElapsedSeconds()
              << " seconds";
  }

  if (!opts.skip_bundle_adjustment) {
    LOG_HEADING1("Running iterative bundle adjustment");
    Timer timer;
    timer.Start();
    bool ok;
    if (use_6dof) {
      ok = IterativeSixDofBundleAdjustment(opts.bundle_adjustment,
                                           prior_ba_opts,
                                           prior_rotation_fallback_stddev_rad,
                                           opts.max_normalized_reproj_error,
                                           opts.min_tri_angle_deg,
                                           opts.ba_num_iterations,
                                           opts.ba_skip_fixed_rotation_stage,
                                           opts.ba_skip_joint_optimization_stage);
    } else if (use_prior) {
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
    if (!ok) {
      return false;
    }
    LOG(INFO) << "Iterative bundle adjustment done in "
              << timer.ElapsedSeconds() << " seconds";
  }

  if (!opts.skip_retriangulation) {
    LOG_HEADING1("Running iterative retriangulation and refinement");
    Timer timer;
    timer.Start();
    bool ok;
    if (use_6dof && opts.use_6dof_retriangulation_refinement) {
      LOG(INFO) << "Retriangulation refinement will keep 6DoF pose priors "
                   "enabled.";
      if (ShouldUseLidarPointToPlaneInRetriangulation(opts)) {
        if (opts.use_lidar_point_to_plane_only_in_final_retriangulation) {
          LOG(INFO) << "Retriangulation refinement will keep LiDAR point-to-"
                       "plane structural alignment only for the final 6DoF BA "
                    << "pass, with reprojection gate "
                    << opts.lidar_retriangulation_max_reprojection_error
                    << " px.";
        } else {
          LOG(INFO) << "Retriangulation refinement will also keep LiDAR point-"
                       "to-plane structural alignment enabled with reprojection "
                    << "gate "
                    << opts.lidar_retriangulation_max_reprojection_error
                    << " px.";
        }
      }
      ok = IterativeRetriangulateAndRefineWithSixDofPriors(
          opts.retriangulation,
          opts.bundle_adjustment,
          opts,
          prior_ba_opts,
          prior_rotation_fallback_stddev_rad,
          opts.max_normalized_reproj_error,
          opts.min_tri_angle_deg);
    } else if (use_6dof) {
      LOG(INFO) << "Retriangulation refinement will use the default pure-"
                   "visual path. Set GlobalMapper.use_6dof_"
           "retriangulation_refinement=1 to keep 6DoF priors active "
           "in this stage.";
      if (opts.log_6dof_retriangulation_debug_snapshots) {
        ok = IterativeRetriangulateAndRefineVisualOnlyWithSnapshots(
            opts.retriangulation,
            opts.bundle_adjustment,
            opts.max_normalized_reproj_error,
            opts.min_tri_angle_deg);
      } else {
        ok = IterativeRetriangulateAndRefine(opts.retriangulation,
                                             opts.bundle_adjustment,
                                             opts.max_normalized_reproj_error,
                                             opts.min_tri_angle_deg);
      }
    } else if (use_prior) {
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
    if (!ok) {
      return false;
    }
    LOG(INFO) << "Iterative retriangulation done in "
              << timer.ElapsedSeconds() << " seconds";
  }

  if (use_prior || use_6dof) {
    const size_t num_pose_enforcements = EnforceMaxPositionPriorDeviation(
        database_cache_->PosePriors(),
        opts.max_position_prior_deviation,
        opts.delete_frames_with_position_prior_deviation_after_optimization,
        opts.clamp_positions_to_prior_after_optimization_max_observations,
        opts.clamp_positions_to_prior_after_optimization,
        "after optimization");
    if (num_pose_enforcements > 0) {
      GetReconstruction().UpdatePoint3DErrors();
      if (opts.run_bundle_adjustment_after_pose_enforcement) {
        if (!RunPostEnforcementCleanup(opts,
                                       prior_ba_opts,
                                       use_prior,
                                       use_6dof,
                                       prior_rotation_fallback_stddev_rad)) {
          return false;
        }
      }
      if (opts.post_enforcement_max_reprojection_error_px > 0) {
        if (!ApplyPostEnforcementObservationResidualFilter(
                opts.post_enforcement_max_reprojection_error_px)) {
          return false;
        }
      }
    }
  }

  return ValidateReconstructionQuality(opts.min_mean_reprojection_error,
                                       opts.min_observations_per_registered_image,
                                       "after optimization");
}

}  // namespace colmap
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

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/enum_utils.h"
#include "colmap/util/types.h"

#include <ostream>
#include <limits>

#include <Eigen/Core>

namespace colmap {

struct PosePrior {
 public:
  MAKE_ENUM_CLASS(CoordinateSystem,
                  -1,
                  UNDEFINED,  // = -1
                  WGS84,      // = 0
                  CARTESIAN   // = 1
  );

  // Position prior (meters).
  Eigen::Vector3d position =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());

  // Optional rotation prior in so(3) tangent / rotation-vector form (radians).
  // If any component is non-finite, rotation is considered "not provided".
  Eigen::Vector3d rotation =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());

  // Unified covariance:
  // - 3x3 -> position-only covariance
  // - 6x6 -> joint [px,py,pz, rx,ry,rz] covariance
  // Any other size is invalid.
  Eigen::MatrixXd covariance =
      Eigen::MatrixXd::Constant(3, 3, std::numeric_limits<double>::quiet_NaN());

  CoordinateSystem coordinate_system = CoordinateSystem::UNDEFINED;

  PosePrior() = default;

  // Position-only constructors
  explicit PosePrior(const Eigen::Vector3d& position)
      : position(position) {}

  PosePrior(const Eigen::Vector3d& position,
            const CoordinateSystem system)
      : position(position), coordinate_system(system) {}

  PosePrior(const Eigen::Vector3d& position,
            const Eigen::Matrix3d& pos_cov)
      : position(position), covariance(pos_cov) {}

  PosePrior(const Eigen::Vector3d& position,
            const Eigen::Matrix3d& pos_cov,
            const CoordinateSystem system)
      : position(position), covariance(pos_cov), coordinate_system(system) {}

  // Position + rotation constructors (6x6 covariance)
  PosePrior(const Eigen::Vector3d& position,
            const Eigen::Vector3d& rotation)
      : position(position), rotation(rotation),
        covariance(Eigen::MatrixXd::Constant(
            6, 6, std::numeric_limits<double>::quiet_NaN())) {}

  PosePrior(const Eigen::Vector3d& position,
            const Eigen::Vector3d& rotation,
            const CoordinateSystem system)
      : position(position), rotation(rotation),
        covariance(Eigen::MatrixXd::Constant(
            6, 6, std::numeric_limits<double>::quiet_NaN())),
        coordinate_system(system) {}

  PosePrior(const Eigen::Vector3d& position,
            const Eigen::Vector3d& rotation,
            const Eigen::Matrix<double, 6, 6>& joint_cov)
      : position(position), rotation(rotation), covariance(joint_cov) {}

  PosePrior(const Eigen::Vector3d& position,
            const Eigen::Vector3d& rotation,
            const Eigen::Matrix<double, 6, 6>& joint_cov,
            const CoordinateSystem system)
      : position(position),
        rotation(rotation),
        covariance(joint_cov),
        coordinate_system(system) {}

  inline bool IsValid() const { return position.allFinite(); }

  inline bool HasRotation() const { return rotation.allFinite(); }

  inline bool IsCovarianceValid() const {
    const bool size_ok =
        (covariance.rows() == 3 && covariance.cols() == 3) ||
        (covariance.rows() == 6 && covariance.cols() == 6);
    return size_ok && covariance.allFinite();
  }

  inline bool operator==(const PosePrior& other) const;
  inline bool operator!=(const PosePrior& other) const;
};

std::ostream& operator<<(std::ostream& stream, const PosePrior& prior);

// Equality operators compare coordinate_system, position, rotation, and covariance (including size).
bool PosePrior::operator==(const PosePrior& other) const {
  if (coordinate_system != other.coordinate_system) return false;
  if (position != other.position) return false;
  if (HasRotation() != other.HasRotation()) return false;
  if (HasRotation() && rotation != other.rotation) return false;
  if (covariance.rows() != other.covariance.rows() ||
      covariance.cols() != other.covariance.cols()) {
    return false;
  }
  return covariance == other.covariance;
}

bool PosePrior::operator!=(const PosePrior& other) const { return !(*this == other); }

}  // namespace colmap
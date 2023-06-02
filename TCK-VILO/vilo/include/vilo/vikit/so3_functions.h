// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <sophus/se3.h>

namespace vilo {

using namespace Eigen;
using namespace std;
using namespace Sophus;

Eigen::Matrix3d expSO3(const double x, const double y, const double z);
Eigen::Matrix3d expSO3(const Eigen::Vector3d &w);

Eigen::Vector3d logSO3(const Eigen::Matrix3d &R);

Eigen::Matrix3d inverseRightJacobianSO3(const Eigen::Vector3d &v);
Eigen::Matrix3d rightJacobianSO3(const Eigen::Vector3d &v);
Eigen::Matrix3d rightJacobianSO3(const double x, const double y, const double z);

Eigen::Matrix3d skew(const Eigen::Vector3d &w);
Eigen::Matrix3d inverseRightJacobianSO3(const double x, const double y, const double z);

Eigen::Matrix3d normalizeRotation(const Eigen::Matrix3d &R);

}

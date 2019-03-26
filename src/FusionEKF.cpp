#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  // measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

	H_laser_ << 1, 0, 0, 0,
							0, 1, 0, 0;

	// Anything left to initialize?
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
			ekf_.x_ << measurement_pack.raw_measurements_[0] * cos(measurement_pack.raw_measurements_[1]),
								 measurement_pack.raw_measurements_[0] * sin(measurement_pack.raw_measurements_[1]),
								 measurement_pack.raw_measurements_[2] * cos(measurement_pack.raw_measurements_[1]),
								 measurement_pack.raw_measurements_[2] * sin(measurement_pack.raw_measurements_[1]);
			ekf_.P_ << 1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 10000, 0,
					0, 0, 0, 10000;
		}
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
			ekf_.x_ << measurement_pack.raw_measurements_[0],
								 measurement_pack.raw_measurements_[1],
								 0,
								 0;
			ekf_.P_ << 1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 10000, 0,
					0, 0, 0, 10000;
		} else {
			cout << "Unrecognized sensor type: " + measurement_pack.sensor_type_ << endl;
			return;
		}

    is_initialized_ = true;
		return; // no need to predict or update
	}

  /**
   * Prediction
   */

  /**
   * TODO: Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * TODO: Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  ekf_.Predict();

  /**
   * Update
   */

  /**
   * TODO:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // TODO: Radar updates

  } else {
    // TODO: Laser updates

  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}

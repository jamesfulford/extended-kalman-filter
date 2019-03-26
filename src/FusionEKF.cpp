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

  // measurement covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // TODO(jafulfor): Anything left to initialize?
  // Hj_ = MatrixXd(3, 4);
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

  // compute the time elapsed between the current and previous measurements
  // dt - expressed in seconds
  float noise_ax = 9;
  float noise_ay = 9;

  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt2 = dt * dt;
  float dt3 = dt2 * dt;
  float dt4 = dt3 * dt;

  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1, 0,
             0, 0, 0, 1;

  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt4 * noise_ax / 4, 0, dt3 * noise_ax / 2, 0,
             0, dt4 * noise_ay / 4, 0, dt3 * noise_ay / 2,
             dt3 * noise_ax / 2, 0, dt2 * noise_ax, 0,
             0, dt3 * noise_ay / 2, 0, dt2 * noise_ay;

  ekf_.Predict();

  /**
   * Update
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.R_ = R_radar_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  } else {
    ekf_.R_ = R_laser_;
    ekf_.H_  = tools.CalculateJacobian(ekf_.x_); // TODO(jamesfulford): Use current state or measurement?
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}

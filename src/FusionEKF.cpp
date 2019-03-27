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

  VectorXd x = VectorXd(4);  // current state, not yet known
  MatrixXd P = MatrixXd(4, 4);  // current covariance matrix, not yet known
  MatrixXd F = MatrixXd(4, 4);
  F <<  1, 0, 8, 0, // 8 will be overwritten with dt on each pass
        0, 1, 0, 8,  // 8 will be overwritten with dt on each pass
        0, 0, 1, 0,
        0, 0, 0, 1;
  MatrixXd H = MatrixXd(2, 4);  // To be overwritten on each pass
  MatrixXd R = MatrixXd(2, 2);  // To be overwritten on each pass
  MatrixXd Q = MatrixXd(4, 4);

  ekf_.Init(
      x,
      P,
      F,
      H,
      R,
      Q
  );

  // measurement covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  // Transform measurement matrix for laser
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // Transform measurement matrix for radar (jacobian)
  Hj_ = MatrixXd(3, 4);  // to be filled later, because nonlinear transform... needs parameters
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
    // cout << "Initializing ";
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // cout << "with radar" << endl;
      ekf_.x_ << measurement_pack.raw_measurements_[0] * cos(measurement_pack.raw_measurements_[1]),
                 measurement_pack.raw_measurements_[0] * sin(measurement_pack.raw_measurements_[1]),
                 0,
                 0;
      ekf_.P_ << 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1000, 0,
          0, 0, 0, 1000;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // cout << "with laser" << endl;
      ekf_.x_ << measurement_pack.raw_measurements_[0],
                 measurement_pack.raw_measurements_[1],
                 0,
                 0;
      ekf_.P_ << 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1000, 0,
          0, 0, 0, 1000;
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

  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;
  // cout << "F" << endl << ekf_.F_ << endl;

  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt4 * noise_ax / 4, 0, dt3 * noise_ax / 2, 0,
             0, dt4 * noise_ay / 4, 0, dt3 * noise_ay / 2,
             dt3 * noise_ax / 2, 0, dt2 * noise_ax, 0,
             0, dt3 * noise_ay / 2, 0, dt2 * noise_ay;
  // cout << "Q" << endl << ekf_.Q_ << endl;

  // cout << "x" << endl << ekf_.x_ << endl;
  ekf_.Predict();

  /**
   * Update
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // cout << "Radar update" << endl;
    ekf_.R_ = R_radar_;
    // cout << "Radar update R" << endl << ekf_.R_ << endl;

    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);  // TODO(jamesfulford): Use current state or measurement?
    // cout << "Radar update H" << endl << ekf_.H_ << endl;

    // cout << "measurement_pack.raw_measurements_" << endl << measurement_pack.raw_measurements_ << endl;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    // cout << "Radar update x" << endl << ekf_.x_ << endl;
  } else { // Laser
    // cout << "Laser update" << endl;
    ekf_.R_ = R_laser_;
    // cout << "Laser update R" << endl << ekf_.R_ << endl;

    ekf_.H_  = H_laser_;
    // cout << "Laser update H" << endl << ekf_.H_ << endl;

    // cout << "measurement_pack.raw_measurements_" << endl << measurement_pack.raw_measurements_ << endl;
    ekf_.Update(measurement_pack.raw_measurements_);
    // cout << "Laser update x" << endl << ekf_.x_ << endl;
  }

  // print the output
  cout << "x_ = " << endl << ekf_.x_ << endl;
  cout << "P_ = " << endl << ekf_.P_ << endl;
}

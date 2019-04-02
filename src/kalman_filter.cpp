#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
 * Please note that the Eigen library does not initialize
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // x_ to polar (to compare prediction to reality, z)
  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];
  float px2py2sqrt = pow(pow(px, 2) + pow(py, 2), 0.5);

  float dpdt;
  if (fabs(px2py2sqrt) < 0.0001) { // you are at the origin
    // then your change in magnitude is equal to the magnitude of your velocity vector
    dpdt = pow(pow(vx, 2) + pow(vy, 2), 0.5);
  } else { // otherwise
    // you have to account for location when translating to polar
    // (note that if we used this at origin, we would have a divide by 0 situation)
    dpdt = ((px * vx) + (py * vy)) / px2py2sqrt;
  }

  VectorXd prediction = VectorXd(3);
  prediction << px2py2sqrt,
                atan2(py, px),
                dpdt;

  // Compare prediction to reality
  VectorXd error = z - prediction;
  // Normalize error[1] (angle) to within -π to π
  while (error[1] < -M_PI || error[1] > M_PI) {
    if (error[1] < -M_PI) {
      error[1] = error[1] + (2 * M_PI);
    } else {
      error[1] = error[1] - (2 * M_PI);
    }
  }

  // Update beliefs
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * error);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

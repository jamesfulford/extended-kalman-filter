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
	x_ = F_ * x_;  // call state transition function on current mean state
	P_ = (F_ * P_ * F_.transpose()) + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	VectorXd prediction = H_ * x_;
	VectorXd error = z - prediction;  // linear
	MatrixXd H_transpose = H_.transpose();
	MatrixXd S = H_ * P_ * H_transpose + R_;
	MatrixXd K = P_ * H_transpose * S.inverse();

	x_ = x_ + (K * error);
	long x_size = x_.size();
	P_ = (MatrixXd::Identity(x_size, x_size) - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
	VectorXd prediction = H_ * x_;

	// Convert prediction to polar
	double px = prediction[0];
	double py = prediction[1];
	double vx = prediction[2];
	double vy = prediction[3];
	VectorXd prediction_polar = VectorXd(3);
	double pxpysqrt = sqrt(pow(px, 2) + pow(px, 2));

	prediction_polar << pxpysqrt,
											atan2(py, px), // between -π and π, handles the divide by 0 case
											((px * vx) + (py * vy)) / pxpysqrt;

	VectorXd y = z - prediction_polar;

	MatrixXd H_transpose = H_.transpose();
	MatrixXd S = H_ * P_ * H_transpose + R_;
	MatrixXd K = P_ * H_transpose * S.inverse();

	x_ = x_ + (K * y);
	long x_size = x_.size();
	P_ = (MatrixXd::Identity(x_size, x_size) - K * H_) * P_;
}

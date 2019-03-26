#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  if (estimations.size() != ground_truth.size() || estimations.size() == 0)
  {
    return rmse;
  }

  for (unsigned int i = 0; i < estimations.size(); ++i)
  {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }
  rmse = rmse.array() / estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3, 4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // check division by zero
  if (px == 0 && py == 0)
  {
    return Hj;
  }

  float pxy = pow(px, 2) + pow(py, 2);
  float pxyroot = pow(pxy, 0.5);

  // compute the Jacobian matrix
  Hj << px / pxyroot, py / pxyroot, 0, 0,
      -py / pxy, px / pxy, 0, 0,
      (py * ((vx * py) - (vy * px))) / pow(pxy, 1.5), (px * ((vy * px) - (vx * py))) / pow(pxy, 1.5), px / pxyroot, py / pxyroot;

  return Hj;
}

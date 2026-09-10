#pragma once
#include "opencv2/core.hpp"
namespace cv {
constexpr int INTER_LINEAR = 1;
void GaussianBlur(const Mat&, Mat&, Size, double, double);
}

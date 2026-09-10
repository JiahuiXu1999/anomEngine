#pragma once
// Declaration-only test surface for compiling unchanged image-processing calls.
// No image computation is simulated or claimed by the routing tests.
#include <cstddef>
#include <exception>
namespace cv {
class Exception : public std::exception {};
struct Size { int width = 0, height = 0; Size() = default; Size(int w, int h) : width(w), height(h) {} };
struct Point { int x = 0, y = 0; Point() = default; Point(int a, int b) : x(a), y(b) {} };
struct Rect { int x = 0, y = 0, width = 0, height = 0; };
struct Scalar { explicit Scalar(double, double = 0, double = 0, double = 0); };
struct Mat {
    int rows = 0, cols = 0;
    Mat() = default;
    Mat(int, int, int, void*, size_t);
    Mat(Size, int, Scalar);
    bool empty() const;
    Mat clone() const;
    size_t step1() const;
    template<class T> T* ptr();
    template<class T> const T* ptr() const;
    Mat& operator+=(const Mat&);
    Mat& operator/=(float);
};
void minMaxLoc(const Mat&, double*, double*);
}
#define CV_32FC1 5
#define CV_8UC3 16

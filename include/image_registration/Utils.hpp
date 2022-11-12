/**
 * @file    Utils.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace _cv
{
inline cv::Mat scaleHomography(const cv::Mat& homography, float scale)
{
    if (scale <= 0) {
        throw std::runtime_error("scale must be > 0");
    }
    cv::Mat scaledHomography = homography.clone();
    scaledHomography.ptr<float>(0)[2] *= scale;
    scaledHomography.ptr<float>(1)[2] *= scale;
    scaledHomography.ptr<float>(2)[0] /= scale;
    scaledHomography.ptr<float>(2)[1] /= scale;

    return scaledHomography;
}
}  // namespace _cv

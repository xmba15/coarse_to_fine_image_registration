/**
 * @file    Types.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

namespace _cv
{
struct KeyPointMatches {
    std::vector<cv::KeyPoint> queryKpts;
    cv::Mat queryDescs;
    std::vector<cv::KeyPoint> refKpts;
    cv::Mat refDescs;
    std::vector<cv::DMatch> matchIndices;
};
}  // namespace _cv

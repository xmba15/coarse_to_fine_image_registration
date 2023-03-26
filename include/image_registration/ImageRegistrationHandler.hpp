/**
 * @file    ImageRegistrationHandler.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <string>

#include <image_registration/SuperGlueMatcher.hpp>
#include <image_registration/Types.hpp>

namespace _cv
{
class ImageRegistrationHandler
{
 public:
    struct Param {
        std::string pathToSuperPointWeights = "";
        std::string pathToSuperGlueWeights = "";

        // superpoint param
        int imageHeight = 480;
        int imageWidth = 640;
        float distThresh = 1;  // nms
        float borderRemove = 2;
        float confidenceThresh = 0.2;
        int gpuIdx = -1;

        // superglue param
        float superGlueConfidenceThresh = 0.7;

        // ransac param
        float reprojThresh = 5.0;

        bool useTPS = true;  // flag to specify whether to use thin plate spline
    };

 public:
    explicit ImageRegistrationHandler(const Param& param);

    cv::Mat run(const cv::Mat& queryImage, const cv::Mat& refImage) const;

 private:
    Param m_param;
    cv::Ptr<SuperGlueMatcher> m_imageMatcher;
};

}  // namespace _cv

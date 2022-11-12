/**
 * @file    SuperGlueMatcher.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <memory>
#include <string>

#include <image_registration/Types.hpp>

namespace _cv
{
class SuperGlueMatcher
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
        float confidenceThresh = 0.01;
        int gpuIdx = -1;
    };

 public:
    explicit SuperGlueMatcher(const Param& param);
    ~SuperGlueMatcher();

    KeyPointMatches run(const cv::Mat& queryImage, const cv::Mat& refImage) const;

 private:
    class SuperGlueMatcherImpl;
    std::unique_ptr<SuperGlueMatcherImpl> m_piml;
};

}  // namespace _cv

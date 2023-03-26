/**
 * @file    ImageRegistrationHandler.cpp
 *
 * @author  btran
 *
 */

#include <image_registration/ImageRegistrationHandler.hpp>
#include <torch_cpp/Utility.hpp>

namespace _cv
{
ImageRegistrationHandler::ImageRegistrationHandler(const Param& param)
    : m_param(param)
    , m_imageMatcher(nullptr)
{
    SuperGlueMatcher::Param imageMatcherParam{.pathToSuperPointWeights = param.pathToSuperPointWeights,
                                              .pathToSuperGlueWeights = param.pathToSuperGlueWeights,
                                              .imageHeight = param.imageHeight,
                                              .imageWidth = param.imageWidth,
                                              .distThresh = param.distThresh,
                                              .borderRemove = param.borderRemove,
                                              .confidenceThresh = param.confidenceThresh,
                                              .superGlueConfidenceThresh = param.superGlueConfidenceThresh,
                                              .gpuIdx = param.gpuIdx};
    m_imageMatcher = cv::makePtr<SuperGlueMatcher>(imageMatcherParam);
}

cv::Mat ImageRegistrationHandler::run(const cv::Mat& queryImage, const cv::Mat& refImage) const
{
    _cv::KeyPointMatches kptMatches = m_imageMatcher->runCoarseMatching(queryImage, refImage);
    INFO_LOG("Number of coarse matches: %lu", kptMatches.size())
    cv::Mat homography = kptMatches.estimateHomography(m_param.reprojThresh);
    if (homography.empty()) {
        return cv::Mat();
    }

    cv::Mat warpedImage;
    cv::warpPerspective(queryImage, warpedImage, homography, refImage.size());

    kptMatches = m_imageMatcher->runFineMatching(warpedImage, refImage);
    INFO_LOG("Number of fine matches: %lu", kptMatches.size())

    if (!m_param.useTPS) {
        homography = kptMatches.estimateHomography(m_param.reprojThresh);
        if (homography.empty()) {
            return warpedImage;
        }
        cv::warpPerspective(warpedImage, warpedImage, homography, refImage.size());
        return warpedImage;
    }

    kptMatches = kptMatches.filterByHomography(m_param.reprojThresh);
    INFO_LOG("Number of fine matches after filtering: %lu", kptMatches.size())

    std::vector<cv::Point2f> queryPts;
    std::vector<cv::Point2f> refPts;
    for (std::size_t i = 0; i < kptMatches.size(); ++i) {
        queryPts.emplace_back(kptMatches.queryKpts[kptMatches.matchIndices[i].queryIdx].pt);
        refPts.emplace_back(kptMatches.refKpts[kptMatches.matchIndices[i].trainIdx].pt);
    }

    auto tps = cv::createThinPlateSplineShapeTransformer(25000);
    INFO_LOG("Estimating Thin Plate Spline model...");
    tps->estimateTransformation(refPts, queryPts, kptMatches.matchIndices);
    INFO_LOG("Warping by Thin Plate Spline model...");
    tps->warpImage(warpedImage, warpedImage);

    return warpedImage;
}
}  // namespace _cv

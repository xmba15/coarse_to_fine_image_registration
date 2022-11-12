/**
 * @file    SuperGlueMatcher.cpp
 *
 * @author  btran
 *
 */

#include <image_registration/SuperGlueMatcher.hpp>

#include <torch_cpp/torch_cpp.hpp>

namespace _cv
{
class SuperGlueMatcher::SuperGlueMatcherImpl
{
 public:
    explicit SuperGlueMatcherImpl(const SuperGlueMatcher::Param& param)
        : m_param(param)
        , m_superPoint(nullptr)
        , m_superGlue(nullptr)
    {
        if (m_param.pathToSuperPointWeights.size() == 0) {
            throw std::runtime_error("Empty path to SuperPoint weights");
        }

        if (m_param.pathToSuperGlueWeights.size() == 0) {
            throw std::runtime_error("Empty path to SuperGlue weights");
        }

        _cv::SuperPoint::Param superPointParam;
        superPointParam.imageHeight = m_param.imageHeight;
        superPointParam.imageWidth = m_param.imageWidth;
        superPointParam.pathToWeights = m_param.pathToSuperPointWeights;
        superPointParam.distThresh = m_param.distThresh;
        superPointParam.borderRemove = m_param.borderRemove;
        superPointParam.confidenceThresh = m_param.confidenceThresh;
        superPointParam.gpuIdx = m_param.gpuIdx;

        m_superPoint = _cv::SuperPoint::create(superPointParam);

        _cv::SuperGlue::Param superGlueParam;
        superGlueParam.pathToWeights = m_param.pathToSuperGlueWeights;
        superGlueParam.gpuIdx = m_param.gpuIdx;
        m_superGlue = _cv::SuperGlue::create(superGlueParam);
    }

    ~SuperGlueMatcherImpl() = default;

 public:
    KeyPointMatches run(const cv::Mat& queryImage, const cv::Mat& refImage) const
    {
        if (queryImage.empty() || queryImage.channels() != 1) {
            throw std::runtime_error("Invalid query image");
        }

        if (refImage.empty() || refImage.channels() != 1) {
            throw std::runtime_error("Invalid reference image");
        }

        KeyPointMatches kptMatches;

        m_superPoint->detectAndCompute(queryImage, cv::Mat(), kptMatches.queryKpts, kptMatches.queryDescs);
        m_superPoint->detectAndCompute(refImage, cv::Mat(), kptMatches.refKpts, kptMatches.refDescs);

        m_superGlue->match(kptMatches.queryDescs, kptMatches.queryKpts, queryImage.size(), kptMatches.refDescs,
                           kptMatches.refKpts, refImage.size(), kptMatches.matchIndices);

        return kptMatches;
    }

 private:
    SuperGlueMatcher::Param m_param;
    cv::Ptr<cv::Feature2D> m_superPoint;
    cv::Ptr<_cv::SuperGlue> m_superGlue;
};

SuperGlueMatcher::SuperGlueMatcher(const Param& param)
    : m_piml(std::make_unique<SuperGlueMatcherImpl>(param))
{
}

SuperGlueMatcher::~SuperGlueMatcher() = default;

KeyPointMatches SuperGlueMatcher::run(const cv::Mat& queryImage, const cv::Mat& refImage) const
{
    return m_piml->run(queryImage, refImage);
}
}  // namespace _cv

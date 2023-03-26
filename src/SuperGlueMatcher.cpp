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
        superGlueParam.matchThreshold = m_param.superGlueConfidenceThresh;
        superGlueParam.gpuIdx = m_param.gpuIdx;
        m_superGlue = _cv::SuperGlue::create(superGlueParam);
    }

    ~SuperGlueMatcherImpl() = default;

 public:
    KeyPointMatches runCoarseMatching(const cv::Mat& queryImage, const cv::Mat& refImage) const
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

    KeyPointMatches runFineMatching(const cv::Mat& queryImage, const cv::Mat& refImage) const
    {
        if (queryImage.empty() || queryImage.channels() != 1) {
            throw std::runtime_error("Invalid query image");
        }

        if (refImage.empty() || refImage.channels() != 1) {
            throw std::runtime_error("Invalid reference image");
        }

        return this->matchBySlidingWindow(queryImage, refImage);
    }

 private:
    KeyPointMatches matchBySlidingWindow(const cv::Mat& queryImage, const cv::Mat& refImage, int numRowDivision = 2,
                                         int numColDivision = 2) const
    {
        if (queryImage.rows != refImage.rows || queryImage.cols != refImage.cols) {
            throw std::runtime_error("Query and reference images must be of the same size");
        }

        if (numRowDivision <= 0 || numColDivision <= 0) {
            throw std::runtime_error("Number of divisions must be more than 1");
        }

        int height = queryImage.rows;
        int width = queryImage.cols;
        int rowStep = height / numRowDivision;
        int colStep = width / numColDivision;

        KeyPointMatches kptMatches;
        int count = 0;

        for (int yMin = 0; yMin < height; yMin += rowStep) {
            for (int xMin = 0; xMin < width; xMin += colStep) {
                int yMax = std::min<int>(yMin + rowStep - 1, height - 1);
                int xMax = std::min<int>(xMin + colStep - 1, width - 1);

                int yMinBuffer = std::max<int>(yMin - m_param.borderRemove, 0);
                int xMinBuffer = std::max<int>(xMin - m_param.borderRemove, 0);

                int yMaxBuffer = std::min<int>(yMax + m_param.borderRemove, height - 1);
                int xMaxBuffer = std::min<int>(xMax + m_param.borderRemove, width - 1);

                auto curWindow = cv::Rect(cv::Point(xMinBuffer, yMinBuffer), cv::Point(xMaxBuffer, yMaxBuffer));

                KeyPointMatches curKptMatches;
                m_superPoint->detectAndCompute(cv::Mat(queryImage, curWindow), cv::Mat(), curKptMatches.queryKpts,
                                               curKptMatches.queryDescs);
                m_superPoint->detectAndCompute(cv::Mat(refImage, curWindow), cv::Mat(), curKptMatches.refKpts,
                                               curKptMatches.refDescs);

                m_superGlue->match(curKptMatches.queryDescs, curKptMatches.queryKpts, curWindow.size(),
                                   curKptMatches.refDescs, curKptMatches.refKpts, curWindow.size(),
                                   curKptMatches.matchIndices);

                for (const auto& curMatch : curKptMatches.matchIndices) {
                    int queryX = curKptMatches.queryKpts[curMatch.queryIdx].pt.x;
                    int queryY = curKptMatches.queryKpts[curMatch.queryIdx].pt.y;

                    int refX = curKptMatches.refKpts[curMatch.trainIdx].pt.x;
                    int refY = curKptMatches.refKpts[curMatch.trainIdx].pt.y;

                    if (queryX < xMin - xMinBuffer || queryX > xMax - xMinBuffer || queryY < yMin - yMinBuffer ||
                        queryY > yMax - yMinBuffer) {
                        continue;
                    }

                    if (refX < xMin - xMinBuffer || refX > xMax - xMinBuffer || refY < yMin - yMinBuffer ||
                        refY > yMax - yMinBuffer) {
                        continue;
                    }

                    cv::KeyPoint queryKpt = curKptMatches.queryKpts[curMatch.queryIdx];
                    cv::KeyPoint refKpt = curKptMatches.refKpts[curMatch.trainIdx];
                    queryKpt.pt.x += xMinBuffer;
                    queryKpt.pt.y += yMinBuffer;
                    refKpt.pt.x += xMinBuffer;
                    refKpt.pt.y += yMinBuffer;
                    kptMatches.queryKpts.emplace_back(std::move(queryKpt));
                    kptMatches.refKpts.emplace_back(std::move(refKpt));
                    kptMatches.queryDescs.push_back(curKptMatches.queryDescs.row(curMatch.queryIdx));
                    kptMatches.refDescs.push_back(curKptMatches.refDescs.row(curMatch.trainIdx));
                    cv::DMatch matchIndex;
                    matchIndex.queryIdx = count;
                    matchIndex.trainIdx = count;
                    matchIndex.distance = curMatch.distance;
                    kptMatches.matchIndices.emplace_back(std::move(matchIndex));
                    count++;
                }
            }
        }

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

KeyPointMatches SuperGlueMatcher::runCoarseMatching(const cv::Mat& queryImage, const cv::Mat& refImage) const
{
    return m_piml->runCoarseMatching(queryImage, refImage);
}

KeyPointMatches SuperGlueMatcher::runFineMatching(const cv::Mat& queryImage, const cv::Mat& refImage) const
{
    return m_piml->runFineMatching(queryImage, refImage);
}
}  // namespace _cv

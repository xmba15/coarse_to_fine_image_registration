/**
 * @file    Types.cpp
 *
 * @author  btran
 *
 */

#include <image_registration/Types.hpp>

namespace _cv
{
std::size_t KeyPointMatches::size() const
{
    return this->matchIndices.size();
}

cv::Mat KeyPointMatches::estimateHomography(float reprojThresh)
{
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    for (std::size_t i = 0; i < this->size(); ++i) {
        pts1.emplace_back(this->queryKpts[this->matchIndices[i].queryIdx].pt);
        pts2.emplace_back(this->refKpts[this->matchIndices[i].trainIdx].pt);
    }

    return cv::findHomography(pts1, pts2, cv::RANSAC, reprojThresh, std::vector<char>(this->size()));
}

KeyPointMatches KeyPointMatches::filterByHomography(float reprojThresh)
{
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    for (std::size_t i = 0; i < this->size(); ++i) {
        pts1.emplace_back(this->queryKpts[this->matchIndices[i].queryIdx].pt);
        pts2.emplace_back(this->refKpts[this->matchIndices[i].trainIdx].pt);
    }

    cv::Mat masks;
    cv::findHomography(pts1, pts2, cv::RANSAC, reprojThresh, masks);

    std::vector<cv::DMatch> inlinerMatch;
    for (int i = 0; i < masks.rows; ++i) {
        uchar* inliner = masks.ptr<uchar>(i);
        if (inliner[0] == 1) {
            inlinerMatch.push_back(this->matchIndices[i]);
        }
    }

    KeyPointMatches outKeyPointMatches = *this;
    outKeyPointMatches.matchIndices = inlinerMatch;

    return outKeyPointMatches;
}
}  // namespace _cv

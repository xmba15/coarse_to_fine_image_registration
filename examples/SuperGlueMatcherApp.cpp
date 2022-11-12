/**
 * @file    SuperGlueMatcherApp.cpp
 *
 * @author  btran
 *
 */

#include <image_registration/image_registration.hpp>

namespace
{
inline cv::Mat findKeyPointsHomography(const _cv::KeyPointMatches kptMatches, const std::vector<char>& matchMask);
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [app] [path/to/superpoint/weights] [path/to/superglue/weights]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::vector<std::string> IMAGE_PATHS = {"./data/00564a_blue.png", "./data/00564a_green.png",
                                                  "./data/00564a_red.png"};

    std::vector<cv::Mat> grays;
    std::transform(IMAGE_PATHS.begin(), IMAGE_PATHS.end(), std::back_inserter(grays),
                   [](const auto& imagePath) { return cv::imread(imagePath, 0); });

    _cv::SuperGlueMatcher::Param param;
    param.pathToSuperPointWeights = argv[1];
    param.pathToSuperGlueWeights = argv[2];

    _cv::SuperGlueMatcher matcher(param);

    auto kptMatches = matcher.run(grays[1], grays[0]);
    cv::Mat H10 = ::findKeyPointsHomography(kptMatches, std::vector<char>(kptMatches.matchIndices.size(), 1));

    kptMatches = matcher.run(grays[2], grays[0]);
    cv::Mat H20 = ::findKeyPointsHomography(kptMatches, std::vector<char>(kptMatches.matchIndices.size(), 1));

    cv::warpPerspective(grays[1], grays[1], H10, grays[0].size());
    cv::warpPerspective(grays[2], grays[2], H20, grays[0].size());

    cv::Mat merged;
    cv::merge(std::vector<cv::Mat>{std::move(grays[0]), std::move(grays[1]), std::move(grays[2])}.data(), 3, merged);
    cv::imwrite("merged.jpg", merged);

    return EXIT_SUCCESS;
}

namespace
{
inline cv::Mat findKeyPointsHomography(const _cv::KeyPointMatches kptMatches, const std::vector<char>& matchMask)
{
    if (matchMask.size() < 3) {
        return cv::Mat();
    }
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    for (std::size_t i = 0; i < kptMatches.matchIndices.size(); ++i) {
        pts1.emplace_back(kptMatches.queryKpts[kptMatches.matchIndices[i].queryIdx].pt);
        pts2.emplace_back(kptMatches.refKpts[kptMatches.matchIndices[i].trainIdx].pt);
    }
    return cv::findHomography(pts1, pts2, cv::RANSAC, 4, matchMask);
}
}  // namespace

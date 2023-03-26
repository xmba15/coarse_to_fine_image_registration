/**
 * @file    SuperGlueMatcherTwoRGBs.cpp
 *
 * @author  btran
 *
 */

#include <image_registration/image_registration.hpp>

int main(int argc, char* argv[])
{
    if (argc != 5) {
        std::cerr << "Usage: [app] [path/to/superpoint/weights] [path/to/superglue/weights] [path/to/image/1] "
                     "[path/to/image/2]"
                  << std::endl;
        return EXIT_FAILURE;
    }
    const std::string SUPERPOINT_WEIGHTS_PATH = argv[1];
    const std::string SUPERGLUE_WEIGHTS_PATH = argv[2];
    const std::vector<std::string> IMAGE_PATHS = {argv[3], argv[4]};

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> grays;
    std::transform(IMAGE_PATHS.begin(), IMAGE_PATHS.end(), std::back_inserter(images),
                   [](const auto& imagePath) { return cv::imread(imagePath); });
    for (int i = 0; i < 2; ++i) {
        if (images[i].empty()) {
            throw std::runtime_error("failed to open " + IMAGE_PATHS[i]);
        }
    }
    std::transform(IMAGE_PATHS.begin(), IMAGE_PATHS.end(), std::back_inserter(grays),
                   [](const auto& imagePath) { return cv::imread(imagePath, 0); });

    _cv::SuperGlueMatcher::Param param;
    param.pathToSuperPointWeights = argv[1];
    param.pathToSuperGlueWeights = argv[2];
    param.borderRemove = 4;

    _cv::SuperGlueMatcher matcher(param);
    _cv::KeyPointMatches kptMatches = matcher.runCoarseMatching(grays[0], grays[1]);
    cv::Mat H = kptMatches.estimateHomography(5.0 /* reprojection threshold */);
    if (H.empty()) {
        std::cerr << "failed to estimate transformation matrix" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "number of matches: " << kptMatches.size() << std::endl;
    cv::Mat warpedImage;
    cv::warpPerspective(images[0], warpedImage, H, images[1].size());
    std::string warpedImagePath = "warped_rgb.jpg";
    std::cout << "Warped image is saved into: " << warpedImagePath << std::endl;
    cv::imwrite(warpedImagePath, warpedImage);

    cv::Mat blended;
    float alpha = 0.5;
    cv::addWeighted(warpedImage, alpha, images[1], 1 - alpha, 0.0, blended);
    std::string blendedImagePath = "blended.jpg";
    std::cout << "Blended image is saved into: " << blendedImagePath << std::endl;
    cv::imwrite(blendedImagePath, blended);

    cv::Mat matchesImage;
    cv::drawMatches(images[0], kptMatches.queryKpts, images[1], kptMatches.refKpts, kptMatches.matchIndices,
                    matchesImage, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    std::string matchedImagePath = "good_matches.jpg";
    std::cout << "Matched image is saved into: " << matchedImagePath << std::endl;
    cv::imwrite(matchedImagePath, matchesImage);

    return EXIT_SUCCESS;
}

/**
 * @file    SuperGlueMatcherApp.cpp
 *
 * @author  btran
 *
 */

#include <image_registration/image_registration.hpp>

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

    _cv::ImageRegistrationHandler::Param param;
    param.pathToSuperPointWeights = argv[1];
    param.pathToSuperGlueWeights = argv[2];
    param.imageHeight = 960;
    param.imageWidth = 1280;
    param.useTPS = false;

    _cv::ImageRegistrationHandler registrationHandler(param);

    grays[1] = registrationHandler.run(grays[1], grays[0]);
    grays[2] = registrationHandler.run(grays[2], grays[0]);

    cv::Mat merged;
    cv::merge(grays.data(), 3, merged);

    std::string mergedImagePath = "merged.jpg";
    std::cout << "Merged image is saved into: " << mergedImagePath << std::endl;
    cv::imwrite(mergedImagePath, merged);

    return EXIT_SUCCESS;
}

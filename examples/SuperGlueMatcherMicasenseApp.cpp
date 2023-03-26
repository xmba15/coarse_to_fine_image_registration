/**
 * @file    SuperGlueMatcherMicasenseApp.cpp
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

    const std::vector<std::string> IMAGE_PATHS = {
        "./data/multispectral/IMG_0008_1.tif", "./data/multispectral/IMG_0008_2.tif",
        "./data/multispectral/IMG_0008_3.tif", "./data/multispectral/IMG_0008_4.tif",
        "./data/multispectral/IMG_0008_5.tif",
    };

    std::vector<cv::Mat> grays;
    std::transform(IMAGE_PATHS.begin(), IMAGE_PATHS.end(), std::back_inserter(grays),
                   [](const auto& imagePath) { return cv::imread(imagePath, 0); });

    _cv::ImageRegistrationHandler::Param param;
    param.pathToSuperPointWeights = argv[1];
    param.pathToSuperGlueWeights = argv[2];
    param.imageHeight = 960;
    param.imageWidth = 1280;
    param.useTPS = true;

    _cv::ImageRegistrationHandler registrationHandler(param);

    for (int i = 1; i < IMAGE_PATHS.size(); ++i) {
        grays[i] = registrationHandler.run(grays[i], grays[0]);
    }

    cv::Mat mergedRGB;
    cv::merge(std::vector<cv::Mat>{grays[0], grays[1], grays[2]}.data(), 3, mergedRGB);
    std::string mergedRGBPath = "micasense_rgb.jpg";
    std::cout << "Merged RGB image is saved into: " << mergedRGBPath << std::endl;
    cv::imwrite(mergedRGBPath, mergedRGB);

    cv::Mat mergedNIRGB;
    cv::merge(std::vector<cv::Mat>{grays[0], grays[1], grays[4]}.data(), 3, mergedNIRGB);
    std::string mergedNIRGBPath = "micasense_nirgb.jpg";
    std::cout << "Merged NIRGB image is saved into: " << mergedNIRGBPath << std::endl;
    cv::imwrite(mergedNIRGBPath, mergedNIRGB);

    return EXIT_SUCCESS;
}

/**
 * @file    TestSuperGlueMatcher.cpp
 *
 * @author  btran
 *
 */

#include <gtest/gtest.h>

#include <experimental/filesystem>

#include <image_registration/image_registration.hpp>

#include "HttpDownloader.hpp"
#include "config.h"

namespace fs = std::experimental::filesystem;

class TestSuperGlueMatcher : public ::testing::Test
{
 protected:
    TestSuperGlueMatcher()
        : m_superpointWeightsFile("/tmp/" + fs::path(TestSuperGlueMatcher::SUPERPOINT_MODEL_URL).filename().string())
        , m_superglueWeightsFile("/tmp/" + fs::path(TestSuperGlueMatcher::SUPERGLUE_MODEL_URL).filename().string())
    {
        HTTPDownloader downloader;
        if (!fs::exists(m_superpointWeightsFile)) {
            downloader.download(TestSuperGlueMatcher::SUPERPOINT_MODEL_URL, m_superpointWeightsFile);
        }

        if (!fs::exists(m_superglueWeightsFile)) {
            downloader.download(TestSuperGlueMatcher::SUPERGLUE_MODEL_URL, m_superglueWeightsFile);
        }
    }

    ~TestSuperGlueMatcher() override
    {
        fs::remove(m_superpointWeightsFile);
        fs::remove(m_superglueWeightsFile);
    }

    static const char SUPERPOINT_MODEL_URL[];
    static const char SUPERGLUE_MODEL_URL[];

    std::string m_superpointWeightsFile;
    std::string m_superglueWeightsFile;
};

const char TestSuperGlueMatcher::SUPERPOINT_MODEL_URL[] =
    "https://github.com/xmba15/torch_cpp/releases/download/0.0.1/superpoint_model.pt";

const char TestSuperGlueMatcher::SUPERGLUE_MODEL_URL[] =
    "https://github.com/xmba15/torch_cpp/releases/download/0.0.1/superglue_model.pt";

TEST_F(TestSuperGlueMatcher, TestInitializationFailure)
{
    _cv::SuperGlueMatcher::Param param;
    EXPECT_ANY_THROW({ _cv::SuperGlueMatcher matcher(param); });
}

TEST_F(TestSuperGlueMatcher, TestInitializationSuccess)
{
    std::unique_ptr<_cv::SuperGlueMatcher> matcher;

    EXPECT_NO_THROW({
        _cv::SuperGlueMatcher::Param param;
        param.pathToSuperPointWeights = m_superpointWeightsFile;
        param.pathToSuperGlueWeights = m_superglueWeightsFile;
        matcher.reset(new _cv::SuperGlueMatcher(param));
    });

    std::string queryImagePath = std::string(DATA_PATH) + "/00564a_blue.png";
    std::string refImagePath = std::string(DATA_PATH) + "/00564a_red.png";

    cv::Mat queryImage = cv::imread(queryImagePath, 0);
    cv::Mat refImage = cv::imread(refImagePath, 0);

    ASSERT_FALSE(queryImage.empty());
    ASSERT_FALSE(refImage.empty());

    auto kptMatches = matcher->runCoarseMatching(queryImage, refImage);

    EXPECT_GT(kptMatches.size(), 0);
}

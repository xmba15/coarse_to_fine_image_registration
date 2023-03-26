#pragma once

#include <string>

class HTTPDownloader
{
 public:
    HTTPDownloader();
    ~HTTPDownloader();
    void download(const std::string& url, const std::string& outputFile);

 private:
    void* curl;
};

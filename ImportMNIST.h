#include <vector>
#include <fstream>
#include <cstdint>
#include <stdexcept>

using namespace std;

// Read 32-bit integer in big-endian format
int readBigEndianInt(ifstream &file) {
    unsigned char bytes[4];
    file.read((char*)bytes, 4);
    return (bytes[0] << 24) |
           (bytes[1] << 16) |
           (bytes[2] << 8)  |
            bytes[3];
}

vector<vector<float>> loadMNISTImages(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file)
        throw runtime_error("Cannot open MNIST image file");

    int magic = readBigEndianInt(file);
    int numImages = readBigEndianInt(file);
    int rows = readBigEndianInt(file);
    int cols = readBigEndianInt(file);

    if (magic != 2051)
        throw runtime_error("Invalid MNIST image file");

    vector<vector<float>> images(numImages, vector<float>(rows * cols));

    for (int i = 0; i < numImages; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, 1);
            images[i][j] = static_cast<float>(pixel) / 255.0f;
        }
    }

    return images;
}

vector<vector<float>> loadMNISTLabels(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file)
        throw runtime_error("Cannot open MNIST label file");

    int magic = readBigEndianInt(file);
    int numLabels = readBigEndianInt(file);

    if (magic != 2049)
        throw runtime_error("Invalid MNIST label file");

    vector<vector<float>> labels(numLabels, vector<float>(10, 0.0f));
    for (int i = 0; i < numLabels; ++i) {
        unsigned char label;
        file.read((char*)&label, 1);
        labels[i][label] = 1.0f;
    }

    return labels;
}

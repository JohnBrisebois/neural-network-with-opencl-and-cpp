#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <cassert>
#include <algorithm>
#include "ImportMNIST.h"
using namespace std;

// Print progress bar
void printProgressBar(int current, int total)
{
    const int barWidth = 50;
    if (total <= 0)
        return;
    int percent = (current * 100) / total;
    if (percent > 100)
        percent = 100;
    cout << "[";
    int filled = (barWidth * percent) / 100;
    for (int i = 0; i < barWidth; ++i)
        cout << (i < filled ? '#' : ' ');
    cout << "] " << percent << "%\r";
    cout.flush();
}

// read kernel file
string readFile(const char *path)
{
    ifstream in(path);
    ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}


class NNGPU
{
    // ##### EDITABLE VALUES:                                          #####
    // ##### LR: learning rate.                                        #####
    // ##### BATCH SIZE: number of data inputs to run in parallel.     #####
    // ##### EPOCHS: number of passes the nn takes when training       #####

    vector<int> topo;
    int L;
    float LR = 0.005f;
    int BATCH_SIZE = 1024;
    int EPOCHS = 100;

    int INPUT_N, OUTPUT_N;
    vector<float> dataset_X, dataset_Y;
    int DATASET_SIZE;

    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    cl_kernel k_forward, k_out_delta, k_hidden_delta, k_grad, k_grad_b, k_apply_weights, k_apply_bias;

    vector<cl_mem> d_weights, d_bias, d_grad, d_grad_b, d_activations, d_deltas;
    cl_mem d_batchX, d_batchY;

public:
    NNGPU(vector<int> topology, vector<vector<float>> input, vector<vector<float>> output)
    {
        topo = topology;
        L = topo.size();
        INPUT_N = input[0].size();
        OUTPUT_N = output[0].size();
        DATASET_SIZE = input.size();

        // Flatten dataset to easily map to buffer
        for (int i = 0; i < DATASET_SIZE; ++i)
        {
            dataset_X.insert(dataset_X.end(), input[i].begin(), input[i].end());
            dataset_Y.insert(dataset_Y.end(), output[i].begin(), output[i].end());
        }

        // OpenCL initialization
        cl_uint num_platforms;
        clGetPlatformIDs(0, nullptr, &num_platforms);
        vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        device = nullptr;
        for (auto p : platforms)
        {
            cl_uint nd = 0;
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd);
            if (nd > 0)
            {
                vector<cl_device_id> devs(nd);
                clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
                device = devs[0];
                break;
            }
        }
        if (!device)
            clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 1, &device, nullptr);

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        queue = clCreateCommandQueue(context, device, 0, nullptr);

        string src = readFile("kernel.cl");
        const char *src_cstr = src.c_str();
        program = clCreateProgramWithSource(context, 1, &src_cstr, nullptr, nullptr);
        clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

        // Set kernels 
        k_forward = clCreateKernel(program, "forward_batch", nullptr);
        k_out_delta = clCreateKernel(program, "output_delta", nullptr);
        k_hidden_delta = clCreateKernel(program, "hidden_delta", nullptr);
        k_grad = clCreateKernel(program, "compute_gradients", nullptr);
        k_grad_b = clCreateKernel(program, "compute_grad_bias", nullptr);
        k_apply_weights = clCreateKernel(program, "apply_weights", nullptr);
        k_apply_bias = clCreateKernel(program, "apply_bias", nullptr);

        // Initialize weights and biases
        d_weights.resize(L);
        d_bias.resize(L);
        d_grad.resize(L);
        d_grad_b.resize(L);
        mt19937 rng(1234);

        // creat buffers for weights, biases and gradients

        for (int l = 1; l < L; ++l)
        {
            int prev_n = topo[l - 1];
            int curr_n = topo[l];
            int fan_in = prev_n;

            vector<float> weights(curr_n * prev_n);
            vector<float> bias(curr_n, 0.0f); // zero biases

            normal_distribution<float> dist(0.0f, sqrt(2.0f / fan_in));

            for (auto &w : weights)
                w = dist(rng);

            d_weights[l] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          sizeof(float) * weights.size(), weights.data(), nullptr);
            d_bias[l] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * bias.size(), bias.data(), nullptr);
            d_grad[l] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * weights.size(), nullptr, nullptr);
            d_grad_b[l] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * bias.size(), nullptr, nullptr);
        }

        // Do the same for activations and deltas
        d_activations.resize(L);
        d_deltas.resize(L);
        for (int l = 0; l < L; ++l)
        {
            int n = topo[l];
            d_activations[l] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * BATCH_SIZE * n, nullptr, nullptr);
            d_deltas[l] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * BATCH_SIZE * n, nullptr, nullptr);
        }

        d_batchX = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * BATCH_SIZE * INPUT_N, nullptr, nullptr);
        d_batchY = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * BATCH_SIZE * OUTPUT_N, nullptr, nullptr);

        // Use vector of indices for easy shuffling
        vector<int> indices(DATASET_SIZE);
        for (int i = 0; i < DATASET_SIZE; ++i)
            indices[i] = i;
        
        auto start_time = chrono::system_clock::now();

        // Training
        for (int epoch = 0; epoch < EPOCHS; ++epoch)
        {
            mt19937 rng(epoch);
            shuffle(indices.begin(), indices.end(), rng);

            for (int start = 0; start < DATASET_SIZE; start += BATCH_SIZE)
            {
                if (start + BATCH_SIZE > DATASET_SIZE)
                    break;

                // Copy batch to GPU
                clEnqueueWriteBuffer(queue, d_batchX, CL_TRUE, 0,
                                     sizeof(float) * BATCH_SIZE * INPUT_N, &dataset_X[indices[start] * INPUT_N],
                                     0, nullptr, nullptr);
                clEnqueueWriteBuffer(queue, d_batchY, CL_TRUE, 0,
                                     sizeof(float) * BATCH_SIZE * OUTPUT_N, &dataset_Y[indices[start] * OUTPUT_N],
                                     0, nullptr, nullptr);

                // Copy batch to input activations
                clEnqueueCopyBuffer(queue, d_batchX, d_activations[0], 0, 0,
                                    sizeof(float) * BATCH_SIZE * INPUT_N, 0, nullptr, nullptr);

                // Forward pass
                for (int l = 1; l < L; ++l)
                {
                    int prev_n = topo[l - 1], curr_n = topo[l];
                    clSetKernelArg(k_forward, 0, sizeof(cl_mem), &d_activations[l - 1]);
                    clSetKernelArg(k_forward, 1, sizeof(cl_mem), &d_weights[l]);
                    clSetKernelArg(k_forward, 2, sizeof(cl_mem), &d_bias[l]);
                    clSetKernelArg(k_forward, 3, sizeof(cl_mem), &d_activations[l]);
                    clSetKernelArg(k_forward, 4, sizeof(int), &prev_n);
                    clSetKernelArg(k_forward, 5, sizeof(int), &curr_n);
                    clSetKernelArg(k_forward, 6, sizeof(int), &BATCH_SIZE);

                    size_t gsize = static_cast<size_t>(BATCH_SIZE * curr_n);
                    clEnqueueNDRangeKernel(queue, k_forward, 1, nullptr, &gsize, nullptr, 0, nullptr, nullptr);
                }

                // Output delta
                int out_n = topo[L - 1];
                clSetKernelArg(k_out_delta, 0, sizeof(cl_mem), &d_activations[L - 1]);
                clSetKernelArg(k_out_delta, 1, sizeof(cl_mem), &d_batchY);
                clSetKernelArg(k_out_delta, 2, sizeof(cl_mem), &d_deltas[L - 1]);
                clSetKernelArg(k_out_delta, 3, sizeof(int), &out_n);
                clSetKernelArg(k_out_delta, 4, sizeof(int), &BATCH_SIZE);

                size_t gsize_out = static_cast<size_t>(BATCH_SIZE * out_n);
                clEnqueueNDRangeKernel(queue, k_out_delta, 1, nullptr, &gsize_out, nullptr, 0, nullptr, nullptr);

                // Hidden deltas
                for (int l = L - 2; l >= 1; --l)
                {
                    int hidden_n = topo[l], next_n = topo[l + 1];
                    clSetKernelArg(k_hidden_delta, 0, sizeof(cl_mem), &d_activations[l]);
                    clSetKernelArg(k_hidden_delta, 1, sizeof(cl_mem), &d_weights[l + 1]);
                    clSetKernelArg(k_hidden_delta, 2, sizeof(cl_mem), &d_deltas[l + 1]);
                    clSetKernelArg(k_hidden_delta, 3, sizeof(cl_mem), &d_deltas[l]);
                    clSetKernelArg(k_hidden_delta, 4, sizeof(int), &hidden_n);
                    clSetKernelArg(k_hidden_delta, 5, sizeof(int), &next_n);
                    clSetKernelArg(k_hidden_delta, 6, sizeof(int), &BATCH_SIZE);

                    size_t gsize_hidden = static_cast<size_t>(BATCH_SIZE * hidden_n);
                    clEnqueueNDRangeKernel(queue, k_hidden_delta, 1, nullptr, &gsize_hidden, nullptr, 0, nullptr, nullptr);
                }

                // Gradients & updates
                for (int l = 1; l < L; ++l)
                {
                    int prev_n = topo[l - 1], curr_n = topo[l];

                    // Compute weight gradients
                    clSetKernelArg(k_grad, 0, sizeof(cl_mem), &d_activations[l - 1]);
                    clSetKernelArg(k_grad, 1, sizeof(cl_mem), &d_deltas[l]);
                    clSetKernelArg(k_grad, 2, sizeof(cl_mem), &d_grad[l]);
                    clSetKernelArg(k_grad, 3, sizeof(int), &prev_n);
                    clSetKernelArg(k_grad, 4, sizeof(int), &curr_n);
                    clSetKernelArg(k_grad, 5, sizeof(int), &BATCH_SIZE);

                    size_t gsize_grad = static_cast<size_t>(curr_n * prev_n);
                    clEnqueueNDRangeKernel(queue, k_grad, 1, nullptr, &gsize_grad, nullptr, 0, nullptr, nullptr);

                    // Compute bias gradients
                    clSetKernelArg(k_grad_b, 0, sizeof(cl_mem), &d_deltas[l]);
                    clSetKernelArg(k_grad_b, 1, sizeof(cl_mem), &d_grad_b[l]);
                    clSetKernelArg(k_grad_b, 2, sizeof(int), &curr_n);
                    clSetKernelArg(k_grad_b, 3, sizeof(int), &BATCH_SIZE);

                    size_t gsize_bias = static_cast<size_t>(curr_n);
                    clEnqueueNDRangeKernel(queue, k_grad_b, 1, nullptr, &gsize_bias, nullptr, 0, nullptr, nullptr);

                    float lr_over_B = LR / BATCH_SIZE;

                    // Apply weight updates
                    clSetKernelArg(k_apply_weights, 0, sizeof(cl_mem), &d_weights[l]);
                    clSetKernelArg(k_apply_weights, 1, sizeof(cl_mem), &d_grad[l]);
                    clSetKernelArg(k_apply_weights, 2, sizeof(float), &lr_over_B);
                    clSetKernelArg(k_apply_weights, 3, sizeof(int), &gsize_grad);
                    clEnqueueNDRangeKernel(queue, k_apply_weights, 1, nullptr, &gsize_grad, nullptr, 0, nullptr, nullptr);

                    // Apply bias updates
                    clSetKernelArg(k_apply_bias, 0, sizeof(cl_mem), &d_bias[l]);
                    clSetKernelArg(k_apply_bias, 1, sizeof(cl_mem), &d_grad_b[l]);
                    clSetKernelArg(k_apply_bias, 2, sizeof(float), &lr_over_B);
                    clSetKernelArg(k_apply_bias, 3, sizeof(int), &curr_n);
                    clEnqueueNDRangeKernel(queue, k_apply_bias, 1, nullptr, &gsize_bias, nullptr, 0, nullptr, nullptr);
                }

                clFinish(queue);
            }
            // Update progress bar (Remove for faster runtime)
            printProgressBar(epoch, EPOCHS - 1);
        }
        cout << endl << "Training done." << endl;
        auto end_time = chrono::system_clock::now();

        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

        cout << "Average Epoch Time: " << duration / EPOCHS << "ms" << endl;
    }

    // Predict output on trained model
    vector<float> predict(const vector<float> &input)
    {
        // Create buffers
        cl_int err;
        vector<cl_mem> d_temp(L);
        for (int l = 0; l < L; ++l)
            d_temp[l] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(float) * (l == 0 ? INPUT_N : topo[l]), nullptr, &err);

        clEnqueueWriteBuffer(queue, d_temp[0], CL_TRUE, 0, sizeof(float) * INPUT_N, input.data(), 0, nullptr, nullptr);
        
        // Forward pass using nn class's weights and biases
        for (int l = 1; l < L; ++l)
        {
            int prev_n = topo[l - 1], curr_n = topo[l];
            int B = 1;
            clSetKernelArg(k_forward, 0, sizeof(cl_mem), &d_temp[l - 1]);
            clSetKernelArg(k_forward, 1, sizeof(cl_mem), &d_weights[l]);
            clSetKernelArg(k_forward, 2, sizeof(cl_mem), &d_bias[l]);
            clSetKernelArg(k_forward, 3, sizeof(cl_mem), &d_temp[l]);
            clSetKernelArg(k_forward, 4, sizeof(int), &prev_n);
            clSetKernelArg(k_forward, 5, sizeof(int), &curr_n);
            clSetKernelArg(k_forward, 6, sizeof(int), &B);
            size_t gsize = curr_n;
            clEnqueueNDRangeKernel(queue, k_forward, 1, nullptr, &gsize, nullptr, 0, nullptr, nullptr);
        }
        // Create output vector
        vector<float> output(topo[L - 1]);
        clEnqueueReadBuffer(queue, d_temp[L - 1], CL_TRUE, 0, sizeof(float) * output.size(), output.data(), 0, nullptr, nullptr);

        // Clear Memory
        for (int l = 0; l < L; ++l)
            clReleaseMemObject(d_temp[l]);
        return output;
    }
};

// Returns index of highest prediction from vector.  This is the number the prediction picked.
int getMax(vector<float> input)
{
    int max = 0;
    for (int i = 0; i < input.size(); i++)
    {
        if (input[i] > input[max])
        {
            max = i;
        }
    }
    return max;
}

int main()
{   
    // Load input and output vectors
    vector<vector<float>> input = loadMNISTImages("train-images.idx3-ubyte");
    vector<vector<float>> output = loadMNISTLabels("train-labels.idx1-ubyte");

    vector<vector<float>> inputTrain(55000);
    vector<vector<float>> outputTrain(55000);

    // only use the first 55000 for training ( so we can test on the rest )
    for (int i = 0; i < 55000; i++) {
        inputTrain[i] = input[i];
        outputTrain[i] = output[i];
    }
    // create and train.  Topology can be any size, although first and last layer must match input/output
    NNGPU nn({784, 256, 128, 10}, inputTrain, outputTrain);

    int start = 55000;
    int end = 56000;

    double errorcnt = 0;

    for (int i = 55000; i < 56000; ++i)
    {
        double res = getMax(nn.predict(input[i]));
        double expected = getMax(output[i]);
        if (res != expected) {
            errorcnt++;
        }
    }
    cout << "Accuracy: " << (end-start-errorcnt) / (end-start) << endl;

    return 0;
}

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread> // For simulating a function with sleep()
#include <complex>
#include <sndfile.h>
#include <fftw3.h>
#include <opencv2/opencv.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <armnn/ArmNN.hpp>

#define SR 8000
#define N_FFT 256
#define HOP_LEN (N_FFT / 2)












// Helper function to make input tensors
armnn::InputTensors MakeInputTensors(const std::pair<armnn::LayerBindingId,
    armnn::TensorInfo>& input,
    const void* inputTensorData)
{
    return { { input.first, armnn::ConstTensor(input.second, inputTensorData) } };
}

// Helper function to make output tensors
armnn::OutputTensors MakeOutputTensors(const std::pair<armnn::LayerBindingId,
    armnn::TensorInfo>& output,
    void* outputTensorData)
{
    return { { output.first, armnn::Tensor(output.second, outputTensorData) } };
}



// Function to preprocess the image into the required format
void PreprocessImage(const cv::Mat& image, std::vector<float>& output) {
    
    std::cout << "PreprocessImage" << std::endl;

    // Resize the image to 129x1251 to match the model's input requirements
    std::cout << "    Resize the image to 129x1251 to match the model's input requirements" << std::endl;
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(129, 1251)); // Resize for the model
    resizedImage.convertTo(resizedImage, CV_32FC3, 1.0 / 255); // Normalize to [0, 1]

    // Flatten the image to a vector
    std::cout << "    Flatten the image to a vector" << std::endl;
    output.resize(resizedImage.total() * resizedImage.channels());
    
    std::memcpy(output.data(), resizedImage.data, output.size() * sizeof(float));
}

// Function to run inference using an ARMNN model
void RunInference(const std::string& modelPath, const cv::Mat& image, const std::string& backendType) {
    std::cout << "RunInference>" << std::endl;
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);

    armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();

    // Read the TFLite model file
    std::cout << "    Read the TFLite model file" << std::endl;
    std::ifstream file(modelPath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open model file: " << modelPath << std::endl;
        return;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> modelData(size);
    file.read(reinterpret_cast<char*>(modelData.data()), size);
    file.close();

    // Create the network from the binary data
    std::cout << "    Create the network from the binary data" << std::endl;
    armnn::INetworkPtr network = parser->CreateNetworkFromBinary(modelData);
       
    // Return the input tensor names for a given subgraph
    std::vector<std::string> InputBindingName = parser->GetSubgraphInputTensorNames(0);
    //const std::string& InputBindingName = parser->GetSubgraphInputTensorNames(0);
  
    // Return the output tensor names for a given subgraph
    std::vector<std::string> OutputBindingName = parser->GetSubgraphOutputTensorNames(0);
    //const std::string& OutputBindingName = parser->GetSubgraphOutputTensorNames(0);
    
    
    std::cout << "    InputBindingName[0] = " << InputBindingName[0] << std::endl;
    std::cout << "    OutputBindingName[0] = " << OutputBindingName[0] << std::endl;
    
    
    size_t numSubgraph = parser->GetSubgraphCount();
    std::cout << "    numSubgraph = " << numSubgraph << std::endl;
    
    for(size_t iter_subgraphs = 0; iter_subgraphs < numSubgraph; iter_subgraphs++){
        std::cout << "    iter_subgraphs = " << iter_subgraphs << std::endl;
        std::cout << "    InputBindingName = " << InputBindingName[iter_subgraphs] << std::endl;
        std::cout << "    OutputBindingName = " << OutputBindingName[iter_subgraphs] << std::endl;
    }
    
    // Find the binding points for the input and output nodes
    armnnTfLiteParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo(0, InputBindingName[0]);
    armnnTfLiteParser::BindingPointInfo outputBindingInfo = parser->GetNetworkOutputBindingInfo(0, OutputBindingName[0]);
    
    
    // Find the binding points for the input and output nodes
    //armnnTfLiteParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo(0, "conv2d_input");
    //armnnTfLiteParser::BindingPointInfo outputBindingInfo = parser->GetNetworkOutputBindingInfo(0, "Identity");


    // Preprocess the input image
    std::cout << "    Preprocess the input image" << std::endl;
    std::vector<float> inputData;
    PreprocessImage(image, inputData);

    // Ensure input data is correct
    /*
    std::cout << "    Ensure input data is correct" << std::endl;
    if (inputData.size() != 129 * 1251 * 3) { // Check if the size is correct
        std::cerr << "Input data size is incorrect. Expected: " << (129 * 1251 * 3) << ", Got: " << inputData.size() << std::endl;
        return;
    }
    */
    std::cout << "    Ensure input data is correct" << std::endl;
    if (inputData.size() != 129 * 1251) { // Check if the size is correct
        std::cerr << "Input data size is incorrect. Expected: " << (129 * 1251) << ", Got: " << inputData.size() << std::endl;
        return;
    }

    // Create TensorInfo for input, ensuring it's a constant tensor
    std::cout << "    Create TensorInfo for input, ensuring it's a constant tensor" << std::endl;
    armnn::TensorInfo inputTensorInfo({1, 129, 1251, 3}, armnn::DataType::Float32);
    //const armnn::TensorInfo inputTensorInfo({1, 150, 150, 3}, armnn::DataType::Float32);
    
    // Create ConstTensor with the correct TensorInfo
    std::cout << "    Create ConstTensor with the correct TensorInfo" << std::endl;
    // Declare inputTensor outside of try-catch block
    //armnn::ConstTensor inputTensor
    
    // Create InputTensors
    std::cout << "    Create InputTensors" << std::endl;
    //armnn::InputTensors inputTensors{ { 0, inputTensor } };
    armnn::InputTensors inputTensor = MakeInputTensors(inputBindingInfo, inputData.data());
    try {
        //armnn::ConstTensor inputTensor(inputTensorInfo, inputData.data());
        //inputTensor = armnn::ConstTensor(inputTensorInfo, inputData.data());
        inputTensorInfo.SetConstant(true);
        } catch (const armnn::InvalidArgumentException& e) {
            std::cerr << "Failed to create ConstTensor: " << e.what() << std::endl;
            return;
    }

    // Allocate memory for output tensor data (2 classes)
    std::cout << "    Allocate memory for output tensor data (2 classes)" << std::endl;
    std::vector<float> outputData(2);
    //armnn::Tensor outputTensor(armnn::TensorInfo({1, 2}, armnn::DataType::Float32), outputData.data());
    // Create OutputTensors
    std::cout << "    Create OutputTensors" << std::endl;
    //armnn::OutputTensors outputTensors{ { 0, armnn::Tensor(run->GetOutputTensorInfo(networkId, 0), outputData.data())} };
    armnn::OutputTensors outputTensor = MakeOutputTensors(outputBindingInfo, outputData.data());

    // Create OutputTensors
    //std::cout << "    Create OutputTensors" << std::endl;
    //armnn::OutputTensors outputTensors{ { 0, outputTensor } };

    // Optimize the network
    std::cout << "    Optimize the network" << std::endl;
    std::vector<armnn::BackendId> backends = {backendType};  // Change based on available backends
    armnn::OptimizerOptionsOpaque optimizerOptionsOpaque;  // Use the ABI stable variant

    armnn::IOptimizedNetworkPtr optimizedNetwork = armnn::Optimize(*network, backends, runtime->GetDeviceSpec(), optimizerOptionsOpaque);
    if (!optimizedNetwork)
    {
        // This shouldn't happen for this simple sample, with reference backend.
        // But in general usage Optimize could fail if the hardware at runtime cannot
        // support the model that has been provided.
        std::cerr << "Error: Failed to optimise the input network." << std::endl;
        return;
    }

    // Load the optimized network
    std::cout << "    Load the optimized network" << std::endl;
    armnn::NetworkId networkId;
    armnn::Status status = runtime->LoadNetwork(networkId, std::move(optimizedNetwork));
    if (status != armnn::Status::Success) {
        std::cerr << "Failed to load network!" << std::endl;
        return;
    }
    
    // Perform inference
    std::cout << "    Perform inference" << std::endl;
    status = runtime->EnqueueWorkload(networkId, inputTensor, outputTensor); //****************************
    if (status != armnn::Status::Success) {
        std::cerr << "Inference failed!" << std::endl;
        return;
    }

    std::cout<< inputBindingInfo.second.GetQuantizationScale() << std::endl;
    std::cout<< inputBindingInfo.second.GetQuantizationOffset() << std::endl;
    std::cout<< inputBindingInfo.second.GetNumBytes() << std::endl;

    std::cout<< outputBindingInfo.second.GetQuantizationScale() << std::endl;
    std::cout<< outputBindingInfo.second.GetQuantizationOffset() << std::endl;
    std::cout<< outputBindingInfo.second.GetNumBytes() << std::endl;

    // Process the output
    std::cout << "    Process the output" << std::endl;
    int predicted_class = std::distance(outputData.begin(), std::max_element(outputData.begin(), outputData.end()));
    std::cout << "Predicted class: " << predicted_class << std::endl; // Will output 0 (clean), 1 (infested)
}




// Function to read audio data
std::vector<double> ReadWaveFile(const std::string& filename, int& sampleRate) {
    std::cout << "ReadWaveFile" << std::endl;
    SF_INFO sfinfo;
    std::memset(&sfinfo, 0, sizeof(SF_INFO)); // Initialize the SF_INFO structure to zero

    // Open audio file for reading
    SNDFILE* infile = sf_open(filename.c_str(), SFM_READ, &sfinfo);
    if (!infile) {
        std::cerr << "Error opening file: " << sf_strerror(nullptr) << std::endl;
        return {}; // Return an empty vector on failure
    }

    sampleRate = sfinfo.samplerate;

    // Check that the file is appropriate
    if (sfinfo.frames <= 0) {
        std::cerr << "Error: The file does not contain audio frames." << std::endl;
        sf_close(infile);
        return {}; // Return an empty vector
    }

    // Allocate vector for samples and read data
    std::vector<double> samples(sfinfo.frames);
    sf_count_t numFramesRead = sf_readf_double(infile, samples.data(), sfinfo.frames);

    // Close the audio file
    sf_close(infile);

    if (numFramesRead < sfinfo.frames) {
        std::cerr << "Warning: Only " << numFramesRead << " frames were read out of " << sfinfo.frames << std::endl;
        samples.resize(numFramesRead); // Resize to the actual number of frames read
    }

    return samples; // Return the audio samples
}

/*
// Function to compute Mel spectrogram
cv::Mat ComputeMelSpectrogram(const std::vector<double>& audio, const std::string& save_path) {
    int mel_fft_size = N_FFT / 2 + 1;
    int num_frames = (audio.size() - N_FFT) / HOP_LEN + 1;

    // Check if there's enough audio data
    if (num_frames <= 0) {
        std::cerr << "Insufficient audio data for spectrogram." << std::endl;
        return {}; // Return an empty matrix
    }

    // Initialize FFTW
    std::vector<fftw_complex> fft_input(N_FFT);  // Use fftw_complex type
    std::vector<fftw_complex> fft_output(mel_fft_size);  // Use fftw_complex type
    
    // Create a matrix to hold the spectrogram
    cv::Mat spectrogram(num_frames, mel_fft_size, CV_32FC1, cv::Scalar(0)); // Num frames, Mel bins

    for (int i = 0; i < num_frames; ++i) {
        // Fill the FFT input with audio data
        for (int j = 0; j < N_FFT; ++j) {
            fft_input[j][0] = (i * HOP_LEN + j < audio.size()) ? audio[i * HOP_LEN + j] : 0.0; // Real part
            fft_input[j][1] = 0.0;  // Imaginary part
        }

        // Perform FFT
        fftw_plan plan = fftw_plan_dft_1d(N_FFT, fft_input.data(), fft_output.data(), FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);

        // Obtain magnitudes and convert them to dB
        for (int j = 0; j < mel_fft_size; ++j) {
            double magnitude = std::sqrt(fft_output[j][0] * fft_output[j][0] + fft_output[j][1] * fft_output[j][1]);
            spectrogram.at<float>(i, j) = 20 * log10(magnitude + 1e-6); // Adding small value to avoid log(0).
        }
    }

    // Process spectrogram size
    if (spectrogram.cols > 1251) {
        spectrogram = spectrogram(cv::Rect(0, 0, 1251, spectrogram.rows)); // Crop to 1251 columns
    }

    // Flip the spectrogram upside down
    cv::flip(spectrogram, spectrogram, 0);

    // Convert the spectrogram to an 8-bit format for saving
    cv::Mat spectrogram_img;
    cv::normalize(spectrogram, spectrogram_img, 0, 255, cv::NORM_MINMAX); // Normalize to range [0, 255]
    spectrogram_img.convertTo(spectrogram_img, CV_8UC1); // Convert to 8-bit single-channel

    // Save the spectrogram image
    cv::imwrite(save_path, spectrogram_img);

    // Expand dimensions (adding channel dimension)
    cv::Mat final_spectrogram = spectrogram_img.reshape(1, spectrogram_img.rows); // Reshape to (129, 1251, 1)

    return final_spectrogram;
}
*/

cv::Mat ComputeMelSpectrogram(const std::vector<double>& audio, const std::string& save_path) {
    std::cout << "ComputeMelSpectrogram" << std::endl;
    int mel_fft_size = N_FFT / 2 + 1;
    int num_frames = (audio.size() - N_FFT) / HOP_LEN + 1;

    // Check if there's enough audio data
    if (num_frames <= 0) {
        std::cerr << "Insufficient audio data for spectrogram." << std::endl;
        return {}; // Return an empty matrix
    }

    // Initialize FFTW input and output buffers correctly
    std::vector<double> fft_input(N_FFT);  // Input buffer for FFT
    std::vector<fftw_complex> fft_output(mel_fft_size);  // FFTW output buffer

    // Create a matrix to hold the spectrogram
    cv::Mat spectrogram(num_frames, mel_fft_size, CV_32FC1, cv::Scalar(0)); // Num frames, Mel bins

    // Perform FFT on each frame
    for (int i = 0; i < num_frames; ++i) {
        // Fill the FFT input buffer
        for (int j = 0; j < N_FFT; ++j) {
            // Ensure we don't exceed audio data bounds
            if (i * HOP_LEN + j < audio.size()) {
                fft_input[j] = audio[i * HOP_LEN + j]; // Keep real part
            } else {
                fft_input[j] = 0.0; // Padding with zeros if out of bounds
            }
        }

        // Perform FFT
        fftw_plan plan = fftw_plan_dft_r2c_1d(N_FFT, fft_input.data(), fft_output.data(), FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);

        // Obtain magnitudes and convert them to dB
        for (int j = 0; j < mel_fft_size; ++j) {
            double magnitude = std::sqrt(fft_output[j][0] * fft_output[j][0] + fft_output[j][1] * fft_output[j][1]);
            spectrogram.at<float>(i, j) = 20 * log10(magnitude + 1e-6); // Adding small value to avoid log(0).
        }
    }

    // Process spectrogram size
    if (spectrogram.cols > 1251) {
        spectrogram = spectrogram(cv::Rect(0, 0, 1251, spectrogram.rows)); // Crop to 1251 columns
    }

    // Flip the spectrogram upside down
    cv::flip(spectrogram, spectrogram, 0);

    // Convert the spectrogram to an 8-bit format for saving
    cv::Mat spectrogram_img;
    cv::normalize(spectrogram, spectrogram_img, 0, 255, cv::NORM_MINMAX); // Normalize to range [0, 255]
    spectrogram_img.convertTo(spectrogram_img, CV_8UC1); // Convert to 8-bit single-channel

    // Save the spectrogram image
    cv::imwrite(save_path, spectrogram_img);

    // Expand dimensions (adding channel dimension)
    cv::Mat final_spectrogram = spectrogram_img.reshape(1, spectrogram_img.rows); // Reshape to (129, 1251, 1)

    return final_spectrogram;
}

// Function to preprocess audio and prepare it for inference
cv::Mat PreprocessAudio(const std::string& audio_file, const std::string& save_spectrogram_path) {
    std::cout << "PreprocessAudio" << std::endl;
    int sampleRate;
    std::vector<double> audio_data = ReadWaveFile(audio_file, sampleRate);
    
    // Randomly roll the audio array
    std::random_shuffle(audio_data.begin(), audio_data.end());

    // Repeat to ensure sufficient length
    if (audio_data.size() < 20 * sampleRate) {
        audio_data.resize(20 * sampleRate);
    }

    // Generate the spectrogram and save it
    cv::Mat spectrogram = ComputeMelSpectrogram(audio_data, save_spectrogram_path);

    return spectrogram;
}

int main(int argc, char** argv) {
    
    // Record the overall start time
    auto start_overall = std::chrono::high_resolution_clock::now();
    
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model.tflite> <audio_path> <backend type (CpuRef or CpuAcc)>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    std::string audioPath = argv[2];
    std::string backendType = argv[3];
    
    
    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "-----------------------inference_RPW_exe-------------------------" << std::endl;
    std::cout << "Input arguments: " << std::endl;
    std::cout << "modelPath: "<< modelPath << "  audioPath: " << audioPath << "  backendType: " << backendType <<std::endl;
    

    std::string audio_file = audioPath; // Replace with your audio file path
    std::string spectrogram_image_path = "spectrogram.png"; // Path to save the spectrogram image
    cv::Mat input_data = PreprocessAudio(audio_file, spectrogram_image_path);

    // Here, you can use input_tensor with your ArmNN model for inference
    
    // Record the inference start time
    auto start_inference = std::chrono::high_resolution_clock::now();
    // Run the inference
    RunInference(modelPath, input_data, backendType);
    // Record the inference end time
    auto end_inference = std::chrono::high_resolution_clock::now();
    // Calculate the duration in milliseconds
    auto duration_inference = std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference);
    std::cout << "Elapsed inference time: " << duration_inference.count() << " milliseconds" << std::endl;

    // Record the overall end time
    auto end_overall = std::chrono::high_resolution_clock::now();
    
    // Calculate the overall duration in milliseconds
    auto duration_overall = std::chrono::duration_cast<std::chrono::milliseconds>(end_overall - start_overall);
    std::cout << "Elapsed overall time: " << duration_overall.count() << " milliseconds" << std::endl;

    return 0;    
}

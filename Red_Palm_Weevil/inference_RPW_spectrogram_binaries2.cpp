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
#include <filesystem>

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










void loadAudioFromFile(const std::string &filename, std::vector<std::vector<std::vector<float>>> &audioData, int rows, int cols) {
    std::ifstream file(filename, std::ios::binary);
    audioData.resize(1, std::vector<std::vector<float>>(rows, std::vector<float>(cols)));

    for (int i = 0; i < rows; i++) {
        file.read(reinterpret_cast<char*>(audioData[0][i].data()), cols * sizeof(float));
    }
    file.close();
}



void loadBinaryToMat(const std::string &filename, cv::Mat &image, int rows, int cols) {
    // Step 1: Read the binary data
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Unable to open file " << filename << std::endl;
        return;
    }

    // Create a vector to hold the image data
    std::vector<float> data(rows * cols); // Adjust the data type as needed (e.g., float, uchar)
    
    // Read the binary data into the vector
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    if (!file) {
        std::cerr << "Error reading file." << std::endl;
        return;
    }
    file.close();

    // Step 2: Create a cv::Mat image from the buffer
    image = cv::Mat(rows, cols, CV_32FC1, data.data()); // For grayscale image
    // Use CV_8UC1 for 8-bit single channel image, CV_8UC3 for 3-channel image, etc.
}




// Function to preprocess the image into the required format
void PreprocessImage(const cv::Mat& image, std::vector<float>& output) {
    
    std::cout << "PreprocessImage" << std::endl;

    // Resize the image to 129rowsx1251cols to match the model's input requirements
    std::cout << "    Resize the image to 129x1251 to match the model's input requirements" << std::endl;
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(1251, 129)); // Resize for the model (width, height)
    //resizedImage.convertTo(resizedImage, CV_32FC1, 1.0 / 255); // Normalize to [0, 1]. CV_32FC1 is for grayscale image like [:::1], instead for RGB channels like [:::3] we would have CV_32FC3
    resizedImage.convertTo(resizedImage, CV_32FC1); //CV_32FC1 is for grayscale image like [:::1], instead for RGB channels like [:::3] we would have CV_32FC3

    // Flatten the image to a vector
    std::cout << "    Flatten the image to a vector" << std::endl;
    //output.resize(resizedImage.total() * resizedImage.channels());
    output.resize(resizedImage.total());
    
    std::memcpy(output.data(), resizedImage.data, output.size() * sizeof(float));
}

// Function to run inference using an ARMNN model
void RunInference(const std::string& modelPath, std::vector<float> inputData, const std::string& backendType) {
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





    //printing image data content
    //std::cout << "image rows*cols = " << image.total() << std::endl;
    
    //std::cout << "Image Data: " << image << std::endl;





    // Preprocess the input image
    //std::cout << "    Preprocess the input image" << std::endl;
    //std::vector<float> inputData;
    //PreprocessImage(image, inputData);

    // Ensure input data is correct
    std::cout << "    Ensure input data is correct" << std::endl;
    if (inputData.size() != 1251 * 129) { // Check if the size is correct
        std::cerr << "Input data size is incorrect. Expected: " << (1251 * 129) << ", Got: " << inputData.size() << std::endl;
        return;
    }
    
    /*
    //printing input data content   
    std::cout << "Input Data: ";
    for (float val : inputData) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    */

    // Create TensorInfo for input, ensuring it's a constant tensor
    std::cout << "    Create TensorInfo for input, ensuring it's a constant tensor" << std::endl;
    armnn::TensorInfo inputTensorInfo({1, 129, 1251, 1}, armnn::DataType::Float32);
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
    
    // Assuming outputData contains the results from the model
    std::cout << "Output Data: ";
    for (float val : outputData) {
        std::cout << val << " ";  // Print raw output values
    }
    std::cout << std::endl;

    std::cout<< "inputBindingInfo.second.GetQuantizationScale(): " << inputBindingInfo.second.GetQuantizationScale() << std::endl;
    std::cout<< "inputBindingInfo.second.GetQuantizationOffset(): " << inputBindingInfo.second.GetQuantizationOffset() << std::endl;
    std::cout<< "inputBindingInfo.second.GetNumBytes(): " << inputBindingInfo.second.GetNumBytes() << std::endl;

    std::cout<< "outputBindingInfo.second.GetQuantizationScale(): " << outputBindingInfo.second.GetQuantizationScale() << std::endl;
    std::cout<< "outputBindingInfo.second.GetQuantizationOffset(): " << outputBindingInfo.second.GetQuantizationOffset() << std::endl;
    std::cout<< "outputBindingInfo.second.GetNumBytes(): " << outputBindingInfo.second.GetNumBytes() << std::endl;

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

    // Flip the spectrogram upside down
    cv::flip(spectrogram, spectrogram, 0);
    //cv::rotate(spectrogram, spectrogram, 2); // 2 --> rotateCode = ROTATE_90_COUNTERCLOCKWISE

    // Process spectrogram size
    if (spectrogram.cols > 1251) {
        spectrogram = spectrogram(cv::Rect(0, 0, 1251, spectrogram.rows)); // Crop to 1251 columns
    }

    // Flip the spectrogram upside down
    //cv::flip(spectrogram, spectrogram, 0);
    //cv::flip(spectrogram, spectrogram, 0);

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


/*
// Function to preprocess audio and prepare it for inference
cv::Mat PreprocessAudio(const std::string& audio_file, const std::string& save_spectrogram_path) {
    std::cout << "PreprocessAudio" << std::endl;
    int sampleRate;
    std::vector<double> audio_data = ReadWaveFile(audio_file, sampleRate);
    
    // Randomly roll the audio array
    //std::random_shuffle(audio_data.begin(), audio_data.end());//*********************************************************************************

    // Repeat to ensure sufficient length
    if (audio_data.size() < 20 * sampleRate) {
        audio_data.resize(20 * sampleRate);
    }

    // Generate the spectrogram and save it
    cv::Mat spectrogram = ComputeMelSpectrogram(audio_data, save_spectrogram_path);

    //return spectrogram;
}
*/


int main(int argc, char** argv) {
    
    // Record the overall start time
    auto start_overall = std::chrono::high_resolution_clock::now();
    
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model.tflite> <spectrograms directoryPath> <backend type (CpuRef or CpuAcc)>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    const char* directoryPath = argv[2];
    std::string backendType = argv[3];
    
    
    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "-----------------------inference_RPW_exe-------------------------" << std::endl;
    std::cout << "Input arguments: " << std::endl;
    std::cout << "modelPath: "<< modelPath << "  spectrograms directoryPath: " << directoryPath << "  backendType: " << backendType <<std::endl;


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // For .npy files
    // Assume you have dimensions to read back
    const int rows = 129;
    const int cols = 1251;

    // Define the directory containing the .bin files
    //const std::string directoryPath = "path/to/your/directory";

    namespace fs = std::filesystem;

    // Iterate over the directory and read .bin files
    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            //readBinaryFile(entry.path());
            //std::string audio_file = entry.path().c_str(); //current audio file path
            
                
            //const char* binPath = entry;

            //std::vector<std::vector<std::vector<float>>> binData;

            //loadAudioFromFile(binPath, binData, rows, cols);
            
            // Ensure data is correctly loaded
            //std::cout << "Loaded audio_bin at (0, 0, 0): " << binData[0][0][0] << std::endl;
            
            

            //cv::Mat image;

            // Convert the binary file to the spectrogram image it represents
            //loadBinaryToMat(binPath, image, rows, cols);
            
            // Check if the spectrogram image is ok
            //if (image.empty()) {
            //    std::cerr << "Failed to load image." << std::endl;
            //    return -1;
            //}
            

            
            std::cout << "----------------------------------------------" << std::endl;
            // Get the path to your binary file
            const char* filename = entry.path().c_str();
            std::cout << filename << std::endl;
            std::cout << "----------------------------------------------" << std::endl;
            
            // Open the binary file in input mode
            std::ifstream file(filename, std::ios::binary);
            if (!file) {
                std::cerr << "Error: Could not open the file!" << std::endl;
                return 1;
            }

            // Use a vector to store the loaded data
            std::vector<float> image;

            // Read the data. Assuming you know the number of elements.
            // You might want to read this from metadata in a real application.
            size_t num_elements = rows*cols; // Change this according to your needs
            image.resize(num_elements);

            // Read the binary data into the vector
            file.read(reinterpret_cast<char*>(image.data()), num_elements * sizeof(float));
            
            // Check if the reading was successful
            if (file.gcount() != num_elements * sizeof(float)) {
                std::cerr << "Error: Could not read enough data!" << std::endl;
                return 1;
            }

            // Close the file
            file.close();
            
            /*
            // Output the read data
            std::cout << "Data read from binary file:" << std::endl;
            for (const auto& value : data) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
            */
            
            // Create a cv::Mat from the vector
            // CV_32F specifies the matrix will hold 32-bit floating point numbers
            //cv::Mat image(rows, cols, CV_32F, data.data());

            // Output the matrix dimensions and data for verification
            std::cout << "Matrix size: " << image.size() << std::endl; // Should output [1251 x 129]
            std::cout << "Data in image" << std::endl;
            //std::cout << image << std::endl;  // This will print the contents of the matrix
            std::cout << "------------------------------------------------------------------------" << std::endl;


            //////////////////////////////////////////////////////////////////////////////////////////////////////////////



            /*
            // added for taking the input image as test
            //cv::Mat spectrogram_input = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
            cv::Mat spectrogram_input = cv::imread(imagePath);
            if (spectrogram_input.empty()) {
                std::cerr << "Failed to load image: " << imagePath << std::endl;
                return 1;
            }
            */
            
            //std::string audio_file = audioPath; // Replace with your audio file path
            //std::string spectrogram_image_path = "spectrogram.png"; // Path to save the spectrogram image
            //cv::Mat input_data = PreprocessAudio(audio_file, spectrogram_image_path);

            // Here, you can use input_tensor with your ArmNN model for inference
            
            // Record the inference start time
            auto start_inference = std::chrono::high_resolution_clock::now();
            // Run the inference
            //RunInference(modelPath, input_data, backendType);
            RunInference(modelPath, image, backendType);
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
                    
        }
    }



    return 0;    
}



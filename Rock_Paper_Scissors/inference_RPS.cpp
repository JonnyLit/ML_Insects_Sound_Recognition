
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread> // For simulating a function with sleep()
#include <armnn/ArmNN.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <opencv2/opencv.hpp>



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

    // Resize the image to 150x150 to match the model's input requirements
    std::cout << "    Resize the image to 150x150 to match the model's input requirements" << std::endl;
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(150, 150)); // Resize for the model
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
    std::cout << "    Ensure input data is correct" << std::endl;
    if (inputData.size() != 150 * 150 * 3) { // Check if the size is correct
        std::cerr << "Input data size is incorrect. Expected: " << (150 * 150 * 3) << ", Got: " << inputData.size() << std::endl;
        return;
    }

    // Create TensorInfo for input, ensuring it's a constant tensor
    std::cout << "    Create TensorInfo for input, ensuring it's a constant tensor" << std::endl;
    armnn::TensorInfo inputTensorInfo({1, 150, 150, 3}, armnn::DataType::Float32);
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

    // Allocate memory for output tensor data (3 classes)
    std::cout << "    Allocate memory for output tensor data (3 classes)" << std::endl;
    std::vector<float> outputData(3);
    //armnn::Tensor outputTensor(armnn::TensorInfo({1, 3}, armnn::DataType::Float32), outputData.data());
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
    // Will output 0 (paper), 1 (rock), 2 (scissors)
    if (predicted_class == 0){
        std::cout << "Predicted class: 0 --> paper" << std::endl;
    }else if (predicted_class == 1){
        std::cout << "Predicted class: 1 --> rock" << std::endl;
    }else if (predicted_class == 2){
        std::cout << "Predicted class: 2 --> scissors" << std::endl;
    }
}

int main(int argc, char** argv) {
    
    // Record the overall start time
    auto start_overall = std::chrono::high_resolution_clock::now();
    
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model.tflite> <image_path> <backend type (CpuRef or CpuAcc)>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    std::string imagePath = argv[2];
    std::string backendType = argv[3];
    
    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "-----------------------inference_RPS_exe-------------------------" << std::endl;
    std::cout << "Input arguments: " << std::endl;
    std::cout << "modelPath: "<< modelPath << "  imagePath: " << imagePath << "  backendType: " << backendType <<std::endl;
    
    

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return 1;
    }

    // Record the inference start time
    auto start_inference = std::chrono::high_resolution_clock::now();

    // Run the inference
    RunInference(modelPath, image, backendType);
    
    // Record the inference end time
    auto end_inference = std::chrono::high_resolution_clock::now();

    // Calculate the inference duration in milliseconds
    auto duration_inference = std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference);
    std::cout << "Elapsed inference time: " << duration_inference.count() << " milliseconds" << std::endl;
    
    // Record the overall end time
    auto end_overall = std::chrono::high_resolution_clock::now();
    
    // Calculate the overall duration in milliseconds
    auto duration_overall = std::chrono::duration_cast<std::chrono::milliseconds>(end_overall - start_overall);
    std::cout << "Elapsed overall time: " << duration_overall.count() << " milliseconds" << std::endl;
       
    return 0;
}

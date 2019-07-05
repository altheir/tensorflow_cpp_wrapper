/*
 * Author: Cameron Chancey
 *
 * Built on Tensorflow 1.10.1
 * A revised example of the tensorflow label image exmample found here:
 * https://github.com/tensorflow/tensorflow/tree/v1.10.1/tensorflow/examples/label_image
 *
 * Cmake compatable and useable outside of the tensorflow repository.
 *
 */

#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"


/**
 * @brief CheckStatus Just handles status checking. In this implementation it throws, I think its clearer than how it was handled in the base case.
 * This also lets you return actual objects from functions, rather than being limited to returning status objects and modifying things by pointer/ref.
 */
void CheckStatus(const tensorflow::Status & status){
    if (not status.ok()){
        throw std::runtime_error(status.error_message());
    }
}

/**
 * @brief ReadFileIntoTensor
 * @param filename
 * @return Tensor (DT_Float, file_size)
 */
tensorflow::Tensor ReadFileIntoTensor(const std::string& filename) {
    tensorflow::uint64 file_size = 0;
    tensorflow::Env* env{tensorflow::Env::Default()};
    CheckStatus(env->GetFileSize(filename, &file_size));
    std::string contents;
    contents.resize(file_size);
    std::unique_ptr<tensorflow::RandomAccessFile> file;
    CheckStatus(env->NewRandomAccessFile(filename, &file));
    tensorflow::StringPiece data;
    CheckStatus(file->Read(0, file_size, &data, &(contents)[0]));
    if (data.size() != file_size) {
        throw std::runtime_error("Error reading file.");
    }
    tensorflow::Tensor output(tensorflow::DT_STRING,tensorflow::TensorShape());
    output.scalar<std::string>()() = data.ToString();
    return output;
}

/**
 * @brief SessionRun Convers the root scope graph and runs the inputs on that graph.
 * @param root scope to convert into a graph.
 * @param op_name which operation to use for the outputs.
 * @param inputs inpputs to run on the graph.
 * @return Run outputs.
 */
std::vector<tensorflow::Tensor> SessionRun(const tensorflow::Scope& root,const std::string& op_name,
                                           const std::vector<std::pair<std::string,tensorflow::Tensor>>&inputs){
    tensorflow::GraphDef graph;
    CheckStatus(root.ToGraphDef(&graph));
    std::unique_ptr<tensorflow::Session> session(
                tensorflow::NewSession(tensorflow::SessionOptions()));
    CheckStatus(session->Create(graph));
    std::vector<tensorflow::Tensor> output;
    CheckStatus(session->Run({inputs}, {op_name}, {}, &output));
    return output;
}


std::vector<std::pair<std::string, tensorflow::Tensor>> CreateTensorInput(const std::string& input_name,const tensorflow::Tensor& input_tensor){
    return {{input_name,input_tensor}};
}

/**
 * @brief Based on the file name , selects which operation to add as the output/ which to add to the root scope with.
 */
tensorflow::Output ChooseOutputType(tensorflow::Scope& root,const std::string& file_name,const tensorflow::Output& file_reader){
    const int wanted_channels = 3;//Default RGB Image.
    tensorflow::Output image_reader;
    if (tensorflow::str_util::EndsWith(file_name, ".png")) {
        image_reader = tensorflow::ops::DecodePng(root.WithOpName("png_reader"), file_reader,
                                 tensorflow::ops::DecodePng::Channels(wanted_channels));
    } else if (tensorflow::str_util::EndsWith(file_name, ".bmp")) {
        image_reader = tensorflow::ops::DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
    } else {
        image_reader = tensorflow::ops::DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                                  tensorflow::ops::DecodeJpeg::Channels(wanted_channels));
    }
    return image_reader;
}

/**
 * @brief FormatOpsForSizeRestrictedGraphs
 * @param root Tensorflow root scope object. This is what you have to append operations to for your graph to inevitably run on.
 * In this case we're taking a root scope object, and adding the formatting operations of casting the tensor to float, expanding the dimensions,
 * resizing it to the image specs, and performing normalzation through the div op object.
 * @param image_reader current output. Outputs are chained together in the graph that you are construction with operations.
 * @param output_name outputname of the "graph" being created.
 * @param input_height Height to change the image tensor to.
 * @param input_width Width to change the image tensor to.
 * @param input_mean Typically unused.
 * @param input_std standard deviation of the image. Typically its 255.
 */
void FormatOpsForSizeRestrictedGraphs(tensorflow::Scope& root,const tensorflow::Output& image_reader,
                           const std::string output_name,const int input_height,
                           const int input_width,const float input_mean,const float input_std){
    auto float_caster =
            tensorflow::ops::Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
    auto dims_expander = tensorflow::ops::ExpandDims(root, float_caster, 0);
    auto resized = tensorflow::ops::ResizeBilinear(
                root, dims_expander,
                tensorflow::ops::Const(root.WithOpName("size"), {input_height, input_width}));
    // Subtract the mean and divide by the scale.
    tensorflow::ops::Div(root.WithOpName(output_name), tensorflow::ops::Sub(root, resized, {input_mean}),
    {input_std});
}

/**
 * @brief ReadTensorFromImageFile : Will read an image file into an appropriate tensor. This is just an example
 * for an image size depend graph, such as a cnn. Those who don't need restricited image sizes should create their own.
 * @param file_name Path to file.
 * @param input_height Height to change the image tensor to.
 * @param input_width Width to change the image tensor to.
 * @param input_mean Typically unused.
 * @param input_std standard deviation of the image. Typically its 255.
 * @return Tensor representing the loaded image in a tensorflow::Tensor(DT_Float, (3,input_heigh,input_width)) (CHW notation.)
 */
std::vector<tensorflow::Tensor> ReadTensorFromImageFile(const std::string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std) {
    auto root = tensorflow::Scope::NewRootScope();
    std::string input_name = "file_reader";
    std::string output_name = "normalized";
    // read file_name into a tensor named input
    tensorflow::Tensor input{ReadFileIntoTensor(file_name)};
    // use a placeholder to read input data
    auto file_reader =
            tensorflow::ops::Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);
    auto  inputs = CreateTensorInput("input",input);
    tensorflow::Output image_reader{ChooseOutputType(root,file_name,file_reader)};
    FormatOpsForSizeRestrictedGraphs(root,image_reader,output_name,input_height,input_width,input_mean,input_std);
    return SessionRun(root,output_name,inputs);
}


/**
 * Reads the ml graph from the graph_file_name into the provided session object.
 */
void LoadGraph(const std::string& graph_file_name,
               std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    CheckStatus(ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def));
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    CheckStatus( (*session)->Create(graph_def));
}


/**
 * @brief maxcord Example of parsing a tensor object. This pattern may be useful for CNN graphs in particular.
 * Alternatives could include creating a map object and mapping defined classes.
 * @param tensors
 * @return Pair of the highest confidence class index, and its confidence.
 */
std::pair<int,float> maxcord(const std::vector<tensorflow::Tensor>& tensors){
    auto flat_tensor = tensors.at(0).flat<float>();//underlying eigen tensor
    std::vector<float> vals;
    vals.reserve(static_cast<size_t>(flat_tensor.size()));//eigen tensor size type is technically not a size_t. but can be converted..
    for (int i=0;i<flat_tensor.size();i++){
        vals.emplace_back(flat_tensor(i));
    }
    auto maxy = std::max_element(vals.begin(),vals.end());
    return {int(std::distance(vals.begin(),maxy)),*maxy};
}

/**
 * @brief The session_wrapper class: Just takes their pointer object and wraps it, allowing for it to be a loaded_graph object.
 */
class LoadedGraph{
public:
    LoadedGraph(const std::string& location){
        LoadGraph(location, &sess);
    }

    std::vector<tensorflow::Tensor> run(std::string input_layer,std::string output_layer,tensorflow::Tensor input){
        std::vector<tensorflow::Tensor> results;
        CheckStatus(sess->Run({{input_layer, input}},{output_layer}, {}, &results));
        return results;
    }

private:
    std::unique_ptr<tensorflow::Session> sess{};
};

int main() {
    std::string image = "/home/cameron/eclipse-workspace/tensorflow_cpp_wrapper/data/grace_hopper.jpg";
    std::string graph =
            "/home/cameron/eclipse-workspace/tensorflow_cpp_wrapper/data/inception_v3_2016_08_28_frozen.pb";
    int input_width = 299;
    int input_height = 299;
    float input_mean = 0;
    float input_std = 255;
    std::string input_layer = "input";
    std::string output_layer = "InceptionV3/Predictions/Reshape_1";
    std::string root_dir = "";
    LoadedGraph sessy(std::move(graph));

    // Get the image from disk as a float array of numbers, resized and normalized
    // to the specifications the main graph expects.
     std::vector<tensorflow::Tensor> resized_tensors =
            ReadTensorFromImageFile(image, input_height, input_width, input_mean,
                                    input_std);

    const tensorflow::Tensor& resized_tensor = resized_tensors[0];

    // Actually run the image through the model.
    std::vector<tensorflow::Tensor> outputs= sessy.run(input_layer,output_layer,resized_tensor);
    //highest val should be 653
    std::cout<<"Class: "<<maxcord(outputs).first<<"\nConfidence: "<<maxcord(outputs).second<<std::endl;

    return 0;
}

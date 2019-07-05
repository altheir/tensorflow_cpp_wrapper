# tensorflow_cpp_wrapper
Library made to demonstrate use of models in c++ using the tensorflow c++ api. 

Useable outside of tensorflow repo, and with cmake.

Tested with tensorflow 1.10.1.

Build the python package first.
Then build the C++ package. "bazel build -c opt //tensorflow:libtensorflow_cc.so"

I suggest locating the install locations and moving them into their own directory for ease of access/ not having it in bazel cache.

Keep the includes with Tensorflow/*
Altering this pattern will break their includes.


See: 
https://github.com/tensorflow/tensorflow/tree/v1.10.1/tensorflow/examples/label_image

Which is what this wrapper is inspired from.Follow the instructions on how to get the model/image used for demonstration.

For additional information regarding tensorflow see the info_about_tensorflow.md . 

Best of luck.

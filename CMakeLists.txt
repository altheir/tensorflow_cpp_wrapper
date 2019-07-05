cmake_minimum_required(VERSION 3.10)
project(tensorflow_cpp_wrapper)

#you will probobly have to modify this to find your install location. I suggest making a tensorflow_includes file and moving the files out of the cache and into there.
include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	"/usr/local/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/projects/inlcude/*.cpp"
	"$ENV{HOME}/tensorflow_includes/bin/tensorflow/"
	"$ENV{HOME}/tensorflow/"
        "$ENV{HOME}/.cache/bazel/abdd949aa584727ffc1bcb44495f14e9/external/com_google_absl/"
	"$ENV{HOME}/tensorflow/"
	"$ENV{HOME}/tensorflow_includes/bin/"
	"$ENV{HOME}/tensorflow_includes/genfiles/"
        "$ENV{HOME}/.local/lib/python3.6/site-packages/tensorflow/include/" #This is explicitly for eigen. If you can't find it on your system try creating the tensorflow library from instructions, in addition to the one specifically for the c++ version.
	"$ENV{HOME}/tensorflow_includes/google"
        "$ENV{HOME}/.cache/bazel/abdd949aa584727ffc1bcb44495f14e9/external/protobuf_archive/src/"
        "$ENV{HOME}/.cache/bazel/abdd949aa584727ffc1bcb44495f14e9/execroot/org_tensorflow/bazel-out/k8-opt/genfiles/"
)

link_directories(
	"/usr/lib"
	"/usr/local/lib"
	"/home/bin/tensorflow/"
	"$ENV{HOME}/tensorflow_includes/bin/tensorflow/"
	"/tensorflow_includes/bin/"
        "$ENV{HOME}/.cache/bazel/abdd949aa584727ffc1bcb44495f14e9/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/"
)

set(CMAKE_CXX_STANDARD 14) #tensorflow doesn't really support C++17 unfortunately. It seems to cause issues with abseil string_view.
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


file(GLOB_RECURSE SOURCES
	"${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)


set(OUTSIDELIBS
	"tensorflow_cc"
	"tensorflow_framework"
)

add_executable(tensorflow_cpp_wrapper ${SOURCES})
target_link_libraries(tensorflow_cpp_wrapper ${OURLIBS} ${OUTSIDELIBS})
SET_TARGET_PROPERTIES(tensorflow_cpp_wrapper PROPERTIES LINKER_LANGUAGE CXX)

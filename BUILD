# Description:
# TensorFlow Lite Example Label wav.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/contrib/lite:build_def.bzl", "tflite_linkopts")

tf_cc_binary(
    name = "label_wav",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "ringbuf.h",
        "ringbuf.cc",
        "label_wav.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":wav_helpers",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:string_util",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "wav_helpers",
    srcs = ["wav_helpers.cc"],
    hdrs = [
        "wav_helpers.h",
        "label_wav.h",
    ],
    deps = [
        "//tensorflow/contrib/lite:builtin_op_data",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:schema_fbs_version",
        "//tensorflow/contrib/lite:string",
        "//tensorflow/contrib/lite:string_util",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
        "//tensorflow/contrib/lite/schema:schema_fbs",
    ],
)


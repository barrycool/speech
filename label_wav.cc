/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)
#include <sys/ipc.h>
#include <sys/shm.h>

#include "ringbuf.h"
#include <fcntl.h>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/optional_debug_tools.h"
#include "tensorflow/contrib/lite/string_util.h"

#include "tensorflow/contrib/lite/examples/label_wav/wav_helpers.h"
#include "tensorflow/contrib/lite/examples/label_wav/get_top_n.h"

#define SHM_BUF_SIZE (96 * 1024)

#define LOG(x) std::cerr

namespace tflite {
namespace label_wav {

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
TfLiteStatus ReadLabelsFile(const string& file_name,
                            std::vector<string>* result,
                            size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    LOG(FATAL) << "Labels file " << file_name << " not found\n";
    return kTfLiteError;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}

void PrintProfilingInfo(const profiling::ProfileEvent* e, uint32_t op_index,
                        TfLiteRegistration registration) {
  // output something like
  // time (ms) , Node xxx, OpCode xxx, symblic name
  //      5.352, Node   5, OpCode   4, DEPTHWISE_CONV_2D

  LOG(INFO) << std::fixed << std::setw(10) << std::setprecision(3)
            << (e->end_timestamp_us - e->begin_timestamp_us) / 1000.0
            << ", Node " << std::setw(3) << std::setprecision(3) << op_index
            << ", OpCode " << std::setw(3) << std::setprecision(3)
            << registration.builtin_code << ", "
            << EnumNameBuiltinOperator(
                   static_cast<BuiltinOperator>(registration.builtin_code))
            << "\n";
}

void RunInference(Settings* s) {
  if (!s->model_name.c_str()) {
    LOG(ERROR) << "no model file name\n";
    exit(-1);
  }

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
  if (!model) {
    LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
    exit(-1);
  }
  LOG(INFO) << "Loaded model " << s->model_name << "\n";
  model->error_reporter();
  LOG(INFO) << "resolved reporter\n";

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
    exit(-1);
  }

  interpreter->UseNNAPI(s->accel);

  if (s->verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
    LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
    LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point << "\n";
    }
  }

  if (s->number_of_threads != -1) {
    interpreter->SetNumThreads(s->number_of_threads);
  }

  uint32_t sample_count;
  uint16_t channel_count;
  uint32_t sample_rate = 16000;

  int input = interpreter->inputs()[0];
  if (s->verbose) LOG(INFO) << "input: " << input << "\n";

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  if (s->verbose) {
    LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

  if (s->verbose) PrintInterpreterState(interpreter.get());

  std::vector<string> labels;
  size_t label_count;

  if (ReadLabelsFile(s->labels_file_name, &labels, &label_count) != kTfLiteOk)
	  exit(-1);

  /*float *raw_data = new float[16000 * 8];*/
  float *input_buf = interpreter->typed_tensor<float>(input);

  /*read_wav(s->input_wav_name, raw_data, &sample_count,
		  &channel_count, &sample_rate, s);

  read_wav(s->input_wav_name_no, raw_data + 16000, &sample_count,
		  &channel_count, &sample_rate, s);

  read_wav(s->input_wav_name_on, raw_data + 16000 * 2, &sample_count,
		  &channel_count, &sample_rate, s);
  
  read_wav(s->input_wav_name_off, raw_data + 16000 * 3, &sample_count,
		  &channel_count, &sample_rate, s);

  read_wav(s->input_wav_name_test, raw_data, &sample_count,
		  &channel_count, &sample_rate, s);*/

  uint32_t clip_duration_ms = 1000;
  const uint32_t clip_duration_samples = (clip_duration_ms * sample_rate) / 1000;
  uint32_t clip_stride_ms = 100;
  const uint32_t clip_stride_samples = (clip_stride_ms * sample_rate) / 1000;
  const uint32_t average_window_duration_ms = 500;
  const uint32_t average_window_duration_sample = (average_window_duration_ms * sample_rate) / 1000;
  const float* start = raw_data;
  const float* end = start + 16000 * 8;
  const float detection_threshold = 0.7;
  const float *last_start = raw_data;
  int last_index = -1;
  uint32_t current_window_duration_sample;

  int shmid = shmget(ftok("/bin/bash", 0), SHM_BUF_SIZE, IPC_CREAT | 0644);
  if (shmid == -1)
  {
	  printf("%s\n", strerror(errno));
	  return 1;
  }

  uint8_t *buf = shmat(shmid, NULL, 0);
  if (buf == NULL)
  {
	  printf("%s\n", strerror(errno));
	  return 2;
  }

  ringbuf_t audio_data =  ringbuf_get(buf);

  int16_t *raw_data = new int16_t[16000];

  while (1)
  {
	  if (ringbuf_bytes_used(audio_data) < clip_duration_samples * 2)
	  {
		  usleep(50000);
		  continue;
	  }

	  ringbuf_copy_data(raw_data, audio_data, clip_duration_samples * 2);
	  ringbuf_skip_buf(raw_data, clip_stride_samples * 2);

	  decode_audio_data(raw_data, clip_duration_samples, input_buf);		

	  interpreter->Invoke();

	  const float threshold = 0.001f;

	  std::vector<std::pair<float, int>> top_results;

	  int output = interpreter->outputs()[0];
	  TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
	  // assume output dims to be something like (1, 1, ... ,size)
	  auto output_size = output_dims->data[output_dims->size - 1];
	  switch (interpreter->tensor(output)->type) {
		  case kTfLiteFloat32:
			  get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
					  s->number_of_results, threshold, &top_results, true);
			  break;
		  case kTfLiteUInt8:
			  get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
					  output_size, s->number_of_results, threshold,
					  &top_results, false);
			  break;
		  default:
			  LOG(FATAL) << "cannot handle output type "
				  << interpreter->tensor(input)->type << " yet";
			  exit(-1);
	  }

	  current_window_duration_sample = start - last_start;

	  if (top_results[0].first > detection_threshold) {
		  /*if (last_index != top_results[0].second || 
			(current_window_duration_sample >= average_window_duration_sample ||
			current_window_duration_sample == 0)) */
		  {

			  /*LOG(INFO) << current_window_duration_sample << "\n";
				start += average_window_duration_sample - clip_stride_samples;
				last_start = start;
				last_index = top_results[0].second;*/

			  for (const auto& result : top_results) {
				  const float confidence = result.first;
				  const int index = result.second;
				  LOG(INFO) << confidence << ": " << index << " " << labels[index] << "\n";
			  }
		  }
	  }
  }
}

void display_usage() {
  LOG(INFO) << "label_wav\n"
            << "--accelerated, -a: [0|1], use Android NNAPI or not\n"
            << "--count, -c: loop interpreter->Invoke() for certain times\n"
            << "--input_mean, -b: input mean\n"
            << "--input_std, -s: input standard deviation\n"
            << "--wav, -w: wav_name.wav\n"
            << "--labels, -l: labels for the model\n"
            << "--tflite_model, -m: model_name.tflite\n"
            << "--profiling, -p: [0|1], profiling or not\n"
            << "--num_results, -r: number of results to show\n"
            << "--threads, -t: number of threads\n"
            << "--verbose, -v: [0|1] print more information\n"
            << "\n";
}

int Main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"accelerated", required_argument, nullptr, 'a'},
        {"count", required_argument, nullptr, 'c'},
        {"verbose", required_argument, nullptr, 'v'},
        {"wav", required_argument, nullptr, 'w'},
        {"labels", required_argument, nullptr, 'l'},
        {"tflite_model", required_argument, nullptr, 'm'},
        {"profiling", required_argument, nullptr, 'p'},
        {"threads", required_argument, nullptr, 't'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"num_results", required_argument, nullptr, 'r'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "a:b:c:f:w:l:m:p:r:s:t:v:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        s.accel = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'b':
        s.input_mean = strtod(optarg, nullptr);
        break;
      case 'c':
        s.loop_count =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'w':
        s.input_wav_name = optarg;
        break;
      case 'l':
        s.labels_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 'p':
        s.profiling =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'r':
        s.number_of_results =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      case 'v':
        s.verbose =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'h':
      case '?':
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
      default:
        exit(-1);
    }
  }
  RunInference(&s);
  return 0;
}

}  // namespace label_image
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::label_wav::Main(argc, argv);
}

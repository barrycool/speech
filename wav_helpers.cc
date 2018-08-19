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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include <unistd.h>  // NOLINT(build/include_order)

#include "tensorflow/contrib/lite/examples/label_wav/wav_helpers.h"

#define LOG(x) std::cerr

namespace tflite {
namespace label_wav {

inline float Int16SampleToFloat(int16_t data) {
  constexpr float kMultiplier = 1.0f / (1 << 15);
  return data * kMultiplier;
}

void read_wav(const std::string& input_bmp_name, float *out, uint32_t* sample_count,
                              uint16_t* channel_count, uint32_t* sample_rate, Settings* s) {
  int begin, end;

  std::ifstream file(input_bmp_name, std::ios::in | std::ios::binary);
  if (!file) {
    LOG(FATAL) << "input file " << input_bmp_name << " not found\n";
    exit(-1);
  }

  begin = file.tellg();
  file.seekg(0, std::ios::end);
  end = file.tellg();
  size_t len = end - begin;

  if (s->verbose) LOG(INFO) << "len: " << len << "\n";

  std::vector<uint8_t> wav_bytes(len);
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(wav_bytes.data()), len);

  *channel_count = wav_bytes[0x16];
  *sample_rate = wav_bytes[0x18] | 
	  (wav_bytes[0x19] << 8) |
	  (wav_bytes[0x1A] << 16) |
	  (wav_bytes[0x1B] << 24);

  uint16_t bit_width = wav_bytes[0x22] / 8;

  if (wav_bytes[0x10] == 16) {
	  *sample_count = wav_bytes[0x28] | 
		  (wav_bytes[0x29] << 8) |
		  (wav_bytes[0x2A] << 16) |
		  (wav_bytes[0x2B] << 24);
  }
  else if (wav_bytes[0x10] == 18) {
	  *sample_count = wav_bytes[0x2A] | 
		  (wav_bytes[0x2B] << 8) |
		  (wav_bytes[0x2C] << 16) |
		  (wav_bytes[0x2D] << 24);
  }
  else {
    LOG(FATAL) << "unknow fmt chunk length:"<< wav_bytes[0x10] << "\n";
    exit(-2);
  }
  *sample_count /= bit_width;
  
  if (s->verbose)
	LOG(INFO) << "sample_count, channel_count, bit_width, sample_rate: " 
		<< *sample_count << ", " 
		<< *channel_count << ", " 
		<< bit_width << ", "
		<< *sample_rate << "\n";

  uint8_t * wav_data = wav_bytes.data();
  wav_data += 0x2C;

  int16_t *single_channel_value = (int16_t*)(wav_data);

  for(uint32_t i = 0; i < *sample_count; i++) {
	out[i] = Int16SampleToFloat(single_channel_value[i]);
  }
}

}  // namespace label_wav
}  // namespace tflite

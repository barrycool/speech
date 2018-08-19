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

#ifndef TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_WAV_HELPERS_H_
#define TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_WAV_HELPERS_H_

#include "tensorflow/contrib/lite/examples/label_wav/label_wav.h"

namespace tflite {
namespace label_wav {

void read_wav(const std::string& input_bmp_name, float *out, uint32_t* sample_count,
                              uint16_t* channel_count, uint32_t* sample_rate, Settings* s);

}  // namespace label_wav
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_WAV_HELPERS_H

#ifndef TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_WAV_H
#define TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_WAV_H

#include <alsa/asoundlib.h>
#include "tensorflow/contrib/lite/string.h"

namespace tflite {
namespace label_wav {

struct Settings {
  bool verbose = true;
  bool accel = false;
  bool input_floating = false;
  bool profiling = false;
  int loop_count = 1;
  float input_mean = 127.5f;
  float input_std = 127.5f;
  string model_name = "/home/barry/workspace/speech_command/speech_recognition.tflite";
  string labels_file_name = "/home/barry/workspace/speech_command/speech_commands_train/conv_labels.txt";
  string input_layer_type = "float";
  string input_wav_name = "/home/barry/workspace/speech_command/speech_dataset/yes/012c8314_nohash_0.wav";
  string input_wav_name_no = "/home/barry/workspace/speech_command/no.wav";
  string input_wav_name_on = "/home/barry/workspace/speech_command/on.wav";
  string input_wav_name_off = "/home/barry/workspace/speech_command/off.wav";
  string input_wav_name_test = "/home/barry/workspace/speech_command/test.wav";

  int number_of_threads = 4;
  int number_of_results = 1;
};

}  // namespace label_image
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_LABEL_WAV_H

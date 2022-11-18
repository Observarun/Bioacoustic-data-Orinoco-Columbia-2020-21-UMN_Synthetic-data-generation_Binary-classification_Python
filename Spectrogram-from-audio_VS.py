# Program to generate spectrograms from an acoustic waveform.



# Imports

from pathlib import Path, PurePath

import numpy

from scipy import signal
import matplotlib.pyplot as plt

try:
  from pydub import AudioSegment
except:
  !pip3 install pydub



# Path variables

# Path for directory containing raw audio clips
audio_clips_path = '/path/to/directory/containing/audioclips'

# Path for directory containing spectrograms generated from raw audio clips
spectrograms_path = '/path/to/directory/for/saving/generated/spectrograms'



# Create an AudioSegment instance from the audio file at the path specified.
# The information in the audio file is then used to create a numpy array, which
# is used as an argument in scipy.signal.spectrogram() to generate the
# corresponding spectrogram.

def generate_spectrogram_from_audio(path, stem):

  instance_AS = AudioSegment.from_file(path)
  samplerate = instance_AS.frame_rate
  data_instance_AS = instance_AS.get_array_of_samples()
  data_instance_AS = numpy.array(data_instance_AS)

  samples = numpy.array([])

  channel_samples = []
  for channel in numpy.arange(instance_AS.channels):
    channel_samples.append(data_instance_AS[channel::instance_AS.channels])
  samples = numpy.array(channel_samples, dtype=array_type)

  frequencies, times, Sxx = signal.spectrogram(data_instance_AS, samplerate, scaling='spectrum')

  plt.pcolormesh(times, frequencies, numpy.log10(Sxx[0]), cmap='jet')
  plt.xticks(numpy.arange(0, instance_AS.duration_seconds, step=.2), rotation = (90))
  plt.tick_params(axis='x', which='major', labelsize=7)
  
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  
  plt.savefig(Path(spectrograms_path, stem +'.png'), format='png')
  plt.show()



# Assuming the directory has both .flac and .WAV audio clips.

for path in Path(audio_clips_path).rglob('*.flac'):
  generate_spectrogram_from_audio(path, PurePath(path).stem)
for path in Path(audio_clips_path).rglob('*.WAV'):
  generate_spectrogram_from_audio(path, PurePath(path).stem)

# Alternatively, can use the following in place of the two for-loops above.
# for path in Path(raw_audio_clips_path).glob('*.[fW][lA][aV]*'):
#   generate_spectrogram_from_audio(path, PurePath(path).stem)
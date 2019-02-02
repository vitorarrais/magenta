import os
from magenta.models.performance_rnn import performance_sequence_generator
from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2
import numpy as np
from scipy.io import wavfile
import magenta.music as mm

_DEFAULT_SAMPLE_RATE = 44100

def gen_wav(seq,
            synth,
            sample_rate=_DEFAULT_SAMPLE_RATE):
  """Generates a wav file

  Args:
    sequence: A music_pb2.NoteSequence to synthesize and play.
    synth: A synthesis function that takes a sequence and sample rate as input.
    sample_rate: The sample rate at which to synthesize.
  """
  array_of_floats = synth(seq, sample_rate=sample_rate)
  normalizer = float(np.iinfo(np.int16).max)
  array_of_ints = np.array(
      np.asarray(array_of_floats) * normalizer, dtype=np.int16)
#   f = open('/tmp/wav.wav', 'w')
  wavfile.write('/tmp/wav.wav', sample_rate, array_of_ints)
#   f.close()

# Constants.
BUNDLE_DIR='/Users/vitorarrais/Projects/Repositories/magenta/magenta/models/performance_rnn/'
# BUNDLE_DIR = '/Users/vitorarrais/Projects/magenta/mag/'
MODEL_NAME = 'multiconditioned_performance_with_dynamics'
BUNDLE_NAME = MODEL_NAME + '.mag'
# OUTPUT_DIR = '/Users/vitorarrais/Projects/magenta/mag/generated/perfomance_rnn'

bundle = mm.sequence_generator_bundle.read_bundle_file(
    os.path.join(BUNDLE_DIR, BUNDLE_NAME))
generator_map = performance_sequence_generator.get_generator_map()
generator = generator_map[MODEL_NAME](checkpoint=None, bundle=bundle)
generator.initialize()
generator_options = generator_pb2.GeneratorOptions()
# Higher is more random; 1.0 is default.
generator_options.args['temperature'].float_value = 1.0
generate_section = generator_options.generate_sections.add(
    start_time=0, end_time=30)
sequence = generator.generate(music_pb2.NoteSequence(), generator_options)
# breakpoint()
gen_wav(sequence, mm.midi_synth.fluidsynth)

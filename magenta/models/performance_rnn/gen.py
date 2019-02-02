import ast
import os
import time
import sys

import tensorflow as tf

import magenta
from magenta.models.performance_rnn import performance_model
from magenta.models.performance_rnn import performance_sequence_generator
from magenta.music import constants
from magenta.music.midi_synth import fluidsynth as synth
from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2
from scipy.io import wavfile
import numpy as np

from magenta.models.performance_rnn import performance_rnn_generate as pgen

_DEFAULT_SAMPLE_RATE = 44100
FLAGS = pgen.FLAGS

def writeStdout():
    print('I am in, bitch')

def run(generator):
  if not FLAGS.output_dir:
    tf.logging.fatal('--output_dir required')
    return
  output_dir = os.path.expanduser(FLAGS.output_dir)

  primer_midi = None
  if FLAGS.primer_midi:
    primer_midi = os.path.expanduser(FLAGS.primer_midi)

  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  primer_sequence = None
  if FLAGS.primer_pitches:
    primer_sequence = music_pb2.NoteSequence()
    primer_sequence.ticks_per_quarter = constants.STANDARD_PPQ
    for pitch in ast.literal_eval(FLAGS.primer_pitches):
      note = primer_sequence.notes.add()
      note.start_time = 0
      note.end_time = 60.0 / magenta.music.DEFAULT_QUARTERS_PER_MINUTE
      note.pitch = pitch
      note.velocity = 100
      primer_sequence.total_time = note.end_time
  elif FLAGS.primer_melody:
    primer_melody = magenta.music.Melody(ast.literal_eval(FLAGS.primer_melody))
    primer_sequence = primer_melody.to_sequence()
  elif primer_midi:
    primer_sequence = magenta.music.midi_file_to_sequence_proto(primer_midi)
  else:
    tf.logging.warning(
        'No priming sequence specified. Defaulting to empty sequence.')
    primer_sequence = music_pb2.NoteSequence()
    primer_sequence.ticks_per_quarter = constants.STANDARD_PPQ

  # Derive the total number of seconds to generate.
  seconds_per_step = 1.0 / generator.steps_per_second
  generate_end_time = FLAGS.num_steps * seconds_per_step

  # Specify start/stop time for generation based on starting generation at the
  # end of the priming sequence and continuing until the sequence is num_steps
  # long.
  generator_options = generator_pb2.GeneratorOptions()
  # Set the start time to begin when the last note ends.
  generate_section = generator_options.generate_sections.add(
      start_time=primer_sequence.total_time,
      end_time=generate_end_time)

  if generate_section.start_time >= generate_section.end_time:
    tf.logging.fatal(
        'Priming sequence is longer than the total number of steps '
        'requested: Priming sequence length: %s, Total length '
        'requested: %s',
        generate_section.start_time, generate_end_time)
    return

  for control_cls in magenta.music.all_performance_control_signals:
    if FLAGS[control_cls.name].value is not None and (
        generator.control_signals is None or not any(
            control.name == control_cls.name
            for control in generator.control_signals)):
      tf.logging.warning(
          'Control signal requested via flag, but generator is not set up to '
          'condition on this control signal. Request will be ignored: %s = %s',
          control_cls.name, FLAGS[control_cls.name].value)

  if (FLAGS.disable_conditioning is not None and
      not generator.optional_conditioning):
    tf.logging.warning(
        'Disable conditioning flag set, but generator is not set up for '
        'optional conditioning. Requested disable conditioning flag will be '
        'ignored: %s', FLAGS.disable_conditioning)

  if generator.control_signals:
    for control in generator.control_signals:
      if FLAGS[control.name].value is not None:
        generator_options.args[control.name].string_value = (
            FLAGS[control.name].value)
  if FLAGS.disable_conditioning is not None:
    generator_options.args['disable_conditioning'].string_value = (
        FLAGS.disable_conditioning)

  generator_options.args['temperature'].float_value = FLAGS.temperature
  generator_options.args['beam_size'].int_value = FLAGS.beam_size
  generator_options.args['branch_factor'].int_value = FLAGS.branch_factor
  generator_options.args[
      'steps_per_iteration'].int_value = FLAGS.steps_per_iteration

  tf.logging.debug('primer_sequence: %s', primer_sequence)
  tf.logging.debug('generator_options: %s', generator_options)

  # Make the generate request num_outputs times and save the output as midi
  # files.
  date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
  digits = len(str(FLAGS.num_outputs))
  for i in range(FLAGS.num_outputs):
    generated_sequence = generator.generate(primer_sequence, generator_options)
    array_of_floats = synth(
        generated_sequence, sample_rate=_DEFAULT_SAMPLE_RATE)
    normalizer = float(np.iinfo(np.int16).max)
    array_of_ints = np.array(
        np.asarray(array_of_floats) * normalizer, dtype=np.int16)
    wav_filename = '%s_%s.wav' % (date_and_time, str(i + 1).zfill(digits))
    wav_path = os.path.join(output_dir, wav_filename)
    wavfile.write(wav_path, _DEFAULT_SAMPLE_RATE, array_of_ints)
    sys.stderr.write('\n++++++++++++++++++++++++++++++++++++++++\n')
    sys.stderr.write('\n++++++++++++++++++++++++++++++++++++++++\n')
    sys.stdout.write(wav_filename)
    sys.stderr.write('\n++++++++++++++++++++++++++++++++++++++++\n')
    sys.stderr.write('\n++++++++++++++++++++++++++++++++++++++++\n')

    # midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
    # midi_path = os.path.join(output_dir, midi_filename)
    # magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)

  tf.logging.info('Wrote %d WAV files to %s',
                  FLAGS.num_outputs, output_dir)

def main(unused_argv):

  bundle = pgen.get_bundle()

  config_id = bundle.generator_details.id if bundle else FLAGS.config
  config = performance_model.default_configs[config_id]
  config.hparams.parse(FLAGS.hparams)
  # Having too large of a batch size will slow generation down unnecessarily.
  config.hparams.batch_size = min(
      config.hparams.batch_size, FLAGS.beam_size * FLAGS.branch_factor)

  generator = performance_sequence_generator.PerformanceRnnSequenceGenerator(
      model=performance_model.PerformanceRnnModel(config),
      details=config.details,
      steps_per_second=config.steps_per_second,
      num_velocity_bins=config.num_velocity_bins,
      control_signals=config.control_signals,
      optional_conditioning=config.optional_conditioning,
      checkpoint=pgen.get_checkpoint(),
      bundle=bundle,
      note_performance=config.note_performance)

  if FLAGS.save_generator_bundle:
    bundle_filename = os.path.expanduser(FLAGS.bundle_file)
    if FLAGS.bundle_description is None:
      tf.logging.warning('No bundle description provided.')
    tf.logging.info('Saving generator bundle to %s', bundle_filename)
    generator.create_bundle_file(bundle_filename, FLAGS.bundle_description)
  else:
    run(generator)


def console_entry_point():
    tf.app.run(main)

if __name__ == '__main__':
    console_entry_point()

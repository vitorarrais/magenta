BUNDLE_PATH='/Users/vitorarrais/Projects/Repositories/magenta/magenta/models/melody_rnn/basic_rnn.mag'
CONFIG='basic_rnn'

melody_rnn_generate \
--config=${CONFIG} \
--bundle_file=${BUNDLE_PATH} \
--output_dir=/tmp/melody_rnn/generated \
--num_outputs=10 \
--num_steps=128 \
--primer_melody="[60]"
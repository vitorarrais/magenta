BUNDLE_PATH='/Users/vitorarrais/Projects/Repositories/magenta/magenta/models/performance_rnn/multiconditioned_performance_with_dynamics.mag'
CONFIG='multiconditioned_performance_with_dynamics'

performance_rnn_generate \
--config=${CONFIG} \
--bundle_file=${BUNDLE_PATH} \
--output_dir=/tmp/performance_rnn/generated \
--num_outputs=10 \
--num_steps=3000 \
--primer_melody="[60,62,64,65,67,69,71,72]"
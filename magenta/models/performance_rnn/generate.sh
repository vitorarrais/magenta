#!/bin/bash

BUNDLE_PATH='/Users/vitorarrais/Projects/Repositories/ai-server/python/magenta/magenta/models/performance_rnn/multiconditioned_performance_with_dynamics.mag'
CONFIG='multiconditioned_performance_with_dynamics'

# performance_rnn_generate \
gen \
--config=${CONFIG} \
--bundle_file=${BUNDLE_PATH} \
--output_dir=/tmp/generated \
--num_outputs=1 \
--num_steps=3000 \
--primer_melody='[60,62,64,65,67,69,71,72]'
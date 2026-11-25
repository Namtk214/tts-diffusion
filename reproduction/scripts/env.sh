# Source this on the VM from any directory. Does not run/download anything.
TTS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TTS_EVAL_PYTHON="$TTS_ROOT/.venvs/eval/bin/python"
export TTS_HPSV3_PYTHON="$TTS_ROOT/.venvs/hpsv3/bin/python"
export SATISDIVE_REWARD_PYTHON="$TTS_HPSV3_PYTHON"
export TTS_HPSV2_PYTHON="$TTS_ROOT/.venvs/hpsv2/bin/python"
export TTS_IMAGEREWARD_PYTHON="$TTS_ROOT/.venvs/imagereward/bin/python"
export TTS_GENEVAL_PYTHON="$TTS_ROOT/.venvs/geneval/bin/python"
export PYTHONPATH="$TTS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$TTS_ROOT/models/torch"
export TOKENIZERS_PARALLELISM=false

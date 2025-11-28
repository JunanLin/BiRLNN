import os
import sys

# Resolve repo root dynamically and ensure model modules are importable
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_DIR = os.path.join(REPO, 'code', 'model')
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from evaluation import Evaluator

# Use repo root as base_path. Experiments are under code/experiments and data under data/.
base_path = REPO
stor_dir = os.path.join(REPO, 'evaluation')

for name in [
    'BIMODAL_SELFIES_fixed_1024',
    'BIMODAL_SELFIES_random_1024',
    'FBRNN_SELFIES_fixed_1024',
    'FBRNN_SELFIES_random_1024',
    'ForwardRNN_SELFIES_1024',
    'BackwardRNN_SELFIES_1024',
    'BIMODAL_fixed_1024',
    'BIMODAL_random_1024',
    'FBRNN_fixed_1024',
    'FBRNN_random_1024',
    'ForwardRNN_1024',
    'BackwardRNN_1024'
    ]:
    e = Evaluator(experiment_name=name, base_path=base_path)
    # evaluation of training and validation losses
    e.eval_training_validation(stor_dir=stor_dir)
    # evaluation of sampled molecules
    e.eval_molecule(stor_dir=stor_dir)
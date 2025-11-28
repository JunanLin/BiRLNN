import sys, os

# Resolve project root dynamically
repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Ensure model package imports find local modules
sys.path.insert(0, os.path.join(repo, 'code', 'model'))

from sample import Sampler

# Example script for unconstrained sampling from multiple models
for model in [
    'ForwardRNN_SELFIES_1024',
    'BackwardRNN_SELFIES_1024',
    'BIMODAL_SELFIES_fixed_1024',
    # 'FBRNN_SELFIES_fixed_1024',
    # 'BIMODAL_SELFIES_random_1024',
    # 'FBRNN_SELFIES_random_1024',
    'ForwardRNN_1024',
    'BackwardRNN_1024',
    'BIMODAL_fixed_1024',
    # 'FBRNN_fixed_1024',
    # 'BIMODAL_random_1024',
    # 'FBRNN_random_1024'
]:
    s = Sampler(model, base_path=repo)
    s.sample(fold=[1, 2, 3, 4, 5], N=200)

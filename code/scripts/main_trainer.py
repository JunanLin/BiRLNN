import sys, os
import time
# Resolve project root dynamically
repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Ensure model package imports find local modules
sys.path.insert(0, os.path.join(repo, 'code', 'model'))
from trainer import Trainer

# Comment to train selected experiments only
for exp in [
            'ForwardRNN_SELFIES_1024',
            'BackwardRNN_SELFIES_1024',
            'BIMODAL_SELFIES_fixed_1024',
            'FBRNN_SELFIES_fixed_1024',
            'BIMODAL_SELFIES_random_1024',
            'FBRNN_SELFIES_random_1024',
            'ForwardRNN_1024',
            'BackwardRNN_1024',
            'BIMODAL_fixed_1024', 
            'FBRNN_fixed_1024', 
            'BIMODAL_random_1024',
            'FBRNN_random_1024',
    ]:
# We also provide a few short training datasets and .ini experiments for quick tests.
# You can also create more custom small datasets by sampling from the full ones.
# for exp in ['BackwardRNN_SELFIES_512_quick']:
    print('\n==== Starting training for', exp, '====')
    start_time = time.time()
    t = Trainer(exp, base_path=repo)
    t.cross_validation()
    # t.single_run()
    end_time = time.time()
    print('==== Finished training for', exp, '====')
    print(f"Training time: {(end_time - start_time)/3600:.2f} hours")
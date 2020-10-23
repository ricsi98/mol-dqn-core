from helpers import MorganFingerprintProvider
from mdp import QEDMolEnv, PenalizedLogpEnv, BenchmarkEnv
from model import BootstrappedDQN
from training_utils import train
import sys

MORGAN_FP_SIZE = 1024
MORGAN_FP_RADIUS = 2
N_EPISODES = 2000
EPISODE_LENGTH = 45
BOOTSTRAP_HEADS = 5

if __name__ == '__main__':
    agent = BootstrappedDQN(MORGAN_FP_SIZE, MORGAN_FP_SIZE, n_heads=BOOTSTRAP_HEADS)
    if sys.argv[1] == 'qed':
        env = QEDMolEnv({'C', 'O', 'N', 'Cl'}, max_steps=EPISODE_LENGTH)
    elif sys.argv[1] == 'penalized_logp':
        env = PenalizedLogpEnv({'C', 'O', 'N', 'Cl'}, max_steps=EPISODE_LENGTH)
    elif sys.argv[1] == 'benchmark':
        env = BenchmarkEnv({'C', 'O', 'N', 'Cl'}, max_steps=EPISODE_LENGTH)
    else:
        print('BAD ARGS')
        sys.exit(1)
    mfp = MorganFingerprintProvider(MORGAN_FP_SIZE, MORGAN_FP_RADIUS)

    molecule_pool = list(train(env, agent, mfp, N_EPISODES, EPISODE_LENGTH))

    with open('OUT_MOLS_%s.smiles' % sys.argv[1], 'w') as f:
        for m in molecule_pool:
            if m is None: continue
            f.write('%s\n' % m)
    print('DONE')
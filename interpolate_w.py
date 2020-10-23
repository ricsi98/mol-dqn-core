from helpers import MorganFingerprintProvider
from mdp import SimVsQedEnv
from model import BootstrappedDQN
import numpy as np
import json
import datetime
import threading
from training_utils import train


MORGAN_FP_SIZE = 1024
MORGAN_FP_RADIUS = 2
N_EPISODES = 2
EPISODE_LENGTH = 45
BOOTSTRAP_HEADS = 5
TARGET_MOL = 'CC(=O)OC1=CC=CC=C1C(=O)O'

if __name__ == '__main__':

    Mols = {}
    threads = []

    for w in np.linspace(0, 1, 6):
        agent = BootstrappedDQN(MORGAN_FP_SIZE, MORGAN_FP_SIZE, n_heads=BOOTSTRAP_HEADS)
        mfp = MorganFingerprintProvider(MORGAN_FP_SIZE, MORGAN_FP_RADIUS)
        env = SimVsQedEnv(TARGET_MOL, mfp, w)

        def t_func(w, agent, mfp, env, Mols):
            print(str(datetime.datetime.now()), ' training with w=%.1f' % w)
            molecules = list(train(env, agent, mfp, N_EPISODES, EPISODE_LENGTH))
            Mols[w] = molecules
            print(str(datetime.datetime.now()), ' training with w=%.1f ENDED' % w)

        t = threading.Thread(target=t_func, args=(w, agent, mfp, env, Mols,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    
    with open('interpolate_w_output.json', 'w') as f:
        data = json.dumps(Mols)
        f.write(data)
            

    print('DONE')
from helpers import MorganFingerprintProvider
from mdp import QEDMolEnv, PenalizedLogpEnv, BenchmarkEnv
from model import BootstrappedDQN
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

    molecule_pool = set()
    bestreward = 0

    for e in range(N_EPISODES):
        env.initialize()
        state = mfp.get_fingerprint(env.state)
        for m in env.get_path():
            molecule_pool.add(m)

        for t in range(EPISODE_LENGTH):

            smiles_actions = list(env.get_valid_actions())
            actions = [mfp.get_fingerprint(m) for m in smiles_actions]
            action = agent.select_action(state, actions)

            result = env.step(smiles_actions[action])

            state_ = mfp.get_fingerprint(result.state)
            reward = result.reward
            done = result.terminated
            actions_ = [mfp.get_fingerprint(m) for m in env.get_valid_actions()]

            agent.remember((state, action, actions, actions_, state_, reward, done))

            if reward > bestreward:
                bestreward = reward

            if done: break

        agent.learn(100)
        if e % 100 == 0:
            print(e, bestreward)

    with open('OUT_MOLS_%s.smiles' % sys.argv[1], 'w') as f:
        for m in molecule_pool:
            if m is None: continue
            f.write('%s\n' % m)
    print('DONE', bestreward)
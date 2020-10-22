from helpers import MorganFingerprintProvider
from mdp import QEDMolEnv
from model import BootstrappedDQN

MORGAN_FP_SIZE = 1024
MORGAN_FP_RADIUS = 2
N_EPISODES = 2000
EPISODE_LENGTH = 45

if __name__ == '__main__':
    agent = BootstrappedDQN(MORGAN_FP_SIZE, MORGAN_FP_SIZE, max_steps=EPISODE_LENGTH)
    env = QEDMolEnv({'C', 'O', 'N', 'Cl'})
    mfp = MorganFingerprintProvider(MORGAN_FP_SIZE, MORGAN_FP_RADIUS)

    molecule_pool = set()

    for e in range(N_EPISODES):
        env.initialize()
        state = mfp.get_fingerprint(env.state)
        if len(env.get_path()) > 0:
            for m in env.get_path():
                molecule_pool.add(m)

        for t in range(EPISODE_LENGTH):

            actions = [mfp.get_fingerprint(m) for m in env.get_valid_actions()]
            action = agent.select_action(state, actions)

            result = env.step(actions[action])

            state_ = result.state
            reward = result.reward
            done = result.terminated
            actions_ = [mfp.get_fingerprint(m) for m in env.get_valid_actions()]

            agent.remember((state, action, actions, actions_, reward, done))

            if done: break

        agent.learn(100)
def train(env, agent, fingerprint_provider, n_episodes, episode_length, episode_callback=None):
    molecule_pool = set()
    bestreward = 0

    for e in range(n_episodes):
        for m in env.get_path():
            molecule_pool.add(m)
        env.initialize()
        state = fingerprint_provider.get_fingerprint(env.state)

        for t in range(episode_length):

            smiles_actions = list(env.get_valid_actions())
            actions = [fingerprint_provider.get_fingerprint(m) for m in smiles_actions]
            action = agent.select_action(state, actions)

            result = env.step(smiles_actions[action])

            state_ = fingerprint_provider.get_fingerprint(result.state)
            reward = result.reward
            done = result.terminated
            actions_ = [fingerprint_provider.get_fingerprint(m) for m in env.get_valid_actions()]

            agent.remember((state, action, actions, actions_, state_, reward, done))

            if reward > bestreward:
                bestreward = reward

            if done: break
            
        if episode_callback is not None:
            episode_callback(reward)

        agent.learn(100)
    
    return molecule_pool
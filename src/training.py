import numpy as np
from collections import deque
import pygame


from enums.algorithm import Algorithm
from world_for_deep_q_learning import BombermanEnv
from dqn import DQN

if __name__ == '__main__':
    LEARNING_RATE = 1e-4
    LEARNING_RATE_DECAY = 0.99
    EXPLORATION_DECAY = 0.95
    GAMMA = 0.975
    UPDATE_TARGET_EVERY = 10

    BATCH_SIZE = 128
    EPISODES = 101

    pygame.display.init()
    INFO = pygame.display.Info()
    TILE_SIZE = int(INFO.current_h * 0.035)
    WINDOW_SIZE = (13 * TILE_SIZE, 13 * TILE_SIZE)
    player_alg = Algorithm.PLAYER
    en1_alg = Algorithm.DIJKSTRA
    en2_alg = Algorithm.DFS
    en3_alg = Algorithm.DIJKSTRA
    show_path = True
    surface = pygame.display.set_mode(WINDOW_SIZE)

    env = BombermanEnv(surface, show_path, player_alg, en1_alg, en2_alg, en3_alg, TILE_SIZE)
    # env = BombermanEnv(None, None, Algorithm.PLAYER, Algorithm.DFS, Algorithm.DIJKSTRA, Algorithm.DFS, None)

    # model_path = "models/100.h5"

    agent = DQN(
        state_shape=env.stateShape,
        action_size=env.actionSpaceSize,
        batch_size=BATCH_SIZE,
        learning_rate_max=LEARNING_RATE,
        learning_rate_decay=LEARNING_RATE_DECAY,
        exploration_decay=EXPLORATION_DECAY,
        gamma=GAMMA
    )
    # agent.load(model_path)
    agent.save(f'models/-1.h5')

    state = env.reset()
    state = np.expand_dims(state, axis=0)

    most_recent_losses = deque(maxlen=BATCH_SIZE)

    log = []

    # fill up memory before training starts
    while agent.memory.length() < BATCH_SIZE:
        action = agent.act(state)
        next_state, reward, done, game_info = env.step(env.actionSpace[action])
        next_state = np.expand_dims(next_state, axis=0)     # Change state shape from (Height, Width) to (Height, Width, 1)
        agent.remember(state, action, reward, next_state, done)
        state = next_state

    for e in range(0, EPISODES):
        state = env.reset()
        state = np.expand_dims(state, axis=0)
        done = False
        step = 0
        ma_loss = None

        while not done:
            action = agent.act(state)
            next_state, reward, done, game_info = env.step(env.actionSpace[action])
            next_state = np.expand_dims(next_state, axis=0)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            step += 1

            loss = agent.replay(episode=e)
            most_recent_losses.append(loss)
            ma_loss = np.array(most_recent_losses).mean()

            if loss != None:
                print(f"Step: {step}. -- Loss: {loss}", end="          \r")

            if done:
                print(f"Episode {e}/{EPISODES-1} completed with {step} steps. LR: {agent.learning_rate:.6f}. EP: {agent.exploration_rate:.2f}. MA loss: {ma_loss:.6f}")
                break

        log.append([e, step, agent.learning_rate, agent.exploration_rate, ma_loss])

        agent.save(f'models/{e}.h5')
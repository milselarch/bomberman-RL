import gym

env = gym.make("CartPole-v1")
state, info = env.reset()

"""
state is a numpy float array with shape (4,)
info is a dictionary
"""

print("Initial state:", state)
print("state shape", state.shape)
print("Info:", info)

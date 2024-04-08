from trainer import Trainer
# from memory_profiler import profile as profile_memory

try:
    from config import incentives
except ImportError as e:
    print('No config module found. '
          'Copy config.example.py to config.py')
    raise e

if __name__ == '__main__':
    # env = BombermanEnv(None, None, Algorithm.PLAYER, Algorithm.DFS, Algorithm.DIJKSTRA, Algorithm.DFS, None)
    # model_path = "models/100.h5"
    trainer = Trainer(incentives=incentives)
    trainer.train()

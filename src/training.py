from trainer import Trainer

try:
    from incentives import incentives
except ImportError as e:
    print('No incentives module found. '
          'Copy incentives.example.py to incentives.py')
    raise e

if __name__ == '__main__':
    # env = BombermanEnv(None, None, Algorithm.PLAYER, Algorithm.DFS, Algorithm.DIJKSTRA, Algorithm.DFS, None)

    # model_path = "models/100.h5"
    trainer = Trainer(incentives=incentives)
    trainer.train()
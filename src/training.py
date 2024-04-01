from trainer import Trainer

if __name__ == '__main__':
    # env = BombermanEnv(None, None, Algorithm.PLAYER, Algorithm.DFS, Algorithm.DIJKSTRA, Algorithm.DFS, None)

    # model_path = "models/100.h5"
    trainer = Trainer()
    trainer.train()
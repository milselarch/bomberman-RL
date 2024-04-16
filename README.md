# bomberman-RL
Learning to play bomberman using reinforcement learning with PyTorch

### Setup Instructions
1. Create python environment  
`python3.10 -m venv venv`
2. Activate python environment  
`source venv/bin/activate`
3. Install libraries  
`python -m pip install -r requirements.txt`

### Run Instructions
1. Activate environment and go to `src` directory  
`source venv/bin/activate && cd src`
2. Create config file  
`cp config.example.py config.py`
3. Run training  
`python training.py`  

Model weight checkpoints are stored in the `saves` directory, 
while tensorboard logs are stored in `logs` directory. The checkpoints / 
tensorboard logs for both will be created in a subdirectory in `saves` / `logs`
named `ddqn-<timestamp>` where `timestamp` is timestamp that `training.py`
was executed, in `YYMMDD-hhmm` format.

### View tensorboard logs
1. `tensorboard --logdir <LOG_DIR>`  
example: 
`tensorboard --logdir logs/ddqn-240401-2155`
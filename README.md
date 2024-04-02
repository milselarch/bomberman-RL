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

### View tensorboard logs
1. `tensorboard --logdir <LOG_DIR>`  
example: 
`tensorboard --logdir src/logs/ddqn-240401-2155`
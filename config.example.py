from game.Incentives import Incentives
from TrainingSettings import TrainingSettings

# reward engineering incentives
incentives = Incentives(
    DEATH_PENALTY=-1,
    BOMB_DEATH_DISTANCE_PENALTY=-2,
    DESTROY_BOX_REWARD=1,
    DESTROY_ENEMY_REWARD=10,
    IN_BOMB_RANGE_PENALTY=0,
    FIRST_CORNER_BOMB_PENALTY=-10
)

# other training hyperparameters
training_settings = TrainingSettings(
    IS_MANUAL_CONTROL=False,
    IS_CHECKING_ILLEGAL_ACTION=True,
    IS_PRESET_GRID=False,
    POOL_TRANSITIONS=True
)
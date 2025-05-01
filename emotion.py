from enum import Enum

class Emotion(str, Enum):
    HAPPY = "HAPPY"
    SAD = "SAD"
    ANGRY = "ANGRY"
    NEUTRAL = "NEUTRAL"
    SURPRISED = "SURPRISED"
# ATRP-RL: Controlling controlled polymerization via ATRP

## ATRP environment: OpenAI-gym environment of an ATRP controller.
Minimal usage:
```
from atrp_env import ATRPEnv
env = ATRPEnv()
state = env.reset()
state_next, reward, done, info = env.step(0)
```


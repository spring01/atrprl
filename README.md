# ATRP-RL: Controlling controlled polymerization via ATRP

## ATRP environment: OpenAI-gym environment of an ATRP controller.
Minimal usage:
```
from atrp_env import ATRPEnv
env = ATRPEnv()
state = env.reset()
new_state = env.action(0)
```


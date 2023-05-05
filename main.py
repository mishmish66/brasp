import brax

import jax.profiler
import jax
from jax import numpy as jnp
from jax import random

from brax.io import image

from envs.fep import brax_fep
from brax import envs

import cv2 as cv
from time import sleep

jax.config.update("jax_platform_name", "cpu")

env = brax_fep()

# env = envs.create(env_name='ant', backend='spring')

state = env.reset()

rand_key = random.PRNGKey(0)


def generate_random_action(key):
    random_actions = random.uniform(
        key, shape=(9,), dtype=jnp.float32, minval=-2.0, maxval=2.0
    )
    new_key = random.split(key)[0]
    return random_actions, new_key


target = jnp.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])

# target = jnp.array([-10, 0, 0, 0, 0, 0, -0.7853, 0.04, 0.04])

for _ in range(int(1e6)):
    random_action, rand_key = generate_random_action(rand_key)
    state = env.step(state, target)
    env.render(state)

jax.profiler.save_device_memory_profile("memory.prof")
pass

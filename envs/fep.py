import brax
from brax.io import mjcf
from brax.positional import pipeline
from brax.io import image
import cv2 as cv

import jax

from jax import numpy as jnp

import gymnasium as gym

import mujoco
import mujoco.viewer

from typing import Optional


class brax_fep(gym.Env):
    def __init__(self):
        # load string from xml file

        self.system = mjcf.load(path="assets/franka_emika_panda/scene.xml")

        self.system = self.system.replace(
            init_q=jnp.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
        )
        # path='../brax/brax/envs/assets/pusher.xml')

        self.mjmodel = None
        self.mjdata = None
        self.viewer = None

    def reset(self):
        pipeline_state = jax.jit(pipeline.init)(
            self.system, self.system.init_q, jnp.zeros_like(self.system.init_q)
        )

        return pipeline_state

    def step(self, state: brax.State, action: jax.Array):
        state = jax.jit(pipeline.step)(self.system, state, action)
        # state = pipeline.step(self.system, state, action)

        return state

    def render(self, state: brax.State, mode="human"):
        # mat = image.render_array(self.system, state, 512, 512)
        # cv.imshow("image", mat)
        # cv.waitKey(1)
        if self.mjmodel is None:
            self.mjmodel = mujoco.MjModel.from_xml_path(
                "assets/franka_emika_panda/scene.xml"
            )
            self.mjdata = mujoco.MjData(self.mjmodel)

        if mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.mjmodel, self.mjdata)

            with self.viewer.lock():
                self.mjdata.qpos[:] = state.q

            mujoco.mj_forward(self.mjmodel, self.mjdata)

            self.viewer.sync()

        # if mode == "human":
        #     self.mjdata.qpos[:] = state.q
        #     renderer = mujoco.Renderer(self.mjmodel, 512, 512)
        #     renderer.update_scene(self.mjdata, camera=0)
        #     array = (
        #         renderer.render()
        #     )  # image.render_array(self.system, state, 1024, 1024)
        #     cv.imshow("image", array)
        #     cv.waitKey(1)

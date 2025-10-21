"""
2D rendering of the Robotic's Warehouse
environment using pyglet
"""

import math
import os
import sys

import numpy as np
import six
from gymnasium import error

from tarware.warehouse import AgentType, Direction

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
    import pyglet.shapes
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import gl
except ImportError as e:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_LIGHTORANGE = (255, 200, 0)
_RED = (255, 0, 0)
_ORANGE = (255, 165, 0)
_DARKORANGE = (255, 140, 0)
_DARKSLATEBLUE = (72, 61, 139)
_TEAL = (0, 128, 128)
_MAROON = (128, 0, 0)
_BLUE = (30,144,255)

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK
_SHELF_COLOR = _DARKSLATEBLUE
_SHELF_REQ_COLOR = _TEAL
_AGENT_COLOR = _DARKORANGE
_AGENT_LOADED_COLOR = _RED
_AGENT_DIR_COLOR = _BLACK
_GOAL_COLOR = (60, 60, 60)
_CHARGING_COLOR = (173, 216, 230)  # Light blue
_CARRIER_COLOR = _MAROON
_LOADER_AGENT = _BLUE

_SHELF_PADDING = 2


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, world_size):
        display = get_display(None)
        self.rows, self.cols = world_size

        self.grid_size = 30
        self.icon_size = 20

        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def render(self, env, return_rgb_array=False):
        gl.glClearColor(*_BACKGROUND_COLOR, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_goals(env)
        self._draw_charging_stations(env)
        self._draw_shelfs(env)
        self._draw_agents(env)

        # Draw battery levels on agents
        for agent in env.agents:
            self._draw_badge(agent.y, agent.x, agent.battery)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        # VERTICAL LINES
        for r in range(self.rows + 1):
            line = pyglet.shapes.Line(0, (self.grid_size + 1) * r + 1, (self.grid_size + 1) * self.cols, (self.grid_size + 1) * r + 1, color=_GRID_COLOR)
            line.draw()

        # HORIZONTAL LINES
        for c in range(self.cols + 1):
            line = pyglet.shapes.Line((self.grid_size + 1) * c + 1, 0, (self.grid_size + 1) * c + 1, (self.grid_size + 1) * self.rows, color=_GRID_COLOR)
            line.draw()

    def _draw_charging_stations(self, env):
        for station in env.charging_stations:
            x, y = station.x, station.y
            y = self.rows - y - 1  # pyglet rendering is reversed
            rect = pyglet.shapes.Rectangle((self.grid_size + 1) * x + 1, (self.grid_size + 1) * y + 1, self.grid_size, self.grid_size, color=_CHARGING_COLOR)
            rect.draw()

    def _draw_shelfs(self, env):
        for shelf in env.shelfs:
            x, y = shelf.x, shelf.y
            y = self.rows - y - 1  # pyglet rendering is reversed
            shelf_color = (
                _SHELF_REQ_COLOR if shelf in env.request_queue else _SHELF_COLOR
            )
            rect = pyglet.shapes.Rectangle((self.grid_size + 1) * x + _SHELF_PADDING + 1, (self.grid_size + 1) * y + _SHELF_PADDING + 1, self.grid_size - 2 * _SHELF_PADDING, self.grid_size - 2 * _SHELF_PADDING, color=shelf_color)
            rect.draw()

    def _draw_goals(self, env):
        for goal in env.goals:
            x, y = goal
            y = self.rows - y - 1  # pyglet rendering is reversed
            rect = pyglet.shapes.Rectangle((self.grid_size + 1) * x + 1, (self.grid_size + 1) * y + 1, self.grid_size, self.grid_size, color=_GOAL_COLOR)
            rect.draw()

    def _draw_agents(self, env):
        radius = self.grid_size / 3

        for agent in env.agents:

            col, row = agent.x, agent.y
            row = self.rows - row - 1  # pyglet rendering is reversed
            
            if agent.type == AgentType.AGV:
                resolution = 6
            elif agent.type == AgentType.PICKER:
                resolution = 4
            elif agent.type == AgentType.AGENT:
                resolution = 8
            else:
                raise ValueError("Agent type not recognized by environment.")
            
            draw_color = _AGENT_LOADED_COLOR if agent.carrying_shelf else _AGENT_COLOR
            center_x = (self.grid_size + 1) * col + self.grid_size // 2 + 1
            center_y = (self.grid_size + 1) * row + self.grid_size // 2 + 1
            self._draw_polygon(center_x, center_y, radius, resolution, draw_color)

        for agent in env.agents:

            col, row = agent.x, agent.y
            row = self.rows - row - 1  # pyglet rendering is reversed

            center_x = (self.grid_size + 1) * col + self.grid_size // 2 + 1
            center_y = (self.grid_size + 1) * row + self.grid_size // 2 + 1
            end_x = center_x + (
                radius if agent.dir.value == Direction.RIGHT.value else 0
            ) + (
                -radius if agent.dir.value == Direction.LEFT.value else 0
            )
            end_y = center_y + (
                radius if agent.dir.value == Direction.UP.value else 0
            ) + (
                -radius if agent.dir.value == Direction.DOWN.value else 0
            )
            line = pyglet.shapes.Line(center_x, center_y, end_x, end_y, color=_AGENT_DIR_COLOR)
            line.draw()

    def _draw_badge(self, row, col, level):
        badge_width = 10
        badge_height = 5
        x_offset = (self.grid_size + 1) * col + self.grid_size // 2 + 5
        y_offset = (self.rows - row - 1) * (self.grid_size + 1) + self.grid_size // 2 - 10
        
        if level > 60:
            color = (0, 255, 0)  # Green
        elif level > 20:
            color = (255, 255, 0)  # Yellow
        else:
            color = (255, 0, 0)  # Red
        
        rect = pyglet.shapes.Rectangle(x_offset, y_offset, badge_width, badge_height, color=color)
        rect.draw()

    def _draw_polygon(self, center_x, center_y, radius, sides, color):
        """Draw a regular polygon using triangles."""
        vertices = []
        for i in range(sides):
            angle = 2 * math.pi * i / sides
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            vertices.append((x, y))
        
        # Draw triangles from center to edges
        for i in range(sides):
            next_i = (i + 1) % sides
            triangle = pyglet.shapes.Triangle(
                center_x, center_y,
                vertices[i][0], vertices[i][1],
                vertices[next_i][0], vertices[next_i][1],
                color=color
            )
            triangle.draw()

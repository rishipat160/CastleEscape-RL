import gym
from gym import spaces
import numpy as np
import random

class CastleEscapeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CastleEscapeEnv, self).__init__()
        # Define a 5x5 grid (numbered from (0,0) to (4,4))
        self.grid_size = 5
        self.rooms = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self.goal_room = (4, 4)  # Define the goal room

        # Define health states
        self.health_states = ['Full', 'Injured', 'Critical']
        self.health_state_to_int = {'Full': 2, 'Injured': 1, 'Critical': 0}
        self.int_to_health_state = {2: 'Full', 1: 'Injured', 0: 'Critical'}

        # Define the guards with their strengths (affects combat) and keenness (affects hiding)
        self.guards = {
            'G1': {'strength': 0.8, 'keenness': 0.1},  # Guard 1
            'G2': {'strength': 0.6, 'keenness': 0.3},  # Guard 2
            'G3': {'strength': 0.9, 'keenness': 0.2},  # Guard 3
            'G4': {'strength': 0.7, 'keenness': 0.5},  # Guard 4
        }
        self.guard_names = list(self.guards.keys())

        # Rewards
        self.rewards = {
            'goal': 10000,
            'combat_win': 10,
            'combat_loss': -1000,
            'defeat': -1000
        }

        # Actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FIGHT', 'HIDE']
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation space
        obs_space_dict = {
            'player_position': spaces.Tuple((spaces.Discrete(self.grid_size), spaces.Discrete(self.grid_size))),
            'player_health': spaces.Discrete(len(self.health_states)),
            'guard_positions': spaces.Dict({
                guard: spaces.Tuple((spaces.Discrete(self.grid_size), spaces.Discrete(self.grid_size)))
                for guard in self.guards
            })
        }
        self.observation_space = spaces.Dict(obs_space_dict)

        # Set initial state
        self.reset()

    def reset(self):
        """Resets the game to the initial state"""
        #print(self.rooms[1:-1])
        rnd_indices = np.random.choice(range(1,len(self.rooms)-1), size=len(self.guards), replace=False)
        guard_pos = [self.rooms[i] for i in rnd_indices]
        self.current_state = {
            'player_position': (0, 0),
            'player_health': 'Full',
            'guard_positions': {
                guard: pos for guard, pos in zip(self.guard_names, guard_pos)
            }  # Guards in random rooms (not the goal or the starting)
        }
        return self.get_observation(), 0, False, {}

    def get_observation(self):
        guard_in_cell = None
        guard_positions = self.current_state['guard_positions']
        player_position = self.current_state['player_position']
        for guard in guard_positions:
            if guard_positions[guard] == player_position:
                guard_in_cell = guard
                break

        obs = {
            'player_position': self.current_state['player_position'],
            'player_health': self.health_state_to_int[self.current_state['player_health']],
            'guard_in_cell': guard_in_cell if guard_in_cell else None,
        }
        return obs

    def is_terminal(self):
        """Check if the game has reached a terminal state"""
        if self.current_state['player_position'] == self.goal_room:  # Reaching the goal means victory
            return 'goal'
        if self.current_state['player_health'] == 'Critical':  # Losing health 3 times results in defeat
            return 'defeat'
        return False

    def move_player(self, action):
        """Move player based on the action, but prevent movement if a guard is in the same room"""
        current_position = self.current_state['player_position']
        guards_in_room = [
            guard for guard in self.guards
            if self.current_state['guard_positions'][guard] == current_position
        ]

        # If there's a guard in the room, the player must fight or hide
        if guards_in_room:
            return f"Guard {guards_in_room[0]} is in the room! You must fight or hide.", 0

        x, y = self.current_state['player_position']
        directions = {
            'UP': (x - 1, y),
            'DOWN': (x + 1, y),
            'LEFT': (x, y - 1),
            'RIGHT': (x, y + 1)
        }

        # Calculate the intended move
        new_position = directions.get(action, self.current_state['player_position'])

        # Ensure new position is within bounds
        if 0 <= new_position[0] < self.grid_size and 0 <= new_position[1] < self.grid_size:
            # 90% chance to move as intended
            if random.random() <= 0.9:
                self.current_state['player_position'] = new_position
            else:
                # 10% chance to move to a random adjacent cell
                adjacent_positions = [
                    directions[act] for act in directions if act != action
                ]
                adjacent_positions = [
                    pos for pos in adjacent_positions
                    if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
                ]
                if adjacent_positions:
                    self.current_state['player_position'] = random.choice(adjacent_positions)
            return f"Moved to {self.current_state['player_position']}", 0
        else:
            return "Out of bounds!", 0

    def move_player_to_random_adjacent(self):
        """Move player to a random adjacent cell without going out of bounds"""
        x, y = self.current_state['player_position']
        directions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        # Filter out-of-bounds positions
        adjacent_positions = [
            pos for pos in directions
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
        ]

        # Move player to a random adjacent position
        if adjacent_positions:
            self.current_state['player_position'] = random.choice(adjacent_positions)

    def try_fight(self):
        """Player chooses to fight the guard"""
        current_position = self.current_state['player_position']
        guards_in_room = [
            guard for guard in self.guards
            if self.current_state['guard_positions'][guard] == current_position
        ]

        if guards_in_room:
            guard = guards_in_room[0]  # Choose one guard to fight
            strength = self.guards[guard]['strength']

            # Player tries to fight the guard
            if random.random() > strength:  # Successful fight
                self.move_player_to_random_adjacent()  # Move player to a random adjacent cell after victory
                return f"Fought {guard} and won!", self.rewards['combat_win']
            else:  # Player loses the fight
                if self.current_state['player_health'] == 'Full':
                    self.current_state['player_health'] = 'Injured'
                elif self.current_state['player_health'] == 'Injured':
                    self.current_state['player_health'] = 'Critical'
                self.move_player_to_random_adjacent()  # Move player to a random adjacent cell after defeat
                return f"Fought {guard} and lost!", self.rewards['combat_loss']
        return "No guard to fight!", 0

    def try_hide(self):
        """Player attempts to hide from the guard"""
        current_position = self.current_state['player_position']
        guards_in_room = [
            guard for guard in self.guards
            if self.current_state['guard_positions'][guard] == current_position
        ]

        if guards_in_room:
            guard = guards_in_room[0]  # Choose one guard to hide from
            keenness = self.guards[guard]['keenness']

            # Player tries to hide
            if random.random() > keenness:  # Successful hide
                self.move_player_to_random_adjacent()  # Move player to a random adjacent cell after successfully hiding
                return f"Successfully hid from {guard}!", 0
            else:
                return self.try_fight()  # Hide failed, must fight
        return "No guard to hide from!", 0

    def play_turn(self, action):
        """Take an action and update the state"""
        if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            return self.move_player(action)
        elif action == 'FIGHT':
            return self.try_fight()
        elif action == 'HIDE':
            return self.try_hide()
        else:
            return "Invalid action!", 0

    def step(self, action):
        """Performs one step in the environment"""

        ## Thisis a fix for gym environment. 
        if (isinstance(action, str)):
            action = self.actions.index(action)

        action_name = self.actions[action]
        result, reward = self.play_turn(action_name)

        done = False
        terminal_state = self.is_terminal()
        if terminal_state == 'goal':
            done = True
            reward += self.rewards['goal']
            result += f" You've reached the goal! {self.rewards['goal']} points!"
        elif terminal_state == 'defeat':
            done = True
            reward += self.rewards['defeat']
            result += f" You've been caught! {self.rewards['combat_loss']} points!"

        observation = self.get_observation()
        info = {'result': result, 'action': action_name}

        return observation, reward, done, info

    def render(self, mode='human'):
        """Renders the current state"""
        print(f"Current state: {self.current_state}")

    def close(self):
        """Performs cleanup"""
        pass


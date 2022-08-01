import gym
from gym import spaces
from gym.utils.renderer import Renderer
import pygame
import numpy as np
import random

import constants
import class_hex
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
#
# Helper objects
#
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def helper_text_objects(incoming_text,
                        incoming_color,
                        font_big = False):
    if font_big:
        Text_surface = constants.FONT_BIG.render(incoming_text, True, incoming_color)
    else:
        Text_surface = constants.FONT_DEBUG_MESSAGE.render(incoming_text, True, incoming_color)

    return Text_surface, Text_surface.get_rect()

def hex_coords(obj1):
    """This definition will find out how far obj1 is from obj2, using the axial coordinte system"""
    obj1_cube = []
    if obj1[1] % 2 == 0:
        obj1_cube.append(obj1[0] - obj1[1] / 2)
    else:
        obj1_cube.append(obj1[0] - (obj1[1] - 1) / 2)
    obj1_cube.append(-obj1_cube[0] - obj1[1])
    obj1_cube.append(obj1[1])
    return obj1_cube

def hex_distance(obj1, obj2):
    """This definition will find out how hard obj1 is from obj2"""

    obj1_coords = hex_coords(obj1)
    obj2_coords = hex_coords(obj2)

    return max([abs(obj1_coords[0] - obj2_coords[0]),
                abs(obj1_coords[1] - obj2_coords[1]),
                abs(obj1_coords[2] - obj2_coords[2])])

def draw_text(display_surface, text_to_display, T_coordinates, text_color, outline = False, center = True, font_big = False):
    """Definition takes in text and displays the text to the screen"""
    # Outline feature from sloth at: https://stackoverflow.com/questions/54363047/how-to-draw-outline-on-the-fontpygame
    _circle_cache = {}

    def _circlepoints(r):
        r = int(round(r))
        if r in _circle_cache:
            return _circle_cache[r]
        x, y, e = r, 0, 1 - r
        _circle_cache[r] = points = []
        while x >= y:
            points.append((x, y))
            y += 1
            if e < 0:
                e += 2 * y - 1
            else:
                x -= 1
                e += 2 * (y - x) - 1
        points += [(y, x) for x, y in points if x > y]
        points += [(-x, y) for x, y in points if x]
        points += [(x, -y) for x, y in points if y]
        points.sort()
        return points

    # --- Create an outline around the name text so you can read it!
    if outline:
        text_surf, text_rect = helper_text_objects(text_to_display, text_color, font_big=font_big)
        text_surf.convert_alpha()
        if center:
            text_rect.center = T_coordinates
        else:
            text_rect.topleft = T_coordinates
        w = text_surf.get_width() + 2 * constants.OUTLINE_SIZE
        h = text_surf.get_height()
        osurf = pygame.Surface((w, h + 2 * constants.OUTLINE_SIZE)).convert_alpha()
        osurf.fill((0, 0, 0, 0))

        surf = osurf.copy()
        outline_surf, outline_rect = helper_text_objects(text_to_display, constants.COLOR_BLACK, font_big=font_big)
        if center:
            outline_rect.center = T_coordinates
        else:
            outline_rect.topleft = T_coordinates
        osurf.blit(outline_surf.convert_alpha(), (0,0))

        for dx, dy in _circlepoints(constants.OUTLINE_SIZE):
            surf.blit(osurf, (dx + constants.OUTLINE_SIZE, dy + constants.OUTLINE_SIZE))

        surf.blit(text_surf, (constants.OUTLINE_SIZE, constants.OUTLINE_SIZE))
        display_surface.blit(surf, outline_rect)

    else:
        text_surf, text_rect = helper_text_objects(text_to_display, text_color, font_big=font_big)
        text_surf.convert_alpha()
        text_rect.center = T_coordinates
        display_surface.blit(text_surf, text_rect)

class C_Sprite(pygame.sprite.Sprite):
    """
    Creatures have health and can damage other objects by attacking them.
    Can also die.
    """

    def __init__(self,
                 x,
                 y,
                 sprite,
                 name_instance,
                 team,
                 hp=10,
                 hp_max = 100,
                 movement_max = 0,
                 strength=20,
                 strength_ranged = 0,
                 dug_in = 0,
                 has_zoc = False):
        self.x = x
        self.y = y
        self.sprite = sprite
        self.name_instance = name_instance
        self.team = team
        self.hp_max = hp_max
        self.hp = hp
        self.movement_max = movement_max
        self.movement = movement_max
        self.strength = strength
        self.strength_ranged = strength_ranged
        self.dug_in = dug_in
        self.has_zoc = has_zoc
        self.alive = True
        self.status = 'alive'
        self.status_default = 'alive'
        self.in_zoc = False


class C_Unit(C_Sprite):
    def __init__(self,
                 x,
                 y,
                 sprite,
                 name_instance,
                 team,
                 hp=100,
                 hp_max=100,
                 movement_max = 0,
                 strength=20,
                 strength_ranged = 0,
                 dug_in = 0,
                 has_zoc = False):
                 
        super().__init__(x,
                         y,
                         sprite,
                         name_instance,
                         team,
                         hp,
                         hp_max,
                         movement_max,
                         strength,
                         strength_ranged,
                         dug_in,
                         has_zoc)
        
    def death_unit(self):
        #print(unit.name_instance + ' is dead!')
        self.alive = False
        self.status = 'dead'
        self.movement = 0
    
    def take_damage(self,
                    damage,
                    aggressor_alive = True):
        # Unit doesn't die if the unit doesn't have enough HP to take it over
        if not aggressor_alive and (self.hp - damage) < 0:
            self.hp = 1
        else:
            self.hp -= damage
        #self.status = 'took damage'
        if self.hp <= 0:
            self.death_unit()


class C_City(C_Sprite):
    def __init__(self,
                 x,
                 y,
                 sprite,
                 name_instance,
                 team='defender',
                 hp=1,
                 hp_max=200,
                 wall_hp=100,
                 strength=18,
                 strength_ranged=0,
                 ranged_combat=False,
                 heal=False,
                 dug_in = 0,
                 has_zoc = True):
        self.wall_hp = wall_hp
        self.ranged_combat = ranged_combat
        self.heal = heal
        self.has_zoc = has_zoc

        super().__init__(x,
                         y,
                         sprite,
                         name_instance,
                         team,
                         hp,
                         hp_max,
                         strength,
                         strength_ranged, 
                         dug_in,
                         has_zoc)
    
    def take_damage(self,
                    damage,
                    aggressor_alive = True):
        # City doesn't die if the unit doesn't have enough HP to take it over
        if not aggressor_alive and (self.hp - damage) < 0:
            self.hp = 1
        else:
            self.hp -= damage
        self.status = 'took damage'

        # --- City dies when it doesn't have HP
        if self.hp <= 0:
            self.hp = 0
            self.death()

    def death(self):
        #print(self.name_instance + ' has been defeated!')
        self.alive = False
        self.status = 'dead'
                         
def attack(aggressor,
           target):
    '''Base attack definition using the formula found on CivFanatics
    TODO: Attack accounting for walls....'''

    defense_bonus = target.dug_in * 3
    fortification_bonus = 0

    if aggressor.strength_ranged > 0:
        if target.__class__ == C_City:
            fortification_bonus = 17

        strength_diff = np.round(aggressor.strength_ranged - fortification_bonus - aggressor.hp / 10) - np.round(target.strength - target.hp / 10 + defense_bonus)
        damage_taken = 0
    else:
        strength_diff = np.round(aggressor.strength - aggressor.hp / 10) - np.round(target.strength - target.hp / 10 + defense_bonus)
        damage_taken = np.round(30 * np.exp(-strength_diff * 0.04) * (random.randint(75, 125) / 100.0))

    damage_out = np.round(30 * np.exp(strength_diff * 0.04) * (random.randint(75, 125) / 100.0))
    #print(f"{aggressor.name_instance} attacked {target.name_instance} with {damage_out} damage")
    

    return damage_out, damage_taken

class GymEnv(gym.Env):
    metadata = {"render_modes": ["show", "hide"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Discrete(36)

        # We have 7 actions for each unit
        self.action_space = spaces.Discrete(7)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._renderer = Renderer(self.render_mode, self._render_frame)

        """
        If human-rendering is used, `self.surface_main` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.surface_main = None
        self.clock = None

    def get_observation(self):
        '''Definition returns the known universe
        positions of each unit and each city
        '''

        # --- Find the distance between the unit and the city
        city_loc = -1

        # dx city, dy city, dx unit 1, dy unit 1, dx unit 2, dy unit 2, ...
        location = np.array([])
        # hp_norm city, hp_norm unit 1, hp_norm unit 2, ...  
        hp = np.array([])
        # movement_norm unit1, movement_norm unit 2,  ...
        movement = np.array([])

        # --- Find the city location in game_objects
        for obj in enumerate(self.game_objects):
            if obj[1].__class__ == C_City:
                city_loc = obj[0]

        # --- Find the space between each unit and the city
        for obj in self.game_objects:
            dx_norm = (self.game_objects[city_loc].x - obj.x) / constants.MAP_WIDTH
            dy_norm = (self.game_objects[city_loc].y - obj.y) / constants.MAP_HEIGHT
            np.append(location, dx_norm)
            np.append(location, dy_norm)
            # --- Normalized HP
            np.append(hp, obj.hp / obj.hp_max)
            # --- Normalized movement point
            np.append(movement, obj.movement / obj.movement_max)
            
        
        return np.concatenate((location, hp, movement), axis=None)

    def reset(self, ep_number = 0, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        episode_number = ep_number
        self.turn_number = 0

        # Choose the agents' location
        self.attacker_location = random.sample(constants.HEX_LOCATIONS_ATTACKER, 5)
        self.defender_location = random.sample(constants.HEX_LOCATIONS_DEFENDER, 3)

        # --- Attacker units
        A_WARRIOR_1 = C_Unit(self.attacker_location[0][0],
                        self.attacker_location[0][1],
                        constants.S_WARRIOR,
                        "A_warrior_1",
                        team='attacker',
                        strength=20,
                        hp=100,
                        hp_max=100,
                        movement_max=2,
                        has_zoc=True)

        A_WARRIOR_2 = C_Unit(self.attacker_location[1][0],
                        self.attacker_location[1][1],
                        constants.S_WARRIOR,
                        "A_warrior_2",
                        team='attacker',
                        strength=20,
                        hp=100,
                        hp_max=100,
                        movement_max=2,
                        has_zoc=True)

        A_WARRIOR_3 = C_Unit(self.attacker_location[2][0],
                        self.attacker_location[2][1],
                        constants.S_WARRIOR,
                        "A_warrior_3",
                        team='attacker',
                        strength=20,
                        hp=100,
                        hp_max=100,
                        movement_max=2,
                        has_zoc=True)

        A_SLINGER_1 = C_Unit(self.attacker_location[3][0],
                        self.attacker_location[3][1],
                        constants.S_SLINGER,
                        "A_slinger_1",
                        team='attacker',
                        strength=5,
                        strength_ranged=15,
                        hp=100,
                        hp_max=100,
                        movement_max=2,
                        has_zoc=False)

        A_SLINGER_2 = C_Unit(self.attacker_location[4][0],
                        self.attacker_location[4][1],
                        constants.S_SLINGER,
                        "A_slinger_2",
                        team='attacker',
                        strength=5,
                        strength_ranged=15,
                        hp=100,
                        hp_max=100,
                        movement_max=2,
                        has_zoc=False)
        # --- Defender units
        D_WARRIOR_1 = C_Unit(self.defender_location[0][0],
                        self.defender_location[0][1],
                        constants.S_WARRIOR,
                        "D_warrior_1",
                        team='defender',
                        strength=20,
                        hp=100,
                        hp_max=100,
                        movement_max=2,
                        has_zoc=True)

        D_WARRIOR_2 = C_Unit(self.defender_location[1][0],
                        self.defender_location[1][1],
                        constants.S_WARRIOR,
                        "D_warrior_2",
                        team='defender',
                        strength=20,
                        hp=100,
                        hp_max=100,
                        movement_max=2,
                        has_zoc=True)

        D_SLINGER_1 = C_Unit(self.defender_location[2][0],
                        self.defender_location[2][1],
                        constants.S_SLINGER,
                        "D_slinger_1",
                        team='defender',
                        strength=5,
                        strength_ranged=15,
                        hp=100,
                        hp_max=100,
                        movement_max=2,
                        has_zoc=False)

        CITY = C_City(constants.LOC_CITY[0],
                      constants.LOC_CITY[1],
                      constants.S_CITY,
                      "City",
                      team='defender',
                      hp=200,
                      strength=28,
                      ranged_combat=False,
                      heal=True,
                      has_zoc=True)

        # Must have units first then the city last
        self.city_objects = [CITY]
        self.attacker_objects = [
                                    A_WARRIOR_1, A_WARRIOR_2, A_WARRIOR_3, A_SLINGER_1, A_SLINGER_2
                                ]
        self.defender_objects = [
                                    #D_WARRIOR_1, D_WARRIOR_2, D_SLINGER_1
                                ]
        self.game_objects = self.attacker_objects + self.defender_objects + self.city_objects
        self.own_objects = {'attacker': self.attacker_objects,
                            'defender': self.defender_objects}
        self.enemy_objects = {'attacker': self.defender_objects,
                              'defender': self.attacker_objects}

        observation = self.get_observation()

        self._renderer.reset()
        self._renderer.render_step()

        return observation

    def get_rewards(self, team):
        '''This definition will return the attacker agent reward status for each step as
        well as the location of the city relative to the attacker agent units'''
        reward = 0
        
        for obj in enumerate(self.city_objects):
                if team == 'attacker':
                    # --- Rewards for city status
                        if obj[1].status == 'dead':
                            reward += 20
                            obj[1].status = None
                        elif obj[1].status == 'took damage':
                            reward += 0.5
                            obj[1].status = obj[1].status_default
                        elif obj[1].status == 'healed':
                            reward -= 0.3
                            obj[1].status = obj[1].status_default
        # print(f"Reward before unit status: {reward}")
        # --- REWARDS for own unit status
        for obj in self.own_objects[team]:
            #print('BEFORE: {} status of {}'.format(obj.name_instance, obj.status))
            if obj.status == 'dead' and team == 'attacker':
                reward -= 1
            elif obj.status == 'took damage':
                reward += 0
                obj.status = obj.status_default
            elif obj.status == 'hit wall':
                reward -= 1
                obj.status = obj.status_default
            elif obj.status == 'healed':
                reward += 0.1
                obj.status = obj.status_default
            elif obj.status == 'attacked':
                reward += 0.2
                obj.status = obj.status_default
        # print(f"Reward after unit status: {reward}")
        # --- REWARDS for opponent unit status
        for obj in self.enemy_objects[team]:
            #print('BEFORE: {} status of {}'.format(obj.name_instance, obj.status))
            if obj.status == 'dead':
                reward += 3
                obj.status = None
        # print(f"Reward before final: {reward}")
        return reward

    def step(self, 
             team, 
             action_input=0):
        # --- agent action definition
        action = 'no-action'
        game_quit = False
        end_turn = False
        unit = None
        reward = 0

        if not any(obj.movement > 0 for obj in self.own_objects[team]):
            end_turn = True
            for obj in self.own_objects[team]:
                if obj.alive == True:
                    obj.movement = obj.movement_max
            if team == 'attacker':
                reward -= 1
                for obj in enumerate(self.city_objects):
                    city_loc = obj[0]
                     # Check to see if the city is dead or not
                    if self.city_objects[city_loc].hp >= 0:
                        # Attempt to heal the city otherwise
                        self.city_take_turn(self.city_objects[city_loc])

                for obj in self.own_objects[team]:
                        # --- Rewards for how far they are away from the city!
                        # - This is a linear reward, 0 for being next to city, -0.5 for maximum distance, per unit
                        dist = hex_distance([obj.x, obj.y], [self.city_objects[city_loc].x, self.city_objects[city_loc].y])
                        dist_reward = float(dist - 1) / (max([constants.MAP_HEIGHT, constants.MAP_WIDTH]) - 2)
                        reward -= dist_reward / 0.5
            self.turn_number += 1
        
        for obj in self.own_objects[team]:
            if obj.alive == True and obj.movement > 0:
                unit = obj
        
        if not game_quit and not end_turn:
            action = self.game_handle_moves_ml_ai(action_input, unit)
            all_attacker_dead = all(obj.hp <= 0 for obj in self.attacker_objects)
            if all_attacker_dead:
                print("All attacker units are dead")
                game_quit = True

            for obj in self.city_objects:
                # Check to see if the city is dead or not
                    if obj.hp <= 0:
                        print("City is destroyed")
                        game_quit = True
        
        if action == 'QUIT':
            game_quit = True
          
        self._renderer.render_step()
        reward += self.get_rewards(team)

        return self.get_observation(), reward, end_turn, game_quit

    def game_handle_moves_ml_ai(self,
                                action,
                                unit):

        # --- Movement commands for attacker
        # --- Determine the parity
        if unit.y % 2 == 0:
            parity = 'EVEN'
        else:
            parity = 'ODD'
        # --- Make a movement
        direction = constants.MOVEMENT_ONE_UNIT[action]
        self.move(unit, constants.MOVEMENT_DIR[direction][parity][0],
                                constants.MOVEMENT_DIR[direction][parity][1])
        

        return "player-moved"
    
    def get_current_state(self):
        """Use this to get unit position as well as health, used for rendering in Blender"""
        temp_data = {}
        for obj in self.game_objects:
            temp_data[obj.name_instance] = {}
            temp_data[obj.name_instance]['health'] = obj.hp
            temp_data[obj.name_instance]['position'] = [obj.x, obj.y]

        return temp_data

    # ---------------------------------------------
    # UNIT ACTION
    # ---------------------------------------------
    def city_take_turn(self, city):
        if city.ranged_combat:
            # Check for a creature that is within two tiles
            items_within_range = []
            for ii in range(-2, 2):
                for jj in range(-2, 2):
                    temp = self.map_check_for_creatures(city.x - ii, city.y - jj, city)
                    # print('checked location', city.x - ii, city.y - jj)
                    if temp:
                        # print(temp.__class__, ' within range')
                        if temp.__class__ == C_Unit:
                            # --- Only add alive units
                            if temp.alive:
                                #print('found {} at {} {}'.format(
                                #    temp.name_instance,
                                #    city.x - ii,
                                #    city.y - jj))
                                items_within_range.append(temp)

            # --- Attack a random creature
            if len(items_within_range) > 0:
                rand_numb = random.randint(0, len(items_within_range) - 1)
                damage_output, damage_taken = attack(city, items_within_range[rand_numb], ranged=True)

                # City should not take any damage for the ranged combat
                items_within_range[rand_numb].take_damage(damage_output)

        if city.heal:
            temp = self.game_map.grid[(city.x, city.y)].get_neighbors(self.game_map.grid)
            tiles_within_range = []
            for ii in range(len(temp)):
                tiles_within_range.append(temp[ii].index)
            # Check to make sure there are three enemy unit with ZOC within one tile
            items_within_range = []
            for obj in self.game_objects:
                for tile in tiles_within_range:
                    if obj.x == tile[0] and obj.y == tile[1] and obj.alive and obj.team != city.team and obj.has_zoc == True:
                        #print(f'position {obj.x} {obj.y} {obj.name_instance}')
                        items_within_range.append(temp)


            # --- Heal if less than 3 tiles are occupied
            if len(items_within_range) < 3:
                city.hp += 20
                city.status = 'healed'
                if city.hp > city.hp_max:
                    city.hp = city.hp_max

    def move(self,
             unit,
             dx,
             dy):

        # --- Check to see if the units is still alive
        if unit.alive:
            #print(f"{unit.name_instance} attempting to make a move, movement: {unit.movement}")
            if unit.movement <= 0:
                # print(f"{unit.name_instance} FAILED to make a move")
                assert(False)
            # --- Heal and fortify the unit if it doesn't move
            if dx == 0 and dy == 0:
                unit.movement = 0
                unit.hp += 10
                unit.status = 'healed'
                if unit.hp > unit.hp_max:
                    unit.hp = unit.hp_max
                    unit.status = unit.status_default
                # If unit is already fortified, add 1 more dug in level (max 2)
                if unit.dug_in < 2:
                    unit.dug_in += 1
                #print(f"{unit.name_instance} fortifying, movement: {unit.movement}")
                return

            # Check to see if the movement is still "in bounds"
            if (int(unit.x + dx), int(unit.y + dy)) not in self.game_map.grid:
                tile_is_wall = True
            else:
                tile_is_wall = False


            # --- set unit status to 'hit wall' if it hit the wall
            if tile_is_wall:
                unit.status = 'hit wall'
                #print(f"{unit.name_instance} hit a wall, movement: {unit.movement}")
                unit.movement = 0
                return
            
            target = self.map_check_for_creatures(unit.x + dx,
                                             unit.y + dy,
                                             unit)


            if target and target.team != unit.team:
                damage_output, damage_taken = attack(unit, target)
                unit.movement = 0
                #print('taken {}, output {}'.format(damage_taken, damage_output))

                # Take the damage
                if damage_taken > 0:
                    #print('damage_taken', damage_taken)
                    unit.take_damage(damage_taken)

                # Have the other object take damage
                if damage_output > 0:
                    target.take_damage(damage_output, unit.alive)
                    unit.status = 'attacked'

                # Move to killed unit location if attacker is a melee unit
                if unit.strength_ranged <= 0 and target.alive == False:
                    unit.x += dx
                    unit.y += dy
                #print(f"{unit.name_instance} made an attack, movement: {unit.movement}")
                return

            # --- Remove dug in bonus if unit moves
            if dx > 0 or dy > 0:
                unit.dug_in = 0

            # --- Move the unit if it can
            if not tile_is_wall and target is None:
                # If target moved into ZoC it should not be allowed to move again
                if not (unit.in_zoc and unit.movement < unit.movement_max):
                    unit.movement -= 1
                    unit.x += dx
                    unit.y += dy
                    self.check_zoc_status(unit)

            #print(f"{unit.name_instance} made a move, movement {unit.movement}")
    
    def check_zoc_status(self, unit):
        '''Check if unit is in ZOC'''
        # Check for enemy with zoc that is within one tiles
        for ii in range(-1, 1):
            for jj in range(-1, 1):
                temp = self.map_check_for_creatures(unit.x - ii, unit.y - jj, unit)
                if temp:
                    if temp.__class__ == C_Unit:
                        # --- Only add alive enemy units with ZOC
                        if temp.alive and temp.team != unit.team and temp.has_zoc:
                            unit.in_zoc = True
                            return
        unit.in_zoc = False

    # ---------------------------------------------
    # MAP
    # ---------------------------------------------
    def map_create(self):

        self.game_map = class_hex.HexMap(constants.MAP_HEIGHT,
                                    constants.MAP_WIDTH,
                                (constants.HEX_SIZE, constants.HEX_SIZE),
                                    constants.EDGE_OFFSET)
        # TODO: block the edge tiles!


    def map_check_for_creatures(self,
                                x,
                                y,
                                actor):
        target = None

        # --- check objectlist to find creature at that location that isn't the actor
        for object in self.game_objects:
            if (object is not actor and
                    object.x == x and
                    object.y == y and
                    object.alive):
                target = object

            if target:
                return target

    # ---------------------------------------------
    # DRAWING
    # ---------------------------------------------
    def draw_unit(self, sprite, x, y):
        """Draw the unit"""

        self.surface_main.blit(sprite,
                          (self.game_map.grid[(int(x), int(y))].rect.x,
                          self.game_map.grid[(int(x), int(y))].rect.y - self.game_map.grid[(int(x), int(y))].rect.h / 4))

    def draw_game(self):
        # --- Return list of all events in the event queue (This is to stop it from crashing the window)
        pygame.event.get()

        # --- Clear the surface
        self.surface_main.fill(constants.COLOR_DEFAULT_BG)

        # --- Draw the map
        self.draw_map()

        # --- Draw the objects
        for obj in self.game_objects:
            self.draw_unit(obj.sprite, obj.x, obj.y)

        # --- Draw the text
        for obj in self.game_objects:
            # Draw the HP above the unit
            color = {'attacker': constants.COLOR_RED,
                    'defender': constants.COLOR_BLUE}
            draw_text(self.surface_main, "{:.0f}/{:.0f}".format(obj.hp, obj.hp_max),
                    (self.game_map.grid[(int(obj.x), int(obj.y))].rect.x + self.game_map.grid[(int(obj.x), int(obj.y))].rect.width / 2,
                    self.game_map.grid[(int(obj.x), int(obj.y))].rect.y - self.game_map.grid[(int(obj.x), int(obj.y))].rect.h*2/7),
                    color[obj.team], outline=True)

            # Draw the units name
            # draw_text(SURFACE_MAIN, obj.name_instance,
            #           (GAME_MAP.grid[(int(obj.x), int(obj.y))].rect.x + GAME_MAP.grid[(int(obj.x), int(obj.y))].rect.width / 2,
            #            GAME_MAP.grid[(int(obj.x), int(obj.y))].rect.y + GAME_MAP.grid[(int(obj.x), int(obj.y))].rect.h*4/5),
            #           constants.COLOR_PURPLE, outline=True)


        # Draw the episode number
        if False:
            draw_text(SURFACE_MAIN, f'Episode: {episode_number}',
                    (constants.EDGE_OFFSET,
                    constants.HEX_SIZE * constants.MAP_HEIGHT - constants.EDGE_OFFSET + constants.HEX_SIZE / 2),
                    constants.COLOR_LIGHT_GREY, outline=True, center=False, font_big = True)

        # Draw the episode number
        draw_text(self.surface_main, f'Turn: {self.turn_number}',
                (constants.EDGE_OFFSET,
                constants.HEX_SIZE * constants.MAP_HEIGHT - constants.EDGE_OFFSET + constants.HEX_SIZE / 7),
                constants.COLOR_LIGHT_GREY, outline=True, center=False, font_big = True)

        # --- Update game display
        pygame.display.flip()


    def draw_map(self):

        for loc in self.game_map.grid:
            if self.surface_main is not None:
                self.surface_main.blit(self.game_map.grid[loc].image, (self.game_map.grid[loc].rect.x, self.game_map.grid[loc].rect.y))
                #SURFACE_MAIN.blit(GAME_MAP.grid[loc].image_outline, (GAME_MAP.grid[loc].rect.x, GAME_MAP.grid[loc].rect.y))

                # Draw the distance between the city and the location on the map,
                if False:
                    draw_text(SURFACE_MAIN, f'{hex_distance(loc, [3,3])}',
                        (GAME_MAP.grid[loc].rect.x + int(constants.HEX_SIZE / 2),
                        GAME_MAP.grid[loc].rect.y + int(constants.HEX_SIZE / 2)),
                        constants.COLOR_BLACK)

                # Draw the map location on the tile #f'{loc[0] - 3},{loc[1] - 3}, {-(loc[0]-3) -(loc[1]-3)}',
                if False:
                    draw_text(SURFACE_MAIN, f'{loc[0]}, {loc[1]}',
                        (GAME_MAP.grid[loc].rect.x + int(constants.HEX_SIZE / 2),
                        GAME_MAP.grid[loc].rect.y + int(constants.HEX_SIZE / 2)),
                        constants.COLOR_BLACK)

    def render(self):
        return self._renderer.get_renders()

    def _render_frame(self, mode):
        assert mode is not None

        if self.surface_main is None and mode == "show":
            pygame.init()
            pygame.display.init()
            self.surface_main = pygame.display.set_mode((constants.MAP_WIDTH
                                                    * constants.HEX_SIZE
                                                    + int(constants.HEX_SIZE / 2)
                                                    + constants.EDGE_OFFSET * 2,
                                                    (constants.MAP_HEIGHT
                                                     * constants.HEX_SIZE)
                                                    - (int(constants.MAP_HEIGHT / 2)
                                                       * int(constants.HEX_SIZE / 2))
                                                    + int(constants.HEX_SIZE / 4)
                                                    + constants.EDGE_OFFSET * 2))#, pygame.FULLSCREEN)
            #SURFACE_MAIN = pygame.display.set_mode((1920,1080), pygame.FULLSCREEN)
        if self.clock is None and mode == "show":
            self.clock = pygame.time.Clock()
        
        self.map_create()
        if mode == "show":
            self.draw_game()


    def close(self):
        if self.surface_main is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    env = GymEnv("show")
    env.reset()
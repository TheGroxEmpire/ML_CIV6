# game.py

import pygame
import constants
import random
import numpy as np
import class_hex


# ------------------------------
# Base class
# ------------------------------

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
                 strength=20,
                 strength_ranged = 0,
                 dug_in = 0):
        self.x = x
        self.y = y
        self.sprite = sprite
        self.name_instance = name_instance
        self.team = team
        self.hp_max = hp_max
        self.hp = hp
        self.strength = strength
        self.strength_ranged = strength_ranged
        self.dug_in = dug_in
        self.alive = True
        self.status = 'alive'
        self.status_default = 'alive'

    def draw(self):
        """Draw the unit"""

        global GAME_MAP

        SURFACE_MAIN.blit(self.sprite,
                          (GAME_MAP.grid[(int(self.x), int(self.y))].rect.x,
                          GAME_MAP.grid[(int(self.x), int(self.y))].rect.y - GAME_MAP.grid[(int(self.x), int(self.y))].rect.h / 4))



class C_Unit(C_Sprite):
    def __init__(self,
                 x,
                 y,
                 sprite,
                 name_instance,
                 team,
                 hp=100,
                 hp_max=100,
                 strength=20,
                 strength_ranged = 0,
                 dug_in = 0):
                 
        super().__init__(x,
                         y,
                         sprite,
                         name_instance,
                         team,
                         hp,
                         hp_max,
                         strength,
                         strength_ranged,
                         dug_in)

    def move(self,
             dx,
             dy):

        # --- Check to see if the units is still alive!!!!
        if self.alive:
            # Check to see if the movement is still "in bounds"
            if (int(self.x + dx), int(self.y + dy)) not in GAME_MAP.grid:
                tile_is_wall = True
            else:
                tile_is_wall = False


            # --- set unit status to 'hit wall' if it hit the wall
            if tile_is_wall:
                self.status = 'hit wall'
            
            target = map_check_for_creatures(self.x + dx,
                                             self.y + dy,
                                             self)


            if target and target.team != self.team:
                damage_output, damage_taken = attack(self, target)
                #print('taken {}, output {}'.format(damage_taken, damage_output))

                # Take the damage
                if damage_taken > 0:
                    #print('damage_taken', damage_taken)
                    self.take_damage(damage_taken)

                # Have the other object take damage
                if damage_output > 0:
                    target.take_damage(damage_output, self.alive)
                    self.status = 'attacked'

            # --- Heal and fortify the unit if it doesn't move
            if dx == 0 and dy == 0:
                self.hp += 10
                self.status = 'healed'
                if self.hp > self.hp_max:
                    self.hp = self.hp_max
                    self.status = self.status_default
                # If unit is already fortified, add 1 more dug in level (max 2)
                if self.dug_in < 2:
                    self.dug_in += 1
            else:
                self.dug_in = 0

            # --- Move the unit if it can
            if not tile_is_wall and target is None:
                self.x += dx
                self.y += dy

    def take_damage(self,
                    damage,
                    aggressor_alive = True):
        # Unit doesn't die if the unit doesn't have enough HP to take it over
        if not aggressor_alive and (self.hp - damage) < 0:
            self.hp = 1
        else:
            self.hp -= damage
        #self.status = 'took damage'


        # --- Unit dies if less than 0 health
        if self.hp <= 0:
            self.death_unit()

    def death_unit(self):
        '''On death, most citys stop moving.'''
        #print(self.name_instance + ' is dead!')
        self.alive = False
        self.status = 'dead'


class C_City(C_Sprite):
    def __init__(self,
                 x,
                 y,
                 sprite,
                 name_instance,
                 team='defender',
                 hp=1,
                 hp_max=100,
                 wall_hp=100,
                 strength=18,
                 strength_ranged=0,
                 ranged_combat=False,
                 heal=False,
                 dug_in = 0):
        self.wall_hp = wall_hp
        self.ranged_combat = ranged_combat
        self.heal = heal

        super().__init__(x,
                         y,
                         sprite,
                         name_instance,
                         team,
                         hp,
                         hp_max,
                         strength,
                         strength_ranged, 
                         dug_in)

    def take_turn(self):

        global GAME_OBJECTS, GAME_MAP

        if self.ranged_combat:
            # Check for a creature that is within two tiles
            items_within_range = []
            for ii in range(-2, 2):
                for jj in range(-2, 2):
                    temp = map_check_for_creatures(self.x - ii, self.y - jj, self)
                    # print('checked location', self.x - ii, self.y - jj)
                    if temp:
                        # print(temp.__class__, ' within range')
                        if temp.__class__ == C_Unit:
                            # --- Only add alive units
                            if temp.alive:
                                #print('found {} at {} {}'.format(
                                #    temp.name_instance,
                                #    self.x - ii,
                                #    self.y - jj))
                                items_within_range.append(temp)

            # --- Attack a random creature
            if len(items_within_range) > 0:
                rand_numb = random.randint(0, len(items_within_range) - 1)
                damage_output, damage_taken = attack(self, items_within_range[rand_numb], ranged=True)

                # City should not take any damage for the ranged combat
                items_within_range[rand_numb].take_damage(damage_output)

        if self.heal:
            temp = GAME_MAP.grid[(self.x, self.y)].get_neighbors(GAME_MAP.grid)
            tiles_within_range = []
            for ii in range(len(temp)):
                tiles_within_range.append(temp[ii].index)
            # Check to make sure there are three enemy unit within one tile
            items_within_range = []
            for obj in GAME_OBJECTS:
                for tile in tiles_within_range:
                    if obj.x == tile[0] and obj.y == tile[1] and obj.alive and obj.team != self.team:
                        #print(f'position {obj.x} {obj.y} {obj.name_instance}')
                        items_within_range.append(temp)


            # --- Heal if less than 3 tiles are occupied
            if len(items_within_range) < 3:
                self.hp += 10
                self.status = 'healed'
                if self.hp > self.hp_max:
                    self.hp = self.hp_max


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
        '''On death, most citys stop moving.'''
        #print(self.name_instance + ' has been defeated!')
        self.alive = False
        self.status = 'dead'


def attack(aggressor,
           target):
    '''Base attack definition using the formula found on CivFanatics
    TODO: Attack accounting for walls....'''

    defense_bonus = target.dug_in * 3

    if aggressor.strength_ranged > 0:
        strength_diff = np.round(aggressor.strength_ranged - aggressor.hp / 10) - np.round(target.strength - target.hp / 10 + defense_bonus)
        damage_taken = 0
    else:
        strength_diff = np.round(aggressor.strength - aggressor.hp / 10) - np.round(target.strength - target.hp / 10 + defense_bonus)
        damage_taken = np.round(30 * np.exp(-strength_diff * 0.04) * (random.randint(75, 125) / 100.0))

    damage_out = np.round(30 * np.exp(strength_diff * 0.04) * (random.randint(75, 125) / 100.0))
    

    return damage_out, damage_taken


# ---------------------------------------------
# MAP
# ---------------------------------------------
def map_create():

    new_map = class_hex.HexMap(constants.MAP_HEIGHT,
                                constants.MAP_WIDTH,
                               (constants.HEX_SIZE, constants.HEX_SIZE),
                                constants.EDGE_OFFSET)
    # TODO: block the edge tiles!


    return new_map


def map_check_for_creatures(x,
                            y,
                            actor):
    target = None

    # --- check objectlist to find creature at that location that isn't the actor
    for object in GAME_OBJECTS:
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
def draw_game():
    global SURFACE_MAIN, episode_number, TURN_NUMBER

    # --- Clear the surface
    SURFACE_MAIN.fill(constants.COLOR_DEFAULT_BG)

    # --- Draw the map
    draw_map(GAME_MAP)

    # --- Draw the objects
    for obj in GAME_OBJECTS:
        obj.draw()

    # --- Draw the text
    for obj in GAME_OBJECTS:
        # Draw the HP above the unit
        color = {'attacker': constants.COLOR_RED,
                'defender': constants.COLOR_BLUE}
        draw_text(SURFACE_MAIN, "{:.0f}/{:.0f}".format(obj.hp, obj.hp_max),
                  (GAME_MAP.grid[(int(obj.x), int(obj.y))].rect.x + GAME_MAP.grid[(int(obj.x), int(obj.y))].rect.width / 2,
                   GAME_MAP.grid[(int(obj.x), int(obj.y))].rect.y - GAME_MAP.grid[(int(obj.x), int(obj.y))].rect.h*2/7),
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
    draw_text(SURFACE_MAIN, f'Turn: {TURN_NUMBER}',
              (constants.EDGE_OFFSET,
               constants.HEX_SIZE * constants.MAP_HEIGHT - constants.EDGE_OFFSET + constants.HEX_SIZE / 7),
              constants.COLOR_LIGHT_GREY, outline=True, center=False, font_big = True)

    # --- Update game display
    pygame.display.flip()


def draw_map(map_to_draw):

    for loc in GAME_MAP.grid:
        if SURFACE_MAIN is not None:
            SURFACE_MAIN.blit(GAME_MAP.grid[loc].image, (GAME_MAP.grid[loc].rect.x, GAME_MAP.grid[loc].rect.y))
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

#   _______      ___      .___  ___.  _______
#  /  _____|    /   \     |   \/   | |   ____|
# |  |  __     /  ^  \    |  \  /  | |  |__
# |  | |_ |   /  /_\  \   |  |\/|  | |   __|
# |  |__| |  /  _____  \  |  |  |  | |  |____
#  \______| /__/     \__\ |__|  |__| |_______|
class Game():
    def __init__(self,
                 human=False,
                 ml_ai=False,
                 render = False):
        self.human = human
        self.ml_ai = ml_ai
        self.quit = False
        self.render = render

        global GAME_OBJECTS

        # initialize pygame
        if self.render or self.human:
            pygame.init()

    def game_main_loop(self,
                       action=0):
        """In this function we loop the main game."""
        game_quit = False

        # --- player action definition
        player_action = 'no-action'

        while not game_quit:

            # --- handle player input
            if self.human:
                player_action = self.game_handle_keys_human(GAME_OBJECTS[0])

            if player_action == 'QUIT':
                game_quit = True


            # --- draw the game
            if self.render:
                draw_game()

            #CLOCK = pygame.time.Clock()
            #CLOCK.tick(constants.GAME_FPS)

        if self.render:
            pygame.quit()

        exit()

    def step(self,
            attacker_action_input=0,
            defender_action_input=0):
        """In this function we take a step for both agent in the main game."""


        global CITY_OBJECTS, TURN_NUMBER

        TURN_NUMBER += 1

        # --- draw the game
        if self.render:
            draw_game()

        # --- agent action definition
        attacker_action = 'no-action'
        defender_action = 'no-action'
        game_quit = False
        attacker_reward = 0
        defender_reward = 0

        if self.ml_ai and not game_quit:
            # --- Attacker agent turn
            attacker_action = self.game_handle_moves_ml_ai('attacker', attacker_action_input)
            # --- Defender agent turn
            defender_action = self.game_handle_moves_ml_ai('defender', defender_action_input)
            all_attacker_dead = all(obj.hp <= 0 for obj in ATTACKER_OBJECTS)
            if all_attacker_dead:
                print("All attacker units are dead")
                game_quit = True
                
            for obj in CITY_OBJECTS:
                # Check to see if the city is dead or not
                if obj.hp <= 0:
                    print("City is destroyed")
                    game_quit = True

                # Attempt to heal the city otherwise
                obj.take_turn()

            if self.render:
                draw_game()         

        if attacker_action == 'QUIT' or defender_action == 'quit':
            game_quit = True

        # --- Get rewards after the city attacks, in case a unit dies
        attacker_reward += self.get_rewards('attacker')
        defender_reward += self.get_rewards('defender')

        # --- Agent reward for the turn
        attacker_reward -= 0.5
        
        #CLOCK = pygame.time.Clock()
        #CLOCK.tick(constants.GAME_FPS)

        return self.get_observation(), attacker_reward, defender_reward, game_quit

    def get_rewards(self, team):
        '''This definition will return the attacker agent reward status for each step as
        well as the location of the city relative to the attacker agent units'''
        global CITY_OBJECTS, ATTACKER_OBJECTS, DEFENDER_OBJECTS

        own_objects = {'attacker': ATTACKER_OBJECTS,
                        'defender': DEFENDER_OBJECTS}
        enemy_objects = {'attacker': DEFENDER_OBJECTS,
                        'defender': ATTACKER_OBJECTS}
        reward = 0

        
        for obj in enumerate(CITY_OBJECTS):
                city_loc = obj[0]
                if team == 'attacker':
                    # --- Rewards for city status
                        if obj[1].status == 'dead':
                            reward += 10
                            obj[1].status = None
                        elif obj[1].status == 'took damage':
                            reward += 0.5
                            obj[1].status = obj[1].status_default
                        elif obj[1].status == 'healed':
                            reward -= 0.3
                            obj[1].status = obj[1].status_default          

        # --- REWARDS for own unit status
        for obj in own_objects[team]:
            #print('BEFORE: {} status of {}'.format(obj.name_instance, obj.status))
            if obj.status == 'dead' and team == 'attacker':
                reward -= 1
                obj.status = None
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

            if team == 'attacker':
                # --- Rewards for how far they are away from the city!
                # - This is a linear reward, 0 for being next to city, -0.5 for maximum distance, per unit
                dist = hex_distance([obj.x, obj.y], [CITY_OBJECTS[city_loc].x,CITY_OBJECTS[city_loc].y])
                dist_reward = float(dist - 1) / (max([constants.MAP_HEIGHT, constants.MAP_WIDTH]) - 2)
                reward -= dist_reward / 0.5
        
        # --- REWARDS for opponent unit status
            for obj in enemy_objects[team]:
                #print('BEFORE: {} status of {}'.format(obj.name_instance, obj.status))
                if obj.status == 'dead':
                    reward += 1
                    obj.status = None


        return reward

    def get_observation(self):
        '''Definition returns the known universe
        positions of each unit and each city
        '''

        global GAME_OBJECTS

        # --- Find the distance between the unit and the city
        loc = -1 # position around the city -1 if not by, 0/8, 1/8, ..., 8/8 otherwise
        city_loc = -1

        observation = [] # city health, dx unit 1, dy unit 1, hp_norm unit 1, dx unit 2, dy unit 2, hp_norm unit 2, ...

        # --- Find the city location in GAME_OBJECTS
        for obj in enumerate(GAME_OBJECTS):
            if obj[1].__class__ == C_City:
                city_loc = obj[0]
                observation.append(obj[1].hp / obj[1].hp_max)

        # --- Find the space between each unit and the city
        for obj in GAME_OBJECTS:
            if obj.__class__ == C_Unit:
                dx_norm = (GAME_OBJECTS[city_loc].x - obj.x) / constants.MAP_WIDTH
                dy_norm = (GAME_OBJECTS[city_loc].y - obj.y) / constants.MAP_HEIGHT
                observation.append(dx_norm)
                observation.append(dy_norm)
                # --- Normalized HP
                observation.append(obj.hp / obj.hp_max)
        
        return np.array(observation)

    def get_current_state(self):
        """Use this to get unit position as well as health, used for rendering in Blender"""
        global GAME_OBJECTS
        temp_data = {}
        for obj in GAME_OBJECTS:
            temp_data[obj.name_instance] = {}
            temp_data[obj.name_instance]['health'] = obj.hp
            temp_data[obj.name_instance]['position'] = [obj.x, obj.y]

        return temp_data


    def game_initialize(self,
                        ep_number = 0):
        """This function initializes the main window, and pygame"""

        global SURFACE_MAIN, GAME_MAP, PLAYER, ENEMY, GAME_OBJECTS, CITY_OBJECTS, DEFENDER_OBJECTS, ATTACKER_OBJECTS, episode_number, TURN_NUMBER#, CLOCK
        #self.episode_number = episode_number

        episode_number = ep_number
        TURN_NUMBER = 0

        # --- Set sufrace dimensions
        if self.render:
            SURFACE_MAIN = pygame.display.set_mode((constants.MAP_WIDTH
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
        else:
            SURFACE_MAIN = None

        # --- Create the game map. Fills the dictionary with values for each tile
        GAME_MAP = map_create()

        # --- Agents main starting location
        ATTACKER_LOCATION = [6, 4]
        DEFENDER_LOCATION = [2, 4]


        # --- Attacker units
        A_WARRIOR_1 = C_Unit(ATTACKER_LOCATION[0],
                        ATTACKER_LOCATION[1]-1,
                        constants.S_WARRIOR,
                        "A_warrior_1",
                        team='attacker',
                        strength=20,
                        hp=100,
                        hp_max=100)

        A_WARRIOR_2 = C_Unit(ATTACKER_LOCATION[0]+1,
                        ATTACKER_LOCATION[1],
                        constants.S_WARRIOR,
                        "A_warrior_2",
                        team='attacker',
                        strength=20,
                        hp=100,
                        hp_max=100)

        A_WARRIOR_3 = C_Unit(ATTACKER_LOCATION[0],
                        ATTACKER_LOCATION[1]+1,
                        constants.S_WARRIOR,
                        "A_warrior_3",
                        team='attacker',
                        strength=20,
                        hp=100,
                        hp_max=100)

        A_SLINGER_1 = C_Unit(ATTACKER_LOCATION[0]+1,
                        ATTACKER_LOCATION[1]+1,
                        constants.S_SLINGER,
                        "A_slinger_1",
                        team='attacker',
                        strength=5,
                        strength_ranged=15,
                        hp=100,
                        hp_max=100)

        A_SLINGER_2 = C_Unit(ATTACKER_LOCATION[0]+1,
                        ATTACKER_LOCATION[1]-1,
                        constants.S_SLINGER,
                        "A_slinger_2",
                        team='attacker',
                        strength=5,
                        strength_ranged=15,
                        hp=100,
                        hp_max=100)
        # --- Defender units
        D_WARRIOR_1 = C_Unit(DEFENDER_LOCATION[0]+1,
                        DEFENDER_LOCATION[1]-1,
                        constants.S_WARRIOR,
                        "D_warrior_1",
                        team='defender',
                        strength=20,
                        hp=100,
                        hp_max=100)

        D_WARRIOR_2 = C_Unit(DEFENDER_LOCATION[0]+1,
                        DEFENDER_LOCATION[1],
                        constants.S_WARRIOR,
                        "D_warrior_2",
                        team='defender',
                        strength=20,
                        hp=100,
                        hp_max=100)

        D_SLINGER_1 = C_Unit(DEFENDER_LOCATION[0]+1,
                        DEFENDER_LOCATION[1]+1,
                        constants.S_SLINGER,
                        "D_slinger_1",
                        team='defender',
                        strength=5,
                        strength_ranged=15,
                        hp=100,
                        hp_max=100)

        CITY = C_City(constants.LOC_CITY[0],
                      constants.LOC_CITY[1],
                      constants.S_CITY,
                      "City",
                      team='defender',
                      hp=100,
                      strength=28,
                      ranged_combat=False,
                      heal=True)

        # Must have units first then the city last!!!
        CITY_OBJECTS = [CITY]
        ATTACKER_OBJECTS = [A_WARRIOR_1, A_WARRIOR_2, A_WARRIOR_3, A_SLINGER_1, A_SLINGER_2]
        DEFENDER_OBJECTS = [D_WARRIOR_1, D_WARRIOR_2, D_SLINGER_1]
        GAME_OBJECTS = ATTACKER_OBJECTS + DEFENDER_OBJECTS + CITY_OBJECTS


    def game_handle_keys_human(self,
                               object):

        # --- check to see if the y coordinate is even or odd
        if object.y % 2 == 0:
            parity = 'EVEN'
            even = True
        else:
            parity = 'ODD'
            even = False

        # get player input
        events_list = pygame.event.get()

        # process input
        for event in events_list:  # loop through all events that have happened
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return 'QUIT'

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    object.move(constants.MOVEMENT_DIR['NW'][parity][0], constants.MOVEMENT_DIR['NW'][parity][1])
                    return "player-moved"

                if event.key == pygame.K_a:
                    object.move(constants.MOVEMENT_DIR['W'][parity][0], constants.MOVEMENT_DIR['W'][parity][1])
                    return "player-moved"

                if event.key == pygame.K_z:
                    object.move(constants.MOVEMENT_DIR['SW'][parity][0], constants.MOVEMENT_DIR['SW'][parity][1])
                    return "player-moved"

                if event.key == pygame.K_e:
                    object.move(constants.MOVEMENT_DIR['NE'][parity][0], constants.MOVEMENT_DIR['NE'][parity][1])
                    return "player-moved"

                if event.key == pygame.K_d:
                    object.move(constants.MOVEMENT_DIR['E'][parity][0], constants.MOVEMENT_DIR['E'][parity][1])
                    return "player-moved"

                if event.key == pygame.K_c:
                    object.move(constants.MOVEMENT_DIR['SE'][parity][0], constants.MOVEMENT_DIR['SE'][parity][1])
                    return "player-moved"

                if event.key == pygame.K_SPACE:
                    if object.hp < PLAYER.hp_max:
                        object.hp += 10
                        if object.hp > object.hp_max:
                            object.hp = object.hp_max
                        object.status = 'healed'
                    return "player-moved"

        return 'no-action'

    def game_handle_moves_ml_ai(self,
                                team,
                                action):

        global ATTACKER_OBJECTS, DEFENDER_OBJECTS

        # --- Movement commands for attacker
        # --- Determine the parity
        if team == 'attacker':
            if ATTACKER_OBJECTS[0].y % 2 == 0:
                parity = 'EVEN'
            else:
                parity = 'ODD'
            # --- Determine the parity
            if ATTACKER_OBJECTS[1].y % 2 == 0:
                parity2 = 'EVEN'
            else:
                parity2 = 'ODD'
            # --- Determine the parity
            if ATTACKER_OBJECTS[2].y % 2 == 0:
                parity3 = 'EVEN'
            else:
                parity3 = 'ODD'
            if ATTACKER_OBJECTS[3].y % 2 == 0:
                parity4 = 'EVEN'
            else:
                parity4 = 'ODD'
            if ATTACKER_OBJECTS[4].y % 2 == 0:
                parity5 = 'EVEN'
            else:
                parity5 = 'ODD'
            # --- Make a movement
            direction = constants.MOVEMENT_FIVE_UNITS[action]
            ATTACKER_OBJECTS[0].move(constants.MOVEMENT_DIR[direction[0]][parity][0],
                                    constants.MOVEMENT_DIR[direction[0]][parity][1])
            ATTACKER_OBJECTS[1].move(constants.MOVEMENT_DIR[direction[1]][parity2][0],
                                    constants.MOVEMENT_DIR[direction[1]][parity2][1])
            ATTACKER_OBJECTS[2].move(constants.MOVEMENT_DIR[direction[2]][parity3][0],
                                    constants.MOVEMENT_DIR[direction[2]][parity3][1])
            ATTACKER_OBJECTS[3].move(constants.MOVEMENT_DIR[direction[3]][parity4][0],
                                    constants.MOVEMENT_DIR[direction[3]][parity4][1])
            ATTACKER_OBJECTS[4].move(constants.MOVEMENT_DIR[direction[4]][parity5][0],
                                    constants.MOVEMENT_DIR[direction[4]][parity5][1])
        else:
            if DEFENDER_OBJECTS[0].y % 2 == 0:
                parity = 'EVEN'
            else:
                parity = 'ODD'
            # --- Determine the parity
            if DEFENDER_OBJECTS[1].y % 2 == 0:
                parity2 = 'EVEN'
            else:
                parity2 = 'ODD'
            # --- Determine the parity
            if DEFENDER_OBJECTS[2].y % 2 == 0:
                parity3 = 'EVEN'
            else:
                parity3 = 'ODD'
            # --- Make a movement
            direction = constants.MOVEMENT_THREE_UNITS[action]
            DEFENDER_OBJECTS[0].move(constants.MOVEMENT_DIR[direction[0]][parity][0],
                                    constants.MOVEMENT_DIR[direction[0]][parity][1])
            DEFENDER_OBJECTS[1].move(constants.MOVEMENT_DIR[direction[1]][parity2][0],
                                    constants.MOVEMENT_DIR[direction[1]][parity2][1])
            DEFENDER_OBJECTS[2].move(constants.MOVEMENT_DIR[direction[2]][parity3][0],
                                    constants.MOVEMENT_DIR[direction[2]][parity3][1])

        return "player-moved"



if __name__ == "__main__":
    env = Game(human = True, render = True)
    env.game_initialize()
    env.game_main_loop(1)
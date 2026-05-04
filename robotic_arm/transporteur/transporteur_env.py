import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import math

class RoboticArmTransporteurEnv(gym.Env):
    
    def __init__(self, segment_lengths=[1.0, 1.0]):
        super(RoboticArmTransporteurEnv, self).__init__()

        self.segment_lengths = np.array(segment_lengths) # Longueurs des segments du bras
        self.number_links = len(self.segment_lengths) # Nombre de segments (= nombre d'articulations)
        
        max_action = 0.3 # Limite maximale de changement d'angle par step

        # L'action_space c'est comme la manette de contrôle qu'on donne à l'IA
        # spaces.Discrete pour un nombre fini de choix (boutons)
        # spaces.Box pour un espace continu (joystick)
        self.action_space = spaces.Box( # Espace d'action : un vecteur de changements d'angles pour chaque articulation
            low=-max_action, 
            high=max_action, 
        # Ici l'agent va choisir une valeur décimal précise entre entre -0.1 et 0.1
        # On borne car une IA non entrainée pourrais faire 500 tours en une fraction de seconde
        # Ce qui n'est pas réaliste et qui casserait la simulation physique
            shape=(self.number_links + 1,), # Le nombre de valeurs à choisir (une pour chaque articulation + une pour la pince)
            dtype=np.float32 # Le type de ces valeurs
        )
        
        max_reach = sum(self.segment_lengths) # Distance maximale de la cible que le bras peut atteindre
        
        # La distance maximale d'un vecteur (d'un bout à l'autre de la zone) est d'environ 2 * max_reach
        max_dist = max_reach * 2.0 

        # Limites minimales de l'observation (angles, 2 vecteurs relatifs, has_object, action_precedente_brute)
        obs_low = np.array(
            [-np.pi] * self.number_links + 
            [-max_dist] * 4 + 
            [0.0] + 
            [-max_action] * (self.number_links + 1)
        ).astype(np.float32)

        # Limites maximales de l'observation
        obs_high = np.array(
            [np.pi] * self.number_links + 
            [max_dist] * 4 + 
            [1.0] + 
            [max_action] * (self.number_links + 1)
        ).astype(np.float32)
        
        # L'observation_space c'est ce que l'IA voit à chaque étape. C'est un espace continu de valeurs possibles
        self.observation_space = spaces.Box(
            low=obs_low, 
            high=obs_high, 
            dtype=np.float32
        )

        self.angles = np.zeros(self.number_links) # Angles initiaux de chaque articulation
        self.object_pos = np.array([0.0, 0.0]) # Position de l'objet cible initiale (x, y)
        self.drop_zone_pos = np.array([0.0, 0.0]) # Position de la zone de dépôt initiale (x, y)
        
        self.previous_action = np.zeros(self.number_links + 1) # Stocke l'action lissée pour la physique
        # Stocke l'action brute (intention de l'IA) pour le calcul du jitter
        self.previous_raw_action = np.zeros(self.number_links + 1) 
        
        self.has_object = False # Indique si le bras tient actuellement l'objet
        self.difficulty = 0.1 # Difficulté initiale pour le curriculum learning
        
        # Variables pour stocker les distances au step précédent (utile pour les récompenses Delta)
        self.previous_distance_to_object = 0.0
        self.previous_distance_to_drop = 0.0

        self.render_mode = "human" # Mode de rendu (affichage à l'écran)
        # Le mode par défaut de Gymnasium est "none", surtout utilisé pour l'entraînement sans affichage.
        # Ici on veut voir le bras bouger, donc on utilise "human" pour afficher une fenêtre et brider le nombre de frames par seconde.

        self.window_size = 600 # Taille de la fenêtre d'affichage
        self.window = None # Fenêtre Pygame (sera créée au premier appel de render)
        self.clock = None # Horloge pour limiter le nombre de frames par seconde
        self.scale = (self.window_size / 2) / (max_reach * 1.2) # Facteur de conversion entre les coordonnées physiques (mètres) et les pixels pour l'affichage

        self.gfx_config = {
            'section':     {'path': '../img/arm-section.png',    'scale': 0.3, 'pivot_x': 80, 'pivot_y': 250},
            'last_section':{'path': '../img/arm-section.png',    'scale': 0.3, 'pivot_x': 80, 'pivot_y': 250, 'crop_percent': 0.72},
            'claw_open':   {'path': '../img/arm-open-claw.png',  'scale': 0.2, 'pivot_x': 420, 'pivot_y': 260},
            'claw_closed': {'path': '../img/arm-close-claw.png', 'scale': 0.2, 'pivot_x': 420, 'pivot_y': 260},
        }
        self.loaded_images = {} # Dictionnaire pour stocker les images chargées

    def set_difficulty(self, difficulty):
        self.difficulty = np.clip(difficulty, 0.1, 1.0)

    def get_end_arm_pos(self):
        """Calcule la position (x, y) du bout du bras en utilisant la Cinématique Directe."""
        
        # La base du robot est fixée à l'origine (centre de l'écran/monde)
        x, y = 0.0, 0.0 
        
        # L'angle global par rapport au point d'origine (0 horizontal)
        cumulative_angle = 0.0 
        
        # On parcourt chaque segment du bras, de la base jusqu'au bout
        for i in range(self.number_links):
            
            # On ajoute l'angle du moteur actuel pour trouver l'orientation 
            # réelle (globale) de ce segment dans l'espace.
            cumulative_angle += self.angles[i]
            
            # Trigonométrie : On calcule la projection en X et Y de ce segment
            # et on l'ajoute à la position de l'articulation précédente.
            x += self.segment_lengths[i] * np.cos(cumulative_angle)
            y += self.segment_lengths[i] * np.sin(cumulative_angle)
            
        # On renvoie les coordonnées finales (l'effecteur)
        return np.array([x, y])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        
        self.angles = np.zeros(self.number_links) 
        max_reach = sum(self.segment_lengths) 
        
        self.has_object = False # On lâche l'objet au début de l'épisode

        # Positionnement de l'objet (avec curriculum learning)
        while True:
            rx = self.np_random.uniform(-max_reach, max_reach)
            ry = self.np_random.uniform(-max_reach, max_reach)
            if np.sqrt(rx**2 + ry**2) <= max_reach:
                x = max_reach + (rx - max_reach) * self.difficulty
                y = 0.0 + (ry - 0.0) * self.difficulty
                self.object_pos = np.array([x, y])
                break
                
        # Positionnement de la zone de dépôt (avec curriculum learning)
        while True:
            rx = self.np_random.uniform(-max_reach, max_reach)
            ry = self.np_random.uniform(-max_reach, max_reach)
            # On s'assure que la zone de dépôt n'est pas générée directement sur l'objet
            if np.sqrt(rx**2 + ry**2) <= max_reach and np.linalg.norm(np.array([rx, ry]) - self.object_pos) > 0.5:
                x = max_reach + (rx - max_reach) * self.difficulty
                y = 0.0 + (ry - 0.0) * self.difficulty
                self.drop_zone_pos = np.array([x, y])
                break

        end_pos = self.get_end_arm_pos() # On calcule la position du bout du bras au reset pour l'inclure dans l'observation initiale.
        
        self.current_step = 0
        self.previous_action = np.zeros(self.number_links + 1) # Réinitialisation de l'action précédente au reset
        self.previous_raw_action = np.zeros(self.number_links + 1) # Réinitialisation de l'action brute

        # Initialisation des distances précédentes
        self.previous_distance_to_object = np.linalg.norm(self.object_pos - end_pos)
        self.previous_distance_to_drop = np.linalg.norm(self.drop_zone_pos - self.object_pos)

        # Calcul des vecteurs relatifs
        vector_to_object = self.object_pos - end_pos
        vector_object_to_drop = self.drop_zone_pos - self.object_pos
        
        # On concatène toutes les variables pour la nouvelle observation
        obs = np.concatenate([
            self.angles, 
            vector_to_object, 
            vector_object_to_drop, 
            [float(self.has_object)],
            self.previous_raw_action # On donne l'action brute à l'IA pour qu'elle comprenne le jitter
        ]).astype(np.float32) 

        return obs, {} 
    
    def step(self, action):
        self.current_step += 1
        
        motors_action = action[:-1] 
        pince_action = action[-1] 

        # Lissage de l'action (Inertie)
        smoothed_action = 0.2 * motors_action + 0.8 * self.previous_action[:-1]
        self.angles += smoothed_action
        self.angles = (self.angles + np.pi) % (2 * np.pi) - np.pi 
        
        end_pos = self.get_end_arm_pos() 
        distance_to_object = np.linalg.norm(end_pos - self.object_pos) 
        
       # -----------------------------------------------------
        # AMÉLIORATION 1 : LA DEADZONE DE LA PINCE (CORRIGÉE !)
        # -----------------------------------------------------
        was_holding_object = self.has_object 
        
        # Le réseau sort des valeurs entre -0.3 et 0.3. 
        # On ferme si > 0.15, on ouvre si < -0.15.
        if pince_action > 0.15: 
            if not self.has_object and distance_to_object < 0.1:
                self.has_object = True 
        elif pince_action < -0.15: 
            self.has_object = False
            
        if self.has_object:
            self.object_pos = end_pos.copy() 

        current_distance_to_drop = np.linalg.norm(self.drop_zone_pos - self.object_pos)

        # -----------------------------------------------------
        # AMÉLIORATION 3 : L'ANTI-TRICHE ET LE BOOST TRANSPORT
        # -----------------------------------------------------
        reward = 0.0

        # 1. Le bonbon de survol (Hover)
        if not self.has_object and distance_to_object < 0.15:
            reward += 0.5 

        # 2. L'Attrape et la Pénalité de Lâché (Le correctif Anti-Triche !)
        if not was_holding_object and self.has_object:
            reward += 20.0 # Grosse récompense pour la saisie
        elif was_holding_object and not self.has_object:
            reward -= 20.0 # Pénalité MASSIVE si elle lâche en cours de route !

        # 3. La Navigation (Deltas)
        if not self.has_object:
            # Phase d'approche
            delta_dist = self.previous_distance_to_object - distance_to_object
            reward += delta_dist * 50.0 
        else:
            # Phase de transport
            delta_dist = self.previous_distance_to_drop - current_distance_to_drop
            # On DOUBLE la récompense de delta pendant le transport !
            reward += delta_dist * 100.0 

        # Mise à jour des distances pour le prochain step
        self.previous_distance_to_object = distance_to_object
        self.previous_distance_to_drop = current_distance_to_drop

        # ... (Les pénalités d'énergie et de jitter restent en dessous comme avant)
        energy_penalty = 0.05 * np.sum(np.square(motors_action))
        jitter_penalty = 0.1 * np.sum(np.square(action[:-1] - self.previous_raw_action[:-1]))
        
        reward = reward - energy_penalty - jitter_penalty
        
        # ... (La sauvegarde de l'action)
        self.previous_action = np.concatenate([smoothed_action, [pince_action]])
        self.previous_raw_action = action.copy()
        
        terminated = False
        truncated = False 

        # 4. Le Pactole de Victoire
        if self.has_object and current_distance_to_drop < 0.1:
            reward += 100.0 # On augmente la prime de victoire !
            terminated = True
            
        # Calcul des vecteurs relatifs pour la nouvelle observation
        vector_to_object = self.object_pos - end_pos
        vector_object_to_drop = self.drop_zone_pos - self.object_pos
            
        # Nouvelle structure d'observation
        obs = np.concatenate([
            self.angles, 
            vector_to_object, 
            vector_object_to_drop, 
            [float(self.has_object)],
            self.previous_raw_action
        ]).astype(np.float32)
        
        return obs, reward, terminated, truncated, {
            "pince_action": pince_action,
            "has_object": self.has_object,
            "energy_penalty": energy_penalty, 
            "jitter_penalty": jitter_penalty
        } 
    
    def draw_image_with_pivot(self, surface, image, pos_screen, pivot_image, angle_degrees):
        """ Dessine une image tournée autour d'un pivot spécifique. """
        image_rect = image.get_rect()
        pivot_vector = pygame.math.Vector2(pivot_image) - image_rect.center
        rotated_image = pygame.transform.rotate(image, angle_degrees)
        rotated_pivot_vector = pivot_vector.rotate(-angle_degrees)
        rotated_rect = rotated_image.get_rect(center=pos_screen - rotated_pivot_vector)
        surface.blit(rotated_image, rotated_rect)

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("RL Robot Arm - Pick & Place")
            self.clock = pygame.time.Clock()

        # Chargement des images si ce n'est pas encore fait
        if not self.loaded_images:
            for key, config in self.gfx_config.items():
                try:
                    img = pygame.image.load(config['path']).convert_alpha()
                    
                    # --- NOUVEAU : Rognage (Crop) de l'image si demandé ---
                    if 'crop_percent' in config:
                        w, h = img.get_size()
                        new_w = max(1, int(w * config['crop_percent']))
                        # subsurface permet de ne garder qu'une portion de l'image (de x=0 à x=new_w)
                        img = img.subsurface((0, 0, new_w, h)).copy()

                    # Mise à l'échelle
                    if config['scale'] != 1.0:
                        w, h = img.get_size()
                        img = pygame.transform.smoothscale(img, (int(w * config['scale']), int(h * config['scale'])))
                    
                    self.loaded_images[key] = img
                except FileNotFoundError:
                    print(f"Attention: {config['path']} introuvable. Remplacement par un rectangle magenta.")
                    surf = pygame.Surface((100, 20), pygame.SRCALPHA)
                    surf.fill((255, 0, 255))
                    self.loaded_images[key] = surf

        self.window.fill((255, 255, 255))

        def to_pixels(x, y):
            px = int(self.window_size / 2 + x * self.scale)
            py = int(self.window_size / 2 - y * self.scale)
            return pygame.math.Vector2(px, py)

        # 1. Dessiner la zone de dépôt (cercle rouge creux)
        drop_px = to_pixels(self.drop_zone_pos[0], self.drop_zone_pos[1])
        pygame.draw.circle(self.window, (255, 0, 0), (int(drop_px.x), int(drop_px.y)), 15, width=3)

        # 2. Dessiner l'objet (cercle bleu)
        obj_px = to_pixels(self.object_pos[0], self.object_pos[1])
        if self.has_object:
            pygame.draw.circle(self.window, (0, 200, 255), (int(obj_px.x), int(obj_px.y)), 10) 
        else:
            pygame.draw.circle(self.window, (0, 100, 255), (int(obj_px.x), int(obj_px.y)), 10)

        # Calculer les positions globales des articulations
        joint_positions = [(0.0, 0.0)] 
        cumulative_angle = 0.0
        
        x, y = 0.0, 0.0
        for i in range(self.number_links):
            cumulative_angle += self.angles[i]
            x += self.segment_lengths[i] * np.cos(cumulative_angle)
            y += self.segment_lengths[i] * np.sin(cumulative_angle)
            joint_positions.append((x, y))

        # 4. Dessiner les SEGMENTS du bras
        cumulative_angle = 0.0
        for i in range(self.number_links):
            cumulative_angle += self.angles[i]
            angle_deg = math.degrees(cumulative_angle)
            
            p_joint = to_pixels(*joint_positions[i])
            
            key = 'last_section' if i == self.number_links - 1 else 'section'
            img_section = self.loaded_images[key]
            
            pivot_section = (self.gfx_config[key]['pivot_x'] * self.gfx_config[key]['scale'], 
                             self.gfx_config[key]['pivot_y'] * self.gfx_config[key]['scale'])
            
            self.draw_image_with_pivot(self.window, img_section, p_joint, pivot_section, angle_deg)
        # 5. Dessiner la PINCE (ouverte ou fermée)
        # La pince est fermée si on tient l'objet OU si l'action IA précédente demandait la fermeture (>0.15)
        is_claw_closed = self.has_object or self.previous_raw_action[-1] > 0.15
        claw_key = 'claw_closed' if is_claw_closed else 'claw_open'
        
        claw_img = self.loaded_images[claw_key]
        claw_pivot = (self.gfx_config[claw_key]['pivot_x'] * self.gfx_config[claw_key]['scale'], 
                      self.gfx_config[claw_key]['pivot_y'] * self.gfx_config[claw_key]['scale'])
        
        p_end = to_pixels(*joint_positions[-1])
        claw_angle_deg = math.degrees(cumulative_angle) 
        
        self.draw_image_with_pivot(self.window, claw_img, p_end, claw_pivot, claw_angle_deg)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

if __name__ == "__main__":
    env = RoboticArmTransporteurEnv(segment_lengths=[1.0, 1.0])
    obs, info = env.reset()
    
    env.render() 
    
    for step in range(500):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
                
        random_action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(random_action)
        
        env.render()
        
        if terminated or truncated:
            obs, info = env.reset()
            
    env.close()

#     def render(self):
#         if self.window is None:
#             pygame.init()
#             pygame.display.init()
#             self.window = pygame.display.set_mode((self.window_size, self.window_size))
#             pygame.display.set_caption("RL Robot Arm - Pick & Place")
#             self.clock = pygame.time.Clock()

#         self.window.fill((255, 255, 255))

#         def to_pixels(x, y):
#             px = int(self.window_size / 2 + x * self.scale)
#             py = int(self.window_size / 2 - y * self.scale)
#             return (px, py)

#         # Dessiner la zone de dépôt (cercle rouge creux)
#         drop_px = to_pixels(self.drop_zone_pos[0], self.drop_zone_pos[1])
#         pygame.draw.circle(self.window, (255, 0, 0), drop_px, 15, width=3)

#         # Dessiner l'objet (carré ou cercle bleu)
#         obj_px = to_pixels(self.object_pos[0], self.object_pos[1])
#         if self.has_object:
#             pygame.draw.circle(self.window, (0, 200, 255), obj_px, 10) # Change de couleur quand attrapé
#         else:
#             pygame.draw.circle(self.window, (0, 100, 255), obj_px, 10)

#         joint_positions = [(0.0, 0.0)] 
#         cumulative_angle = 0.0
        
#         x, y = 0.0, 0.0
#         for i in range(self.number_links):
#             cumulative_angle += self.angles[i]
#             x += self.segment_lengths[i] * np.cos(cumulative_angle)
#             y += self.segment_lengths[i] * np.sin(cumulative_angle)
#             joint_positions.append((x, y))

#         for i in range(len(joint_positions) - 1):
#             p1 = to_pixels(*joint_positions[i])
#             p2 = to_pixels(*joint_positions[i+1])
            
#             pygame.draw.line(self.window, (0, 0, 0), p1, p2, 5)
#             pygame.draw.circle(self.window, (0, 0, 255), p1, 8)

#         end_effector_px = to_pixels(*joint_positions[-1])
#         pygame.draw.circle(self.window, (0, 255, 0), end_effector_px, 8)

#         pygame.display.flip()
#         self.clock.tick(30)

#     def close(self):
#         if self.window is not None:
#             pygame.quit()
#             self.window = None
#             self.clock = None

# if __name__ == "__main__":
#     env = RoboticArmTransporteurEnv(segment_lengths=[1.0, 1.0])
#     obs, info = env.reset()
    
#     env.render() 
    
#     for step in range(500):
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 env.close()
#                 exit()
                
#         random_action = env.action_space.sample() 
#         obs, reward, terminated, truncated, info = env.step(random_action)
        
#         env.render()
        
#         if terminated or truncated:
#             obs, info = env.reset()
            
#     env.close()
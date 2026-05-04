import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

class RoboticArmPointeurEnv(gym.Env):
    
    # Initialisation de l'environnement
    def __init__(self, segment_lengths=[1.0, 1.0]):
        
        super(RoboticArmPointeurEnv, self).__init__()

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
            shape=(self.number_links,), # Le nombre de valeurs à choisir (une pour chaque articulation)
            dtype=np.float32 # Le type de ces valeurs
        )
        
        max_reach = sum(self.segment_lengths) # Distance maximale de la cible que le bras peut atteindre
        
        # Limites minimales de l'observation : angles entre -pi et pi, coordonnées entre -max_reach et max_reach
        obs_low = np.array([-np.pi] * self.number_links + [-max_reach] * 4).astype(np.float32) 

        # Limites maximales de l'observation : angles entre -pi et pi, coordonnées entre -max_reach et max_reach
        obs_high = np.array([np.pi] * self.number_links + [max_reach] * 4).astype(np.float32)
        
        # L'observation_space c'est ce que l'IA voit à chaque étape. C'est un espace continu de valeurs possibles
        self.observation_space = spaces.Box(
            low=obs_low, 
            high=obs_high, 
            dtype=np.float32
        )
        
        self.angles = np.zeros(self.number_links) # Angles initiaux de chaque articulation
        self.target_pos = np.array([0.0, 0.0]) # Position cible initiale (x, y)
        self.previous_action = np.zeros(self.number_links) # Stocke l'action précédente pour calculer la pénalité de jitter
        self.difficulty = 0.1 # Difficulté initiale pour le curriculum learning

        self.render_mode = "none" # Mode de rendu (affichage à l'écran)
        # Le mode par défaut de Gymnasium est "none", surtout utilisé pour l'entraînement sans affichage.
        # Ici on veut voir le bras bouger, donc on utilise "human" pour afficher une fenêtre et brider le nombre de frames par seconde.

        self.window_size = 600 # Taille de la fenêtre d'affichage
        self.window = None # Fenêtre Pygame (sera créée au premier appel de render)
        self.clock = None # Horloge pour limiter le nombre de frames par seconde
        self.scale = (self.window_size / 2) / (max_reach * 1.2) # Facteur de conversion entre les coordonnées physiques (mètres) et les pixels pour l'affichage
        


    def set_difficulty(self, difficulty):
        """Met à jour la difficulté de l'environnement (entre 0.1 et 1.0)."""
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
    
    # La fonction reset() est appelée au début de chaque épisode pour réinitialiser l'état de l'environnement
    def reset(self, seed=None, options=None):

        super().reset(seed=seed) 
        
        self.angles = np.zeros(self.number_links) # On remet tous les angles à zéro (bras tendu vers la droite)
        
        max_reach = sum(self.segment_lengths) # Distance maximale de la cible que le bras peut atteindre

        # # On génère une position cible aléatoire qui soit atteignable par le bras
        # while True:
        #     x = self.np_random.uniform(-max_reach, max_reach)
        #     y = self.np_random.uniform(-max_reach, max_reach)
        #     if np.sqrt(x**2 + y**2) <= max_reach:
        #         self.target_pos = np.array([x, y])
        #         break

        # On utilise le curriculum learning pour faire apparaître la cible progressivement plus loin à mesure que l'agent s'améliore
        while True:
            rx = self.np_random.uniform(-max_reach, max_reach)
            ry = self.np_random.uniform(-max_reach, max_reach)
            if np.sqrt(rx**2 + ry**2) <= max_reach:
                
                # On interpole entre la position de départ du bras (max_reach, 0.0)
                # et la position aléatoire (rx, ry) en fonction de la difficulté.
                # Si difficulty = 0.1 : la cible apparaît très près du bras.
                # Si difficulty = 1.0 : la cible apparaît n'importe où.
                x = max_reach + (rx - max_reach) * self.difficulty
                y = 0.0 + (ry - 0.0) * self.difficulty
                
                self.target_pos = np.array([x, y])
                break
                
        end_pos = self.get_end_arm_pos() # On calcule la position actuelle du bout du bras
        
        # L'observation initiale : les angles (tous à zéro), la position cible, et la position actuelle du bout du bras
        obs = np.concatenate([self.angles, self.target_pos, end_pos]).astype(np.float32) 
        
        self.current_step = 0
        self.previous_action = np.zeros(self.number_links) # On réinitialise l'action précédente

        # On renvoie l'observation initiale et un dictionnaire d'informations (pour le debug par exemple)
        return obs, {} 

    # La fonction step() est appelée à chaque fois que l'IA prend une action pour faire avancer la simulation d'une étape
    def step(self, action):
        self.current_step += 1
        
        ### action est un vecteur de changements d'angles pour chaque articulation, par exemple [0.05, -0.02] pour 2 articulations
        ### self.angles += action # On applique les changements d'angles à l'état actuel du bras

        # Lissage de l'action : on mélange un peu l'action actuelle avec l'action précédente pour donner une inertie au mouvement du bras
        smoothed_action = 0.2 * action + 0.8 * self.previous_action
        self.angles += smoothed_action

        self.angles = (self.angles + np.pi) % (2 * np.pi) - np.pi # On s'assure que les angles restent entre -pi et pi
        
        end_pos = self.get_end_arm_pos() # On calcule la nouvelle position du bout du bras après avoir appliqué l'action

        distance = np.linalg.norm(end_pos - self.target_pos) # Distance entre la position actuelle du bout du bras et la position cible

        ######### REWARD #########

        reward = -distance # On veut minimiser la distance à la cible, donc on donne une récompense négative proportionnelle à cette distance
        
        ### energy_penalty = 1.0 * np.sum(np.square(action)) # Pénalité pour les actions trop grandes (consommation d'énergie)
        
        ### Ici on pénalise les changements brusques d'action (jitter) en comparant l'action actuelle avec l'action précédente.
        ### jitter_penalty = 2.0 * np.sum(np.square(action - self.previous_action))
        
        energy_penalty = 0.5 * np.sum(np.square(action))
        jitter_penalty = 1.0 * np.sum(np.square(action - self.previous_action))

        # On combine la récompense de base avec les pénalités pour obtenir la récompense finale que l'IA recevra à cette étape.
        reward = reward - energy_penalty - jitter_penalty
        
        ##########################
        
        ### self.previous_action = action.copy() # On stocke l'action actuelle pour la comparer à la prochaine étape

        self.previous_action = smoothed_action.copy() # On stocke l'action lissée pour la comparer à la prochaine étape
        
        terminated = False # Indique si l'épisode est terminé (par exemple si la cible est atteinte)

        # if distance < 0.1: 
        #     reward += 10.0
        #     terminated = True # L'épisode se termine si le bout du bras est à moins de 10 cm de la cible

        if distance < 0.1: 
            reward += 1.0 # Bonus continu tant que l'agent reste proche de la cible
            # On ne termine plus l'épisode pour encourager l'agent à rester sur la cible    

        truncated = False # Indique si l'épisode est tronqué (par exemple si on atteint la limite de temps ou de steps)

        if self.current_step >= 200:
            truncated = True # On tronque l'épisode après 200 étapes pour éviter les épisodes trop longs
            
        # L'observation à renvoyer après avoir appliqué l'action : les nouveaux angles, la position cible (inchangée), et la nouvelle position du bout du bras
        obs = np.concatenate([self.angles, self.target_pos, end_pos]).astype(np.float32)
        
        return obs, reward, terminated, truncated, {"distance_to_target": distance, "energy_penalty": energy_penalty, "jitter_penalty": jitter_penalty} 
    
    # La fonction render() est appelée pour afficher l'état actuel de l'environnement à l'écran
    def render(self):

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("RL Robot Arm")
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        def to_pixels(x, y):

            px = int(self.window_size / 2 + x * self.scale)
            py = int(self.window_size / 2 - y * self.scale)
            return (px, py)

        target_px = to_pixels(self.target_pos[0], self.target_pos[1])
        pygame.draw.circle(self.window, (255, 0, 0), target_px, 10)

        joint_positions = [(0.0, 0.0)] 
        cumulative_angle = 0.0
        
        x, y = 0.0, 0.0
        for i in range(self.number_links):
            cumulative_angle += self.angles[i]
            x += self.segment_lengths[i] * np.cos(cumulative_angle)
            y += self.segment_lengths[i] * np.sin(cumulative_angle)
            joint_positions.append((x, y))

        for i in range(len(joint_positions) - 1):
            p1 = to_pixels(*joint_positions[i])
            p2 = to_pixels(*joint_positions[i+1])
            
            pygame.draw.line(self.window, (0, 0, 0), p1, p2, 5)
            
            pygame.draw.circle(self.window, (0, 0, 255), p1, 8)

        end_effector_px = to_pixels(*joint_positions[-1])
        pygame.draw.circle(self.window, (0, 255, 0), end_effector_px, 8)

        pygame.display.flip()
        
        self.clock.tick(30)

    # La fonction close() est appelée pour fermer proprement la fenêtre d'affichage
    def close(self):
        """Ferme proprement la fenêtre Pygame."""
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

if __name__ == "__main__":
    env = RoboticArmPointeurEnv(segment_lengths=[1.0, 1.0])
    obs, info = env.reset()
    
    env.render() 
    
    for step in range(300):
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
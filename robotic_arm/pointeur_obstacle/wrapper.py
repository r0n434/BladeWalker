import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ObstacleWrapper(gym.Wrapper):
    """
    Un Wrapper qui ajoute des obstacles dynamiques, pénalise les collisions 
    et modifie l'observation sans altérer l'environnement d'origine.
    """
    def __init__(self, env, num_obstacles=2, obstacle_radius=0.10):
        super().__init__(env)
        self.num_obstacles = num_obstacles
        self.obstacle_radius = obstacle_radius
        
        # 1. On récupère les limites de l'observation d'origine
        orig_obs_space = self.env.observation_space
        max_reach = sum(self.env.unwrapped.segment_lengths)
        
        # 2. On étend l'espace d'observation pour inclure :
        # - Les positions (x, y) de chaque obstacle
        # - Un indicateur booléen (0.0 ou 1.0) indiquant si le bras est en collision
        low_extras = np.array([-max_reach, -max_reach] * num_obstacles + [0.0]).astype(np.float32)
        high_extras = np.array([max_reach, max_reach] * num_obstacles + [1.0]).astype(np.float32)
        
        new_low = np.concatenate([orig_obs_space.low, low_extras])
        new_high = np.concatenate([orig_obs_space.high, high_extras])
        
        self.observation_space = spaces.Box(low=new_low, high=new_high, dtype=np.float32)
        self.obstacles = []

    def reset(self, **kwargs):
        # On reset l'environnement de base
        obs, info = self.env.reset(**kwargs)
        max_reach = sum(self.env.unwrapped.segment_lengths)
        
        # Génération des positions aléatoires pour les obstacles
        self.obstacles = []
        for _ in range(self.num_obstacles):
            x = self.env.unwrapped.np_random.uniform(-max_reach, max_reach)
            y = self.env.unwrapped.np_random.uniform(-max_reach, max_reach)
            self.obstacles.append(np.array([x, y]))
        
        self.env.unwrapped.obstacles = self.obstacles
        self.env.unwrapped.obstacle_radius = self.obstacle_radius
            
        # Vérification initiale de collision
        collision = self._check_collision()
        
        # On met à jour l'observation avec les obstacles et la collision
        obs = self._get_augmented_obs(obs, collision)
        return obs, info

    def step(self, action):
        # On fait avancer l'environnement de base
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # On vérifie les collisions après le mouvement
        collision = self._check_collision()
        
        if collision:
            reward -= 5.0  # Récompense négative (pénalité) en cas de collision
            # Optionnel : décommenter la ligne ci-dessous pour forcer la fin de l'épisode au crash
            # terminated = True 
            
        info["collision"] = collision  # Utile pour le debug ou les métriques
        
        # On enrichit l'observation
        obs = self._get_augmented_obs(obs, collision)
        
        return obs, reward, terminated, truncated, info

    def _get_augmented_obs(self, base_obs, collision):
        """Fusionne l'observation de base avec les données des obstacles."""
        flat_obstacles = np.concatenate(self.obstacles) if self.obstacles else np.array([])
        return np.concatenate([base_obs, flat_obstacles, [float(collision)]]).astype(np.float32)

    def _check_collision(self):
        """Vérifie si n'importe quel segment du bras touche un obstacle."""
        angles = self.env.unwrapped.angles
        lengths = self.env.unwrapped.segment_lengths
        
        # On recalcule la position de chaque articulation
        joints = [np.array([0.0, 0.0])]
        cumulative_angle = 0.0
        x, y = 0.0, 0.0
        
        for i in range(len(lengths)):
            cumulative_angle += angles[i]
            x += lengths[i] * np.cos(cumulative_angle)
            y += lengths[i] * np.sin(cumulative_angle)
            joints.append(np.array([x, y]))
            
        # Pour chaque obstacle, on vérifie la distance avec chaque segment du bras
        for obs_center in self.obstacles:
            for i in range(len(joints)-1):
                dist = self._dist_point_segment(obs_center, joints[i], joints[i+1])
                if dist < self.obstacle_radius:
                    return True
        return False

    def _dist_point_segment(self, p, a, b):
        """Mathématiques : Calcule la distance la plus courte entre un point P et un segment de droite [AB]"""
        l2 = np.sum((a - b)**2)
        if l2 == 0: return np.linalg.norm(p - a)
        t = max(0, min(1, np.dot(p - a, b - a) / l2))
        proj = a + t * (b - a)
        return np.linalg.norm(p - proj)
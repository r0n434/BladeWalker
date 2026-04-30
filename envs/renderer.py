import pygame
import numpy as np
from Box2D.b2 import polygonShape, circleShape, edgeShape

class Box2DRenderer:
    """Gestionnaire d'affichage Pygame générique et esthétique pour environnements Box2D."""
    
    def __init__(self, window_size=(1000, 600), ppm=50.0, fps=50):
        self.window_size = window_size
        self.ppm = ppm  # Pixels Par Mètre
        self.fps = fps
        self.screen = None
        self.clock = None
        
        # Couleurs stylisées
        self.colors = {
            "sky": (135, 206, 235),
            "ground_base": (46, 139, 87),    # Vert sombre
            "ground_stripe": (60, 179, 113), # Vert clair (effet de damier/vitesse)
            "torso": (220, 90, 90),          # Rouge doux
            "limbs": (150, 150, 150),        # Gris
            "outline": (40, 40, 40),         # Bordure sombre
            "joint": (255, 215, 0)           # Jaune doré pour les moteurs
        }

    def init_pygame(self):
        """Initialise la fenêtre Pygame au premier appel."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("BladeWalker - Render Engine")
            self.clock = pygame.time.Clock()

    def render(self, env):
        """Prend un environnement en paramètre et dessine son état actuel."""
        self.init_pygame()

        # 1. Gestion des événements (pour pouvoir fermer la fenêtre)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False # Indique que l'utilisateur veut quitter

        # 2. Gérer la caméra
        # On essaie de centrer sur le torse, sinon on centre sur 0
        cam_x = 0
        if hasattr(env, 'torse') and env.torse is not None:
            cam_x = env.torse.position.x

        # 3. Dessiner le Ciel
        self.screen.fill(self.colors["sky"])

        # 4. Dessiner le Sol (avec effet de défilement)
        sol_y = int(self.window_size[1] - (0 * self.ppm) - 100) # -100 pour monter un peu le sol
        
        # Fond du sol
        pygame.draw.rect(self.screen, self.colors["ground_base"], 
                         (0, sol_y, self.window_size[0], self.window_size[1] - sol_y))
        
        # Bandes pour voir le défilement et la vitesse (Parallaxe)
        stripe_width = 40
        offset = int((cam_x * self.ppm) % (stripe_width * 2))
        for x in range(-offset, self.window_size[0], stripe_width * 2):
            pygame.draw.rect(self.screen, self.colors["ground_stripe"], 
                             (x, sol_y, stripe_width, self.window_size[1] - sol_y))
            
        # Ligne d'horizon noire
        pygame.draw.line(self.screen, self.colors["outline"], (0, sol_y), (self.window_size[0], sol_y), 4)

        # 5. Dessiner tous les corps dynamiques (le robot)
        for body in env.world.bodies:
            # On ne dessine pas les corps statiques (le sol infini) ici, on l'a déjà fait
            if body.type == 0: 
                continue 
                
            for fixture in body.fixtures:
                shape = fixture.shape
                
                # Choisir la couleur (Torse = Rouge, reste = Gris)
                color = self.colors["torso"] if hasattr(env, 'torse') and body == env.torse else self.colors["limbs"]
                
                if isinstance(shape, polygonShape):
                    self._draw_polygon(shape, body, cam_x, color)
                elif isinstance(shape, circleShape):
                    self._draw_circle(shape, body, cam_x, color)

        # 6. Dessiner les articulations (Joints) par-dessus
        for joint in env.world.joints:
            self._draw_joint(joint, cam_x)

        # 7. Rafraîchissement
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def _draw_polygon(self, polygon, body, cam_x, color):
        """Fonction utilitaire pour dessiner un polygone Box2D."""
        vertices = [(body.transform * v) for v in polygon.vertices]
        pygame_verts = []
        for v in vertices:
            # Conversion mètres -> pixels avec translation de la caméra
            px = int((v[0] - cam_x) * self.ppm + self.window_size[0] / 2)
            py = int(self.window_size[1] - (v[1] * self.ppm) - 100)
            pygame_verts.append((px, py))
        
        pygame.draw.polygon(self.screen, color, pygame_verts)
        pygame.draw.polygon(self.screen, self.colors["outline"], pygame_verts, 2) # Bordure

    def _draw_joint(self, joint, cam_x):
        """Dessine un petit point au niveau des articulations."""
        anchor = joint.anchorA
        px = int((anchor[0] - cam_x) * self.ppm + self.window_size[0] / 2)
        py = int(self.window_size[1] - (anchor[1] * self.ppm) - 100)
        
        pygame.draw.circle(self.screen, self.colors["joint"], (px, py), 4)
        pygame.draw.circle(self.screen, self.colors["outline"], (px, py), 4, 1)

    def close(self):
        """Ferme proprement Pygame."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
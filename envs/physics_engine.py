"""
physics_engine.py
-----------------
Module partagé entre les créatures Box2D/Pygame.
Fournit :
  - capsule()        : corps dynamique en forme de capsule
  - create_world()   : monde Box2D avec sol statique
  - world2screen()   : conversion coordonnées physiques → pixels
  - draw_body()      : rendu générique d'un corps (polygones + cercles)
  - draw_joint()     : pastille de visualisation d'une articulation
  - PhysicsApp       : classe de base pour la boucle principale pygame
"""

import sys
import math

import Box2D
from Box2D.b2 import world, polygonShape, circleShape, fixtureDef, revoluteJointDef
import pygame
import numpy as np


# ---------------------------------------------------------------------------
# Constantes par défaut (peuvent être surchargées dans chaque fichier-créature)
# ---------------------------------------------------------------------------
PPM = 30.0            # pixels par mètre
TARGET_FPS = 50
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700


# ---------------------------------------------------------------------------
# Physique
# ---------------------------------------------------------------------------

def capsule(b2world, position, hx, hy,
            density=1.0, friction=0.5, restitution=0.0, group_index=-1):
    """
    Crée un corps dynamique en forme de capsule (rectangle arrondi).
    Si hy >= hx : capsule verticale, sinon horizontale.
    """
    body = b2world.CreateDynamicBody(position=position)
    filtre = Box2D.b2Filter(groupIndex=group_index)

    fixture_params = dict(
        density=density,
        friction=friction,
        restitution=restitution,
        filter=filtre,
    )

    if hy >= hx:
        radius = hx
        offset = hy - hx
        rect_dims = (hx, offset)
        pos_1 = (0,  offset)
        pos_2 = (0, -offset)
    else:
        radius = hy
        offset = hx - hy
        rect_dims = (offset, hy)
        pos_1 = ( offset, 0)
        pos_2 = (-offset, 0)

    if offset > 0:
        body.CreateFixture(shape=polygonShape(box=rect_dims), **fixture_params)

    body.CreateFixture(shape=circleShape(radius=radius, pos=pos_1), **fixture_params)
    body.CreateFixture(shape=circleShape(radius=radius, pos=pos_2), **fixture_params)

    return body


def create_world(gravity=(0, -10), size_w=50.0, size_h=1.0, friction_ground=0.8):
    """Crée un monde Box2D avec un sol statique rectangulaire."""
    env_world = world(gravity=gravity, doSleep=True)
    ground = env_world.CreateStaticBody(
        position=(0, 0),
        shapes=polygonShape(box=(size_w, size_h)),
    )
    ground.fixtures[0].friction = friction_ground
    return env_world


# ---------------------------------------------------------------------------
# Rendu
# ---------------------------------------------------------------------------

def world2screen(vec, screen_w=SCREEN_WIDTH, screen_h=SCREEN_HEIGHT, ppm=PPM):
    """Convertit un vecteur Box2D (mètres) en coordonnées pixels pygame."""
    return (
        vec[0] * ppm + screen_w / 2,
        screen_h - vec[1] * ppm,
    )


def _convex_hull(points):
    """Enveloppe convexe d'un nuage de points 2-D (algorithme de Andrew)."""
    points = sorted(set((round(p[0], 4), round(p[1], 4)) for p in points))
    if len(points) <= 2:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def draw_body(screen, body, color=None, outline_color=(60, 40, 10),
              screen_w=SCREEN_WIDTH, screen_h=SCREEN_HEIGHT, ppm=PPM):
    """
    Dessine un corps Box2D.

    Si `color` est fourni, chaque fixture est remplie de cette couleur
    (style « chien »). Sinon, seul le contour convexe global est tracé
    (style « poulet »).
    """
    if color is not None:
        # Mode colorié : dessine chaque fixture séparément
        for fixture in body.fixtures:
            shape = fixture.shape
            if isinstance(shape, polygonShape):
                pts = [
                    (int(p[0]), int(p[1]))
                    for p in (
                        world2screen(body.transform * v, screen_w, screen_h, ppm)
                        for v in shape.vertices
                    )
                ]
                pygame.draw.polygon(screen, color, pts, 0)
                pygame.draw.polygon(screen, outline_color, pts, 2)
            elif isinstance(shape, circleShape):
                cx, cy = world2screen(body.transform * shape.pos, screen_w, screen_h, ppm)
                r = int(shape.radius * ppm)
                pygame.draw.circle(screen, color, (int(cx), int(cy)), r)
                pygame.draw.circle(screen, outline_color, (int(cx), int(cy)), r, 2)
    else:
        # Mode contour uniquement (enveloppe convexe)
        pts_local = []
        for fixture in body.fixtures:
            shape = fixture.shape
            if isinstance(shape, polygonShape):
                pts_local.extend(shape.vertices)
            elif isinstance(shape, circleShape):
                cx, cy = shape.pos
                r = shape.radius
                pts_local.extend(
                    (cx + r * math.cos(2 * math.pi * i / 20),
                     cy + r * math.sin(2 * math.pi * i / 20))
                    for i in range(20)
                )

        if not pts_local:
            return

        hull = _convex_hull(pts_local)
        verts = [
            (int(world2screen(body.GetWorldPoint(v), screen_w, screen_h, ppm)[0]),
             int(world2screen(body.GetWorldPoint(v), screen_w, screen_h, ppm)[1]))
            for v in hull
        ]
        if len(verts) > 2:
            pygame.draw.polygon(screen, outline_color, verts, 2)
        elif len(verts) == 2:
            pygame.draw.line(screen, outline_color, verts[0], verts[1], 2)


def draw_joint(screen, joint, color=(220, 50, 50),
               screen_w=SCREEN_WIDTH, screen_h=SCREEN_HEIGHT, ppm=PPM):
    """Pastille colorée à l'ancrage d'une articulation Box2D."""
    ax, ay = world2screen(joint.anchorA, screen_w, screen_h, ppm)
    pygame.draw.circle(screen, color, (int(ax), int(ay)), 4)


# ---------------------------------------------------------------------------
# Boucle principale : classe de base
# ---------------------------------------------------------------------------

class PhysicsApp:
    """
    Classe de base pour une application Box2D/Pygame.

    Sous-classes à implémenter :
      - build(world)          → (joints, bodies_dict, extra…)
      - update_brain(t, dt)   → met à jour les vitesses moteur des joints
      - draw_scene(screen)    → tout le rendu de la créature

    Optionnel :
      - on_event(event)       → gestion d'événements supplémentaires
    """

    TITLE = "Physics"
    BG_COLOR = (255, 255, 255)
    VELOCITY_ITERATIONS = 8
    POSITION_ITERATIONS = 3

    def __init__(self, screen_w=SCREEN_WIDTH, screen_h=SCREEN_HEIGHT):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.time = 0.0

    # ------------------------------------------------------------------
    # Interface à surcharger
    # ------------------------------------------------------------------

    def build(self, b2world):
        """Construit la créature dans le monde. À surcharger."""
        raise NotImplementedError

    def update_brain(self, t, dt):
        """Calcule les vitesses moteur à chaque frame. À surcharger."""
        pass

    def draw_scene(self, screen):
        """Dessine la créature. À surcharger."""
        raise NotImplementedError

    def on_event(self, event):
        """Gestion d'événements optionnelle."""
        pass

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption(self.TITLE)
        clock = pygame.time.Clock()

        env_world = create_world()
        self.build(env_world)
        self._world = env_world

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.on_event(event)

            self.time += TIME_STEP
            self.update_brain(self.time, TIME_STEP)
            env_world.Step(TIME_STEP,
                           self.VELOCITY_ITERATIONS,
                           self.POSITION_ITERATIONS)

            screen.fill(self.BG_COLOR)
            self.draw_scene(screen)
            pygame.display.flip()
            clock.tick(TARGET_FPS)

        pygame.quit()
        sys.exit()

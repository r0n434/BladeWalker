"""
chicken.py
----------
Créature « poule rousse » avec rendu illustré style livre d'images.
Lance : python chicken.py
"""

import math
import numpy as np
import pygame
import pygame.gfxdraw
import Box2D
from Box2D.b2 import revoluteJointDef

from physics_engine import (
    PhysicsApp, capsule,
    world2screen, PPM,
)

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
W, H = 1100, 700

# ---------------------------------------------------------------------------
# Palette « poule rousse »
# ---------------------------------------------------------------------------
SKY_TOP     = (255, 243, 220)
SKY_BOT     = (255, 218, 158)
GROUND_TOP  = (142, 110,  62)
GROUND_MID  = (110,  82,  40)
GROUND_BOT  = ( 78,  58,  28)
GRASS_A     = ( 96, 148,  48)
GRASS_B     = ( 64, 110,  30)

BODY_MAIN   = (210,  98,  30)
BODY_SHADE  = (160,  60,  12)
BODY_LIGHT  = (255, 162,  72)
BELLY       = (245, 195, 108)
NECK_COLOR  = (220, 110,  40)
WING_COLOR  = (170,  70,  18)

HEAD_COLOR  = (218,  92,  25)
BEAK_UP     = (255, 200,  60)
BEAK_LO     = (210, 148,  28)
COMB_COLOR  = (205,  38,  38)
WATTLE      = (190,  30,  30)
EYE_WHITE   = (255, 252, 235)
EYE_IRIS    = ( 55,  25,   8)
EYE_SHINE   = (255, 255, 255)

THIGH_COLOR = (185,  72,  18)
CALF_COLOR  = (195, 158,  72)
CLAW_COLOR  = (210, 170,  65)
OUTLINE     = ( 75,  38,   8)


# ---------------------------------------------------------------------------
# Helpers dessin
# ---------------------------------------------------------------------------

def w2s(vec):
    return world2screen(vec, screen_w=W, screen_h=H, ppm=PPM)


def aacircle(surf, color, center, r, width=0):
    cx, cy = int(center[0]), int(center[1])
    r = max(r, 1)
    if width == 0:
        pygame.gfxdraw.aacircle(surf, cx, cy, r, color)
        pygame.gfxdraw.filled_circle(surf, cx, cy, r, color)
    else:
        for i in range(width):
            pygame.gfxdraw.aacircle(surf, cx, cy, max(1, r - i), color)


def aapoly(surf, color, pts, border=None):
    ipts = [(int(x), int(y)) for x, y in pts]
    pygame.gfxdraw.aapolygon(surf, ipts, color)
    pygame.gfxdraw.filled_polygon(surf, ipts, color)
    if border:
        pygame.gfxdraw.aapolygon(surf, ipts, border)


def lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


# ---------------------------------------------------------------------------
# Fond
# ---------------------------------------------------------------------------

def draw_background(surf, t):
    ground_y = H - int(2.0 * PPM)

    # Ciel dégradé
    for y in range(ground_y):
        pygame.draw.line(surf, lerp(SKY_TOP, SKY_BOT, y / max(ground_y - 1, 1)),
                         (0, y), (W, y))

    # Sol dégradé
    depth = H - ground_y
    for dy in range(depth):
        r = dy / max(depth - 1, 1)
        c = lerp(GROUND_TOP, GROUND_BOT, r)
        pygame.draw.line(surf, c, (0, ground_y + dy), (W, ground_y + dy))

    # Herbe
    rng = np.random.default_rng(42)
    xs = rng.integers(0, W, 200)
    hs = rng.integers(8, 24, 200)
    sw = rng.uniform(-1.0, 1.0, 200)
    for i in range(200):
        sway = int(math.sin(t * 1.4 + sw[i] * 8) * 2.5)
        base = ground_y + 2
        col  = GRASS_B if i % 3 == 0 else GRASS_A
        pygame.draw.line(surf, col,
                         (xs[i], base),
                         (xs[i] + sway, base - hs[i]), 2)

    pygame.draw.line(surf, (100, 72, 32), (0, ground_y), (W, ground_y), 2)


# ---------------------------------------------------------------------------
# Dessin des pièces de la poule
# ---------------------------------------------------------------------------

def draw_shadow(surf, torso):
    tx, _ = w2s(torso.position)
    gy = H - int(2.0 * PPM)
    s = pygame.Surface((100, 22), pygame.SRCALPHA)
    pygame.gfxdraw.filled_ellipse(s, 50, 11, 46, 8, (50, 25, 5, 50))
    surf.blit(s, (int(tx) - 50, gy - 10))


def draw_capsule_limb(surf, body, color, shade, r_px):
    """Dessine un membre capsulaire avec shading latéral."""
    circles = [body.GetWorldPoint(f.shape.pos)
               for f in body.fixtures if hasattr(f.shape, 'pos')]
    if len(circles) < 2:
        return

    p1 = w2s(circles[0])
    p2 = w2s(circles[1])
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    ln = math.hypot(dx, dy)
    if ln < 0.5:
        return
    nx, ny = -dy / ln, dx / ln
    r = r_px

    # Corps du membre
    quad = [
        (p1[0] + nx * r, p1[1] + ny * r),
        (p2[0] + nx * r, p2[1] + ny * r),
        (p2[0] - nx * r, p2[1] - ny * r),
        (p1[0] - nx * r, p1[1] - ny * r),
    ]
    aapoly(surf, color, quad, OUTLINE)

    # Ombre (bord droit)
    shade_q = [
        (p1[0] - nx * r * 0.2, p1[1] - ny * r * 0.2),
        (p2[0] - nx * r * 0.2, p2[1] - ny * r * 0.2),
        (p2[0] - nx * r,       p2[1] - ny * r),
        (p1[0] - nx * r,       p1[1] - ny * r),
    ]
    aapoly(surf, shade, shade_q)

    # Extrémités rondes
    aacircle(surf, color, p1, r)
    aacircle(surf, color, p2, r)
    aacircle(surf, OUTLINE, p1, r, 1)
    aacircle(surf, OUTLINE, p2, r, 1)


def draw_claws(surf, calf_body):
    """Trois griffes en bas du tibia."""
    circles = [calf_body.GetWorldPoint(f.shape.pos)
               for f in calf_body.fixtures if hasattr(f.shape, 'pos')]
    if not circles:
        return
    # Point le plus bas
    bot = min(circles, key=lambda p: p[1])
    sx, sy = w2s(bot)
    ang = calf_body.angle

    for spread in (-0.40, 0.05, 0.50):
        ca = ang + spread + math.pi * 0.55
        mx = sx + math.cos(ca) * 9
        my = sy + math.sin(ca) * 9
        tx = mx + math.cos(ca) * 8
        ty = my + math.sin(ca) * 8
        pygame.draw.line(surf, CLAW_COLOR, (int(sx), int(sy)), (int(tx), int(ty)), 3)
        aacircle(surf, OUTLINE, (tx, ty), 2)
        aacircle(surf, CLAW_COLOR, (tx, ty), 2)


def draw_torso(surf, torso):
    cx, cy = w2s(torso.position)
    ang = torso.angle
    rw = int(1.65 * PPM)
    rh = int(1.15 * PPM)

    tmp = pygame.Surface((rw * 2 + 6, rh * 2 + 6), pygame.SRCALPHA)
    ox, oy = rw + 3, rh + 3

    # Corps principal
    pygame.gfxdraw.filled_ellipse(tmp, ox, oy, rw, rh, BODY_MAIN)
    # Ventre clair
    pygame.gfxdraw.filled_ellipse(tmp, ox + 4, oy + 8, int(rw * 0.60), int(rh * 0.52), BELLY)
    # Reflet
    pygame.gfxdraw.filled_ellipse(tmp, ox - 6, oy - 8, int(rw * 0.30), int(rh * 0.18), BODY_LIGHT)
    # Ombre dessous
    pygame.gfxdraw.filled_ellipse(tmp, ox, oy + int(rh * 0.6), int(rw * 0.80), int(rh * 0.28), BODY_SHADE)
    # Aile esquissée
    wing = [
        (ox - int(rw * 0.2), oy - int(rh * 0.15)),
        (ox - int(rw * 0.85), oy + int(rh * 0.05)),
        (ox - int(rw * 0.65), oy + int(rh * 0.72)),
        (ox - int(rw * 0.05), oy + int(rh * 0.55)),
    ]
    pygame.gfxdraw.aapolygon(tmp, wing, WING_COLOR)
    # Contour
    pygame.gfxdraw.aaellipse(tmp, ox, oy, rw, rh, OUTLINE)

    deg = -math.degrees(ang)
    rot = pygame.transform.rotozoom(tmp, deg, 1.0)
    rx, ry = rot.get_rect().center
    surf.blit(rot, (int(cx) - rx, int(cy) - ry))


def draw_neck(surf, neck_body):
    draw_capsule_limb(surf, neck_body, NECK_COLOR,
                      lerp(NECK_COLOR, OUTLINE, 0.28), int(0.44 * PPM))


def draw_head(surf, head_body):
    cx, cy = w2s(head_body.position)
    ang    = head_body.angle
    r      = int(0.54 * PPM)

    # --- Crête ---
    for i in range(5):
        a = ang - math.pi / 2 + (i - 2) * 0.20
        h = r * (0.65 if i % 2 == 0 else 0.38)
        bx = cx + math.cos(a) * r * 0.55
        by = cy + math.sin(a) * r * 0.55
        tx = cx + math.cos(a) * (r + h)
        ty = cy + math.sin(a) * (r + h)
        perp = a + math.pi / 2
        w2 = 5
        comb_tri = [
            (bx - math.cos(perp) * w2, by - math.sin(perp) * w2),
            (bx + math.cos(perp) * w2, by + math.sin(perp) * w2),
            (tx, ty),
        ]
        aapoly(surf, COMB_COLOR, comb_tri)

    # --- Barbillon ---
    ba = ang - math.pi * 0.25
    bx = cx + math.cos(ba) * r * 0.55
    by = cy + math.sin(ba) * r * 0.55 + r * 0.55
    aacircle(surf, WATTLE, (bx, by),      6)
    aacircle(surf, WATTLE, (bx + 3, by + 7), 4)

    # --- Tête ---
    aacircle(surf, HEAD_COLOR, (cx, cy), r)
    lx = cx + int(math.cos(ang + 0.75) * r * 0.38)
    ly = cy + int(math.sin(ang + 0.75) * r * 0.38)
    aacircle(surf, BODY_LIGHT, (lx, ly), int(r * 0.20))
    aacircle(surf, OUTLINE, (cx, cy), r, 2)

    # --- Bec ---
    bd = ang
    perp = bd + math.pi / 2
    bcx = cx + math.cos(bd) * r * 0.88
    bcy = cy + math.sin(bd) * r * 0.88
    beak_up = [
        (bcx - math.cos(perp) * 5,  bcy - math.sin(perp) * 5),
        (bcx + math.cos(bd) * 14,   bcy + math.sin(bd) * 14),
        (bcx + math.cos(perp) * 3,  bcy + math.sin(perp) * 3),
    ]
    beak_lo = [
        (bcx - math.cos(perp) * 3,     bcy - math.sin(perp) * 3),
        (bcx + math.cos(bd) * 11,      bcy + math.sin(bd) * 11 + 3),
        (bcx + math.cos(perp) * 5,     bcy + math.sin(perp) * 5),
    ]
    aapoly(surf, BEAK_UP, beak_up, OUTLINE)
    aapoly(surf, BEAK_LO, beak_lo, OUTLINE)

    # --- Œil ---
    ex = cx + int(math.cos(ang + 0.42) * r * 0.52)
    ey = cy + int(math.sin(ang + 0.42) * r * 0.52)
    aacircle(surf, EYE_WHITE, (ex, ey), 7)
    aacircle(surf, EYE_IRIS,  (ex, ey), 5)
    aacircle(surf, EYE_SHINE, (ex + 2, ey - 2), 2)
    aacircle(surf, OUTLINE,   (ex, ey), 7, 1)


def draw_chicken(surf, bodies):
    draw_shadow(surf, bodies['torso'])

    # Pattes gauches (derrière)
    draw_capsule_limb(surf, bodies['thighL'], lerp(THIGH_COLOR, (0,0,0), 0.12),
                      lerp(THIGH_COLOR, OUTLINE, 0.4), int(0.37 * PPM))
    draw_capsule_limb(surf, bodies['calfL'], lerp(CALF_COLOR, (0,0,0), 0.1),
                      lerp(CALF_COLOR, OUTLINE, 0.35), int(0.15 * PPM))
    draw_claws(surf, bodies['calfL'])

    # Corps
    draw_torso(surf, bodies['torso'])
    draw_neck(surf, bodies['neck'])

    # Pattes droites (devant)
    draw_capsule_limb(surf, bodies['thighR'], THIGH_COLOR,
                      lerp(THIGH_COLOR, OUTLINE, 0.25), int(0.37 * PPM))
    draw_capsule_limb(surf, bodies['calfR'], CALF_COLOR,
                      lerp(CALF_COLOR, OUTLINE, 0.25), int(0.15 * PPM))
    draw_claws(surf, bodies['calfR'])

    # Tête (tout devant)
    draw_head(surf, bodies['head'])


# ---------------------------------------------------------------------------
# Anatomie
# ---------------------------------------------------------------------------

def build_chicken(b2world):
    SPAWN_Y = 5.0
    TQ = 80.0
    TW, TH = 1.5, 1.0
    NW, NH = 0.5, 1.0
    HR      = 0.5
    FW, FH  = 0.5, 1.0
    CW, CH  = 0.2, 1.0

    torso = capsule(b2world, (0, SPAWN_Y), TW, TH, density=0.8, group_index=-1)
    neck  = capsule(b2world, (3*TW/4, SPAWN_Y + TH/2), NW, NH, density=0.3, group_index=-1)

    head = b2world.CreateDynamicBody(position=(3*TW/4, SPAWN_Y + TH/2 + 3*NH/4))
    head.CreateFixture(shape=Box2D.b2CircleShape(radius=HR),
                       density=0.2, friction=0.5,
                       filter=Box2D.b2Filter(groupIndex=-1))
    b2world.CreateJoint(Box2D.b2WeldJointDef(
        bodyA=neck, bodyB=head,
        localAnchorA=(NW/2, NH/2), localAnchorB=(0, 0)))

    thighR = capsule(b2world, (0, SPAWN_Y - TH/4), FW, FH, density=2.5, group_index=-1)
    thighL = capsule(b2world, (0, SPAWN_Y - TH/4), FW, FH, density=2.5, group_index=-1)
    calfR  = capsule(b2world, (0, SPAWN_Y - TH/4 - 3*FH/4), CW, CH, density=3.0, friction=0.8, group_index=-1)
    calfL  = capsule(b2world, (0, SPAWN_Y - TH/4 - 3*FH/4), CW, CH, density=3.0, friction=0.8, group_index=-1)

    def hip(th):
        return b2world.CreateJoint(revoluteJointDef(
            bodyA=torso, bodyB=th,
            localAnchorA=(0, -TH/4), localAnchorB=(0, FH/2),
            enableLimit=True, lowerAngle=-0.8, upperAngle=1.2,
            enableMotor=True, maxMotorTorque=TQ, motorSpeed=0.0))

    def knee(th, ca):
        return b2world.CreateJoint(revoluteJointDef(
            bodyA=th, bodyB=ca,
            localAnchorA=(0, -FH/2), localAnchorB=(0, CH/2),
            enableLimit=True, lowerAngle=-2.0, upperAngle=0.0,
            enableMotor=True, maxMotorTorque=TQ, motorSpeed=0.0))

    neck_j = b2world.CreateJoint(revoluteJointDef(
        bodyA=torso, bodyB=neck,
        localAnchorA=(TW/2, TH/2), localAnchorB=(0, -3*NH/4),
        enableLimit=True, lowerAngle=-1.0, upperAngle=1.0,
        enableMotor=True, maxMotorTorque=TQ, motorSpeed=0.0))

    joints = [hip(thighR), hip(thighL), knee(thighR, calfR), knee(thighL, calfL), neck_j]
    bodies = dict(torso=torso, neck=neck, head=head,
                  thighR=thighR, thighL=thighL,
                  calfR=calfR,   calfL=calfL)
    return joints, bodies


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class ChickenApp(PhysicsApp):

    TITLE = "La Poule Rousse"

    def __init__(self):
        super().__init__(screen_w=W, screen_h=H)
        self._fonts_loaded = False

    def build(self, b2world):
        joints, self.bodies = build_chicken(b2world)
        (self.hipR, self.hipL,
         self.kneeR, self.kneeL,
         self.neck_joint) = joints

    def update_brain(self, t, dt):
        f, a = 5.0, 3.0
        self.hipR.motorSpeed       =  a * np.sin(t * f)
        self.hipL.motorSpeed       =  a * np.sin(t * f + np.pi)
        self.kneeR.motorSpeed      =  a * np.sin(t * f - 1.5)
        self.kneeL.motorSpeed      =  a * np.sin(t * f + np.pi - 1.5)
        self.neck_joint.motorSpeed =  1.5 * np.sin(t * f * 2)

    def draw_scene(self, screen):
        draw_background(screen, self.time)
        draw_chicken(screen, self.bodies)
        self._draw_ui(screen)

    def _draw_ui(self, screen):
        if not self._fonts_loaded:
            pygame.font.init()
            try:
                self._f1 = pygame.font.SysFont("Georgia", 26, italic=True)
                self._f2 = pygame.font.SysFont("Georgia", 13, italic=True)
            except Exception:
                self._f1 = pygame.font.SysFont(None, 28)
                self._f2 = pygame.font.SysFont(None, 15)
            self._fonts_loaded = True

        # Panneau semi-transparent
        panel = pygame.Surface((290, 56), pygame.SRCALPHA)
        panel.fill((255, 238, 190, 150))
        pygame.draw.rect(panel, (140, 85, 28, 200),
                         panel.get_rect(), 2, border_radius=10)
        screen.blit(panel, (18, 16))

        screen.blit(self._f1.render("La Poule Rousse", True, (90, 44, 8)),  (30, 20))
        screen.blit(self._f2.render(
            f"t = {self.time:.1f}s   ·   Box2D + Pygame", True, (130, 70, 20)),
            (30, 50))


if __name__ == "__main__":
    ChickenApp().run()

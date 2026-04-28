"""
dog.py
------
Créature « chien doré » avec rendu illustré style livre d'images.
Cohérent visuellement avec chicken.py (même fond, mêmes helpers, même UI).
Lance : python dog.py
"""

import math
import numpy as np
import pygame
import pygame.gfxdraw
import Box2D
from Box2D.b2 import polygonShape, circleShape, revoluteJointDef

from physics_engine import (
    PhysicsApp, capsule, world2screen, PPM,
)

# ---------------------------------------------------------------------------
# Dimensions  (identique chicken.py)
# ---------------------------------------------------------------------------
W, H = 1100, 700

# ---------------------------------------------------------------------------
# Palette « chien doré »
# Même ciel/sol/herbe que la poule pour la cohérence de l'univers.
# ---------------------------------------------------------------------------
# -- Fond (identique chicken.py) --
SKY_TOP    = (255, 243, 220)
SKY_BOT    = (255, 218, 158)
GROUND_TOP = (142, 110,  62)
GROUND_MID = (110,  82,  40)
GROUND_BOT = ( 78,  58,  28)
GRASS_A    = ( 96, 148,  48)
GRASS_B    = ( 64, 110,  30)

# -- Corps --
FUR_MAIN   = (210, 162,  62)   # doré chaud
FUR_LIGHT  = (255, 210, 100)   # reflet solaire
FUR_SHADE  = (155, 108,  28)   # ombre pelage
FUR_BELLY  = (245, 220, 150)   # ventre crème
FUR_DARK   = (120,  78,  18)   # pattes sombres

# -- Tête --
HEAD_MAIN  = (220, 168,  65)
HEAD_LIGHT = (255, 215, 110)
MUZZLE     = (240, 200, 130)   # museau crème
NOSE       = ( 48,  32,  20)   # truffe noire
EYE_WHITE  = (255, 252, 235)
EYE_IRIS   = ( 88,  48,  12)   # iris noisette
EYE_PUPIL  = ( 22,  12,   5)
EYE_SHINE  = (255, 255, 255)

# -- Oreilles --
EAR_OUTER  = (178, 118,  38)
EAR_INNER  = (220, 168,  88)

# -- Pattes --
LEG_FRONT  = (195, 148,  52)
LEG_BACK   = (175, 128,  38)
PAW_COLOR  = (230, 190, 110)
PAD_COLOR  = (160,  88,  55)   # coussinet rosé-brun

# -- Queue --
TAIL_BASE  = (200, 155,  55)
TAIL_TIP   = (245, 215, 130)

OUTLINE    = ( 75,  38,   8)   # identique chicken.py


# ---------------------------------------------------------------------------
# Helpers (mêmes signatures que chicken.py)
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
# Fond — IDENTIQUE à chicken.py (même seed, même style)
# ---------------------------------------------------------------------------

def draw_background(surf, t):
    ground_y = H - int(2.0 * PPM)

    for y in range(ground_y):
        pygame.draw.line(surf, lerp(SKY_TOP, SKY_BOT, y / max(ground_y - 1, 1)),
                         (0, y), (W, y))

    depth = H - ground_y
    for dy in range(depth):
        r = dy / max(depth - 1, 1)
        pygame.draw.line(surf, lerp(GROUND_TOP, GROUND_BOT, r),
                         (0, ground_y + dy), (W, ground_y + dy))

    rng = np.random.default_rng(42)
    xs  = rng.integers(0, W, 200)
    hs  = rng.integers(8, 24, 200)
    sw  = rng.uniform(-1.0, 1.0, 200)
    for i in range(200):
        sway = int(math.sin(t * 1.4 + sw[i] * 8) * 2.5)
        col  = GRASS_B if i % 3 == 0 else GRASS_A
        pygame.draw.line(surf, col,
                         (xs[i], ground_y + 2),
                         (xs[i] + sway, ground_y + 2 - hs[i]), 2)

    pygame.draw.line(surf, (100, 72, 32), (0, ground_y), (W, ground_y), 2)


# ---------------------------------------------------------------------------
# Primitive partagée — identique chicken.py (ratio ombre 0.2)
# ---------------------------------------------------------------------------

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

    quad = [
        (p1[0] + nx * r, p1[1] + ny * r),
        (p2[0] + nx * r, p2[1] + ny * r),
        (p2[0] - nx * r, p2[1] - ny * r),
        (p1[0] - nx * r, p1[1] - ny * r),
    ]
    aapoly(surf, color, quad, OUTLINE)

    # ombre côté droit — ratio 0.2 comme chicken.py
    shade_q = [
        (p1[0] - nx * r * 0.2, p1[1] - ny * r * 0.2),
        (p2[0] - nx * r * 0.2, p2[1] - ny * r * 0.2),
        (p2[0] - nx * r,        p2[1] - ny * r),
        (p1[0] - nx * r,        p1[1] - ny * r),
    ]
    aapoly(surf, shade, shade_q)

    aacircle(surf, color,   p1, r)
    aacircle(surf, color,   p2, r)
    aacircle(surf, OUTLINE, p1, r, 1)
    aacircle(surf, OUTLINE, p2, r, 1)


# ---------------------------------------------------------------------------
# Patte — coussinet + orteils
# ---------------------------------------------------------------------------

def draw_paw(surf, tibia_body, front=True):
    """Coussinet + trois orteils arrondis à l'extrémité basse du tibia.

    Correction : en coordonnées monde Box2D, y croît vers le haut.
    Le bas en monde = y minimum → mais world2screen inverse y, donc
    le bas à l'écran vient du point monde à y le PLUS PETIT.
    On prend donc min(circles, key=lambda p: p[1]) en monde.
    """
    circles = [tibia_body.GetWorldPoint(f.shape.pos)
               for f in tibia_body.fixtures if hasattr(f.shape, 'pos')]
    if not circles:
        return
    # y monde le plus petit = bas en monde = bas à l'écran après inversion
    bot_world = min(circles, key=lambda p: p[1])
    px, py = w2s(bot_world)
    ang = tibia_body.angle

    r_paw = 8 if front else 7
    aacircle(surf, PAW_COLOR, (px, py), r_paw)
    aacircle(surf, OUTLINE,   (px, py), r_paw, 1)

    # Coussinet central
    aacircle(surf, PAD_COLOR, (px + 1, py + 1), max(4, r_paw // 2))

    # Trois orteils
    for sp in (-0.38, 0.0, 0.38):
        ta = ang + sp - math.pi * 0.45
        # L'écran a y vers le bas → on inverse sin pour les orteils
        tx = px + math.cos(ta) * (r_paw + 4)
        ty = py - math.sin(ta) * (r_paw + 4)
        aacircle(surf, PAW_COLOR, (tx, ty), 4)
        aacircle(surf, OUTLINE,   (tx, ty), 4, 1)


# ---------------------------------------------------------------------------
# Torse — ellipse avec ventre et reflet, ratio proche de chicken.py
# ---------------------------------------------------------------------------

def draw_torso(surf, torso_body):
    cx, cy = w2s(torso_body.position)
    ang = torso_body.angle

    # Ratio revu : plus proche de la poule (1.65 × 1.15 → 2.2 × 1.0)
    rw = int(2.2 * PPM)
    rh = int(1.0 * PPM)

    tmp = pygame.Surface((rw * 2 + 6, rh * 2 + 6), pygame.SRCALPHA)
    ox, oy = rw + 3, rh + 3

    # Corps principal
    pygame.gfxdraw.filled_ellipse(tmp, ox, oy, rw, rh, FUR_MAIN)
    # Ventre clair
    pygame.gfxdraw.filled_ellipse(tmp, ox + 4, oy + int(rh * 0.42),
                                  int(rw * 0.60), int(rh * 0.50), FUR_BELLY)
    # Reflet dorsal
    pygame.gfxdraw.filled_ellipse(tmp, ox - 8, oy - int(rh * 0.35),
                                  int(rw * 0.30), int(rh * 0.18), FUR_LIGHT)
    # Ombre flanc droit
    pygame.gfxdraw.filled_ellipse(tmp, ox + int(rw * 0.45), oy,
                                  int(rw * 0.35), int(rh * 0.72), FUR_SHADE)
    # Aile esquissée (comme chicken) — on garde sobre
    wing = [
        (ox - int(rw * 0.2), oy - int(rh * 0.12)),
        (ox - int(rw * 0.82), oy + int(rh * 0.05)),
        (ox - int(rw * 0.62), oy + int(rh * 0.70)),
        (ox - int(rw * 0.05), oy + int(rh * 0.52)),
    ]
    pygame.gfxdraw.aapolygon(tmp, wing, FUR_SHADE)
    # Contour
    pygame.gfxdraw.aaellipse(tmp, ox, oy, rw, rh, OUTLINE)

    deg = -math.degrees(ang)
    rot = pygame.transform.rotozoom(tmp, deg, 1.0)
    rx, ry = rot.get_rect().center
    surf.blit(rot, (int(cx) - rx, int(cy) - ry))


# ---------------------------------------------------------------------------
# Cou — capsule Box2D réelle (comme chicken.py / draw_neck)
# ---------------------------------------------------------------------------

def draw_neck(surf, neck_body):
    """Cou capsulaire — identique au style draw_neck de chicken.py."""
    draw_capsule_limb(surf, neck_body, HEAD_MAIN,
                      lerp(HEAD_MAIN, OUTLINE, 0.28), int(0.40 * PPM))


# ---------------------------------------------------------------------------
# Queue — capsule avec touffe
# ---------------------------------------------------------------------------

def draw_tail(surf, tail_body):
    """Queue en capsule avec touffe au sommet.

    Correction : en coordonnées monde, le sommet de la queue (attaché au
    torse) est le point à y le PLUS GRAND (haut en monde).
    La touffe est au point libre (y le plus petit en monde = haut écran).
    """
    circles = [tail_body.GetWorldPoint(f.shape.pos)
               for f in tail_body.fixtures if hasattr(f.shape, 'pos')]

    draw_capsule_limb(surf, tail_body, TAIL_BASE, FUR_SHADE, int(0.18 * PPM))

    if len(circles) < 2:
        return

    # Touffe = point libre = y monde le plus petit (haut à l'écran)
    tip_world = min(circles, key=lambda p: p[1])
    pt = w2s(tip_world)
    for di in range(3):
        off_x = (di - 1) * 4
        aacircle(surf, TAIL_TIP, (pt[0] + off_x, pt[1] - 2), max(1, 5 - di))


# ---------------------------------------------------------------------------
# Ombre portée
# ---------------------------------------------------------------------------

def draw_shadow(surf, torso_body):
    tx, _ = w2s(torso_body.position)
    gy = H - int(2.0 * PPM)
    s = pygame.Surface((120, 24), pygame.SRCALPHA)
    pygame.gfxdraw.filled_ellipse(s, 60, 12, 55, 8, (50, 25, 5, 45))
    surf.blit(s, (int(tx) - 60, gy - 12))


# ---------------------------------------------------------------------------
# Tête — oreilles, museau, truffe, œil expressif
# ---------------------------------------------------------------------------

def draw_head(surf, head_body):
    cx, cy = w2s(head_body.position)
    ang = head_body.angle
    r   = int(0.72 * PPM)   # légèrement plus petit que l'original

    # ---------- Oreilles tombantes ----------
    for side in (-1, 1):
        ear_a  = ang - math.pi / 2 + side * 0.55
        ear_px = cx + math.cos(ear_a) * r * 0.7
        ear_py = cy + math.sin(ear_a) * r * 0.7
        drop_a = ear_a + side * 0.3 + math.pi * 0.15
        tip_x  = ear_px + math.cos(drop_a) * r * 1.35
        tip_y  = ear_py + math.sin(drop_a) * r * 1.35 + 6
        perp   = drop_a + math.pi / 2
        w_b, w_t = 10, 6
        ear_pts = [
            (ear_px - math.cos(perp) * w_b, ear_py - math.sin(perp) * w_b),
            (ear_px + math.cos(perp) * w_b, ear_py + math.sin(perp) * w_b),
            (tip_x  + math.cos(perp) * w_t, tip_y  + math.sin(perp) * w_t),
            (tip_x  - math.cos(perp) * w_t, tip_y  - math.sin(perp) * w_t),
        ]
        aapoly(surf, EAR_OUTER, ear_pts, OUTLINE)
        inner = [(ear_px + (p[0] - ear_px) * 0.55,
                  ear_py + (p[1] - ear_py) * 0.55)
                 for p in ear_pts]
        aapoly(surf, EAR_INNER, inner)

    # ---------- Tête principale ----------
    aacircle(surf, HEAD_MAIN, (cx, cy), r)

    # Reflet crânien
    lx = cx + int(math.cos(ang + 0.7) * r * 0.38)
    ly = cy + int(math.sin(ang + 0.7) * r * 0.38)
    aacircle(surf, HEAD_LIGHT, (lx, ly), int(r * 0.22))

    # Ombre joue basse
    jx = cx + int(math.cos(ang - 0.4) * r * 0.35)
    jy = cy + int(math.sin(ang - 0.4) * r * 0.35)
    aacircle(surf, FUR_SHADE, (jx, jy + 5), int(r * 0.28))

    aacircle(surf, OUTLINE, (cx, cy), r, 2)

    # ---------- Museau ----------
    mus_dir = ang
    mx = cx + math.cos(mus_dir) * r * 0.72
    my = cy + math.sin(mus_dir) * r * 0.72
    m_tmp = pygame.Surface((40, 28), pygame.SRCALPHA)
    pygame.gfxdraw.filled_ellipse(m_tmp, 20, 14, 18, 11, MUZZLE)
    pygame.gfxdraw.aaellipse(m_tmp, 20, 14, 18, 11, OUTLINE)
    m_rot = pygame.transform.rotozoom(m_tmp, -math.degrees(mus_dir), 1.0)
    mrx, mry = m_rot.get_rect().center
    surf.blit(m_rot, (int(mx) - mrx, int(my) - mry))

    # Truffe
    nx_ = mx + math.cos(mus_dir) * 14
    ny_ = my + math.sin(mus_dir) * 14
    aacircle(surf, NOSE, (nx_, ny_), 6)
    aacircle(surf, (100, 70, 50), (nx_ - 1, ny_ - 2), 2)

    # ---------- Œil ----------
    eye_a = ang + 1.8
    ex = cx + int(math.cos(eye_a) * r * 0.50)
    ey = cy + int(math.sin(eye_a) * r * 0.50) - 6
    aacircle(surf, EYE_WHITE, (ex, ey), 8)
    aacircle(surf, EYE_IRIS,  (ex, ey), 6)
    aacircle(surf, EYE_PUPIL, (ex, ey), 4)
    aacircle(surf, EYE_SHINE, (ex + 2, ey - 2), 2)
    aacircle(surf, OUTLINE,   (ex, ey), 8, 1)

    # Sourcil expressif
    brow_x = ex + int(math.cos(ang + 2.0) * 4)
    brow_y = ey + int(math.sin(ang + 2.0) * 4) - 9
    pygame.draw.arc(surf, FUR_DARK,
                    pygame.Rect(brow_x - 7, brow_y - 3, 14, 8),
                    math.pi * 0.1, math.pi * 0.9, 2)


# ---------------------------------------------------------------------------
# Assemblage du chien
# ---------------------------------------------------------------------------

def draw_dog(surf, bodies):
    draw_shadow(surf, bodies['torse'])

    # Pattes arrière gauches (plan du fond, plus sombres)
    draw_capsule_limb(surf, bodies['cuisseARG'],
                      lerp(LEG_BACK, (0,0,0), 0.12),
                      lerp(LEG_BACK, OUTLINE, 0.40), int(0.34 * PPM))
    draw_capsule_limb(surf, bodies['tibiaARG'],
                      lerp(LEG_BACK, (0,0,0), 0.10),
                      lerp(LEG_BACK, OUTLINE, 0.38), int(0.23 * PPM))
    draw_paw(surf, bodies['tibiaARG'], front=False)

    # Pattes avant gauches (plan du fond)
    draw_capsule_limb(surf, bodies['cuisseAVG'],
                      lerp(LEG_FRONT, (0,0,0), 0.10),
                      lerp(LEG_FRONT, OUTLINE, 0.35), int(0.30 * PPM))
    draw_capsule_limb(surf, bodies['tibiaAVG'],
                      lerp(LEG_FRONT, (0,0,0), 0.08),
                      lerp(LEG_FRONT, OUTLINE, 0.32), int(0.20 * PPM))
    draw_paw(surf, bodies['tibiaAVG'], front=True)

    # Queue (avant le torse → cachée par lui)
    draw_tail(surf, bodies['queue'])

    # Torse
    draw_torso(surf, bodies['torse'])

    # Cou (capsule réelle Box2D)
    draw_neck(surf, bodies['cou'])

    # Pattes arrière droites (plan avant)
    draw_capsule_limb(surf, bodies['cuisseARD'], LEG_BACK,
                      lerp(LEG_BACK, OUTLINE, 0.28), int(0.34 * PPM))
    draw_capsule_limb(surf, bodies['tibiaARD'], LEG_BACK,
                      lerp(LEG_BACK, OUTLINE, 0.25), int(0.23 * PPM))
    draw_paw(surf, bodies['tibiaARD'], front=False)

    # Pattes avant droites (plan avant)
    draw_capsule_limb(surf, bodies['cuisseAVD'], LEG_FRONT,
                      lerp(LEG_FRONT, OUTLINE, 0.25), int(0.30 * PPM))
    draw_capsule_limb(surf, bodies['tibiaAVD'], LEG_FRONT,
                      lerp(LEG_FRONT, OUTLINE, 0.22), int(0.20 * PPM))
    draw_paw(surf, bodies['tibiaAVD'], front=True)

    # Tête (tout devant)
    draw_head(surf, bodies['tete'])


# ---------------------------------------------------------------------------
# Anatomie Box2D
# Revu pour être cohérent avec la structure capsule() de chicken.py.
# ---------------------------------------------------------------------------

def build_dog(b2world):
    SPAWN_Y = 5.0
    TQ      = 80.0

    # Torse : ratio proche de la poule (TW, TH en demi-largeur / demi-hauteur)
    TW, TH = 2.0, 0.9

    # Cou — même approche que chicken.py : capsule()
    NW, NH = 0.4, 0.8

    # Tête
    HR = 0.7

    # Pattes
    FW, FH = 0.32, 0.85   # cuisse
    CW, CH = 0.22, 0.85   # tibia

    # ---------- Torse ----------
    torse = b2world.CreateDynamicBody(position=(0, SPAWN_Y))
    torse.CreateFixture(shape=polygonShape(box=(TW, TH)), density=5.0,
                        friction=0.5,
                        filter=Box2D.b2Filter(groupIndex=-1))
    # Extrémités arrondies
    torse.CreateFixture(shape=circleShape(radius=TH, pos=( TW, 0)),
                        density=0.1, friction=0.3,
                        filter=Box2D.b2Filter(groupIndex=-1))
    torse.CreateFixture(shape=circleShape(radius=TH * 0.85, pos=(-TW, 0)),
                        density=0.1, friction=0.3,
                        filter=Box2D.b2Filter(groupIndex=-1))

    # ---------- Cou (capsule réelle) ----------
    cou = capsule(b2world,
                  (TW * 0.8, SPAWN_Y + TH * 0.3),
                  NW, NH, density=0.3, group_index=-1)

    # ---------- Tête ----------
    tete = b2world.CreateDynamicBody(
        position=(TW * 0.8, SPAWN_Y + TH * 0.3 + NH * 0.9))
    tete.CreateFixture(shape=circleShape(radius=HR),
                       density=0.4, friction=0.3,
                       filter=Box2D.b2Filter(groupIndex=-1))

    # Attache cou→torse
    b2world.CreateJoint(revoluteJointDef(
        bodyA=torse, bodyB=cou,
        localAnchorA=(TW * 0.7, TH * 0.3),
        localAnchorB=(0, -NH * 0.8),
        enableLimit=True, lowerAngle=-0.8, upperAngle=0.8,
        enableMotor=True, maxMotorTorque=TQ * 0.5, motorSpeed=0,
    ))
    # Attache tête→cou (soudure souple)
    b2world.CreateJoint(Box2D.b2WeldJointDef(
        bodyA=cou, bodyB=tete,
        localAnchorA=(0, NH * 0.8),
        localAnchorB=(0, 0),
    ))

    # ---------- Queue ----------
    lq, hq = 0.18, 0.9
    queue = b2world.CreateDynamicBody(position=(-TW * 1.1, SPAWN_Y + TH * 0.1))
    queue.CreateFixture(shape=polygonShape(box=(lq, hq)), density=0.2,
                        friction=0.1,
                        filter=Box2D.b2Filter(groupIndex=-1))
    queue.CreateFixture(shape=circleShape(radius=lq, pos=(0,  hq)), density=0.0,
                        filter=Box2D.b2Filter(groupIndex=-1))
    queue.CreateFixture(shape=circleShape(radius=lq, pos=(0, -hq)), density=0.0,
                        filter=Box2D.b2Filter(groupIndex=-1))
    b2world.CreateJoint(revoluteJointDef(
        bodyA=torse, bodyB=queue,
        localAnchorA=(-TW * 0.9, TH * 0.15),
        localAnchorB=(0, hq * 0.85),
        enableLimit=True, lowerAngle=-1.0, upperAngle=1.0,
        enableMotor=False,
    ))

    # ---------- Pattes (même logique que chicken.py) ----------
    xav =  TW * 0.55    # attache avant
    xar = -TW * 0.55    # attache arrière

    def cuisse(x_offset):
        return capsule(b2world,
                       (x_offset, SPAWN_Y - FH * 0.5),
                       FW, FH, density=1.5, group_index=-1)

    def tibia(x_offset):
        return capsule(b2world,
                       (x_offset, SPAWN_Y - FH * 1.5),
                       CW, CH, density=2.0, friction=0.8, group_index=-1)

    cAVG = cuisse(xav);  cAVD = cuisse(xav)
    cARG = cuisse(xar);  cARD = cuisse(xar)
    tAVG = tibia(xav);   tAVD = tibia(xav)
    tARG = tibia(xar);   tARD = tibia(xar)

    def hanche(membre, x):
        return b2world.CreateJoint(revoluteJointDef(
            bodyA=torse, bodyB=membre,
            localAnchorA=(x, -TH * 0.6),
            localAnchorB=(0, FH * 0.6),
            enableLimit=True, lowerAngle=-0.8, upperAngle=1.2,
            enableMotor=True, maxMotorTorque=TQ, motorSpeed=0,
        ))

    def genou(cu, ti):
        return b2world.CreateJoint(revoluteJointDef(
            bodyA=cu, bodyB=ti,
            localAnchorA=(0, -FH * 0.7),
            localAnchorB=(0, CH * 0.7),
            enableLimit=True, lowerAngle=-2.0, upperAngle=0.0,
            enableMotor=True, maxMotorTorque=TQ, motorSpeed=0,
        ))

    joints = [
        hanche(cAVG, xav), hanche(cAVD, xav),
        hanche(cARG, xar), hanche(cARD, xar),
        genou(cAVG, tAVG),  genou(cAVD, tAVD),
        genou(cARG, tARG),  genou(cARD, tARD),
    ]

    corps = dict(
        torse=torse, cou=cou, tete=tete, queue=queue,
        cuisseAVG=cAVG, cuisseAVD=cAVD, cuisseARG=cARG, cuisseARD=cARD,
        tibiaAVG=tAVG, tibiaAVD=tAVD, tibiaARG=tARG, tibiaARD=tARD,
    )
    return joints, corps


# ---------------------------------------------------------------------------
# Application — identique en structure à chicken.py
# ---------------------------------------------------------------------------

class DogApp(PhysicsApp):

    TITLE = "Le Chien Doré"
    VELOCITY_ITERATIONS = 8
    POSITION_ITERATIONS = 3

    def __init__(self):
        super().__init__(screen_w=W, screen_h=H)
        self._fonts_loaded = False

    def build(self, b2world):
        self.joints, self.corps = build_dog(b2world)
        (self._hAVG, self._hAVD,
         self._hARG, self._hARD) = self.joints[:4]
        (self._gAVG, self._gAVD,
         self._gARG, self._gARD) = self.joints[4:]

    def update_brain(self, t, dt):
        """Trot diagonal : AVG+ARD en phase, AVD+ARG en opposition."""
        freq, amp = 4.0, 3.5
        phases = [0, math.pi, math.pi, 0]   # AVG, AVD, ARG, ARD
        for j, ph in zip(self.joints[:4], phases):
            j.motorSpeed = amp * np.sin(t * freq + ph)
        for j, ph in zip(self.joints[4:], phases):
            j.motorSpeed = amp * np.sin(t * freq + ph - 1.5)

        # Queue : balancement autonome
        # (pas de joint motorisé sur la queue — le physics s'en charge)

    def draw_scene(self, screen):
        draw_background(screen, self.time)
        draw_dog(screen, self.corps)
        self._draw_ui(screen)

    def _draw_ui(self, screen):
        """Panneau identique en style à chicken.py."""
        if not self._fonts_loaded:
            pygame.font.init()
            try:
                self._f1 = pygame.font.SysFont("Georgia", 26, italic=True)
                self._f2 = pygame.font.SysFont("Georgia", 13, italic=True)
            except Exception:
                self._f1 = pygame.font.SysFont(None, 28)
                self._f2 = pygame.font.SysFont(None, 15)
            self._fonts_loaded = True

        panel = pygame.Surface((290, 56), pygame.SRCALPHA)
        panel.fill((255, 238, 190, 150))
        pygame.draw.rect(panel, (140, 85, 28, 200),
                         panel.get_rect(), 2, border_radius=10)
        screen.blit(panel, (18, 16))

        screen.blit(self._f1.render("Le Chien Doré", True, (90, 44, 8)),  (30, 20))
        screen.blit(self._f2.render(
            f"t = {self.time:.1f}s   ·   Box2D + Pygame", True, (130, 70, 20)),
            (30, 50))


if __name__ == "__main__":
    DogApp().run()
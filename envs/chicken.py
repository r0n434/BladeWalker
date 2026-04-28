import Box2D
from Box2D.b2 import world, polygonShape, circleShape, fixtureDef, revoluteJointDef
import pygame
import sys
import numpy as np

def capsule(world, position, hx, hy, density=1.0, friction=0.5, restitution=0.0, group_index=-1):
    """Crée un corps dynamique en forme de capsule"""
    body = world.CreateDynamicBody(position=position)
    filtre = Box2D.b2Filter(groupIndex=group_index)
    
    fixture_params = {
        'density': density,
        'friction': friction,
        'restitution': restitution,
        'filter': filtre
    }

    if hy >= hx:
        radius = hx
        offset = hy - hx
        rect_dims = (hx, offset)
        pos_1 = (0, offset) 
        pos_2 = (0, -offset)
    else:
        radius = hy
        offset = hx - hy
        rect_dims = (offset, hy)
        pos_1 = (offset, 0)
        pos_2 = (-offset, 0)

    if offset > 0:
        rect_shape = polygonShape(box=rect_dims)
        body.CreateFixture(shape=rect_shape, **fixture_params)
    
    body.CreateFixture(shape=circleShape(radius=radius, pos=pos_1), **fixture_params)
    body.CreateFixture(shape=circleShape(radius=radius, pos=pos_2), **fixture_params)
    
    return body

def create_world(gravity=(0, -10), size_w=50.0, size_h=1.0, friction_ground=0.8):
    env_world = world(gravity=gravity, doSleep=True)
    ground = env_world.CreateStaticBody(position=(0, 0), shapes=polygonShape(box=(size_w, size_h)))
    ground.fixtures[0].friction = friction_ground
    return env_world

def chicken(world):
    SPAWN_Y = 5.0 
    MOTOR_TORQUE = 80.0 
    
    TORSO_W, TORSO_H = 1.5, 1.0
    NECK_W, NECK_H = 0.5, 1.0    
    HEAD_R = 0.5

    THIGH_W, THIGH_H = 0.5, 1.0
    CALF_W, CALF_H = 0.2, 1.0

    # TORSE
    torso = capsule(world, (0, SPAWN_Y), TORSO_W, TORSO_H, density=0.8, friction=0.5, group_index=-1)
        
    # COU
    neck = capsule(world, (3*TORSO_W/4, SPAWN_Y + TORSO_H/2), NECK_W, NECK_H, density=0.3, friction=0.5, group_index=-1)
    
    # TETE
    head_pos = (3*TORSO_W/4, SPAWN_Y + TORSO_H/2 + 3*NECK_H/4)
    head = world.CreateDynamicBody(position=head_pos)
    filtre_tete = Box2D.b2Filter(groupIndex=-1)
    
    head.CreateFixture(
        shape=circleShape(radius=HEAD_R), 
        density=0.2, 
        friction=0.5, 
        filter=filtre_tete
    )

    world.CreateJoint(Box2D.b2WeldJointDef(
        bodyA=neck, 
        bodyB=head, 
        localAnchorA=(NECK_W/2, NECK_H/2),
        localAnchorB=(0, 0)
    ))

    # JAMBES
    thighR = capsule(world, (0, SPAWN_Y - TORSO_H/4), THIGH_W, THIGH_H, density=2.5, friction=0.5, group_index=-1)
    thighL = capsule(world, (0, SPAWN_Y - TORSO_H/4), THIGH_W, THIGH_H, density=2.5, friction=0.5, group_index=-1)
    calfR = capsule(world, (0, SPAWN_Y - TORSO_H/4 - 3*THIGH_H/4), CALF_W, CALF_H, density=3.0, friction=0.8, group_index=-1)
    calfL = capsule(world, (0, SPAWN_Y - TORSO_H/4 - 3*THIGH_H/4), CALF_W, CALF_H, density=3.0, friction=0.8, group_index=-1)

    # ARTICULATIONS
    hipR = world.CreateJoint(revoluteJointDef(bodyA=torso, bodyB=thighR, localAnchorA=(0, -TORSO_H/4), localAnchorB=(0, THIGH_H/2), enableLimit=True, lowerAngle=-0.8, upperAngle=1.2, enableMotor=True, maxMotorTorque=MOTOR_TORQUE, motorSpeed=0.0))
    hipL = world.CreateJoint(revoluteJointDef(bodyA=torso, bodyB=thighL, localAnchorA=(0, -TORSO_H/4), localAnchorB=(0, THIGH_H/2), enableLimit=True, lowerAngle=-0.8, upperAngle=1.2, enableMotor=True, maxMotorTorque=MOTOR_TORQUE, motorSpeed=0.0))
    kneeR = world.CreateJoint(revoluteJointDef(bodyA=thighR, bodyB=calfR, localAnchorA=(0, -THIGH_H/2), localAnchorB=(0, CALF_H/2), enableLimit=True, lowerAngle=-2.0, upperAngle=0.0, enableMotor=True, maxMotorTorque=MOTOR_TORQUE, motorSpeed=0.0))
    kneeL = world.CreateJoint(revoluteJointDef(bodyA=thighL, bodyB=calfL, localAnchorA=(0, -THIGH_H/2), localAnchorB=(0, CALF_H/2), enableLimit=True, lowerAngle=-2.0, upperAngle=0.0, enableMotor=True, maxMotorTorque=MOTOR_TORQUE, motorSpeed=0.0))
    neck_joint = world.CreateJoint(revoluteJointDef(bodyA=torso, bodyB=neck, localAnchorA=(TORSO_W/2, TORSO_H/2), localAnchorB=(0, -3*NECK_H/4), enableLimit=True, lowerAngle=-1.0, upperAngle=1.0, enableMotor=True, maxMotorTorque=MOTOR_TORQUE, motorSpeed=0.0)) 

    return [hipR, hipL, kneeR, kneeL, neck_joint], torso

PPM = 30.0
TARGET_FPS = 50
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 700

def get_convex_hull(points):
    """Calcule l'enveloppe convexe (contours extérieurs) d'un nuage de points"""
    points = sorted(list(set([(round(p[0], 4), round(p[1], 4)) for p in points])))
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

def draw_body_outline(screen, body):
    """Dessine uniquement le contour global d'un corps"""
    points_local = []
    
    for fixture in body.fixtures:
        shape = fixture.shape
        if isinstance(shape, polygonShape):
            points_local.extend(shape.vertices)
        elif isinstance(shape, circleShape):
            cx, cy = shape.pos
            r = shape.radius
            for i in range(20):
                angle = 2 * np.pi * i / 20
                points_local.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
                
    if not points_local:
        return
        
    hull_local = get_convex_hull(points_local)
    
    vertices_pygame = []
    for v in hull_local:
        world_v = body.GetWorldPoint(v)
        vx_pg = int(world_v[0] * PPM + SCREEN_WIDTH / 2)
        vy_pg = int(SCREEN_HEIGHT - world_v[1] * PPM)
        vertices_pygame.append((vx_pg, vy_pg))
        
    if len(vertices_pygame) > 2:
        pygame.draw.polygon(screen, (0, 0, 0), vertices_pygame, 2)
    elif len(vertices_pygame) == 2:
        pygame.draw.line(screen, (0, 0, 0), vertices_pygame[0], vertices_pygame[1], 2)

def draw_joint(screen, joint):
    """Dessine une petite pastille pour visualiser une articulation Box2D"""
    anchor = joint.anchorA 
    
    pos_x = int(anchor[0] * PPM + SCREEN_WIDTH / 2)
    pos_y = int(SCREEN_HEIGHT - anchor[1] * PPM)
    
    pygame.draw.circle(screen, (220, 50, 50), (pos_x, pos_y), 4)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Créature d'après Esquisse (La Marche !)")
    clock = pygame.time.Clock()

    env_world = create_world()
    
    # On récupère les articulations (dans l'ordre où tu les as retournées)
    joints, torso = chicken(env_world)
    hipR, hipL, kneeR, kneeL, neck_joint = joints

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- LE CERVEAU DE LA CRÉATURE ---
        # On utilise le temps pour créer un cycle de marche continu
        t = pygame.time.get_ticks() / 1000.0  # Temps écoulé en secondes
        
        frequence = 5.0 # Vitesse du cycle (plus c'est haut, plus il pédale vite)
        amplitude = 3.0 # Vitesse maximale du moteur
        
        # Mouvement des Hanches (opposition de phase via + np.pi sur la jambe gauche)
        hipR.motorSpeed = amplitude * np.sin(t * frequence)
        hipL.motorSpeed = amplitude * np.sin(t * frequence + np.pi)

        # Mouvement des Genoux (décalage de phase de -1.5 pour plier le genou au bon moment)
        kneeR.motorSpeed = amplitude * np.sin(t * frequence - 1.5)
        kneeL.motorSpeed = amplitude * np.sin(t * frequence + np.pi - 1.5)
        
        # Bonus : On anime le cou pour lui donner un petit mouvement de tête de poulet
        neck_joint.motorSpeed = 1.5 * np.sin(t * frequence * 2)
        # ---------------------------------

        env_world.Step(TIME_STEP, 6, 2)

        screen.fill((255, 255, 255))

        # Affichage
        for body in env_world.bodies:
            draw_body_outline(screen, body)
            
        for joint in env_world.joints:
            draw_joint(screen, joint)

        pygame.display.flip()
        clock.tick(TARGET_FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
import Box2D
from Box2D.b2 import world, polygonShape, fixtureDef, revoluteJointDef
import pygame
import sys

def creer_monde_et_bipede():
    env_world = world(gravity=(0, -10), doSleep=True)
    
    # Sol
    ground = env_world.CreateStaticBody(position=(0, 0), shapes=polygonShape(box=(50.0, 1.0)))
    ground.fixtures[0].friction = 0.8 
    
    SPAWN_Y = 5.0 
    TORSO_W, TORSO_H = 4, 1
    torse = env_world.CreateDynamicBody(position=(0, SPAWN_Y))
    torse.CreateFixture(shape=polygonShape(box=(TORSO_W, TORSO_H)), density=5.0, friction=0.5)

    CUISSE_W, CUISSE_H = 0.5, 1.0
    cuisseL = env_world.CreateDynamicBody(position=(-TORSO_W/2, SPAWN_Y -CUISSE_H))
    cuisseL.CreateFixture(shape=polygonShape(box=(CUISSE_W, CUISSE_H)), density=1.0, friction=0.5)
    cuisseR = env_world.CreateDynamicBody(position=(TORSO_W/2, SPAWN_Y -CUISSE_H))
    cuisseR.CreateFixture(shape=polygonShape(box=(CUISSE_W, CUISSE_H)), density=1.0, friction=0.5) 

    TIBIA_W, TIBIA_H = 0.4, 1.0
    tibiaL = env_world.CreateDynamicBody(position=(-TORSO_W/2, SPAWN_Y - (CUISSE_H*2)))
    tibiaL.CreateFixture(shape=polygonShape(box=(TIBIA_W, TIBIA_H)), density=1.0, friction=0.8)
    tibiaR = env_world.CreateDynamicBody(position=(TORSO_W/2, SPAWN_Y - (CUISSE_H*2)))
    tibiaR.CreateFixture(shape=polygonShape(box=(TIBIA_W, TIBIA_H)), density=1.0, friction=0.8)

    hipL = env_world.CreateJoint(revoluteJointDef(bodyA=torse, bodyB=cuisseL, localAnchorA=(-TORSO_W/2, -TORSO_H/2), localAnchorB=(0, 3*CUISSE_H/4), enableLimit=True, lowerAngle=-0.8, upperAngle=1.2, enableMotor=True, maxMotorTorque=80.0, motorSpeed=0.0))
    hipR = env_world.CreateJoint(revoluteJointDef(bodyA=torse, bodyB=cuisseR, localAnchorA=(TORSO_W/2, -TORSO_H/2), localAnchorB=(0, 3*CUISSE_H/4), enableLimit=True, lowerAngle=-0.8, upperAngle=1.2, enableMotor=True, maxMotorTorque=80.0, motorSpeed=0.0))
    kneeL = env_world.CreateJoint(revoluteJointDef(bodyA=cuisseL, bodyB=tibiaL, localAnchorA=(0, -CUISSE_H/2), localAnchorB=(0, TIBIA_H/2), enableLimit=True, lowerAngle=-2.0, upperAngle=0.0, enableMotor=True, maxMotorTorque=80.0, motorSpeed=0.0)) 
    kneeR = env_world.CreateJoint(revoluteJointDef(bodyA=cuisseR, bodyB=tibiaR, localAnchorA=(0, -CUISSE_H/2), localAnchorB=(0, TIBIA_H/2), enableLimit=True, lowerAngle=-2.0,upperAngle=0.0, enableMotor=True, maxMotorTorque=80.0, motorSpeed=0.0))
   
    return env_world, [hipL, kneeL, hipR, kneeR], torse

# --- 2. LOGIQUE DE RENDU PYGAME ---

# Constantes d'affichage
PPM = 30.0  # Pixels Par Mètre (Zoom)
TARGET_FPS = 50
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600

def my_draw_polygon(screen, polygon, body):
    """Transforme les coordonnées Box2D en PyGame et dessine le polygone"""
    # 1. On récupère les sommets et on les transforme selon la position/rotation du corps
    vertices = [(body.transform * v) * PPM for v in polygon.vertices]
    
    # 2. On convertit pour l'écran (On centre X, et on inverse Y)
    vertices_pygame = [(v[0] + SCREEN_WIDTH / 2, SCREEN_HEIGHT - v[1]) for v in vertices]
    
    # 3. On dessine (en noir, épaisseur 2)
    pygame.draw.polygon(screen, (0, 0, 0), vertices_pygame, 2)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BladeWalker - Test Physique")
    clock = pygame.time.Clock()

    # On initialise la physique
    env_world, joints, torso = creer_monde_et_bipede()

    running = True
    while running:
        # --- Gestion des événements (fermer la fenêtre) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Moteur Physique ---
        # On fait avancer la simulation (sans action de l'IA pour l'instant)
        env_world.Step(TIME_STEP, 6, 2)

        # --- Rendu Visuel ---
        screen.fill((255, 255, 255)) # Fond blanc

        # On parcourt tous les corps du monde pour les dessiner
        for body in env_world.bodies:
            for fixture in body.fixtures:
                shape = fixture.shape
                # Comme on n'a utilisé que des polygones (box), on appelle notre fonction
                if isinstance(shape, Box2D.b2PolygonShape):
                    my_draw_polygon(screen, shape, body)

        pygame.display.flip()
        clock.tick(TARGET_FPS) # Maintient la vitesse à 50 FPS

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
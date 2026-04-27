import Box2D
from Box2D.b2 import world, polygonShape, revoluteJointDef, circleShape
import pygame
import sys
import math

# couleurs
CIEL_COULEUR    = (135, 180, 235)
SOL_COULEUR     = (55,  60,  80)
TORSE_COULEUR   = (220, 180, 120)
PATTE_COULEUR   = (190, 140,  80)
TIBIA_COULEUR   = (160, 110,  55)
TETE_COULEUR    = (230, 190, 130)
OEIL_COULEUR    = (255, 255, 255)
PUPILLE_COULEUR = (30,   30,  50)
QUEUE_COULEUR   = (200, 155,  90)
JOINT_COULEUR   = (255, 210, 100)
CONTOUR     = (60,   40,  10)

PIXELS_PAR_METRE = 30.0
IMAGES_PAR_SECONDE = 50
PAS_TEMPS = 1.0 / IMAGES_PAR_SECONDE
LARGEUR, HAUTEUR = 900, 600


def box2d_vers_ecran(vecteur):
    return (vecteur[0] * PIXELS_PAR_METRE + LARGEUR / 2, HAUTEUR - vecteur[1] * PIXELS_PAR_METRE)


def dessiner_sol(ecran):
    y_sol = HAUTEUR - int(2.0 * PIXELS_PAR_METRE)
    pygame.draw.rect(ecran, SOL_COULEUR, (0, y_sol, LARGEUR, HAUTEUR - y_sol))
    pygame.draw.line(ecran, (80, 90, 120), (0, y_sol), (LARGEUR, y_sol), 2)

def dessiner_corps(ecran, corps, couleur):
    for fixture in corps.fixtures:
        forme = fixture.shape

        # Polygone
        if isinstance(forme, Box2D.b2PolygonShape):
            points = [box2d_vers_ecran(corps.transform * v) for v in forme.vertices]
            pygame.draw.polygon(ecran, couleur, points, 0)
            pygame.draw.polygon(ecran, CONTOUR, points, 2)

        # Cercle
        elif isinstance(forme, Box2D.b2CircleShape):
            cx, cy = box2d_vers_ecran(corps.transform * forme.pos)
            rayon = int(forme.radius * PIXELS_PAR_METRE)
            pygame.draw.circle(ecran, couleur, (int(cx), int(cy)), rayon)
            pygame.draw.circle(ecran, CONTOUR, (int(cx), int(cy)), rayon, 2)

def dessiner_tete(ecran, tete):
    for fixture in tete.fixtures:
        forme = fixture.shape
        if not isinstance(forme, Box2D.b2CircleShape):
            continue
        centre_x, centre_y = box2d_vers_ecran(tete.transform * forme.pos)
        rayon = int(forme.radius * PIXELS_PAR_METRE)
        angle = tete.angle

        # Oreille gauche
        base_oreille_gauche = pygame.math.Vector2(-rayon * 0.5, -rayon * 0.7).rotate(-math.degrees(angle))
        pointe_oreille_gauche = pygame.math.Vector2(-rayon * 0.5, -rayon * 1.5).rotate(-math.degrees(angle))
        points_oreille_gauche = [
            (centre_x + base_oreille_gauche.x - rayon * 0.3, centre_y + base_oreille_gauche.y),
            (centre_x + base_oreille_gauche.x + rayon * 0.3, centre_y + base_oreille_gauche.y),
            (centre_x + pointe_oreille_gauche.x, centre_y + pointe_oreille_gauche.y),
        ]
        pygame.draw.polygon(ecran, PATTE_COULEUR, points_oreille_gauche)
        pygame.draw.polygon(ecran, CONTOUR, points_oreille_gauche, 2)

        # Oreille droite
        base_oreille_droite = pygame.math.Vector2(rayon * 0.5, -rayon * 0.7).rotate(-math.degrees(angle))
        pointe_oreille_droite = pygame.math.Vector2(rayon * 0.5, -rayon * 1.5).rotate(-math.degrees(angle))
        points_oreille_droite = [
            (centre_x + base_oreille_droite.x - rayon * 0.3, centre_y + base_oreille_droite.y),
            (centre_x + base_oreille_droite.x + rayon * 0.3, centre_y + base_oreille_droite.y),
            (centre_x + pointe_oreille_droite.x, centre_y + pointe_oreille_droite.y),
        ]
        pygame.draw.polygon(ecran, PATTE_COULEUR, points_oreille_droite)
        pygame.draw.polygon(ecran, CONTOUR, points_oreille_droite, 2)

        pygame.draw.circle(ecran, TETE_COULEUR, (int(centre_x), int(centre_y)), rayon)
        pygame.draw.circle(ecran, CONTOUR,  (int(centre_x), int(centre_y)), rayon, 2)

        decalage_oeil = pygame.math.Vector2(rayon * 0.35, rayon * 0.25).rotate(-math.degrees(angle))
        oeil_x, oeil_y = int(centre_x + decalage_oeil.x), int(centre_y + decalage_oeil.y)
        pygame.draw.circle(ecran, OEIL_COULEUR,    (oeil_x, oeil_y), max(3, rayon // 4))
        pygame.draw.circle(ecran, PUPILLE_COULEUR, (oeil_x, oeil_y), max(1, rayon // 7))

        decalage_museau = pygame.math.Vector2(rayon * 0.7, rayon * 0.5).rotate(-math.degrees(angle))
        museau_x, museau_y = int(centre_x + decalage_museau.x), int(centre_y + decalage_museau.y)
        pygame.draw.circle(ecran, (230, 170, 140), (museau_x, museau_y), max(4, rayon // 3))
        pygame.draw.circle(ecran, CONTOUR, (museau_x, museau_y), max(4, rayon // 3), 1)

        decalage_truffe = pygame.math.Vector2(rayon * 0.95, rayon * 0.65).rotate(-math.degrees(angle))
        truffe_x, truffe_y = int(centre_x + decalage_truffe.x), int(centre_y + decalage_truffe.y)
        pygame.draw.circle(ecran, (40, 30, 20), (truffe_x, truffe_y), max(2, rayon // 7))


def dessiner_chien(ecran, dict_corps, joints):
    tibias = ['tibiaAVG', 'tibiaAVD', 'tibiaARG', 'tibiaARD']
    cuisses = ['cuisseAVG', 'cuisseAVD', 'cuisseARG', 'cuisseARD']
    for nom in tibias:
        dessiner_corps(ecran, dict_corps[nom], TIBIA_COULEUR)
    for nom in cuisses:
        dessiner_corps(ecran, dict_corps[nom], PATTE_COULEUR)
    dessiner_corps(ecran, dict_corps['queue'], QUEUE_COULEUR)
    dessiner_corps(ecran, dict_corps['torse'], TORSE_COULEUR)
    dessiner_tete(ecran, dict_corps['tete'])

    for joint in joints:
        ancre_x, ancre_y = box2d_vers_ecran(joint.anchorA)
        pygame.draw.circle(ecran, JOINT_COULEUR, (int(ancre_x), int(ancre_y)), 4)
        pygame.draw.circle(ecran, CONTOUR,   (int(ancre_x), int(ancre_y)), 4, 1)


def creer_monde():
    monde = world(gravity=(0, -10), doSleep=True)
    sol = monde.CreateStaticBody(position=(0, 0), shapes=polygonShape(box=(50.0, 1.0)))
    sol.fixtures[0].friction = 0.8

    hauteur_initiale = 5.0
    largeur_torse, hauteur_torse = 3.5, 0.7

    torse = monde.CreateDynamicBody(position=(0, hauteur_initiale))
    torse.CreateFixture(shape=polygonShape(box=(largeur_torse, hauteur_torse)), density=5.0, friction=0.5)
    torse.CreateFixture(shape=circleShape(radius=hauteur_torse,       pos=( largeur_torse, 0)), density=0.1, friction=0.3)
    torse.CreateFixture(shape=circleShape(radius=hauteur_torse * 0.8, pos=(-largeur_torse, 0)), density=0.1, friction=0.3)

    rayon_tete = 0.75
    tete = monde.CreateDynamicBody(position=(largeur_torse + rayon_tete * 1.5, hauteur_initiale + 0.3))
    tete.CreateFixture(shape=circleShape(radius=rayon_tete), density=1.0, friction=0.3)
    monde.CreateJoint(revoluteJointDef(
        bodyA=torse, bodyB=tete,
        localAnchorA=(largeur_torse, hauteur_torse * 0.2), localAnchorB=(-rayon_tete, 0),
        enableLimit=True, lowerAngle=-0.3, upperAngle=0.3, enableMotor=False
    ))

    largeur_queue, hauteur_queue = 0.2, 1.0
    queue = monde.CreateDynamicBody(position=(-largeur_torse - largeur_queue, hauteur_initiale + 0.4))
    queue.CreateFixture(shape=polygonShape(box=(largeur_queue, hauteur_queue)), density=0.2, friction=0.1)
    queue.CreateFixture(shape=circleShape(radius=largeur_queue, pos=(0, hauteur_queue)), density=0.0)
    monde.CreateJoint(revoluteJointDef(
        bodyA=torse, bodyB=queue,
        localAnchorA=(-largeur_torse, hauteur_torse * 0.2), localAnchorB=(largeur_queue, -hauteur_queue * 0.5),
        enableLimit=True, lowerAngle=-1.0, upperAngle=1.0, enableMotor=False
    ))

    largeur_cuisse, hauteur_cuisse = 0.35, 0.9
    position_x_avant, position_x_arriere = 1.8, -1.8

    def creer_cuisse(x):
        corps_temporaire = monde.CreateDynamicBody(position=(x, hauteur_initiale - hauteur_cuisse))
        corps_temporaire.CreateFixture(shape=polygonShape(box=(largeur_cuisse, hauteur_cuisse)), density=1.0, friction=0.5)
        corps_temporaire.CreateFixture(shape=circleShape(radius=largeur_cuisse, pos=(0, -hauteur_cuisse)), density=0.0)
        return corps_temporaire

    cuisseAVG = creer_cuisse(position_x_avant)
    cuisseAVD = creer_cuisse(position_x_avant)
    cuisseARG = creer_cuisse(position_x_arriere)
    cuisseARD = creer_cuisse(position_x_arriere)

    largeur_tibia, hauteur_tibia = 0.25, 0.9

    def creer_tibia(x):
        corps_temporaire = monde.CreateDynamicBody(position=(x, hauteur_initiale - hauteur_cuisse * 2))
        corps_temporaire.CreateFixture(shape=polygonShape(box=(largeur_tibia, hauteur_tibia)), density=1.0, friction=0.8)
        corps_temporaire.CreateFixture(shape=circleShape(radius=largeur_tibia * 1.5, pos=(0, -hauteur_tibia)), density=0.0)
        return corps_temporaire

    tibiaAVG = creer_tibia(position_x_avant)
    tibiaAVD = creer_tibia(position_x_avant)
    tibiaARG = creer_tibia(position_x_arriere)
    tibiaARD = creer_tibia(position_x_arriere)

    def hanche(corps_membre, x):
        return monde.CreateJoint(revoluteJointDef(
            bodyA=torse, bodyB=corps_membre,
            localAnchorA=(x, -hauteur_torse / 2), localAnchorB=(0, 3 * hauteur_cuisse / 4),
            enableLimit=True, lowerAngle=-0.8, upperAngle=1.2,
            enableMotor=True, maxMotorTorque=80, motorSpeed=0
        ))

    def genou(cuisse, tibia):
        return monde.CreateJoint(revoluteJointDef(
            bodyA=cuisse, bodyB=tibia,
            localAnchorA=(0, -hauteur_cuisse / 2), localAnchorB=(0, hauteur_tibia / 2),
            enableLimit=True, lowerAngle=-2.0, upperAngle=0.0,
            enableMotor=True, maxMotorTorque=80, motorSpeed=0
        ))

    joints = [
        hanche(cuisseAVG, position_x_avant), hanche(cuisseAVD, position_x_avant),
        hanche(cuisseARG, position_x_arriere), hanche(cuisseARD, position_x_arriere),
        genou(cuisseAVG, tibiaAVG), genou(cuisseAVD, tibiaAVD),
        genou(cuisseARG, tibiaARG), genou(cuisseARD, tibiaARD),
    ]

    dict_corps = {
        'torse': torse, 'tete': tete, 'queue': queue,
        'cuisseAVG': cuisseAVG, 'cuisseAVD': cuisseAVD,
        'cuisseARG': cuisseARG, 'cuisseARD': cuisseARD,
        'tibiaAVG': tibiaAVG, 'tibiaAVD': tibiaAVD,
        'tibiaARG': tibiaARG, 'tibiaARD': tibiaARD,
    }
    return monde, joints, torse, dict_corps


def main():
    pygame.init()
    ecran = pygame.display.set_mode((LARGEUR, HAUTEUR))
    pygame.display.set_caption("BladeWalker")
    horloge = pygame.time.Clock()
    police = pygame.font.SysFont("monospace", 14)

    monde, joints, torse, dict_corps = creer_monde()

    temps = 0.0

    en_cours = True
    while en_cours:
        for evenement in pygame.event.get():
            if evenement.type == pygame.QUIT:
                en_cours = False

        monde.Step(PAS_TEMPS, 8, 3)
        temps += PAS_TEMPS

        ecran.fill(CIEL_COULEUR)
        dessiner_sol(ecran)
        dessiner_chien(ecran, dict_corps, joints)

        texte = police.render(f"BladeWalker  |  t={temps:.1f}s", True, (180, 200, 240))
        ecran.blit(texte, (16, 12))

        pygame.display.flip()
        horloge.tick(IMAGES_PAR_SECONDE)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
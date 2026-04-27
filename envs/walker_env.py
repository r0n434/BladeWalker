import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Box2D
from Box2D.b2 import world, polygonShape, revoluteJointDef


class WalkerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super(WalkerEnv, self).__init__()
        self.render_mode = render_mode #mode d'affichage ( human pour affichage classique, rgb_array pour récupérer les images)
        
        # 1. Action Space : 4 joints (Hanche G, Genou G, Hanche D, Genou D)
        # On utilise une plage de -1 à 1 pour faciliter l'apprentissage du réseau
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # 2. Observation Space : 14 variables (angles, vitesses, contacts)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        self.world = world(gravity=(0, -10), doSleep=True)
        self.walker_body = None # Sera initialisé dans reset()

    def _create_robot(self, x_torse, y_torse):
        # --- CORPS ---
        # Torse 
        self.torse = self.world.CreateDynamicBody(
            position=(x_torse, y_torse),
            fixtures=polygonShape(box=(0.25, 0.15), density=5.0, friction=0.1) 
        )

        cuisse_shape = polygonShape(box=(0.06, 0.2), density=1.0, friction=0.2)
        tibia_shape = polygonShape(box=(0.05, 0.2), density=1.0, friction=0.2)

        # Création des membres 
        self.cuisse_left  = self.world.CreateDynamicBody(position=(x_torse, y_torse - 0.3), fixtures=cuisse_shape)
        self.cuisse_right = self.world.CreateDynamicBody(position=(x_torse, y_torse - 0.3), fixtures=cuisse_shape)
        self.tibia_left   = self.world.CreateDynamicBody(position=(x_torse, y_torse - 0.7), fixtures=tibia_shape)
        self.tibia_right  = self.world.CreateDynamicBody(position=(x_torse, y_torse - 0.7), fixtures=tibia_shape)

        # --- JOINTS (Articulations) ---
        # On utilise 'anchor' pour laisser Box2D calculer les points d'attache précis
        
        # Hanches
        hip_params = dict(
            bodyA=self.torse,
            enableMotor=True,
            maxMotorTorque=80.0,
            enableLimit=True,
            lowerAngle=-0.5,
            upperAngle=0.7
        )

        self.joint_hanche_g = self.world.CreateJoint(revoluteJointDef(
            **hip_params, bodyB=self.cuisse_left, anchor=(x_torse, y_torse - 0.15)
        ))
        self.joint_hanche_d = self.world.CreateJoint(revoluteJointDef(
            **hip_params, bodyB=self.cuisse_right, anchor=(x_torse, y_torse - 0.15)
        ))

        # Genoux 
        knee_params = dict(
            enableMotor=True,
            maxMotorTorque=60.0,
            enableLimit=True,
            lowerAngle=-1.0,
            upperAngle=0.0
        )

        self.joint_genou_g = self.world.CreateJoint(revoluteJointDef(
            **knee_params, bodyA=self.cuisse_left, bodyB=self.tibia_left, anchor=(x_torse, y_torse - 0.5)
        ))
        self.joint_genou_d = self.world.CreateJoint(revoluteJointDef(
            **knee_params, bodyA=self.cuisse_right, bodyB=self.tibia_right, anchor=(x_torse, y_torse - 0.5)
        ))

        # On stocke tout pour le nettoyage futur
        self.bodies = [self.torse, self.cuisse_left, self.cuisse_right, self.tibia_left, self.tibia_right]
        
        # On retourne le torse (l'objet principal pour le tracking)
        return self.torse
    
    
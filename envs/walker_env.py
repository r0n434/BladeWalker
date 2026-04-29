import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Box2D
from Box2D.b2 import world, polygonShape, revoluteJointDef, fixtureDef

class WalkerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super(WalkerEnv, self).__init__()
        self.render_mode = render_mode 
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        self.world = world(gravity=(0, -10), doSleep=True)
        self.walker_body = None 
        self.bodies = [] # On initialise la liste ici pour éviter les erreurs !

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 

        # Nettoyage si un robot existait déjà
        if self.walker_body is not None:
            for body in self.bodies:
                self.world.DestroyBody(body)

        # Création du robot
        self.walker_body = self._create_robot(x_torse=5.0, y_torse=10.0)

        # Observation temporaire (14 zéros)
        observation = np.zeros(14, dtype=np.float32)
        
        return observation, {}

    def _create_robot(self, x_torse, y_torse):
        # --- DÉFINITION DES FIXTURES (Propriétés physiques) ---
        
        # Fixture du torse
        hull_fixture = fixtureDef(
            shape=polygonShape(box=(0.25, 0.15)),
            density=5.0,
            friction=0.1
        )

        # Fixture des cuisses
        cuisse_fixture = fixtureDef(
            shape=polygonShape(box=(0.06, 0.2)),
            density=1.0,
            friction=0.2
        )

        # Fixture des tibias
        tibia_fixture = fixtureDef(
            shape=polygonShape(box=(0.05, 0.2)),
            density=1.0,
            friction=0.2
        )

        # --- CRÉATION DES CORPS (En utilisant les fixtures) ---
        self.torse = self.world.CreateDynamicBody(
            position=(x_torse, y_torse),
            fixtures=hull_fixture # On passe la fixture complète ici
        )

        self.cuisse_left  = self.world.CreateDynamicBody(position=(x_torse, y_torse - 0.3), fixtures=cuisse_fixture)
        self.cuisse_right = self.world.CreateDynamicBody(position=(x_torse, y_torse - 0.3), fixtures=cuisse_fixture)
        self.tibia_left   = self.world.CreateDynamicBody(position=(x_torse, y_torse - 0.7), fixtures=tibia_fixture)
        self.tibia_right  = self.world.CreateDynamicBody(position=(x_torse, y_torse - 0.7), fixtures=tibia_fixture)

        # --- JOINTS ---
        hip_params = dict(bodyA=self.torse, enableMotor=True, maxMotorTorque=80.0, enableLimit=True, lowerAngle=-0.5, upperAngle=0.7)

        self.joint_hanche_g = self.world.CreateJoint(revoluteJointDef(**hip_params, bodyB=self.cuisse_left, anchor=(x_torse, y_torse - 0.15)))
        self.joint_hanche_d = self.world.CreateJoint(revoluteJointDef(**hip_params, bodyB=self.cuisse_right, anchor=(x_torse, y_torse - 0.15)))

        knee_params = dict(enableMotor=True, maxMotorTorque=60.0, enableLimit=True, lowerAngle=-1.0, upperAngle=0.0)

        self.joint_genou_g = self.world.CreateJoint(revoluteJointDef(**knee_params, bodyA=self.cuisse_left, bodyB=self.tibia_left, anchor=(x_torse, y_torse - 0.5)))
        self.joint_genou_d = self.world.CreateJoint(revoluteJointDef(**knee_params, bodyA=self.cuisse_right, bodyB=self.tibia_right, anchor=(x_torse, y_torse - 0.5)))

        self.bodies = [self.torse, self.cuisse_left, self.cuisse_right, self.tibia_left, self.tibia_right]
        return self.torse

# ==========================================
# SCRIPT DE TEST (À mettre tout en bas)
# ==========================================
if __name__ == "__main__":
    print(" Test de l'environnement BladeWalker...")
    
    # 1. Instanciation
    env = WalkerEnv()
    
    # 2. Test du Reset
    obs, info = env.reset()
    
    print(" Reset réussi !")
    print(f"Observation (doit être 14 zéros) : {obs}")
    print(f"Position du torse : {env.torse.position}")
    print(f"Nombre de membres créés : {len(env.bodies)}")
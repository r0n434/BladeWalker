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
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # Nettoyage si un robot existait déjà
        if self.walker_body is not None:
            for body in self.bodies:
                self.world.DestroyBody(body)
        self.bodies = []
        # Création du robot
        self.walker_body = self._create_robot(x_torse=5.0, y_torse=10.0)
        self.start_x = float(self.torse.position.x)   # position initiale (≈ 5.0)
        self.prev_x = float(self.torse.position.x)

        observation = self.get_observation()
        
        return observation, {}
    
    def get_observation(self):
        # Récupération des données du torse
        torse_pos = self.torse.position
        torse_vel = self.torse.linearVelocity
        
        # Détection de contact simplifiée (en attendant un ContactListener)
        # On vérifie si le tibia est proche du sol (y=0)
        contact_g = 1.0 if any(contact.touching for contact in self.tibia_left.contacts) else 0.0
        contact_d = 1.0 if any(contact.touching for contact in self.tibia_right.contacts) else 0.0

        observation = np.array([
            # Torse (4)
            self.torse.angle,
            self.torse.angularVelocity,
            torse_vel.x,
            torse_vel.y,
            
            # Jambe Gauche (5)
            self.joint_hanche_g.angle,
            self.joint_hanche_g.speed,
            self.joint_genou_g.angle,
            self.joint_genou_g.speed,
            contact_g,
            
            # Jambe Droite (5)
            self.joint_hanche_d.angle,
            self.joint_hanche_d.speed,
            self.joint_genou_d.angle,
            self.joint_genou_d.speed,
            contact_d
        ], dtype=np.float32)
        
        return observation
    
    def step(self, action):
        # --- 1. APPLIQUER LES ACTIONS ---
        # On transforme les actions [-1, 1] en vitesse pour les joints
        MOTOR_SPEED = 4.0 # rad/s
        
        self.joint_hanche_g.motorSpeed = float(action[0]) * MOTOR_SPEED
        self.joint_genou_g.motorSpeed = float(action[1]) * MOTOR_SPEED
        self.joint_hanche_d.motorSpeed = float(action[2]) * MOTOR_SPEED
        self.joint_genou_d.motorSpeed = float(action[3]) * MOTOR_SPEED

        # --- 2. AVANCER LA PHYSIQUE ---
        # On fait avancer le monde d'un pas (1/50 sec)
        self.world.Step(1.0/50.0, 6, 2)
        self.steps += 1 # Incrémenter notre compteur de temps

        # --- 3. RÉCUPÉRER L'OBSERVATION (14 variables) ---
        # On utilise la logique définie précédemment
        obs = self.get_observation()

        

        # --- 5. DÉTERMINER LA FIN (Terminated / Truncated) ---
        terminated = False
        
        # Chute : le torse penche trop (> 1.0 rad) ou touche le sol
        if abs(self.torse.angle) > 1.0:
            terminated = True
            
        
        # Le robot recule trop
        if self.torse.position.x < self.start_x - 0.5:
            terminated = True

        # Fin de temps : on limite l'essai à 1000 pas (Truncated)
        truncated = False
        if self.steps >= 1000:
            truncated = True
        
        

        # --- 6. RETOURNER LES RÉSULTATS ---
        return obs, self._compute_reward(action, terminated), terminated, truncated, {}

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
    
    def _compute_reward(self, action, terminated):
        """
        Reward conforme à la section '9. La fonction de récompense' du README.

        Étape 1 : avancer (velocity_x) -> version robuste: delta_x / dt ?
        Étape 2 : pénalité énergie: -0.003 * sum(action^2)
        Étape 3 : pénalité chute: -100 si terminated
        Étape 4 : bonus survie (optionnel): +0.001
        """
        # dt doit être cohérent avec ton world.Step(...)
        # (idéalement self.TIME_STEP = 1/50 si tu suis le README)
        dt = getattr(self, "TIME_STEP", 1.0 / self.metadata.get("render_fps", 50))

        x = float(self.torse.position.x)

        # Forward: utiliser delta_x/dt (recommandé dans le README)
        prev_x = getattr(self, "prev_x", x)
        velocity_x = (x - float(prev_x)) / dt
        self.prev_x = x

        # Étape 1
        reward = velocity_x

        # Étape 2: énergie
        action = np.asarray(action, dtype=np.float32)
        reward -= 0.003 * float(np.sum(action ** 2))

        # Étape 3: chute
        if terminated:
            reward -= 100.0

        # Étape 4 (optionnel)
        reward += 0.001

        return float(reward)

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
    print(f"Observation : {obs}")
    print(f"Position du torse : {env.torse.position}")
    print(f"Nombre de membres créés : {len(env.bodies)}")
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from pointeur_env import RoboticArmPointeurEnv
from stable_baselines3 import PPO, SAC
import argparse

from wrapper import ObstacleWrapper

def PPO_train(segment_lengths=[1.0, 1.0], total_timesteps=500000, policy_kwargs=None,n_envs=4,model_name="ppo_arm_obstacles"):
    """
    Fonction d'entraînement avec PPO (Proximal Policy Optimization).
    
    PPO en résumé (La Force Tranquille) :
    - Standard de l'industrie : Robuste et stable.
    - On-Policy : Apprend uniquement de ce qu'il est en train de faire, puis oublie (pas de mémoire à long terme).
    - "Proximal" : Bride ses mises à jour pour avancer à petits pas prudents et ne pas détruire son cerveau sur un "coup de chance".
    - Architecture : Nécessite moins de calculs, un réseau de [64, 64] est souvent suffisant.
    """
    env_kwargs = {"segment_lengths": segment_lengths}
    wrapper_kwargs = {}
    
    vec_env = make_vec_env(
        RoboticArmPointeurEnv, 
        n_envs=n_envs, 
        env_kwargs=env_kwargs,
        wrapper_class=ObstacleWrapper,
        wrapper_kwargs=wrapper_kwargs
    )

    for i in range(n_envs):
        vec_env.env_method("set_difficulty", 0.1, indices=i)

    curriculum_callback = CurriculumCallback(total_timesteps=total_timesteps, initial_difficulty=0.1)
    
    if policy_kwargs is None:
        policy_kwargs = dict(net_arch=[64, 64])

    model = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)

    print(f"--- Début de l'entraînement PPO ( {total_timesteps} étapes sur {n_envs} coeurs ) ---")

    model.learn(total_timesteps=total_timesteps,callback=curriculum_callback) 
    print("--- Entraînement terminé ! ---")

    model.save(f"models/{model_name}")
    print(f"Modèle sauvegardé sous le nom '{model_name}.zip'")

    vec_env.close()

def SAC_train(segment_lengths=[1.0, 1.0], total_timesteps=200000, policy_kwargs=None, model_name="sac_arm_obstacles"):
    """
    Fonction d'entraînement avec SAC (Soft Actor-Critic).
    
    SAC en résumé (L'Explorateur Créatif) :
    - Idéal pour la robotique : Taillé pour les environnements à actions continues.
    - Off-Policy : Possède une mémoire (Replay Buffer) et s'entraîne en piochant dans ses expériences passées.
    - "Soft" (Entropie maximale) : Cherche à maximiser la récompense ET à agir de manière créative/imprévisible pour trouver plusieurs solutions viables.
    - Sample Efficient : Apprend plus vite (nécessite moins d'étapes) mais fait plus de calculs par étape.
    """

    env = RoboticArmPointeurEnv(segment_lengths=segment_lengths)
    env = ObstacleWrapper(env)

    env.unwrapped.set_difficulty(0.1)

    curriculum_callback = CurriculumCallback(total_timesteps=total_timesteps, initial_difficulty=0.1)

    if policy_kwargs is None:
        policy_kwargs = dict(net_arch=[256, 256])

    print(f"--- Début de l'entraînement SAC ( {total_timesteps} étapes ) ---")

    model = SAC("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=total_timesteps,callback=curriculum_callback) 
    print("--- Entraînement terminé ! ---")

    model.save(f"models/{model_name}")
    print(f"Modèle sauvegardé sous le nom '{model_name}.zip'")

    env.close()

class CurriculumCallback(BaseCallback):
    """
    Callback personnalisé qui augmente la difficulté de l'environnement 
    au fur et à mesure de l'entraînement.
    """
    def __init__(self, total_timesteps, initial_difficulty=0.1, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.initial_difficulty = initial_difficulty

    def _on_step(self) -> bool:
        # On calcule le pourcentage d'avancement de l'entraînement
        # On divise par 0.8 pour que la difficulté atteigne 1.0 à 80% du temps total
        progress = self.num_timesteps / (self.total_timesteps * 0.8)
        
        # La difficulté augmente linéairement
        current_difficulty = min(1.0, self.initial_difficulty + progress)

        # On injecte cette nouvelle difficulté dans l'environnement
        # env_method est la fonction de SB3 pour parler à l'environnement
        # Vérification : est-ce un VecEnv (PPO) ou un env normal encapsulé (SAC) ?
        if hasattr(self.training_env, 'env_method'):
            # Cas PPO (VecEnv)
            self.training_env.env_method("set_difficulty", current_difficulty)
        else:
            # Cas SAC (Environnement Gym normal + Wrapper)
            # On passe au travers du Wrapper avec .unwrapped
            if hasattr(self.training_env, 'envs'):
                self.training_env.envs[0].unwrapped.set_difficulty(current_difficulty)

        # Petit affichage tous les 10 000 steps pour voir l'évolution en console
        if self.num_timesteps % 10000 == 0:
            print(f"Step: {self.num_timesteps} | Difficulté: {current_difficulty:.2f}")

        return True
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Entraînement du bras robotique avec PPO ou SAC.")
    
    # --algo : Choix obligatoire entre 'ppo' et 'sac'
    parser.add_argument("--algo", type=str, choices=['ppo', 'sac'], 
                        help="L'algorithme à utiliser : 'ppo' ou 'sac'.")
    
    # --timesteps : Optionnel 
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Nombre total d'étapes (ex: 300000).")
    
    # --segments : Optionnel.
    parser.add_argument("--segments", type=float, nargs='+', default=[1.0, 1.0],
                        help="Longueurs des segments. Exemple: --segments 1.0 1.0 0.5")
    
    # --net_arch : Optionnel.
    parser.add_argument("--net_arch", type=int, nargs='+', default=None,
                        help="Architecture du réseau. Exemple: --net_arch 128 128")
    
    # --n_envs : Optionnel, uniquement pour PPO.
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Nombre d'environnements parallèles pour PPO (ex: 8).")

    # --model_name : Optionnel, nom du fichier de sauvegarde du modèle.
    parser.add_argument("--model_name", type=str, default=None,
                        help="Nom du modèle sauvegardé (sans extension). Exemple: --model_name mon_ppo_arm")
    
    args = parser.parse_args()

    policy_kwargs = None
    if args.net_arch is not None:
        policy_kwargs = dict(net_arch=args.net_arch)

    if args.algo is None or args.algo == 'sac':
        ts = args.timesteps if args.timesteps is not None else 200000
        print(f"Lancement de SAC | Segments: {args.segments} | Architecture: {args.net_arch or '[256, 256] (défaut)'}")
        SAC_train(segment_lengths=args.segments, total_timesteps=ts, policy_kwargs=policy_kwargs, model_name=args.model_name or "sac_arm_obstacles")
    elif args.algo == 'ppo':
        ts = args.timesteps if args.timesteps is not None else 500000
        print(f"Lancement de PPO | Segments: {args.segments} | Architecture: {args.net_arch or '[64, 64] (défaut)'}")
        n_envs = args.n_envs if args.n_envs is not None else 4
        PPO_train(segment_lengths=args.segments, total_timesteps=ts, policy_kwargs=policy_kwargs, n_envs=args.n_envs, model_name=args.model_name or "ppo_arm_obstacles")
        

        
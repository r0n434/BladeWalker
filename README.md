Tâches EvoWalker
🏗️ Setup

T0 — Initialiser le projet, venv, dépendances (gymnasium[box2d], tensorflow, pygame, tensorboard)

🌍 Environnement

T1 — Créer le squelette WalkerEnv(gymnasium.Env) avec observation_space et action_space définis
T2 — Construire le corps Box2D (torse + 2 jambes, 4 segments total, joints motorisés)
T3 — Implémenter reset() et step() (physique + détection chute)
T4 — Implémenter la reward function (vitesse_x uniquement pour commencer)
T5 — Valider l'env avec des actions random + rendu "human"

🧠 Réseau

T6 — Implémenter ActorCritic en TensorFlow (shared backbone, tête acteur gaussienne, tête critique)

⚡ Entraînement

T7 — Implémenter RolloutBuffer (stockage des trajectoires)
T8 [OPTIONNEL SB3] — Boucle d'entraînement avec stable_baselines3.PPO sur l'env custom → valide que l'env fonctionne
T9 — Implémenter PPO from scratch (loss clip + GAE + entropie)
T10 — Boucle train.py principale avec checkpoints

📊 Monitoring

T11 — Logger TensorBoard (reward, loss, entropy par itération)
T12 — RecordVideo wrapper pour capturer les checkpoints
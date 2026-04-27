 EvoWalker 🦿

> Apprentissage par renforcement d'un bipède simulé — from scratch avec Python, TensorFlow, Gymnasium et Box2D.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Stack technique](#2-stack-technique)
3. [Architecture du projet](#3-architecture-du-projet)
4. [Plan de développement (tâches)](#4-plan-de-développement-tâches)
5. [L'environnement Gymnasium](#5-lenvironnement-gymnasium)
6. [Le corps Box2D](#6-le-corps-box2d)
7. [Le réseau de neurones](#7-le-réseau-de-neurones)
8. [L'algorithme PPO](#8-lalgorithme-ppo)
9. [La fonction de récompense](#9-la-fonction-de-récompense)
10. [Entraînement & monitoring](#10-entraînement--monitoring)
11. [Visualisation & vidéos](#11-visualisation--vidéos)
12. [Installation](#12-installation)
13. [Pièges fréquents](#13-pièges-fréquents)
14. [Références](#14-références)

---

## 1. Vue d'ensemble

EvoWalker est un projet d'apprentissage par renforcement (**Reinforcement Learning, RL**) dont l'objectif est d'entraîner un agent bipède simulé à marcher le plus loin possible vers l'avant, uniquement en apprenant par essais et erreurs — sans règles explicites ni démonstrations humaines.

Le projet est conçu pour être **implémenté manuellement** : on n'utilise pas de framework RL clé en main comme Stable-Baselines3 (sauf en phase de validation, voir T8), mais on construit les briques une à une pour comprendre chaque composant.

### Principe général

```
┌─────────────────────────────────────────────────────────────────┐
│                        Boucle RL                                │
│                                                                 │
│   ┌──────────┐   observation s_t   ┌──────────────────────┐    │
│   │  Monde   │ ──────────────────► │  Policy π(a | s)     │    │
│   │  Box2D   │                     │  Réseau acteur-       │    │
│   │          │ ◄────────────────── │  critique (TF)        │    │
│   │          │   action a_t        └──────────────────────┘    │
│   │          │                                                  │
│   │          │ ──────────────────► reward r_t, done, s_{t+1}   │
│   └──────────┘                                                  │
│                                                                 │
│   Après N steps : mise à jour des poids (PPO)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Stack technique

### Langage

| Composant | Choix | Justification |
|---|---|---|
| Langage | **Python 3.11+** | Standard de l'écosystème RL/ML |
| IA (algorithme) | **PPO** (from scratch) | Stable, état de l'art pour l'espace continu |
| IA (fallback) | **Stable-Baselines3** | Validation rapide de l'env avant d'implémenter PPO |
| Réseaux de neurones | **TensorFlow 2.x / Keras** | API haut niveau, compatible avec l'écosystème |
| Environnement | **Gymnasium** | Standard industrie pour l'interface Agent/Monde |
| Moteur physique | **Box2D** (via `gymnasium[box2d]`) | Gravité, friction, collisions, joints motorisés |
| Rendu | **PyGame** (intégré à Gymnasium) | Rendu 2D léger, déjà packagé avec Gym |
| Monitoring | **TensorBoard** | Visualisation des métriques en temps réel |
| Enregistrement | **gymnasium.wrappers.RecordVideo** | Capture MP4 des épisodes |

### Versions recommandées

```
python        >= 3.11
tensorflow    >= 2.15
gymnasium     >= 0.29
box2d-py      >= 2.3.10
pygame        >= 2.5
stable-baselines3 >= 2.3   # pour T8 uniquement
tensorboard   >= 2.15
numpy         >= 1.26
```

---

## 3. Architecture du projet

```
EvoWalker/
│
├── envs/
│   └── walker_env.py           # Environnement Gymnasium custom
│                               # Contient : observation_space, action_space,
│                               # reset(), step(), render(), close()
│                               # Corps Box2D, détection de chute, reward
│
├── models/
│   └── policy_network.py       # Réseau acteur-critique TensorFlow
│                               # Contient : ActorCritic (backbone + têtes),
│                               # méthode call(), get_action(), get_value()
│
├── training/
│   ├── ppo.py                  # Algorithme PPO from scratch
│   │                           # Contient : PPO.update(), calcul GAE,
│   │                           # loss clip, loss critique, loss entropie
│   └── rollout_buffer.py       # Buffer de trajectoires
│                               # Contient : add(), get(), compute_returns()
│
├── utils/
│   ├── logger.py               # Wrapper TensorBoard + console
│   └── normalizer.py           # RunningMeanStd pour normaliser les observations
│
├── checkpoints/                # Sauvegardes des poids (.weights.h5)
│   └── iter_XXXX/
│
├── videos/                     # Rendus MP4 des épisodes
│
├── train.py                    # Point d'entrée principal
├── evaluate.py                 # Chargement d'un checkpoint + rendu
├── requirements.txt
└── README.md
```

### Responsabilités par fichier

#### `envs/walker_env.py`
C'est la pièce centrale. Il hérite de `gymnasium.Env` et encapsule **tout** ce qui concerne le monde physique :
- Création du monde Box2D et du corps du bipède
- Définition de `observation_space` (angles, vitesses, contact sol, etc.)
- Définition de `action_space` (torques appliqués aux joints, espace continu)
- `reset()` : réinitialise le monde et retourne l'observation initiale
- `step(action)` : applique l'action, avance la simulation, calcule la récompense, détecte la fin d'épisode
- `render()` : délègue à PyGame

#### `models/policy_network.py`
Contient uniquement l'architecture du réseau. Aucune logique d'entraînement ici. Le réseau est séparé en acteur (produit une distribution sur les actions) et critique (estime la valeur de l'état).

#### `training/ppo.py`
Contient uniquement la logique de mise à jour PPO. Prend un `RolloutBuffer` en entrée, retourne les métriques de loss. Ne sait rien de l'environnement.

#### `training/rollout_buffer.py`
Stocke les transitions `(s, a, r, done, log_prob, value)` collectées lors d'un rollout. Calcule les returns et avantages (GAE) à la fin de la collecte.

#### `train.py`
Orchestre tout : instancie l'env, le modèle, le buffer, le PPO, et lance la boucle principale.

---

## 4. Plan de développement (tâches)

### 🏗️ Setup
- **T0** — Initialiser le projet, venv, `requirements.txt`, structure des dossiers

### 🌍 Environnement
- **T1** — Squelette `WalkerEnv` : héritages, spaces définis, méthodes vides
- **T2** — Construction du corps Box2D (torse + 2 jambes, joints motorisés)
- **T3** — Implémentation de `reset()` et `step()` (physique + détection chute)
- **T4** — Implémentation de la reward function (itération)
- **T5** — Validation de l'env avec actions random + rendu `"human"`

### 🧠 Réseau
- **T6** — `ActorCritic` en TensorFlow (backbone partagé, tête gaussienne, tête valeur)

### ⚡ Entraînement
- **T7** — `RolloutBuffer` (stockage, GAE, batch)
- **T8 [OPTIONNEL SB3]** — Boucle SB3 sur l'env custom (valide l'env avant PPO custom)
- **T9** — PPO from scratch (clip loss + GAE + entropie)
- **T10** — `train.py` : boucle principale + checkpoints

### 📊 Monitoring
- **T11** — Logger TensorBoard
- **T12** — `RecordVideo` wrapper

> **Stratégie** : T8 peut se faire après T5, sans attendre T6-T9. Ça permet de voir le walker marcher tôt et de valider l'env avant d'y investir plus de temps.

---

## 5. L'environnement Gymnasium

### Interface standard

Tout environnement Gymnasium implémente cette interface :

```python
class WalkerEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        # Définir observation_space et action_space
        ...

    def reset(self, seed=None, options=None):
        # Réinitialiser le monde, retourner (observation, info)
        ...

    def step(self, action):
        # Appliquer action, retourner (obs, reward, terminated, truncated, info)
        ...

    def render(self):
        # Rendu PyGame si render_mode est défini
        ...

    def close(self):
        # Fermer PyGame
        ...
```

### Observation space

L'observation est un vecteur de flottants décrivant l'état complet du bipède :

| Index | Valeur | Range typique |
|---|---|---|
| 0 | Angle du torse (inclinaison) | [-π, π] |
| 1 | Vitesse angulaire du torse | [-∞, ∞] |
| 2 | Vitesse horizontale du torse | [-∞, ∞] |
| 3 | Vitesse verticale du torse | [-∞, ∞] |
| 4 | Angle joint hanche gauche | [-π, π] |
| 5 | Vitesse angulaire hanche gauche | [-∞, ∞] |
| 6 | Angle joint genou gauche | [-π, π] |
| 7 | Vitesse angulaire genou gauche | [-∞, ∞] |
| 8 | Contact sol jambe gauche | {0, 1} |
| 9 | Angle joint hanche droite | [-π, π] |
| 10 | Vitesse angulaire hanche droite | [-∞, ∞] |
| 11 | Angle joint genou droit | [-π, π] |
| 12 | Vitesse angulaire genou droit | [-∞, ∞] |
| 13 | Contact sol jambe droite | {0, 1} |

→ `observation_space = Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)`

### Action space

4 actions continues correspondant aux torques appliqués aux 4 joints (hanche G, genou G, hanche D, genou D) :

→ `action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)`

Les valeurs sont normalisées à [-1, 1] et multipliées par `MOTOR_SPEED` dans `step()`.

### Modes de rendu

- `"human"` : fenêtre PyGame temps réel (pour observer pendant/après entraînement)
- `"rgb_array"` : retourne un array numpy (pour `RecordVideo`)
- `None` : headless (entraînement pur, plus rapide)

---

## 6. Le corps Box2D

### Structure du bipède

```
          [TORSE]
         /       \
    [HANCHE G]  [HANCHE D]
        |             |
    [CUISSE G]   [CUISSE D]
        |             |
    [GENOU G]    [GENOU D]
        |             |
    [TIBIA G]    [TIBIA D]
```

### Segments et joints

| Segment | Corps Box2D | Dimensions approx. |
|---|---|---|
| Torse | `b2Body` rectangulaire | 0.5 × 0.3 m |
| Cuisse (×2) | `b2Body` rectangulaire | 0.12 × 0.4 m |
| Tibia (×2) | `b2Body` rectangulaire | 0.1 × 0.4 m |

Les joints sont des `b2RevoluteJoint` avec :
- Limites angulaires (`lowerAngle`, `upperAngle`) pour éviter des postures impossibles
- Moteur (`enableMotor=True`, `maxMotorTorque`) contrôlé par les actions

### Paramètres physiques importants

```python
GRAVITY       = (0, -10)   # m/s²
SCALE         = 30.0        # pixels par mètre (pour le rendu)
FPS           = 50          # frames par seconde
MOTOR_SPEED   = 4           # rad/s max pour les joints
MOTOR_TORQUE  = 80          # N·m max
HULL_DENSITY  = 5.0
HULL_FRICTION = 0.1
LEG_DENSITY   = 1.0
LEG_FRICTION  = 0.2
GROUND_FRICTION = 0.8
```

### Détection de chute

L'épisode se termine (`terminated = True`) si :
- Le torse touche le sol (contact callback Box2D)
- L'angle du torse dépasse un seuil (ex: |angle| > 1.0 rad ≈ 57°)
- Le walker recule trop (position_x < position_initiale - 0.5)

L'épisode est tronqué (`truncated = True`) si on dépasse `MAX_STEPS = 1000`.

---

## 7. Le réseau de neurones

### Architecture Acteur-Critique

```
Observation (14,)
      │
  Dense(64, tanh)
      │
  Dense(64, tanh)
      │
  ┌───┴───────────────────┐
  │                       │
Acteur                  Critique
Dense(4)                Dense(1)
(mean des actions)      (valeur V(s))
  +
log_std (paramètre
  appris indépendant)
  │
Distribution Normale
N(mean, exp(log_std))
  │
sample → action
log_prob(action)
```

### Détails d'implémentation

```python
class ActorCritic(tf.keras.Model):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Backbone partagé
        self.shared = [
            Dense(64, activation='tanh'),
            Dense(64, activation='tanh'),
        ]
        # Têtes séparées
        self.actor_mean = Dense(act_dim)          # mean de la gaussienne
        self.log_std = tf.Variable(               # std apprise, indépendante de s
            tf.zeros(act_dim), trainable=True
        )
        self.critic = Dense(1)                    # valeur scalaire V(s)

    def call(self, obs):
        x = obs
        for layer in self.shared:
            x = layer(x)
        mean = self.actor_mean(x)
        value = self.critic(x)
        return mean, self.log_std, value

    def get_action(self, obs):
        mean, log_std, value = self(obs)
        std = tf.exp(log_std)
        dist = tfp.distributions.Normal(mean, std)
        action = dist.sample()
        action = tf.clip_by_value(action, -1.0, 1.0)
        log_prob = tf.reduce_sum(dist.log_prob(action), axis=-1)
        return action, log_prob, value
```

### Pourquoi une distribution gaussienne ?

L'espace d'action est **continu** (torques en [-1, 1]). On ne peut pas sortir une action déterministe — on a besoin d'**exploration**. La policy sort donc une distribution normale `N(μ, σ)` dont on sample l'action. La std `σ` contrôle l'exploration : grande au début, elle diminue à mesure que le réseau converge.

### Normalisation des observations

Les valeurs brutes de Box2D ont des échelles très différentes (angles en radians vs vitesses pouvant dépasser 10). Il faut normaliser avec un `RunningMeanStd` :

```python
obs_normalized = (obs - running_mean) / (running_std + 1e-8)
```

Implémenter dans `utils/normalizer.py` et appliquer dans `train.py` avant chaque appel au réseau.

---

## 8. L'algorithme PPO

### Pourquoi PPO ?

PPO (Proximal Policy Optimization, Schulman et al. 2017) est l'algorithme de référence pour les espaces d'action continus. Son avantage par rapport à d'autres algos comme REINFORCE ou A3C : il contrôle la **taille du pas de mise à jour** pour éviter de détruire la policy en un seul gradient step.

### Généralised Advantage Estimation (GAE)

Avant la mise à jour, on calcule l'**avantage** `A(s, a)` qui mesure si l'action prise était meilleure ou moins bonne qu'attendu.

GAE combine les returns à court et long terme via le paramètre `λ` :

```
δ_t   = r_t + γ · V(s_{t+1}) - V(s_t)          # TD error
A_t   = δ_t + (γλ) · δ_{t+1} + (γλ)² · δ_{t+2} + ...
```

En pratique, on calcule ça de façon récursive **en remontant** le rollout :

```python
gae = 0
advantages = []
for t in reversed(range(len(rewards))):
    delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
    gae = delta + gamma * lam * (1 - dones[t]) * gae
    advantages.insert(0, gae)
returns = [adv + val for adv, val in zip(advantages, values[:-1])]
```

### Loss PPO (clippée)

```
ratio     = π_new(a|s) / π_old(a|s)
           = exp(log_prob_new - log_prob_old)

L_clip    = -mean( min(
                ratio * A,
                clip(ratio, 1-ε, 1+ε) * A
            ) )

L_value   = MSE(V(s), returns)

L_entropy = -β · H(π)    # H = -mean(log_prob)

L_total   = L_clip + c1 · L_value - c2 · L_entropy
```

Le **clip** est la clé de PPO : si le ratio s'éloigne trop de 1 (la nouvelle policy diverge trop de l'ancienne), on plafonne le gradient. Ça stabilise l'entraînement.

### Hyperparamètres de départ

| Paramètre | Valeur | Rôle |
|---|---|---|
| `gamma` | 0.99 | Facteur de discount (importance du futur) |
| `lam` | 0.95 | GAE lambda (biais/variance tradeoff) |
| `clip_eps` | 0.2 | Clipping du ratio |
| `lr` | 3e-4 | Learning rate Adam |
| `n_steps` | 2048 | Steps par rollout |
| `batch_size` | 64 | Taille des mini-batches |
| `n_epochs` | 10 | Passes sur le buffer par update |
| `c1` | 0.5 | Coefficient loss critique |
| `c2` | 0.01 | Coefficient entropie |

### Boucle principale

```python
for iteration in range(MAX_ITERATIONS):
    # 1. Collecte du rollout
    rollout = collect_rollout(env, policy, n_steps=2048)

    # 2. Calcul des avantages et returns (GAE)
    rollout.compute_returns_and_advantages(gamma=0.99, lam=0.95)

    # 3. Mise à jour PPO sur K epochs
    for epoch in range(n_epochs):
        for batch in rollout.get_batches(batch_size=64):
            metrics = ppo.update(batch)

    # 4. Logging
    logger.log(iteration, metrics, rollout.mean_reward)

    # 5. Checkpoint
    if iteration % SAVE_EVERY == 0:
        policy.save_weights(f"checkpoints/iter_{iteration:04d}")
```

---

## 9. La fonction de récompense

La reward function est **l'endroit le plus impactant** du projet. Une mauvaise reward produit des comportements dégénérés même avec un algo parfait.

### Évolution progressive recommandée

#### Étape 1 — Reward de base (commencer ici)
```python
reward = velocity_x  # Juste avancer
```
Simple, mais le walker peut "tricher" en sautant sur place ou en se traînant.

#### Étape 2 — Ajouter pénalité d'énergie
```python
reward = velocity_x - 0.003 * np.sum(action**2)
```
Force à économiser les mouvements parasites.

#### Étape 3 — Pénalité de chute
```python
if terminated:
    reward -= 100
```
Décourage fortement la chute.

#### Étape 4 — Récompense de survie (optionnel)
```python
reward += 0.001  # Petit bonus par step survécu
```
Encourage à tenir debout plus longtemps.

### Comportements dégénérés typiques

| Comportement | Cause probable | Correctif |
|---|---|---|
| Le walker saute sur place | Reward de vitesse mal calculée (spike au reset) | Utiliser `delta_x / dt` et non la vitesse instantanée Box2D |
| Il rampe au sol | Pas de pénalité de chute | Ajouter `terminated penalty` |
| Il ne bouge pas | Reward trop faible, std s'effondre | Augmenter `c2` (entropie) |
| Il tombe instantanément | Limites des joints trop larges | Restreindre `lowerAngle/upperAngle` |

---

## 10. Entraînement & monitoring

### Lancer un entraînement

```bash
# Entraînement headless (rapide)
python train.py

# Avec rendu temps réel (lent mais utile pour débugger l'env)
python train.py --render

# Avec SB3 (T8, pour valider l'env)
python train.py --sb3
```

### Métriques à surveiller

| Métrique | Attendu | Problème si... |
|---|---|---|
| `mean_reward` | Croissante sur ~500k steps | Plateaue dès le début → reward function cassée |
| `policy_loss` | Décroissante puis stable | Explose → lr trop élevé |
| `value_loss` | Décroissante | Ne converge pas → backbone trop petit |
| `entropy` | Décroît lentement | Chute rapide → exploration trop faible, augmenter `c2` |
| `episode_length` | Croissante | Reste à 1-2 steps → détection de chute trop agressive |

### TensorBoard

```bash
tensorboard --logdir=runs/
# → http://localhost:6006
```

### Checkpoints

Les poids sont sauvegardés en `.weights.h5` (format TF2 natif) :

```
checkpoints/
├── iter_0000.weights.h5
├── iter_0100.weights.h5
└── iter_0500.weights.h5   ← meilleur checkpoint, on peut comparer
```

---

## 11. Visualisation & vidéos

### Rendu temps réel

```python
env = WalkerEnv(render_mode="human")
```

### Capture vidéo automatique

```python
from gymnasium.wrappers import RecordVideo

env = WalkerEnv(render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder="videos/",
    episode_trigger=lambda ep: ep % 50 == 0  # toutes les 50 épisodes
)
```

### Rejouer un checkpoint

```bash
python evaluate.py --checkpoint checkpoints/iter_0500
```

Le script `evaluate.py` charge les poids, instancie l'env en mode `"human"`, et joue des épisodes en boucle.

---

## 12. Installation

```bash
# 1. Cloner le projet
git clone https://github.com/r0n/EvoWalker.git
cd EvoWalker

# 2. Créer et activer le venv
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

### `requirements.txt`

```
tensorflow>=2.15
gymnasium>=0.29
gymnasium[box2d]
pygame>=2.5
stable-baselines3>=2.3
tensorboard>=2.15
numpy>=1.26
tensorflow-probability>=0.23
```

> **Note Box2D** : `gymnasium[box2d]` installe `box2d-py` via pip. Sur certains systèmes, il faut d'abord `apt install swig` (Linux) ou `brew install swig` (macOS).

---

## 13. Pièges fréquents

### Environnement

- **Ne pas oublier `super().reset(seed=seed)`** dans `reset()` — requis par l'API Gymnasium pour la reproductibilité.
- **Détruire le monde Box2D dans `reset()`** — sinon les corps s'accumulent en mémoire.
- **`terminated` vs `truncated`** : `terminated=True` quand le walker tombe (fin naturelle), `truncated=True` quand on dépasse `MAX_STEPS` (timeout). PPO les traite différemment dans le calcul GAE.

### PPO

- **Ne pas mettre à jour `log_prob_old` pendant les K epochs** — la `π_old` doit rester fixe pendant toute la mise à jour, sinon le ratio est mal calculé.
- **Normaliser les avantages** par batch avant de calculer la loss : `A = (A - mean(A)) / (std(A) + 1e-8)`. Ça stabilise les gradients.
- **Gradient clipping** : `tf.clip_by_global_norm(grads, 0.5)` avant `optimizer.apply_gradients`.

### TensorFlow

- **`@tf.function` sur les méthodes critiques** (`call`, `update`) pour les compiler en graph et gagner en vitesse.
- **Attention aux shapes** : TF est sensible à `(batch, 1)` vs `(batch,)`. Utiliser `.squeeze()` ou `[:, 0]` sur les sorties du critique.

### Performance

- Entraîner en mode **headless** (sans `render_mode`). Le rendu PyGame divise la vitesse par ~10.
- Box2D est monothread. Pour du multi-env parallèle plus tard : `gymnasium.vector.AsyncVectorEnv`.

---

## 14. Références

- **PPO paper** : Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (2017)
- **GAE paper** : Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) (2015)
- **Gymnasium docs** : https://gymnasium.farama.org/
- **Box2D Python** : https://github.com/pybox2d/pybox2d
- **BipedalWalker source** (référence implémentation) : https://github.com/openai/gym/blob/master/gym/envs/box2d/bipedal_walker.py
- **The 37 Implementation Details of PPO** : https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

---

*EvoWalker — projet personnel d'apprentissage RL from scratch.*

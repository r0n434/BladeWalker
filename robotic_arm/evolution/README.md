# EvoArm 🦾

> Evolution morphologique et comportementale d'un bras articulé 2D par NEAT — Python, neat-python, Gymnasium, Pygame.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Stack technique](#2-stack-technique)
3. [Architecture du projet](#3-architecture-du-projet)
4. [Plan de développement (tâches)](#4-plan-de-développement-tâches)
5. [NEAT — principe et fonctionnement](#5-neat--principe-et-fonctionnement)
6. [L'environnement Gymnasium](#6-lenvironnement-gymnasium)
7. [Le génome — morphologie + comportement](#7-le-génome--morphologie--comportement)
8. [La fonction de fitness](#8-la-fonction-de-fitness)
9. [La reward function](#9-la-reward-function)
10. [Configuration NEAT](#10-configuration-neat)
11. [Visualisation](#11-visualisation)
12. [Installation](#12-installation)
13. [Références](#13-références)

---

## 1. Vue d'ensemble

EvoArm est un projet de **morphological evolution** — on ne programme pas comment le bras doit bouger, ni même combien d'articulations il doit avoir. On laisse l'évolution trouver à la fois la **structure** (nombre de joints, longueur des membres) et le **comportement** (comment les bouger) pour atteindre une cible placée aléatoirement.

L'algorithme utilisé est **NEAT** (NeuroEvolution of Augmenting Topologies) — il fait évoluer simultanément la topologie du réseau de neurones et ses poids, ce qui permet d'encoder la morphologie directement dans le génome.

### Principe général

```
┌──────────────────────────────────────────────────────────────┐
│  Population initiale — N individus avec génomes aléatoires   │
│  (nb_joints, longueurs, poids réseau)                        │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│  Évaluation — chaque individu joue K épisodes                │
│  L'env construit le bras selon sa morphologie                │
│  Le réseau contrôle les joints à chaque step                 │
│  Score de fitness = reward cumulée moyenne                   │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│  Sélection + Évolution                                       │
│  - Les meilleurs individus survivent                         │
│  - Mutations : ajouter joints, modifier longueurs, nouveaux  │
│    neurones, nouvelles connexions                            │
│  - Crossover entre individus compatibles (spéciation)        │
└───────────────────────────┬──────────────────────────────────┘
                            │
                    nouvelle génération
                            │
                        (répéter)
```

---

## 2. Stack technique

| Composant | Choix | Justification |
|---|---|---|
| Langage | **Python 3.11+** | Standard écosystème ML |
| Évolution | **neat-python** | Lib NEAT mature, bien documentée |
| Environnement | **Gymnasium** | Interface standard agent/monde |
| Physique | **Numpy** (géométrie 2D) | Pas de gravité → pas besoin de Box2D |
| Rendu | **Pygame** | Rendu 2D léger |
| Monitoring | **Matplotlib** | Courbes de fitness par génération |

### Versions recommandées

```
python       >= 3.11
neat-python  >= 0.92
gymnasium    >= 0.29
numpy        >= 1.26
pygame       >= 2.5
matplotlib   >= 3.8
```

---

## 3. Architecture du projet

```
EvoArm/
│
├── envs/
│   └── arm_env.py              # Environnement Gymnasium custom
│                               # Bras 2D, cinématique directe,
│                               # cible aléatoire, reward function
│
├── evolution/
│   ├── genome.py               # Décodage génome → morphologie
│   ├── evaluator.py            # Évaluation d'un individu NEAT
│   └── neat_config.txt         # Fichier de configuration NEAT
│
├── utils/
│   ├── kinematics.py           # Cinématique directe (FK)
│   └── visualizer.py           # Rendu Pygame + courbes fitness
│
├── checkpoints/                # Sauvegardes des meilleurs génomes
│   └── gen_XXXX.pkl
│
├── train.py                    # Point d'entrée principal
├── evaluate.py                 # Rejouer un génome sauvegardé
├── requirements.txt
└── README.md
```

### Responsabilités par fichier

#### `envs/arm_env.py`
Environnement Gymnasium qui construit le bras selon une morphologie donnée, simule les steps, calcule la reward et détecte la réussite. Interface standard `reset()`, `step()`, `render()`.

#### `evolution/genome.py`
Décode un génome NEAT en morphologie concrète : extrait `nb_joints` et `longueurs` des nœuds d'entrée spéciaux, construit le bras correspondant.

#### `evolution/evaluator.py`
Fonction d'évaluation appelée par NEAT pour chaque individu. Lance K épisodes, retourne le score de fitness moyen.

#### `utils/kinematics.py`
Calcule la position du bout du bras (end-effector) à partir des angles des joints — c'est la **cinématique directe** (Forward Kinematics).

#### `train.py`
Configure NEAT, lance l'évolution, sauvegarde les meilleurs génomes.

---

## 4. Plan de développement (tâches)

### 🏗️ Setup
- **T0** — Initialiser le projet, venv, `requirements.txt`, structure

### 🌍 Environnement
- **T1** — `ArmEnv` squelette : spaces définis, morphologie configurable
- **T2** — Cinématique directe (FK) dans `kinematics.py`
- **T3** — `reset()` et `step()` : appliquer les angles, calculer positions
- **T4** — Reward function complète
- **T5** — Valider l'env avec des actions random + rendu Pygame

### 🧬 Évolution
- **T6** — Fichier de config NEAT (`neat_config.txt`)
- **T7** — `genome.py` : décodage génome → morphologie
- **T8** — `evaluator.py` : fonction de fitness
- **T9** — `train.py` : boucle NEAT principale + checkpoints

### 📊 Monitoring
- **T10** — Courbes de fitness (meilleur, moyenne, médiane par génération)
- **T11** — `evaluate.py` : rejouer un génome sauvegardé avec rendu

---

## 5. NEAT — principe et fonctionnement

### Qu'est-ce que NEAT ?

NEAT (Stanley & Miikkulainen, 2002) est un algorithme évolutionnaire qui fait évoluer des réseaux de neurones. Contrairement à PPO/SAC qui optimisent les **poids d'un réseau fixe**, NEAT fait évoluer simultanément :
- La **topologie** du réseau (nombre de neurones, connexions)
- Les **poids** des connexions

### Le génome

Un génome NEAT encode deux choses :

```
Génome
├── Node genes     → liste des neurones (id, type: input/hidden/output)
└── Connection genes → liste des connexions (from, to, weight, enabled)
```

Dans EvoArm, les nœuds d'entrée spéciaux encodent aussi la morphologie :

```
Inputs du réseau :
  [0]     nb_joints normalisé         ← morphologie
  [1..N]  longueurs des membres       ← morphologie
  [N+1..] angles joints courants      ← état
  [...]   distance cible (x, y)       ← état
  [...]   angle vers cible            ← état

Outputs du réseau :
  [0..N]  vitesses angulaires des joints  ← action
```

### Spéciation

NEAT protège les innovations via la **spéciation** — les individus trop différents sont placés dans des espèces séparées et ne se font pas concurrence directement. Ça donne le temps aux nouvelles structures de se développer avant d'être éliminées.

### Mutations possibles

| Mutation | Effet |
|---|---|
| Add node | Divise une connexion existante, ajoute un neurone |
| Add connection | Ajoute une connexion entre deux neurones existants |
| Modify weight | Perturbe un poids de connexion |
| Enable/disable | Active ou désactive une connexion |

Dans notre cas, `nb_joints` évolue indirectement via les mutations de nœuds — plus le réseau grossit, plus il peut encoder des morphologies complexes.

---

## 6. L'environnement Gymnasium

### Morphologie variable

L'env est instancié avec une morphologie concrète à chaque évaluation :

```python
morphology = {
    "nb_joints" : 3,
    "lengths"   : [1.2, 0.8, 0.5]   # longueur de chaque membre
}
env = ArmEnv(morphology=morphology)
```

### Observation space

L'observation dépend du nombre de joints. Pour `nb_joints = N` :

```
[0]      nb_joints normalisé          (morphologie)
[1..N]   longueurs membres normalisées (morphologie)
[N+1..2N] angles joints courants      (état)
[2N+1]   distance euclidienne cible   (état)
[2N+2]   angle vers cible             (état)
[2N+3]   position x end-effector      (état)
[2N+4]   position y end-effector      (état)
```

Taille totale : `1 + N + N + 4 = 2N + 5`

Pour gérer la taille variable, on fixe une taille maximale et on padde avec des zéros :

```python
MAX_JOINTS = 6
obs_dim    = 2 * MAX_JOINTS + 5   # = 17
```

### Action space

```python
action_space = Box(low=-1.0, high=1.0, shape=(MAX_JOINTS,))
```

Les actions au-delà de `nb_joints` sont ignorées dans `step()`.

### Paramètres

```python
ORIGIN      = (0, 0)      # point d'attache du bras
MIN_JOINTS  = 2
MAX_JOINTS  = 6
MIN_LENGTH  = 0.5
MAX_LENGTH  = 2.0
MAX_STEPS   = 200         # steps par épisode
REACH_DIST  = 0.2         # seuil de succès (distance end-effector/cible)
MOTOR_SPEED = 2.0         # rad/s max par joint
```

---

## 7. Le génome — morphologie + comportement

### Décodage

La morphologie est encodée dans les **poids des connexions vers les nœuds d'entrée spéciaux** :

```python
def decode_genome(genome, config):
    # nb_joints : extrait du premier nœud d'entrée
    # valeur normalisée [0,1] → remappée à [MIN_JOINTS, MAX_JOINTS]
    nb_joints_norm = genome.nodes[INPUT_NODE_ID].bias
    nb_joints = int(round(
        MIN_JOINTS + nb_joints_norm * (MAX_JOINTS - MIN_JOINTS)
    ))

    # longueurs : extraites des nœuds suivants
    lengths = []
    for i in range(nb_joints):
        length_norm = genome.nodes[LENGTH_NODE_IDS[i]].bias
        length = MIN_LENGTH + length_norm * (MAX_LENGTH - MIN_LENGTH)
        lengths.append(length)

    return {"nb_joints": nb_joints, "lengths": lengths}
```

### Évolution de la morphologie

À chaque génération, les mutations NEAT modifient les biais des nœuds spéciaux → `nb_joints` et `lengths` changent progressivement. Une morphologie avec 2 joints peut évoluer vers 4 joints en quelques générations si ça améliore la fitness.

---

## 8. La fonction de fitness

La fitness est ce que NEAT maximise. Elle est calculée par `evaluator.py` :

```python
def evaluate_genome(genome, config):
    morphology = decode_genome(genome, config)
    env        = ArmEnv(morphology=morphology)
    net        = neat.nn.FeedForwardNetwork.create(genome, config)

    total_fitness = 0.0

    for episode in range(N_EVAL_EPISODES):
        obs, _ = env.reset()
        episode_reward = 0.0

        for step in range(MAX_STEPS):
            action = net.activate(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

        total_fitness += episode_reward

    return total_fitness / N_EVAL_EPISODES
```

### Paramètres d'évaluation

```python
N_EVAL_EPISODES = 5    # moyenne sur 5 cibles aléatoires différentes
MAX_STEPS       = 200  # steps max par épisode
```

Évaluer sur plusieurs épisodes évite qu'un individu soit favorisé par une cible facile.

---

## 9. La reward function

```
reward = bonus_proximité - malus_articulations - malus_énergie + bonus_succès
```

### Détail par composante

**`bonus_proximité`** — reward dense à chaque step :
```python
bonus_proximité = (dist_précédente - dist_actuelle) * 10.0
```
Signal continu — NEAT apprend même sans jamais atteindre la cible. Récompense le rapprochement, pénalise l'éloignement.

**`malus_articulations`** — par épisode, calculé une fois au `reset()` :
```python
malus_articulations = nb_joints * 0.1
```
Pression évolutive vers les morphologies simples. Un bras à 2 joints qui atteint la cible sera favorisé sur un bras à 6 joints qui fait la même chose.

**`malus_énergie`** — par step :
```python
malus_énergie = sum(abs(vitesses_angulaires)) * 0.01
```
Encourage les trajectoires fluides. Pénalise les gesticulations inutiles.

**`bonus_succès`** — one-shot quand `distance < REACH_DIST` :
```python
bonus_succès = 100.0
```
Signal fort — la fitness d'un individu qui atteint la cible sera significativement plus haute que celui qui s'en approche seulement.

### Comportements dégénérés à surveiller

| Comportement | Cause | Correctif |
|---|---|---|
| Bras tourne en rond | `malus_énergie` trop faible | Augmenter le coefficient |
| Toujours 2 joints (min) | `malus_articulations` trop fort | Réduire le coefficient |
| Jamais de succès | `bonus_succès` trop faible | Augmenter ou réduire `REACH_DIST` |
| Convergence vers morphologie fixe | Population trop petite | Augmenter `pop_size` dans NEAT |

---

## 10. Configuration NEAT

Le fichier `neat_config.txt` contrôle tous les paramètres de l'évolution :

```ini
[NEAT]
fitness_criterion     = max
fitness_threshold     = 500.0    # stop si un individu atteint ce score
pop_size              = 100      # taille de la population
reset_on_extinction   = True

[DefaultGenome]
# Réseau
num_inputs            = 17       # 2*MAX_JOINTS + 5
num_outputs           = 6        # MAX_JOINTS
num_hidden            = 0        # NEAT part de 0 neurones cachés
feed_forward          = True

# Activation
activation_default    = tanh
activation_mutate_rate = 0.0
activation_options    = tanh

# Connexions
initial_connection    = full
conn_add_prob         = 0.5
conn_delete_prob      = 0.2
weight_mutate_rate    = 0.8
weight_mutate_power   = 0.5
weight_max_value      = 3.0
weight_min_value      = -3.0

# Nœuds
node_add_prob         = 0.2
node_delete_prob      = 0.1
bias_mutate_rate      = 0.7
bias_mutate_power     = 0.5

[DefaultSpeciesSet]
compatibility_threshold = 3.0   # seuil de distance génomique

[DefaultStagnation]
max_stagnation        = 20      # générations sans amélioration avant extinction
species_elitism       = 2       # nb d'espèces protégées

[DefaultReproduction]
elitism               = 2       # nb de meilleurs individus conservés par espèce
survival_threshold    = 0.2     # % de survivants par espèce
```

---

## 11. Visualisation

### Pendant l'entraînement

```
Generation 001 | best: 45.3 | mean: 12.1 | species: 8
Generation 002 | best: 67.2 | mean: 18.4 | species: 9
...
```

### Courbes de fitness

`utils/visualizer.py` génère des courbes matplotlib après chaque génération :
- Fitness du meilleur individu
- Fitness moyenne de la population
- Nombre d'espèces actives

### Rejouer un génome

```bash
python evaluate.py --checkpoint checkpoints/gen_0050.pkl
```

Charge le génome, construit le bras correspondant, lance un épisode avec rendu Pygame.

---

## 12. Installation

```bash
git clone https://github.com/r0n/EvoArm.git
cd EvoArm

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### `requirements.txt`

```
neat-python>=0.92
gymnasium>=0.29
numpy>=1.26
pygame>=2.5
matplotlib>=3.8
```

### Lancer l'entraînement

```bash
python train.py
```

---

## 13. Références

- **NEAT paper** : Stanley & Miikkulainen, [Evolving Neural Networks through Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) (2002)
- **neat-python docs** : https://neat-python.readthedocs.io/
- **Forward Kinematics** : https://en.wikipedia.org/wiki/Forward_kinematics
- **Morphological Evolution survey** : Gupta et al., [Embodied Intelligence via Learning and Evolution](https://arxiv.org/abs/2102.02202) (2021)

---

*EvoArm — évolution morphologique et comportementale d'un bras articulé 2D.*

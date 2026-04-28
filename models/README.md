# T6 — Le réseau ActorCritic

> Module : `models/policy_network.py`

---

## Table des matières

1. [Rôle dans le projet](#1-rôle-dans-le-projet)
2. [Pourquoi un réseau de neurones ?](#2-pourquoi-un-réseau-de-neurones-)
3. [L'architecture Acteur-Critique](#3-larchitecture-acteur-critique)
4. [Le backbone](#4-le-backbone)
5. [La tête Acteur](#5-la-tête-acteur)
6. [La tête Critique](#6-la-tête-critique)
7. [Pourquoi une distribution gaussienne ?](#7-pourquoi-une-distribution-gaussienne-)
8. [log_std — contrôler l'exploration](#8-log_std--contrôler-lexploration)
9. [Les méthodes](#9-les-méthodes)
10. [Shapes de référence](#10-shapes-de-référence)

---

## 1. Rôle dans le projet

Le réseau ActorCritic est **le cerveau du walker**. C'est lui qui, à chaque step de simulation, reçoit l'état du walker (positions, vitesses, contacts) et décide quels torques appliquer aux joints.

Il intervient à deux moments distincts dans la boucle d'entraînement :

```
┌─────────────────────────────────────────────────────┐
│  Phase 1 — Rollout (collecte)                       │
│                                                     │
│  obs → get_action() → action envoyée à Box2D        │
│                    → log_prob stocké dans buffer    │
│                    → value stocké dans buffer       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Phase 2 — Update (mise à jour PPO)                 │
│                                                     │
│  obs + actions du buffer → evaluate_actions()       │
│                         → log_prob recalculé        │
│                         → entropy calculée          │
│                         → value recalculée          │
└─────────────────────────────────────────────────────┘
```

---

## 2. Pourquoi un réseau de neurones ?

Le walker vit dans un espace d'états très grand — toutes les combinaisons possibles d'angles, vitesses et contacts. Il est impossible de définir manuellement une règle pour chaque situation.

Le réseau de neurones est un **approximateur universel de fonctions** : il apprend lui-même la fonction qui mappe un état vers une bonne action, uniquement à partir des récompenses reçues.

Un neurone seul c'est une transformation linéaire :

```
sortie = (x1·w1 + x2·w2 + ... + xn·wn) + biais
```

Empilé en couches avec des fonctions d'activation non-linéaires, le réseau peut approximer des comportements arbitrairement complexes — comme la coordination des jambes pour marcher.

---

## 3. L'architecture Acteur-Critique

Le réseau est divisé en deux rôles distincts :

```
Observation (14,)
      │
  ┌───┴───────────────────┐
  │    Backbone partagé   │
  │    Dense(64, tanh)    │
  │    Dense(64, tanh)    │
  └───┬───────────────────┘
      │
  ┌───┴──────────────────────────┐
  │                              │
Acteur                        Critique
Dense(4)                      Dense(1)
+ log_std                          │
  │                           V(s) = 47.3
Distribution N(mean, std)     "cette situation
  │                            vaut 47 points"
action, log_prob
"je pousse la hanche
 droite à 0.6"
```

**L'Acteur** décide quoi faire. Il produit une distribution sur les actions possibles.

**Le Critique** évalue si la situation est bonne ou mauvaise. Il ne décide rien — il sert de référence pour mesurer si une action était meilleure ou moins bonne qu'attendu.

Les deux partagent le backbone parce qu'ils ont besoin de la même compréhension de l'état. Seule l'interprétation finale diverge.

---

## 4. Le backbone

```python
self.backbone = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
])
```

Le backbone prend l'observation brute `(14,)` et la transforme en une représentation abstraite `(64,)`.

### Pourquoi deux couches ?

Une seule couche Dense, peu importe sa taille, ne peut représenter que des relations **linéaires** entre l'entrée et la sortie. La deuxième couche ne voit plus l'observation brute — elle voit les features extraites par la première, et peut les combiner entre elles.

Exemple concret : la première couche pourrait apprendre "la jambe gauche est en l'air" comme feature. La deuxième couche peut ensuite apprendre que cette feature combinée avec "le torse penche à droite" signifie "pousser avec la hanche droite".

### Pourquoi `tanh` ?

`tanh` est une fonction non-linéaire qui écrase toute valeur dans `[-1, 1]` :

```
 1 |        ████████
   |      ██
   |    ██
 0 |───██──────────────
   |  ██
   |██
-1 |
   └──────────────────
     -3   0   3
```

Sans activation, empiler des Dense revient mathématiquement à une seule transformation linéaire — inutile. `tanh` introduit la non-linéarité nécessaire et évite que les valeurs explosent entre les couches.

### Pourquoi 64 neurones ?

C'est empirique. Pour un espace d'observation de taille 14 avec 4 actions, c'est un bon compromis :
- Assez grand pour capturer des comportements complexes
- Assez petit pour converger rapidement

On ne touche à cette valeur qu'en dernier recours si l'entraînement stagne.

---

## 5. La tête Acteur

```python
self.actor_mean = tf.keras.layers.Dense(act_dim)
```

Prend les 64 features du backbone et sort 4 valeurs — une par joint (hanche G, genou G, hanche D, genou D). Ce sont les **moyennes** des distributions gaussiennes, une par joint.

Pas d'activation sur cette couche — le mean peut être n'importe quel réel. Box2D recevra l'action après sample et clip.

---

## 6. La tête Critique

```python
self.critic = tf.keras.layers.Dense(1)
```

Prend les 64 features du backbone et sort **un seul scalaire** : `V(s)`, la valeur estimée de l'état.

`V(s)` répond à : *"si je suis dans cet état, combien de récompense totale est-ce que j'espère accumuler jusqu'à la fin de l'épisode ?"*

Le critique ne décide rien. Il sert à mesurer si une action était bonne — "meilleure que ce que j'anticipais" ou "moins bonne". C'est ce signal qui permet à PPO d'apprendre efficacement sans attendre la fin de l'épisode.

---

## 7. Pourquoi une distribution gaussienne ?

Le réseau ne sort pas une action fixe. Il sort une **proposition avec une marge d'incertitude**.

Pourquoi ? Parce que sans hasard, le walker ferait exactement la même chose à chaque fois qu'il voit le même état. Il n'explorerait jamais de nouvelles stratégies et resterait bloqué dans le premier comportement qu'il découvre.

La gaussienne `N(mean, std)` permet de **sampler** une action légèrement différente à chaque fois :

```
mean = 0.6, std = 0.5

        ████          ← actions probables autour de 0.6
       ██████
      ████████
─────────────────────
  0.0  0.6  1.0

On peut tirer : 0.4, 0.7, 0.9, 0.5...
Rarement : -0.2 ou 1.3
```

Avec le temps, PPO ajuste `mean` vers les bonnes actions et réduit `std` — le walker devient progressivement plus déterministe et précis.

---

## 8. `log_std` — contrôler l'exploration

```python
self.log_std = tf.Variable(tf.zeros(act_dim), trainable=True)
```

`log_std` est un paramètre appris qui contrôle la largeur de la gaussienne — donc le niveau d'exploration.

### Pourquoi `log_std` et pas `std` directement ?

La std doit toujours être **positive**. Si on apprenait `std` directement, rien n'empêche le gradient de la rendre négative — ce qui n'a pas de sens pour un écart-type.

En apprenant `log_std` qui peut être n'importe quel réel, et en calculant `std = exp(log_std)`, on garantit que std est toujours positive quelle que soit la valeur de `log_std`.

```
log_std =  0  →  std = exp(0)  = 1.0   (début entraînement, exploration neutre)
log_std =  1  →  std = exp(1)  = 2.7   (beaucoup d'exploration)
log_std = -2  →  std = exp(-2) = 0.13  (peu d'exploration, réseau confiant)
```

### `trainable=True`

C'est une variable ordinaire, pas une couche. Sans `trainable=True`, TF ne saurait pas qu'il doit l'optimiser pendant l'entraînement — la std resterait à 1.0 pour toujours.

### Évolution pendant l'entraînement

```
Début    std ≈ 1.0   Walker gesticule dans tous les sens, explore
  │
  │      std ≈ 0.5   Mouvements plus cohérents, exploration réduite
  │
  ▼
Fin      std ≈ 0.1   Walker marche de façon quasi-déterministe
```

---

## 9. Les méthodes

### `call(obs)`

Forward pass de base. Utilisé en interne par `get_action()` et `evaluate_actions()`.

```python
def call(self, obs):
    x     = self.backbone(obs)   # (batch, 14) → (batch, 64)
    mean  = self.actor_mean(x)   # (batch, 64) → (batch, 4)
    value = self.critic(x)       # (batch, 64) → (batch, 1)
    return mean, value
```

---

### `get_action(obs)`

Appelée **pendant le rollout**. Produit une action à envoyer à Box2D et les métadonnées à stocker dans le buffer.

```python
def get_action(self, obs):
    mean, value = self.call(obs)

    std      = tf.exp(self.log_std)                        # log_std → std
    dist     = tfp.distributions.Normal(mean, std)         # gaussienne N(mean, std)
    action   = dist.sample()                               # on tire une action
    action   = tf.clip_by_value(action, -1.0, 1.0)        # contrainte [-1, 1] pour Box2D
    log_prob = tf.reduce_sum(dist.log_prob(action), axis=-1)  # probabilité de cette action
    value    = tf.squeeze(value, axis=-1)                  # (batch, 1) → (batch,)

    return action, log_prob, value
```

**`log_prob`** — répond à "à quel point cette action était-elle probable ?". On l'additionne sur les 4 joints (`reduce_sum`) pour avoir un scalaire par sample. Sera utilisé plus tard par PPO.

**`clip_by_value`** — Box2D attend des torques dans `[-1, 1]`. La gaussienne peut théoriquement sortir n'importe quelle valeur, on force les bornes.

---

### `evaluate_actions(obs, actions)`

Appelée **pendant la mise à jour PPO**. On reprend des observations et actions déjà collectées, et on recalcule leurs probabilités avec les poids actuels du réseau.

```python
def evaluate_actions(self, obs, actions):
    mean, value = self.call(obs)

    std      = tf.exp(self.log_std)
    dist     = tfp.distributions.Normal(mean, std)
    log_prob = tf.reduce_sum(dist.log_prob(actions), axis=-1)  # on ne sample pas
    entropy  = tf.reduce_sum(dist.entropy(), axis=-1)          # mesure d'exploration
    value    = tf.squeeze(value, axis=-1)

    return log_prob, value, entropy
```

Différence clé avec `get_action()` : on ne sample pas. On passe les `actions` déjà collectées et on recalcule leur log_prob avec le réseau **tel qu'il est maintenant**. PPO compare ce log_prob avec celui stocké dans le buffer pour mesurer combien la policy a changé.

**`entropy`** — mesure à quel point la distribution est étalée. PPO ajoute un bonus d'entropie dans sa loss pour empêcher la std de s'effondrer trop vite et éviter la convergence prématurée vers un mauvais comportement.

---

## 10. Shapes de référence

Pour `batch_size = 4`, `obs_dim = 14`, `act_dim = 4` :

| Tenseur | Shape | Description |
|---|---|---|
| `obs` | `(4, 14)` | Batch d'observations |
| `x` (backbone) | `(4, 64)` | Représentation intermédiaire |
| `mean` | `(4, 4)` | Mean gaussienne par joint |
| `value` (avant squeeze) | `(4, 1)` | Valeur critique |
| `value` (après squeeze) | `(4,)` | Valeur critique aplatie |
| `std` | `(4,)` | Broadcast depuis `log_std (4,)` |
| `action` | `(4, 4)` | Action clippée à [-1, 1] |
| `log_prob` | `(4,)` | Log-prob sommée sur les joints |
| `entropy` | `(4,)` | Entropie sommée sur les joints |
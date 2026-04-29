# T7 & T9 — RolloutBuffer et PPO

> Modules : `training/rollout_buffer.py`, `training/ppo.py`

---

## Table des matières

1. [Vue d'ensemble — le cycle d'apprentissage](#1-vue-densemble--le-cycle-dapprentissage)
2. [RolloutBuffer — le carnet de notes](#2-rolloutbuffer--le-carnet-de-notes)
3. [Ce qu'on stocke et pourquoi](#3-ce-quon-stocke-et-pourquoi)
4. [GAE — évaluer les actions passées](#4-gae--évaluer-les-actions-passées)
5. [gamma et lambda](#5-gamma-et-lambda)
6. [get_batches — découper pour apprendre](#6-get_batches--découper-pour-apprendre)
7. [PPO — l'algorithme d'apprentissage](#7-ppo--lalgorithme-dapprentissage)
8. [Les hyperparamètres](#8-les-hyperparamètres)
9. [Les trois losses](#9-les-trois-losses)
10. [GradientTape — comment TF apprend](#10-gradienttape--comment-tf-apprend)
11. [La boucle update()](#11-la-boucle-update)
12. [PPO est on-policy](#12-ppo-est-on-policy)
13. [Flux complet](#13-flux-complet)

---

## 1. Vue d'ensemble — le cycle d'apprentissage

Le cycle d'entraînement PPO se répète indéfiniment jusqu'à convergence :

```
┌─────────────────────────────────────────────────────────────┐
│  1. ROLLOUT — le walker joue 2048 steps                     │
│     à chaque step : buf.add(obs, action, reward, ...)       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  2. GAE — on évalue chaque action                           │
│     buf.compute_returns_and_advantages(last_value)          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  3. UPDATE — PPO ajuste les poids                           │
│     for epoch in n_epochs:                                  │
│         for batch in buf.get_batches():                     │
│             ppo.update(batch)                               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  4. RESET — on vide le buffer                               │
│     buf.reset()                                             │
│     → retour à l'étape 1 avec la nouvelle policy           │
└─────────────────────────────────────────────────────────────┘
```

Le RolloutBuffer gère les étapes 1, 2 et 4. PPO gère l'étape 3.

---

## 2. RolloutBuffer — le carnet de notes

Le buffer est le **pont entre observer et apprendre**. Pendant que le walker joue, on enregistre tout ce qui se passe. Une fois le rollout terminé, PPO pioche dans ces données pour ajuster les poids.

Sans buffer, PPO ne saurait pas quoi corriger — il n'aurait aucune trace de ce que le walker a fait ni si c'était bien ou mal.

```python
class RolloutBuffer:
    def __init__(self, n_steps, obs_dim, act_dim, gamma=0.99, lam=0.95):
        self.n_steps = n_steps   # taille du rollout (ex: 2048)
        self.gamma   = gamma     # facteur de discount
        self.lam     = lam       # GAE lambda
        self.reset()
```

`gamma` et `lam` sont stockés dans le buffer parce que c'est lui qui calcule GAE — il a besoin de ces paramètres pour évaluer les actions collectées.

---

## 3. Ce qu'on stocke et pourquoi

À chaque step du rollout, on appelle `buf.add()` avec six informations :

| Donnée | Source | Utilisé pour |
|---|---|---|
| `obs` | `env.step()` | Recalculer log_prob dans PPO |
| `action` | `model.get_action()` | Recalculer log_prob dans PPO |
| `reward` | `env.step()` | Calculer les returns |
| `done` | `env.step()` | Couper les épisodes dans GAE |
| `log_prob` | `model.get_action()` | Calculer le ratio π_new / π_old |
| `value` | `model.get_action()` | Calculer les avantages GAE |

```python
def add(self, obs, action, reward, done, log_prob, value):
    self.obs.append(obs)
    self.actions.append(action)
    self.rewards.append(reward)
    self.dones.append(done)
    self.log_probs.append(log_prob)
    self.values.append(value)
```

---

## 4. GAE — évaluer les actions passées

Une fois le rollout terminé, on ne sait pas encore si chaque action était bonne ou mauvaise. GAE (Generalised Advantage Estimation) répond à cette question en regardant ce qui s'est passé **après** chaque action.

### Le problème

Regarder uniquement la récompense immédiate d'une action est insuffisant. Marcher d'un pas peut donner une récompense de +0.1 — mais si ce pas mène à une chute 3 steps plus tard, c'était en réalité une mauvaise décision.

GAE agrège les récompenses futures pour évaluer chaque action dans son contexte.

### TD error — "était-ce mieux que prévu ?"

Pour chaque step, on calcule d'abord l'erreur de prédiction du critique :

```
δ_t = r_t + γ · V(s_{t+1}) · (1 - done) - V(s_t)

δ_t > 0  →  mieux que prévu   → avantage positif → PPO favorise cette action
δ_t < 0  →  moins bien        → avantage négatif → PPO défavorise cette action
δ_t = 0  →  exactement prévu  → pas de signal
```

### L'avantage GAE

On ne s'arrête pas à un seul step — on agrège les TD errors suivantes avec un poids décroissant :

```
A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
```

Les steps proches comptent beaucoup, les steps lointains comptent peu.

### Pourquoi calculer à l'envers ?

GAE est récursif — l'avantage au step `t` dépend de celui au step `t+1`. On part donc de la fin et on remonte :

```python
gae = 0
for t in reversed(range(len(self.rewards))):
    delta = rewards[t] + gamma * next_value * (1 - done) - values[t]
    gae   = delta + gamma * lam * (1 - done) * gae
    advantages.insert(0, gae)
```

### `(1 - done)` — couper les épisodes

Si le walker tombe au step `t` (`done = True`), l'épisode suivant est indépendant. `(1 - done)` remet `gae` à zéro pour que l'avantage du nouvel épisode ne contamine pas le calcul de l'ancien.

```
done = False  →  (1 - done) = 1  →  on propage normalement
done = True   →  (1 - done) = 0  →  on coupe, pas de propagation
```

### `last_value` — le bootstrap

Le rollout s'arrête après 2048 steps, pas forcément parce que le walker est tombé. Si il est encore en vie, il aurait pu continuer à accumuler des récompenses. `last_value` est la valeur estimée du dernier état — elle dit à GAE "après le rollout, la situation valait encore X points".

```python
buf.compute_returns_and_advantages(last_value=float(last_value))
```

### Returns

Les returns sont la cible d'entraînement pour le critique :

```
returns = advantages + values
        = (ce qui s'est réellement passé - prédiction) + prédiction
        = ce qui s'est réellement passé
```

PPO dira au critique : "tu avais estimé 47, en réalité c'était 52, corrige-toi."

### Normalisation des avantages

Avant de les utiliser dans PPO, on normalise les avantages :

```python
advantages = (advantages - mean) / (std + 1e-8)
```

Ça centre les avantages autour de 0 avec une std de 1. Sans ça, si tous les avantages sont très grands (ex: entre 50 et 200), les gradients seraient énormes et l'entraînement instable.

---

## 5. gamma et lambda

### `gamma` — facteur de discount

Contrôle à quel point le futur compte. Une récompense lointaine vaut moins qu'une récompense immédiate.

```
gamma = 0.99

récompense dans  1 step   →  0.99¹  × r = 0.99r
récompense dans 10 steps  →  0.99¹⁰ × r = 0.90r
récompense dans 50 steps  →  0.99⁵⁰ × r = 0.60r
récompense dans 200 steps →  0.99²⁰⁰× r = 0.13r
```

Proche de 1 → vision long terme. Proche de 0 → myope, cherche la récompense immédiate.

### `lambda` — tradeoff biais/variance

Contrôle sur combien de steps futurs on se base pour estimer l'avantage.

```
lambda = 0.0  →  un seul step  →  stable mais imprécis  (biais fort)
lambda = 1.0  →  tous les steps →  précis mais bruité   (variance forte)
lambda = 0.95 →  compromis standard PPO
```

`gamma * lambda = 0.99 × 0.95 = 0.94` — ce coefficient fait décroître exponentiellement l'importance des steps futurs.

---

## 6. get_batches — découper pour apprendre

PPO ne passe pas tout le buffer d'un coup — il le découpe en mini-batches.

```python
def get_batches(self, batch_size):
    indices = np.random.permutation(n)   # mélange aléatoire
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield obs[batch_idx], actions[batch_idx], ...
```

### Pourquoi mélanger ?

Sans mélange, les batches seraient temporellement corrélés (steps 0-63, puis 64-127...). PPO apprendrait des patterns liés à l'ordre des steps plutôt qu'aux actions elles-mêmes. Le mélange casse cette corrélation.

### Pourquoi `yield` ?

C'est un générateur Python — les batches sont produits un par un à la demande plutôt que tous en mémoire. Bonne habitude pour les grands buffers.

---

## 7. PPO — l'algorithme d'apprentissage

PPO (Proximal Policy Optimization) ajuste les poids du réseau pour que le walker fasse de meilleures actions. Son principe central : **ne pas trop changer la policy en un seul update**.

Sans cette contrainte, un gradient step trop agressif pourrait détruire un comportement appris en plusieurs milliers de steps.

```python
class PPO:
    def __init__(self, model, lr, clip_eps, c1, c2, n_epochs, batch_size):
        self.model     = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        ...
```

---

## 8. Les hyperparamètres

| Paramètre | Valeur | Rôle |
|---|---|---|
| `lr` | 3e-4 | Taille du pas de gradient |
| `clip_eps` | 0.2 | Seuil de clipping du ratio (ratio autorisé entre 0.8 et 1.2) |
| `c1` | 0.5 | Poids de la loss critique dans la loss totale |
| `c2` | 0.01 | Poids du bonus d'entropie |
| `n_epochs` | 10 | Passes sur le buffer par update |
| `batch_size` | 64 | Taille des mini-batches |

### `n_epochs` — pourquoi repasser plusieurs fois ?

Collecter 2048 steps prend du temps. En repassant 10 fois sur le même buffer, on extrait plus d'information avant de le jeter. C'est pour ça que le clip existe — sans lui, 10 passes feraient trop dériver la policy.

---

## 9. Les trois losses

### Loss acteur — la loss PPO clippée

```
ratio     = exp(log_prob_new - log_prob_old)   # π_new / π_old

unclipped = ratio × advantage
clipped   = clip(ratio, 1-ε, 1+ε) × advantage

loss_actor = -mean( min(unclipped, clipped) )
```

**Le ratio** mesure combien la policy a changé pour cette action :
```
ratio = 1.0  →  policy inchangée
ratio = 1.5  →  π_new trouve cette action 50% plus probable
ratio = 0.5  →  π_new trouve cette action deux fois moins probable
```

**`min(unclipped, clipped)`** choisit toujours la version pessimiste :
- Si l'avantage est positif (bonne action) et ratio > 1+ε → on plafonne, on ne pousse plus
- Si l'avantage est négatif (mauvaise action) et ratio < 1-ε → on plafonne, on ne tire plus

**Le `-`** devant `mean` : on minimise la loss mais on veut maximiser la récompense — le signe inverse.

### Loss critique — MSE

```
loss_critic = mean( (V(s) - returns)² )
```

MSE classique entre la valeur prédite et le return réel. Le critique apprend à mieux estimer `V(s)` pour que GAE soit plus précis au prochain rollout.

### Loss entropie

```
loss_entropy = -mean(entropy)
```

L'entropie mesure à quel point la distribution est étalée. Plus la std est grande, plus l'entropie est haute. Le `-` force PPO à **maximiser** l'entropie — ce qui maintient l'exploration et évite que la std s'effondre trop vite.

### Loss totale

```
loss_total = loss_actor + c1 × loss_critic + c2 × loss_entropy
```

Une seule loss à minimiser qui combine les trois objectifs.

---

## 10. GradientTape — comment TF apprend

Pour ajuster un poids `w`, TF a besoin de savoir dans quelle direction le modifier :

```
nouveau_w = w - lr × (dLoss/dw)
```

`dLoss/dw` c'est le gradient — "si j'augmente ce poids de 0.001, est-ce que la loss monte ou descend ?". TF doit remonter toute la chaîne d'opérations pour calculer ça (rétropropagation).

`GradientTape` dit à TF d'enregistrer toutes les opérations dans son bloc :

```python
with tf.GradientTape() as tape:
    loss_total, ... = self._compute_losses(...)   # TF prend des notes

grads = tape.gradient(loss_total, self.model.trainable_variables)
# TF relit ses notes à l'envers → calcule dLoss/dw pour chaque poids
```

### Gradient clipping

```python
grads, _ = tf.clip_by_global_norm(grads, 0.5)
```

Si un gradient est anormalement grand (spike), il ferait un pas trop grand et déstabiliserait l'entraînement. On plafonne la norme globale des gradients à 0.5.

---

## 11. La boucle update()

```python
def update(self, buffer):
    for epoch in range(self.n_epochs):          # 10 passes sur le buffer
        for batch in buffer.get_batches(...):   # 32 batches de 64 steps
            with tf.GradientTape() as tape:
                loss = self._compute_losses(...)

            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 0.5)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return metrics   # moyennes des losses sur tous les batches
```

Pour un rollout de 2048 steps avec `n_epochs=10` et `batch_size=64` :
```
2048 / 64 = 32 batches par epoch
32 × 10   = 320 gradient steps par update
```

---

## 12. PPO est on-policy

PPO est un algorithme **on-policy** — les données du buffer doivent venir de la policy actuelle. Une fois la mise à jour faite, le buffer est jeté et on collecte un nouveau rollout avec la nouvelle policy.

On ne peut pas réutiliser d'anciens rollouts (contrairement à SAC ou DQN qui gardent des millions de transitions).

La seule exception : les `n_epochs` passes sur le même buffer. C'est toléré parce que le clip empêche la policy de trop s'éloigner des données collectées.

---

## 13. Flux complet

```
env.reset() → obs_0
     │
     ▼
┌─────────────────────────────────────────┐
│  ROLLOUT (2048 steps)                   │
│                                         │
│  obs_t → model.get_action()             │
│       → action, log_prob, value         │
│                                         │
│  env.step(action)                       │
│       → obs_{t+1}, reward, done         │
│                                         │
│  buf.add(obs, action, reward,           │
│          done, log_prob, value)         │
└─────────────────────────────────────────┘
     │
     ▼
buf.compute_returns_and_advantages(last_value)
     │  → advantages (GAE)
     │  → returns
     ▼
┌─────────────────────────────────────────┐
│  UPDATE PPO (10 epochs × 32 batches)    │
│                                         │
│  model.evaluate_actions(obs, actions)   │
│       → log_prob_new, value, entropy    │
│                                         │
│  ratio = exp(log_prob_new - log_prob_old│
│  loss  = clip + critic + entropy        │
│  gradient step → nouveaux poids         │
└─────────────────────────────────────────┘
     │
     ▼
buf.reset() → retour au rollout avec π_new
```

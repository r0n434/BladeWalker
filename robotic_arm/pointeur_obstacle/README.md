# RL Robotic Arm Environment

Ce projet contient un environnement personnalisé d'Apprentissage par Renforcement (Reinforcement Learning) créé avec **Gymnasium**. Il simule un bras robotique planaire (2D) qui doit apprendre à atteindre une cible générée aléatoirement.

## Prérequis

Pour faire tourner l'environnement et l'entraîner, vous aurez besoin des librairies suivantes :

- `numpy` (Calculs mathématiques)
- `gymnasium` (Framework RL)
- `pygame` (Rendu visuel)
- `stable-baselines3` (Algorithmes d'entraînement PPO / SAC)

## Fonctionnement de l'Environnement

L'environnement suit le standard Gymnasium (`Env`) et interagit avec l'Agent via les espaces suivants :

- **Observation Space (Ce que l'IA voit) :** Un vecteur continu (`spaces.Box`) contenant :
  1. Les angles actuels de chaque articulation (en radians).
  2. Les coordonnées (X, Y) de la cible à atteindre.
  3. Les coordonnées (X, Y) du bout du bras (effecteur).
- **Action Space (Ce que l'IA fait) :** Un vecteur continu (`spaces.Box`) définissant le changement d'angle à appliquer à chaque moteur à chaque étape.
- **Moteur Physique :** La position du bout du bras est calculée de manière déterministe grâce à la **Cinématique Directe (Forward Kinematics)**.

## Système de Récompense (Reward)

L'Agent est incité à atteindre la cible de manière fluide et économe en énergie. La récompense à chaque étape (step) est calculée ainsi :

1. **Distance :** Récompense négative proportionnelle à la distance séparant le bout du bras de la cible.
2. **Pénalité d'énergie :** On retire des points si les actions demandées aux moteurs sont trop intenses.
3. **Pénalité de Jitter :** On retire des points si l'Agent fait des mouvements saccadés (changements brusques entre l'action N et l'action N-1).

## Utilisation

Pour tester l'environnement avec un agent aux actions aléatoires :

```bash
python robot_arm_env.py
python train.py --algo sac --timesteps 200000
```

### 2. Le Contexte et les Traces de Modifications

Si tu veux garder une trace de *pourquoi* et *comment* le code a évolué entre ta première version basique et ta version avancée, voici un résumé parfait à ajouter sous le titre "Historique des versions" par exemple.

#### Historique des Modifications : De la V1 (Basique) à la V2 (Réaliste)

Nous y avons apporté des modifications structurelles pour le rendre plus réaliste physiquement et pour forcer l'Agent à accomplir une tâche plus difficile.

**1. Ajout de l'Inertie Physique (Lissage des actions) :**

- **Contexte :** Dans la V1, les moteurs pouvaient changer de direction instantanément.

- **Modification :** Nous avons implémenté une formule de lissage où l'action réelle est composée à 20% de la nouvelle commande et à 80% de l'action précédente (`0.2 * action + 0.8 * self.previous_action`). Le bras glisse désormais avec un effet de poids.
- **Ajustement :** Pour compenser cette lourdeur artificielle, la limite d'action (`max_action`) a été augmentée de `0.1` à `0.3` pour permettre de plus fortes impulsions.

**2. Évolution de l'Objectif : "Hit and Run" vers "Hold and Stabilize" :**

- **Contexte :** Dans la V1, l'épisode se terminait dès que le bras frôlait la cible (distance < 0.1), accordant un gros bonus unique (+10.0). L'IA apprenait juste à se jeter sur le point.
- **Modification :** La condition de fin prématurée (`terminated = True`) a été retirée. Le bonus a été réduit à `+1.0` mais il est distribué **à chaque frame** tant que le bras reste sur la cible. L'IA doit maintenant apprendre l'art difficile de l'arrêt et de la stabilisation en un point précis.

**3. Rééquilibrage des Pénalités :**

- **Contexte :** La V1 punissait sévèrement les actions intenses (coefficient 1.0) et les tremblements (coefficient 2.0).
- **Modification :** L'ajout de l'inertie lissant naturellement les mouvements, l'environnement se charge lui-même de stabiliser les à-coups. Les coefficients de pénalité ont donc été assouplis (0.5 pour l'énergie, 1.0 pour les tremblements) pour ne pas brider excessivement l'exploration de l'IA.

**4. Intégration du Curriculum Learning :**

- **Contexte :** L'entraînement standard de PPO/SAC montrait des difficultés à faire converger l'Agent rapidement lorsque la cible apparaissait dans des zones "difficiles" (nécessitant des contorsions) dès le début.

- **Modification :** Ajout d'une variable `self.difficulty` (de 0.1 à 1.0) dans l'environnement modifiant la logique de spawn de la cible dans la méthode `reset()`. Implémentation d'une classe `CurriculumCallback` côté script d'entraînement pour faire augmenter cette difficulté linéairement jusqu'à 80% des *timesteps* totaux.
- **Résultat :** L'Agent apprend les corrélations motrices de base beaucoup plus vite en début d'entraînement (récompenses positives fréquentes) et devient bien plus robuste sur les cibles éloignées en fin d'entraînement.

## Curriculum Learning (Apprentissage Progressif)

Pour accélérer et stabiliser l'entraînement de l'Agent, ce projet intègre un système de **Curriculum Learning**.

Au lieu de confronter l'IA à la difficulté maximale dès le premier épisode, l'environnement adapte sa complexité en temps réel :

1. **Phase Débutant (Difficulté 0.1) :** Au début de l'entraînement, la cible générée apparaît systématiquement très près de l'effecteur du bras. L'Agent apprend rapidement les bases de la motricité (comment bouger ses moteurs pour faire baisser la distance).
2. **Phase Intermédiaire :** Au fur et à mesure que les *timesteps* s'écoulent, le rayon d'apparition de la cible s'agrandit progressivement. L'Agent doit commencer à plier ses articulations et gérer des trajectoires plus complexes.
3. **Phase Experte (Difficulté 1.0) :** Vers 80% du temps d'entraînement total, la cible peut apparaître n'importe où dans la zone atteignable. L'Agent, ayant déjà maîtrisé les mouvements de base, affine sa précision sur les cas extrêmes (ex: atteindre un point situé derrière sa base).

Cette mécanique est gérée dynamiquement par un **Callback** (Stable Baselines3) qui communique la progression de l'entraînement à l'environnement à chaque étape.

## ROADMAP

### Des applications amusantes

- Le Bras Dessinateur (Waypoints) :
    L'idée : Au lieu de générer une seule cible aléatoire, donne à ton bras une liste de points à suivre dans l'ordre (par exemple, les sommets d'un carré, ou les points du contour d'une lettre).

- Suivi de cible mobile (Tracking) :

    L'idée : La cible n'est plus statique. À chaque step, la cible se déplace légèrement (comme une mouche qui vole).

- Pick and Place (Attraper et relâcher) :

    L'idée : Ajouter une pince (un "gripper") au bout du bras. L'IA doit aller sur un objet, fermer la pince, déplacer l'objet vers une zone de dépôt, et ouvrir la pince.

    *Comment faire :* Ajouter une dimension à l'action_space (un 3ème "moteur" qui contrôle l'ouverture de la pince entre 0 et 1) et modifier la récompense

### Techniques d'entraînement avancées

- Domain Randomization (Robustesse) :

  À chaque reset, modifie légèrement la longueur des segments du bras (ex: entre 0.9 et 1.1). L'IA ne pourra plus apprendre par cœur la taille de son corps et devra développer une stratégie générale et adaptable.

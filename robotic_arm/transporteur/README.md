# 🦾 RL Robotic Arm - Pick & Place Environment

Cet environnement simule un bras robotique articulé planaire (2D) équipé d'une pince. L'objectif de l'Agent d'Apprentissage par Renforcement est complexe : il doit approcher un objet cible, l'attraper, et le transporter jusqu'à une zone de dépôt, le tout en minimisant ses dépenses énergétiques.

## 🧠 Les Espaces d'Interaction

L'environnement interagit avec l'Agent via des espaces continus optimisés pour accélérer la convergence des réseaux de neurones.

### 👁️ Observation Space (État)

L'Agent reçoit un vecteur d'observation relatif à chaque étape. Au lieu de positions absolues, il perçoit des vecteurs directionnels :

1. Les angles actuels des articulations.
2. Le vecteur `[X, Y]` pointant de l'effecteur (bout du bras) vers l'objet à saisir.
3. Le vecteur `[X, Y]` pointant de l'objet vers la zone de dépôt.
4. Un booléen binaire (0 ou 1) indiquant si l'objet est actuellement tenu par la pince.
5. L'action brute demandée à l'étape précédente (pour la conscience du mouvement).

### 🎮 Action Space (Contrôles)

Un vecteur continu `[-0.3, 0.3]` comprenant :

* Les changements d'angles pour chaque articulation (Moteurs).
* Une valeur continue contrôlant l'état de la pince (Lock/Unlock).

## 🎯 Fonction de Récompense (Dense & Sparse)

La tâche "Pick & Place" étant complexe, la récompense guide l'Agent par étapes :

1. **Phase d'approche :** Récompense calculée sur le *Delta* de distance (l'Agent gagne des points s'il réduit la distance entre la pince et l'objet au step N par rapport au step N-1).
2. **Phase de transport :** Une fois l'objet saisi, la boussole change. La récompense suit le Delta de distance entre l'objet et la zone de dépôt.
3. **Pénalités :** L'énergie consommée (mouvements amples) et le Jitter (mouvements saccadés) sont pénalisés à chaque step.
4. **Victoire :** Un bonus majeur (`+50`) est accordé et l'épisode se termine si l'objet est déposé dans la zone cible.

## 🎓 Curriculum Learning

L'environnement intègre un mode de difficulté dynamique (`self.difficulty`). En début d'entraînement, l'objet et la zone de dépôt apparaissent très proches du bras. La zone d'apparition s'élargit progressivement à mesure que l'Agent gagne en compétence.

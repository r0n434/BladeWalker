// Vous pouvez ajouter ce code à particles.js ou dans un script dédié sur levels.html

const robotCharacter = document.getElementById('robot-character');
const levelNodes = document.querySelectorAll('.level-node');
const infoPanelTitle = document.getElementById('selected-title');
const infoPanelDesc = document.getElementById('selected-desc');
const mapContainer = document.querySelector('.world-map');

// 1. Positionner le robot sur le niveau actif initial (ex: Niveau 2)
function initializeRobotPosition() {
    const activeNode = document.querySelector('.level-node.active');
    if (activeNode) {
        moveRobotToNode(activeNode, false); // false = pas d'animation à l'init
    }
}

// 2. Fonction de déplacement du robot
function moveRobotToNode(node, animate = true) {
    // Désactiver l'animation si spécifié
    if (!animate) {
        robotCharacter.style.transition = 'none';
    } else {
        robotCharacter.style.transition = 'left 0.5s ease-out, top 0.5s ease-out, transform 0.3s';
    }

    // Calculer la position relative du nœud par rapport à la carte
    const mapRect = mapContainer.getBoundingClientRect();
    const nodeRect = node.getBoundingClientRect();
    
    const targetX = nodeRect.left - mapRect.left + (nodeRect.width / 2);
    const targetY = nodeRect.top - mapRect.top + (nodeRect.height / 2);

    // Appliquer les nouvelles coordonnées
    robotCharacter.style.left = `${targetX}px`;
    robotCharacter.style.top = `${targetY}px`;
}

// 3. Gestionnaire de clic sur les niveaux
levelNodes.forEach(node => {
    node.addEventListener('click', function(e) {
        const level = this.getAttribute('data-title');
        
        // Empêcher l'action si le niveau est verrouillé
        if (this.classList.contains('locked')) {
            e.preventDefault();
            alert(`${level} est actuellement verrouillé.`);
            return;
        }

        e.preventDefault(); // Empêcher la redirection immédiate
        const targetUrl = this.href;

        // Déplacer le robot avec animation
        moveRobotToNode(this, true);
        robotCharacter.classList.add('moving'); // Optionnel: classe pour animation supplémentaire (ex: tremblement)

        // Mettre à jour le panneau d'information
        infoPanelTitle.innerText = this.getAttribute('data-title');
        infoPanelDesc.innerText = this.getAttribute('data-desc');

        // Attendre la fin de l'animation avant la redirection
        setTimeout(() => {
            robotCharacter.classList.remove('moving');
            window.location.href = targetUrl; // Redirection vers le dashboard correspondant
        }, 550); // Légèrement supérieur à la durée de transition CSS (0.5s)
    });
});

// Initialiser après le chargement de la page
window.addEventListener('load', initializeRobotPosition);
window.addEventListener('resize', initializeRobotPosition); // Recalculer si la fenêtre change de taille
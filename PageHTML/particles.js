const canvas = document.getElementById('canvas-bg');
const ctx = canvas.getContext('2d');
let particles = [];
const mouse = { x: null, y: null, radius: 150 };

// 1. Redimensionnement
function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);
window.addEventListener('mousemove', (e) => { mouse.x = e.x; mouse.y = e.y; });
resize();

// 2. Création des particules
class Particle {
    constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = 2;
        this.vx = (Math.random() - 0.5) * 2;
        this.vy = (Math.random() - 0.5) * 2;
    }
    update() {
        this.x += this.vx;
        this.y += this.vy;
        if (this.x > canvas.width || this.x < 0) this.vx *= -1;
        if (this.y > canvas.height || this.y < 0) this.vy *= -1;
    }
    draw() {
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
    }
}

// 3. Animation et lignes
function init() {
    particles = [];
    for(let i=0; i<100; i++) particles.push(new Particle());
}

function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < particles.length; i++) {
        particles[i].update();
        particles[i].draw();
        
        // Dessiner les lignes (constellation)
        for (let j = i; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const distance = Math.sqrt(dx*dx + dy*dy);
            
            if (distance < 150) {
                ctx.strokeStyle = `rgba(255, 255, 255, ${1 - distance/150})`;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(particles[i].x, particles[i].y);
                ctx.lineTo(particles[j].x, particles[j].y);
                ctx.stroke();
            }
        }
    }
    requestAnimationFrame(animate);
}

// 4. Changement de couleur au scroll
window.addEventListener('scroll', () => {
    if (window.scrollY > window.innerHeight / 2) {
        document.body.classList.add('red-mode');
    } else {
        document.body.classList.remove('red-mode');
    }
});

init();
animate();
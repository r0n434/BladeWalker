from walker_env import WalkerEnv
from renderer import Box2DRenderer

def main():
    print("Initialisation de l'environnement...")
    # On instancie l'environnement (sans se soucier du render_mode pour Pygame)
    env = WalkerEnv()
    obs, info = env.reset()
    
    # On instancie notre moteur de rendu séparé
    renderer = Box2DRenderer(window_size=(1000, 600), ppm=60.0)
    
    running = True
    episodes = 0
    
    while running:
        # Prendre une action aléatoire (à remplacer par ton modèle plus tard)
        action = env.action_space.sample()
        
        # Avancer l'environnement
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Déléguer l'affichage au moteur de rendu en lui passant l'environnement
        # Si render() retourne False, c'est que l'utilisateur a cliqué sur la croix [X]
        running = renderer.render(env)
        
        if terminated or truncated:
            print(f"Épisode {episodes} terminé. Reset.")
            obs, info = env.reset()
            episodes += 1

    # Nettoyage
    renderer.close()
    env.close()
    print("Fermeture.")

if __name__ == "__main__":
    main()
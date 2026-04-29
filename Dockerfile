# On part d'une version stable de Python (3.11 est souvent plus compatible avec TensorFlow/Gym que 3.13)
FROM python:3.11-slim

# On installe les dépendances système (celles qui t'ont posé problème tout à l'heure !)
RUN apt-get update && apt-get install -y \
    swig \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    libfreetype6-dev libportmidi-dev libjpeg-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# On définit le dossier de travail dans le conteneur
WORKDIR /BladeWalker

# On copie le fichier des dépendances Python
COPY requirements.txt .

# On installe les bibliothèques Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Commande par défaut pour garder le conteneur en vie
CMD ["tail", "-f", "/dev/null"]
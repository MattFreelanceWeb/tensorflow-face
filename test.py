import cv2
import dlib
import numpy as np

# Charger le modèle de détection de visage pré-entraîné
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Charger le modèle de prédiction de points de repère du visage
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialiser la webcam
cap = cv2.VideoCapture(0)

# Charger l'image que vous souhaitez afficher
overlay_image = cv2.imread('smiley.png')  # Remplacez 'votre_image.png' par le chemin de votre image

while True:
    # Capture d'une image depuis la webcam
    ret, frame = cap.read()

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Obtenez les points de repère du visage
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = predictor(gray, rect)

        # Convertissez les points de repère en un tableau NumPy
        landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Calculer la largeur et la hauteur de l'image superposée pour couvrir entièrement la tête
        overlay_width = int(1.5 * w)  # Ajustez ce facteur selon vos besoins
        overlay_height = int(1.5 * h)  # Ajustez ce facteur selon vos besoins

        # Redimensionner l'image superposée en fonction des nouvelles dimensions
        overlay_image_resized = cv2.resize(overlay_image, (overlay_width, overlay_height))

        # Définir les coordonnées pour superposer l'image sur la tête
        overlay_x = int(x + w/2 - overlay_width/2)
        overlay_y = int(y + h/2 - overlay_height/2)

        # S'assurer que les coordonnées sont dans les limites de l'image
        overlay_x = max(overlay_x, 0)
        overlay_y = max(overlay_y, 0)

        # Superposer l'image sur le visage
        frame[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width] = overlay_image_resized

    # Afficher le résultat
    cv2.imshow('Facial Landmarks', frame)

    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la webcam et fermer la fenêtre
cap.release()
cv2.destroyAllWindows()

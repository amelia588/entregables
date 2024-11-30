# Importamos las librerías

import cv2

import mediapipe as mp

from ultralytics import YOLO



# Inicializamos MediaPipe Hands

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

mp_drawing = mp.solutions.drawing_utils



# Cargamos el modelo YOLO para reconocimiento adicional

model = YOLO("best.pt")



# Inicializamos la cámara

cap = cv2.VideoCapture(0)



# Bucle principal

while True:

  # Leer fotogramas

  ret, frame = cap.read()



  # Verificar si se capturó el fotograma correctamente

  if not ret:

    print("No se pudo capturar el fotograma.")

    break



  # Volteamos el fotograma horizontalmente para corregir el reflejo

  frame = cv2.flip(frame, 1)



  # Convertimos el fotograma a RGB para MediaPipe

  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



  # Detección de manos con MediaPipe

  resultados = hands.process(rgb_frame)



  # Dibujamos las manos detectadas y obtenemos regiones de interés

  if resultados.multi_hand_landmarks:

    for hand_landmarks in resultados.multi_hand_landmarks:

      # Dibujamos las conexiones de la mano

      mp_drawing.draw_landmarks(

        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS

      )



      # Calculamos el bounding box aproximado de la mano

      h, w, _ = frame.shape

      x_min = min([int(landmark.x * w) for landmark in hand_landmarks.landmark])

      y_min = min([int(landmark.y * h) for landmark in hand_landmarks.landmark])

      x_max = max([int(landmark.x * w) for landmark in hand_landmarks.landmark])

      y_max = max([int(landmark.y * h) for landmark in hand_landmarks.landmark])



      # Aseguramos que el ROI esté dentro de los límites

      x_min, y_min = max(x_min, 0), max(y_min, 0)

      x_max, y_max = min(x_max, w), min(y_max, h)



      # Extraemos el ROI de la mano

      roi = frame[y_min:y_max, x_min:x_max]



      # Procesamos el ROI con YOLO si es suficientemente grande

      if roi.size > 0:

        yolo_resultados = model.predict(roi, imgsz=640, conf=0.4)



        # Dibujamos anotaciones de YOLO dentro del ROI

        anotaciones = yolo_resultados[0].plot()

        frame[y_min:y_max, x_min:x_max] = anotaciones



  # Mostramos los fotogramas con las detecciones

  cv2.imshow("DETECCION", frame)



  # Salir del programa con la tecla ESC (6)

  if cv2.waitKey(1) == 6:

    break



# Liberamos recursos

cap.release()

cv2.destroyAllWindows()
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# model
model = load_model('best_card_recognition_model.keras')

class_names = [
    'aceHearts', '2Hearts', '3Hearts', '4Hearts', '5Hearts', '6Hearts', '7Hearts', '8Hearts', '9Hearts', '10Hearts', 'jackHearts', 'queenHearts', 'kingHearts',
    'aceDiamonds', '2Diamonds', '3Diamonds', '4Diamonds', '5Diamonds', '6Diamonds', '7Diamonds', '8Diamonds', '9Diamonds', '10Diamonds', 'jackDiamonds', 'queenDiamonds', 'kingDiamonds',
    'aceClubs', '2Clubs', '3Clubs', '4Clubs', '5Clubs', '6Clubs', '7Clubs', '8Clubs', '9Clubs', '10Clubs', 'jackClubs', 'queenClubs', 'kingClubs',
    'aceSpades', '2Spades', '3Spades', '4Spades', '5Spades', '6Spades', '7Spades', '8Spades', '9Spades', '10Spades', 'jackSpades', 'queenSpades', 'kingSpades'
]

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
thickness = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_resized = cv2.resize(frame, (224, 224))
    frame_resized = frame_resized / 255.0
    frame_resized = np.expand_dims(frame_resized, axis=0)
    predictions = model.predict(frame_resized)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    cv2.putText(frame, f'Prediction: {predicted_class_name}', (10, 30), font, font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.imshow('Card Scanner', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import pandas as pd
import numpy as np

# Читаем данные акселерометра
accelerometer_data = pd.read_csv('./датчики/Accelerometer.csv')
# Простое масштабирование координат X и Y
x_points = np.array(accelerometer_data['x']) * 100
y_points = np.array(accelerometer_data['y']) * 100

# Открываем видео
cap = cv2.VideoCapture('.\TestTaskVslamAndOdometry\Траектория-_1-2024-03-27_13-11-09.zip')
frame_count = 0

# Создаём фон для отрисовки траектории
# Предполагаем, что размеры кадра 640x480
trajectory_background = np.zeros((480, 640, 3), dtype=np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Проверяем, есть ли у нас координаты для этого фрейма
    if frame_count < len(x_points):
        x = int(x_points[frame_count])
        y = int(y_points[frame_count])
        
        # Рисуем точку на фоне
        cv2.circle(trajectory_background, (200 + x, 200 + y), 5, (0, 255, 0), -1)
    
    # Наложим фон с траекторией на текущий кадр
    frame_with_trajectory = cv2.add(frame, trajectory_background)

    cv2.imshow('Frame with Trajectory', frame_with_trajectory)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):  # Скорость воспроизведения и выход
        break
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import pandas as pd

# Функция для добавления отрисовки траектории XY из файла
def add_xy_trajectory(trajectory_plot, ax, ay, frame_number):
    scale = 10  # масштаб для усиления величин движения
    origin = (100, 100)  # начальная точка

    if frame_number == 0:
        trajectory_plot[origin[1], origin[0]] = (0,0,255)
    else:
        new_point = (int(origin[0] + ax * scale), int(origin[1] + ay * scale))
        cv2.line(trajectory_plot, origin, new_point, (0, 255, 0), 2)
        origin = new_point
    return trajectory_plot

def draw_trajectory_with_csv(video_path, csv_path):
    # Считывание данных из файла
    data = pd.read_csv(csv_path)
    ax = data['x'].values
    ay = data['y'].values

    cap = cv2.VideoCapture(video_path)    
    ret, frame = cap.read()
    if not ret:
        print("Не удалось считать видео")
        return

    scale_percent = 50  # процент от оригинала
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    prev_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    trajectory = np.zeros_like(prev_frame)
    trajectory_plot = np.zeros_like(prev_frame)  # Для отрисовки XY траектории

    frame_number = 0

    while cap.isOpened():
        if frame_number < len(ax):
            trajectory_plot = add_xy_trajectory(trajectory_plot, ax[frame_number], ay[frame_number], frame_number)
            frame_number += 1
            
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        output = cv2.add(frame, trajectory_plot)

        cv2.imshow('Trajectory', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


video_path = './Draw_traectory/TestTaskVslamAndOdometry/20240327_161347_448.mp4'
accelerometer_path = '.\Draw_traectory\датчики\Accelerometer.csv'
draw_trajectory_with_csv(video_path, accelerometer_path)

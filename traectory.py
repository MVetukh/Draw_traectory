import cv2
from pathlib import Path
import numpy as np



# Функция для извлечения кадров из видео
def extract_frames(video_path, output_dir, skip_frames=10):
    """
    Извлекает кадры из видеофайла.

    :param video_path: Путь к файлу видео.
    :param output_dir: Директория для сохранения извлеченных кадров.
    :param skip_frames: Число пропускаемых кадров между сохранениями. Например, если skip_frames=30, 
                        то сохраняется каждый 30-й кадр.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_counter % skip_frames == 0:
            frame_filename = output_dir / f"frame_{frame_counter}.jpg"
            cv2.imwrite(str(frame_filename), frame)

        frame_counter += 1

    cap.release()
    print(f"Кадры успешно сохранены в {output_dir}.")

def draw_trajectory(video_path):
    # Инициализация переменных
    cap = cv2.VideoCapture(video_path)    
    ret, prev_frame = cap.read()
    if not ret:
        print("Не удалось считать видео")
        return
    # Уменьшение размера кадра
    scale_percent = 50  # Процент от исходного размера
    width = int(prev_frame.shape[1] * scale_percent / 100)
    height = int(prev_frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    prev_frame = cv2.resize(prev_frame, dim, interpolation=cv2.INTER_AREA)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    trajectory = np.zeros_like(prev_frame)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Оптический поток
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        magnitude, angle = cv2.cartToPolar(flow[:,:, 0], flow[:,:, 1])
        mean_mag = np.mean(magnitude)
        
        if 'trajectory_point' not in locals():
            trajectory_point = (frame.shape[1]//2, frame.shape[0]//2)
        
        dx, dy = int(mean_mag * np.cos(np.mean(angle))), int(mean_mag * np.sin(np.mean(angle)))
        trajectory_point = (trajectory_point[0] + dx, trajectory_point[1] + dy)
    
        cv2.circle(trajectory, trajectory_point, 5, (0, 0, 255), -1)
        output = cv2.add(frame, trajectory)
    
        cv2.imshow('Trajectory', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        prev_gray = gray
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    video_path = './Draw_traectory/TestTaskVslamAndOdometry/20240327_161347_448.mp4'
    #output_dir = './TestTaskVslamAndOdometry/Draw_traectory/frames'
   # extract_frames(video_path, output_dir, skip_frames=10)

    draw_trajectory(video_path)
   
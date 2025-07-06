import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN

class DroneObjectDetector:
    def __init__(self, video_path, output_video_path, output_map_path):
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.output_map_path = output_map_path
        self.global_coordinates = []  # Хранит все обнаруженные координаты объектов
        self.accumulated_homography = np.eye(3)  # Накопленная гомография для преобразования координат
        self.frame_count = 0

    def detect_objects(self, frame):
        """Обнаружение контрастных объектов с помощью адаптивного порога и морфологических операций"""
        # Преобразование в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Адаптивная бинаризация для выделения объектов
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Морфологические операции для улучшения качества
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        # Поиск контуров
        contours, _ = cv2.findContours(
            processed, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Фильтрация контуров по размеру
        min_area = 100
        max_area = 10000
        valid_contours = [
            cnt for cnt in contours 
            if min_area < cv2.contourArea(cnt) < max_area
        ]
        
        return valid_contours

    def track_camera_movement(self, prev_frame, curr_frame):
        """Отслеживание движения камеры с помощью оптического потока"""
        # Параметры для алгоритма Лукаса-Канаде
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Поиск точек для отслеживания
        prev_pts = cv2.goodFeaturesToTrack(
            prev_frame,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3
        )
        
        if prev_pts is not None:
            # Расчет оптического потока
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_frame, curr_frame, 
                prev_pts, None, **lk_params
            )
            
            # Отбор успешно отслеженных точек
            valid_prev = prev_pts[status == 1]
            valid_curr = curr_pts[status == 1]
            
            if len(valid_prev) > 4:
                # Расчет матрицы преобразования
                H, _ = cv2.estimateAffinePartial2D(
                    valid_prev, valid_curr
                )
                if H is not None:
                    # Преобразование в гомографию
                    H_full = np.vstack([H, [0, 0, 1]])
                    return H_full
        return np.eye(3)

    def process_video(self):
        """Основной метод обработки видео"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError("Ошибка открытия видеофайла")
        
        # Получение параметров видео
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Инициализация видео-писателя
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            self.output_video_path, fourcc, fps, 
            (width, height)
        )
        
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Не удалось прочитать первый кадр")
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frame_count += 1
            
            # 1. Отслеживание движения камеры
            H = self.track_camera_movement(prev_gray, curr_gray)
            self.accumulated_homography = self.accumulated_homography @ H
            
            # 2. Обнаружение объектов
            contours = self.detect_objects(frame)
            
            # 3. Обработка и сохранение координат
            for cnt in contours:
                # Расчет центра объекта
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Преобразование в глобальные координаты
                    global_point = self.accumulated_homography @ np.array([cx, cy, 1])
                    global_point /= global_point[2]  # Нормализация
                    
                    # Сохранение координат
                    self.global_coordinates.append(global_point[:2])
                    
                    # Визуализация на кадре
                    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(
                        frame, f"{int(global_point[0])},{int(global_point[1])}", 
                        (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1
                    )
            
            # Запись обработанного кадра
            out.write(frame)
            
            # Обновление предыдущего кадра
            prev_gray = curr_gray.copy()
            
        # Освобождение ресурсов
        cap.release()
        out.release()
        
        # 4. Постобработка координат
        self.postprocess_coordinates()
        
        # 5. Сохранение координатной карты
        self.save_coordinate_map()

    def postprocess_coordinates(self):
        """Кластеризация координат для объединения дубликатов"""
        if not self.global_coordinates:
            return
            
        coords = np.array(self.global_coordinates)
        
        # Кластеризация DBSCAN для объединения близких точек
        db = DBSCAN(eps=50, min_samples=1).fit(coords)
        labels = db.labels_
        
        # Усреднение координат в кластерах
        unique_labels = set(labels)
        clustered_coords = []
        
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = coords[labels == label]
            mean_coord = np.mean(cluster_points, axis=0)
            clustered_coords.append(mean_coord)
            
        self.global_coordinates = clustered_coords

    def save_coordinate_map(self):
        """Сохранение координатной карты в текстовый файл"""
        with open(self.output_map_path, 'w') as f:
            f.write("Объектная карта координат (относительно точки взлета):\n")
            f.write("X\tY\n")
            for coord in self.global_coordinates:
                f.write(f"{int(coord[0])}\t{int(coord[1])}\n")

if __name__ == "__main__":
    # Параметры обработки
    input_video = "Timeline 1.mp4"
    output_video = "processed_output.avi"
    output_map = "object_coordinates.txt"   
    
    # Обработка видео
    detector = DroneObjectDetector(input_video, output_video, output_map)
    detector.process_video()
    
    print(f"Обработка завершена. Результаты сохранены в {output_video} и {output_map}")

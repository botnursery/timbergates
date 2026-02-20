import cv2
import numpy as np
import torch
from ultralytics import YOLO

def main():
    MIN_CONF = 0.25  # Настройте уверенность один раз здесь
    
    # 1. Загрузка модели
    model_path = r'Drone_Racing_Project\Forest_and_Gates\weights\best.pt'
    model = YOLO(model_path)

    # 2. Настройка видео
    video_path = r'Drone_Racing_Project\17s(2ach&tree).mp4'
    cap = cv2.VideoCapture(video_path)
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 3. Калибровка камеры (упрощенная)
    focal_length = W 
    cam_matrix = np.array([[focal_length, 0, W/2], [0, focal_length, H/2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # 4. 3D модель ворот (1x1 метр)
    obj_pts = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], dtype=np.float32)

    print(f"Обработка видео {W}x{H}. Нажмите 'q' для выхода.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Инициализируем нейтральные сигналы и дистанцию по умолчанию
        # [Throttle, Roll, Pitch, Yaw]
        ctrl = [0.5, 0.5, 0.5, 0.5] 
        current_dist = 0.0
        
        # 1. Запуск трекинга (persist=True обязателен для --track)
        results_list = model.track(frame, conf=MIN_CONF, persist=True, verbose=False, device=0)
        
        annotated_frame = frame.copy()
        # Проверяем, что список не пуст
        if results_list and len(results_list[0].obb) > 0:
            results = results_list[0] # Берем первый (единственный) результат
            # Отрисовываем рамки и скоры (аналог --show_scores)
            annotated_frame = results.plot(conf=True, line_width=2)

            # 2. Поиск ворот (класс 0)
            gate_indices = [i for i, cls in enumerate(results.obb.cls) if int(cls) == 0]
            
            if gate_indices:
                all_obbs = results.obb
                # Берем только площади воротс с самой большой площадью (ближайшие)
                areas = all_obbs.xywhr[gate_indices, 2] * all_obbs.xywhr[gate_indices, 3]
                best_gate_idx = gate_indices[torch.argmax(areas).item()]
                # Получаем 2D точки выбранных ворот
                target_pts = all_obbs.xyxyxyxy[best_gate_idx].cpu().numpy().reshape(4, 2).astype(np.float32)

                # 3. SolvePnP рисуем ПОВЕРХ кадра с рамками
                success, rvec, tvec = cv2.solvePnP(obj_pts, target_pts, cam_matrix, dist_coeffs)

                if success:
                    # Рисуем оси на annotated_frame(X-красный, Y-зеленый, Z-синий) 
                    # Длина осей 0.5 метра
                    cv2.drawFrameAxes(annotated_frame, cam_matrix, dist_coeffs, rvec, tvec, 0.5)
                    # расчет tvec и rvec
                    
                    tx, ty, tz = tvec.flatten()
                    current_dist = tz # Сохраняем дистанцию для вывода

                    # Расчет углов для управления (Yaw и Throttle)
                    yaw_err = np.arctan2(tx, tz)
                    pitch_angle = np.arctan2(-ty, tz)

                    # Формируем сигналы, значения в список по индексам
                    ctrl[0] = float(np.clip(0.5 + (pitch_angle / 0.8), 0, 1)) # Throttle
                    ctrl[1] = 0.5                                            # Roll (нейтраль)
                    ctrl[2] = 0.6 if tz > 1.8 else 0.5                       # Pitch (вперед)
                    ctrl[3] = float(np.clip(0.5 + (yaw_err / 0.8), 0, 1))    # Yaw (поворот)

                    # Текст на экране
                    # Метка цели
                    cx, cy = int(all_obbs.xywhr[best_gate_idx, 0]), int(all_obbs.xywhr[best_gate_idx, 1])
                    cv2.putText(annotated_frame, f"TARGET GATE: {tz:.1f}m", (cx - 50, cy - 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Показываем результат, где есть и рамки от YOLO, и оси от PnP в консоль (обновление строки)
        cv2.imshow('Drone PnP View', annotated_frame)
        
        # ВЫВОД В КОНСОЛЬ (теперь вне всех условий)
        print(f"Dist:{current_dist:4.1f}m | Thr:{ctrl[0]:.2f} | Rol:{ctrl[1]:.2f} | Pit:{ctrl[2]:.2f} | Yaw:{ctrl[3]:.2f}", end='\r')

        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()
    print("\nОбработка завершена.")

if __name__ == '__main__':
    main()

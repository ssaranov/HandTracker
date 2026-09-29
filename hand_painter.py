import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Автоматически привязываемся к текущей папке скрипта
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'hand_landmarker.task')

# ИСПРАВЛЕННАЯ ССЫЛКА НА GOOGLE STORAGE
url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

# Скачивание с обработкой ошибок сети
if not os.path.exists(model_path):
    print("Файл модели не найден. Попытка скачать веса...")
    try:
        urllib.request.urlretrieve(url, model_path)
        print("Модель успешно загружена!")
    except Exception as e:
        print(f"\n[!] Ошибка скачивания: {e}")
        print("Google заблокировал прямую загрузку или нет интернета.")
        print(f"Пожалуйста, скачай файл вручную по ссылке:\n{url}")
        print(f"И просто закинь его в папку: {script_dir}\n")
        model_path = None

# Чтение модели в память
if model_path and os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model_bytes = f.read()
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1
    )
else:
    print("Запуск невозможен без файла модели. Скачай его вручную по ссылке выше.")
    exit()

# Цвета: Графитовый, Синий, Красный, Зеленый
colors = [(60, 60, 60), (242, 100, 25), (38, 38, 240), (40, 190, 80)]
current_color = colors[1]  
brush_thickness = 5
eraser_thickness = 50  # Размер ластика
canvas = None

cap = cv2.VideoCapture(0)
window_name = 'Apple Air Painter'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

px, py = 0, 0

with HandLandmarker.create_from_options(options) as landmarker:
    print("\nAir Painter запущен!")
    print("👉 Один палец вверх — РИСОВАНИЕ")
    print("✌️ Два пальца вверх — ЛАСТИК\n")
    
    while cap.isOpened():
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
            
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp == 0:
            timestamp = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp)
        
        drawing_mode = False
        eraser_mode = False
        cx, cy = 0, 0

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # Координаты указательного пальца (кончик и сустав)
                index_tip = hand_landmarks[8]
                index_pip = hand_landmarks[6]
                
                # Координаты среднего пальца (кончик и сустав)
                middle_tip = hand_landmarks[12]
                middle_pip = hand_landmarks[10]
                
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                
                # Проверяем, какие пальцы подняты
                index_up = index_tip.y < index_pip.y
                middle_up = middle_tip.y < middle_pip.y
                
                if index_up and middle_up:
                    # Поднято два пальца -> Режим ЛАСТИКА
                    eraser_mode = True
                    # Рисуем круг-индикатор ластика вокруг указательного пальца
                    cv2.circle(frame, (cx, cy), eraser_thickness // 2, (200, 200, 200), 2, cv2.LINE_AA)
                elif index_up:
                    # Поднят только указательный -> Режим РИСОВАНИЯ
                    drawing_mode = True
                    cv2.circle(frame, (cx, cy), int(brush_thickness + 2), (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), brush_thickness, current_color, cv2.FILLED, cv2.LINE_AA)

        # Работа с верхним меню (выбор цвета/очистка) работает только в режиме рисования
        if drawing_mode and cy < 70:
            if w - 120 < cx < w - 20:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
            elif 20 < cx < 260:
                idx = (cx - 20) // 60
                if 0 <= idx < len(colors):
                    current_color = colors[idx]

        # Логика рисования / стирания
        if (drawing_mode or eraser_mode) and cy > 70:
            if px == 0 and py == 0:
                px, py = cx, cy
            
            if drawing_mode:
                cv2.line(canvas, (px, py), (cx, cy), current_color, brush_thickness, cv2.LINE_AA)
            elif eraser_mode:
                # Стираем — просто рисуем по холсту черным (0, 0, 0) толстой линией
                cv2.line(canvas, (px, py), (cx, cy), (0, 0, 0), eraser_thickness, cv2.FILLED)
                
            px, py = cx, cy
        else:
            px, py = 0, 0

        # Наложение холста
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        frame = cv2.add(frame_bg, canvas_fg)

        # Меню Apple-style
        menu_overlay = frame.copy()
        cv2.rectangle(menu_overlay, (0, 0), (w, 70), (255, 255, 255), cv2.FILLED)
        cv2.addWeighted(menu_overlay, 0.15, frame, 0.85, 0, frame)
        cv2.line(frame, (0, 70), (w, 70), (200, 200, 200), 1, cv2.LINE_AA)

        for i, col in enumerate(colors):
            center_x = 40 + i * 60
            cv2.circle(frame, (center_x, 35), 18, col, cv2.FILLED, cv2.LINE_AA)
            if col == current_color and not eraser_mode:
                cv2.circle(frame, (center_x, 35), 21, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (w - 120, 20), (w - 20, 50), (40, 40, 40), cv2.FILLED, cv2.LINE_AA)
        cv2.putText(frame, "Очистить", (w - 103, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

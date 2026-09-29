import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
import math

# Библиотеки ИИ
import torch
import torchvision.models as models
import torchvision.transforms as transforms

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'hand_landmarker.task')
url = 'https://googleapis.com'

if not os.path.exists(model_path):
    print("Скачивание модели Mediapipe...")
    try:
        urllib.request.urlretrieve(url, model_path)
    except Exception as e:
        print(f"[!] Ошибка скачивания: {e}")
        exit()

if model_path and os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model_bytes = f.read()
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1
    )
else:
    exit()

# Инициализация ИИ (MobileNetV2)
print("Загрузка ИИ...")
weights = models.MobileNet_V2_Weights.DEFAULT
ai_model = models.mobilenet_v2(weights=weights)
ai_model.eval()
categories = weights.meta["categories"]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Настройки холста
colors = [(60, 60, 60), (242, 100, 25), (38, 38, 240), (40, 190, 80)]
current_color = colors[1]  
brush_thickness = 8
eraser_thickness = 50
canvas = None

canvas_history = []
max_history = 10
was_drawing_last_frame = False
undo_cooldown = 0
ai_prediction = "Рисуйте..."
ai_cooldown = 0  

# Настройки сглаживания
smooth_x, smooth_y = 0, 0
smoothing_factor = 0.25  

cap = cv2.VideoCapture(0)
window_name = 'Apple Air Painter Smooth Pro'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

px, py = 0, 0

def save_to_history(canv):
    global canvas_history
    if len(canvas_history) >= max_history:
        canvas_history.pop(0)
    canvas_history.append(canv.copy())

with HandLandmarker.create_from_options(options) as landmarker:
    print("\n=== Удобный Air Painter с ИИ Запущен ===")
    
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
            save_to_history(canvas)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp == 0:
            timestamp = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp)
        
        drawing_mode = False
        eraser_mode = False
        pinch_mode = False
        fist_mode = False
        
        cx, cy = 0, 0
        status_text = "Просмотр"

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                lm = hand_landmarks
                thumb_tip = lm[4]
                index_tip, index_pip = lm[8], lm[6]
                middle_tip, middle_pip = lm[12], lm[10]
                ring_tip = lm[16]
                pinky_tip = lm[20]
                
                raw_cx, raw_cy = int(index_tip.x * w), int(index_tip.y * h)
                
                # Сглаживание
                if smooth_x == 0 and smooth_y == 0:
                    smooth_x, smooth_y = raw_cx, raw_cy
                else:
                    smooth_x = smooth_x + smoothing_factor * (raw_cx - smooth_x)
                    smooth_y = smooth_y + smoothing_factor * (raw_cy - smooth_y)
                
                cx, cy = int(smooth_x), int(smooth_y)
                
                index_up = index_tip.y < index_pip.y
                middle_up = middle_tip.y < middle_pip.y
                ring_up = ring_tip.y < lm[14].y
                pinky_up = pinky_tip.y < lm[18].y
                
                dist_thumb_index = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y) * w

                if not index_up and not middle_up and not ring_up and not pinky_up:
                    fist_mode = True
                elif index_up and dist_thumb_index < 35:
                    pinch_mode = True
                elif index_up and middle_up:
                    eraser_mode = True
                elif index_up:
                    drawing_mode = True

                if eraser_mode:
                    status_text = "Ластик"
                    cv2.circle(frame, (cx, cy), eraser_thickness // 2, (105, 105, 255), 2, cv2.LINE_AA)
                elif pinch_mode:
                    status_text = "Размер кисти"
                    brush_thickness = int(np.clip(dist_thumb_index / 2, 3, 40))
                    cv2.circle(frame, (cx, cy), brush_thickness, (255, 255, 255), 2, cv2.LINE_AA)
                elif drawing_mode:
                    status_text = "Рисование"
                    cv2.circle(frame, (cx, cy), brush_thickness + 4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), brush_thickness, current_color, cv2.FILLED, cv2.LINE_AA)
        else:
            smooth_x, smooth_y = 0, 0

        if (drawing_mode or eraser_mode) and cy > 70:
            was_drawing_last_frame = True
        else:
            if was_drawing_last_frame:
                save_to_history(canvas)
                was_drawing_last_frame = False

        if drawing_mode and cy < 70:
            if w - 120 < cx < w - 20:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                save_to_history(canvas)
                ai_prediction = "Очищено"
            elif 20 < cx < 260:
                idx = (cx - 20) // 60
                if 0 <= idx < len(colors):
                    current_color = colors[idx]

        if (drawing_mode or eraser_mode) and cy > 70:
            if px == 0 and py == 0:
                px, py = cx, cy
            
            if math.hypot(cx - px, cy - py) < 80:
                if drawing_mode:
                    cv2.line(canvas, (px, py), (cx, cy), current_color, brush_thickness, cv2.LINE_AA)
                elif eraser_mode:
                    cv2.line(canvas, (px, py), (cx, cy), (0, 0, 0), eraser_thickness, cv2.FILLED)
            
            px, py = cx, cy
        else:
            px, py = 0, 0

        if fist_mode and time.time() > undo_cooldown:
            if len(canvas_history) > 1:
                canvas_history.pop()
                canvas = canvas_history[-1].copy()
                status_text = "Отмена (Undo)"
                undo_cooldown = time.time() + 1.2

        # --- ИСПРАВЛЕННЫЙ БЛОК ИИ РАСПОЗНАВАНИЯ ---
        if time.time() > ai_cooldown and np.max(canvas) > 0:
            ai_cooldown = time.time() + 0.6
            try:
                # Преобразуем холст и добавляем размерность пакета ОДИН раз через unsqueeze(0)
                input_tensor = transform(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).unsqueeze(0)
                with torch.no_grad():
                    prediction = ai_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
                    top_prob, top_catid = torch.topk(probabilities, 1)
                    
                    if top_prob.item() > 0.12:
                        ai_prediction = f"{categories[top_catid.item()]} ({top_prob.item()*100:.1f}%)"
                    else:
                        ai_prediction = "Распознавание..."
            except Exception as e:
                pass

        # Эффект неона
        canvas_blur = cv2.GaussianBlur(canvas, (11, 11), 0)
        neon_canvas = cv2.addWeighted(canvas, 1.0, canvas_blur, 1.3, 0)

        # Слияние
        gray_canvas = cv2.cvtColor(neon_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(neon_canvas, neon_canvas, mask=mask)
        frame = cv2.add(frame_bg, canvas_fg)

        # Меню
        menu_overlay = frame.copy()
        cv2.rectangle(menu_overlay, (0, 0), (w, 70), (255, 255, 255), cv2.FILLED)
        cv2.addWeighted(menu_overlay, 0.15, frame, 0.85, 0, frame)
        cv2.line(frame, (0, 70), (w, 70), (220, 220, 220), 1, cv2.LINE_AA)

        for i, col in enumerate(colors):
            center_x = 40 + i * 60
            cv2.circle(frame, (center_x, 35), 18, col, cv2.FILLED, cv2.LINE_AA)
            if col == current_color and not eraser_mode:
                cv2.circle(frame, (center_x, 35), 21, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (w - 120, 20), (w - 20, 50), (40, 40, 40), cv2.FILLED, cv2.LINE_AA)
        cv2.putText(frame, "Очистить", (w - 103, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Вывод текста
        cv2.putText(frame, f"Режим: {status_text}", (280, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(frame, f"ИИ видит: {ai_prediction}", (280, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 140, 30), 1, cv2.LINE_AA)
        
        if brush_thickness > 0 and not eraser_mode:
            cv2.putText(frame, f"Размер: {brush_thickness}px", (w - 320, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

import os
import time
import cv2

# 1. Создаём папку для сохранения, если её ещё нет
save_folder = "captured_photos"
os.makedirs(save_folder, exist_ok=True)

# 2. Открываем камеру (укажите индекс 0 или 1, который сработал на вашем Mac)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
time.sleep(0.5)  # Короткая пауза для инициализации камеры на macOS

if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру.")
    exit()

print("Нажмите 's', чтобы сделать снимок, или 'q', чтобы выйти.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Camera (s - save, q - quit)", frame)
    key = cv2.waitKey(1) & 0xFF

    # Сохраняем кадр при нажатии 's'
    if key == ord('s'):
        # Формируем уникальное имя файла с текущей датой и временем
        filename = f"photo_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(save_folder, filename)

        cv2.imwrite(filepath, frame)
        print(f"Фото сохранено: {filepath}")

    # Выход при нажатии 'q'
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

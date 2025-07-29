import easyocr
import cv2
from PIL import ImageFont, ImageDraw, Image

import numpy as np


# font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
# font = ImageFont.truetype(font_path, size=30)

# img_pil = Image.fromarray(frame)
# frame = ImageDraw.Draw(img_pil)
# frame.text((50, 50), "한글도 잘 나와요!", font=font, fill=(255, 0, 0))


# 다시 OpenCV 이미지로 변환
# img = np.array(img_pil)


reader = easyocr.Reader(['ko', 'en'], gpu=False)
cap = cv2.VideoCapture(0)  # 기본 카메라





if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('camera', frame)
    key = cv2.waitKey(1)

    if key == ord('c'):  # c 누르면 OCR 수행
        results = reader.readtext(frame)


        for (bbox, text, prob) in results:
            print(f"Text: {text}, Confidence: {prob:.2f}")
            pts = [tuple(map(int, point)) for point in bbox]

            # Draw rectangle
            for j in range(4):
                pt1 = pts[j]
                pt2 = pts[(j + 1) % 4]
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)


            # Draw each corner coordinate
            for i, pt in enumerate(pts):
                
                cv2.putText(frame, f'{pt}', (pt[0], pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                

                

            # Draw label: text + confidence
            label = f'Prob: ({prob:.2f})'
            cv2.putText(frame, label, (pts[0][0], pts[0][1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)





        cv2.imshow('OCR result', frame)
        cv2.waitKey(0)

    elif key == 27:  # ESC 종료
        break

cap.release()
cv2.destroyAllWindows()



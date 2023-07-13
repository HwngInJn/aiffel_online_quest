# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 황인준
- 리뷰어 : 서희원


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 네 한줄한줄 주석을 작성하셔서 동작 로직을 이해하기 쉬웠습니다.
  ```python
  list_landmarks = []
      # 랜드마크의 위치를 저장할 list 생성    
  
  # 얼굴 영역 박스 마다 face landmark를 찾아냅니다
  # face landmark 좌표를 저장해둡니다
  for dlib_rect in dlib_rects:
      points = landmark_predictor(img_rgb, dlib_rect)
          # 모든 landmark의 위치정보를 points 변수에 저장
      list_points = list(map(lambda p: (p.x, p.y), points.parts()))
          # 각각의 landmark 위치정보를 (x,y) 형태로 변환하여 list_points 리스트로 저장
      list_landmarks.append(list_points)
          # list_landmarks에 랜드마크 리스트를 저장
  
  print(len(list_landmarks[0]))
      # 얼굴이 n개인 경우 list_landmarks는 n개의 원소를 갖고
      # 각 원소는 68개의 랜드마크 위치가 나열된 list 
      # list_landmarks의 원소가 1개이므로 list_landmarks[1]을 호출하면 IndexError가 발생
  ```
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 네 아주 잘작동했습니다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네 잘 이해해 다른 관점에서 코드를 개발하셨습니다
  ```python
  sticker_area = img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
  
  # img_sticker의 알파 채널만 사용하여 조건을 만든다
  img_sticker_alpha = img_sticker[:, :, 3]
  img_sticker_alpha = np.repeat(img_sticker_alpha[:, :, np.newaxis], 3, axis=2)
  
  # img_sticker를 RGB 이미지로 변환
  img_sticker_rgb = img_sticker[:, :, :3]
  
  img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = np.where(img_sticker_alpha==0,sticker_area,img_sticker_rgb).astype(np.uint8)
  
  plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
  plt.show()
  ```
- [X] 코드가 간결한가요?
  > 네 로직을 함수화해 다양한 이미지를 돌리셨습니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
각 매서드의 의미를 서술

# 참고 링크 및 코드 개선
```python
img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = np.where(img_sticker_alpha==0,sticker_area,img_sticker_rgb).astype(np.uint8) 
img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \ # 위에 코드를 이 코드로 바꿔 스티커의 투명도를 조절하면 자연스럽게 만들 수 있습니다.
    cv2.addWeighted(sticker_area, 0.5, np.where(img_sticker_alpha==0,sticker_area,img_sticker_rgb).astype(np.uint8), 0.5, 0)
```

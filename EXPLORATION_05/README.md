# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 황인준
- 리뷰어 : 조준규

# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 코드 자체의 주석이 많지는 않았지만 markdown으로 각 코드 세션에서 어떤 일을 수행하려하는지 나타나있었기에 쉽게 이해할 수 있었다.
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 반복문을 사용하여 여러 이미지에 대한 처리를 독립적으로 진행하였기 때문에 conflict가 일어날 일이 없었다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > OpenCV, Matplotlib 등의 라이브러리에서 제공하는 함수를 적재적소에 알맞게 사용한 것으로 보아 코드를 잘 이해하고 작성했다고 볼 수 있다.
- [X] 코드가 간결한가요?
  > 코드가 중복되어 여러 번 실행되지 않도록 for loop를 사용하여 코드가 간결하였다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
# 코드 에러 유발 가능성 및 코드 간결성 예시
# 코드블록 하나로 작업을 수행할 수 있도록 하였다.
import matplotlib.pyplot as plt

for file in file_list:
    img_path = os.path.join(dir_path, file)
    
    scale_percent = 0.1     # 이미지 해상도가 높아서 1/10로 리스케일링
    img = cv2.imread(img_path)
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    
    img_orig = cv2.resize(img, dim)

    fig = plt.figure(figsize=(10, 20))  

    ax1 = fig.add_subplot(231)
    ax1.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')

    segvalues, output = model.segmentAsPascalvoc(img_path)
    print(segvalues)
    for class_id in segvalues['class_ids']:
        print(LABEL_NAMES[class_id])

    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    seg_color = (128, 128, 192)
    seg_map_bool = np.all(output == seg_color, axis=-1)
    seg_map_int = seg_map_bool.astype(int)
    seg_map_resized = cv2.resize(seg_map_int, dim, interpolation = cv2.INTER_NEAREST)

    ax2 = fig.add_subplot(232)
    ax2.imshow(seg_map_resized, cmap='gray')
    ax2.set_title('Segmentation Map')

    img_show = img_orig.copy()
    img_mask = seg_map_resized.astype(np.uint8) * 255
    color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)
    img_show = cv2.addWeighted(img_show, 0.6, color_mask, 0.4, 0.0)

    ax3 = fig.add_subplot(233)
    ax3.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
    ax3.set_title('Image with Color Mask')

    img_orig_blur = cv2.blur(img_orig, (13, 13))

    ax4 = fig.add_subplot(234)
    ax4.imshow(cv2.cvtColor(img_orig_blur, cv2.COLOR_BGR2RGB))
    ax4.set_title('Blurred Image')

    img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    img_bg_mask = cv2.bitwise_not(img_mask_color)
    img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)

    ax5 = fig.add_subplot(235)
    ax5.imshow(cv2.cvtColor(img_bg_blur, cv2.COLOR_BGR2RGB))
    ax5.set_title('Background Blur')

    img_concat = np.where(img_mask_color == 255, img_orig, img_bg_blur)

    ax6 = fig.add_subplot(236)
    ax6.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
    ax6.set_title('Final Result')

    plt.tight_layout()
    plt.show()
```

# 참고 링크 및 코드 개선
```python
# 웨딩 이미지에서 손 부분이 집중력을 흡수하여 segmentation 결과가 이상한 것 같다고 말씀해주셨습니다.
# 웨딩 이미지에서 손 부분을 한 번 임의로 가리고 segmentation을 진행하면 어떤 결과가 나올지,
# 온전히 두 사람을 같이 잘 인식할지, 아니면 신랑만 인식할지 궁금해지네요.😊
```

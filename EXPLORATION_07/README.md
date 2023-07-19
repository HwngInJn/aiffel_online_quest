# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 황인준
- 리뷰어 : 김석영


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 네. 에러 없이 코드가 정상적으로 동작하며, 주어진 프로젝트 필수 수행사항 및 평가문항의 내용들을 전반적으로 잘 수행하였습니다.
- [ ] 주석을 보고 작성자의 코드가 이해되었나요?
  > 네. 주석뿐만 아니라 Tast별로 제목과 요약이 잘 작성돼 있어 코드 이해가 잘 됐습니다.
- [ ] 코드가 에러를 유발할 가능성이 없나요?
  > 네. 에러 유발 가능성이 있는 코드나 구조는 없습니다.
- [ ] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네. 설명이 필요한 코드의 경우, 코드별로 주석이 잘 기재돼 있으며, 그 의미 역시 제대로 이해했음을 확인할 수 있습니다.
- [ ] 코드가 간결한가요?
  > 네. 중복되는 코드가 없으며, 필요 라이브러리도 관련된 코드에만 import가 돼 있고, 전체적으로 간결하게 작성된 코드라 생각됩니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
for test_ind in range(5):
    f = glob.glob(f"{val_path}/*")[test_ind]
    seg, orig = load_img(f)

    pred = generator(tf.expand_dims(seg, 0))
    pred = denormalize(pred)

    plt.figure(figsize=(20,10))
    plt.subplot(1,3,1); plt.imshow(denormalize(orig))
    plt.subplot(1,3,2); plt.imshow(pred[0])
    plt.subplot(1,3,3); plt.imshow(denormalize(seg))
```

# 참고 링크 및 코드 개선
```python
로그에 대한 코드를 보완한다면 더 좋을 것 같습니다.
```

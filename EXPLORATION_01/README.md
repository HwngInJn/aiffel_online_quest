# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 황인준
- 리뷰어 : 김민식


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  > 위 항목에 대한 근거 작성 필수
  > Project1번
  > 데이터 가져오는 부분 : 무조건 컬럼들을 다 들고 오는게 아니라 작성자의 판단하에 선형성이 없는 컬럼을 제거하고 데이터를 가져옴
  > 
- [O] 코드가 에러를 유발할 가능성이 없나요?
  >위 항목에 대한 근거 작성 필수
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 위 항목에 대한 근거 작성 필수
- [O] 코드가 간결한가요?
  > 위 항목에 대한 근거 작성 필수

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python

```

# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
# mean_squared_error()로 RMSE 구현하기
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
# 출처: https://growingsaja.tistory.com/233

# Datetime64 형식으로 변경하는 방법
# 참고2: httpsL//blog.naver.com/wideeyed/221603462366
train["datetime"] = pd.to_datetime(train["datetime"], format='%Y-%m-%d %H:%M:%S', errors='raise')

# subplot에서 간격 자동 조절
# 옵션 참고: https://steadiness-193.tistory.com/174
fig, ax = plt.subplots(2, 3, constrained_layout=True)
```

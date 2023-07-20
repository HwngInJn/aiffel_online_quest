# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 황인준
- 리뷰어 : 김석영


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 네. 모든 코드가 정상적으로 동작하며, 세 가지 평가문항 모두 충족하였습니다.
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 네. 주석뿐만이 아니라, 각 Task별 제목도 적정하게 잘 작성하여, 전체 흐름 파악과 코드 이해에 무리가 없었습니다.
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 네. 에러를 유발할 만한 요소는 찾을 수 없는 것 같습니다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네. 데이터 분포 확인에 대한 시각화도 상세히 잘 하였고,
  > 특히 마지막에 Testing 결과에 대한 분석에서 작성한 코드뿐만 아니라, 본 프로젝트에 대한 이해가 충분함을 확인할 수 있습니다.
- [X] 코드가 간결한가요?
  > 네. 코드는 간결합니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
answers = data['A'].apply(preprocess_sentence)
answers
```
```python
sns.histplot(questions.str.split().str.len())

plt.xlabel('Length of question')
plt.ylabel('Frequency')

# Add arrow and text
plt.annotate('Selected Max length', xy=(10, 0.0), xytext=(11, 500),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()
```

# 참고 링크 및 코드 개선
```python

```

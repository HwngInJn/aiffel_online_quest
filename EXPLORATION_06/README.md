# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 황인준
- 리뷰어 : 손정민


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 전처리가 체계적으로 진행되었고, 학습 과정에서의 loss 시각화와 결과 분석까지 성공적으로 수행했다.
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 각 과정마다 어떤 역할을 하는 코드인지 주석으로 설명이 되어 있어 이해하기 쉬웠다,
  ```python
  # data info

  print(data.info)  # [98401 rows x 2 columns]
  data.sample(10)   # columns - > headlines, text
  ```
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 에러 없이 잘 실행되었다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 결과 분석 파트에서 어떤 상황에서 문제가 발생했는지, 어떻게 개선시킬 수 있는지에 대한 언급이 있었고 추상적, 추출적 두 가지 방법의 결과에 대한 성능 평가 또한 포함되어 있었다. 작성한 코드에 대한 이해도가 높아 보인다.
- [X] 코드가 간결한가요?
  > 변수명이 직관적이고 각 변수에 대한 설명도 주석으로 제공했다. 코드 내의 개행이나 공백의 간격이 적절하여 가독성이 좋았다.
  ```python
  total_cnt = len(tar_tokenizer.word_index) # 단어의 수
  rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
  total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
  rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합
  ```


# 참고 링크 및 코드 개선
```python
# Extractive Text Summarization with summa

urllib.request.urlretrieve("https://raw.githubusercontent.com/sunnysai12345/News_Summary/master/news_summary_more.csv", filename="news_summary_more.csv")
summa_data = pd.read_csv('news_summary_more.csv', encoding='iso-8859-1')

for i in range(0,10):
  text = summa_data['text'][i]
  summary = summa_data['headlines'][i]

  print(f'실제 headline: {summary}')
  print(f'에측 headline: {summarize(text, ratio=0.5)}')
```
문장을 하나씩 뽑아 결과를 출력하셨는데, 이렇게 하면 여러 개의 summary를 출력할 수 있을 것 같습니다.
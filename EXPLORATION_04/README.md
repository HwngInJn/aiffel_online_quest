# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 황인준
- 리뷰어 : 조대호 


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  코드가 정상 작동하였고, 모델의 validation accuracy도 85%가 넘어서 문제를 해결했습니다.
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  > 모델 부분이나, 데이터 전처리 부분에 주석을 달아주어서 이해하는데 도움이 되었습니다.
  > ex) vocab_size = 10000       # 단어 주머니 10000개
  > ex) word_vector_dim = 50     # 각 토큰별 특성 갯수
- [O] 코드가 에러를 유발할 가능성이 없나요?
  > 주석도 달아주셨고, 에러가 발생하지 않았습니다.
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 변수이름도 직관적이며, 궁금했던 코드에 대해 답변을 잘해주셔서 코드를 이해했다고 생각합니다.
- [O] 코드가 간결한가요?
  > 필요한 부분만 작성하였고, 중복 되는 부분을 함수로 구현하여 간결하게 작성하였습니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
#데이터셋 내 문장 길이 분포 시각화
import seaborn as sns
import matplotlib.pyplot as plt

total_list = X_train + X_test
words_in_X = [len(s) for s in total_list]
print('max number of words: ', max(words_in_X))
print(Counter(words_in_X).most_common(40))

plt.title('word # in each sentence')
sns.kdeplot(data= words_in_X)
plt.annotate('Select Here', xy=(40, 0), xytext=(50, 0.01), arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()


#각 3가지 모델에 대한 시각화
e_titles = ['e-RNN', 'e-CNN', 'e-MaxPooling']
for i, history in enumerate(e_histories):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1, len(acc) + 1)

     # 그래프 크기를 조절
    plt.figure(figsize=(10, 5))

    
    plt.subplot(1, 2, 1)
    # "bo" 파란점
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # "b" 파란선
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(e_titles[i] + ' Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(1, len(epochs) + 1))  # 에포크를 정수로 표시
    plt.legend()

    plt.subplot(1, 2, 2)
    # "bo" 파란점
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    # "b" 파란선
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title(e_titles[i] + ' Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, len(epochs) + 1))  # 에포크를 정수로 표시
    plt.legend()

    plt.tight_layout()
    plt.show()
```

# 참고 링크 및 코드 개선
```python
# validation set 30000건 분리
X_val = X_train[:30000]   
y_val = y_train[:30000]

# validation set을 제외한 나머지 120000건
partial_X_train = X_train[30000:]  
partial_y_train = y_train[30000:]

print(partial_X_train.shape)
print(partial_y_train.shape)
```
데이터셋의 크기가 약 150000개 이여서 validation set을 10000건 보다는 좀더 높여주는것이 좋아보입니다.

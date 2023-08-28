# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 황인준
- 리뷰어 :황규빈


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 넵, 데이터 로드 및 전처리, Sentencepiece 토큰화, 모델 설계, 평가까지 작업 프로세스가 잘 이루어져 있습니다.
  > 하이퍼파라미터를 변경해 가면서 여러 작업을 하는게 인상 깊게 보았습니다.
  > 이후 데이터 시각화를 통해 확인하는점 좋았습니다
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  >
  ```python
  def preprocess_sentence(sentence):
    # 모든 입력을 소문자로 변환
    sentence = sentence.lower()
    # 알파벳, 문장부호, 한글만 남기고 모두 제거
    sentence = re.sub(r"[^a-zA-Z가-힣?.!,]+", " ", sentence)
    # 문장부호 양옆에 공백 추가
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    # 문장 앞뒤의 불필요한 공백 제거
    sentence = re.sub(r'[" "]+', " ", sentence)
    
    return sentence

  # Sentencepiece를 활용하여 학습한 tokenizer를 생성합니다.
  def generate_tokenizer(corpus, vocab_size, lang="ko", pad_id=0, bos_id=1, eos_id=2, unk_id=3):

    # corpus를 받아 txt 파일로 저장
    temp_file = os.getenv('HOME') + f'/aiffel/transformer/corpus_{lang}.txt'
    
    with open(temp_file, 'w') as f:
        for row in corpus:
            f.write(str(row) + '\n')
    
    # Sentencepiece
    spm.SentencePieceTrainer.Train(
        f'--input={temp_file} --pad_id={pad_id} --bos_id={bos_id} --eos_id={eos_id} \
        --unk_id={unk_id} --model_prefix=/aiffel/aiffel/transformer/spm_{lang} --vocab_size={vocab_size}'
    )
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(f'/aiffel/aiffel/transformer/spm_{lang}.model')

    return tokenizer
  ```
- [O] 코드가 에러를 유발할 가능성이 없나요?
  > 넵, 에러 유발 가능성이 없어 보입니다

- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 넵, 전체적인 프로세스로 볼 때 잘 이해하고 있다고 판단이 됩니다.
  
- [O] 코드가 간결한가요?
  > 
  ```python
      from tqdm import notebook  # Process 과정을 보기 위해
      import tensorflow as tf
      
      src_corpus = []
      tgt_corpus = []
      
      assert len(kor_corpus) == len(eng_corpus)
      
      # 토큰의 길이가 50 이하인 문장만 남깁니다. 
      for idx in notebook.tqdm(range(len(kor_corpus))):
          src = ko_tokenizer.EncodeAsIds(kor_corpus[idx])
          tgt = en_tokenizer.EncodeAsIds(eng_corpus[idx])
          
          if len(src) <= 50 and len(tgt) <= 50:
              src_corpus.append(src)
              tgt_corpus.append(tgt)
      
      
      # 패딩처리를 완료하여 학습용 데이터를 완성합니다. 
      enc_train = tf.keras.preprocessing.sequence.pad_sequences(src_corpus, padding='post')
      dec_train = tf.keras.preprocessing.sequence.pad_sequences(tgt_corpus, padding='post')
  ```
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

```

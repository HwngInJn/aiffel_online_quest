# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 황인준
- 리뷰어 : 김민식


# PRT(Peer Review Template)
- [:large_blue_circle:]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - `1. 챗봇 훈련데이터 전처리 과정이 체계적으로 진행되었는가?` &rarr; OK
    > 다양한 Augmenting을 보여주었고 최종적으로 통합하여 학습이 진행되는 과정을 보여주었다.
    ```python
    # questions와 answers에 대해 lexical_sub 함수로 증강
    questions_augmented = [lexical_sub(question, wv_model) for question in questions]
    answers_augmented = [lexical_sub(answer, wv_model) for answer in answers]
    
    # questions와 answers에 대해 lexical_sub 함수로 증강(한번더)
    questions_augmented2 = [lexical_sub(question, wv_model) for question in questions]
    answers_augmented2 = [lexical_sub(answer, wv_model) for answer in answers]
    
    # questions와 answers에 대해 lexical_sub_0_33 함수로 증강
    questions_augmented_0_33 = [lexical_sub_0_33(question, wv_model) for question in questions]
    answers_augmented_0_33 = [lexical_sub_0_33(answer, wv_model) for answer in answers]
    
    # questions와 answers에 대해 lexical_sub 함수로 증강
    questions_augmented_first = [lexical_sub_first_token(question, wv_model) for question in questions]
    answers_augmented_first = [lexical_sub_first_token(answer, wv_model) for answer in answers]
    
    # 원본 questions와 answers에 증강된 데이터 추가
    questions.extend(questions_augmented)
    questions.extend(questions_augmented2)
    questions.extend(questions_augmented_0_33)
    questions.extend(questions_augmented_first)
    
    answers.extend(answers_augmented)
    answers.extend(answers_augmented2)
    answers.extend(answers_augmented_0_33)
    answers.extend(answers_augmented_first)
    
    print('Total questions data size after augmentation:', len(questions))
    print('Total answers data size after augmentation:', len(answers))
    ```
    - `2. transformer 모델을 활용한 챗봇 모델이 과적합을 피해 안정적으로 훈련되었는가?` &rarr; OK
    > 추가 학습 과정을 보여주었고 결과로 Q&A가 잘 수행됨을 보았다.
    ```python
    ADDITIONAL_EPOCHS = 10  # 추가로 학습할 에포크 수

    for epoch in range(ADDITIONAL_EPOCHS):
        total_loss = 0
    
        idx_list = list(range(0, enc_train.shape[0], BATCH_SIZE))
        random.shuffle(idx_list)
        t = tqdm_notebook(idx_list)
    
        for (batch, idx) in enumerate(t):
            batch_loss, enc_attns, dec_attns, dec_enc_attns = \
            train_step(enc_train[idx:idx+BATCH_SIZE],
                       dec_train[idx:idx+BATCH_SIZE],
                       transformer,
                       optimizer)
            total_loss += batch_loss
    
            t.set_description_str('Additional Epoch %2d' % (epoch + 1))
            t.set_postfix_str('Loss %.4f' % (total_loss.numpy() / (batch + 1)))
    
        # 에포크가 끝날 때마다 예문으로 예측합니다.
        print("\nSample Predictions after additional epoch {}\n".format(epoch+1))
        
        for sentence in sample_sentences:
            prediction = evaluate(sentence, transformer, questions_tokenizer, answers_tokenizer)
            print(f'Input: {sentence}')
            print(f'Output: {prediction}\n')
    ```
    
- [:small_red_triangle:]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
  주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
> 같이 수업을 듣는 부분이라 가장 복잡하고 같은 문제는 없는 편이었지만, 전체적으로 주석 또는 doc string의 부재가 조금 아쉽습니다.

  
- [:large_blue_circle:]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
  ”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    > Augment에 필요한 모듈 사용을 위해 버전 이슈를 확인하였고 해당 부분을 해결함을 설명하였다.
    ```shell
    !pip install gensim==3.8.1 # wv을 쓰기위해서 gensim 버전 다운그레이드
    ```
  
- [:x:]  **4. 회고를 잘 작성했나요?**
> 전반적으로 과정 수행은 잘 되었으나 느낀점, 과정 설명 등의 내용이 아쉽다
    
- [:large_blue_circle:]  **5. 코드가 간결하고 효율적인가요?**
> 학습 모델 부분을 클래스로 모듈화 잘 하였습니다.
```python
class Transformer(tf.keras.Model):
    def __init__(self,
                    n_layers,
                    d_model,
                    n_heads,
                    d_ff,
                    src_vocab_size,
                    tgt_vocab_size,
                    pos_len,
                    dropout=0.2,
                    shared_fc=True,
                    shared_emb=False):
        super(Transformer, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)

        if shared_emb:
            self.enc_emb = self.dec_emb = \
            tf.keras.layers.Embedding(src_vocab_size, d_model)
        else:
            self.enc_emb = tf.keras.layers.Embedding(src_vocab_size, d_model)
            self.dec_emb = tf.keras.layers.Embedding(tgt_vocab_size, d_model)

        self.pos_encoding = positional_encoding(pos_len, d_model)
        self.do = tf.keras.layers.Dropout(dropout)

        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)

        self.fc = tf.keras.layers.Dense(tgt_vocab_size)

        self.shared_fc = shared_fc

        if shared_fc:
            self.fc.set_weights(tf.transpose(self.dec_emb.weights))

    def embedding(self, emb, x):
        seq_len = x.shape[1]

        out = emb(x)

        if self.shared_fc: out *= tf.math.sqrt(self.d_model)

        out += self.pos_encoding[np.newaxis, ...][:, :seq_len, :]
        out = self.do(out)

        return out


    def call(self, enc_in, dec_in, enc_mask, causality_mask, dec_mask):
        enc_in = self.embedding(self.enc_emb, enc_in)
        dec_in = self.embedding(self.dec_emb, dec_in)

        enc_out, enc_attns = self.encoder(enc_in, enc_mask)

        dec_out, dec_attns, dec_enc_attns = \
        self.decoder(dec_in, enc_out, causality_mask, dec_mask)

        logits = self.fc(dec_out)

        return logits, enc_attns, dec_attns, dec_enc_attns
```


# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```

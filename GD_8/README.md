# AIFFEL Campus Online Code Self Review 
- 코더 : 황인준
- 리뷰어 : 황인준


# PRT(Peer Review Template)
- [O]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - NSMC 데이터셋으로 fine-tuning 하여, 모델이 정상적으로 작동하는 것을 확인했고,
    - klue/bert-base 를 이용해서 90% 이상의 정확도를 확인했으나,
    - 기존 방식과 Bucketing task의 fine-tuning 시 연산 속도와 모델 성능 간의 trade-off 관계는 확인 하지 못함.
    
- [O]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
  주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
  - 선택하는 항목들에대해서 추후에 이해하기 쉽도록 주석을 써 놓았다.
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',             # 출력 결과들이 저장될 디렉토리
    num_train_epochs=2,                 # 학습할 총 에포크 수
    per_device_train_batch_size=270,    # 학습에 사용할 배치 크기
    per_device_eval_batch_size=270,     # 평가에 사용할 배치 크기
    evaluation_strategy="epoch",        # 에포크마다 평가
    save_strategy="epoch",              # 에포크마다 모델 저장
    save_total_limit=2,                 # 전체 저장된 모델의 최대 개수
    group_by_length=True                # 같은 길이의 입력 시퀀스끼리 그룹화
)
```
  
- [O]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
  ”새로운 시도 또는 추가 실험을 수행”해봤나요?**
  
```python
# 배치사이즈는 nvidia-smi를 통해서 가용 자원을 확인해가며 조정했다
train_dataloader = DataLoader(train_tensor_dataset, batch_size=270, shuffle=True)
val_dataloader = DataLoader(val_tensor_dataset, batch_size=270)
test_dataloader = DataLoader(test_tensor_dataset, batch_size=270)
```
  
- [O]  **4. 회고를 잘 작성했나요?**
```
두 방법 모두 동일한 성능을 냈다. 어떤 방법을 사용해도 상관 없을것 같다.

max_length를 60으로 작게 잡았기 때문에, group_by_length=True가 큰 의미를 갖지 못한것 같다.
```
    
- [△]  **5. 코드가 간결하고 효율적인가요?**

  - 캡슐화하지 않고 다 한 셀에 넣고 코드를 실행시켜서 재사용성이 떨어지고, 매번 새로운 시도를 할때마다 새로 정의 해줘야 했습니다.
```
from tqdm import tqdm

for epoch in range(3):
    print(f"\nStarting Epoch {epoch + 1}...")
    
    model.train()
    total_train_loss = 0
    print("Training...")
    

    for batch in tqdm(train_dataloader, desc="Training", total=len(train_dataloader)):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"\nEnd of Epoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}")
    
    model.eval()
    total_val_loss = 0
    print("Validating...")
    

    for batch in tqdm(val_dataloader, desc="Validation", total=len(val_dataloader)):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss
        total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"\nEnd of Epoch {epoch + 1}, Average Validation Loss: {avg_val_loss:.4f}\n")
    
    # Test accuracy after each epoch
    correct_predictions = 0
    total_predictions = 0

    for batch in tqdm(test_dataloader, desc="Testing", total=len(test_dataloader)):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
        
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    print(f"\nEnd of Epoch {epoch + 1}, Test Accuracy: {accuracy*100:.2f}%")
    
    # If the test accuracy exceeds 90%, stop training
    if accuracy * 100 > 90:
        print("\nReached 90% accuracy. Stopping training!")
        break
```


# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```

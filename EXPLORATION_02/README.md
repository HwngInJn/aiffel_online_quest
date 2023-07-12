# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 황인준
- 리뷰어 : 서희원


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 주석 이외에도 추가적인 설명을 작성해주셔서 이해하기 쉬웠습니다.
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 네
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네 이해하시고 직접 수정을 진행하시면서 성능을 높이셨습니다.
  > train 과 test 데이터를 합쳐서 코드의 길이를 줄이셨습니다.
```python
def my_GridSearch(models, train, y, param_grid, verbose=2, n_jobs=5):
    results = pd.DataFrame()
    for model in models:
        model_name = model.__class__.__name__
        grid_model = GridSearchCV(model, param_grid=param_grid, \
                                  scoring='neg_mean_squared_error', \
                                  cv=5, verbose=verbose, n_jobs=n_jobs)
        grid_model.fit(train, y)

        params = grid_model.cv_results_['params']
        score = grid_model.cv_results_['mean_test_score']
        
        tmp = pd.DataFrame(params)
        tmp['score'] = score
        tmp['RMSLE'] = np.sqrt(-1 * score)
        tmp['model'] = model_name
        results = pd.concat([results, tmp])

    results_sorted = results.sort_values(by=['RMSLE'])

    return results_sorted
```
- [X] 코드가 간결한가요?
  > train 과 test 데이터를 합쳐서 코드의 길이를 줄이셨습니다.
```python
train_len = len(data)
data = pd.concat((data, sub), axis=0)
```

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.

# 참고 링크 및 코드 개선

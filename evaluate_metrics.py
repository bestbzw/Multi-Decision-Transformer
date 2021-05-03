def calc_acc(predict,gold):
    right = 0.
    for p,g in zip(predict,gold):
        if type(g) == list and p in g:
            right += 1
        elif p == g:
            right += 1
    return right/len(predict)
def evaluate(predict,predicts,gold,metrics):
    results = []
    for i in range(len(predicts)):
        results.append(evaluate_one(predicts[i],gold,metrics))
    result = evaluate_one(predict,gold,metrics)
    return results,result
def evaluate_one(predict,gold,metrics):
    if metrics.lower() == "acc" or metrics.lower() == "accuracy":
        return calc_acc(predict,gold)

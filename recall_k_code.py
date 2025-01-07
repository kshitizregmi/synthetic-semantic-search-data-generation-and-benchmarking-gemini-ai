# recall@k function
def recall(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = round(len(act_set & pred_set) / float(len(act_set)), 2)
    return result


actual = ["2", "4", "5", "7"]
predicted = ["1", "2", "3", "4", "5", "6", "7", "8"]
for k in range(1, 9):
    print(f"Recall@{k} = {recall(actual, predicted, k)}")
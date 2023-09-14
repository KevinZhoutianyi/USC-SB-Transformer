def accuracy_score(all_labels, all_preds):
    #TODO

    correct_count = 0

    for label, pred in zip(all_labels, all_preds):
        if label == pred:
            correct_count += 1
    
    return correct_count/len(all_labels)
import numpy as np

def calc_auc(arr1, arr2, per):
    c_arr = np.hstack((arr1, arr2))
    logit = np.zeros(shape=c_arr.shape)
    for idx, i in enumerate(c_arr):
        #     print(prob_clean.shape, p1[1].shape)
        #     length = len(p1[1][p1[1]<=i])
        # #     print(p[1])
        #     probs_clean = prob_clean[length-2]
        # #     print(p[1], i)
        # #     print(length)
        # #     print(prob_clean)
        #     rand_no = np.random.random_sample()
        if i <= per:
            logit[idx] = 0
        else:
            logit[idx] = 1

    #     if adv_out == 1:
    #         count += 1
    #     if prob_clean[i > p[1] and i < p[1]] : #> prob_adv[p1[1] == i]:
    #         count+=1
    # print(count)

    size = arr1.shape

    target = np.hstack((np.zeros(shape=size), np.ones(shape=size)))
    from sklearn.metrics import roc_auc_score

    return (roc_auc_score(target, logit))

def list_to_np_arr(al):
    list_clean = []
    for i in al:
        tuple_d = i[0]
        for j in tuple_d:
            list_clean.append(j.item())

    arr1 = np.array(list_clean)

    return arr1
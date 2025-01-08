# coding: utf-8
def partition(arr, l, r):
    x = arr[r]
    i = l
    for j in range(l, r):

        if arr[j] <= x:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    arr[i], arr[r] = arr[r], arr[i]
    return i


def QuickSelectWithLR(array, left, right, k):
    if 0 < k <= right - left + 1:
        index = partition(array, left, right)
        if index - left == k - 1:
            return array[index]
        if index - left > k - 1:
            return QuickSelectWithLR(array, left, index - 1, k)
        return QuickSelectWithLR(array, index + 1, right,
                           k - index + left - 1)
def QuickSelect(array,k):
    return QuickSelectWithLR(array,0,len(array)-1,k)
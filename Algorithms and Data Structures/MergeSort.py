# # Use this algorithm to investigate about the divide and conquer method.
# # To use the divide and conquer, first, we would like to DIVIDE the problem into smaller problem (remember 
# # that all the problem and sub-problems share an identical in pattern) then we would CONQUER in terms of
# # manipulating the parameters and merge the solutions.

# Define how the sub-problem works
def merge(array, left, mid, right): 
    leftArray = array[left: mid + 1]
    rightArray = array[mid + 1: right + 1]

    indexOfSubArrayLeft, indexOfSubArrayRight, indexOfMergedArray = 0, 0, 0

    # Compare elements from sub-array to original array, update the original if there is a change.
    while (indexOfSubArrayLeft < len(leftArray)) and (indexOfSubArrayRight < len(rightArray)):
        if(leftArray[indexOfSubArrayLeft] <= rightArray[indexOfSubArrayRight]):
            array[indexOfMergedArray] = leftArray[indexOfSubArrayLeft]
            indexOfSubArrayLeft += 1
        else:
            array[indexOfMergedArray] = rightArray[indexOfSubArrayRight]
            indexOfSubArrayRight += 1

        indexOfMergedArray += 1

    # Copy the remain elements to the original array.
    while(indexOfSubArrayLeft < len(leftArray)):
        array[indexOfMergedArray] = leftArray[indexOfSubArrayLeft]
        indexOfMergedArray += 1
        indexOfSubArrayLeft += 1

    while(indexOfSubArrayRight < len(rightArray)):
        array[indexOfMergedArray] = rightArray[indexOfSubArrayRight]
        indexOfMergedArray += 1
        indexOfSubArrayRight += 1

# Start the DIVIDE and CONQUER
def mergeSort(array, left, right):
    if left >= right:
        return
    mid = (left + right)//2
    # DIVIDE
    mergeSort(array, left, mid)
    mergeSort(array, mid + 1, right)
    # CONQUER
    merge(array, left, mid, right)



arr = [12, 11, 13, 5, 6, 7]

mergeSort(arr, 0, len(arr) - 1)

print(arr)
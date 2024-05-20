"""
This is a "Lomuto" partition scheme, which is one of the most efficient and widely used.

Here's an explanation of how it works:

    1. If the length of the array is 0 or 1, return the original array (since it's already sorted).
    2. Choose the first element of the array as the pivot.
    3. Partition the rest of the array into two lists: less and greater. less contains all elements that are less than or equal to the pivot, and greater contains all elements that are greater than the pivot.
    4. Recursively apply the quicksort algorithm to less and greater.
    5. Combine the results of the recursive calls with the pivot element in the middle.
"""

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    less = [x for x in arr[1:] if x <= pivot]
    greater = [x for x in arr[1:] if x > pivot]
    return quicksort(less) + [pivot] + quicksort(greater)


arr = [3, 6, 1, 8, 2, 4]
arr = quicksort(arr)
print(arr)  # [1, 2, 3, 4, 6, 8]
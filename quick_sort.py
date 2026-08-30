def quick_sort(arr):
    """快速排序算法"""
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x <= pivot]
    right = [x for x in arr[1:] if x > pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)


if __name__ == "__main__":
    test_array = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    result = quick_sort(test_array)
    print(f"原始数组: {test_array}")
    print(f"排序结果: {result}")

    # 验证正确性
    assert result == sorted(test_array), "排序结果不正确!"
    assert result == [1, 1, 2, 3, 3, 4, 5, 5, 6, 9], "排序结果与预期不符!"

    # 边界情况测试
    assert quick_sort([]) == []
    assert quick_sort([1]) == [1]
    assert quick_sort([2, 1]) == [1, 2]
    assert quick_sort([3, 3, 3]) == [3, 3, 3]
    assert quick_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]

    print("所有测试通过，快速排序运行正确!")

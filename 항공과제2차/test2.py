def merge_time(lst, indices):
    indices.sort(reverse=True)  # 인덱스를 큰 값부터 순서대로 정렬
    for idx in indices:
        if idx < len(lst) - 1 and idx + 1 < len(lst) and idx+1 in indices and lst[idx] == lst[idx+1]:
            del lst[idx]
    return lst

lst = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
indices = [0, 1, 4, 5]

merged_lst = merge_time(lst, indices)
print(merged_lst)
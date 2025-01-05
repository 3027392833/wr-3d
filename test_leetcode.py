def lengthOfLongestSubstring( s: str) -> int:
    n = len(s)
    start = end = 0
    ans = list()
    result = len(ans)
    while end < n:
        if s[end] not in ans:
            ans.append(s[end])
            result = max(result, len(ans))
        else:
            start += 1
        end += 1
    return result


print(lengthOfLongestSubstring('pwwkew'))
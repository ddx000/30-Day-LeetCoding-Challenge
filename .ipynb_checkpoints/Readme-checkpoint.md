# 30-Day LeetCoding Challenge
- This 30-Day LeetCoding Challenge starts from April 1st, 00:00 AM and end on April 30th, 2020.
- Participants need to submit the solution every day before the end of the day.
- Because of the COVID-19, lots of people stay at home, so it's a great time for practicing Data Structure and Algorithms.


# Day 1 [Single Number](https://leetcode.com/problems/single-number/)
- 從[1,1,2,2,4,5,5]中找出唯一的那個
- 解法1: 用dic存看過的 再看過一次就刪除 最後就會只剩一個
- 解法2: Bit Operation XOR:
    - a^a = 0
    - a^0 = a


```python
def singleNumber(nums):
    #hashtable
    dic = {}
    for i in nums:
        if dic.get(i): #get不到會返回default None
            del dic[i] #刪去比pop快一些
        else:
            dic[i] = 1
    return dic.popitem()[0]
    #Time:O(n), Space:O(n)
    
    
    #Bit Operation , XOR
    res = 0
    for i in nums:
        res^=i
    return (res)
    #Time:O(n), Space:O(1)
```

# Day 2 [Happy Number](https://leetcode.com/problems/happy-number/)
- 運用dic存看過的去解 有無限loop return false


```python
class Solution:
    def isHappy(self, n: int) -> bool:
        self.repeat = {}
        return self.helper(n)
             
    def helper(self,num):
        if num in self.repeat:
            return False
        if num ==1:
            return True
        
        self.repeat[num] = True
        res = 0
        while num:
            res += (num%10)**2
            num = num//10
        return self.helper(res)
    
#Time:O(n), Space:O(n)
```

# Day3 [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)
思路: 可以用暴力解 但是太久 可以用DP 紀錄之前爬過的Array 


```python
class Solution:
    def maxSubArray(self, nums):
        dp = [0]* len(nums) #DP定義為到目前為止 最大sum的subarray
        dp[0] = nums[0]
        for i in range(1,len(nums)):
            dp[i] = max(dp[i-1]+nums[i],nums[i])
            # 選擇繼承前面的 或者完全放棄
        return max(dp)
```

# Day 4 [Move Zeroes](https://leetcode.com/problems/move-zeroes/)
- 題目要求把0移到後面，直接雙指針，走一遍，把非0的都swap掉
- 時間複雜度O(N)


```python
class Solution:
    def moveZeroes(self, nums):
        idx = 0
        for i in range(len(nums)):
            if nums[i] != 0 :
                nums[i], nums[idx] = nums[idx], nums[i]
                idx+=1
```

# Day5  [Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)
- 這題是這系列題裡面最簡單的一題，基本上其他系列都要用DP
- 取一階微分後把正數加起來就好


```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for idx in range(len(prices)-1):
            diff = prices[idx+1]-prices[idx]
            if diff > 0:
                profit += diff
        return profit
```

# Day 6 [groupAnagrams](https://leetcode.com/problems/group-anagrams/)

- 基本上排序後轉成tuple(list不行 因為mutable), 就會是唯一key
- 用哈希表去存就可以


```python
def groupAnagrams(strs):
    dic = {}
    for s in strs:
        key = tuple(sorted(list(s)))
        if dic.get(key):
            dic[key].append(s)
        else:
            dic[key] = [s]
    return list(dic.values())
groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
```

# Day 7 [Counting Elements](https://leetcode.com/explore/featured/card/30-day-leetcoding-challenge/528/week-1/3289/)

- set 再實現時內部是dict 所以查找的話 O(1)
- 時間複雜度為O(n)


```python
class Solution:
    def countElements(self, arr):
        SET = set(arr)
        res = 0
        for i in arr:
            if i+1 in SET:
                res +=1
        return res

```

# Day 8 [Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)
- 基本上就是快慢指針的簡單題


```python
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
```

# Day 9 [Backspace String Compare](https://leetcode.com/problems/backspace-string-compare/)
- 基本上就是簡單的字串處理，用一個Stack去pop，走兩遍字串就解決
- follow up要求O(1) Space, 就要額外用雙指針+倒序去解，會比較複雜
- 時間複雜度O(N), 空間複雜度O(N)


```python
class Solution:
    def backspaceCompare(self, S: str, T: str) -> bool:
        def _RemoveBS(string):
            lst=[]
            for s in string:
                if s=='#' and lst:
                    lst.pop()
                elif s=='#':
                    continue
                else:
                    lst.append(s)
            return lst
        return True if _RemoveBS(S) == _RemoveBS(T) else False
```

# Day 10 [Min Stack](https://leetcode.com/problems/min-stack/)
- 設計一個物件導向的Class
- 用空間複雜度換取時間複雜度 紀錄多開一個list紀錄min


```python
class MinStack:

    def __init__(self):
        self.stack = []
        self.min   = []
        
    def push(self, x):
        self.stack.append(x)
        if not self.min:
            self.min.append(x)
        elif x <= self.min[-1]:
            self.min.append(x)
        
    def pop(self):
        val  = self.stack.pop()
        if val == self.min[-1]:
            self.min.pop()
        
    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min[-1]
```

# Day 11 [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)
- 這題稍微複雜一點 基本上就是用DFS Recursion
- DFS注意終止條件(沒有node return 0)
- return回去時 左右取一條最大的就行


```python
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.maxD = 0
        self.dfs(root)
        
        return self.maxD
    
    def dfs(self,node):
        if not node:
            return 0
        
        left = self.dfs(node.left)
        right = self.dfs(node.right)
        self.maxD = max(left+right , self.maxD)
        return max(left,right) + 1
```

# Day 12 [Last Stone Weight](https://leetcode.com/problems/last-stone-weight/)
- 這題直覺地解法是每次都sort(nlogn), 整體時間複雜度是n*nlogn, 很久
- 可以用優先隊列去解 時間複雜度是n
- 我有特別寫一篇[Medium](https://medium.com/@jimmy_huang/heaqp-in-python-leetcode-630dfe4773d5)來述說Priority Queue的寫法


```python
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        def helper(stones):
            if len(stones) == 1:
                return stones[0]
            elif len(stones) == 0:
                return 0
            stones.sort() #nlogn
            max1  = stones.pop()
            max2  = stones.pop()
            if max1 != max2:
                stones.append(max1-max2) 
            return helper(stones)
        return helper(stones)
        
```

# Day 13 [contiguous-array](https://leetcode.com/problems/contiguous-array/)
- 這題大概是我這13天唯一一題沒解出來的... 頗有難度  
基本上的概念就是用hash table記憶走過的地方 因為可能有offset, 所以+1-1會回到那個offset  
所以要把所有點記憶下來，從終點扣到起點，求最長距離


```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        
        count = 0
        maxlength = 0
        dic = {0:0}
        
        for idx, num in enumerate(nums,1):
            if num == 0:
                count -=1
            elif num ==1:
                count +=1
                
            if count not in dic:
                dic[count] = idx
            else:
                maxlength = max( idx - dic[count] , maxlength)
            #print(dic)
        return maxlength
```

# Day 14 [Perform String Shifts](https://leetcode.com/explore/featured/card/30-day-leetcoding-challenge/529/week-2/3299/)
- 簡單的字串處理, 字串處理就是python的強項... 秒殺


```python
class Solution:
    def stringShift(self, s: str, shift: List[List[int]]) -> str:
        
        for i in shift:
            if i[0] == 0: #shift left,postive
                move = i[1] 
            else: #shift right, negative
                move = -i[1]
            s = s[move:]+s[:move]
        return s
```


# Day 15[Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)
- 基本上這題如果用除法 就超快，但是題目限定不能用除法，也要O(n)+ no extra space
- 這題五個月前寫過一次，這次竟然再看到也想不出來... 真有魔性XD
- A*B*C*D = DOT 基本上這種題目 就是要拆成兩個array，A,BCD 這樣然後兩次迴圈 一次從左到右 一次從右到左


```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        
        dot = 1
        res = [dot]
        
        # first loop O(n): left to right
        for idx in range(len(nums)-1):
            dot = dot * nums[idx]
            res.append(dot)

        # second loop O(n): r to l
        dot = 1
        for idx in range(len(nums)-1,0,-1):
            dot = dot * nums[idx]
            res[idx-1] = res[idx-1]*dot
        
        #Time O(n)
        #Space O(1)
        
        return res


```
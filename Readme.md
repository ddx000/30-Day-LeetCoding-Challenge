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

# Day 5  [Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)
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


# Day 15 [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)
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


# Day 16 [Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/)
思路1: 字串題+parentness--> stack  
此題關鍵是*可以當作三種符號 如果枚舉的話就是樹狀結構，就是都做一次就是DFS 不知道會不會超時
以下為代碼，後來跑測試案例到52/58時，真的TLE了

```python
class Solution:
    def checkValidString(self, s: str) -> bool:
        return self.dfs([],s)
    
    def dfs(self,stack,s):
        #end condition    
        if  len(s) == 0:
            if len(stack) == 0:
                return True
            else:
                return False
        if s[0] == '(':
            return self.dfs(stack+[s[0]], s[1:])    
            
        elif s[0] == ')':
            if stack:
                return self.dfs(stack[:-1], s[1:])
            else:
                return False
            
        else:
            return self.dfs(stack[:-1], s[1:]) or self.dfs(stack+[s[0]] , s[1:]) or self.dfs(stack, s[1:])
```

腦袋中暫時沒其他解法，只好去討論區看看別人的思路  
另外一個解法是 從頭到尾 用兩個變數存count min 和count max 意思就是到尾巴時 還需要幾個右括號  
count min就是紀錄最少需求 我們把星號都當成右括號 ")"   
count max就是紀錄最多需求 我們把都當成左括號 "("  
一個迴圈  
cmin ++ if "(" else --, if cmin<0, set it to 0  
cmax ++ if "(" or "*" else --, cmax must>0 otherwise return false  

```python
class Solution:
    def checkValidString(self, s: str) -> bool:
        cmin = 0
        cmax = 0
        for i in s: 
            if i =="(":
                cmin += 1
                cmax += 1
                
            elif i == "*":
                cmin -= 1
                cmax += 1    
                
            elif i == ")":
                cmin -=1
                cmax -=1
                
            if cmax<0:
                return False
            if cmin<0:
                cmin = 0
        return True if cmin==0 else False

```


# Day 17 [Numbers of Islands](https://leetcode.com/problems/number-of-islands/)

剛好上個月才寫過這題，這次二刷一次就AC 感覺思路清晰了許多]
- 思路：題目說詢問島嶼數量，基本這種左右走的就是DFS沒錯，所以就兩個迴圈mn,iterate all elements in matrix, 遇到1就去DFS(此時CNT+1), 注意凡走過必留下痕跡，務必把1設為其他數字(有點像是走過就翻面的感覺)，不然會無窮迴圈
- 提醒: 矩陣題型 Corner case(超出邊界) 很重要, 一些變數如果是dfs和main function 共用的話，就要self.classvariable
- 技巧1: 走過就翻面(同448. Find All Numbers Disappeared in an Array) 可以另開一個記憶體arr 但是這樣浪費空間 就直接inplace改
- 技巧2: 迴圈dfs，通常這種都是暴力解的安全牌，同437. Path Sum III，要優化時間複雜度就只能用用看dp可以記住什麼不要重複計算的東西

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
      
        self.cnt = 0
        self.grid = grid
        
        self.m = len(grid)
        if self.m==0:
            return 0
        self.n = len(grid[0])
        
        
        for i in range(self.m):
            for j in range(self.n):
                if self.grid[i][j]=='1': # ==1
                    self.cnt += 1
                    self.dfs(i,j)
                    
        return self.cnt
        
        
    def dfs(self,i,j):
        
        if self.grid[i][j] != '1':
            return 
            
        self.grid[i][j] = '-1'
        if i+1 <= self.m-1:
            self.dfs(i+1,j)
        if i-1 >= 0:
            self.dfs(i-1,j)
        if j+1 <= self.n-1:
            self.dfs(i,j+1)
        if j-1 >= 0:
            self.dfs(i,j-1)
        
        
        return self.cnt
```


# Day 18 [Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)

DP的題目，注意矩陣題都要小心corner case
基本上把邊界先跑過一遍比較安全(就不用太多邊界處理)
不用另外開矩陣，inplace改可以省空間


```python=

def minPathSum(grid):
    m = len(grid)
    if m == 0:
        return 0
    n = len(grid[0])

    for i in range(1,m):
        grid[i][0] = grid[i-1][0] + grid[i][0]
    for j in range(1,n):
        grid[0][j] = grid[0][j-1] + grid[0][j]

    for i in range(1,m):
        for j in range(1,n):
            grid[i][j] = min(grid[i-1][j],grid[i][j-1])+grid[i][j] 
            
    return grid[-1][-1]


```

# Day 19 Search in Rotated Sorted Array

基本上這題一開始的觀念很tricky, 要懂得用中位數和尾巴去判斷前後哪段是有序，因為題目只有一個PIVOT點，所以一定會有一段是有序的，有序的話，馬上就能判斷target是不是在裡面，基本上就是二分查找  
另外一個要注意的點就是，while left<=left這邊要記得加等於，可以參考下面這篇   -->[你真的会二分查找吗](https://blog.csdn.net/int64ago/article/details/7425727/)
- 1. while >=
- 2. left= mid+1, right=mid-1
- 3. mid可以>target或者>=, 大於等於就是YES LEFT



```python=
class Solution:
    def search(self, lst: List[int], target: int) -> int:

        if lst == None:
            return -1
        left = 0
        right = len(lst) -1
        while left<=right:
            mid = (left+right)//2
            if target == lst[mid]:
                return mid
            #右半邊有序
            if lst[mid] < lst[right]:
                if lst[mid]<target<=lst[right]:
                    #刪左邊，留右邊
                    left = mid +1 
                else:
                    #刪右邊，留左邊
                    right = mid - 1

            #左半邊有序
            else:
                if lst[left]<=target<lst[mid]:
                    #刪右邊，留左邊
                    right = mid - 1
                else:
                    #刪左邊，留右邊
                    left = mid +1 
        return -1
```


# Day 20 Build a Binary Search Tree from a Preorder Sequence


Tree題基本上都要一個helper
二元樹就是self.left= helper(...)這樣

開頭寫終止條件return None
最後寫return node把答案回傳回去

題目都在同一個list上作創作，所以基本上不用額外copy記憶體，用idx傳下去

基本二分條件就是看大小，這邊用個迴圈就能決定怎麼拆(因為題目進來是preorder)



```python

class Solution:
    def bstFromPreorder(self, preorder):
        def helper(start,end):
            if start == end:
                return None
            idx = start
            while idx < end:
                if preorder[idx]>preorder[start]:
                    print(preorder[idx],idx)
                    break
                idx+=1

            node = TreeNode(preorder[start])
            node.left = helper(start+1,idx)
            node.right = helper(idx,end)

            return node

        return helper(0,len(preorder))


```

# Day 21 Leftmost Column with at Least a One

這題是互動題，有點像打靶，但是打靶次數有限制，矩陣題比較混淆的是題目給的x是對應到矩陣的m(所以不是一般肉眼見到的XD)  
基本上一定是二分法啦，題目的矩陣同一row是先0後1，所以一定一開始是打一整column的值，全0代表找太左邊了，有一個1的話，代表還是要往左邊找，這樣不斷縮小邊界，當然這時候就不用一次打一整列col了，因為你過去打到0的地方，你往左打，肯定也是0  
(所以這邊用一個newrow去存他)  
大概就是二分左右逼近就搞定了(記得 打到1就把那一COL號碼存起來，最後一定會打到最近的)

```python

class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        row , col = binaryMatrix.dimensions()[0] , binaryMatrix.dimensions()[1]
        self.rows = list(range(row))
        left,right = 0,col-1
        # print(row,col)
        self.res = -1
        
        while left<=right:
            mid = (left+right)//2
            #print(mid,'mid')
            lst = []
            newrow =[]
            for y in self.rows:
                # print(y,mid)
                val = binaryMatrix.get(y,mid)
                lst.append(val)
                if val ==1:
                    newrow.append(y)
            
            # print(lst,'lst')
            if lst == [0]*row: #全零 就是代表找太前面了
                left = mid + 1
            else: #有一個1 !!
                # print('find1')
                self.rows = newrow
                self.res = mid
                right = mid -1
            # print(left,right)
        return self.res
```

# Day 22 [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)

這題真的血淚，因為兩周前做過，結果這次再寫第二遍
**我還是想不出來HASHMAP的做法 真的太奇葩了**

只能先用prefix sum 試試看 

因為可以節省重複的計算
然後記得prefix前面+[0]
[1,1,2,1]
prefix
[0,1,2,4,5]
sumi到j(含)總和sum[0:3] = prefix[j+1]-prefix[i]

結果馬上TLE.........

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        sums = 0
        prefix = [0]
        # n
        for i in nums:
            sums += i
            prefix.append(sums)

        res = 0

        # n^2
        for i in range(len(nums)):
            for j in range(i,len(nums)):
                if k == prefix[j+1] - prefix[i]:
                    res+=1
        return res
```

只好重新定義題目(舉例/相似/簡化/DP/列舉 這五種方法的"簡化法")
題目是問子數組的和為K
大概像這樣
[1,3,5,[SUM = 13 =K ],8,1,2]
但是 這樣就是求
1+3+5 == (21 - 13)
         SUM - K
       ^i         ^j

sum(0,i) = sum(0,j) - k
題目重新定義，有多少從頭開始得子數組 = 特定值呢

這邊再用一個hashmap 紀錄有幾種可能
k =和 v=可能 所以一開始一定有(0,1)

開始走一遍:
把rolling sum加進去dic裡面(初始值是(0,1))
res += dic.get(sum-k,0)
為什麼會去dic找sum-k的概念有點是這樣

以同樣迴圈走到star處 開始往前找的時候，dictionary出現幾個sum-k就代表k有幾種可能
(同一個尾巴star處 往前推伸)
(sum-k.......)(  k)*
(sum-k...)(      k)*

```python
class Solution:
    def subarraySum(self, nums, k):
        dic ={0:1} #形成0有一種可能
        sums = 0
        res = 0
        for i in nums:
            sums += i
            res  += dic.get(sums-k,0)
            dic[sums] = dic.get(sums,0)+1
        return res
```

# Day 23 [Bitwise AND of Numbers Range](https://leetcode.com/problems/bitwise-and-of-numbers-range/)
題目要找一定範圍的BITWISE AND，這時候用舉例法

26: 11010
29: 11101
基本上只有前兩位一樣，後面只要有出現過0的話都是0，那我們把它>>3次 把他們變成11=11 這樣就是重疊的地方了

```python
class Solution:
    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        cnt =0
        while m!=n:
            m = m>>1
            n = n>>1
            cnt+=1
        return m<<cnt
```

# Day 24 [LRU Cache](https://leetcode.com/problems/lru-cache/)
面試出現機率超高的LRU Cache... 又再遇到了  
這次寫有個地方沒注意到，就是dict的key值一定不能讓他重複!!!  
因為重複的話，dict會直接覆蓋過去，但是LRU結構卻會不斷增長  

如何選取數據結構?  
1.  
題目要求的get功能+O(1)就要聯想到hashmap  
2.  
接下來就是put新資料後，要把最少用的資料刪除  
常常刪除，就不可能用array，要用linked list  
而linked list在時間複雜度上，雙向的可以達到任意NODE del都達到O(1)  
由於insert 都是在頭部或尾部插入，所以時間複雜度也是O(1)  
3.  
結合haspmap+double linked list  
可以在O(1) access到該個node，del的話也可以在O(1)的時間內  
4.  
簡單來說，就是用空間換取時間複雜度的概念，hashmap記錄了node在哪裡，node本身又存了key,value和pre, next等指標  
```python

class Node:
    def __init__(self,k,v):
        # double pointer
        self.pre  = None
        self.next = None
        # key and value
        self.k = k
        self.v = v

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        print(capacity)
        # HASHMAP for quickly access where is the node
        #結構:key  :  value = node(k,v)
        self.dic = {}           

        # 一開始要初始化頭尾
        self.head = Node(0,0)
        self.tail = Node(0,0)
        self.head.next = self.tail
        self.tail.pre  = self.head

    def get(self, key: int) -> int:
        if key not in self.dic:
            return -1
        # 移除後，再加進去
        self._remove(self.dic[key])
        self._add(self.dic[key]) #加進尾巴
        # hashmap存的是整個node，現在只要該node的value
        return self.dic[key].v


    def put(self, key: int, value: int) -> None:
        # 務必要檢查重複的情況，因為重複的key在dictionary裡面會直接取代，
        # 但是後面檢查長度時，是用dic的長度去檢查，就會出現重複的值塞進去，dic一直取代，但是長度不變
        if key in self.dic:
            self._remove(self.dic[key])
        newnode = Node(key,value)
        self._add(newnode)
        self.dic[key] = newnode
        if len(self.dic)>self.capacity:
            # 最少使用的就在頭部刪除
            # 字典裏面也要去掉
            del self.dic[self.head.next.k]
            self._remove(self.head.next)


    def _add(self,node):
        # 新進來，或剛使用的，就加進尾部
        self.tail.pre.next = node
        node.pre = self.tail.pre
        self.tail.pre = node
        node.next = self.tail

    def _remove(self, remove_node):
        # 把指標改掉後，剩下的給GC回收
        remove_node.pre.next = remove_node.next
        remove_node.next.pre = remove_node.pre
```

# Day 25 [Jump game](https://leetcode.com/problems/jump-game/)


```python
class Solution(object):
    def canJump(self, nums):
        """
        這題是Greedy算法
        我只要算我能走的最遠的地方，是不是能到終點就好
        """
        maxarrived = 0
        for idx, i in enumerate(nums):
            if idx <= maxarrived:
                maxarrived = max(idx+i,maxarrived)
            
        return True if maxarrived>= len(nums)-1 else False
            
```

# Day 26 Longest Common Subsequence

本題hint幫了我大忙  
Try dynamic programming. DP[i][j] represents the longest common subsequence of text1[0 ... i] & text2[0 ... j].  
轉移方程要注意
錯誤寫法-->(因為這樣會有一個字符重複的問題)  
這題之後會補上圖解  
                # add = 1 if text1[i] == text2[j] else 0   
                # dp[i][j] = max(dp[i][j-1],dp[i-1][j],dp[i-1][j-1]) + add   

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        
        
        I = len(text1)
        J = len(text2)
        #dp = [[0]*J]*I 錯誤寫法!!!
        dp = [[0 for _ in range(J+1)] for _ in range(I+1)]
        #這邊要多開一個陣列，就不要填，避免邊界問題
            
        for i in range(1,I+1):
            for j in range(1,J+1):

                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] +1
                    
                else:
                    dp[i][j] = max(dp[i][j-1],dp[i-1][j]) #絕對不可以偷懶用-1法

        return dp[-1][-1]
```



# Day 27 Maximal Square

DP題，很不好想到
額外開一個dp陣列，儲存以該點位為右下角，往左上延伸時，可以獲得的最大值  
if matrix[m][n]==1: dp[m][n] = min(dp[m-1][n],dp[m][n-1],dp[m-1][n-1])+1  
這個轉移方程畫個圖才有辦法了解...

![](https://i.imgur.com/c0pGrK3.png)
注意corner case 所以在創dp時 會多開一個欄位(這樣左上角才不會超出邊界)  

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:

        m = len(matrix)
        if m==0:
            return 0
        n = len(matrix[0])
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]

        maxvalue = 0
        # dp有多開一個欄位
        for i in range(m):
            for j in range(n):
                if matrix[i][j]== "1":
                    dp[i+1][j+1] = min(dp[i][j+1],dp[i+1][j],dp[i][j]) + 1
                    maxvalue = max(dp[i+1][j+1],maxvalue)
        return maxvalue**2
```

# Day 28 FirstUnique


- TLE的解法，add 是O(1)，search是O(n)
```python
class FirstUnique:

    def __init__(self, nums: List[int]):
        self.lst = nums
        self.dic = {}
        for i in self.lst:
            self.dic[i] = self.dic.get(i,0)+1

    def showFirstUnique(self) -> int:
        # O(n)
        for i in self.lst:
            if self.dic[i] == 1:
                return i
        return -1


    def add(self, value: int) -> None:
        # O(1)
        self.lst.append(value)
        self.dic[value] = self.dic.get(value,0) + 1
```
由於上面這解法TLE，出問題的應該是showFirstUnique這個複雜度必須降到log(n)  
這題有點像是LRU Cache，可以試試看double linked list(為何選擇-->移除新增快)+ hashmap(直接跳選到該node)  
基本上我就是要insert進來後，我可以在O(1)的時間內刪掉重複的node，永遠return linked list的head就可以  
不過這邊練習一下python特有的ordered dict來試試看  

- initial花了最久的時間... 要處理一些重複的問題
- add時間複雜度O(1) show基本上也是O(1) (如果我沒計算錯的話XD)
- 記憶體開銷是比較大了點，會維護兩個hashmap(一個orderdict for 順序，另一個是set，處理一些一開始出現的情況)

```python

class FirstUnique:

    from collections import OrderedDict, Counter

    def __init__(self, nums: List[int]):
        temp_dic = OrderedDict(Counter(nums))
        self.dic = OrderedDict()
        self.set = set(nums)
        for k, v in temp_dic.items():
            if v == 1:
                self.dic[k] = v

    def showFirstUnique(self) -> int:
        if self.dic:
            #return list(self.dic)[0]
            return next(iter(self.dic.keys())) 
        else:
            return -1

    def add(self, value: int) -> None:
        if value in self.dic:
            del self.dic[value]
        elif value in self.set:
            pass
        else:
            self.dic[value] = 1

```
def insertion_sort(nums):
    if nums == []:
        return []
    else:
        for i in range(2,len(nums)+1):
            numb = len(nums)-i
            val1 = nums[numb]
            j = numb + 1
            while j<len(nums) and nums[j] > val1:
                nums[j-1]=nums[j]
                j = j + 1
            nums[j-1]=val1
        return nums


print(insertion_sort([2,3,12,9,21, 17,9 ,9 ,9 ]))


def merge(A,p,q,r):
    L = A[p:q]#python is neat
    R = A[q:r]
    L.append(float("inf"))
    R.append(float("inf"))
    i = 0
    j = 0
    for o in range(p,r):
        if L[i]<=R[j]:
            A[o]=L[i]
            i = i+1
        else:
            A[o]=R[j]
            j=j+1

A =[1,3,8,1,2,4]
merge(A,0,3,len(A))
print(A)
# def merge_mod(A,p,q,r):
#     L = A[p:q]#python is neat
#     R = A[q:r]
#     L_len =len(L)
#     R_len =len(R)
#     i = 0
#     j = 0
#     for o in range(p,r):
#         if R_len == j:
#             A[o:] = L[i:]
#             print("here",A[o])
#             break
#         if L_len == i:
#             A[o:] = R[j:]
#             break
#
#         if L[i]<=R[j]:
#             A[o]=L[i]
#             i = i+1
#         else:
#             A[o]=R[j]
#             j=j+1

#
# def merge_sort(A,p,r):
#     if p < r :
#         q = int((p+r)/2)
#         merge_sort(A,p,q)
#         merge_sort(A,q+1,r)
#         merge_mod(A,p,q,r)
#
# A = [7,2,0,3,1,6,1,7,4,734,62]
# merge_sort(A,0,len(A))
# print(A)


def max_cross(A,low,mid,high):
    left_sum = -float("inf")
    sum = 0
    max_left=None
    for i in range(mid, low - 1, -1):
         # i = mid - low + 1
        sum = sum+A[i]
        if sum > left_sum:
            left_sum = sum
            max_left = mid - i
    right_sum= -float("inf")
    sum = 0
    max_right = None
    for j in range(mid+1, high+1):
        sum =sum + A[j]
        if sum>right_sum:
            right_sum = sum
            max_right = j
    return max_left,max_right, left_sum+right_sum

def max_subarray(A, low, high):
    if low is None:
        low = 0
    if high is None:
        high = len(l) - 1
    if high == low: # could there be an empty list?
        return low,high, A[low-1]
    else:
        mid = (low+high)//2
        left_low,left_high,left_sum = max_subarray(A,low,mid)
        right_low,right_high,right_sum = max_subarray(A,mid+1,high)
        cross_low,cross_high,cross_sum=max_cross(A,low,mid,high)

        if left_sum >= right_sum and left_sum >= cross_sum:
            return left_low, left_high, left_sum
        elif right_sum >= left_sum and right_sum >= cross_sum:
            return right_low, right_high, right_sum
        else:
            return cross_low, cross_high, cross_sum

A = [-1,-2,-10,-20,-30,-50,-30]
print(int(len(A)/2))
print(len(A))
a,b,max =max_cross(A,1,2,3)
print("here",a,b,max)
print(sum(A))
a,b,sum = max_subarray(A,0,len(A)-1)
print("this is the sum", sum)


# def maxProfit(prices):
#     """
#     :type prices: List[int]
#     :rtype: int
#     """
#     difference = []
#     dif_val = 0
#     for i in range(len(prices) - 1):
#         dif = prices[i + 1] - prices[i]
#         if dif > 0:
#             dif_val += dif
#             if i == len(prices) - 2:
#                 difference.append(dif_val)
#         else:
#             if dif_val > 0:
#                 difference.append(dif_val)
#                 dif_val = 0
#     if len(difference) == 1:
#         return difference[0]
#     else:
#         difference = sorted(difference)
#         print(difference)
#         return difference[len(difference) - 1] + difference[len(difference) - 2]



def cw(n):
     return count_ways_mem(n, (n+1)*[-1])

def count_ways_mem(n,memo):
    if n<0:
        return 0
    if n == 0:
        return 1
    if memo[n]> -1:
        return memo[n]
    else:
        memo[n] = count_ways_mem(n - 3, memo) + count_ways_mem(n - 2, memo) + count_ways_mem(n - 1, memo)
        return memo[n]


print("This is cw", cw(509))



def count_sort(A, k):#do I know what the maximum number is
    C = [0]*(k+1)
    for i in A:
        C[i] += 1
    print(C[1],"d")

    for i in range(k):
        C[i] += C[i-1]
    B = [0]*(len(A)+1)

    for i in range(len(A)-1,0,-1):
        print("C[A[i]]", C[A[i]])
        print("A[i]", A[i])
        B[C[A[i]]-1]= A[i]

        C[A[i]] =C[A[i]]-1
    return B[1:]

print(count_sort([1,124,234,2,0,46,33,2], 300))


def merge(A, B, lastA_ind, lastB_ind):

    while lastB_ind >= -1:
        last_ind = lastA_ind + lastB_ind

        if A[lastA_ind] > B[lastB_ind]:
            A[last_ind] = A[lastA_ind]
            lastA_ind -= 1
        else:
            A[last_ind] = B[lastB_ind]
            lastB_ind -= 1


A = [1,3,3,None,None,None,None]
B = [1,2,4,5]
merge(A, B, 2, 3)
print(A)

def is_anagram(wordA, wordB):
    dictA ={}
    if len(wordA) == len(wordB):
        for i in wordA:
            if dictA.get(i) is None:
                dictA[i] =1
            else:
                dictA[i]+=1
        for i in wordB:
            if dictA.get(i) is not None and dictA.get(i)>0:
                dictA[i]-=1
            else:
                return False
        return True
    return False

print(is_anagram("listen", "silent" ))
print(sorted("listen"))


def reverse( x):
    """
    :type x: int
    :rtype: int
    """
    x = str(x)
    if x[0]=="-":
        x = x[1:]
        x = x[::-1]
        if int(x) < 2**31:
            return int("-"+x)
        return 0
    x = x[::-1]
    if int(x) < 2**31:
        return int(x)
    else:
        return 0
print(reverse(1563847412))

print("graphsssss ")


def findOrder(numCourses, prerequisites):
    """
    :type numCourses: int
    :type prerequisites: List[List[int]]
    :rtype: List[int]
    """
    graph = [[] for i in range(numCourses)]
    for v2, v1 in prerequisites:
        graph[v2].append(v1)
    print(graph)

    order = []
    visited = set()
    tempV = set()

    for i in range(numCourses):
        if not dfs(graph,i,visited,order,tempV):
            return []

    return order

def dfs(edge, node, visited, order,tempV):
    if node in tempV:
        return False
    if node in visited:
        return True

    visited.add(node)
    tempV.add(node)

    for nodes in edge[node]:
        if not dfs(edge, nodes,visited,order,tempV ):
            return False

    order.append(node)
    tempV.remove(node)
    return True



print("This is find", findOrder(4, [[1,0],[2,0],[3,1],[3,2]]))


print("neww2w")

def isRoute(n,edges,a,b):
    graph = [[] for i in range(n)]
    for v1,v2 in edges:
        graph[v1-1].append(v2-1)
    visited = set()
    return dfs(a-1,graph,b-1,visited)

def dfs(node, graph, b , visited):
    if node == b:
        return True
    visited.add(node)
    for i in graph[node]:
        if i not in visited:
            return dfs(i,graph, b, visited)
    return False
print(isRoute(4, [[1,2],[2,4],[2,3],[4,3]], 3,4 ))
class Node():
    def __init__(val=None):
        Node.val = val
        Node.right = None
        Node.left = None

def bst(A):
    if A:
        median_ind=len(A)//2
        node = Node()
        node.val = A[median_ind]
        node.left = bst(A[0:median_ind])
        node.right = bst(A[median_ind+1:len(A)])
        return node
    return None
tree = bst([1,2,3])
print(tree.val)
def print_Tree(tree):
    if tree:
        print(tree.val,"-->")
        print_Tree(tree.left)
        print_Tree(tree.right)

print_Tree(tree)


def iot(root):
	node_list = []
	if root:
		iot(root.left)
		node_list.append(root.val)
		iot(root.right)



def findPairs( nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    h = {}
    for i in nums:
        h[i] = True
    count = 0
    for i in nums:
        dif = abs(k - i)
        if h.get(dif):
            count += 1

    return count//2
print(findPairs([1,2,3,4,5],1))



def moveZeroes( nums):
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    A = nums
    if A == []:
        return A
    count = 0
    for i in range(len(A)):
        if A[i] == 0:
            count += 1
        else:
            A[i - count] = A[i]
    A[len(A)-count:] =[0]*count
    return  A
print(moveZeroes([0,1,0,3,12]))

def array_pref(A):
    j=1
    pref=[0]*len(A)
    for i in range(1,A):

        if A[j]==A[i]:
            pref[j] = pref[j-1]+1
            j += 1
        if A[j]!=A[i]:
            if A[pref[j-1]]==A[i]:
                pref[j]= pref[j-1]

# supper i
from collections import deque

def levelOrder( root):
    if not root: return []
    traversal_queue = deque([root])
    ans = []
    while traversal_queue:
        level_len = len(traversal_queue)
        level_nodes = []
        while level_len > 0:
            node = traversal_queue.popleft()
            level_nodes.append(node.val)
            if node.left:
                traversal_queue.append(node.left)
            if node.right:
                traversal_queue.append(node.right)
            level_len -= 1
        ans.append(level_nodes)
    return ans


def search( nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    n = len(nums)//2
    prev = len(nums)//2
    while n>=0 and n<len(nums):
        if nums[n]==target:
            return n
        if prev == n//2 :
            return -1
        if nums[n]<target:
            n += (n//2)
        if nums[n]<target:
            n = n//2
    return -1






def minDistance( word1, word2):
    if word1 == "":
        return len(word2)
    if word2 == "":
        return len(word1)
    if word1[-1]==word2[-1]:
        cost = 0
    else:
        cost = 1
    print("here",word1)
    print("not",word2)
    res = min(minDistance(word1[:-1],word2)+1,
              minDistance(word1, word2[:-1])+1,
              minDistance(word1[:-1], word2[:-1])+cost)
    return res

# print(.minDistance("prosperity","properties"))


def LD(s, t):
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    if s[-1] == t[-1]:
        cost = 0
    else:
        cost = 1
    res = min([LD(s[:-1], t)+1,
               LD(s, t[:-1])+1,
               LD(s[:-1], t[:-1]) + cost])
    return res
 # print(LD("Python", "Peithen"))

# print(LD("prosperity","properties"))



def lengthOfLongestSubstring( s):
    """
    :type s: str
    :rtype: int
    """
    dup = {}
    res = []
    cur_str = ""
    checked = 0
    for i in range(len(s)):
        if s[i] in dup and (checked == 0 or checked < i):
            checked = i
            res.append(cur_str)
            i = dup[s[i]] + 1
            dup.pop(s[i])
            cur_str = ""

        else:
            cur_str += s[i]
            print("this is cur", cur_str)
            dup[s[i]] = i
    res.append(cur_str)
    print(dup)
    ms = ""
    print(res)
    for i in res:
        if len(i) > len(ms):
            ms = i
    return len(ms)

print(lengthOfLongestSubstring("abcabcbb"))
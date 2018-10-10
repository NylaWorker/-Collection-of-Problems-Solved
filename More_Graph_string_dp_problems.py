'''
Collection of random problems.

'''


def numDecodings(self, s):
    """
    :type s: str
    :rtype: int
    """
    return self.helper(s, 1, {})

def helper(self, s, count, memo):
    if int(s) <27:
        print(count)
        memo[s]= count
        count += 1
        return count
    elif int(s) < 26:
        print(count)
        memo[s]= count
        count += 2
        return count
    elif s in memo:
        count += memo[s]
        return count

    count +=  self.helper(s[1:], count, memo)
    memo[s] = count
    print("This is ", memo)
    return count

print(numDecodings("14242"))


def zombieCluster(zombies):
    # Write your code here
    for i in range(len(zombies)):
        zombies[i] = list(zombies[i])
    '''
    This can be done with bfs and modifying the original zombie map. Givent that I don't have contraints I will do that. 
    Instead of keeping track of the visited nodes I will just erase them. 

    Here you have the idea. I need to debug it a little, but because of time I moved to the other problem. 
    '''

    if len(zombies) == 0:
        return 0
    count = 0
    for r in range(len(zombies)):
        for c in range(len(zombies[0])):
            if zombies[r][c] == "1":
                print(zombies[r][c])
                bfs(zombies, (r, c))
                count += 1
    return count


def bfs(board, start):
    q = [(0, 0)]
    drs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while q:
        row, col = q.pop(0)
        board[row][col] = "0"
        for r, c in drs:
            rc = row + r
            cc = col + c
            if is_valid(board, rc, cc) and board[rc][cc] == "1":
                q.append((rc, cc))


def is_valid(board, row, col):
    if row > len(board[0]) - 1 or col > len(board) or row < 0 or col < 0:
        return False
    return True


def connectedCities(n, g, originCities, destinationCities):
    # Write your code here

    '''
    This is another graph problem. I am deciding between bfs or dfs.
    First I create an adjacency list representation.

    '''
    print(n)
    print(g)
    print(originCities)
    print(destinationCities)
    adj = {}
    for i in range(1, n + 1):  # this generates the neighbors
        print(i)
        adj[i] = []
        for element in range(1, (i) // 2 + 1):
            if i % element == 0:
                if i > g:
                    adj[i].append(element)
                    adj[element].append(i)
    print(adj)  # adj is a dictionary with all the adjacent nodes.
    # with adj we can now do a search for each of the cities.  (Important to note this is a directed graph. ) Then i realized it is not a directed graph
    '''
    There are two ways  of doing this. Either bfs or dfs. Given that I coded bfs and it went okay I am just going to do dfs. 
    I am unclear of the threshold is and it is making this fail. 

    Sorry, I am unclear about what the threshold is. If I understood it I would fix it. 
    '''
    originCities = set(originCities)
    res = len(destinationCities) * [0]
    for i in range(len(destinationCities)):
        tempV = set()
        print("dest city ", destinationCities[i])
        if dfs(adj, destinationCities[i], originCities, tempV):
            res[i] = 1
    return res


def dfs(graphD, start, originCities, tempV):
    if start in originCities:
        print("here")
        return True

    for i in graphD[start]:
        if i not in tempV:
            print(i)
            tempV.add(i)
            return dfs(graphD, i, originCities, tempV)
    return False

def backtrack( s):
    memo, n = {}, len(s)
    def n_decodings_from(start):
        # T(n) = O(3^n)
        # S(n) = O(1)
        if start in memo:
            return memo[start]
        if start == n:
            return 1
        count = cand = p = 0
        for j in range(start, min(start + 2, n)):
            cand *= pow(10, p)
            cand += int(s[j])
            if cand < 1 or cand > 26:
                break
            count += n_decodings_from(j + 1)
            p += 1
        memo[start] = count
        return count

    return n_decodings_from(0)

print("this is back", backtrack("14242"))

import collections



def validPalindrome(self, s):
    """
    :type s: str
    :rtype: bool
    """
    if s == "" or s == " ":
        return True

    if s == s[::-1]:
        return True
    else:
        for i in range(0, len(s) - 1):
            if i == 0:
                a = s[1:]
                if a == a[::-1]:
                    return True

            a = s[i] + s[i + 1:]
            if a == a[::-1]:
                return True
        return False



def plusOne(self, digits):
    """
    :type digits: List[int]
    :rtype: List[int]
    """
    carry = 1
    count = 10

    for i in range(len(digits) - 1, -1, -1):
        cur = digits[i] + carry
        if cur < 10:
            digits[i] = cur
            carry = 0
            break
        else:
            count = 1
            carry = cur // 10
            digits[i] = cur % 10
            count = cur % 10
    if carry > 0 and count == 0:
        digits.insert(0, carry)
    return digits

print(plusOne([9,9,9,9]))

print("NEW ")


def trap(self, height):
    """
    :type height: List[int]
    :rtype: int
    """
    if len(height) == 0:
        return 0
    max_n = max(height)
    max_ind = height.index(max_n)
    count = 0
    count += self.findArea(height[max_ind:][::-1])
    print(height[max_ind:][::-1])
    count += self.findArea(height[:max_ind+1])
    print(height[:max_ind+1])
    return count

def findArea(self, height):
    if len(height) ==0: return 0
    de = 0
    tempde = 0
    max_left = height[0]
    print("HEIGHT" ,height)
    start = True
    for i in range(1, len(height)):
        if height[i] >= max_left:
            print("here")
            max_left = height[i]
            de += tempde
            tempde = 0
        else:
            tempde += max_left - height[i]
            print(tempde)
    return de
print(trap([5,4,1,2]))



def mergeKLists(self, lists):
    """
    :type lists: List[ListNode]
    :rtype: ListNode
    """
    if len(lists) == 0: return lists
    if len(lists) == 1: return lists[0]
    head = new = ListNode(-1)
    count = 0

    while count < len(lists):
        min_val = float("inf")
        count = 0
        item = 0
        min_ind = 0
        for i in range(len(lists)):
            if lists[i] == None:
                count += 1
            elif lists[i].val < min_val and item == 0:
                if min_val == lists[i].val:
                    item = 1
                min_val = lists[i].val
                min_ind = i
        if lists[min_ind] != None:
            lists[min_ind] = lists[min_ind].next
        if min_val != float("inf"):
            new.next = ListNode(min_val)
            new = new.next
    return head.next




class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

right = TreeNode(8)
right.left = TreeNode(7)
left = TreeNode(2)
left.left = TreeNode(1)

root = TreeNode(6)
root.left = left
root.right = right




def inorderSuccessor(self, root, p):
    """
    :type root: TreeNode
    :type p: TreeNode
    :rtype: TreeNode
    """
    res = []
    imp = []
    self.inorder(root, res,p,imp)
    print(res)
    print(imp)
    if imp:
        if imp[0] <len(res):
            return res[imp[0]]
    return None

def inorder(self, root, res,p,imp):
    if root:
        self.inorder(root.left,res,p,imp)
        res.append(root.val)
        print("inorderrrr")
        print(p.val)
        print(root.val)
        if root.val==p.val:
            imp = imp.append(len(res))
            print("this impo", imp)
        self.inorder(root.right,res,p,imp)

print(inorderSuccessor(root,TreeNode(7)))


def wordBreak(self, s, wordDict):
    """
    :type s: str
    :type wordDict: List[str]
    :rtype: bool
    """
    d = [False] * len(s)

    for i in range(0, len(s)):
        for word in wordDict:
            if s[i - len(word) + 1:i + 1] == word and (d[i - len(word)] or i - len(word) == -1):
                d[i] = True
    return d[len(s) - 1]



def numDecodings(self, s):
    """
    :type s: str
    :rtype: int
    """
    if s[0] == "0": return 0

    dp1 = dp2 = 1
    for i in range(1, len(s)):
        if s[i] == "0" and (s[i - 1] >= "3" or s[i - 1] == "0"): return 0
        if s[i] == "0":
            dp1, dp2 = [dp2, dp1]
        elif "10" <= s[i - 1:i + 1] <= "26":
            dp1, dp2 = dp2, dp1 + dp2
    return dp2



def swimInWater(self, grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    if len(grid) == 0 or len(grid[0]) == 0: return 0
    maxm = [0]
    start = (0, 0)
    self.dfs(grid, start, maxm, set(),[0])
    return maxm[0]


def dfs(self, grid, start, maxm, visited,prev):
    if start in visited:
        visited.remove(prev[-1])
        print("got here")
    visited.add(start)
    print(start)
    if start == (len(grid) - 1, len(grid) - 1):
        return
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    r, c = start
    min_pair = [start] * 4
    min_path = [float("inf")] * 4
    count = 0
    for rc, cc in dirs:
        rtemp = r + rc
        ctemp = c + cc
        if self.isValid(grid, rtemp, ctemp) and (rtemp, ctemp) not in visited:
            min_path[count] = grid[rtemp][ctemp]
            min_pair[count] = (rtemp, ctemp)
        count += 1
    minv = min(min_path)
    minp = min_pair[min_path.index(minv)]
    maxm[0] = max(maxm[0], minv)
    print("this is prev",prev)

    self.dfs(grid, minp, maxm, visited,prev.append(minp))

def isValid(self, grid, r, c):
    if r >= 0 and c >= 0 and r < len(grid[0]) and c < len(grid):
        return True
    return False

# print(Solution().swimInWater([[7,11,5,3],[2,14,12,8],[4,13,9,10],[6,0,1,15]]))



def checkPossibility(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    if len(nums)<2: return False
    count = 0
    for i in range(len(nums)-1):
        pos = nums[i-1]-nums[i]
        if pos<0:
            count +=1
    if count <2:
        return True
    return False

print(checkPossibility([4,2,1]))


def moves(a):
    # Write your code here
    a_len = len(a)
    leftp = 0
    swap = 0
    print(a)

    for i in range(len(a) // 2):
        cur = True
        print(a[i])
        if a[i] % 2 == 1:
            print("here")
            swap += 1
            cur = False
            leftp += 1
        '''
        This is  not right. 
        '''
        if cur and a[len(a) - i - 1] % 2 == 0:
            swap += 1
            leftp += 1

        if swap == 0 and i == (len(a)) // 2 - 1:
            return swap

        if leftp + i == len(a) - 1:
            print(leftp + i)
            return swap
    return swap

print("NEWWWWW")



print(minWindow("jmeqksfrsdcmsiwvaovztaqenprpvnbstl","l"))
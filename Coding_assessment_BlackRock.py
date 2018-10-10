


class Solution:
    def longestCommonPrefix( strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if strs == []: return ""

        pref = ""
        if strs[0] == "": return ""
        cur = strs[0][0]
        count = 0
        for x in range(len(strs[0])):
            for s in strs:
                if len(s) <= x:
                    count =1
                    break
                if s[x] != cur:
                    count = 1
                    break
            if count != 0:
                break
            else:
                pref += cur
                if len(strs[0]) > x + 1:
                    cur = strs[0][x + 1]
        return pref

print(Solution().longestCommonPrefix([]))


def ascii_deletion_distance(str1, str2):
    if str1=="":
        sum_ord2 = 0
        for i in str2:
            sum_ord2 += ord(i)
        return sum_ord2
    if str2=="":
        sum_ord1 = 0
        for i in str1:
            sum_ord1 += ord(i)
        return sum_ord1
    if str1[-1]== str2[-1]:
        add = 0
    else:
        add = ord(str1[-1])+ord(str2[-1])
    res = ascii_deletion_distance(str1[:-1],str2[:-1])+add
    return res

ascii_deletion_distance("at", "cat")

def four_letter_words(sentence):
    count = 0
    for i in sentence:
        if len(i)==4:
            count+=1
    return count


def smallestDistancePair( nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    nums = sorted(nums)
    low  = nums[-1] - nums[0]
    for i in range(1, len(nums)):
        low = min(low, nums[i] - nums[i - 1])
    high = nums[-1] - nums[0]
    while low < high:
        mid = (low + high )>>1
        if cnt_p(nums, mid) < k:
            low = mid + 1
        else:
            high = mid
    return low

def cnt_p( nums, mid):
    res = 0
    for i in range(len(nums)):
        j = i
        while (j < len(nums) and nums[j] - nums[i] <= mid): j += 1
        res += j - i - 1
    return res

print(smallestDistancePair([62,100,4],2))




def num_to_cash(num):
  cash_available = {.01: 'PENNY', .05: 'NICKEL', .10: 'DIME', .25: 'QUARTER', .50: 'HALF DOLLAR', 1.00: 'ONE',
    2.00: 'TWO', 5.00: 'FIVE', 10.00: 'TEN', 20.00: 'TWENTY', 50.00: 'FIFTY'}
    bills = cash_available.keys()
    ret_cash = []
    while num:
      cur_bill = bills.pop()
        check = num / cur_bill
        if check != 0:
          num = num - check * cur_bill
            bill_needed = [cash_available[cur_bill]] * check
            ret_cash.append(bill_needed)
    return " ".join(ret_cash)


def main():
  PP, CH = line.split(";")
    PP, CH = int(PP), int(CH)
    if CH < PP:
      print("Error")
    if CH == PP:
      print("Error")
    else:
      print(num_to_cash(CH - PP))

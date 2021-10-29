#!/usr/bin/python
# Included in this file are some functions that help to improve the performance of the prediction.
import itertools
import random

#-------------------------------##-------------------------------#
#-------------------------------##-------------------------------#
def addhydrogen(str1): # Improve Uri Alon's hash representations.
    if str1=='H':
        print("Wrong map !!!!!")
        return
    elif str1=="CH@3":
        str1="CH2"
    elif str1=="C@3":
        str1="CH@3"
    elif str1.count('H')==0:
        str1=str1+'H'
    elif str1[len(str1)-1]=='H':
        str1=str1+'2'
    else:
        str1=str1.replace(str1[len(str1)-1],str(int(str1[len(str1)-1])+1))
    return str1
#-------------------------------##-------------------------------#
#-------------------------------##-------------------------------#
def replace_n(str1, n, str2):
    letters = (
        str2 if i == n else char
        for i, char in enumerate(str1)
    )
    return ''.join(letters)
#-------------------------------##-------------------------------#
#-------------------------------##-------------------------------#
def get_num_combinations(list_a):
    count_x=1
    for i in range(len(list_a)):
        count_x=count_x*len(list_a[i])
    return count_x
#-------------------------------##-------------------------------#
#-------------------------------##-------------------------------#
def get_ith_combination(list_a, num_i): #list_a is a list of lists to perform combination
    ith_combination=[]
    length_list=[]
    mod_list=[]

    ith_list=[]
    for one_list in list_a:
        length_list.append(len(one_list))
        mod_list.append(1)

        ith_list.append(0)
    for i in range(len(length_list)):
        for j in range(len(length_list)-i):
            mod_list[i]=length_list[-j-1]*mod_list[i]
    for i in range(len(length_list)-1):
        ith_list[i]=(num_i % mod_list[i]) / mod_list[i+1] + 1
        if (num_i % mod_list[i]) % mod_list[i+1]==0:
            ith_list[i]=ith_list[i]-1
        if num_i % mod_list[i]==0:
            ith_list[i]=length_list[i]
    ith_list[-1]=num_i % mod_list[-1]
    if ith_list[-1]==0:
        ith_list[-1]=length_list[-1]
    for j in range(len(ith_list)):
        ith_combination.append(list_a[j][ith_list[j]-1])
    return ith_combination
#-------------------------------##-------------------------------#
#-------------------------------##-------------------------------#
def cart_prod( alistoflists): # return combinations of elements, each from a different set.
    # [[a,b,c],[d,e,f],[g,h], ...] converts to
    # [(a,d,g),(b,d,g),(c,d,g),(a,d,h),(b,d,h), ...]
    prod_list=alistoflists[0]
    if len(alistoflists)>1:
        for i in range(len(alistoflists)-1):
            prod_list=list(itertools.product(prod_list,alistoflists[i+1]))
            if i==0:
                pass
            else:
                set_list=prod_list
                prod_list=[]
                for j in range(len(set_list)):
                    inbracket=list(set_list[j][0])
                    nobracket=set_list[j][1]
                    inbracket.append(nobracket)
                    prod_list.append(tuple(inbracket))
    else:
        temp_list=[]
        for i in range(len(prod_list)):
            temp_list.append((prod_list[i],))
        prod_list=temp_list
    return prod_list
#-------------------------------##-------------------------------#
#-------------------------------##-------------------------------#
def randomList(a): 
    b = [] 
    for i in range(len(a)): 
        element = random.choice(a) 
        a.remove(element) 
        b.append(element) 
    return b
#-------------------------------##-------------------------------#
#-------------------------------##-------------------------------#
def iftuplestrinlist(a,b): # Check if all strs in one tuple are contained in a list
    allstrsin=False
    countall=0
    for i in range(len(a)):
        if a[i] in b:
            countall+=1
    if countall==len(a):
        allstrsin=True
    return allstrsin
#-------------------------------##-------------------------------#
#-------------------------------##-------------------------------#
def find_nth(string_a, string_b, n): # Find the nth occurence of string_b in string_a (, find_nth('aaaaa','aa',3) = -1).
    s = string_a.find(string_b)
    while s >= 0 and n > 1:
        s = string_a.find(string_b, s+len(string_b))
        n -= 1
    return s
#-------------------------------##-------------------------------#
#-------------------------------##-------------------------------#
def simplify_enzymeid(enzyme_a):
    # For better texture output
    if enzyme_a=='ecKAYLA_backward' or enzyme_a=='ecKAYLA_forward':
        return enzyme_a
    else:
        enzyme_b=enzyme_a
        thethirddotindex=find_nth(enzyme_b,'.',3)
        theshortlineindex=enzyme_b.index('_')
        start=thethirddotindex
        while start<theshortlineindex:
            enzyme_b=replace_n(enzyme_b,thethirddotindex,'')
            start+=1
        return enzyme_b
#-------------------------------##-------------------------------#
#-------------------------------##-------------------------------#
def remove_bkgd_cmpd(hash_list_a):
    bkgd_cmpd_list = ['OH2~','CoAH~','OH,OH,OH,O,P~0000001112','C,O,O~220','NH3~']
    list_cleared=[]
    for i in range(len(hash_list_a)):
        if hash_list_a[i] not in bkgd_cmpd_list:
            list_cleared.append(hash_list_a[i])
    return list_cleared
#-------------------------------##-------------------------------#
#-------------------------------##-------------------------------#
def remove_hydrogen_nodes_in_rule(one_ruleH):
    while (one_ruleH.find('\\')!=-1):
        location=one_ruleH.index('\\')
        one_ruleH=replace_n(one_ruleH,location,'') # Remove '\'
    while (one_ruleH.find('/')!=-1):
        location=one_ruleH.index('/')
        one_ruleH=replace_n(one_ruleH,location,'') # Remove '/'
    while (one_ruleH.find('[H]')!=-1):
        location=one_ruleH.index('[H]')
        one_ruleH=replace_n(one_ruleH,location,'') # Remove '['
        one_ruleH=replace_n(one_ruleH,location,'') # Remove 'H'
        one_ruleH=replace_n(one_ruleH,location,'') # Remove ']'
    while (one_ruleH.find('[H:')!=-1):
        location=one_ruleH.index('[H:')
        if one_ruleH[location+4]=="]":
            one_ruleH=replace_n(one_ruleH,location,'') # Remove '['
            one_ruleH=replace_n(one_ruleH,location,'') # Remove 'H'
            one_ruleH=replace_n(one_ruleH,location,'') # Remove ':'
            one_ruleH=replace_n(one_ruleH,location,'') # Remove '*'
            one_ruleH=replace_n(one_ruleH,location,'') # Remove ']'
        elif one_ruleH[location+5]=="]":
            one_ruleH=replace_n(one_ruleH,location,'') # Remove '['
            one_ruleH=replace_n(one_ruleH,location,'') # Remove 'H'
            one_ruleH=replace_n(one_ruleH,location,'') # Remove ':'
            one_ruleH=replace_n(one_ruleH,location,'') # Remove '*'
            one_ruleH=replace_n(one_ruleH,location,'') # Remove '*'
            one_ruleH=replace_n(one_ruleH,location,'') # Remove ']'
        else:
            print("is that even possible?")
    while (one_ruleH.find('()')!=-1):
        location=one_ruleH.index('()')
        one_ruleH=replace_n(one_ruleH,location,'') # Remove '('
        one_ruleH=replace_n(one_ruleH,location,'') # Remove ')'
    return one_ruleH


###############################################################
###############################################################
from collections import OrderedDict,Counter
class OrderedCounter(Counter, OrderedDict):
     'Counter that remembers the order elements are first encountered'

     def __repr__(self):
         return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

     def __reduce__(self):
         return self.__class__, (OrderedDict(self),)

###############################################################
###############################################################
def test():

    for i in cartesian_product([[0,1],[0,2],[0,3]], unique_values=False):
        print(i)

if (__name__ == '__main__'):
    test()

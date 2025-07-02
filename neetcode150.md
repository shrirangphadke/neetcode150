**94. Binary Tree Inorder Traversal**

Given the `root` of a binary tree, return _the inorder traversal of its nodes' values_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

**Input:** root = [1,null,2,3]
**Output:** [1,3,2]

**Example 2:**
**Input:** root = []
**Output:** []

**Example 3:**
**Input:** root = [1]
**Output:** [1]

**Constraints:**
-   The number of nodes in the tree is in the range `[0, 100]`.
-   `-100 <= Node.val <= 100`

Solution:

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        lis = []
        self.dfs(root, lis)
        return lis

    def dfs(self, root, lis):
        if(root == None):
            return
        self.dfs(root.left, lis)
        lis.append(root.val)
        self.dfs(root.right, lis)
```

**100. Same Tree**

Given the roots of two binary trees `p` and `q`, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/12/20/ex1.jpg)

**Input:** p = [1,2,3], q = [1,2,3]
**Output:** true

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/12/20/ex2.jpg)

**Input:** p = [1,2], q = [1,null,2]
**Output:** false

Solution:

```python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```

**101. Symmetric Tree**

Given the `root` of a binary tree, _check whether it is a mirror of itself_ (i.e., symmetric around its center).

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/02/19/symtree1.jpg)

**Input:** root = [1,2,2,3,4,4,3]
**Output:** true

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/02/19/symtree2.jpg)

**Input:** root = [1,2,2,null,3,null,3]
**Output:** false

Solution:

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        return self.isSame(root, root)
    
    def isSame(self, n1, n2):
        if not n1 and not n2:
            return True
        elif not n1 or not n2:
            return False
        
        return n1.val == n2.val and self.isSame(n1.right, n2.left) and self.isSame(n1.left, n2.right)
```


**104. Maximum Depth of Binary Tree**

Given the `root` of a binary tree, return _its maximum depth_.

A binary tree's **maximum depth** is the number of nodes along the longest path from the root node down to the farthest leaf node.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/11/26/tmp-tree.jpg)

**Input:** root = [3,9,20,null,null,15,7]
**Output:** 3

**Example 2:**
**Input:** root = [1,null,2]
**Output:** 2

Solution:

```python
class Solution:
    maxheight = 0
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def check(root,value):
            if(root):
                value = value +1
                self.maxheight = max(value,self.maxheight)
                check(root.left,value)
                check(root.right,value)
        check(root,0)
        return self.maxheight
```

**108. Convert Sorted Array to Binary Search Tree**

Given an integer array `nums` where the elements are sorted in **ascending order**, convert it to a height-balanced binary search tree.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/02/18/btree1.jpg)

**Input:** nums = [-10,-3,0,5,9]
**Output:** [0,-3,9,-10,null,5]
**Explanation:** [0,-10,5,null,-3,null,9] is also accepted:


![](https://assets.leetcode.com/uploads/2021/02/18/btree2.jpg)

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/02/18/btree.jpg)

**Input:** nums = [1,3]
**Output:** [3,1]
**Explanation:** [1,null,3] and [3,1] are both height-balanced BSTs.

Solution:

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        if not nums:
            return None
        mid = len(nums) // 2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root
```

**110. Balanced Binary Tree**

Given a binary tree, determine if it is height-balanced
**Example 1:**

![](https://assets.leetcode.com/uploads/2020/10/06/balance_1.jpg)

**Input:** root = [3,9,20,null,null,15,7]
**Output:** true

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/10/06/balance_2.jpg)

**Input:** root = [1,2,2,3,3,null,null,4,4]
**Output:** false

**Example 3:**
**Input:** root = []
**Output:** true

Solution:

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def getHeight(node):
            if node is None:
                return 0
            left_height = getHeight(node.left)
            right_height = getHeight(node.right)
            if left_height == -1 or right_height == -1 or abs(left_height - right_height) > 1:
                return -1
            return 1 + max(left_height, right_height)
        return getHeight(root) != -1
```

**111. Minimum Depth of Binary Tree**

Given a binary tree, find its minimum depth.
The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
**Note:** A leaf is a node with no children.

**Example 1:**
![](https://assets.leetcode.com/uploads/2020/10/12/ex_depth.jpg)

**Input:** root = [3,9,20,null,null,15,7]
**Output:** 2

**Example 2:**
**Input:** root = [2,null,3,null,4,null,5,null,6]
**Output:** 5

Solution:

```python
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        def count(root):
            if root is None:
                return 0
            if root.left is None:
                return count(root.right) + 1
            if root.right is None:
                return count(root.left) + 1
            return min(count(root.left), count(root.right)) + 1
        return count(root)
```

**112. Path Sum**

Given the `root` of a binary tree and an integer `targetSum`, return `true` if the tree has a **root-to-leaf** path such that adding up all the values along the path equals `targetSum`.
A **leaf** is a node with no children.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/01/18/pathsum1.jpg)

**Input:** root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
**Output:** true
**Explanation:** The root-to-leaf path with the target sum is shown.

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/01/18/pathsum2.jpg)

**Input:** root = [1,2,3], targetSum = 5
**Output:** false

**Explanation:** There two root-to-leaf paths in the tree:
(1 --> 2): The sum is 3.
(1 --> 3): The sum is 4.
There is no root-to-leaf path with sum = 5.

**Example 3:**

**Input:** root = [], targetSum = 0
**Output:** false
**Explanation:** Since the tree is empty, there are no root-to-leaf paths.

Solution:

```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        
        # Use a list to store the path from root to the current node
        stack = [(root, root.val)]
        
        while stack:
            node, val = stack.pop()
            
            # Check if the current node is a leaf node and its value matches the targetSum
            if not node.left and not node.right and val == targetSum:
                return True
            
            # Add the left and right children to the stack along with their updated path value
            if node.left:
                stack.append((node.left, val + node.left.val))
            if node.right:
                stack.append((node.right, val + node.right.val))
        
        return False
```

**144. Binary Tree Preorder Traversal**

Given the `root` of a binary tree, return _the preorder traversal of its nodes' values_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

**Input:** root = [1,null,2,3]
**Output:** [1,2,3]

**Example 2:**
**Input:** root = []
**Output:** []

**Example 3:**
**Input:** root = [1]
**Output:** [1]

Solution:

```python
class Solution:
    def preorderTraversal(self, root_: TreeNode | None) -> list[int]:
        def preorder(root: TreeNode | None) -> Iterable:
            if not root: return
            yield root.val
            yield from preorder(root.left)
            yield from preorder(root.right)
                
        return list(preorder(root_))
```

**145. Binary Tree Postorder Traversal**

Given the `root` of a binary tree, return _the postorder traversal of its nodes' values_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/08/28/pre1.jpg)

**Input:** root = [1,null,2,3]
**Output:** [3,2,1]

**Example 2:**
**Input:** root = []
**Output:** []

**Example 3:**
**Input:** root = [1]
**Output:** [1]

Solution:

```python
class Solution:
    def __init__(self):
        self.postOrder = []
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        if root:
            self.postorderTraversal(root.left)
            self.postorderTraversal(root.right)
            self.postOrder.append(root.val)
        return self.postOrder
```

**226. Invert Binary Tree**

Given the `root` of a binary tree, invert the tree, and return _its root_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/03/14/invert1-tree.jpg)

**Input:** root = [4,2,7,1,3,6,9]
**Output:** [4,7,2,9,6,3,1]

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/03/14/invert2-tree.jpg)

**Input:** root = [2,1,3]
**Output:** [2,3,1]

**Example 3:**
**Input:** root = []
**Output:** []

Solution:

```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root: #Base Case
            return root
        self.invertTree(root.left) #Call the left substree
        self.invertTree(root.right)  #Call the right substree
        # Swap the nodes
        root.left, root.right = root.right, root.left
        return root # Return the root
```

**257. Binary Tree Paths**

Given the `root` of a binary tree, return _all root-to-leaf paths in **any order**_. A **leaf** is a node with no children.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/03/12/paths-tree.jpg)

**Input:** root = [1,2,3,null,5]
**Output:** ["1->2->5","1->3"]

**Example 2:**
**Input:** root = [1]
**Output:** ["1"]

Solution:

```python
class Solution:
  def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
    ans = []

    def dfs(root: Optional[TreeNode], path: List[str]) -> None:
      if not root:
        return
      if not root.left and not root.right:
        ans.append(''.join(path) + str(root.val))
        return

      path.append(str(root.val) + '->')
      dfs(root.left, path)
      dfs(root.right, path)
      path.pop()

    dfs(root, [])
    return ans
```

**404. Sum of Left Leaves**

Given the `root` of a binary tree, return _the sum of all left leaves._
A **leaf** is a node with no children. A **left leaf** is a leaf that is the left child of another node.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/04/08/leftsum-tree.jpg)

**Input:** root = [3,9,20,null,null,15,7]
**Output:** 24
**Explanation:** There are two left leaves in the binary tree, with values 9 and 15 respectively.

**Example 2:**
**Input:** root = [1]
**Output:** 0

Solution:

```python
class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        # Define a helper function to recursively traverse the tree and compute the sum of left leaves
        def traverse(node, is_left):
            # If the current node is None, return 0
            if not node:
                return 0
            
            # If the current node is a leaf node and is a left child, return its value
            if not node.left and not node.right and is_left:
                return node.val
            
            # Recursively traverse the left and right subtrees
            left_sum = traverse(node.left, True)
            right_sum = traverse(node.right, False)
            
            return left_sum + right_sum
        
        # Call the helper function with the root node and False to indicate that the root is not a left child
        return traverse(root, False)
```

**501. Find Mode in Binary Search Tree**

Given the `root` of a binary search tree (BST) with duplicates, return _all the [mode(s)](https://en.wikipedia.org/wiki/Mode_(statistics)) (i.e., the most frequently occurred element) in it_. If the tree has more than one mode, return them in **any order**.

Assume a BST is defined as follows:
-   The left subtree of a node contains only nodes with keys **less than or equal to** the node's key.
-   The right subtree of a node contains only nodes with keys **greater than or equal to** the node's key.
-   Both the left and right subtrees must also be binary search trees.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/03/11/mode-tree.jpg)

**Input:** root = [1,null,2,2]
**Output:** [2]

**Example 2:**
**Input:** root = [0]
**Output:** [0]

Solution:

```python
class Solution:
    def __init__(self):
        self.d = {}
        self.m = 1
        self.ans = []
        
    def dfs(self, x):
        if not x:
            return
        if x.val not in self.d:
            self.d[x.val] = 1    
        else:
            self.d[x.val] += 1
            self.m = max(self.m, self.d[x.val])
            
        self.dfs(x.left)
        self.dfs(x.right)
    
    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        self.dfs(root)
        for x in self.d:
            if self.d[x] == self.m:
                self.ans.append(x)
        return self.ans
```

**530. Minimum Absolute Difference in BST**

Given the `root` of a Binary Search Tree (BST), return _the minimum absolute difference between the values of any two different nodes in the tree_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/02/05/bst1.jpg)

**Input:** root = [4,2,6,1,3]
**Output:** 1

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/02/05/bst2.jpg)

**Input:** root = [1,0,48,null,null,12,49]
**Output:** 1

Solution:

```python
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        d = float('inf')
        s = []
        if root == None:
            return 
        d = self.traverse(root,d,s)
        return d

    def traverse(self,root,d,s):
        if root.left != None:
            d = self.traverse(root.left,d,s)
        s.append(root.val)
        if len(s)>1:
            diff = s[-1]-s[-2]
            if diff < d:
                d = diff
        if root.right != None:
            d = self.traverse(root.right,d,s) 
        return d
```

**543. Diameter of Binary Tree**

Given the `root` of a binary tree, return _the length of the **diameter** of the tree_.
The **diameter** of a binary tree is the **length** of the longest path between any two nodes in a tree. This path may or may not pass through the `root`.
The **length** of a path between two nodes is represented by the number of edges between them.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/03/06/diamtree.jpg)

**Input:** root = [1,2,3,4,5]
**Output:** 3
**Explanation:** 3 is the length of the path [4,2,1,3] or [5,2,1,3].

**Example 2:**
**Input:** root = [1,2]
**Output:** 1

Solution:

```python
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.ans = 0  # variable to store the maximum diameter found so far
        
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            self.ans = max(self.ans, left + right)  # update ans if new diameter is found
            return max(left, right) + 1  # return the maximum height of the node
            
        dfs(root)
        return self.ans
```

**563. Binary Tree Tilt**

Given the `root` of a binary tree, return _the sum of every tree node's **tilt**._
The **tilt** of a tree node is the **absolute difference** between the sum of all left subtree node **values** and all right subtree node **values**. If a node does not have a left child, then the sum of the left subtree node **values** is treated as `0`. The rule is similar if the node does not have a right child.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/10/20/tilt1.jpg)

**Input:** root = [1,2,3]
**Output:** 1
**Explanation:** 
Tilt of node 2 : |0-0| = 0 (no children)
Tilt of node 3 : |0-0| = 0 (no children)
Tilt of node 1 : |2-3| = 1 (left subtree is just left child, so sum is 2; right subtree is just right child, so sum is 3)
Sum of every tilt : 0 + 0 + 1 = 1

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/10/20/tilt2.jpg)

**Input:** root = [4,2,9,3,5,null,7]
**Output:** 15
**Explanation:** 
Tilt of node 3 : |0-0| = 0 (no children)
Tilt of node 5 : |0-0| = 0 (no children)
Tilt of node 7 : |0-0| = 0 (no children)
Tilt of node 2 : |3-5| = 2 (left subtree is just left child, so sum is 3; right subtree is just right child, so sum is 5)
Tilt of node 9 : |0-7| = 7 (no left child, so sum is 0; right subtree is just right child, so sum is 7)
Tilt of node 4 : |(3+5+2)-(9+7)| = |10-16| = 6 (left subtree values are 3, 5, and 2, which sums to 10; right subtree values are 9 and 7, which sums to 16)
Sum of every tilt : 0 + 0 + 0 + 2 + 7 + 6 = 15

**Example 3:**

![](https://assets.leetcode.com/uploads/2020/10/20/tilt3.jpg)

**Input:** root = [21,7,14,1,1,2,2,3,3]
**Output:** 9

Solution:

```python
class Solution:
    def findTilt(self, root: Optional[TreeNode]) -> int:
        self.total_tilt = 0
        
        def calculate_tilt(node):
            if not node:
                return 0
            
            left_sum = calculate_tilt(node.left)
            right_sum = calculate_tilt(node.right)
            tilt = abs(left_sum - right_sum)
            
            self.total_tilt += tilt
            
            return left_sum + right_sum + node.val
        
        calculate_tilt(root)
        return self.total_tilt
```

**572. Subtree of Another Tree**

Given the roots of two binary trees `root` and `subRoot`, return `true` if there is a subtree of `root` with the same structure and node values of `subRoot` and `false` otherwise.
A subtree of a binary tree `tree` is a tree that consists of a node in `tree` and all of this node's descendants. The tree `tree` could also be considered as a subtree of itself.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/04/28/subtree1-tree.jpg)

**Input:** root = [3,4,5,1,2], subRoot = [4,1,2]
**Output:** true

**Example 2:**
![](https://assets.leetcode.com/uploads/2021/04/28/subtree2-tree.jpg)

**Input:** root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
**Output:** false

```python
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        # Helper function to check if two trees are equal in value and structure
        def checkEqual(root1, root2):
            if not root1 and not root2:
                return True
            elif not root1 or not root2:
                return False
            elif root1.val != root2.val:
                return False
            else:
                return checkEqual(root1.left, root2.left) and checkEqual(root1.right, root2.right)
        
        # Traverse through the root tree and check if any subtree matches the subRoot tree
        if not root:
            return False
        elif checkEqual(root, subRoot):
            return True
        else:
            return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
```

**606. Construct String from Binary Tree**

Given the `root` of a binary tree, construct a string consisting of parenthesis and integers from a binary tree with the preorder traversal way, and return it.
Omit all the empty parenthesis pairs that do not affect the one-to-one mapping relationship between the string and the original binary tree.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/05/03/cons1-tree.jpg)

**Input:** root = [1,2,3,4]
**Output:** "1(2(4))(3)"
**Explanation:** Originally, it needs to be "1(2(4)())(3()())", but you need to omit all the unnecessary empty parenthesis pairs. And it will be "1(2(4))(3)"

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/05/03/cons2-tree.jpg)

**Input:** root = [1,2,3,null,4]
**Output:** "1(2()(4))(3)"
**Explanation:** Almost the same as the first example, except we cannot omit the first parenthesis pair to break the one-to-one mapping relationship between the input and the output.

Solution:

```python
class Solution:
    def tree2str(self, root: Optional[TreeNode]) -> str:
        # Initialising string with root.val
        string = str(root.val)
        # If root has a non-empty left subtree
        if root.left: 
            # we traverse it and wrap everything it returns in ()
            string += "(" + self.tree2str(root.left) + ")"
        # If root has a non-empty right subtree
        if root.right: 
            # If left subtree of root is empty, if we don't add empty () before actual
            # content of right subtree we can't differentiate whether it is from left
            # of right subtree. So, we are adding empty ()
            # Why we don't do like this in left subtree is, consider
            #   1
            # 2
            # Where 2 is left subtree of 1, "1(2)" and "1(2())" doesn't add any new
            # info to identify the tree
            # But, if the tree is like
            #   1
            #     2
            # Where 2 is right subtree of 1, "1(2)" and "1(()(2))" are different. 
            # Because "1(2)" won't tell that 2 is right child of 1.
            if not root.left: string += "()"
            # we traverse right subtree it and wrap everything it returns in ()
            string += "(" + self.tree2str(root.right) + ")"
        # Return string 
        return string
```

**617. Merge Two Binary Trees**

You are given two binary trees `root1` and `root2`.
Imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not. You need to merge the two trees into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of the new tree.
Return _the merged tree_.
**Note:** The merging process must start from the root nodes of both trees.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/02/05/merge.jpg)

**Input:** root1 = [1,3,2,5], root2 = [2,1,3,null,4,null,7]
**Output:** [3,4,5,5,4,null,7]

**Example 2:**

**Input:** root1 = [1], root2 = [1,2]
**Output:** [2,2]

Solution:

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        if not t1:
            return t2
        elif not t2:
            return t1
        else:
            res = TreeNode(t1.val + t2.val)
            res.left = self.mergeTrees(t1.left, t2.left)
            res.right = self.mergeTrees(t1.right, t2.right)
        return res
```

**637. Average of Levels in Binary Tree**

Given the `root` of a binary tree, return _the average value of the nodes on each level in the form of an array_. Answers within `10-5` of the actual answer will be accepted.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/03/09/avg1-tree.jpg)

**Input:** root = [3,9,20,null,null,15,7]
**Output:** [3.00000,14.50000,11.00000]
Explanation: The average value of nodes on level 0 is 3, on level 1 is 14.5, and on level 2 is 11.
Hence return [3, 14.5, 11].

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/03/09/avg2-tree.jpg)

**Input:** root = [3,9,20,15,7]
**Output:** [3.00000,14.50000,11.00000]

Solution:

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        if not root: return []
        res = []
        stack = [(root)]
        while stack:
            lvl = []
            for i in range(len(stack)):
                n = stack.pop(0)
                lvl.append(n.val)
                if n.left: stack.append(n.left)
                if n.right: stack.append(n.right)
            res.append(sum(lvl)/len(lvl))
        return res
```

**653. Two Sum IV - Input is a BST**

Given the `root` of a binary search tree and an integer `k`, return `true` _if there exist two elements in the BST such that their sum is equal to_ `k`, _or_ `false` _otherwise_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/09/21/sum_tree_1.jpg)

**Input:** root = [5,3,6,2,4,null,7], k = 9
**Output:** true

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/09/21/sum_tree_2.jpg)

**Input:** root = [5,3,6,2,4,null,7], k = 28
**Output:** false

Solution:

```python
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        return self.helper(root, k, set())
        
    def helper(self, root, k, seen):
        if not root:
            return None
        if (k - root.val) in seen:
            return True
        seen.add(root.val)
        left = self.helper(root.left, k, seen)
        right = self.helper(root.right, k, seen)
        
        return left or right
```

**671. Second Minimum Node In a Binary Tree**

Given a non-empty special binary tree consisting of nodes with the non-negative value, where each node in this tree has exactly `two` or `zero` sub-node. If the node has two sub-nodes, then this node's value is the smaller value among its two sub-nodes. More formally, the property `root.val = min(root.left.val, root.right.val)` always holds.
Given such a binary tree, you need to output the **second minimum** value in the set made of all the nodes' value in the whole tree.
If no such second minimum value exists, output -1 instead.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/10/15/smbt1.jpg)

**Input:** root = [2,2,5,null,null,5,7]
**Output:** 5
**Explanation:** The smallest value is 2, the second smallest value is 5.

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/10/15/smbt2.jpg)

**Input:** root = [2,2,2]
**Output:** -1
**Explanation:** The smallest value is 2, but there isn't any second smallest value.

Solution:

```python
class Solution:
    def findSecondMinimumValue(self, root: TreeNode) -> int:
        nums = []
        nodes = [root]
        while nodes:
            t = nodes.pop()
            nums.append(t.val)
            if t.left:
                nodes.append(t.left)
            if t.right:
                nodes.append(t.right)
        
        if len(set(nums)) == 1:
            return -1
        
        return sorted(set(nums))[1]
```


**559. Maximum Depth of N-ary Tree**

Given a n-ary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

_Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See examples)._

**Example 1:**

![](https://assets.leetcode.com/uploads/2018/10/12/narytreeexample.png)

**Input:** root = [1,null,3,2,4,null,5,6]
**Output:** 3

**Example 2:**

![](https://assets.leetcode.com/uploads/2019/11/08/sample_4_964.png)

**Input:** root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
**Output:** 5

Solution:

```python
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        maxdepth=0
        for child in root.children:
            maxdepth=max(self.maxDepth(child),maxdepth)
        return maxdepth+1
```


**589. N-ary Tree Preorder Traversal**

Given the `root` of an n-ary tree, return _the preorder traversal of its nodes' values_.

Nary-Tree input serialization is represented in their level order traversal. Each group of children is separated by the null value (See examples)

**Example 1:**

![](https://assets.leetcode.com/uploads/2018/10/12/narytreeexample.png)

**Input:** root = [1,null,3,2,4,null,5,6]
**Output:** [1,3,5,6,2,4]

**Example 2:**

![](https://assets.leetcode.com/uploads/2019/11/08/sample_4_964.png)

**Input:** root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
**Output:** [1,2,3,6,7,11,14,4,8,12,5,9,13,10]

```python
class Solution(object):
    def preorder(self, root):
        # To store the output result...
        output = []
        self.traverse(root, output)
        return output
    def traverse(self, root, output):
        # Base case: If root is none...
        if root is None: return
        # Append the value of the root node to the output...
        output.append(root.val)
        # Recursively traverse each node in the children array...
        for child in root.children:
            self.traverse(child, output)
```


**590. N-ary Tree Postorder Traversal**

Given the `root` of an n-ary tree, return _the postorder traversal of its nodes' values_.

Nary-Tree input serialization is represented in their level order traversal. Each group of children is separated by the null value (See examples)

**Example 1:**

![](https://assets.leetcode.com/uploads/2018/10/12/narytreeexample.png)

**Input:** root = [1,null,3,2,4,null,5,6]
**Output:** [5,6,3,2,4,1]

**Example 2:**

![](https://assets.leetcode.com/uploads/2019/11/08/sample_4_964.png)

**Input:** root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
**Output:** [2,6,14,11,7,3,12,8,4,13,9,10,5,1]

```python
class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        if not root:
            return []
        
        stack = [root]
        res = []
        
        while stack:
            node = stack.pop()
            res.append(node.val)
            
            for child in node.children:
                stack.append(child)
                
        return res[::-1]

```

**700. Search in a Binary Search Tree**

You are given the `root` of a binary search tree (BST) and an integer `val`.

Find the node in the BST that the node's value equals `val` and return the subtree rooted with that node. If such a node does not exist, return `null`.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/01/12/tree1.jpg)

**Input:** root = [4,2,7,1,3], val = 2
**Output:** [2,1,3]

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/01/12/tree2.jpg)

**Input:** root = [4,2,7,1,3], val = 5
**Output:** []

```kotlin
class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        
        def search(root):
            if root == None:
                return None
            if root.val == val:
                return root
            if val < root.val:
                return search(root.left)
            else:
                return search(root.right)
        
        return search(root)
```

**703. Kth Largest Element in a Stream**

Design a class to find the `kth` largest element in a stream. Note that it is the `kth` largest element in the sorted order, not the `kth` distinct element.

Implement `KthLargest` class:

-   `KthLargest(int k, int[] nums)` Initializes the object with the integer `k` and the stream of integers `nums`.
-   `int add(int val)` Appends the integer `val` to the stream and returns the element representing the `kth` largest element in the stream.

**Example 1:**

**Input**
["KthLargest", "add", "add", "add", "add", "add"]
[[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]
**Output**
[null, 4, 5, 5, 8, 8]

**Explanation**
KthLargest kthLargest = new KthLargest(3, [4, 5, 8, 2]);
kthLargest.add(3);   // return 4
kthLargest.add(5);   // return 5
kthLargest.add(10);  // return 5
kthLargest.add(9);   // return 8
kthLargest.add(4);   // return 8

```python
import heapq
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.minHeap = []  # Min Heap
        for num in nums:
            heapq.heappush(self.minHeap, num)   # adding all elements to min heap
            
        while len(self.minHeap) > k:
            heapq.heappop(self.minHeap)         # Only keeping k maximum elements
        
        
    def add(self, val: int) -> int:
        heapq.heappush(self.minHeap, val)       # first add to min heap
        
        if len(self.minHeap) > self.k:          # if length greater pop minimum element as root is the min
            heapq.heappop(self.minHeap)
            
        return self.minHeap[0]                  # root is minHeap[0] as root is k'th max
    
# Time: O(N log(N))     # as heap size is N so heappush takes log(N) time
# Space: O(N)
```


**783. Minimum Distance Between BST Nodes**

Given the `root` of a Binary Search Tree (BST), return _the minimum difference between the values of any two different nodes in the tree_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/02/05/bst1.jpg)

**Input:** root = [4,2,6,1,3]
**Output:** 1

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/02/05/bst2.jpg)

**Input:** root = [1,0,48,null,null,12,49]
**Output:** 1

```java
def minDiffInBST(self, root: Optional[TreeNode]) -> int:
        stack = []
        prev = None
        mindif = float('inf')
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            mindif = min(root.val - prev.val, mindif) if prev else mindif
            prev = root
            root = root.right
        return mindif
```


**872. Leaf-Similar Trees**


Consider all the leaves of a binary tree, from left to right order, the values of those leaves form a **leaf value sequence**_._

![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/16/tree.png)

For example, in the given tree above, the leaf value sequence is `(6, 7, 4, 9, 8)`.

Two binary trees are considered _leaf-similar_ if their leaf value sequence is the same.

Return `true` if and only if the two given trees with head nodes `root1` and `root2` are leaf-similar.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/09/03/leaf-similar-1.jpg)

**Input:** root1 = [3,5,1,6,2,9,8,null,null,7,4], root2 = [3,5,1,6,7,4,2,null,null,null,null,null,null,9,8]
**Output:** true

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/09/03/leaf-similar-2.jpg)

**Input:** root1 = [1,2,3], root2 = [1,3,2]
**Output:** false


```ruby
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        def dfs(node):
            if node.left and node.right:
                dfs(node.left)
                dfs(node.right)
            elif node.left:
                dfs(node.left)
            elif node.right:
                dfs(node.right)
            else:
                self.ans.append(node.val)
        self.ans = []
        dfs(root1)
        length = len(self.ans)
        dfs(root2)
        return self.ans[:length]==self.ans[length:]
```


**897. Increasing Order Search Tree**

Given the `root` of a binary search tree, rearrange the tree in **in-order** so that the leftmost node in the tree is now the root of the tree, and every node has no left child and only one right child.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/11/17/ex1.jpg)

**Input:** root = [5,3,6,2,4,null,8,1,null,null,null,7,9]
**Output:** [1,null,2,null,3,null,4,null,5,null,6,null,7,null,8,null,9]

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/11/17/ex2.jpg)

**Input:** root = [5,1,7]
**Output:** [1,null,5,null,7]


```ruby
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        stack, a, curr, prev = [], [], root, None
        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.right
            curr = stack.pop()
            a.append(curr.val)
            new = TreeNode(curr.val)
            new.right = prev
            prev = new
            curr = curr.left
        return prev
        
```


**938. Range Sum of BST**

Given the `root` node of a binary search tree and two integers `low` and `high`, return _the sum of values of all nodes with a value in the **inclusive** range_ `[low, high]`.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/11/05/bst1.jpg)

**Input:** root = [10,5,15,3,7,null,18], low = 7, high = 15
**Output:** 32
**Explanation:** Nodes 7, 10, and 15 are in the range [7, 15]. 7 + 10 + 15 = 32.

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/11/05/bst2.jpg)

**Input:** root = [10,5,15,3,7,13,18,1,null,6], low = 6, high = 10
**Output:** 23
**Explanation:** Nodes 6, 7, and 10 are in the range [6, 10]. 6 + 7 + 10 = 23.


```ruby
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
        def rangeSumBST(self, root: TreeNode, L: int, R: int) -> int:
            if not root:
                return 0
            elif root.val < L:
                return self.rangeSumBST(root.right, L, R)
            elif root.val > R:
                return self.rangeSumBST(root.left, L, R)
            return root.val + self.rangeSumBST(root.left, L, R) + self.rangeSumBST(root.right, L, R)
```


**965. Univalued Binary Tree**

A binary tree is **uni-valued** if every node in the tree has the same value.

Given the `root` of a binary tree, return `true` _if the given tree is **uni-valued**, or_ `false` _otherwise._

**Example 1:**

![](https://assets.leetcode.com/uploads/2018/12/28/unival_bst_1.png)

**Input:** root = [1,1,1,1,1,null,1]
**Output:** true

**Example 2:**

![](https://assets.leetcode.com/uploads/2018/12/28/unival_bst_2.png)

**Input:** root = [2,2,2,5,2]
**Output:** false

```kotlin
class Solution:
    def in_order(self, root, val):
        if root:
            return self.in_order(root.left, root.val) and self.in_order(root.right, root.val) and root.val == val
        else:
            return True
    def isUnivalTree(self, root: Optional[TreeNode]) -> bool:
        return self.in_order(root.left, root.val) and self.in_order(root.right, root.val)
```


**993. Cousins in Binary Tree**

Given the `root` of a binary tree with unique values and the values of two different nodes of the tree `x` and `y`, return `true` _if the nodes corresponding to the values_ `x` _and_ `y` _in the tree are **cousins**, or_ `false` _otherwise._

Two nodes of a binary tree are **cousins** if they have the same depth with different parents.

Note that in a binary tree, the root node is at the depth `0`, and children of each depth `k` node are at the depth `k + 1`.

**Example 1:**

![](https://assets.leetcode.com/uploads/2019/02/12/q1248-01.png)

**Input:** root = [1,2,3,4], x = 4, y = 3
**Output:** false

**Example 2:**

![](https://assets.leetcode.com/uploads/2019/02/12/q1248-02.png)

**Input:** root = [1,2,3,null,4,null,5], x = 5, y = 4
**Output:** true

**Example 3:**

![](https://assets.leetcode.com/uploads/2019/02/13/q1248-03.png)

**Input:** root = [1,2,3,null,4], x = 2, y = 3
**Output:** false


```ruby
class Solution:
    def level_order(self, root, parent, height):
        if root:
            self.map[root.val] = (parent, height)
            self.level_order(root.left, root, height + 1)
            self.level_order(root.right, root, height + 1)

    def isCousins(self, root: Optional[TreeNode], x: int, y: int) -> bool:
        self.map = {}
        self.level_order(root, None, 0)
        x_parent, x_depth = self.map[x]
        y_parent, y_depth = self.map[y]
        return (x_depth == y_depth) and (x_parent != y_parent)
```

**1022. Sum of Root To Leaf Binary Numbers**


You are given the `root` of a binary tree where each node has a value `0` or `1`. Each root-to-leaf path represents a binary number starting with the most significant bit.

-   For example, if the path is `0 -> 1 -> 1 -> 0 -> 1`, then this could represent `01101` in binary, which is `13`.

For all leaves in the tree, consider the numbers represented by the path from the root to that leaf. Return _the sum of these numbers_.

The test cases are generated so that the answer fits in a **32-bits** integer.

**Example 1:**

![](https://assets.leetcode.com/uploads/2019/04/04/sum-of-root-to-leaf-binary-numbers.png)

**Input:** root = [1,0,1,0,1,0,1]
**Output:** 22
**Explanation:** (100) + (101) + (110) + (111) = 4 + 5 + 6 + 7 = 22

**Example 2:**

**Input:** root = [0]
**Output:** 0


```ruby
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
        self.final = []

        def dfs(self,root,path):
            if not root:
                return None

            if not root.left and not root.right:
                self.final.append(int(''.join(path),2))
            
            if root.left:
                dfs(self,root.left,path+[str(root.left.val)])
            
            if root.right:
                dfs(self,root.right,path+[str(root.right.val)])
            

        dfs(self,root,[str(root.val)])

        #print(self.final)
        return sum(self.final)

```


**1379. Find a Corresponding Node of a Binary Tree in a Clone of That Tree**


Given two binary trees `original` and `cloned` and given a reference to a node `target` in the original tree.

The `cloned` tree is a **copy of** the `original` tree.

Return _a reference to the same node_ in the `cloned` tree.

**Note** that you are **not allowed** to change any of the two trees or the `target` node and the answer **must be** a reference to a node in the `cloned` tree.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/02/21/e1.png)

**Input:** tree = [7,4,3,null,null,6,19], target = 3
**Output:** 3
**Explanation:** In all examples the original and cloned trees are shown. The target node is a green node from the original tree. The answer is the yellow node from the cloned tree.

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/02/21/e2.png)

**Input:** tree = [7], target =  7
**Output:** 7

**Example 3:**

![](https://assets.leetcode.com/uploads/2020/02/21/e3.png)

**Input:** tree = [8,null,6,null,5,null,4,null,3,null,2,null,1], target = 4
**Output:** 4


```php
def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
    # the basic idea is to iterate over both of the binary trees and then return the target node in the cloned
    
    que = deque()
    
    que.append((original,cloned))
    
    while que:
        origion, clone = que.popleft()
        
        if origion == target:
            return clone
        
        if origion.left:
            que.append((origion.left,clone.left))
        
        if origion.right:
            que.append((origion.right,clone.right))
```



**2236. Root Equals Sum of Children**

You are given the `root` of a **binary tree** that consists of exactly `3` nodes: the root, its left child, and its right child.

Return `true` _if the value of the root is equal to the **sum** of the values of its two children, or_ `false` _otherwise_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2022/04/08/graph3drawio.png)

**Input:** root = [10,4,6]
**Output:** true
**Explanation:** The values of the root, its left child, and its right child are 10, 4, and 6, respectively.
10 is equal to 4 + 6, so we return true.

**Example 2:**

![](https://assets.leetcode.com/uploads/2022/04/08/graph3drawio-1.png)

**Input:** root = [5,3,1]
**Output:** false
**Explanation:** The values of the root, its left child, and its right child are 5, 3, and 1, respectively.
5 is not equal to 3 + 1, so we return false.


```kotlin
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def checkTree(self, root: Optional[TreeNode]) -> bool:
        if root.val==root.left.val + root.right.val:
            return True
        else:
            return False
```


**2331. Evaluate Boolean Binary Tree**


You are given the `root` of a **full binary tree** with the following properties:

-   **Leaf nodes** have either the value `0` or `1`, where `0` represents `False` and `1` represents `True`.
-   **Non-leaf nodes** have either the value `2` or `3`, where `2` represents the boolean `OR` and `3` represents the boolean `AND`.

The **evaluation** of a node is as follows:

-   If the node is a leaf node, the evaluation is the **value** of the node, i.e. `True` or `False`.
-   Otherwise, **evaluate** the node's two children and **apply** the boolean operation of its value with the children's evaluations.

Return _the boolean result of **evaluating** the_ `root` _node._

A **full binary tree** is a binary tree where each node has either `0` or `2` children.

A **leaf node** is a node that has zero children.

**Example 1:**

![](https://assets.leetcode.com/uploads/2022/05/16/example1drawio1.png)

**Input:** root = [2,1,3,null,null,0,1]
**Output:** true
**Explanation:** The above diagram illustrates the evaluation process.
The AND node evaluates to False AND True = False.
The OR node evaluates to True OR False = True.
The root node evaluates to True, so we return true.

**Example 2:**

**Input:** root = [0]
**Output:** false
**Explanation:** The root node is a leaf node and it evaluates to false, so we return false.


```ruby
class Solution:
    def evaluateTree(self, root: Optional[TreeNode]) -> bool:
        if root.val==3:
            return self.evaluateTree(root.left) and self.evaluateTree(root.right)
        elif root.val==2:
            return self.evaluateTree(root.left) or self.evaluateTree(root.right)
        return root.val
```






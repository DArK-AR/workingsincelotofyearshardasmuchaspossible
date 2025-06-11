# OOPS algo

class User:
    def __init__(self, user_id, name, email):
        self.user_id = user_id
        self.name = name
        self.email = email

    def __str__(self):
        return f"User({self.user_id}, {self.name}, {self.email})"


class UserService:
    def __init__(self):
        self.users = {}  # Dictionary to store users with user_id as the key

    def create_user(self, user_id, name, email):
        if user_id in self.users:
            raise Exception("User ID already exists.")
        self.users[user_id] = User(user_id, name, email)
        return self.users[user_id]

    def read_user(self, user_id):
        if user_id not in self.users:
            raise Exception("User not found.")
        return self.users[user_id]

    def update_user(self, user_id, name=None, email=None):
        if user_id not in self.users:
            raise Exception("User not found.")
        if name:
            self.users[user_id].name = name
        if email:
            self.users[user_id].email = email
        return self.users[user_id]

    def delete_user(self, user_id):
        if user_id not in self.users:
            raise Exception("User not found.")
        return self.users.pop(user_id)

# Example Usage
if __name__ == "__main__":
    service = UserService()

    # Create user
    user = service.create_user(1, "Alice", "alice@example.com")
    print("Created:", user)

    # Read user
    user = service.read_user(1)
    print("Read:", user)

    # Update user
    user = service.update_user(1, name="Alice Smith")
    print("Updated:", user)

    # Delete user
    deleted_user = service.delete_user(1)
    print("Deleted:", deleted_user)


# Graph

class Vertex:
    def __init__(self, value):
        self.value = value
        self.neighbors = []

    def add_neighbor(self, vertex):
        self.neighbors.append(vertex)

    def remove_neighbor(self, vertex):
        if vertex in self.neighbors:
            self.neighbors.remove(vertex)

class Graph:
    def __init__(self):
        self.vertices = {}

    def add_vertex(self, value):
        if value not in self.vertices:
            self.vertices[value] = Vertex(value)

    def remove_vertex(self, value):
        if value in self.vertices:
            for vertex in self.vertices.values():
                vertex.remove_neighbor(self.vertices[value])
            del self.vertices[value]

    def add_edge(self, value1, value2):
        if value1 in self.vertices and value2 in self.vertices:
            self.vertices[value1].add_neighbor(self.vertices[value2])
            self.vertices[value2].add_neighbor(self.vertices[value1])

    def remove_edge(self, value1, value2):
        if value1 in self.vertices and value2 in self.vertices:
            self.vertices[value1].remove_neighbor(self.vertices[value2])
            self.vertices[value2].remove_neighbor(self.vertices[value1])

    def depth_first_search(self, start_value):
        visited = set()
        stack = [self.vertices[start_value]]
        result = []
        while stack:
            current = stack.pop()
            if current.value not in visited:
                visited.add(current.value)
                result.append(current.value)
                for neighbor in current.neighbors:
                    stack.append(neighbor)
        return result

    def breadth_first_search(self, start_value):
        visited = set()
        queue = [self.vertices[start_value]]
        result = []
        while queue:
            current = queue.pop(0)
            if current.value not in visited:
                visited.add(current.value)
                result.append(current.value)
                for neighbor in current.neighbors:
                    queue.append(neighbor)
        return result

    def get_vertices(self):
        return list(self.vertices.keys())

    def get_edges(self):
        edges = []
        for vertex in self.vertices.values():
            for neighbor in vertex.neighbors:
                edges.append((vertex.value, neighbor.value))
        return edges

    def __str__(self):
        graph_representation = {}
        for value, vertex in self.vertices.items():
            graph_representation[value] = [neighbor.value for neighbor in vertex.neighbors]
        return str(graph_representation)

# Example 

if __name__ == "__main__":
    graph = Graph()

    # Adding vertices
    graph.add_vertex("A")
    graph.add_vertex("B")
    graph.add_vertex("C")
    graph.add_vertex("D")

    # Adding edges
    graph.add_edge("A", "B")
    graph.add_edge("A", "C")
    graph.add_edge("B", "D")

    # Displaying the graph
    print("Graph representation:", graph)

    # Performing Depth-First Search (DFS)
    dfs_result = graph.depth_first_search("A")
    print("DFS starting from A:", dfs_result)

    # Performing Breadth-First Search (BFS)
    bfs_result = graph.breadth_first_search("A")
    print("BFS starting from A:", bfs_result)


# Tree 

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.root = None

    def add(self, value):
        if not self.root:
            self.root = TreeNode(value)
        else:
            self._add_recursive(self.root, value)

    def _add_recursive(self, node, value):
        if value < node.value:
            if node.left:
                self._add_recursive(node.left, value)
            else:
                node.left = TreeNode(value)
        else:
            if node.right:
                self._add_recursive(node.right, value)
            else:
                node.right = TreeNode(value)

    def search(self, value):
        return self._search_recursive(self.root, value)

    def _search_recursive(self, node, value):
        if not node:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)

    def pre_order_traversal(self):
        result = []
        self._pre_order_recursive(self.root, result)
        return result

    def _pre_order_recursive(self, node, result):
        if node:
            result.append(node.value)
            self._pre_order_recursive(node.left, result)
            self._pre_order_recursive(node.right, result)

    def in_order_traversal(self):
        result = []
        self._in_order_recursive(self.root, result)
        return result

    def _in_order_recursive(self, node, result):
        if node:
            self._in_order_recursive(node.left, result)
            result.append(node.value)
            self._in_order_recursive(node.right, result)

    def post_order_traversal(self):
        result = []
        self._post_order_recursive(self.root, result)
        return result

    def _post_order_recursive(self, node, result):
        if node:
            self._post_order_recursive(node.left, result)
            self._post_order_recursive(node.right, result)
            result.append(node.value)

if __name__ == "__main__":
    # Initialize the tree
    tree = BinaryTree()

    # Add nodes
    tree.add(50)
    tree.add(30)
    tree.add(70)
    tree.add(20)
    tree.add(40)
    tree.add(60)
    tree.add(80)

    # Search for a value
    print("Search 40:", tree.search(40))  # Output: True
    print("Search 100:", tree.search(100))  # Output: False

    # Traversals
    print("Pre-order Traversal:", tree.pre_order_traversal())  # Output: [50, 30, 20, 40, 70, 60, 80]
    print("In-order Traversal:", tree.in_order_traversal())  # Output: [20, 30, 40, 50, 60, 70, 80]
    print("Post-order Traversal:", tree.post_order_traversal())  # Output: [20, 40, 30, 60, 80, 70, 50]


# Two Pointer

class TwoPointer:
    def __init__(self, data):
        self.data = sorted(data)

    def find_pair_with_sum(self, target_sum):
        left = 0
        right = len(self.data) - 1
        result = []
        while left < right:
            current_sum = self.data[left] + self.data[right]
            if current_sum == target_sum:
                result.append((self.data[left], self.data[right]))
                left += 1
                right -= 1
            elif current_sum < target_sum:
                left += 1
            else:
                right -= 1
        return result if result else None

    def find_triplets_with_sum(self, target_sum):
        result = []
        for i in range(len(self.data)):
            left = i + 1
            right = len(self.data) - 1
            while left < right:
                current_sum = self.data[i] + self.data[left] + self.data[right]
                if current_sum == target_sum:
                    result.append((self.data[i], self.data[left], self.data[right]))
                    left += 1
                    right -= 1
                elif current_sum < target_sum:
                    left += 1
                else:
                    right -= 1
        return result if result else None

    def find_subarray_with_sum(self, target_sum):
        left = 0
        current_sum = 0
        for right in range(len(self.data)):
            current_sum += self.data[right]
            while current_sum > target_sum and left <= right:
                current_sum -= self.data[left]
                left += 1
            if current_sum == target_sum:
                return self.data[left:right + 1]
        return None

    def closest_pair_to_target(self, target_sum):
        left = 0
        right = len(self.data) - 1
        closest_pair = None
        closest_difference = float('inf')
        while left < right:
            current_sum = self.data[left] + self.data[right]
            difference = abs(target_sum - current_sum)
            if difference < closest_difference:
                closest_difference = difference
                closest_pair = (self.data[left], self.data[right])
            if current_sum < target_sum:
                left += 1
            else:
                right -= 1
        return closest_pair


if __name__ == "__main__":
    data = [10, 22, 28, 29, 30, 40]
    target_sum = 54

    tp = TwoPointer(data)

    print("Array:", data)
    print("\nPairs with Sum:")
    print(tp.find_pair_with_sum(target_sum))

    print("\nTriplets with Sum:")
    print(tp.find_triplets_with_sum(target_sum))

    print("\nSubarray with Sum:")
    print(tp.find_subarray_with_sum(target_sum))

    print("\nClosest Pair to Target:")
    print(tp.closest_pair_to_target(target_sum))


# Sliding window 

class SlidingWindow:
    def __init__(self, data):
        self.data = data

    def max_subarray_sum(self, k):
        if len(self.data) < k:
            raise ValueError("Window size must be smaller than or equal to the size of the data.")
        window_sum = 0
        max_sum = float('-inf')
        for i in range(len(self.data)):
            window_sum += self.data[i]
            if i >= k - 1:
                max_sum = max(max_sum, window_sum)
                window_sum -= self.data[i - (k - 1)]
        return max_sum

    def longest_unique_substring(self):
        char_set = set()
        left = 0
        max_length = 0
        for right in range(len(self.data)):
            while self.data[right] in char_set:
                char_set.remove(self.data[left])
                left += 1
            char_set.add(self.data[right])
            max_length = max(max_length, right - left + 1)
        return max_length

    def count_subarrays_with_sum(self, target_sum):
        count = 0
        current_sum = 0
        prefix_sum = {0: 1}
        for i in range(len(self.data)):
            current_sum += self.data[i]
            if current_sum - target_sum in prefix_sum:
                count += prefix_sum[current_sum - target_sum]
            prefix_sum[current_sum] = prefix_sum.get(current_sum, 0) + 1
        return count

    def smallest_subarray_with_sum(self, target_sum):
        min_length = float('inf')
        current_sum = 0
        left = 0
        for right in range(len(self.data)):
            current_sum += self.data[right]
            while current_sum >= target_sum:
                min_length = min(min_length, right - left + 1)
                current_sum -= self.data[left]
                left += 1
        return min_length if min_length != float('inf') else 0

if __name__ == "__main__":
    data_array = [2, 1, 5, 2, 3, 2]
    target_sum = 7
    window_size = 3
    data_string = "abcabcbb"

    sliding_window = SlidingWindow(data_array)
    print("Maximum Subarray Sum:", sliding_window.max_subarray_sum(window_size))
    sliding_window_string = SlidingWindow(data_string)
    print("Longest Unique Substring Length:", sliding_window_string.longest_unique_substring())
    print("Count of Subarrays with Sum:", sliding_window.count_subarrays_with_sum(target_sum))
    print("Smallest Subarray with Sum:", sliding_window.smallest_subarray_with_sum(target_sum))


# Dynamic Programming 
# A coumputer programming technique that breaks down problems into smaller, more manageable subproblems.


import heapq
from datetime import datetime, timedelta

class Task:
    def __init__(self, task_id, description, priority, execution_time):
        self.task_id = task_id
        self.description = description
        self.priority = priority  # Lower value means higher priority
        self.execution_time = execution_time

    def __lt__(self, other):
        return self.priority < other.priority

    def execute(self):
        print(f"Executing Task {self.task_id}: {self.description} (Priority: {self.priority})")

class TaskScheduler:
    def __init__(self):
        self.task_queue = []
        self.task_history = []

    def add_task(self, task_id, description, priority, delay_seconds):
        execution_time = datetime.now() + timedelta(seconds=delay_seconds)
        task = Task(task_id, description, priority, execution_time)
        heapq.heappush(self.task_queue, (execution_time, task))
        print(f"Task {task_id} scheduled at {execution_time} with priority {priority}.")

    def remove_task(self, task_id):
        self.task_queue = [(time, task) for time, task in self.task_queue if task.task_id != task_id]
        heapq.heapify(self.task_queue)
        print(f"Task {task_id} removed from the queue.")

    def execute_tasks(self):
        now = datetime.now()
        while self.task_queue and self.task_queue[0][0] <= now:
            _, task = heapq.heappop(self.task_queue)
            task.execute()
            self.task_history.append(task)
        print("All tasks ready for execution have been processed.")

    def get_pending_tasks(self):
        return [(task.task_id, task.description, task.priority, task.execution_time) for _, task in self.task_queue]

    def get_task_history(self):
        return [(task.task_id, task.description, task.priority, task.execution_time) for task in self.task_history]


if __name__ == "__main__":
    scheduler = TaskScheduler()

    # Add tasks with varying priorities and delays
    scheduler.add_task(task_id=1, description="Backup Database", priority=2, delay_seconds=3)
    scheduler.add_task(task_id=2, description="Send Email Notifications", priority=1, delay_seconds=5)
    scheduler.add_task(task_id=3, description="Clean Temporary Files", priority=3, delay_seconds=1)

    # Simulate waiting for tasks to execute
    import time
    time.sleep(6)

    # Execute tasks
    scheduler.execute_tasks()

    # Display pending tasks
    print("\nPending Tasks:")
    for task in scheduler.get_pending_tasks():
        print(task)

    # Display task history
    print("\nTask History:")
    for task in scheduler.get_task_history():
        print(task)


# Top K element 

import heapq

class TopKElements:
    def __init__(self, data):
        self.data = data

    def find_top_k_largest(self, k):
        if k > len(self.data):
            raise ValueError("K must not be greater than the size of the data.")
        return heapq.nlargest(k, self.data)

    def find_top_k_smallest(self, k):
        if k > len(self.data):
            raise ValueError("K must not be greater than the size of the data.")
        return heapq.nsmallest(k, self.data)

    def find_k_frequent_elements(self, k):
        frequency_map = {}
        for num in self.data:
            frequency_map[num] = frequency_map.get(num, 0) + 1
        
        frequency_heap = [(-freq, num) for num, freq in frequency_map.items()]
        heapq.heapify(frequency_heap)

        result = []
        for _ in range(k):
            if frequency_heap:
                freq, num = heapq.heappop(frequency_heap)
                result.append(num)
        return result

    def find_top_k_with_custom_key(self, k, key=lambda x: x):
        if k > len(self.data):
            raise ValueError("K must not be greater than the size of the data.")
        return heapq.nlargest(k, self.data, key=key)


if __name__ == "__main__":
    data_array = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    top_k = TopKElements(data_array)

    # Finding Top K Largest Elements
    k = 3
    print("Top K Largest Elements:", top_k.find_top_k_largest(k))

    # Finding Top K Smallest Elements
    print("Top K Smallest Elements:", top_k.find_top_k_smallest(k))

    # Finding K Most Frequent Elements
    print("Top K Frequent Elements:", top_k.find_k_frequent_elements(k))

    # Finding Top K Elements with Custom Key
    data_objects = [{'value': 10}, {'value': 20}, {'value': 15}]
    top_k_objects = TopKElements(data_objects)
    print("Top K Largest Objects by 'value':", 
          top_k_objects.find_top_k_with_custom_key(k=2, key=lambda x: x['value']))


# Prefix Sum 

class PrefixSum:
    def __init__(self, data):
        self.data = data
        self.prefix_sums = []
        self._calculate_prefix_sums()

    def _calculate_prefix_sums(self):
        current_sum = 0
        for num in self.data:
            current_sum += num
            self.prefix_sums.append(current_sum)

    def range_sum(self, left, right):
        if left < 0 or right >= len(self.data) or left > right:
            raise ValueError("Invalid range provided.")
        if left == 0:
            return self.prefix_sums[right]
        return self.prefix_sums[right] - self.prefix_sums[left - 1]

    def update(self, index, value):
        if index < 0 or index >= len(self.data):
            raise IndexError("Index out of bounds.")
        original_value = self.data[index]
        self.data[index] = value
        difference = value - original_value
        for i in range(index, len(self.data)):
            self.prefix_sums[i] += difference

    def find_equal_sum_subarrays(self, target):
        results = []
        for i in range(len(self.data)):
            for j in range(i, len(self.data)):
                if self.range_sum(i, j) == target:
                    results.append((i, j))
        return results

    def max_subarray_sum(self):
        max_sum = float('-inf')
        min_prefix = 0
        for prefix in self.prefix_sums:
            max_sum = max(max_sum, prefix - min_prefix)
            min_prefix = min(min_prefix, prefix)
        return max_sum

    def print_prefix_sums(self):
        return self.prefix_sums

if __name__ == "__main__":
    data = [2, 4, 6, 8, 10]
    prefix_sum = PrefixSum(data)

    # Print the calculated prefix sums
    print("Prefix Sums:", prefix_sum.print_prefix_sums())

    # Query the range sum from index 1 to 3
    print("Range Sum (1 to 3):", prefix_sum.range_sum(1, 3))

    # Update the value at index 2
    print("Update index 2 to value 5")
    prefix_sum.update(2, 5)
    print("Updated Prefix Sums:", prefix_sum.print_prefix_sums())

    # Query the range sum again after update
    print("Range Sum (1 to 3) after update:", prefix_sum.range_sum(1, 3))

    # Find all subarrays with a target sum
    target_sum = 10
    print(f"Subarrays with sum {target_sum}:", prefix_sum.find_equal_sum_subarrays(target_sum))

    # Find the maximum subarray sum
    print("Maximum Subarray Sum:", prefix_sum.max_subarray_sum())



# Stack 

class Stack:
    def __init__(self, capacity=None):
        self.stack = []
        self.capacity = capacity

    def push(self, item):
        if self.capacity and len(self.stack) >= self.capacity:
            raise OverflowError("Stack is full.")
        self.stack.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Cannot pop from empty stack.")
        return self.stack.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Cannot peek in an empty stack.")
        return self.stack[-1]

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)

    def clear(self):
        self.stack = []

    def get_elements(self):
        return list(self.stack)

class StackCalculator:
    def __init__(self):
        self.stack = Stack()

    def evaluate_postfix_expression(self, expression):
        for char in expression:
            if char.isdigit():
                self.stack.push(int(char))
            else:
                operand2 = self.stack.pop()
                operand1 = self.stack.pop()
                if char == '+':
                    self.stack.push(operand1 + operand2)
                elif char == '-':
                    self.stack.push(operand1 - operand2)
                elif char == '*':
                    self.stack.push(operand1 * operand2)
                elif char == '/':
                    self.stack.push(operand1 // operand2)
        return self.stack.pop()

    def reverse_stack(self):
        temp_stack = Stack()
        while not self.stack.is_empty():
            temp_stack.push(self.stack.pop())
        self.stack = temp_stack

    def sort_stack(self):
        temp_stack = Stack()
        while not self.stack.is_empty():
            current = self.stack.pop()
            while not temp_stack.is_empty() and temp_stack.peek() > current:
                self.stack.push(temp_stack.pop())
            temp_stack.push(current)
        self.stack = temp_stack

if __name__ == "__main__":
    s = Stack(capacity=10)
    s.push(5)
    s.push(3)
    s.push(8)
    print("Stack elements:", s.get_elements())
    print("Popped element:", s.pop())
    print("Peek element:", s.peek())
    print("Stack size:", s.size())
    s.clear()
    print("Stack after clearing:", s.get_elements())

    calc = StackCalculator()
    postfix_expr = "53+82-*"
    print("Postfix expression result:", calc.evaluate_postfix_expression(postfix_expr))
    calc.stack.push(4)
    calc.stack.push(7)
    calc.stack.push(1)
    calc.stack.push(9)
    print("Stack before sorting:", calc.stack.get_elements())
    calc.sort_stack()
    print("Stack after sorting:", calc.stack.get_elements())

# B+ Tree 

class BPlusTreeNode:
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.keys = []
        self.children = []

class BPlusTree:
    def __init__(self, order):
        self.root = BPlusTreeNode(is_leaf=True)
        self.order = order

    def search(self, key, node=None):
        node = node or self.root
        if node.is_leaf:
            for i, item in enumerate(node.keys):
                if key == item:
                    return True
                elif key < item:
                    return False
            return False
        else:
            for i, item in enumerate(node.keys):
                if key < item:
                    return self.search(key, node.children[i])
            return self.search(key, node.children[-1])

    def insert(self, key):
        root = self.root
        if len(root.keys) == self.order - 1:
            new_root = BPlusTreeNode()
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key)

    def _insert_non_full(self, node, key):
        if node.is_leaf:
            i = 0
            while i < len(node.keys) and key > node.keys[i]:
                i += 1
            node.keys.insert(i, key)
        else:
            i = 0
            while i < len(node.keys) and key > node.keys[i]:
                i += 1
            if len(node.children[i].keys) == self.order - 1:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key)

    def _split_child(self, parent, index):
        order = self.order
        node = parent.children[index]
        new_node = BPlusTreeNode(is_leaf=node.is_leaf)

        mid = len(node.keys) // 2
        if node.is_leaf:
            new_node.keys = node.keys[mid:]
            node.keys = node.keys[:mid]
            new_node.children = node.children[mid:]
            node.children = node.children[:mid]
            new_node.children.append(None)
        else:
            parent.keys.insert(index, node.keys[mid])
            new_node.keys = node.keys[mid + 1:]
            node.keys = node.keys[:mid]
            new_node.children = node.children[mid + 1:]
            node.children = node.children[:mid + 1]

        parent.children.insert(index + 1, new_node)

    def display(self, node=None, level=0):
        node = node or self.root
        print("Level", level, "Keys:", node.keys)
        if not node.is_leaf:
            for child in node.children:
                self.display(child, level + 1)


if __name__ == "__main__":
    # Create a B+ Tree with order 4
    bptree = BPlusTree(order=4)

    # Insert keys into the B+ Tree
    keys_to_insert = [10, 20, 5, 6, 12, 30, 7, 17]
    for key in keys_to_insert:
        bptree.insert(key)

    # Display the tree structure
    print("B+ Tree Structure:")
    bptree.display()

    # Search for keys
    print("Search for key 6:", bptree.search(6))  # Output: True
    print("Search for key 15:", bptree.search(15))  # Output: False


# Linked List

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def prepend(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def insert_after_node(self, prev_node_data, data):
        if not self.head:
            return
        current = self.head
        while current and current.data != prev_node_data:
            current = current.next
        if not current:
            raise ValueError("The given previous node is not in the list.")
        new_node = Node(data)
        new_node.next = current.next
        current.next = new_node

    def delete_node(self, data):
        if not self.head:
            return
        if self.head.data == data:
            self.head = self.head.next
            return
        current = self.head
        while current.next and current.next.data != data:
            current = current.next
        if not current.next:
            raise ValueError("The data to delete is not found in the list.")
        current.next = current.next.next

    def delete_at_position(self, position):
        if not self.head:
            return
        if position == 0:
            self.head = self.head.next
            return
        current = self.head
        for _ in range(position - 1):
            if not current.next:
                raise IndexError("The position is out of range.")
            current = current.next
        if not current.next:
            raise IndexError("The position is out of range.")
        current.next = current.next.next

    def length(self):
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count

    def reverse(self):
        previous = None
        current = self.head
        while current:
            next_node = current.next
            current.next = previous
            previous = current
            current = next_node
        self.head = previous

    def find(self, data):
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False

    def get_position(self, data):
        current = self.head
        position = 0
        while current:
            if current.data == data:
                return position
            current = current.next
            position += 1
        raise ValueError("The data is not found in the list.")

    def print_list(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return elements


if __name__ == "__main__":
    # Initialize the linked list
    ll = LinkedList()

    # Append elements to the list
    ll.append(10)
    ll.append(20)
    ll.append(30)
    print("Linked List after appending 10, 20, 30:", ll.print_list())

    # Prepend an element
    ll.prepend(5)
    print("Linked List after prepending 5:", ll.print_list())

    # Insert after a specific node
    ll.insert_after_node(10, 15)
    print("Linked List after inserting 15 after 10:", ll.print_list())

    # Delete a specific node
    ll.delete_node(20)
    print("Linked List after deleting 20:", ll.print_list())

    # Delete a node at a specific position
    ll.delete_at_position(2)
    print("Linked List after deleting at position 2:", ll.print_list())

    # Reverse the linked list
    ll.reverse()
    print("Reversed Linked List:", ll.print_list())

    # Check the length of the list
    print("Length of Linked List:", ll.length())

    # Find a value in the list
    print("Is 10 in the list?", ll.find(10))
    print("Is 50 in the list?", ll.find(50))

    # Get the position of a value
    print("Position of 5 in the list:", ll.get_position(5))


# Binary Search 


class BinarySearch:
    def __init__(self, data):
        self.data = sorted(data)

    def iterative_search(self, target):
        left = 0
        right = len(self.data) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.data[mid] == target:
                return mid
            elif self.data[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    def recursive_search(self, target):
        return self._recursive_search_helper(target, 0, len(self.data) - 1)

    def _recursive_search_helper(self, target, left, right):
        if left > right:
            return -1
        mid = (left + right) // 2
        if self.data[mid] == target:
            return mid
        elif self.data[mid] < target:
            return self._recursive_search_helper(target, mid + 1, right)
        else:
            return self._recursive_search_helper(target, left, mid - 1)

    def find_range(self, target):
        start = self._find_first_occurrence(target)
        end = self._find_last_occurrence(target)
        if start == -1:
            return (-1, -1)
        return (start, end)

    def _find_first_occurrence(self, target):
        left = 0
        right = len(self.data) - 1
        result = -1
        while left <= right:
            mid = (left + right) // 2
            if self.data[mid] == target:
                result = mid
                right = mid - 1
            elif self.data[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return result

    def _find_last_occurrence(self, target):
        left = 0
        right = len(self.data) - 1
        result = -1
        while left <= right:
            mid = (left + right) // 2
            if self.data[mid] == target:
                result = mid
                left = mid + 1
            elif self.data[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return result

    def is_present(self, target):
        return self.iterative_search(target) != -1

    def count_occurrences(self, target):
        range_indices = self.find_range(target)
        if range_indices == (-1, -1):
            return 0
        return range_indices[1] - range_indices[0] + 1

    def print_data(self):
        return self.data


if __name__ == "__main__":
    # Input data for binary search
    data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    target = 50

    # Initialize the BinarySearch class
    binary_search = BinarySearch(data)

    # Print sorted data
    print("Data:", binary_search.print_data())

    # Iterative Search
    index_iterative = binary_search.iterative_search(target)
    print(f"Iterative Search: Element {target} found at index {index_iterative}")

    # Recursive Search
    index_recursive = binary_search.recursive_search(target)
    print(f"Recursive Search: Element {target} found at index {index_recursive}")

    # Range Query
    range_indices = binary_search.find_range(target)
    print(f"Range of Element {target}: {range_indices}")

    # Count Occurrences
    count = binary_search.count_occurrences(target)
    print(f"Element {target} occurs {count} times")

    # Check if Element is Present
    presence = binary_search.is_present(target)
    print(f"Is Element {target} present?: {presence}")

    # Test for a missing element
    missing_target = 100
    print(f"Search for missing element {missing_target}: {binary_search.iterative_search(missing_target)}")

# Greedy Algo 

# A problem-solving technique that chooses the best option at each step to get to an approximate solution.
class Activity:
    def __init__(self, name, start, end, location=None, priority=0):
        self.name = name
        self.start = start
        self.end = end
        self.location = location
        self.priority = priority

    def duration(self):
        return self.end - self.start

    def __repr__(self):
        return (f"Activity(Name: {self.name}, Start: {self.start}, End: {self.end},"
                f"Location: {self.location}, Priority: {self.priority})")


class ActivitySelector:
    def __init__(self, activities):
        self.activities = sorted(activities, key=lambda x: (x.end, -x.priority))

    def select_activities(self):
        selected_activities = []
        last_end_time = 0

        for activity in self.activities:
            if activity.start >= last_end_time:
                selected_activities.append(activity)
                last_end_time = activity.end

        return selected_activities

    def select_highest_priority(self):
        selected_activities = []
        last_end_time = 0

        for activity in sorted(self.activities, key=lambda x: (-x.priority, x.end)):
            if activity.start >= last_end_time:
                selected_activities.append(activity)
                last_end_time = activity.end

        return selected_activities

    def find_longest_activity(self):
        if not self.activities:
            return None
        return max(self.activities, key=lambda x: x.duration())

    def filter_by_location(self, location):
        return [activity for activity in self.activities if activity.location == location]

    def find_conflicting_activities(self):
        conflicts = []
        for i in range(len(self.activities)):
            for j in range(i + 1, len(self.activities)):
                if self.activities[j].start < self.activities[i].end:
                    conflicts.append((self.activities[i], self.activities[j]))
        return conflicts

    def print_activities(self, activities):
        for activity in activities:
            print(f"Selected: {activity.name} (Start: {activity.start}, End: {activity.end}, "
                  f"Location: {activity.location}, Priority: {activity.priority})")


if __name__ == "__main__":
    # Define activities with additional parameters
    activity_list = [
        Activity("A1", 1, 3, location="Room1", priority=2),
        Activity("A2", 2, 5, location="Room2", priority=3),
        Activity("A3", 4, 7, location="Room1", priority=1),
        Activity("A4", 1, 8, location="Room3", priority=5),
        Activity("A5", 8, 10, location="Room1", priority=4),
        Activity("A6", 9, 11, location="Room2", priority=1),
    ]

    # Initialize ActivitySelector
    selector = ActivitySelector(activity_list)

    # Select non-overlapping activities based on end time
    selected = selector.select_activities()
    print("Selected Activities Based on End Time:")
    selector.print_activities(selected)

    # Select highest-priority non-overlapping activities
    highest_priority_selected = selector.select_highest_priority()
    print("\nSelected Activities Based on Priority:")
    selector.print_activities(highest_priority_selected)

    # Find the longest activity
    longest_activity = selector.find_longest_activity()
    print("\nLongest Activity:")
    print(longest_activity)

    # Filter activities by location
    room1_activities = selector.filter_by_location("Room1")
    print("\nActivities in Room1:")
    selector.print_activities(room1_activities)

    # Find conflicting activities
    conflicts = selector.find_conflicting_activities()
    print("\nConflicting Activities:")
    for conflict in conflicts:
        print(f"Conflict: {conflict[0].name} overlaps with {conflict[1].name}")


# Quick Sort 

class QuickSort:
    def __init__(self, data):
        self.data = data

    def sort(self, ascending=True):
        self._quick_sort(0, len(self.data) - 1, ascending)

    def _quick_sort(self, low, high, ascending):
        if low < high:
            partition_index = self._partition(low, high, ascending)
            self._quick_sort(low, partition_index - 1, ascending)
            self._quick_sort(partition_index + 1, high, ascending)

    def _partition(self, low, high, ascending):
        pivot = self.data[high]
        i = low - 1
        for j in range(low, high):
            if (self.data[j] <= pivot and ascending) or (self.data[j] >= pivot and not ascending):
                i += 1
                self.data[i], self.data[j] = self.data[j], self.data[i]
        self.data[i + 1], self.data[high] = self.data[high], self.data[i + 1]
        return i + 1

    def find_kth_smallest(self, k):
        if k < 1 or k > len(self.data):
            raise ValueError("k is out of range")
        return self._kth_smallest(0, len(self.data) - 1, k - 1)

    def _kth_smallest(self, low, high, k):
        if low <= high:
            partition_index = self._partition(low, high, ascending=True)
            if partition_index == k:
                return self.data[partition_index]
            elif partition_index > k:
                return self._kth_smallest(low, partition_index - 1, k)
            else:
                return self._kth_smallest(partition_index + 1, high, k)

    def find_kth_largest(self, k):
        if k < 1 or k > len(self.data):
            raise ValueError("k is out of range")
        return self._kth_largest(0, len(self.data) - 1, len(self.data) - k)

    def _kth_largest(self, low, high, k):
        if low <= high:
            partition_index = self._partition(low, high, ascending=True)
            if partition_index == k:
                return self.data[partition_index]
            elif partition_index > k:
                return self._kth_largest(low, partition_index - 1, k)
            else:
                return self._kth_largest(partition_index + 1, high, k)

    def print_sorted_data(self):
        return self.data

    def reset_data(self, new_data):
        self.data = new_data

    def is_sorted(self, ascending=True):
        for i in range(len(self.data) - 1):
            if (self.data[i] > self.data[i + 1] and ascending) or (self.data[i] < self.data[i + 1] and not ascending):
                return False
        return True


if __name__ == "__main__":
    # Example data
    data = [10, 80, 30, 90, 40, 50, 70]

    # Initialize QuickSort
    quick_sort = QuickSort(data)

    # Sort the data in ascending order
    print("Original Data:", data)
    quick_sort.sort(ascending=True)
    print("Sorted Data (Ascending):", quick_sort.print_sorted_data())

    # Sort the data in descending order
    quick_sort.reset_data(data)
    quick_sort.sort(ascending=False)
    print("Sorted Data (Descending):", quick_sort.print_sorted_data())

    # Find the 3rd smallest element
    quick_sort.reset_data(data)
    kth_smallest = quick_sort.find_kth_smallest(3)
    print("3rd Smallest Element:", kth_smallest)

    # Find the 2nd largest element
    kth_largest = quick_sort.find_kth_largest(2)
    print("2nd Largest Element:", kth_largest)

    # Check if data is sorted
    is_sorted = quick_sort.is_sorted(ascending=False)
    print("Is data sorted (Descending)?", is_sorted)

    # Reset and sort again for validation
    quick_sort.reset_data(data)
    quick_sort.sort(ascending=True)
    print("Is data sorted (Ascending)?", quick_sort.is_sorted(ascending=True))

# first step


import uuid
from django.db import models
from django.contrib.auth.models import (
    AbstractBaseUser, AbstractUser, BaseUserManager
)


def user_profile_picture(instance, filename):
    image_extension = filename.split('.')[-1]
    image_name = f'user_profile_picture/{instance.user_uid}/{instance.user_uid}.{image_extension}'

    return image_name


class UserManager(BaseUserManager):
    def create_user(self, username, password=None, **extra_fields):
        if not username:
            raise ValueError('User must have a Username')

        user = self.model(username=username)
        user.set_password(password)
        user.save(using=self._db)

        return user

    def create_superuser(self, username, password, **extra_fields):
        user = self.create_user(
            username, password=password
        )

        user.is_admin = True
        user.save(using=self._db)

        return user


class User(AbstractBaseUser):
    user_uid = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False)
    username = models.CharField(max_length=25, unique=True)
    profile_picture = models.ImageField(
        upload_to=user_profile_picture, null=True, blank=True)
    email_id = models.EmailField(verbose_name='Email ID', unique=True)
    name = models.CharField(max_length=100)
    about_me = models.CharField(
        max_length=256, default="Hey, I'm using this app.")
    phone_number = models.CharField(max_length=20, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_verified = models.BooleanField(default=False)

    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_admin = models.BooleanField(default=False)

    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['email_id', 'name']

    def get_full_name(self):
        full_name = self.name
        return full_name

    def get_short_name(self):
        return self.name

    def __str__(self):
        return self.username

    def has_perm(self, perm):
        return True

    def has_module_perms(self, app_label):
        return True

    @property
    def is_staff(self):
        return self.is_admin

    objects = UserManager()

#  Serializer User 

import email
from rest_framework import serializers
from rest_framework import status
from django.contrib.auth import get_user_model
import re

User = get_user_model()


def password_validator(password):
    if re.findall('[a-z]', password):
        if re.findall('[A-Z]', password):
            if re.findall('[0-9]', password):
                if re.findall("[~\!@#\$%\^&\*\(\)_\+{}\":;'\[\]]", password):
                    if len(password) > 7:
                        return True
    return False

def check_username(username):
    instagram_username_pattern = bool(re.match(r'^([A-Za-z0-9_](?:(?:[A-Za-z0-9_]|(?:\.(?!\.))){0,28}(?:[A-Za-z0-9_]))?)$', username))
    return instagram_username_pattern


class RegisterUserSerializer(serializers.Serializer):
    username = serializers.CharField(error_messages={'message': 'Username Required.'})
    email_id = serializers.EmailField(error_messages={'message': 'Username Required.'})
    name = serializers.CharField(error_messages={'message': 'Username Required.'})
    password = serializers.CharField(error_messages={'message': 'Username Required.'})
    confirm_password = serializers.CharField(error_messages={'message': 'Username Required.'})

    def validate(self, data):
        username = data.get('username')
        email_id = data.get('email_id')
        password = data.get('password')
        confirm_password = data.get('confirm_password')

        try:
            user = User.objects.get(username=username)
        except:
            user = None

        try:
            user_email = User.objects.get(email_id=email_id)
        except:
            user_email = None

        if user:
            raise serializers.ValidationError('User with given username already exists.')

        if user_email:
            raise serializers.ValidationError('Email ID already belongs to an account.')
            
        if not check_username(username):
            raise serializers.ValidationError('Not a valid username.')

        if not password or not confirm_password:
            raise serializers.ValidationError("Please provide all details.")

        if not password_validator(password):
            raise serializers.ValidationError('Password must contain 1 number, 1 upper-case and lower-case letter and a special character.')

        if password == email_id:
            raise serializers.ValidationError('Password cannot be your Email ID.')

        if password != confirm_password:
            raise serializers.ValidationError('Password fields did not match.')

        return data

class LoginUserSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField()

    def validate(self, data):
        username = data.get('username')
        password = data.get('password')

        try:
            user = User.objects.get(username=username)
        except:
            user = None

        if not user:
            raise serializers.ValidationError({'message': 'Account with username does not exists.'}, code=status.HTTP_404_NOT_FOUND)

        if not user.check_password(password):
            raise serializers.ValidationError({'message': 'Password is incorrect.'}, code=status.HTTP_401_UNAUTHORIZED)
        
        # if not user.is_verified:
        #     raise serializers.ValidationError('User is not verified.')
        
        if not user.is_active:
            raise serializers.ValidationError('User is not active.')

        return data

#  views User 

from rest_framework import status
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from rest_framework.authtoken.models import Token
from rest_framework.permissions import AllowAny, IsAuthenticated
from .serializers import RegisterUserSerializer, LoginUserSerializer

User = get_user_model()


class RegisterUserView(APIView):
    permission_classes = (AllowAny,)
    serializer_class = RegisterUserSerializer

    def post(self, request):
        serializer = self.serializer_class(data=request.data)

        if serializer.is_valid():
            username = serializer.validated_data['username']
            email_id = serializer.validated_data['email_id']
            name = serializer.validated_data['name']

            password = request.data.get('password')

            user = User.objects.create(
                username=username,
                email_id=email_id,
                name=name,
            )

            user.set_password(password)
            user.save()

            token, created = Token.objects.get_or_create(user=user)

            content = {
                'token': token.key,
                'user_uid': user.user_uid,
                'username': username,
                'email_id': email_id,
                'name': name,
                'about_me': user.about_me,
                'profile_picture': user.profile_picture.url if user.profile_picture else None,
            }

            response_content = {
                'status': True,
                'message': 'User registered successfully.',
                'data': content
            }

            return Response(response_content, status=status.HTTP_201_CREATED)

        else:
            response_content = {
                'status': False,
                'message': serializer.errors,
            }

            print(response_content)
            return Response(response_content, status=status.HTTP_400_BAD_REQUEST)


class LoginUserView(APIView):
    permission_classes = (AllowAny,)
    serializer_class = LoginUserSerializer

    def post(self, request):
        serializer = self.serializer_class(data=request.data)

        try:
            if serializer.is_valid():
                username = serializer.validated_data['username']

                user = User.objects.get(username=username)

                token, created = Token.objects.get_or_create(user=user)

                content = {
                    'token': token.key,
                    'user_uid': user.user_uid,
                    'username': username,
                    'email_id': user.email_id,
                    'name': user.name,
                    'about_me': user.about_me,
                    'profile_picture': user.profile_picture.url if user.profile_picture != '' else None
                }

                response_content = {
                    'status': True,
                    'message': 'User logged in successfully.',
                    'data': content
                }

                return Response(response_content, status=status.HTTP_200_OK)

            else:
                response_content = {
                    'status': False,
                    'message': serializer.errors,
                }

                return Response(response_content, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            print(e)



class LogoutUserView(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        user = request.user

        token = Token.objects.get(user=user)

        token.delete()

        response_content = {
            'status': True,
            'message': 'User logged out successfully.'
        }

        return Response(response_content, status=status.HTTP_202_ACCEPTED)
    
# Bit Manipulation 
# Optimized technique for handling binary operations.

class BitManipulation:
    def __init__(self, number):
        self.number = number

    def set_bit(self, pos):
        """Sets the bit at a given position."""
        self.number |= (1 << pos)

    def clear_bit(self, pos):
        """Clears the bit at a given position."""
        self.number &= ~(1 << pos)

    def toggle_bit(self, pos):
        """Toggles the bit at a given position."""
        self.number ^= (1 << pos)

    def is_bit_set(self, pos):
        """Checks if a bit at a given position is set."""
        return (self.number & (1 << pos)) != 0

    def count_set_bits(self):
        """Recursively counts the number of set bits using bitwise AND."""
        return self._count_set_bits_recursive(self.number)

    def _count_set_bits_recursive(self, num):
        """Helper function for counting set bits recursively."""
        if num == 0:
            return 0
        return 1 + self._count_set_bits_recursive(num & (num - 1))

    def reverse_bits(self, bit_length=32):
        """Reverses the bits in the given number considering bit_length."""
        reversed_num = 0
        for i in range(bit_length):
            reversed_num |= ((self.number >> i) & 1) << (bit_length - 1 - i)
        return reversed_num

    def swap_numbers(self, num1, num2):
        """Swaps two numbers using XOR without temporary storage."""
        num1 ^= num2
        num2 ^= num1
        num1 ^= num2
        return num1, num2

    def is_power_of_two(self):
        """Checks if the number is a power of two."""
        return self.number > 0 and (self.number & (self.number - 1)) == 0

    def get_number(self):
        return self.number

    def reset_number(self, new_number):
        self.number = new_number


# Example Usage:
bit_manip = BitManipulation(29)  # 29 in binary: 11101
print("Bit 2 is set?", bit_manip.is_bit_set(2))
bit_manip.toggle_bit(2)
print("After toggling bit 2:", bit_manip.get_number())
print("Number of set bits:", bit_manip.count_set_bits())
print("Reversed Bits:", bin(bit_manip.reverse_bits()))
print("Is power of two?", bit_manip.is_power_of_two())
print("Swapped numbers:", bit_manip.swap_numbers(45, 78))

# Dynamic Programming 
# Solves problems by breaking them into overlapping subproblems.


class KnapsackDP:
    def __init__(self, weights, values, capacity):
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.n = len(weights)
        self.memo = [[-1] * (capacity + 1) for _ in range(self.n)]

    # Memoization (Top-Down Recursive)
    def knapsack_memo(self, index, remaining_capacity):
        if index < 0 or remaining_capacity == 0:
            return 0
        if self.memo[index][remaining_capacity] != -1:
            return self.memo[index][remaining_capacity]

        # Case 1: Skip current item
        exclude_item = self.knapsack_memo(index - 1, remaining_capacity)

        # Case 2: Include current item if it fits
        include_item = 0
        if self.weights[index] <= remaining_capacity:
            include_item = self.values[index] + self.knapsack_memo(index - 1, remaining_capacity - self.weights[index])

        # Store the result
        self.memo[index][remaining_capacity] = max(include_item, exclude_item)
        return self.memo[index][remaining_capacity]

    # Tabulation (Bottom-Up Approach)
    def knapsack_tabulation(self):
        dp = [[0] * (self.capacity + 1) for _ in range(self.n + 1)]

        for i in range(1, self.n + 1):
            for w in range(self.capacity + 1):
                if self.weights[i - 1] <= w:
                    dp[i][w] = max(dp[i - 1][w], self.values[i - 1] + dp[i - 1][w - self.weights[i - 1]])
                else:
                    dp[i][w] = dp[i - 1][w]

        return dp[self.n][self.capacity]

    # Space Optimized DP (Using a single array)
    def knapsack_space_optimized(self):
        dp = [0] * (self.capacity + 1)

        for i in range(self.n):
            for w in range(self.capacity, self.weights[i] - 1, -1):
                dp[w] = max(dp[w], self.values[i] + dp[w - self.weights[i]])

        return dp[self.capacity]

    def get_optimal_value(self):
        return {
            "Memoization Result": self.knapsack_memo(self.n - 1, self.capacity),
            "Tabulation Result": self.knapsack_tabulation(),
            "Space Optimized Result": self.knapsack_space_optimized()
        }


# Example Usage:
weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
capacity = 8

knapsack = KnapsackDP(weights, values, capacity)
results = knapsack.get_optimal_value()

print("Knapsack Results using DP:")
for method, value in results.items():
    print(f"{method}: {value}")


# Heap-based Algo 
# Used for priority queues and finding Kth largest/smallest elements.
""" 
Heap Construction (Min-Heap & Maz-Heap)
Heap Sort Algorithm
Priority Queue Implementation using Heaps
Extracting K smallest/Largest Elements using Heaps
"""

import heapq  # Python's built-in heap library

class HeapAlgorithms:
    def __init__(self, data):
        self.data = data
        self.min_heap = []
        self.max_heap = []

    # Build Min Heap
    def build_min_heap(self):
        """Creates a Min-Heap from data"""
        self.min_heap = self.data[:]
        heapq.heapify(self.min_heap)  # Converts list into a valid Min-Heap

    # Build Max Heap (Using Negative Values Trick)
    def build_max_heap(self):
        """Creates a Max-Heap from data (Using Negative Values)"""
        self.max_heap = [-num for num in self.data]
        heapq.heapify(self.max_heap)

    # Heap Sort (Using Min-Heap)
    def heap_sort_min(self):
        """Heap Sort using Min-Heap"""
        sorted_list = []
        temp_heap = self.min_heap[:]  # Copy heap to avoid modifying original
        while temp_heap:
            sorted_list.append(heapq.heappop(temp_heap))
        return sorted_list

    # Heap Sort (Using Max-Heap)
    def heap_sort_max(self):
        """Heap Sort using Max-Heap"""
        sorted_list = []
        temp_heap = self.max_heap[:]  # Copy heap to avoid modifying original
        while temp_heap:
            sorted_list.append(-heapq.heappop(temp_heap))  # Convert back to positive
        return sorted_list

    # Extract K Smallest Elements
    def extract_k_smallest(self, k):
        """Extract K smallest elements from heap"""
        return heapq.nsmallest(k, self.data)

    # Extract K Largest Elements
    def extract_k_largest(self, k):
        """Extract K largest elements from heap"""
        return heapq.nlargest(k, self.data)

    # Priority Queue Implementation (Using Min-Heap)
    def priority_queue(self):
        """Simulates a Priority Queue using Min-Heap"""
        pq = []
        for num in self.data:
            heapq.heappush(pq, num)  # Push into Min-Heap (Priority Queue)
        
        print("Priority Queue Output:")
        while pq:
            print(heapq.heappop(pq), end=" ")  # Pop elements in sorted order

    def display_results(self):
        """Runs all heap operations and displays results"""
        self.build_min_heap()
        self.build_max_heap()

        print("Original Data:", self.data)
        print("Min-Heap:", self.min_heap)
        print("Max-Heap:", [-x for x in self.max_heap])  # Convert back to positive
        print("Heap Sort (Min-Heap):", self.heap_sort_min())
        print("Heap Sort (Max-Heap):", self.heap_sort_max())
        print("3 Smallest Elements:", self.extract_k_smallest(3))
        print("3 Largest Elements:", self.extract_k_largest(3))
        self.priority_queue()


# Example Usage:
data = [10, 20, 15, 30, 40, 25]
heap_algo = HeapAlgorithms(data)
heap_algo.display_results()


# Topologial Sorting
# Used For scheduling tasks with dependencies
"""
Graph Class (Encapsulating graph operations)
Topological Sorting using Kahn's Algorithm (BFS)
Topological Sorting using DFS
Displaying the sorted order
"""

from collections import deque

class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adj_list = {i: [] for i in range(vertices)}  # Initialize adjacency list
        self.in_degree = {i: 0 for i in range(vertices)}  # Track in-degree for Kahn's algorithm

    def add_edge(self, u, v):
        """Adds a directed edge from u -> v"""
        self.adj_list[u].append(v)
        self.in_degree[v] += 1

    def topological_sort_bfs(self):
        """Performs Topological Sort using Kahn's Algorithm (BFS)"""
        queue = deque()
        sorted_order = []

        # Add nodes with zero in-degree to the queue
        for node in self.in_degree:
            if self.in_degree[node] == 0:
                queue.append(node)

        while queue:
            current = queue.popleft()
            sorted_order.append(current)

            for neighbor in self.adj_list[current]:
                self.in_degree[neighbor] -= 1
                if self.in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_order) == self.vertices:
            return sorted_order
        else:
            return "Cycle detected! Topological sorting not possible."

    def topological_sort_dfs_util(self, node, visited, stack):
        """DFS helper function for Topological Sorting"""
        visited[node] = True
        for neighbor in self.adj_list[node]:
            if not visited[neighbor]:
                self.topological_sort_dfs_util(neighbor, visited, stack)
        stack.append(node)

    def topological_sort_dfs(self):
        """Performs Topological Sort using DFS"""
        visited = {i: False for i in range(self.vertices)}
        stack = []

        for node in range(self.vertices):
            if not visited[node]:
                self.topological_sort_dfs_util(node, visited, stack)

        return stack[::-1]  # Reverse for correct ordering

    def display_graph(self):
        """Displays the adjacency list of the graph"""
        print("Graph adjacency list:")
        for node in self.adj_list:
            print(f"{node} -> {self.adj_list[node]}")


# Example usage:
graph = Graph(6)
graph.add_edge(5, 2)
graph.add_edge(5, 0)
graph.add_edge(4, 0)
graph.add_edge(4, 1)
graph.add_edge(2, 3)
graph.add_edge(3, 1)

graph.display_graph()
print("Topological Sort using BFS (Kahn’s Algorithm):", graph.topological_sort_bfs())
print("Topological Sort using DFS:", graph.topological_sort_dfs())


# Prim's Algorithm 
# Another approach for Minimum Spanning Tree
"""
Graph Representation and MST Computation in One Class
Efficient Priority-Based Edge Selection
Verbose Execution to show Step-by-Step Selection
Handling Edge Cases Like Unconnected Components
"""


import heapq

class PrimAlgorithm:
    """Represents a weighted undirected graph and computes the Minimum Spanning Tree (MST) using Prim's Algorithm."""
    
    def __init__(self, vertices):
        """Initialize graph representation with adjacency lists."""
        self.vertices = vertices
        self.adj_list = {i: [] for i in range(vertices)}
        self.visited = [False] * vertices
        self.mst_edges = []
    
    def add_edge(self, u, v, weight):
        """Add an undirected edge with given weight."""
        self.adj_list[u].append((v, weight))
        self.adj_list[v].append((u, weight))

    def compute_mst(self):
        """Execute Prim's Algorithm to compute the Minimum Spanning Tree."""
        self._initialize_priority_queue()
        
        while self.min_heap:
            weight, current, parent = heapq.heappop(self.min_heap)

            if self.visited[current]:
                continue
            
            self.visited[current] = True
            if parent != -1:
                self.mst_edges.append((parent, current, weight))

            self._process_edges(current)

    def _initialize_priority_queue(self):
        """Initialize the priority queue with the first node."""
        self.min_heap = [(0, 0, -1)]  # (Weight, Current Node, Parent Node)

    def _process_edges(self, current):
        """Push valid edges into the priority queue."""
        for neighbor, weight in self.adj_list[current]:
            if not self.visited[neighbor]:
                heapq.heappush(self.min_heap, (weight, neighbor, current))

    def display_graph(self):
        """Display the adjacency list representation of the graph."""
        print("\nGraph Representation:")
        for node in self.adj_list:
            print(f"{node}: {self.adj_list[node]}")

    def display_mst(self):
        """Displays the computed Minimum Spanning Tree."""
        self.compute_mst()
        print("\nMinimum Spanning Tree (Prim’s Algorithm):")
        for u, v, w in self.mst_edges:
            print(f"Edge {u} -- {v}  (Weight: {w})")

# Example Usage:
graph = PrimAlgorithm(6)
graph.add_edge(0, 1, 4)
graph.add_edge(0, 2, 2)
graph.add_edge(1, 2, 5)
graph.add_edge(1, 3, 10)
graph.add_edge(2, 3, 3)
graph.add_edge(3, 4, 7)
graph.add_edge(4, 5, 8)
graph.add_edge(3, 5, 6)

graph.display_graph()
graph.display_mst()


# Kruskal's Algorithm
# Finds the Minimum Spanning Tree using Union-Find


class KruskalMST:
    def __init__(self, vertices):
        self.V = vertices
        self.result = []
        self.edges = []
        self.parent = [i for i in range(self.V)]
        self.rank = [0] * self.V

    def add_edge(self, u, v, weight):
        self.edges.append((weight, u, v))

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, u_root, v_root):
        if self.rank[u_root] < self.rank[v_root]:
            self.parent[u_root] = v_root
        elif self.rank[u_root] > self.rank[v_root]:
            self.parent[v_root] = u_root
        else:
            self.parent[v_root] = u_root
            self.rank[u_root] += 1

    def compute_mst(self):
        self.edges.sort()
        num_edges_in_mst = 0
        i = 0
        while num_edges_in_mst < self.V - 1 and i < len(self.edges):
            weight, u, v = self.edges[i]
            i += 1
            u_root = self.find(u)
            v_root = self.find(v)
            if u_root != v_root:
                self.result.append((u, v, weight))
                self.union(u_root, v_root)
                num_edges_in_mst += 1

    def print_mst(self):
        print("Edges in the constructed Minimum Spanning Tree:")
        total_weight = 0
        for u, v, weight in self.result:
            print(f"{u} -- {v} == {weight}")
            total_weight += weight
        print(f"Total weight of MST: {total_weight}")

if __name__ == "__main__":
    g = KruskalMST(6)
    g.add_edge(0, 1, 4)
    g.add_edge(0, 2, 4)
    g.add_edge(1, 2, 2)
    g.add_edge(1, 0, 4)
    g.add_edge(2, 0, 4)
    g.add_edge(2, 1, 2)
    g.add_edge(2, 3, 3)
    g.add_edge(2, 5, 2)
    g.add_edge(2, 4, 4)
    g.add_edge(3, 2, 3)
    g.add_edge(3, 4, 3)
    g.add_edge(4, 2, 4)
    g.add_edge(4, 3, 3)
    g.add_edge(5, 2, 2)
    g.add_edge(5, 4, 3)
    g.compute_mst()
    g.print_mst()


# Floyd-Warshall Algorithm
# Computes shortest paths between all pairs of nodes

INF = float('inf')

def initialize_distance_matrix(vertices):
    matrix = []
    for i in range(vertices):
        row = []
        for j in range(vertices):
            if i == j:
                row.append(0)
            else:
                row.append(INF)
        matrix.append(row)
    return matrix

def apply_edges(matrix, edges):
    for edge in edges:
        u = edge[0]
        v = edge[1]
        weight = edge[2]
        matrix[u][v] = weight

def floyd_warshall_algorithm(matrix, vertices):
    for k in range(vertices):
        for i in range(vertices):
            for j in range(vertices):
                if matrix[i][k] != INF and matrix[k][j] != INF:
                    if matrix[i][j] > matrix[i][k] + matrix[k][j]:
                        matrix[i][j] = matrix[i][k] + matrix[k][j]

def print_distance_matrix(matrix, vertices):
    print("Shortest distance matrix:")
    for i in range(vertices):
        row_values = []
        for j in range(vertices):
            if matrix[i][j] == INF:
                row_values.append("INF")
            else:
                row_values.append(str(matrix[i][j]))
        print(" ".join(row_values))

def main():
    vertices = 4

    edges = [
        (0, 1, 5),
        (0, 3, 10),
        (1, 2, 3),
        (2, 3, 1)
    ]

    distance_matrix = initialize_distance_matrix(vertices)
    apply_edges(distance_matrix, edges)
    floyd_warshall_algorithm(distance_matrix, vertices)
    print_distance_matrix(distance_matrix, vertices)

if __name__ == "__main__":
    main()


# Bellman-Gord Algorithm 
#  Solves shortest path problems with negative weights.

INF = float('inf')

def initialize_distances(vertices, source):
    distances = []
    for i in range(vertices):
        if i == source:
            distances.append(0)
        else:
            distances.append(INF)
    return distances

def relax_edges(vertices, edges, distances):
    for _ in range(vertices - 1):
        for edge in edges:
            u = edge[0]
            v = edge[1]
            weight = edge[2]
            if distances[u] != INF and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight

def detect_negative_cycle(vertices, edges, distances):
    for edge in edges:
        u = edge[0]
        v = edge[1]
        weight = edge[2]
        if distances[u] != INF and distances[u] + weight < distances[v]:
            return True
    return False

def print_distances(distances, source):
    print(f"Shortest distances from source vertex {source}:")
    for i in range(len(distances)):
        if distances[i] == INF:
            print(f"Vertex {i}: INF")
        else:
            print(f"Vertex {i}: {distances[i]}")

def bellman_ford(vertices, edges, source):
    distances = initialize_distances(vertices, source)
    relax_edges(vertices, edges, distances)
    has_negative_cycle = detect_negative_cycle(vertices, edges, distances)
    if has_negative_cycle:
        print("Graph contains a negative weight cycle.")
    else:
        print_distances(distances, source)

def main():
    vertices = 5
    edges = [
        (0, 1, -1),
        (0, 2, 4),
        (1, 2, 3),
        (1, 3, 2),
        (1, 4, 2),
        (3, 2, 5),
        (3, 1, 1),
        (4, 3, -3)
    ]
    source = 0
    bellman_ford(vertices, edges, source)

if __name__ == "__main__":
    main()

# Dijkstra's Algorithm 
# Finds teh shortest path in weighted graphs

import heapq

class Dijkstra:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = {i: [] for i in range(vertices)}
        self.distances = [float('inf')] * vertices

    def add_edge(self, u, v, weight):
        self.graph[u].append((v, weight))

    def compute(self, source):
        self.distances = [float('inf')] * self.V
        self.distances[source] = 0

        priority_queue = []
        heapq.heappush(priority_queue, (0, source))

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            if current_distance > self.distances[current_vertex]:
                continue

            for neighbor, weight in self.graph[current_vertex]:
                distance = current_distance + weight
                if distance < self.distances[neighbor]:
                    self.distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

    def get_distances(self):
        return [None if d == float('inf') else d for d in self.distances]

if __name__ == "__main__":
    g = Dijkstra(5)
    g.add_edge(0, 1, 10)
    g.add_edge(0, 4, 5)
    g.add_edge(1, 2, 1)
    g.add_edge(1, 4, 2)
    g.add_edge(2, 3, 4)
    g.add_edge(3, 2, 6)
    g.add_edge(4, 1, 3)
    g.add_edge(4, 2, 9)
    g.add_edge(4, 3, 2)

    g.compute(0)
    print("Shortest distances from node 0:")
    print(g.get_distances())


# Manacher's Algorithm 
# Finds the longest palindromic substring in linear time

class Manacher:
    def __init__(self, s):
        self.original = s
        self.processed = self._preprocess(s)
        self.p = [0] * len(self.processed)
        self.center = 0
        self.right = 0

    def _preprocess(self, s):
        return '^#' + '#'.join(s) + '#$'

    def compute(self):
        for i in range(1, len(self.processed) - 1):
            mirror = 2 * self.center - i

            if i < self.right:
                self.p[i] = min(self.right - i, self.p[mirror])

            while self.processed[i + self.p[i] + 1] == self.processed[i - self.p[i] - 1]:
                self.p[i] += 1

            if i + self.p[i] > self.right:
                self.center = i
                self.right = i + self.p[i]

    def longest_palindromic_substring(self):
        self.compute()
        max_len = max(self.p)
        center_index = self.p.index(max_len)
        start = (center_index - max_len) // 2
        return self.original[start:start + max_len]

    def get_all_palindromic_lengths(self):
        return self.p[2:-2:2]  # skip special characters and reduce to original indices


if __name__ == "__main__":
    s = "babad"
    manacher = Manacher(s)
    print("Longest palindromic substring:", manacher.longest_palindromic_substring())
    print("Palindromic lengths at original character positions:", manacher.get_all_palindromic_lengths())


# Kadane's Algorithm 
# Used for finding the maximum sun subarray efficiently


class Kadane:
    def __init__(self, array):
        self.array = array

    def max_subarray_sum(self):
        max_sum = current_sum = self.array[0]
        for num in self.array[1:]:
            current_sum = max(num, current_sum + num)
            max_sum = max(max_sum, current_sum)
        return max_sum

    def max_subarray_with_indices(self):
        max_sum = current_sum = self.array[0]
        start = end = temp_start = 0
        for i in range(1, len(self.array)):
            if self.array[i] > current_sum + self.array[i]:
                current_sum = self.array[i]
                temp_start = i
            else:
                current_sum += self.array[i]

            if current_sum > max_sum:
                max_sum = current_sum
                start = temp_start
                end = i
        return max_sum, start, end, self.array[start:end+1]

if __name__ == "__main__":
    arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    kadane = Kadane(arr)

    max_sum = kadane.max_subarray_sum()
    print("Maximum subarray sum:", max_sum)

    max_sum, start, end, subarray = kadane.max_subarray_with_indices()
    print(f"Max subarray sum: {max_sum}, from index {start} to {end}")
    print("Subarray:", subarray)


# Monotonic Stack
# Helps in solving problems related to next greater/smaller elements

class MonotonicStack:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)

    def _init_result(self):
        return [-1] * self.n, [-1] * self.n

    def next_greater_elements(self):
        stack = []
        values, indices = self._init_result()

        for i in range(self.n - 1, -1, -1):
            current = self.nums[i]

            while stack and self.nums[stack[-1]] <= current:
                stack.pop()

            if stack:
                indices[i] = stack[-1]
                values[i] = self.nums[stack[-1]]

            stack.append(i)

        return values, indices

    def previous_greater_elements(self):
        stack = []
        values, indices = self._init_result()

        for i in range(self.n):
            current = self.nums[i]

            while stack and self.nums[stack[-1]] <= current:
                stack.pop()

            if stack:
                indices[i] = stack[-1]
                values[i] = self.nums[stack[-1]]

            stack.append(i)

        return values, indices

    def next_lesser_elements(self):
        stack = []
        values, indices = self._init_result()

        for i in range(self.n - 1, -1, -1):
            current = self.nums[i]

            while stack and self.nums[stack[-1]] >= current:
                stack.pop()

            if stack:
                indices[i] = stack[-1]
                values[i] = self.nums[stack[-1]]

            stack.append(i)

        return values, indices

    def previous_lesser_elements(self):
        stack = []
        values, indices = self._init_result()

        for i in range(self.n):
            current = self.nums[i]

            while stack and self.nums[stack[-1]] >= current:
                stack.pop()

            if stack:
                indices[i] = stack[-1]
                values[i] = self.nums[stack[-1]]

            stack.append(i)

        return values, indices

    def debug_print_all(self):
        print("Original array:", self.nums)
        n_g_vals, n_g_idx = self.next_greater_elements()
        print("Next Greater Values:  ", n_g_vals)
        print("Next Greater Indices: ", n_g_idx)

        p_g_vals, p_g_idx = self.previous_greater_elements()
        print("Previous Greater Values:  ", p_g_vals)
        print("Previous Greater Indices: ", p_g_idx)

        n_l_vals, n_l_idx = self.next_lesser_elements()
        print("Next Lesser Values:  ", n_l_vals)
        print("Next Lesser Indices: ", n_l_idx)

        p_l_vals, p_l_idx = self.previous_lesser_elements()
        print("Previous Lesser Values:  ", p_l_vals)
        print("Previous Lesser Indices: ", p_l_idx)

if __name__ == "__main__":
    nums = [2, 1, 2, 4, 3]
    ms = MonotonicStack(nums)
    ms.debug_print_all()

# Fenwick Tree (Binary Indexed Tree)
# Optimized for cumulative frequency queries.

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

    def update(self, index, delta):
        index += 1  # Convert to 1-based
        while index <= self.n:
            self.tree[index] += delta
            index += index & -index  # Move to parent

    def query(self, index):
        index += 1  # Convert to 1-based
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & -index  # Move to child
        return result

    def range_query(self, left, right):
        return self.query(right) - self.query(left - 1)

    def build_from_list(self, data):
        for i, val in enumerate(data):
            self.update(i, val)

    def get_tree(self):
        return self.tree[1:]  # return internal structure (1-based)

if __name__ == "__main__":
    data = [3, 2, -1, 6, 5, 4, -3, 3, 7, 2, 3]
    ft = FenwickTree(len(data))
    ft.build_from_list(data)

    print("Fenwick Tree internal structure:", ft.get_tree())

    print("Prefix sum [0..5]:", ft.query(5))         # sum of data[0..5]
    print("Range sum [3..8]:", ft.range_query(3, 8))  # sum of data[3..8]

    print("Updating index 4 (+2)...")
    ft.update(4, 2)
    print("New prefix sum [0..5]:", ft.query(5))

# Segment Tree
# Used for range queries and updates in arrays.

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.data = data[:]
        self.tree = [0] * (4 * self.n)  # Enough size for segment tree
        self._build(0, 0, self.n - 1)

    def _build(self, node, l, r):
        if l == r:
            self.tree[node] = self.data[l]
        else:
            mid = (l + r) // 2
            self._build(2 * node + 1, l, mid)
            self._build(2 * node + 2, mid + 1, r)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def _query(self, node, l, r, ql, qr):
        if ql > r or qr < l:
            return 0  # No overlap
        if ql <= l and r <= qr:
            return self.tree[node]  # Total overlap
        mid = (l + r) // 2
        left_sum = self._query(2 * node + 1, l, mid, ql, qr)
        right_sum = self._query(2 * node + 2, mid + 1, r, ql, qr)
        return left_sum + right_sum  # Partial overlap

    def query(self, left, right):
        if left < 0 or right >= self.n or left > right:
            raise IndexError("Invalid query range")
        return self._query(0, 0, self.n - 1, left, right)

    def _update(self, node, l, r, index, value):
        if l == r:
            self.tree[node] = value
            self.data[index] = value
        else:
            mid = (l + r) // 2
            if index <= mid:
                self._update(2 * node + 1, l, mid, index, value)
            else:
                self._update(2 * node + 2, mid + 1, r, index, value)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def update(self, index, value):
        if index < 0 or index >= self.n:
            raise IndexError("Index out of bounds")
        self._update(0, 0, self.n - 1, index, value)

    def get_tree(self):
        return self.tree[:]

if __name__ == "__main__":
    arr = [1, 3, 5, 7, 9, 11]
    st = SegmentTree(arr)

    print("Segment Tree structure:", st.get_tree())
    print("Sum of values from index 1 to 3:", st.query(1, 3))  # 3 + 5 + 7 = 15

    print("Updating index 1 to value 10...")
    st.update(1, 10)

    print("Sum of values from index 1 to 3 after update:", st.query(1, 3))  # 10 + 5 + 7 = 22


# Trie (Prefix Tree)
# Efficient For searching words and autocomplete features.

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_end_of_word = True

    def search(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        return current.is_end_of_word

    def starts_with(self, prefix):
        current = self.root
        for char in prefix:
            if char not in current.children:
                return False
            current = current.children[char]
        return True

    def delete(self, word):
        def _delete(node, word, depth):
            if node is None:
                return False

            if depth == len(word):
                if not node.is_end_of_word:
                    return False  # word doesn't exist
                node.is_end_of_word = False
                return len(node.children) == 0

            char = word[depth]
            if char in node.children:
                should_delete_child = _delete(node.children[char], word, depth + 1)

                if should_delete_child:
                    del node.children[char]
                    return not node.is_end_of_word and len(node.children) == 0

            return False

        _delete(self.root, word, 0)

    def collect_all_words(self):
        def _collect(node, path, words):
            if node.is_end_of_word:
                words.append(''.join(path))
            for char, child in node.children.items():
                path.append(char)
                _collect(child, path, words)
                path.pop()

        words = []
        _collect(self.root, [], words)
        return words


if __name__ == "__main__":
    trie = Trie()
    trie.insert("apple")
    trie.insert("app")
    trie.insert("apt")
    trie.insert("bat")
    trie.insert("batch")

    print("Search 'app':", trie.search("app"))           # True
    print("Search 'apple':", trie.search("apple"))       # True
    print("Search 'ap':", trie.search("ap"))             # False
    print("Starts with 'ap':", trie.starts_with("ap"))   # True
    print("Starts with 'ba':", trie.starts_with("ba"))   # True
    print("Starts with 'cat':", trie.starts_with("cat")) # False

    print("All words in trie:", trie.collect_all_words())  # ['apple', 'app', 'apt', 'bat', 'batch']

    trie.delete("apple")
    print("After deleting 'apple':", trie.collect_all_words())  # ['app', 'apt', 'bat', 'batch']
    print("Search 'apple':", trie.search("apple"))  # False


# Union-Find (Disjoint Set)
# Helps in detecting cycles and solving connectivity problems.


class UnionFind:
    def __init__(self, size):
        self.parent = [i for i in range(size)]
        self.rank = [0] * size  # Used for union by rank

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already in the same set

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_all_groups(self):
        from collections import defaultdict
        groups = defaultdict(list)
        for node in range(len(self.parent)):
            root = self.find(node)
            groups[root].append(node)
        return list(groups.values())


if __name__ == "__main__":
    uf = UnionFind(10)

    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(3, 4)
    uf.union(5, 6)
    uf.union(6, 7)
    uf.union(7, 8)

    print("Are 0 and 2 connected?", uf.connected(0, 2))  # True
    print("Are 0 and 3 connected?", uf.connected(0, 3))  # False
    print("All groups after unions:", uf.get_all_groups())

    uf.union(2, 3)  # Now connects 0-1-2 with 3-4

    print("Are 0 and 4 connected after union(2, 3)?", uf.connected(0, 4))  # True
    print("All groups after more unions:", uf.get_all_groups())


# Graph Traversal (BFS & DFS) 
# Used for exploring graphs, shortest paths, and connectivity

from collections import deque, defaultdict

class Graph:
    def __init__(self, directed=False):
        self.adj = defaultdict(list)
        self.directed = directed

    def add_edge(self, u, v):
        self.adj[u].append(v)
        if not self.directed:
            self.adj[v].append(u)

    def bfs(self, start):
        visited = set()
        queue = deque([start])
        traversal = []

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                traversal.append(node)
                for neighbor in self.adj[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        return traversal

    def dfs(self, start):
        visited = set()
        traversal = []

        def _dfs(node):
            visited.add(node)
            traversal.append(node)
            for neighbor in self.adj[node]:
                if neighbor not in visited:
                    _dfs(neighbor)

        _dfs(start)
        return traversal

    def bfs_all(self):
        visited = set()
        traversal = []

        for node in self.adj:
            if node not in visited:
                queue = deque([node])
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        traversal.append(current)
                        for neighbor in self.adj[current]:
                            if neighbor not in visited:
                                queue.append(neighbor)
        return traversal

    def dfs_all(self):
        visited = set()
        traversal = []

        def _dfs(node):
            visited.add(node)
            traversal.append(node)
            for neighbor in self.adj[node]:
                if neighbor not in visited:
                    _dfs(neighbor)

        for node in self.adj:
            if node not in visited:
                _dfs(node)

        return traversal

    def show(self):
        return dict(self.adj)

if __name__ == "__main__":
    g = Graph(directed=False)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 3)
    g.add_edge(1, 4)
    g.add_edge(2, 5)
    g.add_edge(3, 6)

    print("Graph structure:", g.show())
    print("BFS from node 0:", g.bfs(0))
    print("DFS from node 0:", g.dfs(0))
    print("BFS for disconnected graph:", g.bfs_all())
    print("DFS for disconnected graph:", g.dfs_all())


# Divide and Conquer 
# Breaks problems into smaller subproblems (e.g., Merg Sort, Quick Sort)

class DivideAndConquer:
    def __init__(self, array):
        self.array = array

    def max_crossing_sum(self, left, mid, right):
        left_sum = float('-inf')
        current_sum = 0
        max_left = mid

        for i in range(mid, left - 1, -1):
            current_sum += self.array[i]
            if current_sum > left_sum:
                left_sum = current_sum
                max_left = i

        right_sum = float('-inf')
        current_sum = 0
        max_right = mid + 1

        for i in range(mid + 1, right + 1):
            current_sum += self.array[i]
            if current_sum > right_sum:
                right_sum = current_sum
                max_right = i

        total_sum = left_sum + right_sum
        return total_sum, max_left, max_right

    def max_subarray_sum(self, left, right):
        if left == right:
            return self.array[left], left, right

        mid = (left + right) // 2

        left_sum, left_start, left_end = self.max_subarray_sum(left, mid)
        right_sum, right_start, right_end = self.max_subarray_sum(mid + 1, right)
        cross_sum, cross_start, cross_end = self.max_crossing_sum(left, mid, right)

        if left_sum >= right_sum and left_sum >= cross_sum:
            return left_sum, left_start, left_end
        elif right_sum >= left_sum and right_sum >= cross_sum:
            return right_sum, right_start, right_end
        else:
            return cross_sum, cross_start, cross_end

    def solve(self):
        if not self.array:
            return 0, -1, -1
        return self.max_subarray_sum(0, len(self.array) - 1)

if __name__ == "__main__":
    array = [13, -3, -25, 20, -3, -16, -23, 18,
             20, -7, 12, -5, -22, 15, -4, 7]
    
    dac = DivideAndConquer(array)
    max_sum, start, end = dac.solve()
    print("Maximum Subarray Sum:", max_sum)
    print(f"Subarray indices: [{start}:{end}]")
    print("Subarray:", array[start:end+1])


# Backtracking 
# Used for problems involving permutations, combinations, and constraint satisfaction.

class NQueensBacktracking:
    def __init__(self, n):
        self.n = n
        self.board = [['.' for _ in range(n)] for _ in range(n)]
        self.solutions = []

    def is_safe(self, row, col):
        # Check column
        for r in range(row):
            if self.board[r][col] == 'Q':
                return False

        # Check upper-left diagonal
        r, c = row - 1, col - 1
        while r >= 0 and c >= 0:
            if self.board[r][c] == 'Q':
                return False
            r -= 1
            c -= 1

        # Check upper-right diagonal
        r, c = row - 1, col + 1
        while r >= 0 and c < self.n:
            if self.board[r][c] == 'Q':
                return False
            r -= 1
            c += 1

        return True

    def solve_row(self, row=0):
        if row == self.n:
            # Found a solution, convert board to list of strings and save
            solution = [''.join(r) for r in self.board]
            self.solutions.append(solution)
            return

        for col in range(self.n):
            if self.is_safe(row, col):
                self.board[row][col] = 'Q'
                self.solve_row(row + 1)
                self.board[row][col] = '.'  # Backtrack

    def solve(self):
        self.solve_row()
        return self.solutions

if __name__ == "__main__":
    n = 4
    solver = NQueensBacktracking(n)
    solutions = solver.solve()
    print(f"Total solutions for {n}-Queens:", len(solutions))
    for idx, sol in enumerate(solutions, 1):
        print(f"Solution {idx}:")
        for row in sol:
            print(row)
        print()


# Gabow's Algorithm


class GabowSCC:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adj_list = {i: [] for i in range(vertices)}
        self.index_counter = 0
        self.stack = []
        self.component_stack = []
        self.index = [-1] * vertices
        self.component_map = [-1] * vertices
        self.sccs = []
        self.execution_steps = []

    def add_edge(self, u, v):
        self.adj_list[u].append(v)
        self.execution_steps.append((u, v, "Added"))

    def compute_sccs(self):
        for v in range(self.vertices):
            if self.index[v] == -1:
                self._dfs(v)

    def _dfs(self, v):
        self.index[v] = self.index_counter
        self.index_counter += 1
        self.stack.append(v)
        self.component_stack.append(v)

        for neighbor in self.adj_list[v]:
            if self.index[neighbor] == -1:
                self._dfs(neighbor)
            elif self.component_map[neighbor] == -1:
                while self.index[neighbor] < self.index[self.component_stack[-1]]:
                    self.component_stack.pop()

        if v == self.component_stack[-1]:
            self.component_stack.pop()
            scc = []
            while True:
                w = self.stack.pop()
                self.component_map[w] = len(self.sccs)
                scc.append(w)
                if w == v:
                    break
            self.sccs.append(scc)

    def get_sccs(self):
        scc_structure = {}
        component_sizes = {}
        
        for component_id in range(len(self.sccs)):
            component_nodes = []
            for node, comp in enumerate(self.component_map):
                if comp == component_id:
                    component_nodes.append(node)
            
            scc_structure[component_id] = component_nodes
            component_sizes[component_id] = len(component_nodes)
        
        return {
            "Total_SCCs": len(self.sccs),
            "SCC_Details": scc_structure,
            "Component_Sizes": component_sizes
        }

    def get_graph(self):
        structured_graph = {"Graph_Nodes": {}}
        edge_count = 0

        for node in self.adj_list:
            edges = []
            for neighbor in self.adj_list[node]:
                edges.append(neighbor)
                edge_count += 1
            
            structured_graph["Graph_Nodes"][node] = {
                "Outgoing_Connections": edges,
                "Total_Outgoing": len(edges)
            }
        
        structured_graph["Total_Edges"] = edge_count
        structured_graph["Total_Vertices"] = self.vertices
        
        return structured_graph

    def get_execution_steps(self):
        execution_flow = {"Steps": [], "Operation_Count": {}}
        operation_tracking = {"Edge_Additions": 0, "DFS_Calls": 0, "SCC_Creations": 0}

        for step_id, step in enumerate(self.execution_steps):
            if step[2] == "Added":
                operation_tracking["Edge_Additions"] += 1
                execution_flow["Steps"].append({"Step": step_id, "Action": "Edge Addition", "Nodes": (step[0], step[1])})

        for node in range(self.vertices):
            if self.index[node] != -1:
                operation_tracking["DFS_Calls"] += 1
                execution_flow["Steps"].append({"Step": len(execution_flow["Steps"]), "Action": "DFS Call", "Node": node})

        execution_flow["Operation_Count"] = operation_tracking
        return execution_flow


graph = GabowSCC(5)
graph.add_edge(0, 1)
graph.add_edge(1, 2)
graph.add_edge(2, 0)
graph.add_edge(2, 3)
graph.add_edge(3, 4)

graph.compute_sccs()

scc_result = graph.get_sccs()
graph_data = graph.get_graph()
execution_log = graph.get_execution_steps()


# Decomposition (HLD) Heavy-Light Decomposition


class HeavyLightDecomposition:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adj_list = {i: [] for i in range(vertices)}
        self.parent = [-1] * vertices
        self.depth = [0] * vertices
        self.subtree_size = [0] * vertices
        self.chain_head = [-1] * vertices
        self.chain_index = [0] * vertices
        self.position = [-1] * vertices
        self.current_pos = 0
        self.segment_tree = [0] * (4 * vertices)
        self.original_values = [0] * vertices

    def add_edge(self, u, v):
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)

    def compute_subtree_sizes(self, node, parent):
        self.subtree_size[node] = 1
        self.parent[node] = parent
        for neighbor in self.adj_list[node]:
            if neighbor == parent:
                continue
            self.depth[neighbor] = self.depth[node] + 1
            self.compute_subtree_sizes(neighbor, node)
            self.subtree_size[node] += self.subtree_size[neighbor]

    def decompose(self, node, chain_head):
        self.chain_head[node] = chain_head
        self.position[node] = self.current_pos
        self.current_pos += 1
        heavy_child = -1
        max_size = -1

        for neighbor in self.adj_list[node]:
            if neighbor == self.parent[node]:
                continue
            if self.subtree_size[neighbor] > max_size:
                max_size = self.subtree_size[neighbor]
                heavy_child = neighbor

        if heavy_child != -1:
            self.decompose(heavy_child, chain_head)

        for neighbor in self.adj_list[node]:
            if neighbor == self.parent[node] or neighbor == heavy_child:
                continue
            self.decompose(neighbor, neighbor)

    def build_segment_tree(self, start, end, index):
        if start == end:
            self.segment_tree[index] = self.original_values[self.position[start]]
            return
        mid = (start + end) // 2
        self.build_segment_tree(start, mid, 2 * index + 1)
        self.build_segment_tree(mid + 1, end, 2 * index + 2)
        self.segment_tree[index] = self.segment_tree[2 * index + 1] + self.segment_tree[2 * index + 2]

    def update_segment_tree(self, start, end, index, pos, value):
        if start == end:
            self.segment_tree[index] = value
            return
        mid = (start + end) // 2
        if pos <= mid:
            self.update_segment_tree(start, mid, 2 * index + 1, pos, value)
        else:
            self.update_segment_tree(mid + 1, end, 2 * index + 2, pos, value)
        self.segment_tree[index] = self.segment_tree[2 * index + 1] + self.segment_tree[2 * index + 2]

    def query_segment_tree(self, start, end, index, left, right):
        if left > end or right < start:
            return 0
        if left <= start and right >= end:
            return self.segment_tree[index]
        mid = (start + end) // 2
        left_query = self.query_segment_tree(start, mid, 2 * index + 1, left, right)
        right_query = self.query_segment_tree(mid + 1, end, 2 * index + 2, left, right)
        return left_query + right_query

    def path_query(self, u, v):
        result = 0
        while self.chain_head[u] != self.chain_head[v]:
            if self.depth[self.chain_head[u]] < self.depth[self.chain_head[v]]:
                u, v = v, u
            start = self.position[self.chain_head[u]]
            end = self.position[u]
            result += self.query_segment_tree(0, self.vertices - 1, 0, start, end)
            u = self.parent[self.chain_head[u]]
        start = min(self.position[u], self.position[v])
        end = max(self.position[u], self.position[v])
        result += self.query_segment_tree(0, self.vertices - 1, 0, start, end)
        return result

    def get_hierarchy(self):
        hierarchy = {}
        for node in range(self.vertices):
            hierarchy[node] = {
                "Parent": self.parent[node],
                "Depth": self.depth[node],
                "Subtree_Size": self.subtree_size[node],
                "Chain_Head": self.chain_head[node],
                "Position": self.position[node]
            }
        return hierarchy

    def get_segment_tree(self):
        return {"Segment_Tree_Data": self.segment_tree, "Total_Nodes": len(self.segment_tree)}

    def get_execution_log(self):
        log_details = {"Edges_Processed": [], "Tree_Building": {}, "Decomposition_Steps": {}}
        for node in range(self.vertices):
            log_details["Edges_Processed"].append(self.adj_list[node])
            log_details["Tree_Building"][node] = {
                "Parent": self.parent[node],
                "Depth": self.depth[node],
                "Chain_Head": self.chain_head[node]
            }
            log_details["Decomposition_Steps"][node] = {
                "Heavy_Child": max(self.adj_list[node], key=lambda x: self.subtree_size[x], default=-1),
                "Processed_Position": self.position[node]
            }
        return log_details


# Example Usage:
hld = HeavyLightDecomposition(7)
hld.add_edge(0, 1)
hld.add_edge(1, 2)
hld.add_edge(1, 3)
hld.add_edge(3, 4)
hld.add_edge(3, 5)
hld.add_edge(5, 6)

hld.compute_subtree_sizes(0, -1)
hld.decompose(0, 0)

hierarchy_result = hld.get_hierarchy()
segment_tree_data = hld.get_segment_tree()
execution_steps = hld.get_execution_log()


# Wavelet-Tree

class WaveletTree:
    def __init__(self, data, low=None, high=None):
        self.data = data
        self.low = min(data) if low is None else low
        self.high = max(data) if high is None else high
        self.left = None
        self.right = None
        self.bitmap = []

        if self.low == self.high or not data:
            return
        
        mid = (self.low + self.high) // 2
        left_partition = []
        right_partition = []
        
        for val in data:
            if val <= mid:
                self.bitmap.append(0)
                left_partition.append(val)
            else:
                self.bitmap.append(1)
                right_partition.append(val)
        
        if left_partition:
            self.left = WaveletTree(left_partition, self.low, mid)
        if right_partition:
            self.right = WaveletTree(right_partition, mid + 1, self.high)

    def rank(self, value, index):
        if self.low == self.high:
            return index

        mid = (self.low + self.high) // 2
        rank_count = sum(1 for i in range(index) if self.bitmap[i] == (value > mid))

        if value <= mid:
            return self.left.rank(value, rank_count) if self.left else 0
        else:
            return self.right.rank(value, rank_count) if self.right else 0

    def select(self, value, rank):
        if self.low == self.high:
            return rank
        
        mid = (self.low + self.high) // 2
        matching_positions = [i for i in range(len(self.bitmap)) if self.bitmap[i] == (value > mid)]
        
        if rank >= len(matching_positions):
            return -1  

        selected_index = matching_positions[rank]

        if value <= mid:
            return self.left.select(value, rank) if self.left else -1
        else:
            return self.right.select(value, rank) if self.right else -1

    def access(self, index):
        if self.low == self.high:
            return self.low

        mid = (self.low + self.high) // 2
        bit_value = self.bitmap[index]
        rank_index = sum(1 for i in range(index) if self.bitmap[i] == bit_value)

        return self.left.access(rank_index) if bit_value == 0 and self.left else self.right.access(rank_index)


# Example Usage:
data = [3, 5, 2, 8, 5, 7, 6, 1, 4]
wavelet_tree = WaveletTree(data)

rank_5 = wavelet_tree.rank(5, 6)
select_5 = wavelet_tree.select(5, 2)
access_4 = wavelet_tree.access(4)

structured_tree = {
    "Rank of 5 in first 6 elements": rank_5,
    "Position of 2nd occurrence of 5": select_5,
    "Element at index 4": access_4
}


# Range Minimum Query (RMQ)

class RangeMinimumQuery:
    def __init__(self, arr):
        """Initialize RMQ with segment tree"""
        self.n = len(arr)
        self.arr = arr
        self.segment_tree = [float("inf")] * (4 * self.n)
        self.lazy_tree = [0] * (4 * self.n)
        self._build_segment_tree(0, self.n - 1, 0)

    def _build_segment_tree(self, start, end, index):
        """Builds the segment tree recursively"""
        if start == end:
            self.segment_tree[index] = self.arr[start]
            return
        mid = (start + end) // 2
        self._build_segment_tree(start, mid, 2 * index + 1)
        self._build_segment_tree(mid + 1, end, 2 * index + 2)
        self.segment_tree[index] = min(self.segment_tree[2 * index + 1], self.segment_tree[2 * index + 2])

    def _range_query(self, start, end, index, left, right):
        """Queries the minimum value in a given range"""
        if left > end or right < start:
            return float("inf")
        if left <= start and right >= end:
            return self.segment_tree[index]
        mid = (start + end) // 2
        left_query = self._range_query(start, mid, 2 * index + 1, left, right)
        right_query = self._range_query(mid + 1, end, 2 * index + 2, left, right)
        return min(left_query, right_query)

    def _update_segment_tree(self, start, end, index, pos, value):
        """Updates the segment tree when an element is modified"""
        if start == end:
            self.segment_tree[index] = value
            return
        mid = (start + end) // 2
        if pos <= mid:
            self._update_segment_tree(start, mid, 2 * index + 1, pos, value)
        else:
            self._update_segment_tree(mid + 1, end, 2 * index + 2, pos, value)
        self.segment_tree[index] = min(self.segment_tree[2 * index + 1], self.segment_tree[2 * index + 2])

    def _lazy_propagation(self, start, end, index):
        """Handles lazy propagation updates"""
        if self.lazy_tree[index] != 0:
            self.segment_tree[index] += self.lazy_tree[index]
            if start != end:
                self.lazy_tree[2 * index + 1] += self.lazy_tree[index]
                self.lazy_tree[2 * index + 2] += self.lazy_tree[index]
            self.lazy_tree[index] = 0

    def range_update(self, start, end, index, left, right, value):
        """Applies range-based updates lazily"""
        self._lazy_propagation(start, end, index)
        if left > end or right < start:
            return
        if left <= start and right >= end:
            self.segment_tree[index] += value
            if start != end:
                self.lazy_tree[2 * index + 1] += value
                self.lazy_tree[2 * index + 2] += value
            return
        mid = (start + end) // 2
        self.range_update(start, mid, 2 * index + 1, left, right, value)
        self.range_update(mid + 1, end, 2 * index + 2, left, right, value)
        self.segment_tree[index] = min(self.segment_tree[2 * index + 1], self.segment_tree[2 * index + 2])

    def get_minimum(self, left, right):
        """Public method for querying RMQ"""
        return self._range_query(0, self.n - 1, 0, left, right)

    def update(self, pos, value):
        """Public method for updating segment tree"""
        self._update_segment_tree(0, self.n - 1, 0, pos, value)

    def get_segment_tree(self):
        """Structured retrieval of segment tree data"""
        return {"Segment_Tree": self.segment_tree, "Lazy_Propagation": self.lazy_tree, "Total_Nodes": len(self.segment_tree)}

    def get_execution_log(self):
        """Returns detailed execution steps tracking"""
        log_details = {"Tree_Building": {}, "Lazy_Operations": {}, "Updates": {}}
        for node in range(self.n):
            log_details["Tree_Building"][node] = {"Original_Value": self.arr[node], "Segment_Tree_Position": node}
            log_details["Lazy_Operations"][node] = {"Lazy_Value": self.lazy_tree[node]}
        return log_details


# Example Usage:
arr = [1, 3, 2, 7, 9, 11, 4, 8]
rmq = RangeMinimumQuery(arr)

query_result = rmq.get_minimum(2, 6)
rmq.update(3, 5)
updated_tree = rmq.get_segment_tree()
execution_steps = rmq.get_execution_log()


# Bentley-Ottmann Algorithm


import heapq

class BentleyOttmann:
    def __init__(self, segments):
        self.segments = segments
        self.event_queue = []
        self.active_segments = {}
        self.intersections = []
        self._initialize_events()
        self.execution_steps = []

    def _initialize_events(self):
        for i, (p1, p2) in enumerate(self.segments):
            heapq.heappush(self.event_queue, (p1[0], 'start', i, p1, p2))
            heapq.heappush(self.event_queue, (p2[0], 'end', i, p1, p2))
            self.execution_steps.append(("Event Added", p1, p2))

    def _find_intersection(self, s1, s2):
        (x1, y1), (x2, y2) = s1
        (x3, y3), (x4, y4) = s2

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None

        intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

        if (min(x1, x2) <= intersect_x <= max(x1, x2) and min(y1, y2) <= intersect_y <= max(y1, y2) and
                min(x3, x4) <= intersect_x <= max(x3, x4) and min(y3, y4) <= intersect_y <= max(y3, y4)):
            return (intersect_x, intersect_y)
        return None

    def _process_event(self, event_type, seg_id, p1, p2):
        if event_type == "start":
            self.active_segments[seg_id] = (p1, p2)
            self.execution_steps.append(("Segment Added", seg_id, p1, p2))
            neighbors = list(self.active_segments.values())

            for i in range(len(neighbors) - 1):
                for j in range(i + 1, len(neighbors)):
                    intersection = self._find_intersection(neighbors[i], neighbors[j])
                    if intersection:
                        self.intersections.append(intersection)
                        self.execution_steps.append(("Intersection Found", neighbors[i], neighbors[j], intersection))

        elif event_type == "end":
            if seg_id in self.active_segments:
                del self.active_segments[seg_id]
                self.execution_steps.append(("Segment Removed", seg_id, p1, p2))

    def process(self):
        while self.event_queue:
            x, event_type, seg_id, p1, p2 = heapq.heappop(self.event_queue)
            self._process_event(event_type, seg_id, p1, p2)
        return self.intersections

    def get_intersections(self):
        result = {"Total Intersections": len(self.intersections), "Details": {}}
        for idx, intersection in enumerate(self.intersections):
            result["Details"][idx] = {"Intersection Point": intersection, "Segments Involved": []}
            for seg_id, segment in self.active_segments.items():
                if self._find_intersection(segment, intersection):
                    result["Details"][idx]["Segments Involved"].append(seg_id)
        return result

    def get_active_segments(self):
        active_details = {"Total Active": len(self.active_segments), "Segment Data": {}}
        for seg_id, segment in self.active_segments.items():
            active_details["Segment Data"][seg_id] = {"Start Point": segment[0], "End Point": segment[1]}
        return active_details

    def get_execution_steps(self):
        execution_log = {"Total Steps": len(self.execution_steps), "Detailed Steps": {}}
        step_counter = 0
        for step in self.execution_steps:
            execution_log["Detailed Steps"][step_counter] = {"Action": step[0], "Details": step[1:]}
            step_counter += 1
        return execution_log


# Example Usage:
segments = [
    ((0, 0), (5, 5)),
    ((2, 6), (4, 2)),
    ((1, 4), (6, 3)),
    ((3, 0), (3, 7))
]

bo_algorithm = BentleyOttmann(segments)
bo_algorithm.process()

intersections_result = bo_algorithm.get_intersections()
active_segments_result = bo_algorithm.get_active_segments()
execution_log_result = bo_algorithm.get_execution_steps()

# Z-algorithm

def z_algorithm(s):
    n = len(s)
    z = [0] * n
    left, right = 0, 0
    execution_steps = []

    print(f"Input string: {s}")
    print("Initializing Z-array...\n")

    for i in range(1, n):
        if i <= right:
            z[i] = min(right - i + 1, z[i - left])
            execution_steps.append(("Using previously computed Z-value", i, z[i], left, right))

        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
            execution_steps.append(("Extending Z-value", i, z[i], left, right))

        if i + z[i] - 1 > right:
            left, right = i, i + z[i] - 1
            execution_steps.append(("Updating Left and Right Boundaries", i, left, right))

        print(f"Step {i}: Z[{i}] = {z[i]}, Left = {left}, Right = {right}")

    print("\nExecution Steps Log:")
    for step in execution_steps:
        print(step)

    print("\nFinal Z-array:", z)
    return z


def compare_substring_occurrences(text, pattern):
    combined = pattern + "$" + text
    z_values = z_algorithm(combined)
    matches = [i - len(pattern) - 1 for i in range(len(z_values)) if z_values[i] == len(pattern)]

    print("\nPattern Occurrences:")
    for match in matches:
        print(f"Pattern found at index {match}")

    return matches


def print_execution_summary(execution_steps):
    print("\nDetailed Execution Summary:")
    step_count = 1
    operation_summary = {"Using previously computed Z-value": 0, "Extending Z-value": 0, "Updating Left and Right Boundaries": 0}

    for step in execution_steps:
        operation_summary[step[0]] += 1
        print(f"Step {step_count}: {step}")
        step_count += 1

    print("\nOperations Summary:")
    for operation, count in operation_summary.items():
        print(f"{operation}: {count} times")


def find_pattern_positions(text, pattern):
    print("\nFinding occurrences of pattern:", pattern)
    matches = compare_substring_occurrences(text, pattern)

    print("\nIdentified Pattern Positions:")
    position_details = {"Pattern": pattern, "Occurrences": matches, "Total Matches": len(matches)}
    return position_details


def validate_pattern(text, pattern):
    if len(pattern) > len(text):
        print("\nError: Pattern length is greater than text length!")
        return False
    if not pattern or not text:
        print("\nError: Text or pattern is empty!")
        return False
    return True


def process_z_algorithm(text, pattern):
    if not validate_pattern(text, pattern):
        return None

    print("\nProcessing Z-algorithm for:", text)
    z_values = z_algorithm(text)

    pattern_positions = find_pattern_positions(text, pattern)

    print("\nFinal Pattern Position Details:")
    print(pattern_positions)

    return pattern_positions


# Example Usage:
test_string = "abacaba"
test_pattern = "aba"

process_z_algorithm(test_string, test_pattern)



# Suffix Tree 

class SuffixTreeNode:
    def __init__(self):
        self.children = {}
        self.start = None
        self.end = None
        self.link = None


class SuffixTree:
    def __init__(self, text):
        self.text = text + "$"  # Append terminal symbol
        self.root = SuffixTreeNode()
        self.active_node = self.root
        self.active_edge = 0
        self.active_length = 0
        self.remaining_suffix_count = 0
        self.end = -1
        self.last_created_node = None
        self.build_tree()

    def build_tree(self):
        for i in range(len(self.text)):
            self.extend_suffix_tree(i)

    def extend_suffix_tree(self, pos):
        self.end += 1
        self.remaining_suffix_count += 1
        self.last_created_node = None

        while self.remaining_suffix_count > 0:
            if self.active_length == 0:
                self.active_edge = pos

            if self.text[self.active_edge] not in self.active_node.children:
                new_node = SuffixTreeNode()
                new_node.start = pos
                new_node.end = self.end
                self.active_node.children[self.text[self.active_edge]] = new_node

                if self.last_created_node:
                    self.last_created_node.link = self.active_node
                    self.last_created_node = None
            else:
                next_node = self.active_node.children[self.text[self.active_edge]]
                edge_length = next_node.end - next_node.start + 1

                if self.active_length >= edge_length:
                    self.active_edge += edge_length
                    self.active_length -= edge_length
                    self.active_node = next_node
                    continue

                if self.text[next_node.start + self.active_length] == self.text[pos]:
                    self.active_length += 1
                    if self.last_created_node:
                        self.last_created_node.link = self.active_node
                        self.last_created_node = None
                    break

                split_node = SuffixTreeNode()
                split_node.start = next_node.start
                split_node.end = next_node.start + self.active_length - 1
                self.active_node.children[self.text[self.active_edge]] = split_node

                next_node.start += self.active_length
                split_node.children[self.text[next_node.start]] = next_node

                leaf = SuffixTreeNode()
                leaf.start = pos
                leaf.end = self.end
                split_node.children[self.text[pos]] = leaf

                if self.last_created_node:
                    self.last_created_node.link = split_node

                self.last_created_node = split_node

            self.remaining_suffix_count -= 1

            if self.active_node == self.root and self.active_length > 0:
                self.active_length -= 1
                self.active_edge = pos - self.remaining_suffix_count + 1
            elif self.active_node.link:
                self.active_node = self.active_node.link
            else:
                self.active_node = self.root

    def search(self, pattern):
        current = self.root
        i = 0

        while i < len(pattern):
            if pattern[i] in current.children:
                node = current.children[pattern[i]]
                j = node.start
                while j <= node.end and i < len(pattern) and self.text[j] == pattern[i]:
                    j += 1
                    i += 1
                if j > node.end:
                    current = node
                else:
                    return False
            else:
                return False
        return True

    def get_longest_common_substring(self):
        stack = [(self.root, 0)]
        max_length = 0
        longest_substr = ""

        while stack:
            node, depth = stack.pop()

            if len(node.children) > 1 and depth > max_length:
                max_length = depth
                longest_substr = self.text[node.start:node.start + depth]

            for child in node.children.values():
                stack.append((child, depth + (child.end - child.start + 1)))

        return longest_substr


# Example Usage:
suffix_tree = SuffixTree("banana")
print("\nSearching for 'nan':", suffix_tree.search("nan"))
print("Searching for 'apple':", suffix_tree.search("apple"))
print("Longest Common Substring:", suffix_tree.get_longest_common_substring())


# Persistent Segment Trees

class PersistentSegmentTree:
    class Node:
        def __init__(self, value=0, left=None, right=None):
            self.value = value
            self.left = left
            self.right = right

    def __init__(self, size):
        self.size = size
        self.roots = [self._build(0, size - 1)]

    def _build(self, start, end):
        if start == end:
            return self.Node()
        mid = (start + end) // 2
        left_child = self._build(start, mid)
        right_child = self._build(mid + 1, end)
        return self.Node(left_child.value + right_child.value, left_child, right_child)

    def update(self, prev_version, pos, value):
        new_root = self._update(self.roots[prev_version], 0, self.size - 1, pos, value)
        self.roots.append(new_root)

    def _update(self, node, start, end, pos, value):
        if start == end:
            return self.Node(value)
        mid = (start + end) // 2
        if pos <= mid:
            new_left = self._update(node.left, start, mid, pos, value)
            return self.Node(new_left.value + node.right.value, new_left, node.right)
        else:
            new_right = self._update(node.right, mid + 1, end, pos, value)
            return self.Node(node.left.value + new_right.value, node.left, new_right)

    def query(self, version, left, right):
        return self._query(self.roots[version], 0, self.size - 1, left, right)

    def _query(self, node, start, end, left, right):
        if left > end or right < start:
            return 0
        if left <= start and right >= end:
            return node.value
        mid = (start + end) // 2
        return self._query(node.left, start, mid, left, right) + self._query(node.right, mid + 1, end, left, right)

    def get_versions_count(self):
        return len(self.roots)

    def get_structure_at_version(self, version):
        structure = {}
        self._traverse(self.roots[version], 0, self.size - 1, structure)
        return structure

    def _traverse(self, node, start, end, structure):
        structure[start, end] = node.value
        if start != end:
            mid = (start + end) // 2
            self._traverse(node.left, start, mid, structure)
            self._traverse(node.right, mid + 1, end, structure)

    def get_execution_log(self):
        log_details = {"Total_Versions": self.get_versions_count(), "Version_History": {}}
        for version in range(len(self.roots)):
            log_details["Version_History"][version] = {"Segment_Tree_Structure": self.get_structure_at_version(version)}
        return log_details


# Example Usage:
size = 10
pst = PersistentSegmentTree(size)

pst.update(0, 3, 5)
pst.update(1, 6, 8)
pst.update(2, 2, 7)

query_result_1 = pst.query(0, 0, 5)
query_result_2 = pst.query(1, 3, 7)
query_result_3 = pst.query(2, 2, 6)

versions_count = pst.get_versions_count()
structure_at_version_2 = pst.get_structure_at_version(2)
execution_log_result = pst.get_execution_log()


# Dynamic Connectivity Algorithm


import random

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
        self.connected_components = size

    def find(self, node):
        path_traversed = []
        while self.parent[node] != node:
            path_traversed.append(node)
            node = self.parent[node]
        for traversed_node in path_traversed:
            self.parent[traversed_node] = node
        return node

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
            self.connected_components -= 1

    def connected(self, u, v):
        return self.find(u) == self.find(v)

    def count_components(self):
        return self.connected_components

    def get_structure(self):
        return {"Parents": self.parent, "Ranks": self.rank, "Total Components": self.connected_components}


class LinkCutTree:
    class Node:
        def __init__(self, value):
            self.value = value
            self.parent = None
            self.left = None
            self.right = None
            self.reversed = False

    def __init__(self, size):
        self.nodes = [self.Node(i) for i in range(size)]

    def find_root(self, node):
        path = []
        while self.nodes[node].parent:
            path.append(node)
            node = self.nodes[node].parent.value
        return node

    def link(self, u, v):
        root_u = self.find_root(u)
        root_v = self.find_root(v)
        if root_u != root_v:
            self.nodes[root_u].parent = self.nodes[root_v]

    def cut(self, u, v):
        if self.nodes[u].parent == self.nodes[v]:
            self.nodes[u].parent = None
        elif self.nodes[v].parent == self.nodes[u]:
            self.nodes[v].parent = None

    def connected(self, u, v):
        return self.find_root(u) == self.find_root(v)

    def get_tree_structure(self):
        tree_representation = {}
        for node in self.nodes:
            tree_representation[node.value] = {
                "Parent": node.parent.value if node.parent else None,
                "Left Child": node.left.value if node.left else None,
                "Right Child": node.right.value if node.right else None
            }
        return tree_representation


class DynamicConnectivity:
    def __init__(self, size):
        self.union_find = UnionFind(size)
        self.link_cut_tree = LinkCutTree(size)
        self.execution_log = []

    def add_edge(self, u, v):
        self.union_find.union(u, v)
        self.link_cut_tree.link(u, v)
        self.execution_log.append(("Added Edge", u, v))

    def remove_edge(self, u, v):
        self.link_cut_tree.cut(u, v)
        self.execution_log.append(("Removed Edge", u, v))

    def query_connected(self, u, v):
        result = self.union_find.connected(u, v) or self.link_cut_tree.connected(u, v)
        self.execution_log.append(("Connectivity Check", u, v, result))
        return result

    def total_components(self):
        component_count = self.union_find.count_components()
        self.execution_log.append(("Total Components", component_count))
        return component_count

    def get_execution_log(self):
        return {"Execution Steps": self.execution_log, "Total Steps": len(self.execution_log)}

    def get_full_structure(self):
        return {
            "Union-Find Structure": self.union_find.get_structure(),
            "Link-Cut Tree Structure": self.link_cut_tree.get_tree_structure(),
            "Execution History": self.execution_log
        }


# Example Usage:
dc = DynamicConnectivity(10)

edges = [(random.randint(0, 9), random.randint(0, 9)) for _ in range(7)]
for u, v in edges:
    dc.add_edge(u, v)

print("Connected (0, 1):", dc.query_connected(0, 1))
print("Connected (2, 7):", dc.query_connected(2, 7))

dc.remove_edge(edges[2][0], edges[2][1])

print("Connected (2, 7) after removal:", dc.query_connected(2, 7))
print("Total Components:", dc.total_components())

execution_log = dc.get_execution_log()
full_structure = dc.get_full_structure()



# Chained Hash table


class HashNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None

class ChainedHashTable:
    def __init__(self, initial_capacity=10):
        self.capacity = initial_capacity
        self.size = 0
        self.table = [None] * self.capacity
        self.execution_log = []

    def _hash(self, key):
        return hash(key) % self.capacity

    def insert(self, key, value):
        index = self._hash(key)
        node = self.table[index]
        
        if node is None:
            self.table[index] = HashNode(key, value)
            self.size += 1
            self.execution_log.append(("Inserted", key, value))
            return

        prev = None
        while node:
            if node.key == key:
                node.value = value
                self.execution_log.append(("Updated", key, value))
                return
            prev = node
            node = node.next
        
        prev.next = HashNode(key, value)
        self.size += 1
        self.execution_log.append(("Inserted at Chain", key, value))

    def search(self, key):
        index = self._hash(key)
        node = self.table[index]
        
        while node:
            if node.key == key:
                self.execution_log.append(("Search Found", key, node.value))
                return node.value
            node = node.next
        
        self.execution_log.append(("Search Not Found", key))
        return None

    def delete(self, key):
        index = self._hash(key)
        node = self.table[index]
        prev = None

        while node:
            if node.key == key:
                if prev:
                    prev.next = node.next
                else:
                    self.table[index] = node.next
                self.size -= 1
                self.execution_log.append(("Deleted", key))
                return True
            prev = node
            node = node.next

        self.execution_log.append(("Deletion Failed", key))
        return False

    def resize(self):
        old_table = self.table
        self.capacity *= 2
        self.table = [None] * self.capacity
        self.size = 0
        self.execution_log.append(("Resized Table", self.capacity))

        for node in old_table:
            while node:
                self.insert(node.key, node.value)
                node = node.next

    def load_factor(self):
        return self.size / self.capacity

    def check_resize(self):
        if self.load_factor() > 0.7:
            self.resize()

    def get_structure(self):
        structure = {}
        for i, node in enumerate(self.table):
            chain = []
            while node:
                chain.append((node.key, node.value))
                node = node.next
            structure[i] = chain
        return structure

    def get_execution_log(self):
        return {"Execution Steps": self.execution_log, "Total Steps": len(self.execution_log)}


# Example Usage:
hash_table = ChainedHashTable()

keys = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew"]
values = [5, 3, 7, 6, 2, 8, 1, 4]

for k, v in zip(keys, values):
    hash_table.insert(k, v)
    hash_table.check_resize()

print("\nSearch 'banana':", hash_table.search("banana"))
print("Search 'mango':", hash_table.search("mango"))

hash_table.delete("cherry")
print("Search after deletion 'cherry':", hash_table.search("cherry"))

print("\nHash Table Structure:", hash_table.get_structure())
print("\nExecution Log:", hash_table.get_execution_log())


# Double ended Queue 


class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None

class Deque:
    def __init__(self):
        self.front = None
        self.rear = None
        self.size = 0
        self.execution_log = []

    def is_empty(self):
        return self.size == 0

    def add_front(self, value):
        new_node = Node(value)
        if self.is_empty():
            self.front = self.rear = new_node
        else:
            new_node.next = self.front
            self.front.prev = new_node
            self.front = new_node
        self.size += 1
        self.execution_log.append(("Added Front", value))

    def add_rear(self, value):
        new_node = Node(value)
        if self.is_empty():
            self.front = self.rear = new_node
        else:
            new_node.prev = self.rear
            self.rear.next = new_node
            self.rear = new_node
        self.size += 1
        self.execution_log.append(("Added Rear", value))

    def remove_front(self):
        if self.is_empty():
            return None
        value = self.front.value
        self.front = self.front.next
        if self.front:
            self.front.prev = None
        else:
            self.rear = None
        self.size -= 1
        self.execution_log.append(("Removed Front", value))
        return value

    def remove_rear(self):
        if self.is_empty():
            return None
        value = self.rear.value
        self.rear = self.rear.prev
        if self.rear:
            self.rear.next = None
        else:
            self.front = None
        self.size -= 1
        self.execution_log.append(("Removed Rear", value))
        return value

    def peek_front(self):
        if self.is_empty():
            self.execution_log.append(("Peek Front Failed", "Deque is Empty"))
            return None
        self.execution_log.append(("Peek Front", self.front.value))
        return self.front.value

    def peek_rear(self):
        if self.is_empty():
            self.execution_log.append(("Peek Rear Failed", "Deque is Empty"))
            return None
        self.execution_log.append(("Peek Rear", self.rear.value))
        return self.rear.value

    def get_size(self):
        size_details = {
            "Current Size": self.size,
            "Deque Empty": self.is_empty(),
            "Front Value": self.peek_front(),
            "Rear Value": self.peek_rear()
        }
        self.execution_log.append(("Size Checked", size_details))
        return size_details

    def reverse_deque(self):
        current = self.front
        while current:
            current.next, current.prev = current.prev, current.next
            current = current.prev
        self.front, self.rear = self.rear, self.front
        self.execution_log.append(("Reversed Deque",))

    def contains(self, value):
        current = self.front
        while current:
            if current.value == value:
                self.execution_log.append(("Found", value))
                return True
            current = current.next
        self.execution_log.append(("Not Found", value))
        return False

    def display(self):
        elements = []
        current = self.front
        while current:
            elements.append(current.value)
            current = current.next
        self.execution_log.append(("Displayed Deque", elements))
        return elements

    def get_execution_log(self):
        return {"Execution Steps": self.execution_log, "Total Steps": len(self.execution_log)}


# Example Usage:
dq = Deque()

dq.add_front(10)
dq.add_front(20)
dq.add_rear(30)
dq.add_rear(40)

print("\nDeque Elements:", dq.display())
print("Front Element:", dq.peek_front())
print("Rear Element:", dq.peek_rear())

dq.remove_front()
dq.remove_rear()

print("\nDeque Elements after removals:", dq.display())
print("Size of Deque:", dq.get_size())

dq.add_front(50)
dq.add_rear(60)
dq.reverse_deque()

print("\nDeque Elements after reversing:", dq.display())

print("\nChecking if '30' exists in Deque:", dq.contains(30))
print("\nExecution Log:", dq.get_execution_log())


# Van Emde Boas Tree

class VanEmdeBoas:
    def __init__(self, universe_size):
        self.u = universe_size
        self.min = None
        self.max = None

        if universe_size <= 2:
            self.summary = None
            self.cluster = None
        else:
            sqrt_u = int(universe_size ** 0.5)
            self.summary = VanEmdeBoas(sqrt_u)
            self.cluster = [VanEmdeBoas(sqrt_u) for _ in range(sqrt_u)]

    def high(self, x):
        if x < 0 or x >= self.u:
            raise ValueError(f"Value {x} is out of bounds for universe size {self.u}")
        
        root_size = int(self.u ** 0.5)
        high_value = x // root_size

        tracking_data = {
            "Input": x,
            "Computed Root Size": root_size,
            "High Value": high_value
        }

        return high_value, tracking_data

    def low(self, x):
        if x < 0 or x >= self.u:
            raise ValueError(f"Value {x} is out of bounds for universe size {self.u}")
        
        root_size = int(self.u ** 0.5)
        low_value = x % root_size

        tracking_data = {
            "Input": x,
            "Computed Root Size": root_size,
            "Low Value": low_value
        }

        return low_value, tracking_data

    def index(self, high, low):
        if high < 0 or low < 0 or high >= int(self.u ** 0.5) or low >= int(self.u ** 0.5):
            raise ValueError(f"Invalid index computation with high={high}, low={low} for universe {self.u}")

        root_size = int(self.u ** 0.5)
        computed_index = high * root_size + low

        tracking_data = {
            "High Component": high,
            "Low Component": low,
            "Computed Root Size": root_size,
            "Final Index": computed_index
        }

        return computed_index, tracking_data

    def insert(self, x):
        if self.min is None:
            self.min = self.max = x
        else:
            if x < self.min:
                x, self.min = self.min, x
            if x > self.max:
                self.max = x
            
            if self.u > 2:
                high_x, high_tracking = self.high(x)
                low_x, low_tracking = self.low(x)

                if self.cluster[high_x].min is None:
                    self.summary.insert(high_x)
                    self.cluster[high_x].insert(low_x)
                else:
                    self.cluster[high_x].insert(low_x)

    def contains(self, x):
        if x == self.min or x == self.max:
            return True
        elif self.u <= 2:
            return False
        else:
            high_x, _ = self.high(x)
            low_x, _ = self.low(x)
            return self.cluster[high_x].contains(low_x)

    def successor(self, x):
        if self.u <= 2:
            if x == 0 and self.max == 1:
                return 1
            return None
        elif self.min is not None and x < self.min:
            return self.min
        else:
            high_x, _ = self.high(x)
            low_x, _ = self.low(x)

            if low_x < self.cluster[high_x].max:
                succ_low = self.cluster[high_x].successor(low_x)
                final_index, index_tracking = self.index(high_x, succ_low)
                return final_index
            else:
                succ_high = self.summary.successor(high_x)
                if succ_high is None:
                    return None
                final_index, index_tracking = self.index(succ_high, self.cluster[succ_high].min)
                return final_index

    def predecessor(self, x):
        if self.u <= 2:
            if x == 1 and self.min == 0:
                return 0
            return None
        elif self.max is not None and x > self.max:
            return self.max
        else:
            high_x, _ = self.high(x)
            low_x, _ = self.low(x)

            if low_x > self.cluster[high_x].min:
                pred_low = self.cluster[high_x].predecessor(low_x)
                final_index, index_tracking = self.index(high_x, pred_low)
                return final_index
            else:
                pred_high = self.summary.predecessor(high_x)
                if pred_high is None:
                    return None
                final_index, index_tracking = self.index(pred_high, self.cluster[pred_high].max)
                return final_index


# Example Usage:
veb_tree = VanEmdeBoas(16)

veb_tree.insert(3)
veb_tree.insert(7)
veb_tree.insert(10)
veb_tree.insert(12)

print("\nContains 7:", veb_tree.contains(7))
print("Contains 5:", veb_tree.contains(5))

print("\nSuccessor of 7:", veb_tree.successor(7))
print("Predecessor of 7:", veb_tree.predecessor(7))


# Sparse Table

import math

class SparseTable:
    def __init__(self, arr):
        self.n = len(arr)
        self.log = [0] * (self.n + 1)
        self.table = [[0] * (math.ceil(math.log2(self.n)) + 1) for _ in range(self.n)]
        self.execution_log = []
        self._precompute_logs()
        self._build_table(arr)

    def _precompute_logs(self):
        self.execution_log.append("Starting log precomputation")
        for i in range(2, self.n + 1):
            self.log[i] = self.log[i // 2] + 1
            self.execution_log.append(f"Computed log[{i}] = {self.log[i]}")

    def _build_table(self, arr):
        self.execution_log.append("Starting Sparse Table construction")
        for i in range(self.n):
            self.table[i][0] = arr[i]
            self.execution_log.append(f"Set table[{i}][0] = {arr[i]}")

        for j in range(1, math.ceil(math.log2(self.n)) + 1):
            for i in range(self.n - (1 << j) + 1):
                self.table[i][j] = min(self.table[i][j - 1], self.table[i + (1 << (j - 1))][j - 1])
                self.execution_log.append(f"Computed table[{i}][{j}] = {self.table[i][j]}")

    def query(self, left, right):
        if left < 0 or right >= self.n or left > right:
            self.execution_log.append(f"Invalid query ({left}, {right})")
            return None

        j = self.log[right - left + 1]
        result = min(self.table[left][j], self.table[right - (1 << j) + 1][j])
        self.execution_log.append(f"Query ({left}, {right}) => min({self.table[left][j]}, {self.table[right - (1 << j) + 1][j]}) = {result}")
        return result

    def update_table(self, index, value):
        if index < 0 or index >= self.n:
            self.execution_log.append(f"Update failed: Index {index} out of bounds")
            return

        self.table[index][0] = value
        self.execution_log.append(f"Updated table[{index}][0] = {value}")

        for j in range(1, math.ceil(math.log2(self.n)) + 1):
            for i in range(self.n - (1 << j) + 1):
                self.table[i][j] = min(self.table[i][j - 1], self.table[i + (1 << (j - 1))][j - 1])
                self.execution_log.append(f"Recomputed table[{i}][{j}] after update")

    def get_table_structure(self):
        return {"Sparse Table": self.table, "Precomputed Log Values": self.log}

    def get_execution_log(self):
        return {"Execution Steps": self.execution_log, "Total Steps": len(self.execution_log)}


# Example Usage:
arr = [1, 3, 2, 7, 9, 11, 4, 8]
sparse_table = SparseTable(arr)

query_result_1 = sparse_table.query(2, 6)
query_result_2 = sparse_table.query(0, 5)

sparse_table.update_table(4, 5)

table_data = sparse_table.get_table_structure()
execution_log = sparse_table.get_execution_log()


# Aho-Corasick Algorithm


from collections import deque

class TrieNode:
    def __init__(self):
        self.children = {}
        self.failure_link = None
        self.output = []

class AhoCorasick:
    def __init__(self, patterns):
        self.root = TrieNode()
        self.execution_log = []
        self._build_trie(patterns)
        self._build_failure_links()

    def _build_trie(self, patterns):
        self.execution_log.append("Starting Trie Construction")
        for pattern in patterns:
            node = self.root
            for char in pattern:
                if char not in node.children:
                    node.children[char] = TrieNode()
                    self.execution_log.append(f"Inserted '{char}' in Trie")
                node = node.children[char]
            node.output.append(pattern)
            self.execution_log.append(f"Pattern '{pattern}' added to output list")

    def _build_failure_links(self):
        self.execution_log.append("Starting Failure Link Construction")
        queue = deque()
        for child in self.root.children.values():
            child.failure_link = self.root
            queue.append(child)

        while queue:
            node = queue.popleft()
            for char, child in node.children.items():
                queue.append(child)
                failure = node.failure_link
                while failure is not None and char not in failure.children:
                    failure = failure.failure_link
                child.failure_link = failure.children[char] if failure else self.root
                child.output += child.failure_link.output
                self.execution_log.append(f"Set failure link for '{char}'")

    def search(self, text):
        self.execution_log.append(f"Starting Search in '{text}'")
        node = self.root
        results = []
        
        for i, char in enumerate(text):
            while node is not None and char not in node.children:
                node = node.failure_link
            if node is None:
                node = self.root
                continue
            node = node.children[char]
            for pattern in node.output:
                results.append((i - len(pattern) + 1, pattern))
                self.execution_log.append(f"Found pattern '{pattern}' at position {i - len(pattern) + 1}")

        return results

    def get_execution_log(self):
        return {"Execution Steps": self.execution_log, "Total Steps": len(self.execution_log)}


# Example Usage:
patterns = ["he", "she", "hers", "his"]
text = "she is searching for his and hers"

aho_corasick = AhoCorasick(patterns)
search_results = aho_corasick.search(text)

print("\nSearch Results:", search_results)
print("\nExecution Log:", aho_corasick.get_execution_log())



# Karatsuba Algorithm


def karatsuba(x, y):
    """Recursive Karatsuba multiplication algorithm"""
    if x < 10 or y < 10:  # Base case: single-digit multiplication
        return x * y

    n = max(len(str(x)), len(str(y)))
    half_n = n // 2

    # Split x and y into two parts
    x_high, x_low = divmod(x, 10 ** half_n)
    y_high, y_low = divmod(y, 10 ** half_n)

    # Compute three multiplications
    z0 = karatsuba(x_low, y_low)
    z1 = karatsuba((x_low + x_high), (y_low + y_high))
    z2 = karatsuba(x_high, y_high)

    # Karatsuba formula
    result = (z2 * 10 ** (2 * half_n)) + ((z1 - z2 - z0) * 10 ** half_n) + z0
    
    return result


class KaratsubaMultiplication:
    """Encapsulated class implementation of Karatsuba Algorithm"""
    
    def __init__(self, num1, num2):
        self.num1 = num1
        self.num2 = num2
        self.execution_log = []
        
    def compute(self):
        """Compute Karatsuba multiplication with logging"""
        result = self._karatsuba(self.num1, self.num2)
        self.execution_log.append(("Final Result", result))
        return result

    def _karatsuba(self, x, y):
        """Recursive Karatsuba multiplication with detailed logging"""
        if x < 10 or y < 10:
            self.execution_log.append(("Base Case Multiplication", x, y, x * y))
            return x * y

        n = max(len(str(x)), len(str(y)))
        half_n = n // 2

        x_high, x_low = divmod(x, 10 ** half_n)
        y_high, y_low = divmod(y, 10 ** half_n)

        z0 = self._karatsuba(x_low, y_low)
        z1 = self._karatsuba((x_low + x_high), (y_low + y_high))
        z2 = self._karatsuba(x_high, y_high)

        result = (z2 * 10 ** (2 * half_n)) + ((z1 - z2 - z0) * 10 ** half_n) + z0
        
        self.execution_log.append(("Karatsuba Step", x, y, result))
        return result

    def get_execution_log(self):
        """Retrieve detailed execution steps"""
        return {"Execution Steps": self.execution_log, "Total Steps": len(self.execution_log)}


# Example Usage:
num1 = 12345678
num2 = 87654321

karatsuba_instance = KaratsubaMultiplication(num1, num2)
final_result = karatsuba_instance.compute()

print("\nFinal Karatsuba Multiplication Result:", final_result)
print("\nExecution Log:", karatsuba_instance.get_execution_log())


# Circular Buffer

class CircularBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0
        self.execution_log = []

    def is_empty(self):
        empty_status = self.size == 0
        self.execution_log.append(("Checked if Buffer is Empty", empty_status))
        return empty_status

    def is_full(self):
        full_status = self.size == self.capacity
        self.execution_log.append(("Checked if Buffer is Full", full_status))
        return full_status

    def enqueue(self, value):
        if self.is_full():
            self.execution_log.append(f"Enqueue Failed: Buffer is Full (Attempted value: {value})")
            return False

        self.buffer[self.tail] = value
        self.execution_log.append(f"Enqueued {value} at position {self.tail}")
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1
        self.execution_log.append(("Buffer Size Updated", self.size))
        return True

    def dequeue(self):
        if self.is_empty():
            self.execution_log.append("Dequeue Failed: Buffer is Empty")
            return None

        value = self.buffer[self.head]
        self.buffer[self.head] = None
        self.execution_log.append(f"Dequeued {value} from position {self.head}")
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        self.execution_log.append(("Buffer Size Updated", self.size))
        return value

    def peek(self):
        if self.is_empty():
            self.execution_log.append("Peek Failed: Buffer is Empty")
            return None
        
        peek_value = self.buffer[self.head]
        self.execution_log.append(("Peeked at Front Element", peek_value))
        return peek_value

    def get_buffer_structure(self):
        buffer_state = {
            "Buffer Data": self.buffer[:],
            "Head Position": self.head,
            "Tail Position": self.tail,
            "Current Size": self.size
        }
        self.execution_log.append(("Retrieved Buffer Structure", buffer_state))
        return buffer_state

    def get_execution_log(self):
        execution_data = {
            "Execution Steps": self.execution_log,
            "Total Steps Recorded": len(self.execution_log)
        }
        return execution_data

    def clear(self):
        self.buffer = [None] * self.capacity
        self.head = 0
        self.tail = 0
        self.size = 0
        self.execution_log.append("Buffer Cleared")

    def available_space(self):
        """Returns the number of available slots in the buffer."""
        free_space = self.capacity - self.size
        self.execution_log.append(("Checked Available Space", free_space))
        return free_space

    def reverse_buffer(self):
        """Reverses the buffer's contents while maintaining cyclic structure."""
        reversed_data = [None] * self.capacity
        temp = self.head
        for i in range(self.size):
            reversed_data[i] = self.buffer[temp]
            temp = (temp + 1) % self.capacity
        self.buffer = reversed_data
        self.head = 0
        self.tail = self.size
        self.execution_log.append(("Reversed Buffer Content", self.buffer[:]))


# Example Usage:
cb = CircularBuffer(5)

cb.enqueue(10)
cb.enqueue(20)
cb.enqueue(30)
cb.dequeue()
cb.enqueue(40)
cb.enqueue(50)
cb.enqueue(60)  # Should fail since the buffer is full
cb.dequeue()
cb.peek()

print("\nCircular Buffer State:", cb.get_buffer_structure())
print("\nAvailable Space:", cb.available_space())

cb.reverse_buffer()
print("\nReversed Buffer State:", cb.get_buffer_structure())

print("\nExecution Log:", cb.get_execution_log())



# Longest Palindromic Substring

class Manacher:
    def __init__(self, s):
        self.original = s
        self.processed = self._transform_string(s)
        self.n = len(self.processed)
        self.palindrome_lengths = [0] * self.n
        self.execution_log = []
        self.center = 0
        self.right = 0
        self.longest_palindrome = ""

    def _transform_string(self, s):
        """Transforms input string to handle even-length cases."""
        transformed = "^#" + "#".join(s) + "#$"
        self.execution_log.append(("Transformed String", transformed))
        return transformed

    def compute_longest_palindrome(self):
        """Manacher’s Algorithm for finding longest palindromic substring."""
        self.execution_log.append("Starting computation for longest palindrome.")

        for i in range(1, self.n - 1):
            mirror = 2 * self.center - i
            
            if i < self.right:
                self.palindrome_lengths[i] = min(self.right - i, self.palindrome_lengths[mirror])
                self.execution_log.append((f"Mirroring at index {i}", f"Mirrored Length: {self.palindrome_lengths[i]}"))

            while self.processed[i + self.palindrome_lengths[i] + 1] == self.processed[i - self.palindrome_lengths[i] - 1]:
                self.palindrome_lengths[i] += 1
                self.execution_log.append(("Expanding Palindrome", i, self.palindrome_lengths[i]))

            if i + self.palindrome_lengths[i] > self.right:
                self.center, self.right = i, i + self.palindrome_lengths[i]
                self.execution_log.append(("Updated Center & Right Boundaries", self.center, self.right))

        max_length, center_index = max((val, idx) for idx, val in enumerate(self.palindrome_lengths))
        start = (center_index - max_length) // 2  # Convert back to original index
        self.longest_palindrome = self.original[start:start + max_length]

        self.execution_log.append(("Longest Palindromic Substring Found", self.longest_palindrome))
        return self.longest_palindrome

    def query_substring(self, start, end):
        """Checks if a given substring is palindromic."""
        if start < 0 or end >= len(self.original) or start > end:
            self.execution_log.append(("Invalid Query", start, end))
            return None

        substring = self.original[start:end+1]
        is_palindrome = substring == substring[::-1]
        self.execution_log.append(("Checked Substring", substring, "Is Palindrome:", is_palindrome))
        return is_palindrome

    def get_execution_log(self):
        """Retrieves execution steps."""
        return {"Execution Steps": self.execution_log, "Total Steps": len(self.execution_log)}

    def get_full_structure(self):
        """Retrieves the full data structure of palindrome computations."""
        return {
            "Processed String": self.processed,
            "Palindrome Lengths": self.palindrome_lengths,
            "Original Input": self.original,
            "Execution Log": self.execution_log
        }


# Example Usage:
text = "cbbd"
manacher_instance = Manacher(text)
longest_palindrome = manacher_instance.compute_longest_palindrome()

print("\nLongest Palindromic Substring:", longest_palindrome)
print("\nQuerying 'bb' for Palindrome:", manacher_instance.query_substring(1, 2))
print("\nFull Structure:", manacher_instance.get_full_structure())
print("\nExecution Log:", manacher_instance.get_execution_log())


# Sudoku Solver


class SudokuSolver:
    def __init__(self, board):
        self.board = board
        self.execution_log = []
        self.size = 9
        self.subgrid_size = 3

    def solve(self):
        """Solves the Sudoku puzzle using backtracking."""
        self.execution_log.append("Starting Sudoku Solver")
        if self._backtrack():
            self.execution_log.append("Sudoku Solved Successfully")
            return self.board
        else:
            self.execution_log.append("No Solution Found")
            return None

    def _backtrack(self):
        """Recursive function to solve Sudoku using backtracking."""
        empty_cell = self._find_empty()
        if not empty_cell:
            return True  # Puzzle solved
        
        row, col = empty_cell
        for num in range(1, 10):
            if self._is_valid(num, row, col):
                self.board[row][col] = num
                self.execution_log.append(f"Placed {num} at ({row}, {col})")

                if self._backtrack():
                    return True
                
                self.board[row][col] = 0
                self.execution_log.append(f"Backtracked at ({row}, {col})")

        return False

    def _find_empty(self):
        """Finds an empty cell in the board."""
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == 0:
                    return row, col
        return None

    def _is_valid(self, num, row, col):
        """Checks if placing 'num' in (row, col) is valid."""
        if num in self.board[row]:  # Check row
            return False
        
        if num in [self.board[i][col] for i in range(self.size)]:  # Check column
            return False

        box_row, box_col = row // self.subgrid_size * self.subgrid_size, col // self.subgrid_size * self.subgrid_size
        for i in range(self.subgrid_size):
            for j in range(self.subgrid_size):
                if self.board[box_row + i][box_col + j] == num:
                    return False
        
        return True

    def validate_board(self):
        """Validates the initial Sudoku board setup."""
        for row in range(self.size):
            row_values = [num for num in self.board[row] if num != 0]
            if len(row_values) != len(set(row_values)):  # Check duplicate numbers in rows
                self.execution_log.append(f"Invalid Row: {row}")
                return False

        for col in range(self.size):
            col_values = [self.board[row][col] for row in range(self.size) if self.board[row][col] != 0]
            if len(col_values) != len(set(col_values)):  # Check duplicate numbers in columns
                self.execution_log.append(f"Invalid Column: {col}")
                return False

        for box_row in range(0, self.size, self.subgrid_size):
            for box_col in range(0, self.size, self.subgrid_size):
                subgrid_values = []
                for i in range(self.subgrid_size):
                    for j in range(self.subgrid_size):
                        value = self.board[box_row + i][box_col + j]
                        if value != 0:
                            subgrid_values.append(value)
                if len(subgrid_values) != len(set(subgrid_values)):  # Check duplicate numbers in subgrid
                    self.execution_log.append(f"Invalid Subgrid at ({box_row}, {box_col})")
                    return False
        
        return True

    def get_execution_log(self):
        """Retrieves execution steps for debugging."""
        return {"Execution Steps": self.execution_log, "Total Steps": len(self.execution_log)}

    def display_board(self):
        """Displays the Sudoku board."""
        print("\nCurrent Sudoku Board:")
        for row in self.board:
            print(" ".join(str(cell) if cell != 0 else "." for cell in row))


# Example Usage:
puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

solver = SudokuSolver(puzzle)

if solver.validate_board():
    solved_board = solver.solve()
    solver.display_board()
else:
    print("\nInvalid Sudoku Board Configuration")

print("\nExecution Log:", solver.get_execution_log())


# Permutation


class PermutationGenerator:
    def __init__(self, elements):
        self.elements = elements
        self.execution_log = []
        self.results = []
        self.used = [False] * len(elements)

    def generate(self):
        """Generates all permutations using recursive backtracking."""
        self.execution_log.append("Starting permutation computation.")
        self._backtrack([], self.used)
        self.execution_log.append("Completed permutation computation.")
        return self.results

    def _backtrack(self, path, used):
        """Recursive function to explore permutations."""
        if len(path) == len(self.elements):
            self.results.append(path[:])
            self.execution_log.append(("Permutation Found", path[:]))
            return
        
        for i in range(len(self.elements)):
            if used[i]:  # Skip already used elements
                continue
            
            used[i] = True
            path.append(self.elements[i])
            self.execution_log.append(("Added Element", self.elements[i], "Current Path", path[:]))

            self._backtrack(path, used)

            used[i] = False
            path.pop()
            self.execution_log.append(("Backtracked", path[:]))

    def query_permutation(self, target):
        """Checks if a specific permutation exists."""
        exists = target in self.results
        self.execution_log.append(("Checked Permutation", target, "Exists:", exists))
        return exists

    def get_execution_log(self):
        """Retrieves execution steps."""
        return {"Execution Steps": self.execution_log, "Total Steps": len(self.execution_log)}

    def get_all_permutations(self):
        """Retrieves all computed permutations."""
        return {"Total Permutations": len(self.results), "Permutations": self.results}


# Example Usage:
elements = [1, 2, 3]
perm_generator = PermutationGenerator(elements)

all_permutations = perm_generator.generate()
print("\nAll Permutations:", all_permutations)

query_result = perm_generator.query_permutation([3, 1, 2])
print("\nQuerying if [3, 1, 2] exists:", query_result)

execution_log = perm_generator.get_execution_log()
print("\nExecution Log:", execution_log)


# SubSet

class SubsetGenerator:
    def __init__(self, elements):
        self.elements = elements
        self.execution_log = []
        self.results = []

    def generate(self):
        """Generates all subsets using recursive backtracking."""
        self.execution_log.append("Starting subset computation.")
        self._backtrack(0, [])
        self.execution_log.append("Completed subset computation.")
        return self.results

    def _backtrack(self, index, path):
        """Recursive function to explore subsets."""
        self.results.append(path[:])
        self.execution_log.append(("Subset Found", path[:]))

        for i in range(index, len(self.elements)):
            path.append(self.elements[i])
            self.execution_log.append(("Added Element", self.elements[i], "Current Path", path[:]))

            self._backtrack(i + 1, path)

            path.pop()
            self.execution_log.append(("Backtracked", path[:]))

    def query_subset(self, target):
        """Checks if a specific subset exists with structured validation."""
        if not isinstance(target, list):
            self.execution_log.append(("Invalid Query Type", target, "Expected a list"))
            return False
        
        exists = target in self.results
        self.execution_log.append(("Checked Subset", target, "Exists:", exists))
        
        query_analysis = {
            "Requested Subset": target,
            "Exists": exists,
            "Subset Length": len(target),
            "Total Computed Subsets": len(self.results),
            "Execution Log": len(self.execution_log),
        }

        return query_analysis

    def get_execution_log(self):
        """Retrieves execution steps with detailed insights."""
        execution_data = {
            "Execution Steps": self.execution_log,
            "Total Steps Recorded": len(self.execution_log),
            "Operations Breakdown": {
                "Subset Computations": sum(1 for log in self.execution_log if "Subset Found" in log),
                "Element Additions": sum(1 for log in self.execution_log if "Added Element" in log),
                "Backtracking Operations": sum(1 for log in self.execution_log if "Backtracked" in log),
            }
        }

        return execution_data

    def get_all_subsets(self):
        """Retrieves all subsets with structured data representation."""
        subset_statistics = {
            "Total Subsets Generated": len(self.results),
            "Largest Subset Length": max((len(subset) for subset in self.results), default=0),
            "Smallest Subset Length": min((len(subset) for subset in self.results), default=0),
            "Computed Subsets": self.results,
        }

        self.execution_log.append(("Retrieved All Subsets", subset_statistics))
        return subset_statistics


# Example Usage:
elements = [1, 2, 3]
subset_generator = SubsetGenerator(elements)

subset_generator.generate()

query_result = subset_generator.query_subset([1, 3])
print("\nQuerying if [1, 3] exists:", query_result)

execution_log = subset_generator.get_execution_log()
print("\nExecution Log:", execution_log)

all_subsets = subset_generator.get_all_subsets()
print("\nSubset Statistics:", all_subsets)


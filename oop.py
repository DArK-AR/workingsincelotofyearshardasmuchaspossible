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

making request

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  // ignore: library_private_types_in_public_api
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String? upiLink;

  Future<void> initiateUPIPayment() async {
    final response = await http.post(
      Uri.parse("http://127.0.0.1:8000/api/create-upi-payment/"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({
        "amount": 1,
        "customer_name": "joe doe",
        "customer_phone": "9999999999",
      }),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      setState(() {
        upiLink = data["upi_link"];
      });
    } else {
      setState(() {
        upiLink = "Error fetching UPI payment link";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(   flutter post method

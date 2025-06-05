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

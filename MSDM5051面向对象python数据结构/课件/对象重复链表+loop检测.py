class Node:
    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node  

class SinglyLinkedList:
    def __init__(self, head=None):
        self.head = head
        
    ###########################################################################
    
    def duplicate(self):
        
        # no duplication if the head is undefined
        if self.head == None:
            return
        
        # iterate until the final node, while copying each node's data to create a new linked list
        curr = self.head
        temp_head = Node(curr.data)
        curr2 = temp_head
        while curr.next:
            curr = curr.next
            curr2.next = Node(curr.data)
            curr2 = curr2.next
        
        # append temp_head back to the tail of the original list
        curr.next = temp_head
        
    ###########################################################################
    # for printing the linked list only
    
    def print_list(self):
        print_node = self.head
        while print_node:
            print(print_node.data)
            print_node = print_node.next 
            

my_SLL = SinglyLinkedList()
n1 = Node(True)
n2 = Node("I love Python")
n3 = Node(5051)
n4 = Node("but I don't want to quiz")

my_SLL.head = n1
n1.next = n2
n2.next = n3
n3.next = n4

my_SLL.duplicate()
my_SLL.print_list()





class Node:
    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node  

class SinglyLinkedList:
    def __init__(self, head=None):
        self.head = head
        
    ###########################################################################
    
    def detect_loop(self):
        
        curr = self.head
        visited = set()
        while curr not in visited:
            
            if curr.next == None:
                return "N"
            
            visited.add(curr)
            curr = curr.next
            
        return "Y"
        
    ###########################################################################
    # for printing the linked list only
    
    def print_list(self):
        print_node = self.head
        while print_node:
            print(print_node.data)
            print_node = print_node.next 
            
my_SLL = SinglyLinkedList()
n1 = Node(True)
n2 = Node("I love Python")
n3 = Node(5051)
n4 = Node("but I don't want to quiz")

# Linking by direct assigning the nodes' property
my_SLL.head = n1
n1.next = n2
n2.next = n3
n3.next = n4
n4.next = n2 # this link forms a loop

# Detect loop
my_SLL.detect_loop()

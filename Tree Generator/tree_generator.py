import binarytree
from sympy import simplify
from random import randint

class data: #In charge of managing the data that is generated
    
    def __init__(self, number_of_data_points, name_of_file): #Initialize the class
        self.number = number_of_data_points #Number of data points we want to generate
        self.name = name_of_file #Name of the CSV file that we want to store the data points in
        
    def create_empty_csv(self): #Initalize the CSV file
        pass
    
    def update_csv(self, data): #Update the CSV file every time we generate a new data point
        pass
    
class generator:
    
    def __init__(self, number_of_variables): #Initialize the class
        self.num_vars = number_of_variables #Number of variables in the function
        self.unary = ["abs", "cos", "sin", "tan", "sec", "csc", "cot", "cosh", "sinh", "tanh", "sech", "csch", "coth", "exp", "sqrt", "ln"] #Unary operands
        self.binary = ["add", "sub", "mult", "div", "pow"] #Binary operands
        self.variables = [] #Variables in the function
        self.digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #digits
        self.constants = ["pi", "e"]
        
    def generate_vars(self): #generate variables
        pass
    
    def generate_tree(self): #genenate a random BST
        height = randint(1, 5)
        tree = binarytree.bst(height = height, is_perfect = False, letters = False)
        return tree
    
    def populate_tree(self, tree): #populate the tree
        for i in list(tree):
            if i.is_leaf():
                i.value = self.generate_constant()
            elif i.is_binary():
                i.value = self.random_binary()
            else:
                i.value = self.random_unary()
        return tree

    def random_unary(self): #pick a random unary operand
        pass
    
    def random_binary(self): #pick a random binary operand
        pass
    
    def iterate_tree(self): #i don't think i'll need this actually
        pass
    
    def generate_leaf_node(self): #generate a variable or constant
        pass
    
    def generate_digits(self): #generate a digit
        pass
    
    def generate_random_variable(sefl): #pick a variable at random
        pass
    
    def generate_random_constant(self): #pick a random digit or known constant i.e. pi
        pass
    
    def is_valid_tree(self): #use SymPy to verify that the tree is a valid mathematical expression
        pass
    
    def tree_to_exp(self): #convert the tree into something that sympy can read
        pass
    
    def expressions_to_include(self): #decide if the expression will include trig functions or log
        pass
    
    def is_leaf(self): #check to see if the node is a leaf
        return self.left == None and self.right == None
    
    def is_binary(self): #check to see if the node has two children
        return self.left != None and self.right != None
            
    #tree.values prints out the tree like how they are expressed in leet code
        
        #unary goes to a binary operation
        
        

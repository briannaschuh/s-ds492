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
        self.constants= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "pi", "e"] #Constants
        
    def generate_vars(self): #generate variables
        pass
    
    def generate_tree(self): #genenate a random BST
        height = randint(1, 9)
        tree = binarytree.bst(height = height, is_perfect = False, letters = True)
        return tree
    
    def populate_tree(self, tree): #populate the tree
        pass
        
    def random_unary(self): #pick a random unary operand
        pass
    
    def random_binary(self): #pick a random binary operand
        pass
    
    
    
        
        #unary goes to a binary operation
        

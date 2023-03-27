import binarytree
import sympy as sp
from random import randint, choices
import numpy as np
from operands import operands_dict
from timeout import timeout

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
        self.unary = ["cos", "sin", "tan", "sec", "csc", "cot", "cosh", "sinh", "tanh", "sech", "csch", "coth", "exp", "sqrt", "ln"] #Unary operands
        self.unary_weights = [12, 12, 10, 8, 8, 8, 2, 2, 2, 2, 2, 2, 10, 10, 10]
        self.binary = ["add", "sub", "mult", "div", "pow"] #Binary operands
        self.binary_weights = [20, 20, 20, 20, 20]
        self.operands_dict = {'add': '+', 'sub': '-', 'mult': '*', 'div': '/', 'pow': '**', 'cos': 'cos', 'sin': 'sin', 'tan': 'tan', 'sec': 'sec', 'csc': 'csc', 'cot': 'cot', 'cosh': 'cosh', 'sinh': 'sinh', 'tanh': 'tanh', 'sech': 'sech', 'csch': 'csch', 'coth': 'coth', 'exp':'exp', 'sqrt': 'sqrt', 'ln': 'ln'}
        self.variables = [] #Variables in the function
        self.alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"] #alphabet
        self.digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #digits
        self.digit_prob = 0.4
        self.power = False
        self.power_count = 0
    
    #capped at 26 variables but I will work on a function towards the end that can generate more than 26 variables
    #This creates a list that has all of the variables for the function
    def generate_vars(self): #generables
        self.variables = self.alphabet[0:int(self.num_vars)]
    
    #generate a generic binary tree. we will later populate this tree so that it represents a random expresssion
    def generate_tree(self, perfect_bool): #genenate a random BST
        height = randint(1, 5)#choose a random height
        tree = binarytree.bst(height = height, is_perfect = perfect_bool, letters = False) 
        return tree
    
    #populate a binary tree so that it represents an expression
    def populate_tree(self, tree): #populate the tree
        for i in list(tree): #go through each node in the tree
            if is_leaf(i): #check to see if it is a leaf
                i.value = self.generate_leaf_node() #if it is a leaf node, generate a random leaf value
                if self.power: #check to see if the parent node is a power
                    if self.power_count == 1:
                        self.power_count = 0
                        self.power = False
                    else:
                        self.power_count = 1
            elif is_binary(i): #if it is not a leaf, check to see if it has two children
                i.value = self.random_binary() #choose a random binary
                if i.value == "pow": #keep track if it is a power
                    self.power = True
            else: #if it is neither, it must have one child
                i.value = self.random_unary() #generate a random unary operand
        return tree #return the tree
    #TODO: rethink how you handle powers? maybe at the end if there's time?

    def random_unary(self): #pick a random unary operand
        return choices(self.unary, weights = self.unary_weights)[0]
    
    def random_binary(self): #pick a random binary operand
        return choices(self.binary, weights = self.binary_weights)[0]
    
    def generate_leaf_node(self): #generate a variable or constant
        if self.power_count == 1:
            weight_constant = 0.8
            weight_var = 0.2
            normalized_weights = [weight_constant, weight_var]
        else:
            weight_constant = 30
            weight_var = 20*len(self.variables)
            normalized_weights = self.normalization(weight_constant, weight_var)
        gen_var = choices([0,1], weights = normalized_weights)
        if gen_var[0]:
            return self.generate_random_variable()
        else:
            return self.generate_random_constant()           
    #TODO: think of a better algo for generating constants
    
    def generate_digits(self): #generate a digit
        num = self.generate_first_digit()
        prob = randint(0,100)
        while prob < 40:
            num += choices(self.digits)
            prob = randint(0,100)
        return num
    #TODO: think of a better algo for generating digit? maybe change the probability?
        
    #generate the first digit of a constant
    def generate_first_digit(self):
        return choices(self.digits[1:])[0]
    
    #pick a variable at random
    def generate_random_variable(self): 
        return choices(self.variables)[0]
                
    #normamlize weights for node generation
    def normalization(self, weight_one, weight_two):
        list_weights = np.array([weight_one, weight_two])
        weight_sum = sum(list_weights)
        return list((1/weight_sum) * list_weights)
    #TODO: think of a better algo?
    
    #randomly decide to generate pi or a random digit
    def generate_random_constant(self): #pick a random digit or known constant i.e. pi
        prob = randint(0, 100)
        if prob < 95:
            return self.generate_random_digit()
        else:
            return("pi")
    
    #generate a constant that is not pi        
    def generate_random_digit(self):
        digit = str(self.generate_first_digit())
        prob = randint(0, 100)
        while prob < 20:
            digit += str(choices(self.digits)[0])
            prob = randint(0, 100)
        return int(digit)
    #TODO: function name is misleading and should be renamed
        
def is_leaf(node): #check to see if the node is a leaf
    return node.left == None and node.right == None

def is_binary(node): #check to see if the node has two children
    return node.left != None and node.right != None
       
def tree_to_exp(tree): #convert the tree into something that sympy can read
    left = tree_to_exp(tree.left) if tree.left != None else None
    right = tree_to_exp(tree.right) if tree.right != None else None
    if left != None and right != None:
        return "(" + left + operands_dict[tree.value] + right + ")"
    elif left == None and right != None:
        return operands_dict[tree.value] + "(" + right + ")"
    elif left != None and right == None:
        return operands_dict[tree.value] + "(" + left + ")"
    else:
        return str(tree.value)

def simplify_expr(expr):    
    @timeout()
    def simplify_wrapped(expr):
        try:
            simplified_expr = sp.simplify(expr)
            return simplified_expr
        except TimeoutError:
            return False
    return simplify_wrapped(expr)

        

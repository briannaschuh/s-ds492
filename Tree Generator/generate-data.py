import sys
from tree_generator import data, generator
import sample_expressions
import sympy

if __name__ == "__main__":
    l = len(sys.argv)
    
    #Error checking
    try: #Check to see if the first argument is valid
        inputtest = sys.argv[1]
        inputtest = int(inputtest)
    except IndexError:
        print("The first argument specifies the  number of variables in the function. ")
        sys.exit(1)
    except ValueError:
        print("The first argument specifies the number of variables in the function. ")
        sys.exit(1)

    try: #The second argument specifes the number of data points to be generated
        inputtest = sys.argv[2]
        inputtest = int(inputtest)
    except IndexError:
        print("The second argument specifies the number of data points to be generated. ")
        sys.exit(1)
    except ValueError:
        print("The second argument specifies the number of data points to be generated. ")
        sys.exit(1)

    try: #The third argument gives the name of the CSV file that will store all of the generated data
    #TODO: rewrite this code to check to make sure that the name entered can be a valid CSV file name
        inputtest = sys.argv[3]
    except IndexError: 
        print("The third argument specifes the name of the CSV file that will store all of the generated data. ")
        sys.exit(1)
    
        
    obj = generator(sys.argv[1])
    tree = obj.generate_tree()
    print(list(tree))
    print(tree)
    
    count = 0
    
    print(tree.values)
    
    for i in list(tree):
        print(i.value)
        print("left", i.left)
        print("right", i.right)
        print("we are now changing the value")
        i.value = 1
        print("new value", i.value)
        
    print(tree)
    
    #you can also use levelorder
    
    #NEXT STEPS: iterate through the binary tree
    #leaves should be constants. internal nodes should be operators
    #design an algo that is able to tell if an internal node should be an internal operator or not
    
    #use sympy to check if the expression is valid
    #then we can start generating data
        
    
    ## Generate expressions

#Use command line arguments to specify the number of internal nodes
#And the number of variables

#n = number of interal nodes

#program in the different sets?


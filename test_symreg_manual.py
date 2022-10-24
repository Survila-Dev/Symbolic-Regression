"""
Showcase the documentation of python code.

The public class, methods and modules should entail documentation, which is
used by the compiler to assist programming.

Classes:
    
    my_class
        Description here.
    
Functions:
    
    my_method
        Description here.
    
Misc variables:
    
    None
        Description here.
    
Author: Eimantas Survila
Date: Tue Dec 22 07:06:19 2020
"""

from symbolicregression import Node, SymbRegIndividual

Node.operator_types = {
    "plus": [lambda x: sum(x), "({0} + {1})", 2],
    "minus": [lambda x: x[0] - x[1], "({0} - {1})", 2],
    "mult": [lambda x: x[0] * x[1], "({0} * {1})", 2],
    "div": [lambda x: x[0]/x[1], "({0} / {1})", 2]}

Node.variables = ["x", "t", "p", "E"]
Node.const_boundaries = [0, 0.1, 5]


first_idvi = SymbRegIndividual()
first_idvi.generate_random_gencode()
print(first_idvi.gencode.write())

second_idvi = SymbRegIndividual()
second_idvi.generate_random_gencode()
print(second_idvi.gencode.write())

cross_over = first_idvi.crossover(second_idvi)
print(cross_over.gencode.write())

print(first_idvi.gencode.write())
print(second_idvi.gencode.write())

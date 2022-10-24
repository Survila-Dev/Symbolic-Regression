"""
Implement SymRegIndividual Class

Classes:
    
    SymRegIndividual
        Subclass of Individual for use in GenAlg.
    Node
        Node for tree structure.
    
Author: Eimantas Survila
Date: Mon Dec 21 15:30:13 2020
"""
import genalg_def as ga
import random
import numpy as np

# %%


class Node():

    operator_types = {}
    variables = []
    const_boundaries = []
    configuration = {
        "probability_for_constant_not_var": 0.2,
        "prob_exchange_node": 0.4,
        "mutate_random_node_depth": 2,
        "max_recursive_depth": 6,
        "initial_depth": 1,
        "prob_change_current_node_with_single_node": 0.2}

    # %%
    def __init__(
            self,
            node_type,
            children=[],
            **kwargs):

        self.type = node_type
        self.children = children

        if node_type == "const":
            self.value = kwargs["value"]

        if node_type == "var":
            self.var_key = kwargs["var"]

    # %%
    def __call__(
            self,
            gencode={},
            **kwargs):

        if self.type == "const":
            return self.value

        elif self.type == "var":
            return kwargs[self.var_key]

        else:
            return Node.operator_types[self.type][0](
                [i(**kwargs) for i in self.children])

    # %%
    def write(self):

        if self.type == "const":
            return str(self.value)
        elif self.type == "var":
            return str(self.var_key)
        else:
            output_str = Node.operator_types[self.type][1].format(
                *[child.write() for child in self.children])

            return output_str

    # %%
    def select_random_child(self):
        return self.children[random.randint(0, len(self.children) - 1)]

    # %%
    def random_tree_traverse(
            self,
            tree_max_depth,
            max_depth_to_find,
            precise_traverse_to_depth=False):

        # if traverse_to_depth == None:

        cur_depth = 0
        if max_depth_to_find == 0:
            return self, None, cur_depth

        cur_node = self
        cur_par_node = None

        if not precise_traverse_to_depth:
            max_depth_to_find = random.randint(1, max_depth_to_find)

        for _ in range(tree_max_depth):

            if cur_depth == max_depth_to_find:
                return cur_node, cur_par_node, cur_depth
            else:
                cur_depth += 1
                cur_par_node = cur_node
                cur_node = cur_node.select_random_child()
                if len(cur_node.children) == 0:
                    return cur_node, cur_par_node, cur_depth

    # %%
    def copy(self):

        # if node_type == "const":
        #     self.value = kwargs["value"]

        # if node_type == "var":
        #     self.var_key = kwargs["var"]

        if self.type == "const":
            copy_node = Node(
                node_type="const",
                value=self.value)

        elif self.type == "var":
            copy_node = Node(
                node_type="var",
                var=self.var_key)

        else:
            copy_node = Node(
                node_type=self.type,
                children=[
                    i.copy() for i in self.children])

        return copy_node

    # %%

    def random_node(
            allowed_depth):

        def create_node(
                cur_depth
        ):

            nonlocal allowed_depth

            if allowed_depth == 0:
                prob_oper = 0
            else:
                prob_oper = 1 * float(allowed_depth -
                                      cur_depth) / float(allowed_depth)
            prob_const = Node.configuration["probability_for_constant_not_var"] * (
                1 - prob_oper)
            prob_var = 1 - prob_oper - prob_const

            assert (np.round(prob_const + prob_var + prob_oper, 3)
                    == 1), "Probabilites must add up to 1."

            rnd_no = random.random()
            if rnd_no < prob_const:

                bound = Node.const_boundaries
                if len(bound) == 2:
                    val = bound[0] + (bound[-1] - bound[0]) * random.random()
                if len(bound) == 3:
                    val = bound[0] + int(((bound[-1] - bound[0])
                                         * random.random()) // bound[1]) * bound[1]
                    val = np.round(val, 3)

                out = Node(
                    node_type="const",
                    value=val)

            if rnd_no > prob_const and rnd_no < (prob_const + prob_var):

                out = Node(
                    node_type="var",
                    var=Node.variables[random.randint(0, len(Node.variables) - 1)])

            # Recursive call
            if rnd_no > (prob_const + prob_var):

                oper_types = list(Node.operator_types.keys())
                oper_type = oper_types[random.randint(0, len(oper_types) - 1)]

                out = Node(
                    node_type=oper_type,
                    children=[
                        create_node(cur_depth + 1) for _ in range(
                            Node.operator_types[oper_type][2])])

            return out

        return create_node(cur_depth=0)

# %%


class SymbRegIndividual(ga.Individual):

    # %%
    def __call__(self, **kwargs):
        return self.gencode(**kwargs)

    def func_out(self):
        return self

    def fitness(self):
        out = self.func_fit(self)
        return self.func_fit(self)

    def generate_random_gencode(self):
        INITIAL_DEPTH = Node.configuration["initial_depth"]

        self.gencode = Node.random_node(INITIAL_DEPTH)
        self.tree_depth = INITIAL_DEPTH

    def write(self):
        return self.gencode.write()

    def signature(self):
        return self.gencode.write()

    # %%
    def mutate(self):

        new_gencode = self.gencode.copy()

        old_child, parent, child_depth = new_gencode.random_tree_traverse(
            self.tree_depth,
            self.tree_depth)

        tree_growth = 0

        # Change random node with another node, which has the same argument no.
        if random.random() < Node.configuration["prob_exchange_node"]:

            if old_child.type == "var" or old_child.type == "const":
                new_child = Node.random_node(0)
                parent.children[parent.children.index(old_child)] = new_child
                # break

            else:
                no_of_args = Node.operator_types[old_child.type][2]

                oper_types_edit = {
                    key: val for key, val in Node.operator_types.items() if val[2] == no_of_args}
                key_list = list(oper_types_edit.keys())
                key_list.remove(old_child.type)

                if len(key_list) != 0:
                    if len(key_list) == 1:
                        rnd_type = key_list[0]
                    else:
                        rnd_type = key_list[random.randint(
                            0, len(key_list) - 1)]
                else:
                    rnd_type = old_child.type

                new_child = Node(
                    node_type=rnd_type,
                    children=old_child.children)
                parent.children[parent.children.index(old_child)] = new_child

        # Change random node with whole new node.
        else:

            if random.random() < Node.configuration["prob_change_current_node_with_single_node"]:
                new_child = Node.random_node(0)
            else:
                max_mutant_depth = min(
                    Node.configuration["mutate_random_node_depth"],
                    Node.configuration["max_recursive_depth"] - child_depth)
                new_child = Node.random_node(max_mutant_depth)
                tree_growth = max_mutant_depth

            parent.children[parent.children.index(old_child)] = new_child

        mutant = SymbRegIndividual(
            func_fit=self.func_fit)

        mutant.gencode = new_gencode
        mutant.tree_depth = max(self.tree_depth, child_depth + tree_growth)
        return mutant

    # %%
    def crossover(self, other):

        new_gencode = self.gencode.copy()

        depth_of_exchange = random.randint(
            1, Node.configuration["max_recursive_depth"])

        child_self, parent_self, _ = new_gencode.random_tree_traverse(
            max(self.tree_depth, other.tree_depth),
            depth_of_exchange,
            precise_traverse_to_depth=True)

        child_other, _, _ = new_gencode.random_tree_traverse(
            max(self.tree_depth, other.tree_depth),
            depth_of_exchange,
            precise_traverse_to_depth=True)

        parent_self.children[parent_self.children.index(
            child_self)] = child_other

        mutant = SymbRegIndividual(
            func_fit=self.func_fit)
        mutant.gencode = new_gencode
        mutant.tree_depth = self.tree_depth

        return mutant

    # %%
    def elitechild(self):

        child = SymbRegIndividual(
            func_fit=self.func_fit)
        child.gencode = self.gencode
        child.tree_depth = self.tree_depth

        return child

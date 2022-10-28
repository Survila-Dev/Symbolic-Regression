"""
Testing the genetical algorithm modul.

Author: Eimantas Survila
Date: Sat Dec 19 08:30:42 2020
"""

import unittest
import genalg_def as genalg
import symbolicregression as symreg

import numpy as np


class GenAlgTest(unittest.TestCase):

    # %%
    def test_circle_func(self):
        """
        Test if float variable optimization for circle functions works.
        """

        def func_to_optimize_1(a, b):
            return (a**2 + b**2)**0.5

        vartype = {
            "x": float,
            "y": float}

        varbounds = {
            "x": [-50, 50],
            "y": [-50, 50]}

        def func_in(gencode):
            return gencode["x"], gencode["y"]

        def func_fit(func_output):
            return abs(func_output)

        optimizer = genalg.Evolver(
            var_type=vartype,
            var_bounds=varbounds,
            f=func_to_optimize_1,
            f_in=func_in,
            f_fit=func_fit)

        optimizer.configuration["population_size"] = 100
        optimizer.configuration["maximal_iterations"] = 20

        optimizer.run()

        [a_best, b_best] = optimizer.bestvars()
        best_f_out = optimizer.bestoutput()

        self.assertAlmostEqual(a_best, 0, 0)
        self.assertAlmostEqual(b_best, 0, 0)
        self.assertAlmostEqual(best_f_out, 0, 0)

    # %%
    def test_discrete_vals(self):
        """
        Test if optimization with discrete variabels from a list works.
        """

        def func_to_optimize_2(a, b):

            if a == "Vorname" and b == "Nachname":
                return 0.1
            else:
                return 0.5

        vartype = {
            "first_name": list,
            "last_name": list}

        varbounds = {
            "first_name": ["Name", "Titel", "Nachname", "Vorname"],
            "last_name": ["Vorname", "Doktor", "Professor", "Nachname"]}

        def func_in(gencode):
            return gencode["first_name"], gencode["last_name"]

        def func_fit(func_output):
            return func_output

        optimizer = genalg.Evolver(
            var_type=vartype,
            var_bounds=varbounds,
            f=func_to_optimize_2,
            f_in=func_in,
            f_fit=func_fit)

        optimizer.configuration["population_size"] = 50
        optimizer.configuration["maximal_iterations"] = 10

        optimizer.run()

        [first_best, last_best] = optimizer.bestvars()
        best_f_out = optimizer.bestoutput()

        self.assertEqual(first_best, "Vorname")
        self.assertEqual(last_best, "Nachname")
        self.assertEqual(best_f_out, 0.1)

    # %%
    def test_mixed_variables(self):
        """
        Test if optimization with discrete values works.
        """

        def func_to_optimize_3(no_bricks, brick_type, wall_height):

            if brick_type == "brick1":
                brick_len = 10
            if brick_type == "brick2":
                brick_len = 12
            if brick_type == "brick3":
                brick_len = 8

            return 0, 0, brick_len * float(no_bricks) * wall_height

        vartype = {
            "no_bricks": int,
            "brick_type": list,
            "wall_height": float}

        varbounds = {
            "no_bricks": [1, 21],  # The last value is not inclusive.
            "brick_type": ["brick1", "brick2", "brick3"],
            "wall_height": [1, 0.2, 10]}

        def func_in(gencode):
            return gencode["no_bricks"], gencode["brick_type"], gencode["wall_height"]

        def func_fit(func_output):
            return 1/func_output[2]

        optimizer = genalg.Evolver(
            var_type=vartype,
            var_bounds=varbounds,
            f=func_to_optimize_3,
            f_in=func_in,
            f_fit=func_fit)

        optimizer.configuration["population_size"] = 150
        optimizer.configuration["maximal_iterations"] = 80

        optimizer.run()

        [best_int, best_brick, best_wall_h] = optimizer.bestvars()
        [out1, out2, out3] = optimizer.bestoutput()

        self.assertEqual(best_int, 20)
        self.assertAlmostEqual(best_wall_h, 10, 1)
        self.assertEqual(best_brick, "brick2")
        self.assertEqual(out1, 0)
        self.assertEqual(out2, 0)
        self.assertAlmostEqual(out3, 2400, 0)

    # %%
    def test_bifurcating_arguments(self):
        """
        Test if bifurcating arguments are handled right.
        """

        def func_to_optimize_4(subfunc_type, *args):

            def square(*args):
                if len(args) != 2:
                    raise "Too many variables"
                return args[0] * args[1]

            def cube(*args):
                return args[0] * args[1] * args[2]

            if subfunc_type == "Square":

                return square(*args) * 20

            if subfunc_type == "Cube":

                return cube(*args)

        vartype = {
            "Subfunc_type": list,
            "length": float,
            "width": float,
            "height": float}

        varbounds = {
            "Subfunc_type": ["Square", "Cube"],
            "length": [1, 20],
            "width": [1, 19],
            "height": [1, 21]}

        def func_in(gencode):

            if gencode["Subfunc_type"] == "Square":

                return \
                    gencode["Subfunc_type"], \
                    gencode["length"], \
                    gencode["width"]

            if gencode["Subfunc_type"] == "Cube":

                return \
                    gencode["Subfunc_type"], \
                    gencode["length"], \
                    gencode["width"], \
                    gencode["height"]

        def func_fit(func_output):
            return 1/func_output

        optimizer = genalg.Evolver(
            var_type=vartype,
            var_bounds=varbounds,
            f=func_to_optimize_4,
            f_in=func_in,
            f_fit=func_fit)

        optimizer.configuration["population_size"] = 200
        optimizer.configuration["maximal_iterations"] = 50

        optimizer.run()

        best_vars = optimizer.bestvars()
        if len(best_vars) == 3:
            out_form, out_l, out_w = best_vars
        if len(best_vars) == 4:
            out_form, out_l, out_w, out_h = best_vars

        best_f_out = optimizer.bestoutput()
        if out_form == "Cube":
            self.assertAlmostEqual(best_f_out, 7980, 0)
            self.assertEqual(out_form, "Cube")
            self.assertAlmostEqual(out_l, 20, 0)
            self.assertAlmostEqual(out_w, 19, 0)
            self.assertAlmostEqual(out_h, 21, 0)
        if out_form == "Square":
            self.assertAlmostEqual(best_f_out, 7600, 0)
            self.assertEqual(out_form, "Square")
            self.assertAlmostEqual(out_l, 20, 0)
            self.assertAlmostEqual(out_w, 19, 0)

    # %%
    def test_int_boundaries(self):
        """
        Test if integers boundaries and step size is handled correctly.
        """

        def func_to_optimize_5(h, v):
            return abs((h-5)*v)

        vartype = {
            "horz": int,
            "vert": int}

        varbounds = {
            "horz": [2, 5],
            "vert": [6, 12]}

        def func_in(gencode):
            return gencode["horz"], gencode["vert"]

        def func_fit(func_output):
            return func_output

        optimizer = genalg.Evolver(
            var_type=vartype,
            var_bounds=varbounds,
            f=func_to_optimize_5,
            f_in=func_in,
            f_fit=func_fit)

        optimizer.configuration["population_size"] = 100
        optimizer.configuration["maximal_iterations"] = 50

        optimizer.run()

        [h_best, v_best] = optimizer.bestvars()
        best_f_out = optimizer.bestoutput()

        # print("h best is {} and v_best is {}".format(h_best, v_best))
        self.assertTrue(h_best == 4 or h_best == 6)
        self.assertAlmostEqual(v_best, 6)
        self.assertAlmostEqual(best_f_out, 6, 0)

    # %%
    def test_float_step_sizes(self):
        """
        Test if float variable step sizes are handled correctly.
        """

        def func_float_step_size(x, y):
            return abs((x - 0.75)*(y - 0.75))

        vartype = {
            "x": float,
            "y": float}

        varbounds = {
            "x": [-60, 0.5, 60],
            "y": [-70, 1/3, 70]}

        def func_in(gencode):
            return gencode["x"], gencode["y"]

        def func_fit(func_output):
            return func_output

        optimizer = genalg.Evolver(
            var_type=vartype,
            var_bounds=varbounds,
            f=func_float_step_size,
            f_in=func_in,
            f_fit=func_fit)

        optimizer.configuration["population_size"] = 100
        optimizer.configuration["maximal_iterations"] = 50
        optimizer.configuration["log_results"] = False

        optimizer.run()

        [x_best, y_best] = optimizer.bestvars()
        best_f_out = optimizer.bestoutput()

        self.assertTrue(x_best == 0.5 or x_best == 1)
        self.assertAlmostEqual(y_best, 2/3, 1)
        self.assertAlmostEqual(best_f_out, 0.0208, 2)

# %%


class TestSymbolicRegression(unittest.TestCase):

    # %%
    def test_tree_class_call_binary_and_unary(self):

        symreg.Node.operator_types = {
            "plus": [lambda x: sum(x), "{0} + {1}", 2],
            "minus": [lambda x: x[0] - x[1], "{0} - {1}", 2],
            "exp": [lambda x: np.e**x[0], "e^{0}", 1]}

        n_zero = symreg.Node(
            node_type="const",
            value=0)
        self.assertEqual(n_zero(), 0)

        n_exp = symreg.Node(
            node_type="exp",
            children=[n_zero])

        self.assertEqual(n_exp(), 1)

        n_minus = symreg.Node(
            node_type="minus",
            children=[
                symreg.Node(
                    node_type="const",
                    value=6),
                symreg.Node(
                    node_type="const",
                    value=2)
            ])
        self.assertEqual(n_minus(), 4)

        n_root = symreg.Node(
            node_type="plus",
            children=[n_minus, n_exp])

        self.assertEqual(n_root(), 5)
        self.assertEqual(n_root.write(), "6 - 2 + e^0")

    # %%
    def test_tree_class_call_multiply_arguments(self):

        def func_condition(x):
            if x[0] > 5:
                return 2.5
            else:
                return 1

        symreg.Node.operator_types = {
            "mean": [lambda x: sum(x)/len(x), "mean({0}, {1}, {2}, {3})", 4],
            "condition": [func_condition, "cond({0})", 1],
            "max": [lambda x: max(x), "max({0}, {1}, {2})", 3],
            "sum": [lambda x: sum(x), "sum({0}, {1}, {2})", 3]}

        n_mean = symreg.Node(
            node_type="mean",
            children=[
                symreg.Node(
                    node_type="const",
                    value=1),
                symreg.Node(
                    node_type="const",
                    value=2),
                symreg.Node(
                    node_type="const",
                    value=3),
                symreg.Node(
                    node_type="const",
                    value=4)
            ])
        self.assertEqual(n_mean(), 2.5)

        n_condition = symreg.Node(
            node_type="condition",
            children=[
                symreg.Node(
                    node_type="const",
                    value=6)
            ])
        self.assertEqual(n_condition(), 2.5)

        n_max = symreg.Node(
            node_type="max",
            children=[
                symreg.Node(
                    node_type="const",
                    value=2),
                symreg.Node(
                    node_type="const",
                    value=6),
                symreg.Node(
                    node_type="const",
                    value=12)
            ])
        self.assertEqual(n_max(), 12)

        n_root = symreg.Node(
            node_type="sum",
            children=[n_mean, n_condition, n_max])

        self.assertEqual(n_root(), 17)
        self.assertEqual(
            n_root.write(), "sum(mean(1, 2, 3, 4), cond(6), max(2, 6, 12))")

# %%


class TestSymbolicRegression(unittest.TestCase):

    # %%
    def test_symbolic_regress_depth_3(self):

        def func_to_find(x):
            return (1/x) * abs(x)

        def exp_with_overflow_control(x):
            OVERFLOW_LIM = 10
            if x[0] > OVERFLOW_LIM:
                res = np.e**OVERFLOW_LIM
            else:
                res = np.e**x[0]
            return res

        def divide_with_zero_control(x):
            ARBITRARY_HIGH_NO = 10**9
            if x[1] == 0:
                return ARBITRARY_HIGH_NO
            else:
                return x[0] / x[1]

        symreg.Node.operator_types = {
            "plus": [lambda x: sum(x), "({0} + {1})", 2],
            "div": [divide_with_zero_control, "({0} / {1})", 2],
            "abs": [lambda x: abs(x[0]), "abs({0})", 1]}

        symreg.Node.variables = ["x"]
        symreg.Node.const_boundaries = [-2, 0.5, 2]

        def func_random_name_for_testing(individual_node):
            res = 0
            for x in [-5, -3, 2, 5, 6, 9, 5, 6]:
                res += abs(individual_node(x=x) - func_to_find(x))
            return res

        optimizer = genalg.Evolver(
            individual_class=symreg.SymbRegIndividual,
            f_fit=func_random_name_for_testing)

        symreg.Node.configuration["max_recursive_depth"] = 4
        optimizer.configuration["population_size"] = 30
        optimizer.configuration["maximal_iterations"] = 10

        optimizer.run()

        best_solution = optimizer.bestoutput()

        self.assertEqual(best_solution(x=1), func_to_find(1))

    def test_symbolic_regress_depth_3_1(self):

        def power_with_overflow_control(x):
            OVERFLOW_LIM = 10
            if x[1] > OVERFLOW_LIM:
                res = x[0]**OVERFLOW_LIM
            else:
                res = x[0]**x[1]
            return res

        symreg.Node.operator_types = {
            "plus": [lambda x: sum(x), "({0} + {1})", 2],
            "abs": [lambda x: abs(x[0]), "abs({0})", 1],
            "mult": [lambda x: x[0]*x[1], "({0}*{1})", 2]}

        symreg.Node.variables = ["x", "t"]
        symreg.Node.const_boundaries = [-2, 0.5, 5]

        def func_random_name_for_testing(individual_node):
            values = [
                [1, 1, 3],
                [-1, -1, 1],
                [2, 2, 10],
                [2, -2, 10],
                [-2, 2, 6]]
            res = 0
            for x, t, f in values:
                res += abs(individual_node(x=x, t=t) - f)
            return res

        optimizer = genalg.Evolver(
            individual_class=symreg.SymbRegIndividual,
            f_fit=func_random_name_for_testing)

        symreg.Node.configuration["max_recursive_depth"] = 4
        optimizer.configuration["population_size"] = 100
        optimizer.configuration["maximal_iterations"] = 50

        optimizer.run()

        best_solution = optimizer.bestoutput()

        values = [
            [1, 1, 3],
            [-1, -1, 1],
            [2, 2, 10],
            [2, -2, 10],
            [-2, 2, 6]]

        for x, t, f in values:
            self.assertEqual(best_solution(x=x, t=t), f)


# %%
if __name__ == "__main__":
    unittest.main()

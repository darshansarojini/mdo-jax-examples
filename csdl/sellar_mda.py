import numpy as np

import csdl
import python_csdl_backend as pcb
from csdl import NewtonSolver, ScipyKrylov, NonlinearBlockGS

class Discipline1(csdl.Model):
    def define(self):
        z = self.declare_variable('z', shape=(2, ), val=np.array([5., 2.]))
        x = self.declare_variable('x', shape=(1,), val=1.)
        y2 = self.declare_variable('y2_state', shape=(1,))

        y1 = z[0] ** 2 + z[1] + x - 0.2 * y2
        self.register_output(name='y1', var=y1)
        return


class Discipline2(csdl.Model):
    def define(self):
        z = self.declare_variable('z', shape=(2,), val=np.array([5., 2.]))
        x = self.declare_variable('x', shape=(1,), val=1.)
        y1 = self.declare_variable('y1_state', shape=(1,))

        y2 = y1**.5 + z[0] + z[1]
        self.register_output(name='y2', var=y2)
        return


class SellarMDA(csdl.Model):
    def define(self):

        self.add(submodel=Discipline1(), name='Discipline1', promotes=['x', 'z', 'y1', 'y2_state'])
        self.add(submodel=Discipline2(), name='Discipline2', promotes=['x', 'z', 'y2', 'y1_state'])

        y1_state  = self.declare_variable('y1_state')
        y1  = self.declare_variable('y1')
        self.register_output('y1_res', y1_state - y1)

        y2_state  = self.declare_variable('y2_state')
        y2  = self.declare_variable('y2')
        self.register_output('y2_res', y2_state - y2)

        return

class SolveSellarMDA(csdl.Model):
    def define(self):
        solve_sellar_mda = self.create_implicit_operation(model=SellarMDA())
        solve_sellar_mda.declare_state(state='y2_state', residual='y2_res', val = 1)
        solve_sellar_mda.declare_state(state='y1_state', residual='y1_res', val = 1)
        solve_sellar_mda.nonlinear_solver = NonlinearBlockGS(maxiter=100)
        # solve_sellar_mda.nonlinear_solver = NewtonSolver()
        # solve_sellar_mda.linear_solver = ScipyKrylov()

        y1,y2 = solve_sellar_mda()
        return


if __name__ == "__main__":
    sim = pcb.Simulator(SolveSellarMDA())
    sim.run()
    print(sim['y1_state'])
    print(sim['y2_state'])

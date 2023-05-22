from dataclasses import dataclass
from time import perf_counter

import plotly.graph_objects as go
from plotly.offline import init_notebook_mode

from pyscheduling.core.listeners import BaseListener


@dataclass
class PlotListener(BaseListener):

    def on_start(self, solve_result, start_time):
        super().on_start(solve_result, start_time)
        self.time_axis = []
        self.y_axis = []
        
        init_notebook_mode(connected = True)
        self.fig = go.Figure()
        
        # Edit the layout
        self.fig.update_layout(title='Objective value VS Time (s)',
                        xaxis_title='Time (s)',
                        yaxis_title='Objective value')
        
    def on_complete(self, end_time):
        super().on_complete(end_time)
        
        self.fig.add_trace(go.Scatter(x=self.time_axis, y=self.y_axis, name='Objective value',
                         line=dict(color='firebrick', width=4)))

        self.fig.show()

    def on_solution_found(self, new_solution, time_found):
        self._nb_sol += 1
        # Check if the new solution is the best one
        found_new_best = self.check_best_sol(new_solution)
        
        self.time_axis.append(time_found)
        self.y_axis.append(new_solution.objective_value)


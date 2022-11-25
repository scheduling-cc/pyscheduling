from pyscheduling.PMSP import RmSijkCmax as rm_sijk_cmax
from pyscheduling.PMSP import RmridiSijkWiTi as rm_ri_di_sijk_witi
import pickle
file = open('instance', 'rb')
instance = pickle.load(file)
#instance = rm_ri_di_sijk_witi.RmridiSijkWiTi_Instance.generate_random(50, 2)

results = rm_ri_di_sijk_witi.Heuristics.list_heuristic(instance,1)
print(results)
print(results.objective)
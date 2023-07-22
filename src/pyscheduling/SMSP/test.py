import pyscheduling.SMSP.risijwiTi as sm
instance = sm.risijwiTi_Instance.generate_random(50,Wmax=10,Pmin=1, Pmax=50,alpha=100)
instance.to_txt("deleteMe.txt")
solution = sm.Heuristics.ACTS_WSECi(instance)
print(solution)
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:06:52 2020
Graph file in stochastic shortest path
This file currently reads a network file and
creates random probabilities and costs of links
@author: Pramesh Kumar
"""

import numpy as np
import itertools
import time
from gurobipy import *
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve



class Node:
    def __init__(self, _tmpIn):
        self.nodeId = _tmpIn
        self.outLinks = []
        self.inLinks = []
        self.label = 0
        self.pred = ""
        self.name = ""
            
class Link:
    def __init__(self, _tmpIn):
        self.fromNode = _tmpIn[0]
        self.toNode = _tmpIn[1]
        self.cap = float(_tmpIn[2]) 
        self.dist = float(_tmpIn[3])
        self.time = float(_tmpIn[4])
        self.alpha = float(_tmpIn[5])
        self.beta = float(_tmpIn[6])
        
        
class State:
    def __init__(self, _tmpIn):
        self.nodeId = _tmpIn[0]
        self.theta =  _tmpIn[1]
        self.controls = _tmpIn[2]
        self.cost = _tmpIn[3]
        self.probs = _tmpIn[4]
        self.J = 0.0
        self.tempJ = 0.0
        self.policy = ""

        
####################################################################################################################################################        
def readData():
    inFile = open("network.dat")
    tmpIn = inFile.readline().strip().split("\t")
    for x in inFile:
        tmpIn = x.strip().split("\t")
        if tmpIn[0] not in nodeSet:
            nodeSet[tmpIn[0]] = Node(tmpIn[0])
        if tmpIn[1] not in nodeSet:
            nodeSet[tmpIn[1]] = Node(tmpIn[1])
        if tmpIn[1] not in nodeSet[tmpIn[0]].outLinks:
            nodeSet[tmpIn[0]].outLinks.append(tmpIn[1])
        if tmpIn[0] not in nodeSet[tmpIn[1]].inLinks:
            nodeSet[tmpIn[1]].inLinks.append(tmpIn[0])
        linkSet[tmpIn[0], tmpIn[1]] = Link(tmpIn)  
    inFile.close()
    
    
def generateProbs(no):
    randN = np.random.uniform(low=0, high=1, size=(no,))
    return randN/sum(randN)
    
        
def createStates():
    idx = 0
    for i in nodeSet:
        if i != dest:            
            outCosts = {(i, j): [round(linkSet[i, j].time*alpha, 2) for alpha in fractions] for j in nodeSet[i].outLinks}
            outProbs = {(i, j): generateProbs(len(fractions)) for j in nodeSet[i].outLinks}
            costCrossProd = list(itertools.product(*outCosts.values()))
            probCrossProd = list(itertools.product(*outProbs.values()))
            for c in range(len(probCrossProd)):
                costDict = dict(zip(list(outCosts.keys()), costCrossProd[c]))
                probDict = dict(zip(list(outCosts.keys()), probCrossProd[c]))
                stateSet[idx] = State([i, np.prod(probCrossProd[c]), nodeSet[i].outLinks, costDict, probDict])
                idx += 1
        else:
            stateSet[idx] = State([dest, 1.0, [dest], {(dest, dest):0.0}, {(dest, dest):1.0}])
            idx += 1
            
###################################################################################################################################################
def BellmanOperator(s, alg):
    costs = {}; i = stateSet[s].nodeId
    for j in stateSet[s].controls:
        jStates = [k for k in stateSet if stateSet[k].nodeId == j]
        costs[j] = stateSet[s].cost[i, j] + sum([stateSet[k].theta * stateSet[k].J for k in jStates])     
    u = min(costs, key=costs.get)
    if alg == 'GSVI':
        stateSet[s].J = costs[u]
    if alg == 'VI':
        stateSet[s].tempJ = costs[u]
    stateSet[s].policy = u

def BellmanOperatorMu(s):
    costs = {}; i = stateSet[s].nodeId
    j = stateSet[s].policy
    jStates = [k for k in stateSet if stateSet[k].nodeId == j]
    stateSet[s].J = stateSet[s].cost[i, j] + sum([stateSet[k].theta * stateSet[k].J for k in jStates])     

        
    

def GaussSeidelVI(dest, maxIter, eps=0.1):
    # Initialize
    tol = float("inf"); it = 0; oldJ = float("inf")
    for i in stateSet:
        if stateSet[i].nodeId == dest:
            stateSet[i].J = 0.0
        else:
            stateSet[i].J = float("inf");
        stateSet[i].policy = ""; 
    
    while (tol > eps) and (it < maxIter):
        it += 1
        for s in stateSet:
            BellmanOperator(s,'GSVI')
            
        if sum([stateSet[k].J for k in stateSet]) != float("inf"):
            newJ= round(sum([stateSet[k].J for k in stateSet]))
            tol = newJ - oldJ
            oldJ = newJ
    return(newJ,it)

def VI(dest, maxIter, eps=0.1):
    # Initialize
    tol = float("inf"); it = 0; oldJ = float("inf")
    for i in stateSet:
        if stateSet[i].nodeId == dest:
            stateSet[i].J = 0.0
        else:
            stateSet[i].J = float("inf");
        stateSet[i].policy = ""; 
    
    while (tol > eps) and (it < maxIter):
        it += 1
        for s in stateSet:
            BellmanOperator(s,'VI')
        for s in stateSet:
            stateSet[s].J = stateSet[s].tempJ
            
        if sum([stateSet[k].J for k in stateSet]) != float("inf"):
            newJ= round(sum([stateSet[k].J for k in stateSet]))
            tol = newJ - oldJ
            oldJ = newJ
    return(newJ,it)


def evaluatePolicy(alg, eps):
    if alg == 'PI':        
        S = len(stateSet)
        P = np.identity(S)
        g = np.zeros(S)
        for s1 in stateSet:
            i = stateSet[s1].nodeId
            jStates = [k for k in stateSet if stateSet[k].nodeId == stateSet[s1].policy]
            for s2 in jStates:
                P[s1, s2] -= stateSet[s2].theta
            j = stateSet[s1].policy
            g[s1] = stateSet[s1].cost[i, j]
        P = np.delete(P, destState, 0); P = csc_matrix(np.delete(P, destState, 1), dtype=float)
        g = np.transpose(csc_matrix(np.delete(g, destState, 0), dtype=float))
        J = spsolve(P, g)
    if alg == 'OPI':
        it = 0
        tol = float("inf"); oldJ = float("inf")
        for i in stateSet:
            if stateSet[i].nodeId == dest:
                stateSet[i].J = 0.0
            else:
                stateSet[i].J = float("inf");
        
        while (tol > eps):
            it += 1
            for s in stateSet:
                BellmanOperatorMu(s)
            for s in stateSet:
                stateSet[s].J = stateSet[s].tempJ
                
            if sum([stateSet[k].J for k in stateSet]) != float("inf"):
                newJ= round(sum([stateSet[k].J for k in stateSet]))
                tol = newJ - oldJ
                oldJ = newJ
        

    
    
def improvePolicy():
    changed = 0
    for s in stateSet:        
        costs = {}; i = stateSet[s].nodeId
        for j in stateSet[s].controls:
            jStates = [k for k in stateSet if stateSet[k].nodeId == j]
            costs[j] = stateSet[s].cost[i, j] + sum([stateSet[k].theta * stateSet[k].J for k in jStates])     
        u = min(costs, key=costs.get)
        if stateSet[s].policy != u:
            stateSet[s].policy = u
            changed += 1
    return changed
        

            
    

def PI(maxIter, eps = 0.1):
    # Creating intial policy
    for s in stateSet:
        stateSet[s].policy = stateSet[s].controls[0]        
    changed = float("inf"); it = 0
    while changed > 0 and it < maxIter:
        it += 1
        evaluatePolicy('PI', eps)
        changed = improvePolicy()
    return(round(sum([stateSet[k].J for k in stateSet])), it)


def optimisticPI(maxIter, eps = 0.1):
    # Creating intial policy
    for s in stateSet:
        stateSet[s].policy = stateSet[s].controls[0]        
    changed = float("inf"); it = 0
    while changed > 0 and it < maxIter:
        it += 1
        evaluatePolicy('OPI', eps)
        changed = improvePolicy()
    return(round(sum([stateSet[k].J for k in stateSet])), it)
            
def LP():
    m = Model()
    J = {};
    for s in stateSet: J[s] = m.addVar(vtype = GRB.CONTINUOUS, name = str(s), lb = 0.0)
    for s in stateSet:
        if s != destState:            
            costs = {}; i = stateSet[s].nodeId
            for j in stateSet[s].controls:
                jStates = [k for k in stateSet if stateSet[k].nodeId == j]            
                m.addConstr(J[s] <= stateSet[s].cost[i, j] + sum([stateSet[k].theta * J[k] for k in jStates]))
                
    m.addConstr(J[destState] == 0)
    obj = 0; 
    for d in J: obj += J[d]
    m.setObjective(obj, sense=GRB.MAXIMIZE)
    m.update()
    m.Params.OutputFlag = 0
    m.Params.DualReductions = 0
    m.optimize()
    if m.status == 3:
        m.computeIIS()
        m.write("Infeasible_model.ilp")
    else:
        for s in stateSet:
            BellmanOperator(s, 'GSVI')
    return sum([J[d].x for d in J])
    
    
def runTestsPrintResults():
    algorithms = ['VI', 'GaussSeidel VI', 'PI', 'Optimistic PI', 'LP']; totJ = []; solTimes = []; iters = []
    solveTime = time.time()
    j, its = VI(dest, maxIter=24)
    totJ.append(j); iters.append(its); solTimes.append(round(time.time() - solveTime))
    solveTime = time.time()
    j, its = GaussSeidelVI(dest, maxIter=24);
    totJ.append(j); iters.append(its); solTimes.append(round(time.time() - solveTime))
    solveTime = time.time()
    j, its = PI(maxIter=24);
    totJ.append(j); iters.append(its); solTimes.append(round(time.time() - solveTime))
    solveTime = time.time()
    j, its = optimisticPI(maxIter=24);
    totJ.append(j); iters.append(its); solTimes.append(round(time.time() - solveTime))
    solveTime = time.time()
    totJ.append(round(LP())); iters.append('-'); solTimes.append(round(time.time() - solveTime))
    row_format ="{:>15}" * (4)
    print(row_format.format('Method', *['Total J', 'Iterations', 'CompTime(s)']))
    for a, b, c, d in zip(algorithms, totJ, iters, solTimes):
        print(row_format.format(a, *[b, c, d]))
    print("---------------------------------------------------------------------")
          
        
            
        
###################################################################################################################################################
print("---------------------------------------------------------------------")
readTime = time.time()
nodeSet ={}; linkSet = {}; stateSet = {}; dest ='24'; fractions = fractions = [0.7, 1.0, 1.3] #[0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7] #  
readData(); createStates(); destState = [s for s in stateSet if stateSet[s].nodeId == dest and stateSet[s].controls[0] == dest][0]
print("There are %d nodes, %d links, and %d node states"%(len(nodeSet), len(linkSet), len(stateSet)))
print("It took %d sec to read the data and create states"%(time.time() - readTime))
print("---------------------------------------------------------------------")
runTestsPrintResults()

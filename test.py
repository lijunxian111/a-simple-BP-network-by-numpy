# -*- coding: utf-8 -*-
import math
import numpy as np
from sympy import *

class BPNet:
    def __init__(self,input_size,hidden_size,out_size,yita=0.0001):
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.out_size=out_size
        self.yita=yita

        self.in_weight=np.random.normal(0.0,1.0,size=(self.input_size,self.hidden_size)).astype(np.float32)
        self.out_weight=np.random.normal(0.0,1.0,size=(self.hidden_size,self.out_size)).astype(np.float32)
        self.relu=self.active
        self.params=list()


    def active(self,x):
        return 1./(1.+np.exp(-x))

    def forward(self,x):
        self.params.clear()
        self.params.append(x)
        x=x @ self.in_weight
        x=self.relu(x)
        self.params.append(x)
        x=x @ self.out_weight
        self.params.append(x)
        return x

    def backward(self,loss):  #loss=yi-oi
        div_out=np.multiply(np.multiply(self.params[2],1.-self.params[2]),loss)
        delta_W_out=self.yita*(self.params[1].T @ div_out)
        delta_hidden=div_out @ self.out_weight.T
        div_hidden=np.multiply(np.multiply(self.params[1],1.-self.params[1]),delta_hidden)
        delta_W_in=self.yita*(self.params[0].T @ div_hidden)
        self.out_weight = self.out_weight + delta_W_out
        self.in_weight=self.in_weight+delta_W_in

if __name__=="__main__":
    My_Net=BPNet(8,32,1)
    #print(My_Net.in_weight)
    #print(My_Net.out_weight)
    print("----------------------------------------------------------")
    data=np.random.normal(0.0,1.0,size=(30,8))

    y=np.random.normal(0.0,1.0,size=(30,1))
    for i in range(50):
        out=My_Net.forward(data)
        loss=abs(y-out)
        print(loss.mean())
        My_Net.backward(loss)

    #print(My_Net.in_weight)
    #print(My_Net.out_weight)


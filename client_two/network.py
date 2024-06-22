import torch
from torch import nn


class MADE(nn.Module):
    def __init__(self, input_feat, num_units, num_layer, ordering):
        super(MADE,self).__init__()
        
        self.m_k=[]
        # first m_k is initial ordering
        self.m_k.append(ordering)
        D=input_feat-1
        # sample m_k for every layer between min(m_k_prev) and D-1
        for j in range(num_layer-1):
            self.m_k.append(torch.randint(low=min(self.m_k[j-1]),high=D,size=(num_units,)))
        

        layers=[]
        # first layer has input_feat inputs and num_units outputs
        layers.append(MaskedLayer(in_feat=input_feat,out_feat=num_units,m_k=self.m_k[1],m_k_prev=self.m_k[0],layer_type="hidden"))
        layers.append(nn.ReLU())
        # every other layer has num_units inputs/ outputs
        for i in range(2,num_layer):
            layers.append(MaskedLayer(num_units,num_units,m_k=self.m_k[i],m_k_prev=self.m_k[i-1],layer_type="hidden"))
            layers.append(nn.ReLU())

        # last layer has different layer_type and input_feat outputs
        layers.append(MaskedLayer(num_units,input_feat,m_k=ordering,m_k_prev=self.m_k[num_layer-1],layer_type="output"))
        layers.append(nn.Sigmoid())


        self.layer=nn.ModuleList(layers)

    def forward(self,input):
        x = input
        for layer in self.layer:
            x = layer(x)
        return x
    

class MaskedLayer(nn.Linear):
    def __init__(self, in_feat, out_feat, m_k, m_k_prev, layer_type):
        super().__init__(in_features=in_feat, out_features=out_feat)
        # now the mask is a mxn matrix with only zeros
        self.register_buffer('mask', torch.zeros(out_feat, in_feat))

        # create mask according to algorithm in MADE paper
        for j in range(in_feat):
            for i in range(out_feat):
                if layer_type=="output":
                    if m_k[i]>m_k_prev[j]:
                        self.mask[i,j]=1
                else:
                    if m_k[i]>=m_k_prev[j]:
                        self.mask[i,j]=1
         
    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask, self.bias)

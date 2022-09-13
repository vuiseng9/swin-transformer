import onnx
from collections import OrderedDict
from onnx import numpy_helper

from onnx.parser import parse_model 
# print(onnx.helper.printable_graph(onnx_model.graph))

import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
import torch.nn as nn

class SparsityReporter():
    def __init__(
        self,
        model: nn.Module) -> None:
        
        self.model = model
        self.sparsity_df = self._get_layer_wise_sparsity()

    @staticmethod
    def calc_sparsity(tensor):
        if isinstance(tensor, torch.Tensor):
            rate = 1-(tensor.count_nonzero()/tensor.numel())
            return rate.item()
        else:
            rate = 1-(np.count_nonzero(tensor)/tensor.size)
            return rate

    @staticmethod
    def per_item_sparsity(state_dict):
        dlist=[]
        for key, param in state_dict.items():
            l = OrderedDict()
            l['layer_id'] = key
            l['shape'] = list(param.shape)
            l['nparam'] = np.prod(l['shape'])
            if isinstance(param, torch.Tensor):
                l['nnz'] = param.count_nonzero().item()
            else:
                l['nnz'] = np.count_nonzero(param)
            l['sparsity'] = SparsityReporter.calc_sparsity(param)
            dlist.append(l)
        df = pd.DataFrame.from_dict(dlist)
        return df

    def _get_layer_wise_sparsity(self):
        dlist=[]
        for n, m in self.model.named_modules():
            
            if hasattr(m, 'weight'):
                l = OrderedDict()
                l['layer_id'] = n
                l['layer_type'] = m.__class__.__name__
                l['param_type'] = 'weight'
                l['shape'] = list(m.weight.shape)
                l['nparam'] = np.prod(l['shape'])
                l['nnz'] = m.weight.count_nonzero().item()
                l['sparsity'] = self.calc_sparsity(m.weight)
                dlist.append(l)

            if hasattr(m, 'bias'):
                l = OrderedDict()
                l['layer_id'] = n
                l['layer_type'] = m.__class__.__name__
                l['param_type'] = 'bias'
                l['shape'] = list(m.bias.shape)
                l['nparam'] = np.prod(l['shape'])
                l['nnz'] = m.bias.count_nonzero().item()
                l['sparsity'] = self.calc_sparsity(m.bias)
                dlist.append(l)
                
        df = pd.DataFrame.from_dict(dlist)
        return df

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', '{:20,.2f}'.format)
pd.set_option('display.max_colwidth', None)

def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

def find_node_by_output(output :str, graph):
    # assuming output node is a singleton
    for node in graph.node:
        if output in node.output:
            break
    return node

def find_node(node_name, graph):
    for node in graph.node:
        if node_name == node.name:
            break
    return node

def find_initializer_node(init_node_name, graph):
    for init in graph.initializer:
        if init.name == init_node_name:
            break
    return init

class bert_onnx_mapper:
    def __init__(self, onnx_pth, variant=None):
        self.onnx_pth = onnx_pth
        self.variant = None
        onnx_model = onnx.load(onnx_pth)
        self.onnx_model = onnx_model

        if variant == 'nncf-quantized':
            quantized_tensor_nodes = OrderedDict()
            for node in onnx_model.graph.node:
                if 'fakequantize' in node.name.lower():
                    # From visual inspection, node.input[0] first input appears to be the constant tensor to be quantized
                    # From NodeProto of onnx, input is a list of string where each string is the name of output node of Constant Node (this is not initializer)
                    constant_node = find_node_by_output(node.input[0], onnx_model.graph)
                    if  constant_node.op_type == 'Constant':
                        if len(constant_node.attribute[0].t.dims) > 1:
                            print(node.name, 
                                ", input[0]:", node.input[0], 
                                ", constant_node:", constant_node.name, 
                                ", constant_node_type:", constant_node.op_type, 
                                ", dims:", constant_node.attribute[0].t.dims)
                        quantized_tensor_nodes[constant_node.name] = numpy_helper.to_array(constant_node.attribute[0].t)

            for init in onnx_model.graph.initializer:
                if 'bias' in init.name and 'bert.encoder' in init.name and 'LayerNorm' not in init.name:
                    print("bias_init_node:", init.name, ", dims:", init.dims)
                    quantized_tensor_nodes[init.name] = numpy_helper.to_array(init)

            self.quantized_tensor_nodes = None
            if len(quantized_tensor_nodes) > 0:
                self.quantized_tensor_nodes = quantized_tensor_nodes
        elif variant == 'nncf-sparsified':
            tensor_nodes = OrderedDict()
            for node in onnx_model.graph.node:
                if 'add' in node.name.lower():
                    for innode in node.input:
                        if 'bias' in innode and 'LayerNorm' not in innode:
                            # print(node.name, node.input)

                            bias_init_node = find_initializer_node(node.input[0], onnx_model.graph)
                            print("bias_init_node:", bias_init_node.name, ", dims:", bias_init_node.dims)
                            tensor_nodes[bias_init_node.name] = numpy_helper.to_array(bias_init_node)

                            weight_init_node_name = node.input[0].replace('bias','weight')
                            weight_init_node = find_initializer_node(weight_init_node_name, onnx_model.graph)
                            print("weight_init_node:", weight_init_node.name, ", dims:", weight_init_node.dims)
                            tensor_nodes[weight_init_node.name] = numpy_helper.to_array(weight_init_node)
            if len(tensor_nodes) > 0:
                self.tensor_nodes = tensor_nodes

if __name__ == "__main__":
    onnx_pth = "/tmp/vscode-dev/msft-swin-mvmt/mvmt-swin-b-p4-w7-224_22kto1k/default/ir/NNCFNetwork.fp32.onnx"
    mapper = bert_onnx_mapper(onnx_pth=onnx_pth, variant='nncf-sparsified')

    df = SparsityReporter.per_item_sparsity(mapper.tensor_nodes)

    # onnx_model = onnx.load(onnx_pth)

    print("dummy")
import os
import torch
import pandas as pd
from nncf.torch.layers import NNCFLinear

class SwinCropper:
    def __init__(self) -> None:
        pass

    def __call__(self, compression_ctrl, model, config):
        def get_parent(model, named_mod):
            scope_tokens = named_mod.split(".")
            child_attr = scope_tokens[-1]
            parent = model
            for attr in scope_tokens[:-1]:
                parent = parent._modules[attr]
            return parent, child_attr

        def get_tensor_shape_tuple(tensor):
            _torchsize = tensor.shape
            return tuple([_torchsize[dim] for dim in range(len(_torchsize))])
            
        mvmt_ctrl = None
        is_quantized = False
        if hasattr(compression_ctrl, 'child_ctrls'):
            for ctrl in compression_ctrl.child_ctrls:
                if ctrl.__class__.__name__ == 'QuantizationController':
                    is_quantized=True
                    for k, wqinfo in ctrl.weight_quantizers.items():
                        assert wqinfo.quantizer_module_ref.per_channel == False, "Per-channel wt.q {}".format(k.target_node_name)
                    for k, aqinfo in ctrl.non_weight_quantizers.items():
                        assert aqinfo.quantizer_module_ref.per_channel == False, "Per-channel act.q {}".format(k.target_node_name)
                elif ctrl.__class__.__name__ == 'MovementSparsityController':
                    mvmt_ctrl = ctrl
        elif compression_ctrl.__class__.__name__ == 'MovementSparsityController':
            mvmt_ctrl = compression_ctrl

        # if model_args.skip_quantize is True:
        #         is_quantized = False

        if mvmt_ctrl is not None:
            mvmt_ctrl._propagate_masks()

        structure_rpt = []
        with torch.no_grad():
            for group_id, ctxes in mvmt_ctrl.structured_ctx_by_group.items():
                for ctx in ctxes:
                    nncf_graph_node_name = ctx.sparsifying_node_name
                    named_mod = mvmt_ctrl.op2namedmodule[nncf_graph_node_name]
                    block_id = group_id

                    if any(map(nncf_graph_node_name.__contains__, ['fc1','fc2'])):
                        if nncf_graph_node_name.__contains__('fc1'):
                            # prune row
                            row_ids_to_keep = ctx.sparse_module_info.operand.weight_ctx.binary_mask.amax(dim=1).nonzero().reshape(-1)
                            finalw = ctx.sparse_module_info.module.weight[row_ids_to_keep, :].clone().detach()
                            finalb = ctx.sparse_module_info.module.bias[row_ids_to_keep].clone().detach()
                            assert finalw.shape[0] == finalb.shape[0], "mismatch bias to weight"
                            ofeat, ifeat = finalw.shape
                            new_module = torch.nn.Linear(ifeat, ofeat)
                            new_sd = new_module.state_dict()
                            new_sd['weight'] = finalw
                            new_sd['bias'] = finalb
                            new_module.load_state_dict(new_sd)
                            new_wrap_module = NNCFLinear.from_module(new_module)
                            parent, child_attr = get_parent(mvmt_ctrl.model, named_mod)
                            original_wrapped_module = getattr(parent, child_attr)
                            if is_quantized is True:
                                wq_mod = None
                                for preop, mod in original_wrapped_module.pre_ops.items():
                                    if mod.__class__.__name__ == 'UpdateWeight':
                                        wq_mod = mod
                                        break
                                if wq_mod is not None:
                                    new_wrap_module.pre_ops = torch.nn.ModuleDict({'0':wq_mod})
                            setattr(parent, child_attr, new_wrap_module)
                            # parent, child_attr = get_parent(final_model, named_mod.replace('nncf_module.',""))
                            # setattr(parent, child_attr, new_module)

                            orig_w_shape = get_tensor_shape_tuple(ctx.sparse_module_info.module.weight)
                            orig_b_shape = get_tensor_shape_tuple(ctx.sparse_module_info.module.bias)
                            final_w_shape = get_tensor_shape_tuple(finalw)
                            final_b_shape = get_tensor_shape_tuple(finalb)

                            structure_rpt.append(
                                dict(
                                    pt_module_name=named_mod,
                                    block_id=group_id,
                                    orig_w_shape=orig_w_shape,
                                    final_w_shape=final_w_shape,
                                    orig_b_shape=orig_b_shape,
                                    final_b_shape=final_b_shape,
                                    prune_by="row",
                                    id_to_keep=row_ids_to_keep.tolist(),
                                    head_id_to_keep=None,
                                    nncf_graph_node=nncf_graph_node_name,
                                )
                            )
                        else:
                            # prune col
                            col_ids_to_keep = ctx.sparse_module_info.operand.weight_ctx.binary_mask.amax(dim=0).nonzero().reshape(-1)
                            finalw = ctx.sparse_module_info.module.weight[:, col_ids_to_keep].clone().detach()
                            finalb = ctx.sparse_module_info.module.bias[:].clone().detach()
                            assert finalw.shape[0] == finalb.shape[0], "mismatch bias to weight"
                            ofeat, ifeat = finalw.shape
                            new_module = torch.nn.Linear(ifeat, ofeat)
                            new_sd = new_module.state_dict()
                            new_sd['weight'] = finalw
                            new_sd['bias'] = finalb
                            new_module.load_state_dict(new_sd)
                            new_wrap_module = NNCFLinear.from_module(new_module)
                            parent, child_attr = get_parent(mvmt_ctrl.model, named_mod)
                            original_wrapped_module = getattr(parent, child_attr)
                            if is_quantized is True:
                                wq_mod = None
                                for preop, mod in original_wrapped_module.pre_ops.items():
                                    if mod.__class__.__name__ == 'UpdateWeight':
                                        wq_mod = mod
                                        break
                                if wq_mod is not None:
                                    new_wrap_module.pre_ops = torch.nn.ModuleDict({'0':wq_mod})
                            setattr(parent, child_attr, new_wrap_module)
                            # parent, child_attr = get_parent(final_model, named_mod.replace('nncf_module.',""))
                            # setattr(parent, child_attr, new_module)

                            orig_w_shape = get_tensor_shape_tuple(ctx.sparse_module_info.module.weight)
                            orig_b_shape = get_tensor_shape_tuple(ctx.sparse_module_info.module.bias)
                            final_w_shape = get_tensor_shape_tuple(finalw)
                            final_b_shape = get_tensor_shape_tuple(finalb)

                            structure_rpt.append(
                                dict(
                                    pt_module_name=named_mod,
                                    block_id=group_id,
                                    orig_w_shape=orig_w_shape,
                                    final_w_shape=final_w_shape,
                                    orig_b_shape=orig_b_shape,
                                    final_b_shape=final_b_shape,
                                    prune_by="col",
                                    id_to_keep=col_ids_to_keep.tolist(),
                                    head_id_to_keep=None,
                                    nncf_graph_node=nncf_graph_node_name,
                                )
                            )
                    else:
                        # ndiv = ctx.dependent_structured_mask.reshape(-1).shape[0]
                        # head_id_to_keep = torch.masked_select(torch.range(0, ndiv-1, dtype=int), 
                        #                     ctx.dependent_structured_mask.reshape(-1).cpu().to(bool)).tolist()

                        if any(map(nncf_graph_node_name.__contains__, ['qkv'])):
                            # prune row
                            row_ids_to_keep = ctx.sparse_module_info.operand.weight_ctx.binary_mask.amax(dim=1).nonzero().reshape(-1)
                            finalw = ctx.sparse_module_info.module.weight[row_ids_to_keep, :].clone().detach()
                            finalb = ctx.sparse_module_info.module.bias[row_ids_to_keep].clone().detach()
                            assert finalw.shape[0] == finalb.shape[0], "mismatch bias to weight"
                            ofeat, ifeat = finalw.shape
                            new_module = torch.nn.Linear(ifeat, ofeat)
                            new_sd = new_module.state_dict()
                            new_sd['weight'] = finalw
                            new_sd['bias'] = finalb
                            new_module.load_state_dict(new_sd)
                            new_wrap_module = NNCFLinear.from_module(new_module)
                            parent, child_attr = get_parent(mvmt_ctrl.model, named_mod)
                            original_wrapped_module = getattr(parent, child_attr)
                            if is_quantized is True:
                                wq_mod = None
                                for preop, mod in original_wrapped_module.pre_ops.items():
                                    if mod.__class__.__name__ == 'UpdateWeight':
                                        wq_mod = mod
                                        break
                                if wq_mod is not None:
                                    new_wrap_module.pre_ops = torch.nn.ModuleDict({'0':wq_mod})
                            setattr(parent, child_attr, new_wrap_module)
                            # parent, child_attr = get_parent(final_model, named_mod.replace('nncf_module.',""))
                            # setattr(parent, child_attr, new_module)
                            dim_per_head=32
                            parent.all_head_size = int(ofeat)
                            parent.num_attention_heads = int(ofeat/dim_per_head)
                            
                            orig_w_shape = get_tensor_shape_tuple(ctx.sparse_module_info.module.weight)
                            orig_b_shape = get_tensor_shape_tuple(ctx.sparse_module_info.module.bias)
                            final_w_shape = get_tensor_shape_tuple(finalw)
                            final_b_shape = get_tensor_shape_tuple(finalb)

                            structure_rpt.append(
                                dict(
                                    pt_module_name=named_mod,
                                    block_id=group_id,
                                    orig_w_shape=orig_w_shape,
                                    final_w_shape=final_w_shape,
                                    orig_b_shape=orig_b_shape,
                                    final_b_shape=final_b_shape,
                                    prune_by="group of {} rows".format(dim_per_head),
                                    id_to_keep=row_ids_to_keep.tolist(),
                                    head_id_to_keep=(row_ids_to_keep//dim_per_head)[(row_ids_to_keep%dim_per_head)==0].tolist(),
                                    nncf_graph_node=nncf_graph_node_name,
                                )
                            )
                        else:
                            # prune col
                            col_ids_to_keep = ctx.sparse_module_info.operand.weight_ctx.binary_mask.amax(dim=0).nonzero().reshape(-1)
                            finalw = ctx.sparse_module_info.module.weight[:, col_ids_to_keep].clone().detach()
                            finalb = ctx.sparse_module_info.module.bias[:].clone().detach()
                            assert finalw.shape[0] == finalb.shape[0], "mismatch bias to weight"
                            ofeat, ifeat = finalw.shape
                            new_module = torch.nn.Linear(ifeat, ofeat)
                            new_sd = new_module.state_dict()
                            new_sd['weight'] = finalw
                            new_sd['bias'] = finalb
                            new_module.load_state_dict(new_sd)
                            new_wrap_module = NNCFLinear.from_module(new_module)
                            parent, child_attr = get_parent(mvmt_ctrl.model, named_mod)
                            original_wrapped_module = getattr(parent, child_attr)
                            if is_quantized is True:
                                wq_mod = None
                                for preop, mod in original_wrapped_module.pre_ops.items():
                                    if mod.__class__.__name__ == 'UpdateWeight':
                                        wq_mod = mod
                                        break
                                if wq_mod is not None:
                                    new_wrap_module.pre_ops = torch.nn.ModuleDict({'0':wq_mod})
                            setattr(parent, child_attr, new_wrap_module)
                            # parent, child_attr = get_parent(final_model, named_mod.replace('nncf_module.',""))
                            # setattr(parent, child_attr, new_module)

                            orig_w_shape = get_tensor_shape_tuple(ctx.sparse_module_info.module.weight)
                            orig_b_shape = get_tensor_shape_tuple(ctx.sparse_module_info.module.bias)
                            final_w_shape = get_tensor_shape_tuple(finalw)
                            final_b_shape = get_tensor_shape_tuple(finalb)

                            structure_rpt.append(
                                dict(
                                    pt_module_name=named_mod,
                                    block_id=group_id,
                                    orig_w_shape=orig_w_shape,
                                    final_w_shape=final_w_shape,
                                    orig_b_shape=orig_b_shape,
                                    final_b_shape=final_b_shape,
                                    prune_by="group of {} cols".format(dim_per_head),
                                    id_to_keep=col_ids_to_keep.tolist(),
                                    head_id_to_keep=(col_ids_to_keep//dim_per_head)[(col_ids_to_keep%dim_per_head)==0].tolist(),
                                    nncf_graph_node=nncf_graph_node_name,
                                )
                            )
            ir_dir = os.path.join(config.OUTPUT, "ir")
            os.makedirs(ir_dir, exist_ok=True)
            torch.save(structure_rpt, os.path.join(ir_dir, "sparsity_structures.pkl"))
            structure_df = pd.DataFrame.from_dict(structure_rpt)
            structure_df.id_to_keep = "See pkl"

            structure_df.to_csv(
                os.path.join(ir_dir, "sparsity_structures.csv"), index=False)
            with open(os.path.join(ir_dir, 'sparsity_structures.md'), 'w') as f:
                structure_df.to_markdown(f)

        return model

import os
import torch

from collections import OrderedDict

input_ckpt_pth="/data2/vchua/run/dgx4-swin-opt/swin/pretrained/swin_base_patch4_window7_224_22k.pth"

def convert_swin_ckpt_to_split_qkv(ckpt_pth):
    if os.path.exists(ckpt_pth) is False:
        raise ValueError("Invalid path: {}".format(ckpt_pth))
    
    sd = torch.load(ckpt_pth)

    if 'model' in sd:
        net_sd = sd['model']

        if sum(list(map(lambda key: 'attn.qkv' in key, list(net_sd.keys())))) == 0:
            raise ValueError("Unsupported input ckpt structures, expecting attn.qkv in model state dict")

        new_net_sd = OrderedDict()

        with torch.no_grad():
            for param_name, param_tensor in net_sd.items():
                if 'attn.qkv' in param_name:
                    if 'weight' in param_name or 'bias' in param_name:
                        q, k, v = param_tensor.chunk(3, dim=0)
                        dim_per_proj = param_tensor.shape[0]//3
                        assert (param_tensor[              :dim_per_proj,  ] == q).sum().item() == q.numel()
                        assert (param_tensor[  dim_per_proj:dim_per_proj*2,] == k).sum().item() == k.numel()
                        assert (param_tensor[dim_per_proj*2:dim_per_proj*3,] == v).sum().item() == v.numel()

                        new_net_sd[ param_name.replace("attn.qkv", "attn.q") ] = q
                        new_net_sd[ param_name.replace("attn.qkv", "attn.k") ] = k
                        new_net_sd[ param_name.replace("attn.qkv", "attn.v") ] = v
                else:
                    new_net_sd[param_name]=param_tensor

        sd['model'] = new_net_sd
        dirname = os.path.dirname(input_ckpt_pth)
        filename = os.path.basename(input_ckpt_pth)
        torch.save(sd, os.path.join(dirname, "split_qkv_{}".format(filename)))
    else:
        raise ValueError("Unsupported input ckpt structures, expecting model key")

convert_swin_ckpt_to_split_qkv(input_ckpt_pth)

print("done")
import os
from tabnanny import check
import torch

from collections import OrderedDict

def check_path(pth):
    if os.path.exists(pth) is False:
        raise ValueError("Invalid path: {}".format(pth))

def convert_swin_mvmt_ckpt_to_vanilla_ckpt(refckpt_pth, nncfckpt_pth, allow_shape_mismatch=True):
    # Important! we assume the parameters are sparsified (zero has been burnt in from mask)
    check_path(refckpt_pth)
    check_path(nncfckpt_pth)
    
    refsd = torch.load(refckpt_pth)
    nncfsd = torch.load(nncfckpt_pth)

    ref_netsd=refsd['model']
    nncf_netsd=nncfsd['model']

    new_netsd = OrderedDict()

    with torch.no_grad():
        for param_name, param_tensor in nncf_netsd.items():
            if 'pre_ops' not in param_name:
                new_param_name = param_name.split("nncf_module.")[-1]
                new_netsd[new_param_name] = param_tensor
                
    assert len(new_netsd) == len(ref_netsd), "number of keys unmatched with reference ckpt, pls debug"

    for k, v in new_netsd.items():
        if allow_shape_mismatch is False:
            assert v.shape == ref_netsd[k].shape, "tensor of {} dimension mismatched".format(k)
        else:
            assert k in ref_netsd, "{} key not found in reference state dict".format(k) # will error out if key mismatch 

    nncfsd['model'] = new_netsd
    dirname = os.path.dirname(nncfckpt_pth)
    filename = os.path.basename(nncfckpt_pth)
    torch.save(nncfsd, os.path.join(dirname, "unwrapped_nncf_{}".format(filename)))

def convert_mvmt_sd_to_jpq_sd(mvmt_pth, jpq_path):
    # Assumption preops 0 is moment
    check_path(mvmt_pth)
    check_path(jpq_path)

    mvmtsd = torch.load(mvmt_pth)
    jpqsd = torch.load(jpq_path)

    mvmt_netsd=mvmtsd['model']
    jpq_netsd=jpqsd['model']

    with torch.no_grad():
        for k, v in mvmt_netsd.items():
            assert k in jpq_netsd, "Assumption is broken (key must exist), pls debug"
            assert v.shape == jpq_netsd[k].shape, "Assumption is broken (shape must match), pls debug"
            # overwrite existing jpq_netsd
            jpq_netsd[k] = v

    jpqsd['model'] = jpq_netsd
    dirname = os.path.dirname(jpq_path)
    filename = os.path.basename(jpq_path)
    torch.save(jpqsd, os.path.join(dirname, "inherit_mvmt_sd_{}".format(filename)))

    print("yatta")
# refckpt_pth="/data5/vchua/run/test-a100x6/swin/swin-b-p4-w7-r224-22kto1k-ftuned-bs128-regression/swin_base_patch4_window7_224_22kto1k_finetune/default/ckpt_epoch_28.pth"
# nncfckpt_pth="/data5/vchua/run/test-a100x6/swin/mvmt-swin-b-bs128-r0.010-mhsa16x16-40eph/mvmt-swin-b-p4-w7-224_22kto1k/default/ckpt_epoch_37.pth"
# convert_swin_mvmt_ckpt_to_vanilla_ckpt(refckpt_pth, nncfckpt_pth)


mvmt_pth="/data5/vchua/run/test-a100x6/swin/mvmt-swin-b-bs128-r0.010-mhsa16x16-60eph/mvmt-swin-b-p4-w7-224_22kto1k/default/ckpt_epoch_58.pth"
jpq_path="/tmp/vscode-dev/msft-swin-jpq-ckpt/jpq-swin-b-p4-w7-224_22kto1k/default/ckpt_epoch_0.pth"
convert_mvmt_sd_to_jpq_sd(mvmt_pth, jpq_path)

print("done")
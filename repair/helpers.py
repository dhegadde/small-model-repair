"""Small helpers copied from the lab's other code so this package stands alone."""
import numpy as np


def auroc(conf, correct):
    """Area under the ROC curve of `conf` for separating correct from incorrect (ties get average ranks)."""
    pos, neg = conf[correct], conf[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def hqq_quantize_(model, nbits, group_size=64):
    """Replace every nn.Linear inside the decoder layers with an HQQ-quantized copy (in place); embeddings and lm_head
    stay in bf16."""
    import torch
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
    cfg = BaseQuantizeConfig(nbits=nbits, group_size=group_size)
    targets = [(name, mod) for name, mod in model.named_modules()
               if isinstance(mod, torch.nn.Linear) and ".layers." in name]
    for name, mod in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        dev = mod.weight.device
        with torch.cuda.device(dev):  # HQQ allocates on the CURRENT device: set it for layers on a 2nd GPU
            setattr(parent, name.rsplit(".", 1)[1],
                    HQQLinear(mod, cfg, compute_dtype=torch.bfloat16, device=str(dev), del_orig=True))
    torch.cuda.empty_cache()
    return len(targets)


def peft_without_torchao():
    """peft refuses torchao < 0.16 even when torchao-quantized layers are not used; tell it torchao is absent."""
    import peft.import_utils as iu
    import peft.tuners.lora.torchao as lt
    iu.is_torchao_available = lambda: False
    lt.is_torchao_available = lambda: False

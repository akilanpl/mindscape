"""Engineering checks on MPS only; random untrained outputs never count as research success."""

import torch

from .model import GraphTransition, tensors


def check(graphs):
    with torch.device("mps"):
        model = GraphTransition(101)
    records = []
    for graph in graphs:
        domain, x, masks, rel, active, action = tensors([graph])
        output, precondition = model(domain, x, masks, rel, active, action)
        loss = output.square().sum() + precondition.square().sum()
        model.zero_grad(set_to_none=True)
        loss.backward()
        if (
            model.core.action.weight.grad is None
            or not torch.isfinite(model.core.action.weight.grad).all()
        ):
            raise ValueError("Missing shared-core action gradient")
        n = len(graph["entities"])
        order = torch.arange(x.shape[1], device="mps")
        order[:n] = order[:n].flip(0)
        with torch.inference_mode():
            permuted, pvalid = model(
                domain,
                x[:, order],
                masks[:, order],
                rel[:, order][:, :, order],
                active[:, order],
                action,
            )
            changed = action.clone()
            changed[:, 4] += 0.25
            altered, _ = model(domain, x, masks, rel, active, changed)
        permutation_error = float((permuted - output[:, order]).abs().max().item())
        precondition_error = float((pvalid - precondition).abs().max().item())
        if permutation_error > 1e-5 or precondition_error > 1e-5:
            raise ValueError("Entity-order equivariance violated")
        sensitivity = float((altered - output).abs().max().item())
        if sensitivity <= 1e-8:
            raise ValueError("Action ignored")
        records.append(
            {
                "domain": domain,
                "permutation_error": permutation_error,
                "action_sensitivity": sensitivity,
                "shared_action_gradient_norm": float(model.core.action.weight.grad.norm().item()),
            }
        )
    return {
        "status": "PASS",
        "checks": records,
        "parameters": sum(p.numel() for p in model.parameters()),
        "scientific_result": False,
    }

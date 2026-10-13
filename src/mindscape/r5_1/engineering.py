"""Actual MPS shape/gradient/connectivity tests on unchanged V5; zero updates."""

import torch
import torch.nn.functional as F

from mindscape.v3.physics_world.accelerator import require_mps, verify_placement
from mindscape.v5.data import graph
from mindscape.v5.model import GraphTransition, tensors


def norms(model):
    return {
        name: float(p.grad.norm().item()) if p.grad is not None else None
        for name, p in model.named_parameters()
    }


def run(manifest):
    require_mps()
    protocol = manifest["protocol"]["engineering"]
    threshold = protocol["minimum_gradient_norm"]
    with torch.device("mps"):
        model = GraphTransition(protocol["seed"])
    verify_placement(model)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != protocol["expected_parameters"]:
        raise ValueError("Actual parameter count differs")
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    records = []
    for domain in ("mathematics", "coding", "physics"):
        selected = [
            t for t in manifest["tasks"] if t["domain"] == domain and t["split"] == "composition"
        ][:2]
        # These probes read graphs only, never execute a held-out action or derive its targets.
        graphs = [graph(t) for t in selected]
        inputs = tensors(graphs)
        domain, x, masks, rel, active, action = inputs
        x = x.detach().requires_grad_(True)
        rel = rel.detach().requires_grad_(True)
        action = action.detach().requires_grad_(True)
        inputs = (domain, x, masks, rel, active, action)
        expected_input = (2, 7, protocol["encoder_feature_widths"][domain])
        expected_output = (2, 7, protocol["effect_widths"][domain])
        if (
            tuple(x.shape) != expected_input
            or tuple(action.shape) != (2, 8)
            or tuple(rel.shape) != (2, 7, 7, 3)
        ):
            raise ValueError("Actual neural input shape mismatch")
        gradient_receipts = {}
        for path in ("effect", "precondition"):
            model.zero_grad(set_to_none=True)
            for value in (x, rel, action):
                value.grad = None
            effect, valid = model(*inputs)
            if tuple(effect.shape) != expected_output or tuple(valid.shape) != (2,):
                raise ValueError("Actual neural output shape mismatch")
            if not torch.isfinite(effect).all() or not torch.isfinite(valid).all():
                raise ValueError("Nonfinite neural output")
            # Fixed synthetic engineering probes are not data supervision or optimizer updates.
            loss = (
                ((effect - 0.137) ** 2 * active[..., None]).sum()
                if path == "effect"
                else F.binary_cross_entropy_with_logits(
                    valid, torch.tensor([0.2, 0.8], device="mps")
                )
            )
            loss.backward()
            gradient = norms(model)
            for name, p in model.named_parameters():
                expected = (
                    name.startswith(("core.", "encoders." + domain + "."))
                    or path == "effect"
                    and name.startswith("heads." + domain + ".")
                )
                if path == "effect" and name.startswith("core.precondition."):
                    expected = False
                value = gradient[name]
                if expected:
                    if (
                        value is None
                        or not torch.isfinite(p.grad).all()
                        or p.grad.device.type != "mps"
                        or value <= threshold
                    ):
                        raise ValueError(
                            f"Missing/zero/nonfinite {domain} {path} gradient: {name}={value}"
                        )
                elif value is not None:
                    raise ValueError("Inactive domain/head received gradient: " + name)
            for label, value in (
                ("entity_input", x),
                ("relation_input", rel),
                ("action_input", action),
            ):
                if (
                    value.grad is None
                    or not torch.isfinite(value.grad).all()
                    or value.grad.norm().item() <= threshold
                ):
                    raise ValueError("Disconnected input gradient: " + label)
            if (x.grad * (1 - masks)).abs().max().item() != 0:
                raise ValueError("Hidden/masked input affected output")
            gradient_receipts[path] = {
                "parameters": gradient,
                "input_norms": {
                    label: float(value.grad.norm().item())
                    for label, value in (
                        ("entity_input", x),
                        ("relation_input", rel),
                        ("action_input", action),
                    )
                },
            }
        with torch.inference_mode():
            effect, valid = model(*inputs)
            changed = action.clone()
            changed[:, 4] += 0.25
            altered, _ = model(domain, x, masks, rel, active, changed)
            no_relation = torch.zeros_like(rel)
            disconnected, _ = model(domain, x, masks, no_relation, active, action)
            action_difference = float((altered - effect).abs().max().item())
            relation_difference = float((disconnected - effect).abs().max().item())
            permutation = torch.arange(7, device="mps").flip(0)
            permuted, pvalid = model(
                domain,
                x[:, permutation],
                masks[:, permutation],
                rel[:, permutation][:, :, permutation],
                active[:, permutation],
                action,
            )
            permutation_error = float((permuted - effect[:, permutation]).abs().max().item())
            precondition_error = float((pvalid - valid).abs().max().item())
        if (
            min(action_difference, relation_difference)
            <= protocol["minimum_action_or_relation_sensitivity"]
        ):
            raise ValueError("Action/relation disconnected")
        if max(permutation_error, precondition_error) > protocol["maximum_permutation_error"]:
            raise ValueError("Node-order equivariance failed")
        records.append(
            {
                "domain": domain,
                "input_shape": list(x.shape),
                "masked_encoder_width": 2 * x.shape[-1],
                "effect_shape": list(effect.shape),
                "precondition_shape": list(valid.shape),
                "gradients": gradient_receipts,
                "action_sensitivity": action_difference,
                "relation_sensitivity": relation_difference,
                "permutation_error": permutation_error,
            }
        )
    if any(not torch.equal(initial[name], p) for name, p in model.named_parameters()):
        raise ValueError("Engineering check changed weights")
    return {
        "status": "PASS",
        "device": "mps",
        "parameters": parameters,
        "domain_checks": records,
        "optimizer_steps": 0,
        "weights_unchanged": True,
        "composition_targets_accessed": False,
        "meaning": "Engineering connectivity only, not learning, accuracy or transfer evidence.",
    }

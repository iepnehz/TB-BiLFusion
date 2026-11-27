from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

TensorList = List[torch.Tensor]


@dataclass
class TaskStats:
    direct_norm: float = 0.0
    implicit_norm: float = 0.0
    hyper_norm: float = 0.0
    lower_grad_norm: float = 0.0
    lower_loss: float = 0.0
    weight: float = 0.0
    cosine_before: float = 0.0
    cosine_after: float = 0.0
    upper_loss: float = 0.0
    robust_spread: float = 0.0
    curvature_rayleigh: float = 0.0
    fd_radius: float = 0.0
    tail_count: int = 1
    raw_cosine_vp: float = 0.0
    solver_skipped: int = 0
    probe_cosine_gq: float = 0.0
    galerkin_alpha: float = 0.0
    galerkin_alpha_raw: float = 0.0
    trust_delta: float = 0.0
    alpha_clipped: int = 0
    rayleigh_ratio: float = 0.0
    lower_feasible_before: float = 0.0
    lower_feasible_after: float = 0.0
    lower_projection_active: int = 0


class Architect:
    def __init__(self, model, args):
        self.model = model
        self.lr = float(getattr(args, "arch_learning_rate", 1e-4))
        self.weight_decay = float(getattr(args, "arch_weight_decay", 1e-3))

        self.fd_radius = float(getattr(args, "fd_radius", 1e-2))
        self.fd_eps = float(getattr(args, "hypergrad_eps", 1e-8))
        self.implicit_scale = float(getattr(args, "implicit_scale", 1.0))
        self.hypergrad_clip = float(getattr(args, "hypergrad_clip", 5.0))

        self.cvar_alpha = float(getattr(args, "cvar_alpha", 0.40))
        self.cvar_tau = float(getattr(args, "cvar_tau", 0.05))
        self.robust_topk = max(1, int(getattr(args, "robust_topk", 3)))
        self.ir_attn = float(getattr(args, "ir_attenuation", 0.70))
        self.vis_attn = float(getattr(args, "vis_attenuation", 0.70))

        self.optimizer = torch.optim.SGD(
            self.model.enhance_net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.last_stats: Dict[str, TaskStats] = {
            "visual": TaskStats(),
            "perception": TaskStats(),
        }

    @staticmethod
    def _params(module) -> List[torch.nn.Parameter]:
        return [p for p in module.parameters() if p.requires_grad]

    @staticmethod
    def _replace_none(grads, params) -> TensorList:
        return [
            torch.zeros_like(p) if g is None else g
            for g, p in zip(grads, params)
        ]

    @staticmethod
    def _norm(xs: Sequence[torch.Tensor]) -> torch.Tensor:
        if not xs:
            return torch.tensor(0.0)
        value = torch.zeros((), device=xs[0].device, dtype=xs[0].dtype)
        for item in xs:
            value = value + torch.sum(item.detach() * item.detach())
        return torch.sqrt(value + 1e-12)

    @staticmethod
    def _dot(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
        value = torch.zeros((), device=a[0].device, dtype=a[0].dtype)
        for x, y in zip(a, b):
            value = value + torch.sum(x * y)
        return value

    @staticmethod
    def _clone(xs: Sequence[torch.Tensor]) -> TensorList:
        return [item.detach().clone() for item in xs]

    def _clip_hypergradient(self, hyper: TensorList) -> TensorList:
        if self.hypergrad_clip <= 0:
            return hyper
        hyper_norm = self._norm(hyper)
        factor = torch.clamp(
            torch.as_tensor(
                self.hypergrad_clip,
                device=hyper_norm.device,
                dtype=hyper_norm.dtype,
            )
            / (hyper_norm + self.fd_eps),
            max=1.0,
        )
        return [item * factor for item in hyper]

    @staticmethod
    def _auto_fd_radius(xs: Sequence[torch.Tensor]) -> float:
        if not xs:
            return 1e-4
        numel = sum(item.numel() for item in xs)
        norm = torch.sqrt(
            sum(torch.sum(item.detach().pow(2)) for item in xs)
        )
        finfo = torch.finfo(xs[0].dtype) if xs[0].is_floating_point() else None
        eps = finfo.eps if finfo is not None else 1e-7
        return float(
            (eps ** 0.5) * (1.0 + norm.item() / max(numel ** 0.5, 1.0))
        )

    def _snapshot_buffers(self, modules) -> Dict[int, torch.Tensor]:
        state: Dict[int, torch.Tensor] = {}
        seen = set()
        for module in modules:
            for buffer in module.buffers():
                if buffer is None or id(buffer) in seen or not torch.is_tensor(buffer):
                    continue
                seen.add(id(buffer))
                state[id(buffer)] = buffer.detach().clone()
        return state

    @staticmethod
    def _restore_buffers(modules, state: Dict[int, torch.Tensor]) -> None:
        seen = set()
        for module in modules:
            for buffer in module.buffers():
                if buffer is None or id(buffer) in seen or id(buffer) not in state:
                    continue
                seen.add(id(buffer))
                with torch.no_grad():
                    buffer.copy_(state[id(buffer)])

    @staticmethod
    def _snapshot_rng():
        state = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng(state) -> None:
        if not state:
            return
        torch.set_rng_state(state["cpu"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    @staticmethod
    def _apply_delta(params, directions, scale: float) -> None:
        with torch.no_grad():
            for param, direction in zip(params, directions):
                param.add_(direction, alpha=scale)

    def _grad_pair_at_perturbation(
        self,
        lower_loss_fn,
        theta,
        omega,
        direction,
        radius: float,
        buffer_modules,
    ):
        direction_norm = self._norm(direction).detach()
        eps = float(radius) / float(direction_norm.item() + self.fd_eps)
        eps = max(eps, 1e-7)

        if buffer_modules:
            base_buffers = self._snapshot_buffers(buffer_modules)
        self._apply_delta(theta, direction, +eps)
        try:
            lower_loss = lower_loss_fn()
            grads_plus = torch.autograd.grad(
                lower_loss,
                tuple(theta) + tuple(omega),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        finally:
            self._apply_delta(theta, direction, -eps)
            if buffer_modules:
                self._restore_buffers(buffer_modules, base_buffers)

        num_theta = len(theta)
        g_theta_plus = self._replace_none(grads_plus[:num_theta], theta)
        g_omega_plus = self._replace_none(grads_plus[num_theta:], omega)
        return g_theta_plus, g_omega_plus, eps

    def _task_hypergradient(self, upper_loss, lower_loss_fn, lower_module):
        omega = self._params(self.model.enhance_net)
        theta = self._params(lower_module)
        if not omega or not theta:
            return [torch.zeros_like(param) for param in omega], TaskStats(), []

        upper_grads = torch.autograd.grad(
            upper_loss,
            tuple(theta) + tuple(omega),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        num_theta = len(theta)
        q = self._replace_none(upper_grads[:num_theta], theta)
        direct = self._replace_none(upper_grads[num_theta:], omega)

        rng_base = self._snapshot_rng()
        lower_loss = lower_loss_fn()
        lower_grads = torch.autograd.grad(
            lower_loss,
            tuple(theta) + tuple(omega),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        rng_after_base = self._snapshot_rng()
        g = self._replace_none(lower_grads[:num_theta], theta)
        g_omega = self._replace_none(lower_grads[num_theta:], omega)

        q_norm = self._norm(q)
        g_norm = self._norm(g)
        probe_cosine_gq = float(
            (self._dot(g, q) / (g_norm * q_norm + self.fd_eps))
            .clamp(-1.0, 1.0)
            .item()
        )

        if q_norm.item() > self.fd_eps:
            probe = [item / q_norm for item in q]
        else:
            probe = [torch.zeros_like(item) for item in q]

        hu_theta = [torch.zeros_like(item) for item in g]
        hu_omega = [torch.zeros_like(item) for item in direct]
        used_eps = 0.0
        curvature = torch.zeros(
            (), device=omega[0].device, dtype=omega[0].dtype
        )
        projection = torch.zeros_like(curvature)

        if q_norm.item() > self.fd_eps:
            radius = self._auto_fd_radius(theta)
            self._restore_rng(rng_base)
            try:
                g_theta_plus, g_omega_plus, used_eps = (
                    self._grad_pair_at_perturbation(
                        lower_loss_fn,
                        theta,
                        omega,
                        probe,
                        radius,
                        [lower_module, self.model.enhance_net],
                    )
                )
            finally:
                self._restore_rng(rng_after_base)
            inverse_eps = 1.0 / max(used_eps, self.fd_eps)
            hu_theta = [
                (plus - base) * inverse_eps
                for plus, base in zip(g_theta_plus, g)
            ]
            hu_omega = [
                (plus - base) * inverse_eps
                for plus, base in zip(g_omega_plus, g_omega)
            ]
            curvature = self._dot(probe, hu_theta).detach()
            projection = self._dot(probe, q).detach()

        hu_norm = self._norm(hu_theta).detach()
        curvature_floor = torch.as_tensor(
            torch.finfo(curvature.dtype).eps ** 0.5,
            device=curvature.device,
            dtype=curvature.dtype,
        ) * (hu_norm + self.fd_eps)
        alpha_raw = torch.where(
            curvature.abs() > curvature_floor,
            projection
            / torch.where(
                curvature.abs() > curvature_floor,
                curvature,
                torch.ones_like(curvature),
            ),
            torch.zeros_like(curvature),
        )

        hw_norm = self._norm(hu_omega).detach()
        direct_norm = self._norm(direct).detach()
        trust_delta_tensor = torch.where(
            hw_norm > self.fd_eps,
            direct_norm / (hw_norm + self.fd_eps),
            torch.zeros_like(direct_norm),
        )
        alpha_linear = torch.clamp(
            alpha_raw, -trust_delta_tensor, trust_delta_tensor
        )
        alpha_concave = torch.sign(projection) * trust_delta_tensor
        alpha = torch.where(curvature > 0.0, alpha_linear, alpha_concave)

        solver_skipped = int(
            q_norm.item() <= self.fd_eps
            or curvature.abs().item() <= curvature_floor.item()
        )
        mixed = [alpha * item for item in hu_omega]
        hyper = [
            direct_item - self.implicit_scale * mixed_item
            for direct_item, mixed_item in zip(direct, mixed)
        ]
        hyper = self._clip_hypergradient(hyper)

        stats = TaskStats(
            direct_norm=float(self._norm(direct).item()),
            implicit_norm=float(self._norm(mixed).item()),
            hyper_norm=float(self._norm(hyper).item()),
            lower_grad_norm=float(g_norm.item()),
            lower_loss=float(lower_loss.detach().item()),
            curvature_rayleigh=float(curvature.item()),
            fd_radius=float(used_eps),
            solver_skipped=solver_skipped,
            probe_cosine_gq=probe_cosine_gq,
            galerkin_alpha=float(alpha.item()),
            galerkin_alpha_raw=float(alpha_raw.item()),
            trust_delta=float(trust_delta_tensor.item()),
            alpha_clipped=int(
                (
                    curvature.item() > 0.0
                    and alpha_raw.abs().item() > trust_delta_tensor.item()
                )
                or (curvature.item() <= 0.0 and projection.item() != 0.0)
            ),
            rayleigh_ratio=float(
                (curvature.abs() / (hu_norm + self.fd_eps)).item()
            ),
        )
        return hyper, stats, g_omega

    @staticmethod
    def _scenario_inputs(
        ir,
        vis,
        scenario,
        ir_attn=0.7,
        vis_attn=0.7,
        ir_contrast_alpha=0.5,
        ir_contrast_kernel=7,
        ir_gaussian_sigma=0.02,
        ir_stripe_level=0.02,
        vis_exposure_gain=0.8,
        vis_poisson_gain=15,
        vis_read_sigma=0.02,
        fog_transmittance=0.6,
        fog_atm_light=0.9,
    ):
        device, dtype = ir.device, ir.dtype
        noise_dtype = (
            dtype
            if dtype in (torch.float32, torch.float64)
            else torch.float32
        )

        if scenario == "nominal":
            return ir, vis

        if scenario == "ir_low_contrast":
            padding = ir_contrast_kernel // 2
            ir_pad = (
                F.pad(
                    ir,
                    (padding, padding, padding, padding),
                    mode="replicate",
                )
                if padding > 0
                else ir
            )
            local_mean = F.avg_pool2d(
                ir_pad,
                kernel_size=ir_contrast_kernel,
                stride=1,
            )
            ir_out = local_mean + ir_contrast_alpha * (ir - local_mean)
            return ir_out.clamp(0.0, 1.0), vis

        if scenario == "ir_stripenoise":
            batch, _, height, width = ir.shape
            row_noise = (
                torch.randn(
                    batch, 1, height, 1, device=device, dtype=noise_dtype
                ).to(dtype)
                * ir_stripe_level
            )
            col_noise = (
                torch.randn(
                    batch, 1, 1, width, device=device, dtype=noise_dtype
                ).to(dtype)
                * ir_stripe_level
            )
            gaussian = (
                torch.randn(ir.shape, device=device, dtype=noise_dtype).to(dtype)
                * ir_gaussian_sigma
            )
            return (ir + row_noise + col_noise + gaussian).clamp(0.0, 1.0), vis

        if scenario == "vis_gaussiannoise":
            vis_work = vis.to(noise_dtype)
            vis_dim = vis_work * vis_exposure_gain
            poisson_sample = (
                torch.poisson(vis_dim * vis_poisson_gain) / vis_poisson_gain
            )
            poisson_noise = poisson_sample - vis_dim
            read_noise = (
                torch.randn(
                    vis_work.shape, device=device, dtype=noise_dtype
                )
                * vis_read_sigma
            )
            vis_out = vis_dim + poisson_noise + read_noise
            return ir, vis_out.clamp(0.0, 1.0).to(dtype)

        if scenario == "vis_fog":
            vis_out = (
                vis * fog_transmittance
                + fog_atm_light * (1.0 - fog_transmittance)
            )
            return ir, vis_out.clamp(0.0, 1.0)

        raise ValueError(f"Unknown scenario type: {scenario}")

    def _scenario_loss_from_inputs(
        self,
        task,
        ir_s,
        vis_s,
        mask,
        boxes,
        classes,
        use_generator_loss=False,
    ):
        if task == "visual":
            loss_fn = (
                self.model._generator_loss
                if use_generator_loss
                else self.model._discriminator_loss
            )
            return loss_fn(ir_s, vis_s, mask).mean()
        return self.model._detection_loss(ir_s, vis_s, boxes, classes).mean()

    def _robust_upper_select_tail(self, task, ir, vis, mask, boxes, classes):
        scenarios = [
            "nominal",
            "ir_low_contrast",
            "ir_stripenoise",
            "vis_gaussiannoise",
            "vis_fog",
        ]
        modules = [self.model.enhance_net]
        if task == "visual":
            modules.append(self.model.discriminator)
        else:
            modules.append(self.model.denoise_net)
        buffer_state = self._snapshot_buffers(modules)

        scenario_cache = {}
        scores = []
        with torch.no_grad():
            for scenario in scenarios:
                ir_scenario, vis_scenario = self._scenario_inputs(
                    ir, vis, scenario, self.ir_attn, self.vis_attn
                )
                score = self._scenario_loss_from_inputs(
                    task,
                    ir_scenario,
                    vis_scenario,
                    mask,
                    boxes,
                    classes,
                    use_generator_loss=(task == "visual"),
                )
                scenario_cache[scenario] = (ir_scenario, vis_scenario)
                scores.append(float(score.item()))
                self._restore_buffers(modules, buffer_state)

        order = sorted(
            range(len(scenarios)),
            key=lambda index: scores[index],
            reverse=True,
        )
        tail_scenarios = [
            scenarios[index]
            for index in order[: min(self.robust_topk, len(scenarios))]
        ]
        graph_losses = [
            self._scenario_loss_from_inputs(
                task,
                scenario_cache[scenario][0],
                scenario_cache[scenario][1],
                mask,
                boxes,
                classes,
                use_generator_loss=(task == "visual"),
            )
            for scenario in tail_scenarios
        ]

        values = torch.tensor(
            scores, device=ir.device, dtype=graph_losses[0].dtype
        )
        alpha = min(max(self.cvar_alpha, 0.0), 0.99)
        rank_index = min(len(scores) - 1, int(alpha * len(scores)))
        quantile = torch.sort(values.detach())[0][rank_index]
        tau = max(self.cvar_tau, 1e-5)
        robust = quantile + torch.stack(
            [
                tau * F.softplus((loss - quantile) / tau)
                for loss in graph_losses
            ]
        ).mean() / max(1.0 - alpha, 1e-3)
        spread = max(scores) - min(scores)
        return robust, float(spread), len(tail_scenarios), tail_scenarios, scores

    def _conflict_project(self, a, b):
        dot = self._dot(a, b)
        if dot.item() >= 0:
            return self._clone(a), self._clone(b)
        denom_b = self._dot(b, b).clamp_min(1e-12)
        denom_a = self._dot(a, a).clamp_min(1e-12)
        a2 = [x - (dot / denom_b) * y for x, y in zip(a, b)]
        b2 = [y - (dot / denom_a) * x for x, y in zip(a, b)]
        return self._clone(a2), self._clone(b2)

    def _project_lower_feasible(self, g, bv, bp, stats_v, stats_p):
        if not g or not bv or not bp:
            return g

        dtype = g[0].dtype
        eps = torch.finfo(dtype).eps
        r1 = self._dot(g, bv)
        r2 = self._dot(g, bp)
        d11 = self._dot(bv, bv)
        d22 = self._dot(bp, bp)
        d12 = self._dot(bv, bp)

        g_norm = self._norm(g)
        bv_norm = self._norm(bv)
        bp_norm = self._norm(bp)
        tol1 = eps * (g_norm * bv_norm + self.fd_eps)
        tol2 = eps * (g_norm * bp_norm + self.fd_eps)

        candidates = []
        if r1.item() >= -tol1.item() and r2.item() >= -tol2.item():
            candidates.append((self._clone(g), 0))

        if r1.item() < -tol1.item() and d11.item() > self.fd_eps:
            lam1 = -r1 / d11
            candidate = [x + lam1 * y for x, y in zip(g, bv)]
            if (
                lam1.item() >= 0.0
                and self._dot(candidate, bp).item() >= -tol2.item()
            ):
                candidates.append((candidate, 1))

        if r2.item() < -tol2.item() and d22.item() > self.fd_eps:
            lam2 = -r2 / d22
            candidate = [x + lam2 * y for x, y in zip(g, bp)]
            if (
                lam2.item() >= 0.0
                and self._dot(candidate, bv).item() >= -tol1.item()
            ):
                candidates.append((candidate, 2))

        if r1.item() < -tol1.item() and r2.item() < -tol2.item():
            gram = torch.stack(
                [
                    torch.stack([d11, d12]),
                    torch.stack([d12, d22]),
                ]
            )
            rhs = -torch.stack([r1, r2])
            try:
                lam = torch.linalg.pinv(gram) @ rhs
                if lam[0].item() >= 0.0 and lam[1].item() >= 0.0:
                    candidate = [
                        x + lam[0] * y + lam[1] * z
                        for x, y, z in zip(g, bv, bp)
                    ]
                    if (
                        self._dot(candidate, bv).item() >= -tol1.item()
                        and self._dot(candidate, bp).item() >= -tol2.item()
                    ):
                        candidates.append((candidate, 2))
            except RuntimeError:
                pass

        if not candidates:
            direction = bv if r1.item() <= r2.item() else bp
            denom = self._dot(direction, direction).clamp_min(self.fd_eps)
            lam = -torch.minimum(r1, r2) / denom
            candidates.append(
                (
                    [
                        x + lam * y
                        for x, y in zip(g, direction)
                    ],
                    1,
                )
            )

        def correction_norm(item):
            candidate = item[0]
            return float(
                self._norm(
                    [x - y for x, y in zip(candidate, g)]
                ).item()
            )

        best, active = min(candidates, key=correction_norm)
        before = float(min(r1.item(), r2.item()))
        after = float(
            min(self._dot(best, bv).item(), self._dot(best, bp).item())
        )
        stats_v.lower_feasible_before = before
        stats_p.lower_feasible_before = before
        stats_v.lower_feasible_after = after
        stats_p.lower_feasible_after = after
        stats_v.lower_projection_active = int(active)
        stats_p.lower_projection_active = int(active)
        return best

    def _aggregate(self, gv, gp, stats_v, stats_p):
        raw_nv = self._norm(gv) + self.fd_eps
        raw_np = self._norm(gp) + self.fd_eps
        raw_cosine = (
            self._dot(gv, gp) / (raw_nv * raw_np)
        ).clamp(-1.0, 1.0)

        gv2, gp2 = self._conflict_project(gv, gp)
        inverse_norms = torch.tensor(
            [
                1.0 / max(stats_v.hyper_norm, self.fd_eps),
                1.0 / max(stats_p.hyper_norm, self.fd_eps),
            ],
            device=gv[0].device,
            dtype=gv[0].dtype,
        )
        weights = F.softmax(inverse_norms, dim=0)
        combined = [
            weights[0] * first + weights[1] * second
            for first, second in zip(gv2, gp2)
        ]

        combined_norm = self._norm(combined) + self.fd_eps
        nv = self._norm(gv2) + self.fd_eps
        np_ = self._norm(gp2) + self.fd_eps
        cosine_before = (
            self._dot(gv2, gp2) / (nv * np_)
        ).clamp(-1.0, 1.0)
        cosine_after = (
            0.5
            * (
                self._dot(gv2, combined) / (nv * combined_norm)
                + self._dot(gp2, combined) / (np_ * combined_norm)
            )
        ).clamp(-1.0, 1.0)

        stats_v.weight = float(weights[0].item())
        stats_p.weight = float(weights[1].item())
        stats_v.cosine_before = float(cosine_before.item())
        stats_p.cosine_before = float(cosine_before.item())
        stats_v.cosine_after = float(cosine_after.item())
        stats_p.cosine_after = float(cosine_after.item())
        stats_v.raw_cosine_vp = float(raw_cosine.item())
        stats_p.raw_cosine_vp = float(raw_cosine.item())
        return combined

    def step(
        self,
        batch_imgs_ir,
        batch_imgs_vis,
        mask,
        batch_boxes,
        batch_classes,
        imgs_ir,
        image_vis,
        mask_,
        boxes,
        classes,
        lr_new=1e-4,
    ):
        self.optimizer.param_groups[0]["lr"] = lr_new
        self.optimizer.zero_grad(set_to_none=True)

        upper_visual, spread_v, tail_v, _, _ = self._robust_upper_select_tail(
            "visual", imgs_ir, image_vis, mask_, None, None
        )

        def lower_visual_fn():
            return self.model._discriminator_loss(
                batch_imgs_ir, batch_imgs_vis, mask
            ).mean()

        gv, stats_v, bv = self._task_hypergradient(
            upper_visual,
            lower_visual_fn,
            self.model.discriminator,
        )
        stats_v.upper_loss = float(upper_visual.detach().item())
        stats_v.robust_spread = float(spread_v)
        stats_v.tail_count = int(tail_v)
        del upper_visual

        upper_perception, spread_p, tail_p, _, _ = (
            self._robust_upper_select_tail(
                "perception", imgs_ir, image_vis, None, boxes, classes
            )
        )

        def lower_perception_fn():
            return self.model._detection_loss(
                batch_imgs_ir,
                batch_imgs_vis,
                batch_boxes,
                batch_classes,
            ).mean()

        gp, stats_p, bp = self._task_hypergradient(
            upper_perception,
            lower_perception_fn,
            self.model.denoise_net,
        )
        stats_p.upper_loss = float(upper_perception.detach().item())
        stats_p.robust_spread = float(spread_p)
        stats_p.tail_count = int(tail_p)
        del upper_perception

        combined = self._aggregate(gv, gp, stats_v, stats_p)
        combined = self._project_lower_feasible(
            combined, bv, bp, stats_v, stats_p
        )
        omega = self._params(self.model.enhance_net)
        for parameter, gradient in zip(omega, combined):
            parameter.grad = gradient.detach().clone()

        if self.hypergrad_clip > 0:
            torch.nn.utils.clip_grad_norm_(omega, self.hypergrad_clip)
        self.optimizer.step()

        self.last_stats = {"visual": stats_v, "perception": stats_p}
        return self.last_stats

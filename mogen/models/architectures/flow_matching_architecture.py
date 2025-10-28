import math
import time

import torch

from .base_architecture import BaseArchitecture
from ..builder import (
    ARCHITECTURES,
    build_submodule,
    build_loss,
)


@ARCHITECTURES.register_module()
class MotionFlowMatching(BaseArchitecture):
    """Flow-matching architecture built on top of the existing transformer."""

    def __init__(
        self,
        model=None,
        loss_recon=None,
        loss_weight=None,
        flow=None,
        init_cfg=None,
        index_num=None,
        motion_crop=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.model = build_submodule(model)
        self.loss_recon = build_loss(loss_recon)
        self.loss_weight = loss_weight or {}
        self.index_num = index_num
        self.motion_start = motion_crop[0]
        self.motion_end = motion_crop[1]

        flow = flow or {}
        solver = flow.get("solver", {})
        self.time_scale = float(flow.get("time_scale", 1000.0))
        self.flow_kind = flow.get("kind", "rectified").lower()
        if self.flow_kind not in {"rectified", "linear", "naive", "meanflow"}:
            raise ValueError(f"Unsupported flow kind '{self.flow_kind}'.")

        # Alias "naive" to the original linear path for backwards compatibility.
        if self.flow_kind == "naive":
            self.flow_kind = "linear"

        path_cfg = flow.get("path", {})
        self.path_type = path_cfg.get("type", "rectified").lower()
        if self.path_type not in {"rectified", "linear"}:
            raise ValueError(f"Unsupported flow path type '{self.path_type}'.")
        self.path_exponent = float(path_cfg.get("exponent", 2.0))
        self.path_epsilon = float(path_cfg.get("epsilon", 1e-4))
        if self.path_type == "rectified" and self.path_epsilon <= 0:
            raise ValueError("rectified paths require path.epsilon > 0 to avoid undefined derivatives.")

        self.alpha_dot_floor = float(flow.get("alpha_dot_floor", 1e-12))

        objective_cfg = flow.get("objective", {})
        motion_weight = float(objective_cfg.get("motion", 1.0))
        velocity_weight = float(objective_cfg.get("velocity", 1.0))
        total = motion_weight + velocity_weight
        if total <= 0:
            raise ValueError("At least one of motion or velocity objective weights must be positive.")
        self.motion_loss_weight = motion_weight / total
        self.velocity_loss_weight = velocity_weight / total

        variance_cfg = flow.get("variance", {})
        self.variance_type = variance_cfg.get("type", "alpha").lower()
        if self.variance_type not in {"alpha", "linear"}:
            raise ValueError(f"Unsupported variance schedule '{self.variance_type}'.")
        self.variance_floor = float(variance_cfg.get("floor", 1e-4))

        self.default_solver_steps = int(solver.get("num_steps", 50))
        self.default_solver_type = solver.get("type", "euler").lower()

        # cache stickman geometry for default fallbacks during inference
        if hasattr(self.model, "multistick_encoder"):
            stick_encoder = self.model.multistick_encoder.stick_encoder
            # stick encoder consumes (point_len * 2) features per polyline
            point_len = stick_encoder.proj_in.in_features // 2
            self._stickman_shape = (self.index_num, 6, point_len, 2)
        else:
            self._stickman_shape = None
        self.last_solver_steps = None

    # ------------------------------------------------------------------
    # helper utilities
    # ------------------------------------------------------------------
    def _get_default_stickman(self, batch_size, device):
        if self._stickman_shape is None:
            return None
        index_num, num_segments, point_len, point_dim = self._stickman_shape
        return torch.zeros(
            batch_size,
            index_num,
            num_segments,
            point_len,
            point_dim,
            device=device,
        )

    # ------------------------------------------------------------------
    # device helpers
    # ------------------------------------------------------------------
    def others_cuda(self):
        device = next(self.model.parameters()).device
        self.model.to(device)

    # ------------------------------------------------------------------
    # path helpers
    # ------------------------------------------------------------------
    def _scale_time(self, t):
        return t * self.time_scale

    def _alpha(self, t):
        if self.path_type == "rectified":
            return t.pow(self.path_exponent)
        return t

    def _alpha_dot(self, t):
        if self.path_type == "rectified":
            exponent = self.path_exponent
            if math.isclose(exponent, 1.0):
                return torch.ones_like(t)
            return exponent * t.clamp_min(0).pow(exponent - 1)
        return torch.ones_like(t)

    def _clamp_t(self, t):
        if self.path_epsilon <= 0:
            return t
        return t.clamp(self.path_epsilon, 1.0 - self.path_epsilon)

    def _sigma(self, alpha):
        if self.variance_type == "alpha":
            variance = 1.0 - alpha.pow(2)
        else:
            variance = 1.0 - alpha
        variance = torch.clamp(variance, min=self.variance_floor)
        return variance.sqrt()

    # ------------------------------------------------------------------
    # core forward
    # ------------------------------------------------------------------
    def forward(self, **kwargs):
        return_loss = kwargs.pop("return_loss", True)
        motion = kwargs["motion"].float()
        motion_mask = kwargs["motion_mask"].float()
        motion_length = kwargs["motion_length"]
        specified_idx = kwargs.get("specified_idx", None)
        stickman_tracks = kwargs.get("stickman_tracks", None)
        sample_idx = kwargs.get("sample_idx", None)
        clip_feat = kwargs.get("clip_feat", None)
        text = [meta.get("text", "") for meta in kwargs.get("motion_metas", [{} for _ in range(motion.shape[0])])]

        device = motion.device
        B, T, dim_pose = motion.shape

        if specified_idx is None:
            specified_idx = torch.zeros(B, self.index_num, dtype=torch.long, device=device)
        else:
            specified_idx = specified_idx.to(device=device, dtype=torch.long)

        if stickman_tracks is None:
            default_track = self._get_default_stickman(B, device)
            if default_track is None:
                raise ValueError("stickman_tracks must be provided for this configuration.")
            stickman_tracks = default_track
        else:
            stickman_tracks = stickman_tracks.to(device=device, dtype=torch.float32)

        if sample_idx is not None:
            sample_idx = sample_idx.to(device)

        if clip_feat is not None:
            clip_feat = clip_feat.to(device)

        if self.training and return_loss:
            return self._forward_train(
                motion=motion,
                motion_mask=motion_mask,
                motion_length=motion_length,
                specified_idx=specified_idx,
                stickman_tracks=stickman_tracks,
                sample_idx=sample_idx,
                clip_feat=clip_feat,
                text=text,
            )
        else:
            return self._forward_test(
                motion=motion,
                motion_mask=motion_mask,
                motion_length=motion_length,
                specified_idx=specified_idx,
                stickman_tracks=stickman_tracks,
                sample_idx=sample_idx,
                clip_feat=clip_feat,
                text=text,
                inference_kwargs=kwargs.get("inference_kwargs", {}),
                raw_kwargs=kwargs,
            )

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def _forward_train(
        self,
        motion,
        motion_mask,
        motion_length,
        specified_idx,
        stickman_tracks,
        sample_idx,
        clip_feat,
        text,
    ):
        device = motion.device
        B = motion.shape[0]

        base_sample = torch.randn_like(motion)
        t = self._clamp_t(torch.rand(B, device=device))
        alpha = self._alpha(t)
        alpha_dot = self._alpha_dot(t)

        if self.flow_kind == "meanflow":
            sigma = self._sigma(alpha)[:, None, None]
            xt = alpha[:, None, None] * motion + sigma * base_sample
        else:
            xt = (1.0 - alpha)[:, None, None] * base_sample + alpha[:, None, None] * motion

        scaled_t = self._scale_time(t)

        pred_velocity, index, p_batch, stick_mask = self.model(
            xt,
            scaled_t,
            motion_mask=motion_mask,
            motion_length=motion_length,
            text=text,
            specified_idx=specified_idx,
            stickman_tracks=stickman_tracks,
            sample_idx=sample_idx,
            clip_feat=clip_feat,
        )

        if self.flow_kind == "meanflow":
            pred_motion = xt + pred_velocity
            target = alpha[:, None, None] * motion
            target_velocity = target - xt
        else:
            alpha_dot = alpha_dot[:, None, None]
            denom = torch.where(
                alpha_dot.abs() < self.alpha_dot_floor,
                alpha_dot.new_full(alpha_dot.shape, self.alpha_dot_floor),
                alpha_dot,
            )
            pred_motion = base_sample + pred_velocity / denom
            target = motion
            target_velocity = alpha_dot * (motion - base_sample)

        all_loss = 0.0
        loss = {}
        all_loss_batch = self.loss_recon(pred_motion, target, reduction_override="none")
        velocity_loss_batch = self.loss_recon(pred_velocity, target_velocity, reduction_override="none")
        specified_motion = motion[torch.arange(B, device=device)[:, None], specified_idx, :]

        loss_item = ["text_loss", "both_loss", "stick_loss", "none_loss"]
        assert len(p_batch) == len(loss_item)
        all_batch = sum(p_batch)
        start = 0
        for i, batch in enumerate(p_batch):
            if batch == 0:
                loss[loss_item[i]] = torch.tensor(0.0, device=device)
                continue
            segment_slice = slice(start, start + batch)
            if loss_item[i] in {"both_loss", "stick_loss"}:
                spec_m = specified_motion[segment_slice, :, self.motion_start : self.motion_end]
                pred_m = pred_motion[segment_slice, :, self.motion_start : self.motion_end]
                spec_loss = []
                for j in range(self.index_num):
                    diff = (spec_m[:, j, None] - pred_m).pow(2).mean(-1)
                    w_m = index[segment_slice, :, j]
                    stick_cond_mask = stick_mask[segment_slice, j, 0]
                    spec_loss.append(((diff * w_m).sum(-1) * stick_cond_mask).sum() / batch)
                loss[f"identity_{loss_item[i]}"] = sum(spec_loss) / len(spec_loss)
                all_loss = all_loss + batch / all_batch * loss[f"identity_{loss_item[i]}"] * self.loss_weight.get("motion_w", 1.0)
            motion_term = None
            if self.motion_loss_weight > 0:
                motion_term = (
                    all_loss_batch[segment_slice].mean(-1) * motion_mask[segment_slice]
                ).sum() / motion_mask[segment_slice].sum()
                loss[f"{loss_item[i]}_motion"] = motion_term

            velocity_term = None
            if self.velocity_loss_weight > 0:
                velocity_term = (
                    velocity_loss_batch[segment_slice].mean(-1) * motion_mask[segment_slice]
                ).sum() / motion_mask[segment_slice].sum()
                loss[f"{loss_item[i]}_velocity"] = velocity_term

            masked_mse = 0.0
            if motion_term is not None:
                masked_mse += self.motion_loss_weight * motion_term
            if velocity_term is not None:
                masked_mse += self.velocity_loss_weight * velocity_term

            loss[loss_item[i]] = masked_mse
            all_loss = all_loss + batch / all_batch * masked_mse
            start += batch
        loss["all_loss"] = all_loss
        return loss

    # ------------------------------------------------------------------
    # inference utilities
    # ------------------------------------------------------------------
    def _flow_solver(self, x, motion_mask, motion_length, specified_idx, stickman_tracks, sample_idx, clip_feat, text, inference_kwargs):
        device = x.device
        solver_cfg = inference_kwargs.get("solver", {})
        method = solver_cfg.get("type", self.default_solver_type)
        steps = int(solver_cfg.get("num_steps", self.default_solver_steps))
        if steps <= 0:
            raise ValueError("Number of solver steps must be positive.")
        if method != "euler":
            raise NotImplementedError(f"Solver '{method}' is not supported yet.")

        if motion_mask.dim() == 2:
            mask = motion_mask.unsqueeze(-1)
        else:
            mask = motion_mask

        precomputed = self.model.get_precompute_condition(
            text=text,
            stickman_tracks=stickman_tracks,
            motion_length=motion_length,
            device=device,
            sample_idx=sample_idx,
            clip_feat=clip_feat,
        )
        self.last_solver_steps = steps
        step_times = torch.linspace(0.0, 1.0, steps + 1, device=device)
        for i in range(steps):
            t_cur = step_times[i]
            t_next = step_times[i + 1]
            dt = t_next - t_cur
            t_batch = torch.full((x.shape[0],), self._scale_time(t_cur), device=device)
            velocity, index = self.model(
                x,
                t_batch,
                motion_mask=motion_mask,
                motion_length=motion_length,
                xf_out=precomputed["xf_out"],
                stick_encoder=precomputed["stick_encoder"],
                sample_idx=sample_idx,
            )
            if mask is not None:
                velocity = velocity * mask
            x = x + dt * velocity
        final_t = torch.full((x.shape[0],), self.time_scale, device=device)
        _, index = self.model(
            x,
            final_t,
            motion_mask=motion_mask,
            motion_length=motion_length,
            xf_out=precomputed["xf_out"],
            stick_encoder=precomputed["stick_encoder"],
            sample_idx=sample_idx,
        )
        return x, index, steps

    def _forward_test(
        self,
        motion,
        motion_mask,
        motion_length,
        specified_idx,
        stickman_tracks,
        sample_idx,
        clip_feat,
        text,
        inference_kwargs,
        raw_kwargs,
    ):
        device = motion.device
        B, T, D = motion.shape

        base = inference_kwargs.get("base", "standard_normal")
        if base != "standard_normal":
            raise NotImplementedError(f"Base distribution '{base}' is not supported.")
        x = torch.randn(B, T, D, device=device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()

        pred_motion, pred_index, steps = self._flow_solver(
            x,
            motion_mask,
            motion_length,
            specified_idx,
            stickman_tracks,
            sample_idx,
            clip_feat,
            text,
            inference_kwargs,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        per_sample_time = elapsed / max(B, 1)

        if getattr(self.model, "post_process", None) is not None:
            pred_motion = self.model.post_process(pred_motion)

        results = dict(raw_kwargs)
        results.pop("return_loss", None)
        results["pred_motion"] = pred_motion
        results["pred_index"] = pred_index
        results["solver_steps"] = pred_motion.new_full((B,), float(steps))
        results["inference_time"] = pred_motion.new_full((B,), per_sample_time)
        return self.split_results(results)


@ARCHITECTURES.register_module()
class MotionNaiveFlowMatching(MotionFlowMatching):
    """Linear (a.k.a. naive) flow-matching variant."""

    def __init__(self, flow=None, **kwargs):
        flow_cfg = dict(flow or {})
        flow_cfg.setdefault("kind", "linear")
        flow_cfg.setdefault("path", {}).setdefault("type", "linear")
        super().__init__(flow=flow_cfg, **kwargs)


@ARCHITECTURES.register_module()
class MotionRectifiedFlowMatching(MotionFlowMatching):
    """Rectified flow-matching variant with velocity supervision."""

    def __init__(self, flow=None, **kwargs):
        flow_cfg = dict(flow or {})
        flow_cfg.setdefault("kind", "rectified")
        flow_cfg.setdefault("path", {}).setdefault("type", "rectified")
        super().__init__(flow=flow_cfg, **kwargs)


@ARCHITECTURES.register_module()
class MotionMeanFlowMatching(MotionFlowMatching):
    """MeanFlow-style supervision with optional rectified time reparameterisation."""

    def __init__(self, flow=None, **kwargs):
        flow_cfg = dict(flow or {})
        flow_cfg.setdefault("kind", "meanflow")
        super().__init__(flow=flow_cfg, **kwargs)

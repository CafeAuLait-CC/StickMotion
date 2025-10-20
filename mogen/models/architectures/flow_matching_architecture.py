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
        t = torch.rand(B, device=device)
        # Linear conditional flow path: x_t = (1 - t) * x0 + t * x1
        xt = (1.0 - t)[:, None, None] * base_sample + t[:, None, None] * motion
        scaled_t = t * self.time_scale

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
        pred = base_sample + pred_velocity
        target = motion

        all_loss = 0.0
        loss = {}
        all_loss_batch = self.loss_recon(pred, target, reduction_override="none")
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
                pred_m = pred[segment_slice, :, self.motion_start : self.motion_end]
                spec_loss = []
                for j in range(self.index_num):
                    diff = (spec_m[:, j, None] - pred_m).pow(2).mean(-1)
                    w_m = index[segment_slice, :, j]
                    stick_cond_mask = stick_mask[segment_slice, j, 0]
                    spec_loss.append(((diff * w_m).sum(-1) * stick_cond_mask).sum() / batch)
                loss[f"identity_{loss_item[i]}"] = sum(spec_loss) / len(spec_loss)
                all_loss = all_loss + batch / all_batch * loss[f"identity_{loss_item[i]}"] * self.loss_weight.get("motion_w", 1.0)
            masked_mse = (
                all_loss_batch[segment_slice].mean(-1) * motion_mask[segment_slice]
            ).sum() / motion_mask[segment_slice].sum()
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
        step_times = torch.linspace(0.0, 1.0, steps + 1, device=device)
        for i in range(steps):
            t_cur = step_times[i]
            t_next = step_times[i + 1]
            dt = t_next - t_cur
            t_batch = torch.full((x.shape[0],), t_cur * self.time_scale, device=device)
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
        return x, index

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

        pred_motion, pred_index = self._flow_solver(
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

        if getattr(self.model, "post_process", None) is not None:
            pred_motion = self.model.post_process(pred_motion)

        results = dict(raw_kwargs)
        results.pop("return_loss", None)
        results["pred_motion"] = pred_motion
        results["pred_index"] = pred_index
        return self.split_results(results)

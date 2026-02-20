# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

import os
import logging
from pathlib import Path
from contextlib import nullcontext
from typing import Callable

# Reduce noisy OpenCV logging if present
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# --- DDP environment (torchrun / Slurm) ---
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

# Make dir for logs
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Define logger (avoid multiple ranks clobbering the same file)
log_file = log_dir / (f"train_p9_rank{RANK}.log" if WORLD_SIZE > 1 else "train_p9.log")
logging.basicConfig(
    level=(logging.INFO if RANK == 0 else logging.WARNING),
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode="w"),
    ],
)
logger = logging.getLogger(__name__)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import argparse

from utils import AverageMeter, Loss_MRAE, Loss_PSNR, Loss_RMSE
from mstpp.mstpp import MST_Plus_Plus
from data_carrier import load_east_kaz, load_sri_lanka, load_weedy_rice, DataCarrier



class TransferLearning:
    def __init__(self, args):
        # Args
        self.cluster: bool = bool(args.cluster)
        self.seed: int = int(getattr(args, "seed", 42))

        # DDP state (torchrun sets these env vars)
        self.rank: int = RANK
        self.world_size: int = WORLD_SIZE
        self.local_rank: int = LOCAL_RANK

        # Enable DDP only when running on cluster AND torchrun/Slurm requested multiple processes
        self.ddp: bool = self.cluster and torch.cuda.is_available() and self.world_size > 1

        # Choose device
        if torch.cuda.is_available():
            if self.ddp:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.is_main_process: bool = (self.rank == 0)

        # Init process group (must happen before using DistributedSampler / DDP)
        if self.ddp and not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
            dist.barrier()

        if self.is_main_process:
            logger.info(
                f"[Device] {self.device} | cluster={self.cluster} | ddp={self.ddp} | world_size={self.world_size}"
            )

        # Stage 1 config
        self.stage1_data_path = Path(args.stage1_data_path)
        self.stage1_data_type = args.stage1_data_type
        self.stage1_non_resize_picture = args.stage1_non_resize
        self.stage_1_epochs = args.stage1_epochs
        self.stage_1_lr = args.stage1_lr

        # Stage 2 config
        self.stage2_data_path = Path(args.stage2_data_path)
        self.stage2_data_type = args.stage2_data_type
        self.stage2_non_resize_picture = args.stage2_non_resize
        self.stage2_model = args.stage2_model
        self.stage_2_epochs = args.stage2_epochs
        self.stage_2_lr = args.stage2_lr

        # Stage 3 config
        self.stage3_data_path = Path(args.stage3_data_path)
        self.stage3_data_type = args.stage3_data_type
        self.stage3_non_resize_picture = args.stage3_non_resize
        self.stage3_model = args.stage3_model
        self.stage_3_epochs = args.stage3_epochs
        self.stage_3_lr = args.stage3_lr

        # Model details
        self.model = None
        self.dataset: DataCarrier = None
        self.optimizer: torch.optim.Adam = None
        self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR = None

        # Loss functions
        self.criterion_mrae: Loss_MRAE = None
        self.criterion_rmse: Loss_RMSE = None
        self.criterion_psnr: Loss_PSNR = None

        # Logging
        self.logWriter = SummaryWriter(log_dir="logs/transfer_learning/") if self.is_main_process else None

        # DataLoader args dependent on if on cluster or not.
        base_workers = 15 if self.cluster else 1
        if self.ddp and base_workers > 1:
            base_workers = max(1, base_workers // self.world_size)

        self.loader_kwargs = dict(
            num_workers=base_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(base_workers > 0),
        )
        if base_workers > 0:
            self.loader_kwargs["prefetch_factor"] = 4 if self.cluster else 2

    # ---------------- DDP helpers ----------------
    def cleanup(self):
        if self.ddp and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def _unwrap_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _maybe_wrap_ddp(self):
        if self.ddp and not isinstance(self.model, DDP):
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
            )
            if self.is_main_process:
                logger.info(f"[DDP] Wrapped model in DistributedDataParallel on local_rank={self.local_rank}")

    def _autocast(self):
        # AMP only on CUDA
        if self.device.type == "cuda":
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _broadcast_object(self, obj):
        if not self.ddp:
            return obj
        buf = [obj] if self.is_main_process else [None]
        dist.broadcast_object_list(buf, src=0)
        return buf[0]

    def _broadcast_path(self, path):
        return self._broadcast_object(path)

    def _set_sampler_epoch(self, dataloader, epoch: int):
        # Ensure different shuffles each epoch under DistributedSampler
        sampler = getattr(dataloader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

    def _make_dataloaders(self, train_dataset, val_dataset, train_bs_global=12, val_bs_global=4):
        # Keep *global* batch sizes similar between 1-GPU and DDP by dividing across ranks.
        train_bs = train_bs_global
        val_bs = val_bs_global
        if self.ddp and self.world_size > 0:
            train_bs = max(1, train_bs_global // self.world_size)
            val_bs = max(1, val_bs_global // self.world_size)

        if self.is_main_process:
            eff_train = train_bs * (self.world_size if self.ddp else 1)
            eff_val = val_bs * (self.world_size if self.ddp else 1)
            logger.info(f"[DataLoader] per-rank train_bs={train_bs} (effective={eff_train}), val_bs={val_bs} (effective={eff_val})")

        train_sampler = (
            DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            if self.ddp
            else None
        )
        val_sampler = (
            DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            if (self.ddp and val_dataset is not None)
            else None
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=train_bs,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            **self.loader_kwargs,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=val_bs,
                shuffle=False,
                sampler=val_sampler,
                **self.loader_kwargs,
            )

        return train_loader, val_loader
    def _load_pretrained(self, checkpoint_path, learning_rate):
        # Build base model (unwrapped) to keep checkpoint keys simple
        self.model = MST_Plus_Plus(in_channels=3, out_channels=4, n_feat=4, stage=3)
        self.model = self.model.to(self.device, memory_format=torch.channels_last)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "model" in checkpoint:
            pretrained_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            pretrained_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            pretrained_dict = checkpoint["state_dict"]
        else:
            pretrained_dict = checkpoint

        # Strip a leading 'module.' if present
        cleaned = {}
        for k, v in pretrained_dict.items():
            nk = k[len("module."):] if k.startswith("module.") else k
            cleaned[nk] = v

        model_state = self.model.state_dict()
        filtered = {}
        skipped = []

        for k, v in cleaned.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)

        model_state.update(filtered)
        self.model.load_state_dict(model_state, strict=True)

        if self.is_main_process:
            logger.info(
                f"[Pretrained loading] Loaded {len(filtered)} params, skipped {len(skipped)} params (missing/incompatible)."
            )

        # Wrap in DDP (if enabled) before creating the optimizer
        self._maybe_wrap_ddp()
        self.setup_optimizer(learning_rate)

    def _get_loader_function(self, data_type: str) -> Callable[[Path], tuple[list[Path], list[Path]]]:
        match data_type:
            case "Sri-Lanka":
                return load_sri_lanka
            case "Kazakhstan":
                return load_east_kaz
            case "Weedy-Rice":
                return load_weedy_rice
            case _:
                raise ValueError(f"Unknown dataset type: {data_type}")

    def load_mstpp(self, learning_rate, total_steps):
        self.model = MST_Plus_Plus(in_channels=3, out_channels=4, n_feat=4, stage=3)
        self.model = self.model.to(self.device, memory_format=torch.channels_last)

        self._maybe_wrap_ddp()

        self.setup_optimizer(learning_rate)
        self.setup_scheduler(total_steps, eta_min=1e-6)

    def set_requires_grad(self, module, requires_grad: bool):
        """Recursively set requires_grad for all parameters in a module."""
        for p in module.parameters():
            p.requires_grad = requires_grad

    def save_model(self, save_path, stage_name, epoch=None):
        # Only rank 0 writes checkpoints to disk
        if not self.is_main_process:
            return None

        os.makedirs(save_path, exist_ok=True)

        if epoch is not None:
            filename = f"{stage_name}_epoch_{epoch}.pth"
        else:
            filename = f"{stage_name}_final.pth"

        full_path = os.path.join(save_path, filename)

        checkpoint = {
            "model_state_dict": self._unwrap_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "stage": stage_name,
            "epoch": epoch,
        }

        torch.save(checkpoint, full_path)
        logger.info(f"[Saved] Model saved to {full_path}")
        return full_path

    def freeze_all_except_decoder(self):
        """
        Freeze all layers except the decoder (conv_out layer).
        This is used in Stage 2 of transfer learning.
        """
        m = self._unwrap_model()

        # Freeze conv_in and body
        self.set_requires_grad(m.conv_in, False)
        self.set_requires_grad(m.body, False)

        # Unfreeze conv_out (decoder)
        self.set_requires_grad(m.conv_out, True)

        trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in m.parameters())

        if self.is_main_process:
            logger.info(f"[Freeze] Decoder only: {trainable_params}/{total_params} parameters trainable")

    def unfreeze_all(self):
        m = self._unwrap_model()
        self.set_requires_grad(m, True)
        trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        if self.is_main_process:
            logger.info(f"[Unfreeze] All layers: {trainable_params} parameters trainable")

    def setup_optimizer(self, learning_rate):
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        logger.info(f"[Optimizer] Adam optimizer set with lr={learning_rate}")

    def setup_scheduler(self, total_steps, eta_min):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=total_steps,
            eta_min=eta_min
        )
        logger.info(f"[Scheduler] CosineAnnealinLR schedular set with eta_min={eta_min}")

    def setup_criterion(self):
        self.criterion_mrae = Loss_MRAE()
        self.criterion_rmse = Loss_RMSE()
        self.criterion_psnr = Loss_PSNR()

        # These losses are modules; move to CUDA if needed
        if self.device.type == "cuda":
            self.criterion_mrae = self.criterion_mrae.to(self.device)
            self.criterion_rmse = self.criterion_rmse.to(self.device)
            self.criterion_psnr = self.criterion_psnr.to(self.device)

    def stage_reset(self):
        self.scheduler = None
        self.dataset = None

    def train_epoch(self, dataloader):
        self.model.train(mode=True)

        scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))
        loss_sum = 0.0
        n_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            # Empty cache if on desktop (CUDA only)
            if (not self.cluster) and self.device.type == "cuda":
                torch.cuda.empty_cache()

            inputs = batch["rgb"].to(self.device, non_blocking=True)
            targets = batch["ms"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast():
                outputs = self.model(inputs)
                loss = self.criterion_mrae(outputs, targets)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            bs = int(inputs.shape[0])
            loss_sum += float(loss.item()) * bs
            n_samples += bs

            if self.scheduler is not None:
                self.scheduler.step()

            if self.is_main_process and (batch_idx % 10 == 0):
                logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.6f}")

        # Reduce across ranks so logs reflect the global average loss
        if self.ddp:
            t = torch.tensor([loss_sum, float(n_samples)], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            loss_sum = float(t[0].item())
            n_samples = int(t[1].item())

        return loss_sum / max(1, n_samples)

    def validate_epoch(self, dataloader):
        self.model.eval()

        mrae_sum = 0.0
        rmse_sum = 0.0
        psnr_sum = 0.0
        n_samples = 0

        with torch.no_grad():
            with self._autocast():
                for batch in dataloader:
                    inputs = batch["rgb"].to(self.device, non_blocking=True)
                    targets = batch["ms"].to(self.device, non_blocking=True)

                    outputs = self.model(inputs)
                    loss_mrae = self.criterion_mrae(outputs, targets)
                    loss_rmse = self.criterion_rmse(outputs, targets)
                    loss_psnr = self.criterion_psnr(outputs, targets)

                    bs = int(inputs.shape[0])
                    mrae_sum += float(loss_mrae.item()) * bs
                    rmse_sum += float(loss_rmse.item()) * bs
                    psnr_sum += float(loss_psnr.item()) * bs
                    n_samples += bs

        # Reduce across ranks
        if self.ddp:
            t = torch.tensor([mrae_sum, rmse_sum, psnr_sum, float(n_samples)], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            mrae_sum = float(t[0].item())
            rmse_sum = float(t[1].item())
            psnr_sum = float(t[2].item())
            n_samples = int(t[3].item())

        denom = max(1, n_samples)
        return (mrae_sum / denom), (rmse_sum / denom), (psnr_sum / denom)

    def load_dataset(self, root_dir: Path, loader: Callable[[Path], tuple[list[Path], list[Path]]], non_resize_picture=False):
        self.dataset = DataCarrier(root_dir, loader, resize=(not non_resize_picture)) # non_resize_picture is default False
        logger.info(f"[Loaded] Dataset loaded with {len(self.dataset)} samples.")

    def train_from_scratch(self, train_dataloader, epochs, val_dataloader=None, save_dir="checkpoints", save_every=10):
        if self.is_main_process:
            logger.info("=" * 60)
            logger.info("STAGE 1: Train from scratch")
            logger.info("=" * 60)

        # Set model mode train
        self.model.train(mode=True)

        # Unfreeze all layers
        self.unfreeze_all()

        best_val_loss = float("inf")
        best_model_path = None

        for epoch in range(epochs):
            self._set_sampler_epoch(train_dataloader, epoch)

            if self.is_main_process:
                logger.info(f"[Stage 1] Epoch {epoch + 1}/{epochs}")

            train_loss = self.train_epoch(train_dataloader)

            if self.is_main_process:
                logger.info(f"[Stage 1] Epoch {epoch + 1} - Train Loss: {train_loss:.6f}")
                if self.scheduler is not None:
                    logger.info(f"[Stage 1] Scheduler LR: {self.scheduler.get_last_lr()}")

            # Validation
            if val_dataloader is not None:
                mrae_loss, rmse_loss, psnr_loss = self.validate_epoch(val_dataloader)

                if self.is_main_process:
                    logger.info(
                        f"[Stage 1] Epoch {epoch + 1} - MRAE: {mrae_loss:.6f}, RMSE: {rmse_loss:.6f}, PSNR: {psnr_loss:.6f}"
                    )

                if self.logWriter is not None:
                    self.logWriter.add_scalar("Stage1/Train_Loss", train_loss, epoch)
                    self.logWriter.add_scalar("Stage1/MRAE_Loss", mrae_loss, epoch)
                    self.logWriter.add_scalar("Stage1/RMSE_Loss", rmse_loss, epoch)
                    self.logWriter.add_scalar("Stage1/PSNR_Loss", psnr_loss, epoch)

                if self.is_main_process and (mrae_loss < best_val_loss):
                    best_val_loss = mrae_loss
                    best_model_path = self.save_model(save_dir, "stage1_best")
                    logger.info(f"[Stage 1] New best model saved! Val MRAE: {mrae_loss:.6f}")

            else:
                if self.is_main_process:
                    logger.info("[Stage 1] There is no validate dataloader")
                if self.logWriter is not None:
                    self.logWriter.add_scalar("Stage1/Train_Loss", train_loss, epoch)

            # Save checkpoint periodically (rank 0 only)
            if self.is_main_process and ((epoch + 1) % save_every == 0):
                self.save_model(save_dir, "stage1", epoch + 1)

        final_path = self.save_model(save_dir, "stage1") if self.is_main_process else None

        if self.is_main_process:
            if val_dataloader is not None:
                logger.info(f"[Stage 1] Training completed. Best MRAE: {best_val_loss:.6f}")
                logger.info(f"[Stage 1] Best model: {best_model_path}")
            else:
                logger.info("[Stage 1] Training completed. (No validate dataloader)")
            logger.info(f"[Stage 1] Final model: {final_path}")

        ret = best_model_path if best_model_path else final_path
        ret = self._broadcast_path(ret)
        return ret

    def run_stage_2(self, train_dataloader, epochs, val_dataloader=None, save_dir="checkpoints", save_every=10):
        if self.is_main_process:
            logger.info("=" * 60)
            logger.info("STAGE 2: Decoder Training (Frozen Encoder)")
            logger.info("=" * 60)

        self.model.train(mode=True)
        self.freeze_all_except_decoder()

        best_val_loss = float("inf")
        best_model_path = None

        for epoch in range(epochs):
            self._set_sampler_epoch(train_dataloader, epoch)

            if self.is_main_process:
                logger.info(f"[Stage 2] Epoch {epoch + 1}/{epochs}")

            train_loss = self.train_epoch(train_dataloader)

            if self.is_main_process:
                logger.info(f"[Stage 2] Epoch {epoch + 1} - Train Loss: {train_loss:.6f}")

            if val_dataloader is not None:
                mrae_loss, rmse_loss, psnr_loss = self.validate_epoch(val_dataloader)

                if self.is_main_process:
                    logger.info(
                        f"[Stage 2] Epoch {epoch + 1} - MRAE: {mrae_loss:.6f}, RMSE: {rmse_loss:.6f}, PSNR: {psnr_loss:.6f}"
                    )

                if self.logWriter is not None:
                    self.logWriter.add_scalar("Stage2/Train_Loss", train_loss, epoch)
                    self.logWriter.add_scalar("Stage2/MRAE_Loss", mrae_loss, epoch)
                    self.logWriter.add_scalar("Stage2/RMSE_Loss", rmse_loss, epoch)
                    self.logWriter.add_scalar("Stage2/PSNR_Loss", psnr_loss, epoch)

                if self.is_main_process and (mrae_loss < best_val_loss):
                    best_val_loss = mrae_loss
                    best_model_path = self.save_model(save_dir, "stage2_best")
                    logger.info(f"[Stage 2] New best model saved! Val MRAE: {mrae_loss:.6f}")

            else:
                if self.is_main_process:
                    logger.info("[Stage 2] There is no validate dataloader")
                if self.logWriter is not None:
                    self.logWriter.add_scalar("Stage2/Train_Loss", train_loss, epoch)

            if self.is_main_process and ((epoch + 1) % save_every == 0):
                self.save_model(save_dir, "stage2", epoch + 1)

        final_path = self.save_model(save_dir, "stage2") if self.is_main_process else None

        if self.is_main_process:
            if val_dataloader is not None:
                logger.info(f"[Stage 2] Training completed. Best MRAE: {best_val_loss:.6f}")
                logger.info(f"[Stage 2] Best model: {best_model_path}")
            else:
                logger.info("[Stage 2] Training completed. (No validate dataloader)")
            logger.info(f"[Stage 2] Final model: {final_path}")

        ret = best_model_path if best_model_path else final_path
        ret = self._broadcast_path(ret)
        return ret

    def run_stage_3(self, train_dataloader, epochs, val_dataloader=None, save_dir="checkpoints", save_every=10):
        if self.is_main_process:
            logger.info("=" * 60)
            logger.info("STAGE 3: Full Model Fine-tuning (All Layers Unfrozen)")
            logger.info("=" * 60)

        self.model.train(mode=True)
        self.unfreeze_all()

        best_val_loss = float("inf")
        best_model_path = None

        for epoch in range(epochs):
            self._set_sampler_epoch(train_dataloader, epoch)

            if self.is_main_process:
                logger.info(f"[Stage 3] Epoch {epoch + 1}/{epochs}")

            train_loss = self.train_epoch(train_dataloader)

            if self.is_main_process:
                logger.info(f"[Stage 3] Epoch {epoch + 1} - Train Loss: {train_loss:.6f}")

            if val_dataloader is not None:
                mrae_loss, rmse_loss, psnr_loss = self.validate_epoch(val_dataloader)

                if self.is_main_process:
                    logger.info(
                        f"[Stage 3] Epoch {epoch + 1} - MRAE: {mrae_loss:.6f}, RMSE: {rmse_loss:.6f}, PSNR: {psnr_loss:.6f}"
                    )

                if self.logWriter is not None:
                    self.logWriter.add_scalar("Stage3/Train_Loss", train_loss, epoch)
                    self.logWriter.add_scalar("Stage3/MRAE_Loss", mrae_loss, epoch)
                    self.logWriter.add_scalar("Stage3/RMSE_Loss", rmse_loss, epoch)
                    self.logWriter.add_scalar("Stage3/PSNR_Loss", psnr_loss, epoch)

                if self.is_main_process and (mrae_loss < best_val_loss):
                    best_val_loss = mrae_loss
                    best_model_path = self.save_model(save_dir, "stage3_best")
                    logger.info(f"[Stage 3] New best model saved! Val MRAE: {mrae_loss:.6f}")

            else:
                if self.is_main_process:
                    logger.info("[Stage 3] There is no validate dataloader")
                if self.logWriter is not None:
                    self.logWriter.add_scalar("Stage3/Train_Loss", train_loss, epoch)

            if self.is_main_process and ((epoch + 1) % save_every == 0):
                self.save_model(save_dir, "stage3", epoch + 1)

        final_path = self.save_model(save_dir, "stage3") if self.is_main_process else None

        if self.is_main_process:
            if val_dataloader is not None:
                logger.info(f"[Stage 3] Training completed. Best MRAE: {best_val_loss:.6f}")
                logger.info(f"[Stage 3] Best model: {best_model_path}")
            else:
                logger.info("[Stage 3] Training completed. (No validate dataloader)")
            logger.info(f"[Stage 3] Final model: {final_path}")

        ret = best_model_path if best_model_path else final_path
        ret = self._broadcast_path(ret)
        return ret

    def run_full_pipeline(self,
                          stage1_epochs=100,
                          stage1_lr=1e-5,
                          stage2_epochs=100,
                          stage2_lr=1e-5,
                          stage3_epochs=100,
                          stage3_lr=1e-7,
                          save_dir="checkpoints"):
        logger.info("="*70)
        logger.info(" STAGED TRANSFER LEARNING PIPELINE")
        logger.info("="*70)

        results = {}

        # --------------------------------
        # Stage 1 train model from scratch
        # --------------------------------
        loader = self._get_loader_function(self.stage1_data_type)
        self.load_dataset(root_dir=self.stage1_data_path, loader=loader, non_resize_picture=self.stage1_non_resize_picture)

        total_steps = stage1_epochs * len(self.dataset)
        self.load_mstpp(learning_rate=stage1_lr, total_steps=total_steps)

        total_len = len(self.dataset)
        val_len = max(1, int(0.1 * total_len))
        train_len = total_len - val_len
        gen = torch.Generator().manual_seed(self.seed + 1)
        train_dataset, val_dataset = random_split(self.dataset, [train_len, val_len], generator=gen)

        # Prepare your dataloaders (DDP uses DistributedSampler)
        train_dataloader, val_dataloader = self._make_dataloaders(
            train_dataset,
            val_dataset,
            train_bs_global=72, # TRAIN BATCH SIZE
            val_bs_global=16    # VAL   BATCH SIZE
        )

        results['stage1'] = self.train_from_scratch(
            train_dataloader=train_dataloader,
            epochs=stage1_epochs,
            val_dataloader=val_dataloader,
            save_dir=save_dir
        )

        # --------------------------------
        # Stage 2: Decoder training
        # --------------------------------
        self.stage_reset()
        loader = self._get_loader_function(self.stage2_data_type)
        self.load_dataset(root_dir=self.stage2_data_path, loader=loader, non_resize_picture=self.stage2_non_resize_picture)

        model_path = results["stage1"] if self.stage2_model is None else self.stage2_model
        self._load_pretrained(model_path, learning_rate=stage2_lr)

        total_len = len(self.dataset)
        val_len = max(1, int(0.1 * total_len))
        train_len = total_len - val_len
        gen = torch.Generator().manual_seed(self.seed + 2)
        train_dataset, val_dataset = random_split(self.dataset, [train_len, val_len], generator=gen)

        # Prepare your dataloaders (DDP uses DistributedSampler)
        train_dataloader, val_dataloader = self._make_dataloaders(
            train_dataset,
            val_dataset,
            train_bs_global=64, # TRAIN BATCH SIZE
            val_bs_global=8     # VAL   BATCH SIZE
        )

        results['stage2'] = self.run_stage_2(
            train_dataloader=train_dataloader,
            epochs=stage2_epochs,
            val_dataloader=val_dataloader,
            save_dir=save_dir
        )


        # --------------------------------
        # Stage 3: Full fine-tuning
        # --------------------------------
        self.stage_reset()
        loader = self._get_loader_function(self.stage3_data_type)
        self.load_dataset(root_dir=self.stage3_data_path, loader=loader, non_resize_picture=self.stage3_non_resize_picture)
        
        model_path = results["stage2"] if self.stage3_model is None else self.stage3_model
        self._load_pretrained(model_path, learning_rate=stage3_lr)

        total_len = len(self.dataset)
        val_len = max(1, int(0.1 * total_len))
        train_len = total_len - val_len
        gen = torch.Generator().manual_seed(self.seed + 3)
        train_dataset, val_dataset = random_split(self.dataset, [train_len, val_len], generator=gen)

        # Prepare your dataloaders (DDP uses DistributedSampler)
        train_dataloader, val_dataloader = self._make_dataloaders(
            train_dataset,
            val_dataset,
            train_bs_global=64, # TRAIN BATCH SIZE
            val_bs_global=8     # VAL   BATCH SIZE
        )

        results['stage3'] = self.run_stage_3(
            train_dataloader=train_dataloader,
            epochs=stage3_epochs,
            val_dataloader=val_dataloader,
            save_dir=save_dir
        )

        logger.info("="*70)
        logger.info(" PIPELINE COMPLETED")
        logger.info("="*70)
        logger.info("Saved models:")
        for stage, path in results.items():
            logger.info(f"  {stage}: {path}")

        return results


# Usage example
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get data paths.")
    parser.add_argument("--cluster", type=bool, help="Script intended for cluster, default=False", action=argparse.BooleanOptionalAction)

    parser.add_argument("--seed", type=int, default=42, help="Random seed (used for deterministic train/val split across DDP ranks)")

    parser.add_argument("--stage1_data_path", default="data/East-Kaza")
    parser.add_argument("--stage1_data_type", help="Which dataset", default="Kazakhstan")
    parser.add_argument("--stage1_non_resize", type=bool, help="Use non-resized pictures, default=False", action=argparse.BooleanOptionalAction)
    parser.add_argument("--stage1_epochs", type=int, help="Number of epochs for stage 1 (train from scratch). To skip set epochs to '0'", default=300)
    parser.add_argument("--stage1_lr", type=float, help="Learning rate for stage 1 (train from scratch)", default=4e-4)
    
    parser.add_argument("--stage2_data_path", default="data/")
    parser.add_argument("--stage2_data_type", help="Which dataset", default="Kazakhstan")
    parser.add_argument("--stage2_non_resize", type=bool, help="Use non-resized pictures, default=False", action=argparse.BooleanOptionalAction)
    parser.add_argument("--stage2_model", type=str, help="(Optional) Path to model for stage 2", default=None)
    parser.add_argument("--stage2_epochs", type=int, help="Number of epochs for stage 2. To skip set epochs to '0'", default=0)
    parser.add_argument("--stage2_lr", type=float, help="Learning rate for stage 2", default=1e-5)

    parser.add_argument("--stage3_data_path", default="data/")
    parser.add_argument("--stage3_data_type", help="Which dataset Sri-Lanka or Kazakhstan, default=Sri-Lanka", default="Weedy-Rice")
    parser.add_argument("--stage3_non_resize", type=bool, help="Use non-resized pictures, default=False", action=argparse.BooleanOptionalAction)
    parser.add_argument("--stage3_model", type=str, help="(Optional) Path to model for stage 3", default=None)
    parser.add_argument("--stage3_epochs", type=int, help="Number of epochs for stage 3. To skip set epochs to '0'", default=0)
    parser.add_argument("--stage3_lr", type=float, help="Learning rate for stage 3", default=1e-7)

    # Initialize the transfer learning pipeline
    tl = TransferLearning(parser.parse_args())

    # Setup criterion
    tl.setup_criterion()

    # Run the full 3-stage pipeline with validation
    results = tl.run_full_pipeline(
        stage1_epochs=tl.stage_1_epochs,
        stage1_lr=tl.stage_1_lr,
        stage2_epochs=tl.stage_2_epochs, # Train decoder for 50 epochs
        stage2_lr=tl.stage_2_lr,         # Medium-high learning rate for stage 2
        stage3_epochs=tl.stage_3_epochs, # Fine-tune all layers for 30 epochs
        stage3_lr=tl.stage_3_lr,         # Low learning rate for stage 3
        save_dir="checkpoints"
    )

    # Clean up DDP (if enabled) and close TensorBoard writer
    if tl.logWriter is not None:
        tl.logWriter.close()
    tl.cleanup()

from contextlib import nullcontext
from dataclasses import dataclass, replace
import os
import logging
from pathlib import Path
import re
from typing import Any, Callable, Literal, Optional, get_args

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Reduce noisy OpenCV logging if present
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# --- DDP environment (torchrun / Slurm) ---
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

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
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.amp.grad_scaler import GradScaler

import argparse

from utils import Loss_MRAE, Loss_PSNR, Loss_RMSE, Loss_NDVI, Loss_NDRE
from mstpp.mstpp import MST_Plus_Plus
from data_carrier import load_east_kaz, load_sri_lanka, load_weedy_rice, DataCarrier

DatasetType = Literal["Sri-Lanka", "Kazakhstan", "Weedy-Rice"]

@dataclass(frozen=True)
class DatasetConfig:
    data_path: Path
    data_type: DatasetType
    non_resize: bool
    ddp: bool
    is_main_process: bool
    device_type: str
    cluster: bool
    seed: int

@dataclass(frozen=True)
class DDPContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    is_main: bool

    def barrier(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.barrier()

    def broadcast_object(self, obj):
        if not self.enabled:
            return obj
        buf = [obj] if self.is_main else [None]
        dist.broadcast_object_list(buf, src=0)
        return buf[0]

def _get_loader_function(data_type: DatasetType) -> Callable[[Path], tuple[list[Path], list[Path]]]:
    match data_type:
        case "Sri-Lanka":
            return load_sri_lanka
        case "Kazakhstan":
            return load_east_kaz
        case "Weedy-Rice":
            return load_weedy_rice
        case _:
            raise ValueError(f"Unknown dataset type: {data_type}")

def _get_dataset(root_dir: Path, loader: Callable[[Path], tuple[list[Path], list[Path]]], non_resize_picture=False) -> DataCarrier:
    dataset = DataCarrier(root_dir, loader, resize=(not non_resize_picture))  # non_resize_picture is default False
    logger.info(f"[Loaded] Dataset loaded with {len(dataset)} samples.")
    return dataset

def _make_dataloaders(
    train_dataset,
    val_dataset,
    ddp: bool,
    world_size: int,
    rank: int,
    is_main_process: bool,
    train_bs_global=12,
    val_bs_global=4,
    loader_kwargs: Optional[dict[str, Any]]=None
    ) -> tuple[DataLoader[Any], DataLoader[Any]]:
        # Keep *global* batch sizes similar between 1-GPU and DDP by dividing across ranks.
        if loader_kwargs is None:
            loader_kwargs = {}

        train_bs = train_bs_global
        val_bs = val_bs_global
        if ddp and world_size > 0:
            train_bs = max(1, train_bs_global // world_size)
            val_bs = max(1, val_bs_global // world_size)

        if is_main_process:
            eff_train = train_bs * (world_size if ddp else 1)
            eff_val = val_bs * (world_size if ddp else 1)
            logger.info(f"[DataLoader] per-rank train_bs={train_bs} (effective={eff_train}), val_bs={val_bs} (effective={eff_val})")

        train_sampler = (
            DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            if ddp
            else None
        )
        val_sampler = (
            DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            if (ddp and val_dataset is not None)
            else None
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=train_bs,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=val_bs,
            shuffle=False,
            sampler=val_sampler,
            **loader_kwargs,
        )

        return train_loader, val_loader

def load_datasets(
    train_batch_size: int,
    val_batch_size: int,
    config: DatasetConfig,
    ddp: DDPContext
) -> tuple[DataLoader[Any], DataLoader[Any], int]:
    dataset: DataCarrier = _get_dataset(
        config.data_path,
        _get_loader_function(config.data_type),
        config.non_resize
    )

    total_len = len(dataset)
    val_len = max(1, int(0.1 * total_len))
    train_len = total_len - val_len
    gen = torch.Generator().manual_seed(config.seed + 1)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_len, val_len],
        generator=gen
    )

    # DataLoader args dependent on if on cluster or not.
    base_workers = 15 if config.cluster else 1
    if ddp.enabled and base_workers > 1:
        base_workers = max(1, base_workers // ddp.world_size)

    train, val = _make_dataloaders(
        train_dataset,
        val_dataset,
        ddp.enabled,
        ddp.world_size,
        ddp.rank,
        ddp.is_main,
        train_bs_global=train_batch_size,
        val_bs_global=val_batch_size,
        loader_kwargs= dict(
            num_workers=base_workers,
            pin_memory=(config.device_type == "cuda"),
            persistent_workers=(base_workers > 0),
            prefetch_factor = 4 if config.cluster else 2
        )
    )

    return train, val, total_len

def load_mstpp(device: torch.device) -> MST_Plus_Plus:
    model = MST_Plus_Plus(in_channels=3, out_channels=4, n_feat=4, stage=3)
    return model.to(device, memory_format=torch.channels_last)

def load_pretrained_model(device: torch.device, checkpoint_path: Path, is_main_process: bool) -> MST_Plus_Plus:
    # Build base model (unwrapped) to keep checkpoint keys simple
    model = load_mstpp(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" in checkpoint:
        pretrained_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        pretrained_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        pretrained_dict = checkpoint["state_dict"]
    else:
        pretrained_dict = checkpoint

    # Strip a leading "module" if present
    cleaned = {}
    for k, v in pretrained_dict.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        cleaned[nk] = v

    model_state = model.state_dict()
    filtered = {}
    skipped = []

    for k, v in cleaned.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    model_state.update(filtered)
    model.load_state_dict(model_state, strict=True)

    if is_main_process:
        logger.info(
            f"[Pretrained loading] Loaded {len(filtered)} params, skipped {len(skipped)} params (missing/incompatible)."
        )

    return model

def get_optimizer(model: nn.Module, learning_rate: float) -> Adam:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params=trainable_params,
        lr=learning_rate,
        betas=(0.9, 0.999)
    )
    logger.info(
        f"[Optimizer] Adam with {len(trainable_params)} trainable parameter tensors | lr={learning_rate}"
    )

    return optimizer

def get_scheduler(optimizer: Adam, total_steps: int, eta_min: float) -> CosineAnnealingLR:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=total_steps,
        eta_min=eta_min
    )
    logger.info(
        f"[Scheduler] CosineAnnealinLR schedular set with eta_min={eta_min}"
    )

    return scheduler

def set_requires_grad(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad

def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model

def maybe_wrap_ddp(model: nn.Module, ctx: DDPContext) -> DDP | nn.Module:
    if ctx.enabled and not isinstance(model, DDP):
        model = DDP(
            model,
            device_ids=[ctx.local_rank],
            output_device=ctx.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )
        if ctx.is_main:
            logger.info(f"[DDP] Wrapped model in DistributedDataParallel on local_rank={ctx.local_rank}")

    return model

def unfreeze_all(model: nn.Module, is_main_process: bool):
    m = unwrap_model(model)
    set_requires_grad(m, True)
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    if is_main_process:
        logger.info(f"[Unfreeze] All layers: {trainable_params} parameters trainable")

def set_sampler_epoch(dataloader: DataLoader, epoch: int):
    # Ensure different shuffles each epoch under DistributedSampler
    sampler = getattr(dataloader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)

def auto_cast(device_type: str) -> torch.autocast | nullcontext[None]:
    if device_type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

def save_model(model: nn.Module, optimizer: Adam, save_path: Path, model_name: str, epoch: int) -> Path:
    os.makedirs(save_path, exist_ok=True)

    filename = f"{model_name}.pth"

    match = re.search(r"stage(\d+)", model_name)
    stage_number: int = int(match.group(1)) if match else -1 # if not found -1 for error

    full_path = os.path.join(save_path, filename)

    checkpoint = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "stage": stage_number,
        "epoch": epoch,
    }

    torch.save(checkpoint, full_path)
    logger.info(f"[Saved] Model saved to {full_path}")

    return Path(full_path)

def init_ddp(cluster: bool) -> DDPContext:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    ddp_enabled = cluster and torch.cuda.is_available() and world_size > 1

    # device selection
    if torch.cuda.is_available():
        if ddp_enabled:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # init process group once
    if ddp_enabled and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()

    return DDPContext(
        enabled=ddp_enabled,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        is_main=(rank == 0),
    )

def cleanup_ddp(ctx: DDPContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

@dataclass(frozen=True)
class StageConfig:
    stage_id: str
    run_stage: bool
    model_name: str
    pretrained_model_path: Path | None
    dir_name: str
    data_path: Path
    data_type: DatasetType
    batch_size_train: int
    batch_size_val: int
    epochs: int
    learning_rate: float
    non_resize:bool
    use_scheduler: bool
    loss_mrae_w: float
    loss_ndvi_w: float
    loss_ndre_w: float
    seed: int
    cluster: bool
    band_idx_red: int
    band_idx_rededge: int
    band_idx_nir: int

class Stage:
    def __init__(self, config: StageConfig, ddp: DDPContext):
        # Stage identification
        self.stage_id: str = config.stage_id

        # Model config
        self.model_name: str = config.model_name
        self.save_dir: Path  = Path(f"checkpoints/{config.dir_name}")

        # Training config
        self.data_path: Path        = config.data_path
        self.data_type: DatasetType = config.data_type
        self.epochs: int            = config.epochs
        self.lr: float              = config.learning_rate
        self.non_resize: bool       = config.non_resize
        self.use_scheduler: bool    = config.use_scheduler

        # Composite loss weights
        self.loss_mrae_w: float = config.loss_mrae_w
        self.loss_ndvi_w: float = config.loss_ndvi_w
        self.loss_ndre_w: float = config.loss_ndre_w

        # Other args
        self.seed: int     = config.seed
        self.cluster: bool = config.cluster

        # DDP state (torchrun sets these env vars)
        self.rank       = ddp.rank
        self.world_size = ddp.world_size
        self.local_rank = ddp.local_rank

        # Choose device
        self.ddp = ddp
        self.device = ddp.device
        self.is_main_process: bool = ddp.is_main

        if self.is_main_process:
            logger.info(
                f"[Device] {self.device} | cluster={self.cluster} | ddp={self.ddp.enabled} | world_size={self.world_size}"
            )

        # Band indices for VI losses
        # Default assumes ms channels: [G, R, RE, NIR]
        self.band_idx_red     = config.band_idx_red
        self.band_idx_rededge = config.band_idx_rededge
        self.band_idx_nir     = config.band_idx_nir


        # Load model
        self.model: nn.Module = load_pretrained_model(self.device, config.pretrained_model_path, self.is_main_process) if config.pretrained_model_path else load_mstpp(self.device)
        self.model = maybe_wrap_ddp(self.model, self.ddp)


        # Load dataset
        self.train_dataloader, self.val_dataloader, total_datapoints = load_datasets(
            train_batch_size=config.batch_size_train,
            val_batch_size=config.batch_size_val,
            config=DatasetConfig(
                self.data_path,
                self.data_type,
                self.non_resize,
                self.ddp.enabled,
                self.is_main_process,
                self.device.type,
                self.cluster,
                self.seed
            ),
            ddp=self.ddp
        )


        # Setup optimizer and scheduler
        self.total_steps: int = self.epochs * total_datapoints
        
        self.configure_trainables()
        
        self._rebuild_optimizer()


        # Setup loss functions
        self.criterion_mrae = Loss_MRAE()
        self.criterion_rmse = Loss_RMSE()
        self.criterion_psnr = Loss_PSNR()

        self.criterion_ndvi = Loss_NDVI(
            self.band_idx_nir,
            self.band_idx_red
        )

        self.criterion_ndre = Loss_NDRE(
            self.band_idx_nir,
            self.band_idx_rededge
        )

        # These losses are modules; move to selected device
        self.criterion_mrae = self.criterion_mrae.to(self.device)
        self.criterion_rmse = self.criterion_rmse.to(self.device)
        self.criterion_psnr = self.criterion_psnr.to(self.device)
        self.criterion_ndvi = self.criterion_ndvi.to(self.device)
        self.criterion_ndre = self.criterion_ndre.to(self.device)

        if self.is_main_process:
            logger.info(
                "[Loss] Composite training loss = "
                f"{self.loss_mrae_w}*MRAE + {self.loss_ndvi_w}*NDVI + {self.loss_ndre_w}*NDRE | "
                f"Band indices: RED={self.band_idx_red}, RED_EDGE={self.band_idx_rededge}, NIR={self.band_idx_nir}"
            )

    def train(self):
        raise NotImplementedError()

    def _train_epoch(self) -> tuple[float, float, float, float]:
        self.model.train()
        scaler = GradScaler(enabled=(self.device.type == "cuda"))

        total_sum: float = 0.0
        mrae_sum: float  = 0.0
        ndvi_sum: float  = 0.0
        ndre_sum: float  = 0.0
        n_samples: int   = 0

        for batch_idx, batch in enumerate(self.train_dataloader):
            # Empty cache if on desktop (CUDA only)
            if (not self.cluster) and self.device.type == "cuda":
                torch.cuda.empty_cache()

            inputs = batch["rgb"].to(self.device, non_blocking=True)
            targets = batch["ms"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with auto_cast(self.device.type):
                outputs = self.model(inputs)
                loss_total, loss_mrae, loss_ndvi, loss_ndre = self._compute_composite_loss(outputs, targets)

            skip_scheduler = False
            if scaler.is_enabled():
                scaler.scale(loss_total).backward()

                prev_scale = scaler.get_scale()
                scaler.step(self.optimizer)
                scaler.update()

                skip_scheduler = scaler.get_scale() < prev_scale
            else:
                loss_total.backward()
                self.optimizer.step()

            bs = int(inputs.shape[0])

            total_sum += float(loss_total.item()) * bs
            mrae_sum += float(loss_mrae.item()) * bs
            ndvi_sum += float(loss_ndvi.item()) * bs
            ndre_sum += float(loss_ndre.item()) * bs
            n_samples += bs

            if self.scheduler is not None and not skip_scheduler:
                self.scheduler.step()

            if self.is_main_process and (batch_idx % 10 == 0):
                logger.info(
                    f"Batch {batch_idx + 1}/{len(self.train_dataloader)} | "
                    f"Total: {loss_total.item():.6f} | "
                    f"MRAE: {loss_mrae.item():.6f} | "
                    f"NDVI: {loss_ndvi.item():.6f} | "
                    f"NDRE: {loss_ndre.item():.6f}"
                )

        # Reduce across ranks so logs reflect the global average loss
        if self.ddp.enabled:
            t = torch.tensor([total_sum, mrae_sum, ndvi_sum, ndre_sum, float(n_samples)], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_sum = float(t[0].item())
            mrae_sum = float(t[1].item())
            ndvi_sum = float(t[2].item())
            ndre_sum = float(t[3].item())
            n_samples = int(t[4].item())

        denom = max(1, n_samples)
        return (
            total_sum / denom,  # train_total
            mrae_sum / denom,   # train_mrae
            ndvi_sum / denom,   # train_ndvi
            ndre_sum / denom    # train_ndre
        )

    def validate_epoch(self) -> tuple[float, float, float, float, float, float]:
        self.model.eval()

        total_sum: float = 0.0
        mrae_sum: float  = 0.0
        rmse_sum: float  = 0.0
        psnr_sum: float  = 0.0
        ndvi_sum: float  = 0.0
        ndre_sum: float  = 0.0
        n_samples: int   = 0

        with torch.no_grad():
            with auto_cast(self.device.type):
                for batch in self.val_dataloader:
                    inputs = batch["rgb"].to(self.device, non_blocking=True)
                    targets = batch["ms"].to(self.device, non_blocking=True)

                    outputs: Tensor = self.model(inputs)

                    loss_mrae: Tensor = self.criterion_mrae(outputs, targets)
                    loss_rmse: Tensor = self.criterion_rmse(outputs, targets)
                    loss_psnr: Tensor = self.criterion_psnr(outputs, targets)

                    zero: Tensor = outputs.new_zeros(())
                    loss_ndvi: Tensor = self.criterion_ndvi(outputs, targets) if self.loss_ndvi_w != 0.0 else zero
                    loss_ndre: Tensor = self.criterion_ndre(outputs, targets) if self.loss_ndre_w != 0.0 else zero

                    loss_total: Tensor = (
                        self.loss_mrae_w * loss_mrae +
                        self.loss_ndvi_w * loss_ndvi +
                        self.loss_ndre_w * loss_ndre
                    )

                    bs = int(inputs.shape[0])
                    total_sum += float(loss_total.item()) * bs
                    mrae_sum += float(loss_mrae.item()) * bs
                    rmse_sum += float(loss_rmse.item()) * bs
                    psnr_sum += float(loss_psnr.item()) * bs
                    ndvi_sum += float(loss_ndvi.item()) * bs
                    ndre_sum += float(loss_ndre.item()) * bs
                    n_samples += bs

        # Reduce across ranks
        if self.ddp.enabled:
            t = torch.tensor(
                [total_sum, mrae_sum, rmse_sum, psnr_sum, ndvi_sum, ndre_sum, float(n_samples)],
                device=self.device
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_sum = float(t[0].item())
            mrae_sum = float(t[1].item())
            rmse_sum = float(t[2].item())
            psnr_sum = float(t[3].item())
            ndvi_sum = float(t[4].item())
            ndre_sum = float(t[5].item())
            n_samples = int(t[6].item())

        denom = max(1, n_samples)
        return (
            total_sum / denom,  # val_total
            mrae_sum / denom,
            rmse_sum / denom,
            psnr_sum / denom,
            ndvi_sum / denom,
            ndre_sum / denom,
        )

    def _compute_composite_loss(self, outputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        loss_mrae: Tensor = self.criterion_mrae(outputs, targets)

        zero = outputs.new_zeros(())
        loss_ndvi: Tensor = self.criterion_ndvi(outputs, targets) if self.loss_ndvi_w != 0.0 else zero
        loss_ndre: Tensor = self.criterion_ndre(outputs, targets) if self.loss_ndre_w != 0.0 else zero

        total_loss = (
            self.loss_mrae_w * loss_mrae +
            self.loss_ndvi_w * loss_ndvi +
            self.loss_ndre_w * loss_ndre
        )

        return total_loss, loss_mrae, loss_ndvi, loss_ndre

    def configure_trainables(self) -> None:
        raise NotImplementedError()
    
    def _rebuild_optimizer(self) -> None:
        self.optimizer = get_optimizer(self.model, self.lr)
        self.scheduler = (
            get_scheduler(self.optimizer, self.total_steps, 1e-6)
            if self.use_scheduler
            else None
        )

class Stage1(Stage):
    def train(self):
        if self.is_main_process:
            logger.info("=" * 27)
            logger.info("STAGE 1: Train from scratch")
            logger.info("=" * 27)

        # Set model to train mode
        self.model.train()

        best_val_loss = float("inf")
        best_model_path: Path | None = None

        for epoch in range(self.epochs):
            set_sampler_epoch(self.train_dataloader, epoch)

            if self.is_main_process:
                logger.info(f"[Stage 1] Epoch {epoch + 1}/{self.epochs}")

            # Training
            train_loss, train_mrae, train_ndvi, train_ndre = self._train_epoch()

            if self.is_main_process:
                logger.info(
                    f"[Stage 1] Epoch {epoch + 1} - "
                    f"Train Total: {train_loss:.6f}, MRAE: {train_mrae:.6f}, "
                    f"NDVI: {train_ndvi:.6f}, NDRE: {train_ndre:.6f}"
                )
                if self.scheduler is not None:
                    logger.info(f"[Stage 1] Scheduler LR: {self.scheduler.get_last_lr()}")

            # Validation
            if self.val_dataloader is not None:
                val_total, mrae_loss, rmse_loss, psnr_loss, ndvi_loss, ndre_loss = self.validate_epoch()

                if self.is_main_process:
                    logger.info(
                        f"[Stage 1] Epoch {epoch + 1} - "
                        f"Val Total: {val_total:.6f}, MRAE: {mrae_loss:.6f}, RMSE: {rmse_loss:.6f}, "
                        f"PSNR: {psnr_loss:.6f}, NDVI: {ndvi_loss:.6f}, NDRE: {ndre_loss:.6f}"
                    )

                if self.is_main_process and (val_total < best_val_loss):
                    best_val_loss = val_total
                    best_model_path = save_model(
                        self.model,
                        self.optimizer,
                        self.save_dir,
                        f"{self.model_name}_stage1_best",
                        epoch
                    )
                    logger.info(f"[Stage 1] New best model saved! Val composite loss: {val_total:.6f}")

            else:
                if self.is_main_process:
                    logger.info("[Stage 1] There is no validate dataloader")

            # Save checkpoint periodically (rank 0 only)
            if self.is_main_process and ((epoch + 1) % 10 == 0): # save every 10 epoch
                save_model(
                    self.model,
                    self.optimizer,
                    self.save_dir / "all-models",
                    f"{self.model_name}_stage1_epoch_{epoch}",
                    epoch
                )

        final_path: Path | None = save_model(
            self.model,
            self.optimizer,
            self.save_dir / "all-models",
            f"{self.model_name}_stage1_final",
            self.epochs
        ) if self.is_main_process else None

        if self.is_main_process:
            if self.val_dataloader is not None:
                logger.info(f"[Stage 1] Training completed. Best composite val loss: {best_val_loss:.6f}")
                logger.info(f"[Stage 1] Best model: {best_model_path}")
            else:
                logger.info("[Stage 1] Training completed. (No validate dataloader)")
            logger.info(f"[Stage 1] Final model: {final_path}")

        ret = best_model_path if best_model_path else final_path
        ret = self.ddp.broadcast_object(ret)
        return ret

    def configure_trainables(self) -> None:
        unfreeze_all(self.model, self.is_main_process)

class Stage2(Stage):
    def train(self):
        if self.is_main_process:
            logger.info("=" * 27)
            logger.info("STAGE 2: Train decoder only")
            logger.info("=" * 27)

        # Set model to train mode
        self.model.train()

        best_val_loss = float("inf")
        best_model_path = None

        for epoch in range(self.epochs):
            set_sampler_epoch(self.train_dataloader, epoch)

            if self.is_main_process:
                logger.info(f"[Stage 2] Epoch {epoch + 1}/{self.epochs}")

            # Training
            train_loss, train_mrae, train_ndvi, train_ndre = self._train_epoch()

            if self.is_main_process:
                logger.info(
                    f"[Stage 2] Epoch {epoch + 1} - "
                    f"Train Total: {train_loss:.6f}, MRAE: {train_mrae:.6f}, "
                    f"NDVI: {train_ndvi:.6f}, NDRE: {train_ndre:.6f}"
                )
                if self.scheduler is not None:
                    logger.info(f"[Stage 2] Scheduler LR: {self.scheduler.get_last_lr()}")

            # Validation
            if self.val_dataloader is not None:
                val_total, mrae_loss, rmse_loss, psnr_loss, ndvi_loss, ndre_loss = self.validate_epoch()

                if self.is_main_process:
                    logger.info(
                        f"[Stage 2] Epoch {epoch + 1} - "
                        f"Val Total: {val_total:.6f}, MRAE: {mrae_loss:.6f}, RMSE: {rmse_loss:.6f}, "
                        f"PSNR: {psnr_loss:.6f}, NDVI: {ndvi_loss:.6f}, NDRE: {ndre_loss:.6f}"
                    )

                if self.is_main_process and (val_total < best_val_loss):
                    best_val_loss = val_total
                    best_model_path = save_model(
                        self.model,
                        self.optimizer,
                        self.save_dir,
                        f"{self.model_name}_stage2_best",
                        epoch
                    )
                    logger.info(f"[Stage 2] New best model saved! Val composite loss: {val_total:.6f}")

            else:
                if self.is_main_process:
                    logger.info("[Stage 2] There is no validate dataloader")

            # Save checkpoint periodically (rank 0 only)
            if self.is_main_process and ((epoch + 1) % 10 == 0): # save every 10 epoch
                save_model(
                    self.model,
                    self.optimizer,
                    self.save_dir / "all-models",
                    f"{self.model_name}_stage2_epoch_{epoch}",
                    epoch
                )

        final_path = save_model(
            self.model,
            self.optimizer,
            self.save_dir / "all-models",
            f"{self.model_name}_stage2_final",
            self.epochs
        ) if self.is_main_process else None

        if self.is_main_process:
            if self.val_dataloader is not None:
                logger.info(f"[Stage 2] Training completed. Best composite val loss: {best_val_loss:.6f}")
                logger.info(f"[Stage 2] Best model: {best_model_path}")
            else:
                logger.info("[Stage 2] Training completed. (No validate dataloader)")
            logger.info(f"[Stage 2] Final model: {final_path}")

        ret = best_model_path if best_model_path else final_path
        ret = self.ddp.broadcast_object(ret)
        return ret

    def freeze_all_except_decoder(self):
        """
        Freeze all layers except the decoder (conv_out layer).
        This is used in Stage 2 of transfer learning.
        """
        m = unwrap_model(self.model)

        # Freeze conv_in and body
        set_requires_grad(m.conv_in, False)
        set_requires_grad(m.body, False)

        # Unfreeze conv_out (decoder)
        set_requires_grad(m.conv_out, True)


        if self.is_main_process:
            trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in m.parameters())
            logger.info(f"[Freeze] Decoder only: {trainable_params}/{total_params} parameters trainable")

    def configure_trainables(self) -> None:
        self.freeze_all_except_decoder()

class Stage3(Stage):
    def train(self):
        if self.is_main_process:
            logger.info("=" * 27)
            logger.info("STAGE 3: Train full model")
            logger.info("=" * 27)

        # Set model to train mode
        self.model.train()

        best_val_loss = float("inf")
        best_model_path = None

        for epoch in range(self.epochs):
            set_sampler_epoch(self.train_dataloader, epoch)

            if self.is_main_process:
                logger.info(f"[Stage 3] Epoch {epoch + 1}/{self.epochs}")

            # Training
            train_loss, train_mrae, train_ndvi, train_ndre = self._train_epoch()

            if self.is_main_process:
                logger.info(
                    f"[Stage 3] Epoch {epoch + 1} - "
                    f"Train Total: {train_loss:.6f}, MRAE: {train_mrae:.6f}, "
                    f"NDVI: {train_ndvi:.6f}, NDRE: {train_ndre:.6f}"
                )
                if self.scheduler is not None:
                    logger.info(f"[Stage 3] Scheduler LR: {self.scheduler.get_last_lr()}")

            # Validation
            if self.val_dataloader is not None:
                val_total, mrae_loss, rmse_loss, psnr_loss, ndvi_loss, ndre_loss = self.validate_epoch()

                if self.is_main_process:
                    logger.info(
                        f"[Stage 3] Epoch {epoch + 1} - "
                        f"Val Total: {val_total:.6f}, MRAE: {mrae_loss:.6f}, RMSE: {rmse_loss:.6f}, "
                        f"PSNR: {psnr_loss:.6f}, NDVI: {ndvi_loss:.6f}, NDRE: {ndre_loss:.6f}"
                    )

                if self.is_main_process and (val_total < best_val_loss):
                    best_val_loss = val_total
                    best_model_path = save_model(
                        self.model,
                        self.optimizer,
                        self.save_dir,
                        f"{self.model_name}_stage3_best",
                        epoch
                    )
                    logger.info(f"[Stage 3] New best model saved! Val composite loss: {val_total:.6f}")

            else:
                if self.is_main_process:
                    logger.info("[Stage 3] There is no validate dataloader")

            # Save checkpoint periodically (rank 0 only)
            if self.is_main_process and ((epoch + 1) % 10 == 0): # save every 10 epoch
                save_model(
                    self.model,
                    self.optimizer,
                    self.save_dir / "all-models",
                    f"{self.model_name}_stage3_epoch_{epoch}",
                    epoch
                )

        final_path = save_model(
            self.model,
            self.optimizer,
            self.save_dir / "all-models",
            f"{self.model_name}_stage3_final",
            self.epochs
        ) if self.is_main_process else None

        if self.is_main_process:
            if self.val_dataloader is not None:
                logger.info(f"[Stage 3] Training completed. Best composite val loss: {best_val_loss:.6f}")
                logger.info(f"[Stage 3] Best model: {best_model_path}")
            else:
                logger.info("[Stage 3] Training completed. (No validate dataloader)")
            logger.info(f"[Stage 3] Final model: {final_path}")

        ret = best_model_path if best_model_path else final_path
        ret = self.ddp.broadcast_object(ret)
        return ret

    def configure_trainables(self) -> None:
        unfreeze_all(self.model, self.is_main_process)

@dataclass(frozen=True)
class MamiConfig:
    cluster: bool
    stage1_config: StageConfig
    stage2_config: StageConfig
    stage3_config: StageConfig

class Mami:
    def __init__(self, config: MamiConfig):
        self.ddp = init_ddp(config.cluster)

        all_stages: list[tuple[type[Stage], StageConfig]] = [
            (Stage1, config.stage1_config),
            (Stage2, config.stage2_config),
            (Stage3, config.stage3_config),
        ]

        self.stages: list[tuple[type[Stage], StageConfig]] = [set for set in all_stages if set[1].run_stage]

    def run_pipeline(self):
        results = {}
        prev_path: Path | None = None
        try:
            for stage_class, config in self.stages:
                # if config.pretrained_model_path is None, inherit prev_path
                effective_cfg = config
                if effective_cfg.pretrained_model_path is None:
                    effective_cfg = replace(config, pretrained_model_path=prev_path)

                if stage_class in (Stage2, Stage3) and effective_cfg.pretrained_model_path is None:
                    raise ValueError(f"{effective_cfg.stage_id} requires a pretrained model path (either from previous stage or explicit).")

                stage = stage_class(effective_cfg, self.ddp)
                results[stage.stage_id] = stage.train()
                prev_path = results[stage.stage_id]

                # ensure everyone finishes the stage before next one starts
                self.ddp.barrier()

            return results
        finally:
            cleanup_ddp(self.ddp)

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arguments for running the Mami pipeline")

    parser.add_argument("--cluster", default=False, action=argparse.BooleanOptionalAction, help="Script intended for cluster, default=False")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (used for deterministic train/val split across DDP ranks)")

    parser.add_argument("--model_name", type=str, required=True, help="The name of the model produced. Do NOT include the file extension to the name")
    parser.add_argument("--dir_name", type=str, required=True, help="The name of the directory for storing the model under the directory 'checkpoints'")

    parser.add_argument("--stage1_data_path", default="data/East-Kaza")
    parser.add_argument("--stage1_data_type", default="Kazakhstan", choices=get_args(DatasetType), help="Which dataset")
    parser.add_argument("--stage1_non_resize", default=False, action=argparse.BooleanOptionalAction, help="Use non-resized pictures, default=False")
    parser.add_argument("--stage1_epochs", type=int, default=0, help="Number of epochs for stage 1 (train from scratch). To skip set epochs to '0'")
    parser.add_argument("--stage1_lr", type=float, default=4e-4, help="Learning rate for stage 1 (train from scratch)")
    parser.add_argument("--stage1_loss_mrae_w", type=float, default=1.0, help="MRAE loss weight for Stage 1")
    parser.add_argument("--stage1_loss_ndvi_w", type=float, default=0.1, help="NDVI loss weight for Stage 1")
    parser.add_argument("--stage1_loss_ndre_w", type=float, default=0.1, help="NDRE loss weight for Stage 1")

    parser.add_argument("--stage2_data_path", default="data/East-Kaza")
    parser.add_argument("--stage2_data_type", default="Kazakhstan", choices=get_args(DatasetType), help="Which dataset")
    parser.add_argument("--stage2_non_resize", default=False, action=argparse.BooleanOptionalAction, help="Use non-resized pictures, default=False")
    parser.add_argument("--stage2_model", type=str, default=None, help="Path to model for stage 2")
    parser.add_argument("--stage2_epochs", type=int, default=0, help="Number of epochs for stage 2. To skip set epochs to '0'")
    parser.add_argument("--stage2_lr", type=float, default=1e-5, help="Learning rate for stage 2")
    parser.add_argument("--stage2_use_scheduler", type=bool,default=False,  action=argparse.BooleanOptionalAction, help="Use a scheduler for the learning rate, default=False")
    parser.add_argument("--stage2_loss_mrae_w", type=float, default=1.0, help="MRAE loss weight for Stage 2")
    parser.add_argument("--stage2_loss_ndvi_w", type=float, default=0.1, help="NDVI loss weight for Stage 2")
    parser.add_argument("--stage2_loss_ndre_w", type=float, default=0.1, help="NDRE loss weight for Stage 2")

    parser.add_argument("--stage3_data_path", default="data/East-Kaza")
    parser.add_argument("--stage3_data_type", default="Weedy-Rice", choices=get_args(DatasetType), help="Which dataset Sri-Lanka or Kazakhstan, default=Sri-Lanka")
    parser.add_argument("--stage3_non_resize", default=False, action=argparse.BooleanOptionalAction, help="Use non-resized pictures, default=False")
    parser.add_argument("--stage3_model", type=str, default=None, help="Path to model for stage 3")
    parser.add_argument("--stage3_epochs", type=int, default=0, help="Number of epochs for stage 3. To skip set epochs to '0'")
    parser.add_argument("--stage3_lr", type=float, default=1e-7, help="Learning rate for stage 3")
    parser.add_argument("--stage3_use_scheduler", default=False, action=argparse.BooleanOptionalAction, help="Use a scheduler for the learning rate, default=False")
    parser.add_argument("--stage3_loss_mrae_w", type=float, default=1.0, help="MRAE loss weight for Stage 3")
    parser.add_argument("--stage3_loss_ndvi_w", type=float, default=0.1, help="NDVI loss weight for Stage 3")
    parser.add_argument("--stage3_loss_ndre_w", type=float, default=0.1, help="NDRE loss weight for Stage 3")


    # Band indices in target/predicted ms tensor [B, C, H, W]
    # Defaults assume channel order [G, R, RE, NIR]
    parser.add_argument("--band_idx_red", type=int, default=1, help="Red band channel index in ms tensor")
    parser.add_argument("--band_idx_rededge", type=int, default=2, help="Red-edge band channel index in ms tensor")
    parser.add_argument("--band_idx_nir", type=int, default=3, help="NIR band channel index in ms tensor")


    return parser.parse_args()

def build_mami_config(args: argparse.Namespace) -> MamiConfig:
    stage1: StageConfig = StageConfig(
        "stage1",
        0 < args.stage1_epochs, # IF TO RUN STAGE
        args.model_name,
        None,
        args.dir_name,
        args.stage1_data_path,
        args.stage1_data_type,
        72,                     # BATCH SIZE TRAIN
        16,                     # BATCH SIZE VAL
        args.stage1_epochs,
        args.stage1_lr,
        args.stage1_non_resize,
        True,                   # USE SCHEDULER
        args.stage1_loss_mrae_w,
        args.stage1_loss_ndvi_w,
        args.stage1_loss_ndre_w,
        args.seed + 0,
        args.cluster,
        args.band_idx_red,
        args.band_idx_rededge,
        args.band_idx_nir,
    )

    stage2: StageConfig = StageConfig(
        "stage2",
        0 < args.stage2_epochs, # IF TO RUN STAGE
        args.model_name,
        Path(args.stage2_model) if args.stage2_model is not None else None,
        args.dir_name,
        args.stage2_data_path,
        args.stage2_data_type,
        72,                     # BATCH SIZE TRAIN
        16,                     # BATCH SIZE VAL
        args.stage2_epochs,
        args.stage2_lr,
        args.stage2_non_resize,
        args.stage2_use_scheduler,
        args.stage2_loss_mrae_w,
        args.stage2_loss_ndvi_w,
        args.stage2_loss_ndre_w,
        args.seed + 1,
        args.cluster,
        args.band_idx_red,
        args.band_idx_rededge,
        args.band_idx_nir,
    )

    stage3: StageConfig = StageConfig(
        "stage3",
        0 < args.stage3_epochs, # IF TO RUN STAGE
        args.model_name,
        Path(args.stage3_model) if args.stage3_model is not None else None,
        args.dir_name,
        args.stage3_data_path,
        args.stage3_data_type,
        72,                     # BATCH SIZE TRAIN
        16,                     # BATCH SIZE VAL
        args.stage3_epochs,
        args.stage3_lr,
        args.stage3_non_resize,
        args.stage3_use_scheduler,
        args.stage3_loss_mrae_w,
        args.stage3_loss_ndvi_w,
        args.stage3_loss_ndre_w,
        args.seed + 2,
        args.cluster,
        args.band_idx_red,
        args.band_idx_rededge,
        args.band_idx_nir,
    )

    return MamiConfig(args.cluster, stage1, stage2, stage3)

if __name__ == "__main__":
    args = get_arguments()
    mami_config = build_mami_config(args)

    mami: Mami = Mami(mami_config)
    mami.run_pipeline()
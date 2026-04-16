import psutil
import os
import sys
import torch
import gc


class MemoryCallback:
    def __init__(
        self,
        ram_threshold=95.0,
        vram_min_free_gb=2.0,
        checkpoint_path="emergency_ckpt.pth",
    ):
        self.ram_threshold = ram_threshold
        self.vram_min_free_gb = vram_min_free_gb
        self.checkpoint_path = checkpoint_path

    def check_and_safe_exit(self, model, optimizer=None, epoch=0, batch_idx=0):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # CPU RAM Usage
        mem_usage = psutil.virtual_memory().percent
        if mem_usage > self.ram_threshold:
            print(
                f"\n[CRITICAL] System RAM is almost full ({mem_usage}%). Saving checkpoint..."
            )
            self._save_and_exit(model, optimizer, epoch, batch_idx, f"RAM_{mem_usage}%")

        # GPU VRAM Usage
        if torch.cuda.is_available():
            # Get free and total memory in bytes, convert to GB
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024**3)

            if free_gb < self.vram_min_free_gb:
                print(
                    f"\n[CRITICAL] GPU memory is almost full ({free_gb:.2f} GB remain). Saving checkpoint..."
                )
                self._save_and_exit(
                    model, optimizer, epoch, batch_idx, f"VRAM_{free_gb:.2f}GB"
                )

    def _save_and_exit(self, model, optimizer, epoch, batch_idx, reason):
        state = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model_state_dict": model.state_dict(),
            "reason": reason,
        }
        if optimizer:
            state["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(state, self.checkpoint_path)
        print(f"Emergency checkpoint saved at: {self.checkpoint_path}")
        print("Exiting process to prevent system crash.")
        sys.exit(1)

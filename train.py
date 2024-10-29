import os

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from data_utils import cleaning_seq, seq_to_muspy, to_prompt
from dataset import get_dataloaders
from metrics import get_obj_metrics_all
from models import init_model
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


def get_lr_multiplier(step, cfg) -> float:
    """Get learning rate multiplier."""
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    if step > cfg.decay_end_steps:
        return cfg.decay_end_multiplier
    position = (step - cfg.warmup_steps) / (cfg.decay_end_steps - cfg.warmup_steps)
    return 1 - (1 - cfg.decay_end_multiplier) * position


class TrainConfig:
    def __init__(self, cfg, fn, eval_only=False, gen_only=False):
        self.save_path = os.path.join(cfg.save_dir, fn)
        if os.path.exists(os.path.join(self.save_path, "model_best.pt")) and not eval_only and not gen_only:
            print("trained model already exists")
            if input("overwrite? (y/n): ") != "y":
                return
        os.makedirs(self.save_path, exist_ok=True)
        print("save_path:", self.save_path)

        self.music_rep = cfg.music_rep
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(cfg.data, self.music_rep)
        self.fn = fn
        self.cfg_model = cfg.model

        if not eval_only and not gen_only:
            self.model, self.dp = init_model(cfg.model, self.music_rep, device=device, use_dp=True)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.train.lr)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: get_lr_multiplier(step, cfg.train))
            self.optimizer.zero_grad()
            self.max_steps = cfg.train.max_steps
            self.val_steps = cfg.train.val_steps
            self.acc_steps = cfg.train.accumulation_steps
            self.early_stopping_patience = cfg.train.early_stopping_patience
            self.step = 0
            self.real_step = 0
            self.val_loss_list = []
            self.not_improved_cnt = 0
            self.recent_loss = []
            self.stop_training = False

        if cfg.wandb.use and not gen_only:
            self.use_wandb = cfg.wandb.use
            self.wandb_logger = WandBLogger(cfg, fn)

    def train(self):
        self.model.train()
        while True:
            for batch in tqdm(self.train_loader, desc="train"):
                self._train_step(batch)
                if (self.real_step % self.val_steps == 0 or self.real_step == self.max_steps) and self.real_step != 0:
                    self.model.eval()
                    self._valid()
                    self.model.train()
                if self.real_step == self.max_steps or self.stop_training:
                    return

    def _train_step(self, batch):
        self.step += 1
        loss_list = self.model(batch.to(device)) / self.acc_steps
        loss_list = loss_list.mean(dim=0)
        loss_list.sum().backward()
        if self.music_rep == "NoteTuple":
            self.recent_loss.append([float(loss) for loss in loss_list])
        elif self.music_rep in ["PianoRoll", "REMI+"]:
            self.recent_loss.append(float(loss_list))
        else:
            raise ValueError(f"invalid music representation: {self.music_rep}")

        if self.step % self.acc_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.real_step += 1
            if self.use_wandb:
                train_loss = np.sum(self.recent_loss, axis=0)
                self.recent_loss = []
                self.wandb_logger.train_log(self.real_step, train_loss, "train", self.scheduler.get_last_lr()[0])

    def _valid(self):
        val_loss = self._valid_loss(self.val_loader)
        self.val_loss_list.append(np.sum(val_loss))

        if self.use_wandb:
            self.wandb_logger.train_log(self.real_step, val_loss, "val")

        if len(self.val_loss_list) == 1 or self.val_loss_list[-1] < min(self.val_loss_list[:-1]):
            self.not_improved_cnt = 0
            tqdm.write(f"step: {self.real_step}, val loss improved, saving model...")
            if self.dp:
                torch.save(self.model.module.state_dict(), f"{self.save_path}/model_best.pt")
            else:
                torch.save(self.model.state_dict(), f"{self.save_path}/model_best.pt")
        else:
            self.not_improved_cnt += 1
            if self.not_improved_cnt > self.early_stopping_patience:
                print("early stopping")
                self.stop_training = True

    def _valid_loss(self, dataloader):
        val_loss = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="validation", leave=False):
                loss_list = self.model(batch.to(device))
                loss_list = loss_list.mean(dim=0)
                if self.music_rep == "NoteTuple":
                    val_loss.append([float(loss) for loss in loss_list])
                elif self.music_rep in ["PianoRoll", "REMI+"]:
                    val_loss.append(float(loss_list))
        val_loss = np.mean(val_loss, axis=0)
        return val_loss

    def evaluate(self, sampling_method="top_p", threshold=0.2, eval_bar=16):
        self.load_model()
        gen_seq_all, tgt_seq_all = self._eval_generation(sampling_method, threshold, eval_bar)
        test_loss = self._valid_loss(self.test_loader)
        obj_results = get_obj_metrics_all(gen_seq_all, tgt_seq_all)
        if self.use_wandb:
            self.wandb_logger.eval_log(test_loss, obj_results, sampling_method, threshold, eval_bar)
        else:
            print("test loss:", test_loss)
            print("obj results:", obj_results)

    def _eval_generation(self, sampling_method, threshold, eval_bar):
        gen_seq_all = []
        tgt_seq_all = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="generation"):
                prompt = to_prompt(batch[0], self.music_rep, eval_bar)
                prompt = prompt.unsqueeze(0).to(device)
                if self.music_rep in ["NoteTuple", "REMI+"]:
                    gen_seq = self.model.generate(prompt, sampling_method=sampling_method)
                else:
                    gen_seq = self.model.generate(prompt, threshold=threshold)
                gen_seq = cleaning_seq(gen_seq[0].cpu(), self.music_rep, eval_bar)
                tgt_seq = cleaning_seq(batch[0], self.music_rep, eval_bar)
                gen_seq_all.append(gen_seq)
                tgt_seq_all.append(tgt_seq)
        return gen_seq_all, tgt_seq_all

    def generate(self, sampling_method="top_p", threshold=0.2, given_bar=4, gen_num=10):
        if self.music_rep in ["NoteTuple", "REMI+"]:
            decoding_str = sampling_method
        else:
            decoding_str = threshold
        gen_save_path = os.path.join("./examples/generated/", self.fn + f"_{given_bar}_{decoding_str}")
        org_save_path = "./examples/original/"
        os.makedirs(gen_save_path, exist_ok=True)
        os.makedirs(org_save_path, exist_ok=True)

        self.load_model()
        note_num_all = 0
        time_all = 0.0
        with torch.no_grad():
            torch.manual_seed(0)
            self.test_loader.dataset.output_tempo = True
            for idx, batch_tempo in tqdm(enumerate(self.test_loader)):
                batch, tempo = batch_tempo
                prompt = to_prompt(batch[0], self.music_rep, given_bar + 1)
                prompt = prompt.unsqueeze(0).to(device)
                time_start = time.time()
                if self.music_rep in ["NoteTuple", "REMI+"]:
                    gen_seq = self.model.generate(prompt, sampling_method=sampling_method)
                else:
                    gen_seq = self.model.generate(prompt, threshold=threshold)
                time_end = time.time()
                gen_seq = cleaning_seq(gen_seq[0].cpu(), self.music_rep, None)
                note_num_all += len(gen_seq)
                time_all += time_end - time_start
                org_seq = cleaning_seq(batch[0], self.music_rep, None)
                
                if given_bar == 0:
                    gen_music = seq_to_muspy(gen_seq)
                else:
                    gen_music = seq_to_muspy(gen_seq, tempo.item())
                org_music = seq_to_muspy(org_seq, tempo.item())
                gen_music.write_midi(os.path.join(gen_save_path, f"{idx}.mid"))
                if not os.path.exists(os.path.join(org_save_path, f"{idx}.mid")):
                    org_music.write_midi(os.path.join(org_save_path, f"{idx}.mid"))
                if idx == gen_num - 1:
                    print("average time per note:", time_all / note_num_all)
                    return

    def load_model(self):
        self.model, _ = init_model(self.cfg_model, self.music_rep, device=device, use_dp=False)
        self.model.load_state_dict(torch.load(f"{self.save_path}/model_best.pt"))
        self.model.eval()


class WandBLogger:
    def __init__(self, cfg, fn):
        self.music_rep = cfg.music_rep
        wandb.init(project=cfg.wandb.project, name=fn, config=OmegaConf.to_container(cfg))
        wandb.define_metric("step")
        wandb.define_metric("train/loss", step_metric="step")
        wandb.define_metric("train/lr", step_metric="step")
        wandb.define_metric("val/loss", step_metric="step")
        if self.music_rep == "NoteTuple":
            self.attributes = ["meta", "bar", "position", "track", "pitch", "duration"]
            for attr in self.attributes:
                wandb.define_metric(f"train/loss_{attr}", step_metric="step")
                wandb.define_metric(f"val/loss_{attr}", step_metric="step")

    def train_log(self, step, loss_list, mode, lr=None):
        loss = loss_list.sum()
        wandb.log({f"{mode}/loss": loss, "step": step})
        if mode == "train":
            wandb.log({"train/lr": lr, "step": step})
        if self.music_rep == "NoteTuple":
            for attr in self.attributes:
                wandb.log({f"{mode}/loss_{attr}": loss_list[self.attributes.index(attr)], "step": step})

    def eval_log(self, loss_list, obj_results, sampling_method, threshold, eval_bar):
        loss = loss_list.sum()
        wandb.log({"test/loss": loss})
        if self.music_rep == "NoteTuple":
            for attr in self.attributes:
                wandb.log({f"test/loss_{attr}": loss_list[self.attributes.index(attr)]})
        wandb.log(obj_results)
        wandb.log({"sampling_method": sampling_method, "threshold": threshold, "eval_bar": eval_bar})

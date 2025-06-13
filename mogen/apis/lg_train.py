import torch
import torch.optim as optim
from mogen.core.optimizer.builder import build_optimizers
from mogen.models import build_architecture
from lightning import LightningModule, LightningDataModule
import pickle
from lightning.fabric.fabric import Fabric
import os
import time

class LgModel(LightningModule):

    def __init__(self, cfg, dataset=None, unit=0):
        super().__init__()
        self.save_hyperparameters(cfg._cfg_dict)
        self.cfg = cfg
        self.model = build_architecture(cfg.model)
        self.dataset = dataset
        self.outputs = []
        self.unit = unit

    
    def on_train_epoch_start(self) -> None:
        self.model.others_cuda()

    

    def on_validation_start(self) -> None:
        self.model.others_cuda()


    def configure_optimizers(self):
        optimizer = build_optimizers(self.model, self.cfg.optimizer)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs*2)
        milestone = 5/6 * self.trainer.max_epochs
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        loss_dict = self.model.forward(**batch)
        all_loss = loss_dict['all_loss']
        for name, value in loss_dict.items():
            if name == 'all_loss': continue
            self.log(name, value, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('all_loss', all_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return all_loss
    
    def validation_step(self, batch):
        output = self.model(return_loss=False, **batch)
        self.outputs.append(output)

    # def on_validation_end(self) -> None:
    #     # gather the results from all processes
    def on_validation_epoch_end(self) -> None:
        self.outputs = [i for j in self.outputs for i in j]
        tmp_file = f'/dev/shm/{self.unit}_{self.global_rank}.pkl'
        pickle.dump(self.outputs, open(tmp_file, 'wb'))
        self.trainer.strategy.barrier()
        part_list = []
        if self.global_rank == 0:
            for rank in range(self.trainer.num_devices):
                tmp_file = f'/dev/shm/{self.unit}_{rank}.pkl'
                outputs = pickle.load(open(tmp_file, 'rb'))
                os.remove(tmp_file)
                part_list.append(outputs)
            ordered_results = []
            for res in zip(*part_list):
                ordered_results.extend(list(res))
            ordered_results = ordered_results[:len(self.dataset)]
            print(f'StiSim:{1-evalute_sim(ordered_results, joints_num=21)/evalute_mean(ordered_results, joints_num=21)}')
            results = self.dataset.evaluate(ordered_results)
            for k, v in results.items():
                print(f'\n{k} : {v:.4f}')


def evalute_sim(results, joints_num):
    
    sim_list = []
    for result in results:
        length = result['motion_length'].item()
        gt_motion = result['motion'][:,4:3*joints_num+4]
        index = result['specified_idx']
        gt_stick_motion = gt_motion[index]
        
        pred_motion = result['pred_motion'][:,4:3*joints_num+4]
        pred_index = result['pred_index']
        pred_id = pred_index.max(0)[1]
        pred_stcik_motion = pred_motion[pred_id].double()
        
        dist = (pred_stcik_motion - gt_stick_motion).view(-1,joints_num,3).pow(2).sum(-1).mean()
        
        sim_list.append(dist.item())
    return sum(sim_list)/len(sim_list)


def evalute_mean(results,joints_num):

    sim_list = []
    for result in results:
        length = result['motion_length'].item()
        gt_motion = result['motion'][:,4:3*joints_num+4]
        index = result['specified_idx']
        gt_stick_motion = gt_motion[index]
        
        pred_motion = result['pred_motion'][:,4:3*joints_num+4]
        dist = (pred_motion[:length,None] - gt_stick_motion[None]).view(-1,joints_num,3).pow(2).sum(-1).mean()
        sim_list.append(dist.item())
    return sum(sim_list)/len(sim_list)

# (evalute_sim(ordered_results, joints_num=21), evalute_mean(ordered_results, joints_num=21))
# 1-evalute_sim(ordered_results, joints_num=21)/evalute_mean(ordered_results, joints_num=21)
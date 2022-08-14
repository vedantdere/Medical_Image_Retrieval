import torch
from torch.optim import Adam
from CLIP.model import CLIP
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import wandb
import argparse
from CLIP.clip import _transform

wandb.init(project="CLIP")

class Dataset(Dataset):
    def __init__(self , 
                 root_dir_A , 
                 root_dir_B , 
                 transforms = None):
        super(DataSet , self).__init__()
        
        self.root_dir_A = root_dir_A
        self.root_dir_B = root_dir_B
        self.folder_a = os.listdir(root_dir_A)
        self.folder_b = os.listdir(root_dir_B)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.folder_a)

    def __getitem__(self , idx):
        img_1 = self.folder_a[idx]
        img_2 = self.folder_b[idx]
        img_1 = Image.open(os.path.join(self.root_dir_A , img_1))
        img_2 = Image.open(os.path.join(self.root_dir_B , img_2))
        
        if transforms:
            img_1 = transforms(img_1)
            img_2 = transforms(img_2)
        return img_1 , img_2
    
    
def train(model , 
          dataloader , 
          optimizer , 
          criterion , 
          epoch , 
          device , 
          ckpt_dir , 
          load_model_path = None):
    os.mkdir(path=ckpt_dir , exist_ok=True)
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))
    model.train()
    for i , (img_1 , img_2 , label) in enumerate(dataloader):
        img_1 = img_1.to(device)
        img_2 = img_2.to(device)
        optimizer.zero_grad()
        logits_per_image , image_features2 = model(img_1 , img_2)
        loss_1 = criterion(logits_per_image , label)
        loss_2 = criterion(image_features2 , label)
        loss = (loss_1 + loss_2)/2
        
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            #print('[Epoch %d , Batch %d] loss: %.4f' % (epoch , i , loss.item()))
            wandb.log({'loss': loss.item()})
            wandb.Image(img_1[0] , caption=label)
            wandb.Image(img_2[0] , caption=label)
        if i % 1000 == 0:
            torch.save(model.state_dict() , f'./{ckpt_dir}/model_%d.pth' % i)
          
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP')
    
    ## Model Hyperparameters
    parser.add_argument('--model', type=str, default='CLIP', help='model name')
    parser.add_argument('--embedding_dim', type=int, default=512, help='embedding_dim')
    parser.add_argument('--image_resolutin', type=int, default=512, help='image_resolutin')
    parser.add_argument('--vision_width', type=int, default=512, help='vision_width')
    parser.add_argument('--vision_patch_size', type=int, default=64, help='vision_patch_size')
    parser.add_argument('--vision_layers', type=int, default=3, help='vision_layers')
    
    ## Training Parameters
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--betas', type=tuple, default=(0.9 , 0.999), help='betas')
    parser.add_argument('--load_model_path', type=str, default=None, help='load_model')
    
    ## Dataset Parameters
    parser.add_argument('--root_dir_A', type=str, default='/home/ubuntu/data/A', help='root dir of A')
    parser.add_argument('--root_dir_B', type=str, default='/home/ubuntu/data/B', help='root dir of B')
    
    ## Misc Parameters
    parser.add_argument('--ckpt_dir', type=str, default='ckpt', help='ckpt dir')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    
    args = parser.parse_args()
    
    model = CLIP(embedding_dim=args.embedding_dim ,
                 image_resolution=args.image_resolution ,
                 vision_layers=args.vision_layers ,
                 vision_width=args.vision_width ,
                 vision_patch_size=args.vision_patch_size)
    
    optimizer = Adam(model.parameters() , lr=args.lr , betas=args.betas)
    criterion = torch.nn.CrossEntropyLoss()
    root_dir_A = args.root_dir_A
    root_dir_b = args.root_dir_B
    dataset = Dataset(root_dir_A, root_dir_B , transforms=_transform)
    dataloaer = DataLoader(dataset , batch_size=args.batch_size , shuffle=True , num_workers=args.num_workers)
    epoch = args.epochs
    device = args.device
    train(model , dataloaer , optimizer , criterion , epoch , device , args.ckpt_dir , args.load_model_path)





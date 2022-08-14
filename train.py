import torch
from torch.optim import Adam
from CLIP.model import CLIP
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import wandb

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
          device):
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
          
if __name__ == '__main__':
    model = CLIP()
    optimizer = Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    root_dir_A = ''
    root_dir_b = ''
    dataset = Dataset(root_dir_A, root_dir_B)
    dataloaer = DataLoader(dataset , batch_size=128 , shuffle=True)
    epoch = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(model , dataloaer , optimizer , criterion , epoch , device)





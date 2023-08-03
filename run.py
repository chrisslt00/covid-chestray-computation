import torch
import numpy as np 
from dataset import ChestImage
import argparse
import os
import random
from torch.utils.data import DataLoader
from model import MINIVGG, CUSTOM_MINIVGG
import torchvision
from torch import nn 
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sn
import pandas as pd
from transformers import ViTConfig, ViTModel, ViTImageProcessor

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(model, image, label, criterion, optimizer, args, proc):
    model.train()
    optimizer.zero_grad()
    if args.model == "visionTransformer":
        out = []
        for im in image:
            new_im = proc(im, return_tensors="pt")
            out.append(model(**new_im).last_hidden_state)
        out = np.asarray(out)
    else:
        out = model(image)
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()

    y_pred = out.argmax(dim=-1)

    covid_total_num = torch.sum((label==0)).item()
    normal_total_num = torch.sum((label==1)).item()
    virus_total_num = torch.sum((label==2)).item()

    covid_correct_pred = torch.sum((y_pred == label) * (label==0))
    normal_correct_pred = torch.sum((y_pred == label) * (label==1))
    virus_correct_pred = torch.sum((y_pred == label) * (label==2))
    
    
    return loss, covid_correct_pred, covid_total_num, normal_correct_pred, normal_total_num, virus_correct_pred, virus_total_num

@torch.no_grad()
def test(model, image, label, vis):
    model.eval()
    if vis == None:
        y_pred = model(image).argmax(dim=-1)
    else:
        image = vis(image, return_tensors="pt")
        y_pred = model(**image).argmax(dim=-1).item()

    covid_total_num = torch.sum((label==0)).item()
    normal_total_num = torch.sum((label==1)).item()
    virus_total_num = torch.sum((label==2)).item()

    covid_correct_pred = torch.sum((y_pred == label) * (label==0))
    normal_correct_pred = torch.sum((y_pred == label) * (label==1))
    virus_correct_pred = torch.sum((y_pred == label) * (label==2))

    return torch.eq(label.view(-1).int(), y_pred).sum().item(), len(y_pred),  covid_correct_pred, covid_total_num, normal_correct_pred, normal_total_num, virus_correct_pred, virus_total_num, y_pred

def main(args):
    seed_everything(args.seed)

    dataset = ChestImage(args)
    
    train_set, test_set = torch.utils.data.random_split(dataset, [1500, 323])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    
    vision = False
    if args.model == "resnet18":
        # implemented for our case
        model = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
        model.conv1 = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        d = model.fc.in_features
        model.fc = nn.Linear(d, 3)
    elif args.model == "miniVGG":
        # This is written by us 
        # get rid of padding 
        model = MINIVGG()
    elif args.model == "visionTransformer":
        # TODO: import library
        # config = ViTConfig(num_channels=1, return_dict = False)
        feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        vision = True
    elif args.model == "naive_CNN":
        # TODO: more baseline
        pass
    elif args.model == "custom_minivgg":
        model = CUSTOM_MINIVGG()
    else:
        print("This model is not implemented!")
        raise NotImplementedError
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    test_correct, test_total = 0, 0 

    total_loss = 0


    best_test_accuracy = -100 

    save_dir = f"./results/{args.model}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    
    for curr_epoch in range(args.epoch):
        epoch_training_loss = 0
        epoch_training_covid_correct, epoch_training_covid_total = 0, 0 
        epoch_training_normal_correct, epoch_training_normal_total = 0, 0 
        epoch_training_virus_correct, epoch_training_virus_total = 0, 0 

        epoch_test_covid_correct, epoch_test_covid_total = 0, 0 
        epoch_test_normal_correct, epoch_test_normal_total = 0, 0 
        epoch_test_virus_correct, epoch_test_virus_total = 0, 0 
        for batch in train_loader:
            (image, label) = batch
            image = image.to(device)
            label = label.to(device)
            # image = torch.squeeze(image, dim=0)
            if args.model == "visionTransformer":
                loss, covid_correct_pred, covid_total_num, normal_correct_pred, normal_total_num, virus_correct_pred, virus_total_num = train(model, image, label, criterion, optimizer, args, feature_extractor)
            else:
                loss, covid_correct_pred, covid_total_num, normal_correct_pred, normal_total_num, virus_correct_pred, virus_total_num = train(model, image, label, criterion, optimizer, args, None)
            epoch_training_covid_correct += covid_correct_pred 
            epoch_training_covid_total += covid_total_num 
            epoch_training_normal_correct += normal_correct_pred 
            epoch_training_normal_total += normal_total_num
            epoch_training_virus_correct += virus_correct_pred 
            epoch_training_virus_total += virus_total_num

            epoch_training_loss += loss 
            total_loss += loss 
        
        print(f"Epoch [{curr_epoch}]:")
        print(f"Average Training Accuracy = {(epoch_training_covid_correct + epoch_training_normal_correct + epoch_training_virus_correct)/(epoch_training_covid_total + epoch_training_normal_total + epoch_training_virus_total)}")
        print(f"  Training Covid Correct Prediction = {epoch_training_covid_correct}, Total Covid Samples = {epoch_training_covid_total}, Covid Training Acc = {epoch_training_covid_correct/epoch_training_covid_total}")
        print(f"  Training Normal Correct Prediction = {epoch_training_normal_correct}, Total Normal Samples = {epoch_training_normal_total}, Normal Training Acc = {epoch_training_normal_correct/epoch_training_normal_total}")
        print(f"  Training Virus Correct Prediction = {epoch_training_virus_correct}, Total Virus Samples = {epoch_training_virus_total}, Virus Training Acc = {epoch_training_covid_correct/epoch_training_virus_total}")
        print(f"Current epoch {curr_epoch} total training loss: {epoch_training_loss}")
        print("\n")
        
        y_pred = []
        y_true = []
        for batch in test_loader:
            
            (image, label) = batch
            image = image.to(device)
            label = label.to(device)

           
            if args.model == "visionTransformer":
                curr_correct, curr_total, covid_correct_pred, covid_total_num, normal_correct_pred, normal_total_num, virus_correct_pred, virus_total_num, curr_y_pred = test(model, image, label, feature_extractor)
            else:
                curr_correct, curr_total, covid_correct_pred, covid_total_num, normal_correct_pred, normal_total_num, virus_correct_pred, virus_total_num, curr_y_pred = test(model, image, label, None)
            test_correct += curr_correct
            test_total += curr_total

            y_true.extend(label.cpu().numpy())
            y_pred.extend(curr_y_pred.cpu().numpy())

            epoch_test_covid_correct += covid_correct_pred 
            epoch_test_covid_total += covid_total_num 
            epoch_test_normal_correct += normal_correct_pred 
            epoch_test_normal_total += normal_total_num
            epoch_test_virus_correct += virus_correct_pred 
            epoch_test_virus_total += virus_total_num
        
        print(f"Average Test Accuracy = {test_correct / test_total}")
        print(f"  Testing Covid Correct Prediction = {epoch_test_covid_correct}, Total Covid Samples = {epoch_test_covid_total}, Covid Testing Acc = {epoch_test_covid_correct/epoch_test_covid_total}")
        print(f"  Testing Normal Correct Prediction = {epoch_test_normal_correct}, Total Normal Samples = {epoch_test_normal_total}, Normal Testing Acc = {epoch_test_normal_correct/epoch_test_normal_total}")
        print(f"  Testing Virus Correct Prediction = {epoch_test_virus_correct}, Total Virus Samples = {epoch_test_virus_total}, Virus Testing Acc = {epoch_test_covid_correct/epoch_test_virus_total}")
        print("\n"*3)

        curr_test_acc = test_correct / test_total 

        if curr_test_acc > best_test_accuracy:
            # save best model 
            torch.save(model, os.path.join(save_dir, "model.pth"))

            # Save Confusion matrix 
            classes = ('Covid', 'Normal', 'Virus')

            cf_matrix = confusion_matrix(y_true, y_pred)
            df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 10, index = [i for i in classes], columns = [i for i in classes])
            
            plt.figure(figsize=(12,7))
            sn.heatmap(df_cm, annot=True)
            plt.savefig(os.path.join(save_dir, "confusion_matrix.pdf"))
            plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))

            # TODO: save ROC figure
            f_p_r = []
            t_p_r = []
            roc_auc = []
            for i in range(3):
                y_true_tmp = []
                for j in y_true:
                    if j == i:
                        y_true_tmp.append(1)
                    else:
                        y_true_tmp.append(0)

                y_pred_tmp = []
                for j in y_pred:
                    if j == i:
                        y_pred_tmp.append(1)
                    else:
                        y_pred_tmp.append(0)

                false_positive_rate, true_positive_rate, threshold = roc_curve(y_true_tmp, y_pred_tmp)
                f_p_r.append(false_positive_rate)
                t_p_r.append(true_positive_rate)
                roc_auc.append(auc(false_positive_rate, true_positive_rate))
            
            for i in range(3):
                plt.figure()
                plt.plot(f_p_r[i], t_p_r[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic example')
                plt.legend(loc="lower right")
                path = "roc_out_"+str(i)+".png"
                plt.savefig(os.path.join(save_dir, path))
                plt.close()


    

if __name__ == "__main__":

    # Change parameters -> lr, epoch num 
    # python run.py --lr xxx --model xxx 
    # model: {miniVGG, resnet18}

    parser = argparse.ArgumentParser(description='442 Project')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--model", type=str, default="miniVGG")
    parser.add_argument("--preprocess", type=str, default="cannyedge") 
    # --modeJack
    args = parser.parse_args()
    main(args)
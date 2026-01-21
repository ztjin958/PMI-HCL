import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import sys
from torch_geometric.data import Data
import shutil
import random
from torch import nn
from Prepare import *
from Model import Model
from sklearn.model_selection import StratifiedKFold
from utils import *
device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
meta_features = meta_large_model_features
protein_features = protein_large_model_features
protein_dim = protein_features.shape[1]
meta_dim = meta_features.shape[1]

R = 10
n_splits = 5
lr = 0.001
weight_dacay = 0.01
step_size = 30
threshold_cur_epoch = 30
threshold_i = 240

kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
for item in range(R):
    y,delete_index = balance_tensor(torch.tensor(Y))
    x_protein = np.delete(np.array(X_proteins), delete_index)
    x_meta = np.delete(np.array(X_metas), delete_index)

    for fold, (train_idx, test_idx) in enumerate(kf.split(x_protein, y)):

        print(f"RUS {item+1} Fold {fold + 1}")
        model = Model(protein_dim, meta_dim).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_dacay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
        criterion = nn.BCEWithLogitsLoss().to(device)


        train_protein_idx = x_protein[train_idx].flatten()
        train_meta_idx = x_meta[train_idx].flatten()
        test_protein_idx = x_protein[test_idx].flatten()
        test_meta_idx = x_meta[test_idx].flatten()
        y_train, y_val = np.array([y[i] for i in train_idx]), np.array([y[i] for i in test_idx])
        y_train = torch.tensor(y_train).to(device)
        y_val = torch.tensor(y_val).to(device)



        protein_dis_knn_from_sim_data = Data(
            x=torch.tensor(protein_features, dtype=torch.float32).to(device),
            hyperedge_index=hyperedge_protein_dis_knn_from_sim_index).to(device)

        meta_dis_knn_from_X_sim_data = Data(
            x=torch.tensor(meta_features, dtype=torch.float32).to(device),
            hyperedge_index=hyperedge_meta_dis_knn_from_sim_index).to(device)




        data = {
            "protein_dis_knn_from_sim_data": protein_dis_knn_from_sim_data,
            "meta_dis_knn_from_sim_data": meta_dis_knn_from_X_sim_data,
        }

        # 训练模型
        model.train()
        print("开始训练")
        if not os.path.exists("model"):
            os.mkdir("model")
        if os.path.exists(f"model/fold_{fold + 1}"):
            shutil.rmtree(f"model/fold_{fold + 1}")
        os.mkdir(f"model/fold_{fold + 1}")

        best_loss = 1e5
        cur_epoch = 0
        best_epoch = 0

        from sklearn.metrics import accuracy_score, recall_score, f1_score
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import matthews_corrcoef

        for i in tqdm(range(1000), file=sys.stdout):

            output,cl_loss = model(data, index=(train_protein_idx,train_meta_idx))
            loss = criterion(output.view(-1), y_train.float())
            optimizer.zero_grad()
            total_loss = loss+cl_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            labels = y_train.float().to("cpu").flatten()
            scores = nn.functional.sigmoid(output.detach().flatten().to("cpu"))
            predicted = (nn.functional.sigmoid(output.detach().flatten().to("cpu")) >= 0.5)
            mcc = matthews_corrcoef(labels, predicted)
            recall = recall_score(labels, predicted)
            roc_auc = roc_auc_score(labels, scores)
            accuracy = accuracy_score(labels, predicted)
            f1 = f1_score(labels, predicted, average='binary')
            if i and i % 50 == 0: print(mcc, roc_auc, accuracy, total_loss.item())

            # print(mcc, roc_auc, accuracy, total_loss.item())
            if best_loss > total_loss.item():
                best_loss = total_loss.item()
                cur_epoch = 0
                best_epoch = i + 1
                torch.save(model.state_dict(), f"model/fold_{fold + 1}/best_model" + '.pth')
            else:
                cur_epoch += 1
                if cur_epoch > threshold_cur_epoch and i > threshold_i:
                    break

        # 测试模型
        model.eval()
        model.load_state_dict(torch.load(f"model/fold_{fold + 1}/best_model" + '.pth'))
        print("开始测试")
        ################################################################

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import roc_auc_score, auc
        from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_curve

        with torch.no_grad():
            output,_ = model(data, index=(test_protein_idx,test_meta_idx))
            labels = y_val.float().to("cpu").flatten()
            scores = nn.functional.sigmoid(output.detach().flatten().to("cpu"))
            predicted = (nn.functional.sigmoid(output.detach().flatten().to("cpu")) >= 0.5)
            mcc = matthews_corrcoef(labels, predicted)
            precision_score = precision_score(labels, predicted)
            recall = recall_score(labels, predicted)
            precision, recall_aupr, _ = precision_recall_curve(labels, scores)
            aupr = auc(recall_aupr, precision)
            roc_auc = roc_auc_score(labels, scores)
            accuracy = accuracy_score(labels, predicted)
            f1 = f1_score(labels, predicted, average='binary')
            tn, fp, fn, tp = confusion_matrix(labels, predicted).ravel()
            specificity = tn / (tn + fp)
            print( f"accuracy:{accuracy}", f"f1:{f1}", f"roc_auc:{roc_auc}", f"aupr:{aupr}",sep=" ")
            print( f"mcc:{mcc}", f"specificity:{specificity}", f"precision_score:{precision_score}",sep=" ")
            accuracy_scores.append(accuracy)
            specificity_scores.append(specificity)
            precision_score_scores.append(precision_score)
            recall_scores.append(recall)
            roc_auc_scores.append(roc_auc)
            f1_scores.append(f1)
            mcc_scores.append(mcc)
            aupr_scores.append(aupr)


print("*********************************************")
print("accuracy_scores:", process_data(accuracy_scores,R,n_splits))
print("specificity_scores:", process_data(specificity_scores,R,n_splits))
print("precision_score_scores:", process_data(precision_score_scores,R,n_splits))
print("aupr_scores:", process_data(aupr_scores,R,n_splits))
print("recall_scores:", process_data(recall_scores,R,n_splits))
print("roc_auc_scores:", process_data(roc_auc_scores,R,n_splits))
print("f1_scores:", process_data(f1_scores,R,n_splits))
print("mcc_scores:", process_data(mcc_scores,R,n_splits))

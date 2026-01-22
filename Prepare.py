import warnings
warnings.filterwarnings('ignore')
from utils import *
import pandas as pd
import numpy as np
import torch
np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

cs = "_400"
path = r'path of your dataset'

df = pd.read_csv(os.path.join(path,f"m_p_links{cs}.csv"))
# df = pd.read_csv(os.path.join(path,f"m_p_links.csv"))

# df = pd.read_excel(os.path.join(path, "m_p_links.xlsx"))
df = df.dropna(subset=[df.columns[0], df.columns[1]], how='any')
edges_df = df.iloc[:, [0, 1]]


protein_dict = {}
meta_dict = {}

df_pmi_meta_list = df.iloc[:, 0].drop_duplicates().values.tolist()
df_pmi_protein_list = df.iloc[:, 1].drop_duplicates().values.tolist()
for idx, id in enumerate(df_pmi_protein_list):
    protein_dict[id] = idx

for idx, id in enumerate(df_pmi_meta_list):
    meta_dict[id] = idx


with open(os.path.join(path, "edges.csv"), "w") as f:
    for i in range(len(edges_df)):
        if int(df.iloc[i, 2]) == 1:
            f.write(f"{meta_dict[edges_df.iloc[i, 0]]},{protein_dict[edges_df.iloc[i, 1]]},1\n")
        else:
            f.write(f"{meta_dict[edges_df.iloc[i, 0]]},{protein_dict[edges_df.iloc[i, 1]]},0\n")

df_edges = pd.read_csv(os.path.join(path, "edges.csv"), header=None, names=['node1', 'node2', "weight"])
df_meta = pd.read_csv(os.path.join(path, "meta.smi"), sep=" ", header=None)



df_protein_edgelist = pd.read_csv(os.path.join(path, "p_p_links.tsv"), header=None, sep='\t',
                                  names=["node1", "node2", "weight"])
df_meta_edgelist = pd.read_csv(os.path.join(path, "m_m_links.tsv"), header=None, sep='\t',
                               names=["node1", "node2", "weight"])
if not os.path.exists(os.path.join(path, "protein_edge.edgelist")):
    with open(os.path.join(path, "protein_edge.edgelist"), "w") as f:
        for i in range(len(df_protein_edgelist)):
            try:
                f.write(
                    f"{protein_dict[df_protein_edgelist.iloc[i, 0]]} {protein_dict[df_protein_edgelist.iloc[i, 1]]} {df_protein_edgelist.iloc[i, 2]}\n")
            except:
                pass
if not os.path.exists(os.path.join(path, "meta_edge.edgelist")):
    with open(os.path.join(path, "meta_edge.edgelist"), "w") as f:
        for i in range(len(df_meta_edgelist)):
            try:
                f.write(
                    f"{meta_dict[df_meta_edgelist.iloc[i, 0]]} {meta_dict[df_meta_edgelist.iloc[i, 1]]} {df_meta_edgelist.iloc[i, 2]}\n")
            except:
                pass


df_embedding_protein = edgelist_to_matrix(len(df_pmi_protein_list),os.path.join(path, "protein_edge.edgelist"))
df_embedding_meta = edgelist_to_matrix(len(df_pmi_meta_list),os.path.join(path, "meta_edge.edgelist"))
protein_name_df = pd.read_csv(os.path.join(path, "matched_protein_sequences.csv"))


X_protein_large_model = {}  # dim 1024
df_protein_large_model = np.load(os.path.join(path, "protein_large_model.npy"))
pro_count = 0
for key, value in protein_dict.items():  # 获取蛋白质名称：编号
    try:  
        index_in_folder = protein_name_df.loc[protein_name_df.iloc[:, 0] == key].index
        X_protein_large_model[value] = df_protein_large_model[index_in_folder.item()]
    except:
        X_protein_large_model[value] = np.random.rand(df_protein_large_model.shape[1]).flatten()
        pro_count += 1
if (pro_count):
    print(f"Not in protein_large_model:{pro_count}")


X_protein_sim = {}
pro_count = 0
for key, value in protein_dict.items():  
    try:
        index_in_folder = protein_name_df.loc[protein_name_df.iloc[:, 0] == key].index
        X_protein_sim[value] = df_embedding_protein[value]
    except:
        X_protein_sim[value] = np.random.rand(len(df_pmi_protein_list)).flatten()
        pro_count += 1
if pro_count:
    print(f"Not in X_protein_sim:{pro_count}")



df_meta_large_model = np.load(os.path.join(path, "meta_ChemGPT-19M.npy"))
X_meta_large_model = {}
pro_count = 0
for key, value in meta_dict.items():
    try:
        index = df_meta[df_meta.iloc[:,1] == key].index[0]
        X_meta_large_model[value] = df_meta_large_model[index]
    except:
        X_meta_large_model[value] = np.random.rand(df_meta_large_model.shape[1]).flatten()
        pro_count += 1
if (pro_count):
    print(f"Not in meta_large_model:{pro_count}")


X_meta_sim = {}
pro_count = 0
for key, value in meta_dict.items():
    try:
        X_meta_sim[value] = df_embedding_meta[value]
    except:
        X_meta_sim[value] = np.random.rand(len(df_pmi_meta_list)).flatten()
        pro_count += 1
if (pro_count):
    print(f"Not in embeddings_meta:{pro_count}")




X_proteins = df_edges.iloc[:, 1].values.tolist()
X_metas = df_edges.iloc[:, 0].values.tolist()
Y = np.array(df_edges.iloc[:, 2].values.tolist())





K_meta = [4,9]
K_protein = [7,10]

# protein_large_model
protein_large_model_features = np.array([X_protein_large_model[i] for i in range(len(df_pmi_protein_list))])
# meta_large_model
meta_large_model_features = np.array([X_meta_large_model[i] for i in range(len(df_pmi_meta_list))])


# sim
sim_protein_features = np.array([X_protein_sim[i] for i in range(len(df_pmi_protein_list))])
H_protein_dis_knn_from_sim= construct_H_with_KNN(sim_protein_features,K_protein,metric = 'cosine')
hyperedge_protein_dis_knn_from_sim_index = convert_adjacency_matrix(
    H_protein_dis_knn_from_sim)



sim_meta_features = np.array([X_meta_sim[i] for i in range(len(df_pmi_meta_list))])
H_meta_dis_knn_from_sim = construct_H_with_KNN(sim_meta_features,K_meta,metric = 'cosine')
hyperedge_meta_dis_knn_from_sim_index = convert_adjacency_matrix(
    H_meta_dis_knn_from_sim)



recall_scores = []
aupr_scores = []
f1_scores = []
mcc_scores = []
prescision_scores = []
roc_scores = []
precision_scores = []
precision_recall_scores = []
roc_auc_scores = []
accuracy_scores = []
specificity_scores = []
precision_score_scores = []
print(f"meta_nums:{len(df_pmi_meta_list)}  protein_nums:{len(df_pmi_protein_list)}")
print("Prepare is done")




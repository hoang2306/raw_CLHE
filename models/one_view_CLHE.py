import numpy as np
from collections import OrderedDict
import os
import scipy.sparse as sp 

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import TransformerEncoder 
from models.utils import to_tensor
from models.gnn import Amatrix, AsymMatrix


eps = 1e-9


def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)


def recon_loss_function(recon_x, x):
    negLogLike = torch.sum(F.log_softmax(recon_x, 1) * x, -1) / x.sum(dim=-1)
    negLogLike = -torch.mean(negLogLike)
    return negLogLike


infonce_criterion = nn.CrossEntropyLoss()


def cl_loss_function(a, b, temp=0.2):
    a = nn.functional.normalize(a, dim=-1)
    b = nn.functional.normalize(b, dim=-1)
    logits = torch.mm(a, b.T)
    logits /= temp
    labels = torch.arange(a.shape[0]).to(a.device)
    return infonce_criterion(logits, labels)

# LightGCN module
def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph

def get_item_level_graph(graph, device):
    ui_graph = graph
    item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
    item_level_graph = to_tensor(laplace_transform(item_level_graph)).to(device)
    return item_level_graph

class LightGCN(nn.Module):
    def __init__(self, conf, graph, n_layers=1):
        super().__init__()
        self.conf = conf
        self.graph = graph # graph shape: [n_edges, 2]
        self.device = self.conf['device']
        self.n_layers = n_layers
        self.process_graph()

    def process_graph(self):
        values = np.ones(len(self.graph), dtype=np.float32)
        # values
        indice = self.graph 
        n_item = self.conf['num_items']
        self.iui_graph_sparse = sp.csr_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(n_item, n_item)
        )
        self.ii_graph = get_item_level_graph(graph=self.iui_graph_sparse, device=self.device)

    def one_propagate(self, a_feature, b_feature, graph, n_layers=1):
        features = torch.cat([a_feature, b_feature], dim=0)
        # print(features.shape)
        all_features = [features]

        for i in range(n_layers):
            features = torch.spmm(graph, features)
            all_features.append(features)

        all_features = torch.stack(all_features, dim=1)
        # print(all_features.shape)
        all_features = torch.mean(all_features, dim=1)
        # print(all_features.shape)

        a_feature, b_feature = torch.split(all_features, (a_feature.shape[0], b_feature.shape[0]), 0)
        return a_feature, b_feature

    def forward(self, a_feature, b_feature):
        a_feature, b_feature = self.one_propagate(
            a_feature, b_feature, self.ii_graph, self.n_layers
        )
        return a_feature, b_feature

class HierachicalEncoder(nn.Module):
    def __init__(self, conf, raw_graph, features):
        super(HierachicalEncoder, self).__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device
        self.num_user = self.conf["num_users"]
        self.num_bundle = self.conf["num_bundles"]
        self.num_item = self.conf["num_items"]
        self.embedding_size = 64
        self.ui_graph, self.bi_graph_train, self.bi_graph_seen = raw_graph
        self.attention_components = self.conf["attention"]

        self.content_feature, self.text_feature, self.cf_feature = features

        items_in_train = self.bi_graph_train.sum(axis=0, dtype=bool)
        self.warm_indices = torch.LongTensor(
            np.argwhere(items_in_train)[:, 1]).to(device)
        self.cold_indices = torch.LongTensor(
            np.argwhere(~items_in_train)[:, 1]).to(device)

        # MM >>>
        self.content_feature = nn.functional.normalize(
            self.content_feature, dim=-1)
        self.text_feature = nn.functional.normalize(self.text_feature, dim=-1)

        def dense(feature):
            module = nn.Sequential(OrderedDict([
                ('w1', nn.Linear(feature.shape[1], feature.shape[1])),
                ('act1', nn.ReLU()),
                ('w2', nn.Linear(feature.shape[1], 256)),
                ('act2', nn.ReLU()),
                ('w3', nn.Linear(256, 64)),
            ]))

            for m in module:
                init(m)
            return module

        # encoders for media feature
        self.c_encoder = dense(self.content_feature)
        self.t_encoder = dense(self.text_feature)

        self.multimodal_feature_dim = self.embedding_size
        # MM <<<

        # BI >>>
        self.item_embeddings = nn.Parameter(
            torch.FloatTensor(self.num_item, self.embedding_size))
        init(self.item_embeddings)
        self.multimodal_feature_dim += self.embedding_size
        # BI <<<

        # UI >>>
        self.cf_transformation = nn.Linear(
            self.embedding_size, self.embedding_size)
        init(self.cf_transformation)
        items_in_cf = self.ui_graph.sum(axis=0, dtype=bool)
        self.warm_indices_cf = torch.LongTensor(
            np.argwhere(items_in_cf)[:, 1]).to(device)
        self.cold_indices_cf = torch.LongTensor(
            np.argwhere(~items_in_cf)[:, 1]).to(device)
        self.multimodal_feature_dim += self.embedding_size
        # UI <<<

        # Multimodal Fusion:
        self.w_q = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_q)
        self.w_k = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_k)
        self.w_v = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_v)
        self.ln = nn.LayerNorm(self.embedding_size, elementwise_affine=False)


        # gnn
        if self.conf['use_iui_graph']:
            print('use iui graph gnn module')
            self.iui_graph_path = self.conf['iui_graph_path']
            print(f'iui graph path: {self.iui_graph_path}')
            self.iui_graph = np.load(self.iui_graph_path, allow_pickle=True)
            # self.iui_graph = torch.tensor(self.iui_graph).to(self.device)

            if self.conf['type_gnn_implement'] == 'torch_geometric':
                self.iui_gnn_conv = Amatrix(
                    in_dim=64,
                    out_dim=64,
                    n_layer=1,
                    dropout=0.1,
                    heads=2, 
                    concat=False,
                    self_loop=False,
                    extra_layer=True,
                    # type_gnn='light_gcn'
                    type_gnn=self.conf['iui_gnn_type']
                )
            if self.conf['type_gnn_implement'] == 'self_implement':
                self.iui_gnn_conv = LightGCN(
                    conf=self.conf,
                    graph=self.iui_graph.T,
                    n_layers=1
                )
            self.item_iui_gnn_emb = nn.Parameter(
                torch.FloatTensor(self.num_item, self.embedding_size)
            )
            init(self.item_iui_gnn_emb)

        self.get_bundle_agg_graph_ori(self.bi_graph_seen)
    
    def selfAttention(self, features):
        # features: [bs, #modality, d]
        if "layernorm" in self.attention_components:
            features = self.ln(features)
        q = self.w_q(features)
        k = self.w_k(features)
        if "w_v" in self.attention_components:
            v = self.w_v(features)
        else:
            v = features
        # [bs, #modality, #modality]
        attn = q.mul(self.embedding_size ** -0.5) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)

        features = attn @ v  # [bs, #modality, d]
        # average pooling
        y = features.mean(dim=-2)  # [bs, d]

        return y
    
    def get_bundle_agg_graph_ori(self, graph):
        bi_graph = graph
        device = self.device
        eps = 1e-8
        bundle_size = bi_graph.sum(axis=1) + eps # calculate size for each bundle 
        # print(f"bundle size: {bundle_size.shape}")
        # print(f"diag bundle: {sp.diags(1/bundle_size.A.ravel()).shape}")
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph # sp.diags(1/bundle_size.A.ravel()): D^-1 
        # print(f'graph: {graph}')
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device) 

    def forward_all(self):
        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        features = [mm_feature_full]
        features.append(self.item_embeddings)

        if self.conf['use_item_pretrained_emb']:
            cf_feature_full = self.cf_transformation(self.cf_feature)
            cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
            features.append(cf_feature_full)

        features = torch.stack(features, dim=-2)  # [bs, #modality, d]

        # multimodal fusion >>>
        final_feature = self.selfAttention(F.normalize(features, dim=-1))
        # multimodal fusion <<<

        # run iui graph gnn
        if self.conf['use_iui_graph']:
            # item_iui_gnn_feat, _ = self.iui_gnn_conv(
            #     self.item_iui_gnn_emb, 
            #     self.iui_graph, 
            #     return_attention_weights=True 
            # )  # [n_items, d]
            item_iui_gnn_feat, _ = self.iui_gnn_conv(
                a_feature=self.item_iui_gnn_emb,
                b_feature=self.item_iui_gnn_emb
            )
            # final_feature = final_feature + item_iui_gnn_feat

        # return final_feature # [n_items, d]

        if self.conf['use_iui_graph']:
            return final_feature, item_iui_gnn_feat

        return final_feature, final_feature

    def forward(self, seq_modify, all=False):
        # for bundle-view 
        if all is True:
            return self.forward_all()

        modify_mask = seq_modify == self.num_item
        seq_modify.masked_fill_(modify_mask, 0)

        final_feature, item_iui_gnn_feat = self.forward_all()
        final_feature = final_feature[seq_modify] 
        bs, n_token, d = final_feature.shape
        final_feature = final_feature.view(bs, n_token, d)
        # multimodal fusion <<<

        bundle_iui_gnn_feat = self.bundle_agg_graph_ori @ item_iui_gnn_feat  # [n_bundles, d]
        if self.conf['use_iui_graph']:
            return final_feature, bundle_iui_gnn_feat
        return final_feature, final_feature
    

    def generate_two_subs(self, dropout_ratio=0):
        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        # early-fusion
        mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        features = [mm_feature_full]
        features.append(self.item_embeddings)

        if self.conf['use_item_pretrained_emb']:
            cf_feature_full = self.cf_transformation(self.cf_feature)
            cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
            features.append(cf_feature_full)

        features = torch.stack(features, dim=-2)  # [bs, #modality, d]
        size = features.shape[:2]  # (bs, #modality)

        def random_mask():
            random_tensor = torch.rand(size).to(features.device)
            mask_bool = random_tensor < dropout_ratio  # the remainders are true
            masked_feat = features.masked_fill(mask_bool.unsqueeze(-1), 0)

            # multimodal fusion >>>
            final_feature = self.selfAttention(
                F.normalize(masked_feat, dim=-1))
            # multimodal fusion <<<
            return final_feature

        return random_mask(), random_mask()


class CLHE(nn.Module):
    def __init__(self, conf, raw_graph, features):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device
        self.num_user = self.conf["num_users"]
        self.num_bundle = self.conf["num_bundles"]
        self.num_item = self.conf["num_items"]
        self.embedding_size = 64
        self.ui_graph, self.bi_graph_train, self.bi_graph_seen = raw_graph
        self.item_augmentation = self.conf["item_augment"]

        self.encoder = HierachicalEncoder(conf, raw_graph, features)
        # decoder has the similar structure of the encoder
        if self.conf['view_mode'] == 'dual_view':
            self.decoder = HierachicalEncoder(conf, raw_graph, features)

        self.bundle_encode = TransformerEncoder(conf={
            "n_layer": conf["trans_layer"],
            "dim": 64,
            "num_token": 100,
            "device": self.device,
        }, data={"sp_graph": self.bi_graph_seen})

        self.cl_temp = conf['cl_temp']
        self.cl_alpha = conf['cl_alpha']

        self.bundle_cl_temp = conf['bundle_cl_temp']
        self.bundle_cl_alpha = conf['bundle_cl_alpha']
        self.cl_projector = nn.Linear(self.embedding_size, self.embedding_size)
        init(self.cl_projector)
        if self.item_augmentation in ["FD", "MD"]:
            self.dropout_rate = conf["dropout_rate"]
            self.dropout = nn.Dropout(p=self.dropout_rate)
        elif self.item_augmentation in ["FN"]:
            self.noise_weight = conf['noise_weight']

    def save_embedding(self, log_path):
        try:
            # print(f'log path: {log_path}') # ./save/pog/CLHE
            feat_retrival_view_path = os.path.join(log_path, 'item_feat_retrival_view.pt')
            feat_retrival_view = self.decoder(None, all=True) # run forward to get emb
            torch.save(feat_retrival_view, feat_retrival_view_path)
            print(f'saved {feat_retrival_view_path}')

            item_embedding_path = os.path.join(log_path, 'item_embedding.pt')
            # get embedding from encoder
            item_embedding = self.decoder.item_embeddings
            torch.save(item_embedding, item_embedding_path)
            print(f'saved {item_embedding_path}')
        except Exception as e:
            pass

    def forward(self, batch):
        idx, full, seq_full, modify, seq_modify = batch  # x: [bs, #items]
        mask = seq_full == self.num_item

        if self.conf['use_iui_graph']:
            feat_bundle_view, bundle_iui_gnn_feat = self.encoder(seq_full)  # [bs, n_token, d]
        else:
            feat_bundle_view, _ = self.encoder(seq_full)  # [bs, n_token, d]

        # bundle feature construction >>>
        bundle_feature = self.bundle_encode(feat_bundle_view, mask=mask)

        if self.conf['view_mode'] == 'dual_view':
            # feat_retrival_view = self.decoder(batch, all=True) # [n_items, d]
            if self.conf['use_iui_graph']:
                # return: final_feature, item_iui_gnn_feat
                feat_retrival_view, item_iui_gnn_feat = self.decoder(batch, all=True) # [n_items, d]
            else:
                feat_retrival_view, _ = self.decoder(batch, all=True) # [n_items, d]
        else:
            if self.conf['use_iui_graph']:
                feat_retrival_view, item_iui_gnn_feat = self.encoder(batch, all=True) # [n_items, d]
            else:
                feat_retrival_view, _ = self.encoder(batch, all=True) # [n_items, d]
        # self.feat_retrival_view = feat_retrival_view # to save model

        # fusion
        if self.conf['use_iui_graph']:
            feat_retrival_view = feat_retrival_view + item_iui_gnn_feat
            bundle_feature = bundle_feature + bundle_iui_gnn_feat[idx]

        # compute loss >>>
        logits = bundle_feature @ feat_retrival_view.transpose(0, 1)
        loss = recon_loss_function(logits, full)  # main_loss

        # # item-level contrastive learning >>>
        items_in_batch = torch.argwhere(full.sum(dim=0)).squeeze()
        item_loss = torch.tensor(0).to(self.device)
        if self.cl_alpha > 0:
            if self.item_augmentation == "FD":
                item_features = self.encoder(batch, all=True)[items_in_batch]
                sub1 = self.cl_projector(self.dropout(item_features))
                sub2 = self.cl_projector(self.dropout(item_features))
                item_loss = self.cl_alpha * cl_loss_function(
                    sub1.view(-1, self.embedding_size), sub2.view(-1, self.embedding_size), self.cl_temp)
            elif self.item_augmentation == "NA":
                item_features = self.encoder(batch, all=True)[items_in_batch]
                item_loss = self.cl_alpha * cl_loss_function(
                    item_features.view(-1, self.embedding_size), item_features.view(-1, self.embedding_size), self.cl_temp)
            elif self.item_augmentation == "FN":
                item_features, _ = self.encoder(batch, all=True)
                item_features = item_features[items_in_batch]
                sub1 = self.cl_projector(
                    self.noise_weight * torch.randn_like(item_features) + item_features)
                sub2 = self.cl_projector(
                    self.noise_weight * torch.randn_like(item_features) + item_features)
                item_loss = self.cl_alpha * cl_loss_function(
                    sub1.view(-1, self.embedding_size), sub2.view(-1, self.embedding_size), self.cl_temp)
            elif self.item_augmentation == "MD":
                sub1, sub2 = self.encoder.generate_two_subs(self.dropout_rate)
                sub1 = sub1[items_in_batch]
                sub2 = sub2[items_in_batch]
                item_loss = self.cl_alpha * cl_loss_function(
                    sub1.view(-1, self.embedding_size), sub2.view(-1, self.embedding_size), self.cl_temp)
        # # item-level contrastive learning <<<

        # bundle-level contrastive learning >>>
        bundle_loss = torch.tensor(0).to(self.device)
        if self.bundle_cl_alpha > 0:
            feat_bundle_view2, _ = self.encoder(seq_modify)  # [bs, n_token, d]
            bundle_feature2 = self.bundle_encode(feat_bundle_view2, mask=mask)
            bundle_loss = self.bundle_cl_alpha * cl_loss_function(
                bundle_feature.view(-1, self.embedding_size), bundle_feature2.view(-1, self.embedding_size), self.bundle_cl_temp)
        # bundle-level contrastive learning <<<

        learn_loss = loss + item_loss + bundle_loss if self.conf['loss_mode'] == 'full_loss' else loss
        return {
            # 'loss': loss + item_loss + bundle_loss,
            'loss': learn_loss,
            'item_loss': item_loss.detach(),
            'bundle_loss': bundle_loss.detach()
        }

    def evaluate(self, _, batch):
        idx, x, seq_x = batch
        mask = seq_x == self.num_item
        # feat_bundle_view = self.encoder(seq_x)

        if self.conf['use_iui_graph']:
            feat_bundle_view, bundle_iui_gnn_feat = self.encoder(seq_x)  # [bs, n_token, d]
        else:
            feat_bundle_view, _ = self.encoder(seq_x)  # [bs, n_token, d]

        bundle_feature = self.bundle_encode(feat_bundle_view, mask=mask)

        if self.conf['view_mode'] == 'dual_view':
            # feat_retrival_view = self.decoder((idx, x, seq_x, None, None), all=True)
            if self.conf['use_iui_graph']:
                feat_retrival_view, item_iui_gnn_feat = self.decoder((idx, x, seq_x, None, None), all=True)
            else:
                feat_retrival_view, _ = self.decoder((idx, x, seq_x, None, None), all=True)
        else:
            # feat_retrival_view = self.encoder((idx, x, seq_x, None, None), all=True)
            if self.conf['use_iui_graph']:
                feat_retrival_view, item_iui_gnn_feat = self.encoder((idx, x, seq_x, None, None), all=True)
            else:
                feat_retrival_view, _ = self.encoder((idx, x, seq_x, None, None), all=True)

        # fusion
        if self.conf['use_iui_graph']:
            feat_retrival_view = feat_retrival_view + item_iui_gnn_feat
            bundle_feature = bundle_feature + bundle_iui_gnn_feat[idx]
        
        # cal score 
        logits = bundle_feature @ feat_retrival_view.transpose(0, 1)

        return logits

    def propagate(self, test=False):
        return None

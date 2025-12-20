import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import TransformerEncoder
from collections import OrderedDict
import os

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
        # self.c_encoder = dense(self.content_feature)
        # self.t_encoder = dense(self.text_feature)

        self.c_encoder = MoE_Layer(
            input_dim=self.content_feature.shape[1],
            output_dim=self.embedding_size,
            num_experts=4,
            top_k=2,
            alpha_noise=conf['alpha_noise_moe']
        )
        self.t_encoder = MoE_Layer(
            input_dim=self.text_feature.shape[1],
            output_dim=self.embedding_size,
            num_experts=4,
            top_k=2,
            alpha_noise=conf['alpha_noise_moe']
        )

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

    def forward_all(self):
        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        features = [mm_feature_full]
        features.append(self.item_embeddings)

        cf_feature_full = self.cf_transformation(self.cf_feature)
        cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        # features.append(cf_feature_full)
        if self.conf['use_item_pretrained']:
            features.append(cf_feature_full)

        features = torch.stack(features, dim=-2)  # [bs, #modality, d]

        # multimodal fusion >>>
        final_feature = self.selfAttention(F.normalize(features, dim=-1))
        # multimodal fusion <<<

        return final_feature # [n_items, d]

    def forward(self, seq_modify, all=False):
        if all is True:
            return self.forward_all()

        modify_mask = seq_modify == self.num_item
        seq_modify.masked_fill_(modify_mask, 0)

        # c_feature = self.c_encoder(self.content_feature)
        # t_feature = self.t_encoder(self.text_feature)

        # mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        # # mm_feature = mm_feature_full[seq_modify]  # [bs, n_token, d]

        # features = []
        # features.append(mm_feature_full)
        # bi_feature_full = self.item_embeddings
        # # bi_feature = bi_feature_full[seq_modify]
        # features.append(bi_feature_full)

        # cf_feature_full = self.cf_transformation(self.cf_feature)
        # cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        # # cf_feature = cf_feature_full[seq_modify]
        # features.append(cf_feature_full)

        # features = torch.stack(features, dim=-2)  # [n_item, #modality, d]

        # # multimodal fusion >>>
        # final_feature = self.selfAttention(F.normalize(features, dim=-1))

        final_feature = self.forward_all()
        final_feature = final_feature[seq_modify] 
        bs, n_token, d = final_feature.shape
        final_feature = final_feature.view(bs, n_token, d)
        # multimodal fusion <<<

        return final_feature

    def generate_two_subs(self, dropout_ratio=0):
        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        # early-fusion
        mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        features = [mm_feature_full]

        features.append(self.item_embeddings)

        cf_feature_full = self.cf_transformation(self.cf_feature)
        cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        if self.conf['use_item_pretrained']:
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


class MLP_(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(MLP_, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        init(self.fc1)
        init(self.fc2)

    def forward(self, x):
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x
    
class MoE_Layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, top_k=2, alpha_noise=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        assert top_k <= num_experts, "top_k must be less than or equal to num_experts"

        """
        nn.Linear(self.bundle_sum_emb.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, self.embedding_size)
        """
        self.experts = torch.nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )
            # nn.Linear(input_dim, output_dim)
            for _ in range(num_experts)
        ])
        
        self.w_gate = nn.Parameter(
            torch.zeros(input_dim, num_experts), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(input_dim, num_experts), requires_grad=True 
        )
        self.alpha_noise = alpha_noise

        self.softplus = nn.Softplus()

    def forward(self, x, noise_epsilon=1e-2):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # gate_logits = self.gate(x)
        # add noise to gate logits for exploration
        # 1e-1: good

        # noise = torch.randn_like(gate_logits) * self.alpha_noise
        # noise = torch.randn_like(gate_logits)
        # gate_logits = gate_logits + noise

        clean_logits = x @ self.w_gate
        if self.training:
            raw_noise_std = x @ self.w_noise
            noise_std = self.softplus(raw_noise_std) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_std
            logits = noisy_logits
        else:
            logits = clean_logits

        aux_loss = self._compute_load_balancing_loss(logits)

        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=1)
        
        topk_weights = F.softmax(topk_logits, dim=1)  # [batch_size, top_k]
        
        selected_experts = expert_outputs.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)))
        
        topk_weights = topk_weights.unsqueeze(-1)  # [batch_size, top_k, 1]
        output = torch.sum(topk_weights * selected_experts, dim=1)  # [batch_size, output_dim]
        
        return output, aux_loss
        # return output 
    
    def _compute_load_balancing_loss(self, gate_logits):
        gates = F.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]
        
        importance_per_expert = gates.mean(dim=0)  # [num_experts]

        target_importance = torch.ones_like(importance_per_expert) / self.num_experts
        importance_loss = F.kl_div(
                importance_per_expert.log(),
                target_importance,
                reduction='sum'
        )
        
        return importance_loss

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

        # load bundle summary emb:
        self.bundle_sum_emb = torch.load(
            os.path.join('datasets', conf['dataset'], f'{conf["dataset"]}_bundle_sum_emb.pt')
        ).to(device)
        print(f'bundle emb shape: {self.bundle_sum_emb.shape}')

        if conf['type_adapter'] == 'linear':
            self.bundle_adapter = nn.Linear(
                self.bundle_sum_emb.shape[1], self.embedding_size
            )
        if conf['type_adapter'] == 'MLP':
            # self.bundle_adapter = MLP_(
            #     input_dim=self.bundle_sum_emb.shape[1], # 384 
            #     hidden_dim=64,
            #     output_dim=self.embedding_size,
            #     dropout=0.2
            # )
            self.bundle_adapter = nn.Sequential(
                nn.Linear(self.bundle_sum_emb.shape[1], 128),
                nn.ReLU(),
                nn.Linear(128, self.embedding_size)
            )
        if conf['type_adapter'] == 'MoE':
            self.bundle_adapter = MoE_Layer(
                input_dim=self.bundle_sum_emb.shape[1], # 384 
                output_dim=self.embedding_size,
                num_experts=4,
                top_k=2,
                alpha_noise=conf['alpha_noise_moe']
            )

        # bundle sum alpha
        # self.bundle_sum_alpha=0.2
        self.bundle_sum_alpha = conf['alpha_bundle_sum']
        self.alpha_balance_loss = conf['alpha_balance_loss']

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
        feat_bundle_view = self.encoder(seq_full)  # [bs, n_token, d]

        # bundle feature construction >>>
        bundle_feature = self.bundle_encode(feat_bundle_view, mask=mask)

        if self.conf['view_mode'] == 'dual_view':
            feat_retrival_view = self.decoder(batch, all=True) # [n_items, d]
        else:
            feat_retrival_view = self.encoder(batch, all=True) # [n_items, d]
        # self.feat_retrival_view = feat_retrival_view # to save model

        if self.conf['type_adapter'] == 'MoE':
            bundle_sum_emb, balance_loss = self.bundle_adapter(self.bundle_sum_emb[idx])  # [n_bundles, d]
        else:
            bundle_sum_emb = self.bundle_adapter(self.bundle_sum_emb[idx])  # [n_bundles, d]
        bundle_feature = bundle_feature + self.bundle_sum_alpha*bundle_sum_emb

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
                item_features = self.encoder(batch, all=True)[items_in_batch]
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
            feat_bundle_view2 = self.encoder(seq_modify)  # [bs, n_token, d]
            bundle_feature2 = self.bundle_encode(feat_bundle_view2, mask=mask)
            bundle_sum_emb = self.bundle_adapter(self.bundle_sum_emb[idx])  # [n_bundles, d]
            bundle_feature2 = bundle_feature2 + self.bundle_sum_alpha*bundle_sum_emb
            bundle_loss = self.bundle_cl_alpha * cl_loss_function(
                bundle_feature.view(-1, self.embedding_size), bundle_feature2.view(-1, self.embedding_size), self.bundle_cl_temp)
        # bundle-level contrastive learning <<<

        learn_loss = loss + item_loss + bundle_loss + self.alpha_balance_loss*balance_loss if self.conf['loss_mode'] == 'full_loss' else loss
        return {
            # 'loss': loss + item_loss + bundle_loss,
            'loss': learn_loss,
            'item_loss': item_loss.detach(),
            'bundle_loss': bundle_loss.detach()
        }

    def evaluate(self, _, batch):
        idx, x, seq_x = batch
        mask = seq_x == self.num_item
        feat_bundle_view = self.encoder(seq_x)

        bundle_feature = self.bundle_encode(feat_bundle_view, mask=mask)
        bundle_sum_emb = self.bundle_adapter(self.bundle_sum_emb[idx])  # [n_bundles, d]
        if self.conf['type_adapter'] == 'MoE':
            bundle_sum_emb, _ = bundle_sum_emb  # unpack output from MoE
    
        bundle_feature = bundle_feature + self.bundle_sum_alpha*bundle_sum_emb

        if self.conf['view_mode'] == 'dual_view':
            feat_retrival_view = self.decoder((idx, x, seq_x, None, None), all=True)
        else:
            feat_retrival_view = self.encoder((idx, x, seq_x, None, None), all=True)

        logits = bundle_feature @ feat_retrival_view.transpose(0, 1)

        return logits

    def propagate(self, test=False):
        return None
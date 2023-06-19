"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import utils
from time import time
from Similar import RegularSimilar
import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg, unique_user=None, pos_item_index=None, pos_item_mask=None):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
            unique_user: unique user list
            pos_item_index: positive items for corresponding users
            pos_item_mask: positive items mask for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class PureMF(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.user_privacy_ration = dataset.userPrivacySetting
        self.latent_dim = config['latent_dim_rec']
        self.replace_ratio = config['replace_ratio']
        # init modules
        self.regularSimilar = RegularSimilar(self.latent_dim, dataset.userSimMax, dataset.userSimMin)
        self.select_layer = nn.Sequential(
            nn.Linear(2 * self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, 1),
            nn.LeakyReLU()
        )
        self.feature_transform = nn.Sequential(
            nn.BatchNorm1d(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        self.feature_loss = nn.MSELoss()

        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

        # select node to replace according the item score
    def sample_low_score_pos_item(self, users, sorted_item_score, pos_item_index, all_users, all_items, train_pos):
        users = users.detach().cpu().numpy()
        sorted_pos_score = sorted_item_score[0].detach().cpu().numpy()
        sorted_pos_index = sorted_item_score[1].detach().cpu().numpy()
        pos_item_index = pos_item_index.long().detach().cpu().numpy()
        train_pos = train_pos.long().detach().cpu().numpy()
        # select the item that less contribute to user preference
        need_replace = utils.construct_need_replace_user_item(
            users, sorted_pos_score, sorted_pos_index,
            pos_item_index, self.replace_ratio,
            train_pos
        )
        need_replace = np.array(need_replace)

        if need_replace.shape[0] == 0:
            return need_replace, torch.tensor([]).cuda(), torch.tensor([]).cuda(), 0., 0.


        # get user and item list
        users_index = need_replace[:, 0]
        items_index = need_replace[:, 1]
        all_items = all_items.detach()
        # get user and item embeddings
        users_emb = all_users[users_index].detach()
        items_emb = all_items[items_index]
        # get user privacy setting
        privacy_settings = self.user_privacy_ration[users_index]
        need_replace_feature = torch.cat([users_emb, items_emb], dim=1)
        # free needless paras
        del pos_item_index
        del users_emb
        del items_emb
        del all_users

        # replace item that less contribute to user preference via generator
        replaceable_items, replaceable_items_feature, similarity_loss, similarity = \
            self.regularSimilar.choose_replaceable_item(need_replace, need_replace_feature, all_items, privacy_settings)

        return need_replace, replaceable_items, replaceable_items_feature, similarity_loss, similarity

    # get the score of user with positive items.
    def computer_pos_score(self, users, pos_item_index, pos_item_mask, train_pos):
        # start_time = time()
        all_users = self.embedding_user.weight
        all_items = self.embedding_item.weight
        users = torch.tensor(list(users)).long()
        pos_item_index = torch.from_numpy(pos_item_index).cuda()
        pos_item_mask = torch.from_numpy(pos_item_mask).cuda()
        max_len = pos_item_index.size(1)
        batch_size = pos_item_index.size(0)
        # get the mebedding of positive items and reshape them.
        pos_emb = all_items[pos_item_index.long()].detach()
        users_emb = all_users[users].detach()
        users_expand_emb = users_emb.view(batch_size, 1, self.latent_dim)
        users_expand_emb = users_expand_emb.expand(batch_size, max_len, self.latent_dim)
        user_pos_item_feature = torch.cat([users_expand_emb, pos_emb], dim=-1)
        # user_pos_item_feature = user_pos_item_feature.reshape(-1, 2 * self.latent_dim)
        # get the attention vector between the positive items and user.
        user_item_scores = self.select_layer(user_pos_item_feature)
        del user_pos_item_feature
        user_item_scores = user_item_scores.reshape(batch_size, max_len)
        # fill the valid items using a smaller score.
        user_item_scores = user_item_scores.masked_fill(mask=(pos_item_mask == 0), value=-1e9)
        attention_vector = F.softmax(user_item_scores, dim=-1)

        # sort the item scores
        sorted_pos_cores = torch.sort(attention_vector, dim=1, descending=True)

        # select the item that less contribute to user preference
        need_replace, replaceable_items, replaceable_items_feature, similarity_loss, similarity = \
            self.sample_low_score_pos_item(users, sorted_pos_cores, pos_item_index, all_users, all_items, train_pos)

        # compute the feature loss between the united item embeeding and user embeeding
        pos_emb = pos_emb * attention_vector.reshape(batch_size, max_len, 1)
        user_items_feature = pos_emb.sum(dim=1) # littlemilk: 缺少 (3) equation 中的 |I_u|
        user_items_transform_feature = self.feature_transform(user_items_feature)
        feature_loss = self.feature_loss(user_items_transform_feature, users_emb) # littlemilk: MSELoss


        del users_emb
        del user_items_feature
        del user_items_transform_feature

        return need_replace, replaceable_items, replaceable_items_feature, similarity_loss, similarity, feature_loss

    # replace the need item with selected items
    def replace_pos_items(self, users, pos, neg, need_replace, replaceable_items):
        # littlemilk: 沒有用到 replaceable item 的原因是，後面才會將聚這個 mask 去 replaceable_item_features 撈資料（model.py 213）
        numpy_users = users.detach().cpu().numpy()
        numpy_pos = pos.detach().cpu().numpy()

        pos_mask, replaceable_mask = \
            utils.replace_original_to_replaceable(numpy_users, numpy_pos, need_replace)

        users = users[pos_mask == 1]
        neg = neg[pos_mask == 1]

        return users, neg, replaceable_mask

    def get_original_bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos)
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))

        return loss, reg_loss


    def bpr_loss(self, users, pos, neg, unique_user=None, pos_item_index=None, pos_item_mask=None):
        # start_time = time()
        CF1_loss, CF1_reg_loss = self.get_original_bpr_loss(users, pos, neg)
        # get the need item list and replace item list
        need_replace, replaceable_items, replaceable_items_feature, similarity_loss, similarity, std_loss = \
            self.computer_pos_score(unique_user, pos_item_index, pos_item_mask, pos)
        # replace items
        users, neg, replaceable_mask \
            = self.replace_pos_items(users, pos, neg, need_replace, replaceable_items)
        # recompute the bpr loss of the replaced items.
        users_emb = self.embedding_user(users.long()).detach()
        pos_emb = replaceable_items_feature[replaceable_mask]
        neg_emb = self.embedding_item(neg.long()).detach()
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        CF2_loss = torch.mean(nn.functional.softplus(- pos_scores)) # littlemilk: why mean? / softplus 相當於先做 sigmoid 再做 ln
        # CF2_loss = torch.mean(torch.log(1 + torch.sigmoid(- pos_scores)))
        return CF1_loss, CF1_reg_loss, CF2_loss, similarity_loss, similarity, std_loss


    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)

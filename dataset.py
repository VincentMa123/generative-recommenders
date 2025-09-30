import gc
import json
import pickle
import struct
import threading
from collections import defaultdict
from pathlib import Path
import random

import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F

class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, interaction_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.interaction_dir = Path(interaction_dir)
        self._load_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        self.item_feat_dict = json.load(open(Path(interaction_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.dwell_time_bins = torch.tensor([3, 10, 30, 60, 180], dtype=torch.float32)

        self.interaction_feature_ids = ['101', '117', '100']
        self.interaction_vocab_dict = {}  # Use a dict to hold all vocabs
        print("Loading all interaction vocabularies...")
        for feature_id in self.interaction_feature_ids:
            vocab_path = self.interaction_dir / f'interaction_vocab_{feature_id}.pkl'
            if vocab_path.exists():
                with open(vocab_path, 'rb') as f:
                    self.interaction_vocab_dict[feature_id] = pickle.load(f)
                print(f"  ✅ Loaded vocab for feature '{feature_id}'")
            else:
                print(f"  [WARNING] Vocab for feature '{feature_id}' not found. That interaction will be skipped.")

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

        self._thread_local = threading.local()
        self.all_feature_ids = set()
        for ids in self.feature_types.values():
            self.all_feature_ids.update(ids)


    def _load_offsets(self):
        """
        只加载偏移量，不保持文件句柄打开
        """
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)
        # Store the file path for later use
        self.data_file_path = self.data_dir / "seq.jsonl"

    def _get_file_handle(self):
        """
        获取当前线程的文件句柄，如果不存在则创建
        """
        if not hasattr(self._thread_local, 'data_file'):
            self._thread_local.data_file = open(self.data_file_path, 'rb')
        return self._thread_local.data_file

    def _load_user_data(self, uid):
        """
        线程安全地从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据
        """
        file_handle = self._get_file_handle()
        file_handle.seek(self.seq_offsets[uid])
        line = file_handle.readline()
        data = json.loads(line)
        return data

    def __del__(self):
        """
        清理线程本地文件句柄
        """
        if hasattr(self, '_thread_local') and hasattr(self._thread_local, 'data_file'):
            try:
                self._thread_local.data_file.close()
            except:
                pass

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def _prepare_features(self, feature_list):
        """
        Converts a list of feature dictionaries into a dictionary of padded numpy arrays.
        This logic is moved from the model's feat2tensor method.
        """
        prepared_feats = {k: [] for k in self.all_feature_ids}

        # Determine max array lengths for padding
        max_array_lengths = defaultdict(int)
        array_feat_ids = set(self.feature_types.get('user_array', []) + self.feature_types.get('item_array', []))

        for feat_dict in feature_list:
            for k, v in feat_dict.items():
                if k in array_feat_ids:
                    max_array_lengths[k] = max(max_array_lengths[k], len(v))

        # Populate and pad features
        for feat_dict in feature_list:
            for k in self.all_feature_ids:
                val = feat_dict.get(k, self.feature_default_value[k])
                if k in array_feat_ids:
                    padded_val = np.zeros(max_array_lengths[k], dtype=np.int64)
                    actual_len = min(len(val), max_array_lengths[k])
                    if actual_len > 0:
                        padded_val[:actual_len] = val[:actual_len]
                    prepared_feats[k].append(padded_val)
                elif k in self.feature_types.get('item_emb', []):
                    # Ensure consistent shape for multimodal embeddings
                    emb_shape = self.mm_emb_dict[k][next(iter(self.mm_emb_dict[k]))].shape
                    if isinstance(val, list) and len(val) == 1 and val[0] == 0:  # Default value
                        prepared_feats[k].append(np.zeros(emb_shape, dtype=np.float32))
                    else:
                        prepared_feats[k].append(val)
                else:
                    prepared_feats[k].append(val)

        # Convert lists to numpy arrays
        for k, v in prepared_feats.items():
            try:
                prepared_feats[k] = np.array(v)
            except Exception as e:
                print(f"Error converting feature '{k}' to numpy array: {e}")
                # Handle potential inconsistencies, e.g., by creating an empty array
                # This part may need debugging based on your specific data.
                if prepared_feats[k]:
                    # Fallback for complex object arrays
                    dt = object if not isinstance(prepared_feats[k][0], (int, float)) else type(prepared_feats[k][0])
                    prepared_feats[k] = np.array(v, dtype=dt)
                else:
                    prepared_feats[k] = np.array([])

        return prepared_feats

    def __getitem__(self, uid):
        """
        Modified to return features as padded numpy arrays instead of lists of dicts.
        """
        # ... (All the logic at the beginning of __getitem__ to build ext_user_sequence remains the same) ...
        user_sequence = self._load_user_data(uid)
        user_reid = uid
        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type, timestamp))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type, timestamp))

        seq = np.zeros([self.maxlen + 1], dtype=np.int64)
        pos = np.zeros([self.maxlen + 1], dtype=np.int64)
        neg = np.zeros([self.maxlen + 1], dtype=np.int64)
        dwell_bins = np.zeros([self.maxlen + 1], dtype=np.int64)
        targets = np.zeros_like(seq)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int64)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int64)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int64)
        action_type_arr = np.zeros([self.maxlen + 1], dtype=np.int64)
        ts = np.zeros([self.maxlen + 1], dtype=np.int64)

        # These are now lists of dicts, which will be processed at the end
        raw_seq_feat = [self.feature_default_value] * (self.maxlen + 1)
        raw_pos_feat = [self.feature_default_value] * (self.maxlen + 1)
        raw_neg_feat = [self.feature_default_value] * (self.maxlen + 1)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts_set = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts_set.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type, timestamp = record_tuple
            next_i, next_feat, next_type, next_act_type, next_timestamp = nxt
            ts[idx] = timestamp
            feat = self.fill_missing_feat(feat, i)
            if timestamp > 0:
                feat = self.add_time_features(feat, timestamp)

            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            action_type_arr[idx] = act_type if act_type is not None else 0
            if next_act_type is not None:
                next_action_type[idx] = next_act_type

            raw_seq_feat[idx] = feat

            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                raw_pos_feat[idx] = next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts_set)
                neg[idx] = neg_id
                raw_neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)

            if type_ == 1:
                for feature_id in self.interaction_feature_ids:
                    if feature_id in self.interaction_vocab_dict and feature_id in feat:
                        feature_value = feat[feature_id]
                        interaction_pair = (user_reid, feature_value)
                        interaction_id = self.interaction_vocab_dict[feature_id].get(interaction_pair, 0)
                        feat[f'interaction_user_{feature_id}'] = interaction_id

            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        # Dwell time calculation
        valid_ts_mask = (ts > 0)
        dwell_times = np.zeros_like(ts, dtype=np.float32)
        dwell_times[valid_ts_mask] = np.append(np.diff(ts[valid_ts_mask]), 0)
        dwell_bins_tensor = torch.bucketize(torch.from_numpy(dwell_times), self.dwell_time_bins) + 1
        dwell_bins_tensor[torch.from_numpy(ts == 0)] = 0
        dwell_bins = dwell_bins_tensor.numpy()

        targets[:-1] = seq[1:]

        # *** KEY CHANGE: Process features into padded numpy arrays here ***
        seq_feat_prepared = self._prepare_features(raw_seq_feat)
        pos_feat_prepared = self._prepare_features(raw_pos_feat)
        neg_feat_prepared = self._prepare_features(raw_neg_feat)

        return (seq, targets, pos, neg, token_type, next_token_type,
                next_action_type, seq_feat_prepared, pos_feat_prepared, neg_feat_prepared,
                ts, action_type_arr, dwell_bins)

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []
        feat_types['item_sparse'] += ['300', '301', '302']

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        feat_statistics['300'] = 16
        feat_statistics['301'] = 16
        feat_statistics['302'] = 16

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.feature_types.valudataset.pyes():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    @staticmethod
    def collate_fn(batch):
        """
        Modified to handle padding of array features at the batch level.
        """
        # Unzip the batch
        (seq, targets, pos, neg, token_type, next_token_type,
         next_action_type, seq_feat, pos_feat, neg_feat, ts, action_type, dwell_bins) = zip(*batch)

        # Stack simple numpy arrays into tensors (this part is fine)
        seq = torch.from_numpy(np.array(seq))
        targets = torch.from_numpy(np.array(targets))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        action_type = torch.from_numpy(np.array(action_type))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        dwell_bins = torch.from_numpy(np.array(dwell_bins))
        ts = torch.from_numpy(np.stack(ts, axis=0))

        # --- KEY CHANGE: Intelligent collation for feature dictionaries ---

        def collate_feature_dicts(feature_dicts_tuple):
            """
            Collates a tuple of feature dictionaries, handling padding for arrays.
            """
            collated = {}
            # Get all feature keys from the first item in the batch
            feature_keys = feature_dicts_tuple[0].keys()

            for k in feature_keys:
                # Get the list of arrays for this feature across the batch
                arrays = [d[k] for d in feature_dicts_tuple]

                # Check if the arrays have more than 1 dimension.
                # Sparse features will be 1D (seq_len,) and won't need this padding.
                # Array features will be 2D (seq_len, num_tags) and will need it.
                if arrays[0].ndim > 1:
                    # This is an array-like feature that might have a ragged dimension.
                    # Find the maximum size of the last dimension across the batch.
                    max_last_dim = 0
                    for arr in arrays:
                        max_last_dim = max(max_last_dim, arr.shape[-1])

                    # Pad each array to this max size and collect them
                    padded_arrays = []
                    for arr in arrays:
                        padding_needed = max_last_dim - arr.shape[-1]
                        if padding_needed > 0:
                            # np.pad format: ((before_dim1, after_dim1), (before_dim2, after_dim2), ...)
                            pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, padding_needed)]
                            padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
                            padded_arrays.append(padded_arr)
                        else:
                            padded_arrays.append(arr)

                    # Now that all arrays are the same shape, we can stack them
                    collated[k] = torch.from_numpy(np.stack(padded_arrays))
                else:
                    # This is a sparse feature, shapes should already match.
                    collated[k] = torch.from_numpy(np.stack(arrays))

            return collated

        collated_seq_feat = collate_feature_dicts(seq_feat)
        collated_pos_feat = collate_feature_dicts(pos_feat)
        collated_neg_feat = collate_feature_dicts(neg_feat)

        return (seq, targets, pos, neg, token_type, next_token_type,
                next_action_type, collated_seq_feat, collated_pos_feat, collated_neg_feat,
                ts, action_type, dwell_bins)

    def add_time_features(self, feat, timestamp):
        """Add time-based features to item features"""
        if timestamp > 0:
            dt = datetime.fromtimestamp(timestamp)
            feat['hour'] = dt.hour
            feat['weekday'] = dt.weekday()
            feat['is_weekend'] = int(dt.weekday() >= 5)
        return feat


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

        self.id_to_retrieval_id = {
            internal_id: creative_id
            for creative_id, internal_id in self.indexer['i'].items()
        }

        # Also add a mapping for the padding token
        self.id_to_retrieval_id[0] = 0  # Or another suitable placeholder

        print(f"✅ Mapping created successfully. Found {len(self.id_to_retrieval_id) - 1} mappings.")

        self.oov_indices = {}
        all_sparse_array_ids = (
                self.feature_types['user_sparse'] + self.feature_types['item_sparse'] +
                self.feature_types['user_array'] + self.feature_types['item_array']
        )
        for feat_id in all_sparse_array_ids:
            self.oov_indices[feat_id] = len(self.indexer['f'][feat_id]) + 1



    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，
        现在转换为专用的OOV token index。
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            # If this feature doesn't have an OOV index (e.g., it's a continual feature),
            # just copy its value.
            if feat_id not in self.oov_indices:
                processed_feat[feat_id] = feat_value
                continue

            oov_token_index = self.oov_indices[feat_id]

            if isinstance(feat_value, list):
                # For array features, replace any string with the OOV index
                value_list = [v if not isinstance(v, str) else oov_token_index for v in feat_value]
                processed_feat[feat_id] = value_list
            elif isinstance(feat_value, str):
                # For sparse features, replace the string with the OOV index
                processed_feat[feat_id] = oov_token_index
            else:
                # The value is already a valid integer index
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type, timestamp))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1, action_type, timestamp))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        time = np.zeros([self.maxlen + 1], dtype=np.int32)
        action_type = np.zeros([self.maxlen + 1], dtype=np.int64)

        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type, timestamp = record_tuple
            feat = self.fill_missing_feat(feat, i)
            time[idx] = timestamp
            action_type[idx] = act_type if act_type is not None else 0
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        valid_ts_mask = (ts > 0)
        dwell_times = np.zeros_like(ts, dtype=np.float32)
        if np.any(valid_ts_mask):
            valid_timestamps = ts[valid_ts_mask]
            # Calculate differences and pad the last element with 0
            dwell_times[valid_ts_mask] = np.append(np.diff(valid_timestamps), 0)

        # Bucketize the dwell times using the bins defined in __init__
        dwell_bins_tensor = torch.bucketize(torch.from_numpy(dwell_times), self.dwell_time_bins) + 1

        # Set padded positions (where timestamp was 0) back to bin 0
        dwell_bins_tensor[torch.from_numpy(ts == 0)] = 0
        dwell_bins = dwell_bins_tensor.numpy()


        return seq, token_type, seq_feat, user_id, time, action_type, dwell_bins

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            user_id: user_id, str
        """
        seq, token_type, seq_feat, user_id, timestamp, action_type, dwell_bins = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        action_type = torch.from_numpy(np.array(action_type))
        seq_feat = list(seq_feat)
        timestamp = list(timestamp)
        timestamp = np.stack(timestamp, axis=0)
        timestamp = torch.from_numpy(timestamp)
        return seq, token_type, seq_feat, user_id, timestamp, action_type, dwell_bins




def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict


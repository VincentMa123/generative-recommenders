from embedding import SharedEmbeddingConfig, SharedEmbeddingModule
import argparse
import json
import os
import struct
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset2 import MyTestDataset, save_emb
from hydra import Hydra
from hstu import CrossAttentionReranker

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith("unified_recommendation_system.pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--maxlen', default=50, type=int)

    # Enhanced Model construction
    parser.add_argument('--hstu_hidden_units', default=60, type=int)
    parser.add_argument('--hstu_num_epochs', default=3, type=int)
    parser.add_argument('--hstu_num_heads', default=2, type=int)
    parser.add_argument('--hstu_dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=1e-6, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_actions', default=5, type=int, help='Number of distinct action types.')
    parser.add_argument('--num_candidates', default=500, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    # HSTU specific
    parser.add_argument('--num_local_blocks', default=2, type=int)
    parser.add_argument('--num_global_blocks', default=2, type=int)
    parser.add_argument('--time_span', default=512, type=int)  # For time embeddings
    # Hydra
    parser.add_argument('--hydra_hidden_units', default=60, type=int)
    parser.add_argument('--hydra_num_blocks', default=2, type=int)
    parser.add_argument('--hydra_num_epochs', default=3, type=int)
    parser.add_argument('--hydra_num_heads', default=2, type=int)
    parser.add_argument('--hydra_dropout_rate', default=0.2, type=float)

    # Training enhancements
    parser.add_argument('--warmup_steps', default=1000, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--accumulation_steps', default=1, type=int,
                        help='Accumulate gradients over N steps to simulate larger batch size.')

    parser.add_argument(
        '--evaluate',
        type=str,
        default='true',
        choices=['true', 'false'],
        help='Enable train/test split and FINAL evaluation only (true/false)',
    )
    parser.add_argument('--eval_batch_size', default=64, type=int)
    parser.add_argument('--test_split', default=0.1, type=float)

    # Validation during training (optional, separate from test set)
    parser.add_argument(
        '--use_validation',
        type=str,
        default='true',
        choices=['true', 'false'],
        help='Use validation set for monitoring during training (true/false)',
    )
    parser.add_argument('--val_split', default=0.1, type=float)
    parser.add_argument(
        '--val_every', default=1, type=int, help='Validate every N epochs'
    )

    # Early stopping (only works with validation)
    parser.add_argument(
        '--early_stopping', type=str, default='true', choices=['true', 'false']
    )
    parser.add_argument('--patience', default=3, type=int)

    # Other improvements
    parser.add_argument(
        '--mm_emb_id',
        nargs='+',
        default=['81', '82'],
        type=str,
        choices=[str(s) for s in range(81, 87)],
    )
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument(
        '--norm_first',
        type=str,
        default='true',
        choices=['true', 'false'],
        help='Use pre-LayerNorm architecture for stability (true/false)',
    )

    # Debugging
    parser.add_argument(
        '--debug',
        type=str,
        default='false',
        choices=['true', 'false'],
        help='Enable detailed debugging for a specific step.',
    )
    parser.add_argument(
        '--debug_step',
        type=int,
        default=0,
        help='The global step number to activate debugging on.',
    )
    parser.add_argument(
        '--exit_after_debug',
        type=str,
        default='true',
        choices=['true', 'false'],
        help='If true, the program will exit after the debug step is complete.',
    )

    args = parser.parse_args()
    # Convert string booleans to actual booleans
    args.evaluate = args.evaluate.lower() == 'true'
    args.use_validation = args.use_validation.lower() == 'true'
    args.early_stopping = args.early_stopping.lower() == 'true'
    args.norm_first = args.norm_first.lower() == 'true'
    args.debug = args.debug.lower() == 'true'
    args.exit_after_debug = args.exit_after_debug.lower() == 'true'

    return args


def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (num_points_query and FLAGS_query_ann_top_k)
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # Calculate how many result_ids there are (num_points_query * query_ann_top_k)
        num_result_ids = num_points_query * query_ann_top_k

        # Read result_ids (uint64_t, 8 bytes per value)
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))


def process_cold_start_feat(feat):
    """
    处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
    """
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if type(feat_value) == list:
            value_list = []
            for v in feat_value:
                if type(v) == str:
                    value_list.append(0)
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif type(feat_value) == str:
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


def load_rqvae_model(checkpoint, args):
    """Load and initialize RQ-VAE model from checkpoint"""
    from model_rqvae import RQVAE

    # Initialize RQ-VAE with same parameters as training
    rqvae_model = RQVAE(
        input_dim=32,  # Adjust based on your embedding dimension
        hidden_channels=[16, 16],
        latent_dim=16,
        num_codebooks=3,
        codebook_size=[16, 16, 16],
        shared_codebook=False,
        kmeans_method="kmeans",
        kmeans_iters=20,
        distances_method="cosine",
        loss_beta=0.25,
        device=args.device,
    ).to(args.device)

    # Load the trained weights
    rqvae_state_dict = checkpoint.get('rqvae_state_dict')
    if rqvae_state_dict:
        # Initialize codebooks first by creating dummy parameters
        for i in range(3):  # num_codebooks
            codebook_key = f"rq.vqmodules.{i}.codebook"
            if codebook_key in rqvae_state_dict:
                # Create the codebook parameter in the model
                codebook_tensor = rqvae_state_dict[codebook_key]
                rqvae_model.rq.vqmodules[i].codebook = torch.nn.Parameter(
                    codebook_tensor.clone().to(args.device)
                )

        # Load the state dict with strict=False to handle any remaining mismatches
        missing_keys, unexpected_keys = rqvae_model.load_state_dict(rqvae_state_dict, strict=False)

        if missing_keys:
            print(f"Missing keys in RQ-VAE model: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in RQ-VAE state dict: {unexpected_keys}")

        rqvae_model.eval()
        print("✅ RQ-VAE model loaded successfully")
        return rqvae_model
    else:
        print("⚠️ WARNING: No RQ-VAE state dict found in checkpoint")
        return None


def generate_semantic_ids_for_candidates(rqvae_model, candidate_features, mm_emb_dict, args):
    """Generate semantic IDs for candidate items using RQ-VAE"""
    if rqvae_model is None:
        print("⚠️ RQ-VAE model not available, skipping semantic ID generation")
        return candidate_features

    EMB_FEATURE_ID = '81'  # Adjust based on your multimodal embedding feature
    batch_size = 1024

    # Collect embeddings for candidates that have multimodal features
    candidate_embs = []
    candidate_keys = []

    for creative_id, features in candidate_features.items():
        if EMB_FEATURE_ID in features and creative_id in mm_emb_dict[EMB_FEATURE_ID]:
            emb = torch.tensor(mm_emb_dict[EMB_FEATURE_ID][creative_id], dtype=torch.float32)
            candidate_embs.append(emb)
            candidate_keys.append(creative_id)

    if not candidate_embs:
        print("⚠️ No multimodal embeddings found for candidates")
        return candidate_features

    # Generate semantic IDs in batches
    candidate_embs = torch.stack(candidate_embs).to(args.device)
    all_semantic_ids = []

    with torch.no_grad():
        for i in range(0, len(candidate_embs), batch_size):
            batch_embs = candidate_embs[i:i + batch_size]
            semantic_ids = rqvae_model._get_codebook(batch_embs)
            all_semantic_ids.append(semantic_ids.cpu())

    all_semantic_ids = torch.cat(all_semantic_ids, dim=0).numpy().astype(int)

    # Add semantic IDs to candidate features
    enhanced_features = candidate_features.copy()
    for i, creative_id in enumerate(candidate_keys):
        if creative_id in enhanced_features:
            enhanced_features[creative_id]["300"] = int(all_semantic_ids[i, 0])
            enhanced_features[creative_id]["301"] = int(all_semantic_ids[i, 1])
            enhanced_features[creative_id]["302"] = int(all_semantic_ids[i, 2])

    print(f"✅ Generated semantic IDs for {len(candidate_keys)} candidate items")
    return enhanced_features


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model, rqvae_model, args):
    """
    Enhanced version that generates semantic IDs for candidate items
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}
    candidate_features_dict = {}

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0

            # Process missing fields
            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]

            # Add multimodal embeddings
            for feat_id in feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            candidate_features_dict[creative_id] = feature
            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # ✨ NEW: Generate semantic IDs for candidates using RQ-VAE
    enhanced_candidate_features = generate_semantic_ids_for_candidates(
        rqvae_model, candidate_features_dict, mm_emb_dict, args
    )

    # Update features list with enhanced features
    enhanced_features = []
    for creative_id in creative_ids:
        enhanced_features.append(enhanced_candidate_features[creative_id])

    # Save embeddings with enhanced features
    model.save_item_emb(item_ids, retrieval_ids, enhanced_features, os.environ.get('EVAL_RESULT_PATH'))

    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)

    return retrieve_id2creative_id, enhanced_candidate_features


def infer():
    args = get_args()
    data_path = os.environ.get('EVAL_DATA_PATH')
    ckpt_path = get_ckpt_path()

    checkpoint = torch.load(ckpt_path, map_location=torch.device(args.device), weights_only=False)

    interaction_vocabs = checkpoint.get('interaction_vocabs', {})
    if interaction_vocabs:
        print("✅ Interaction vocabularies loaded from checkpoint.")
    else:
        print("⚠️ WARNING: No interaction vocabularies found in checkpoint.")

    rqvae_model = load_rqvae_model(checkpoint, args)

    test_dataset = MyTestDataset(data_path, args)
    test_dataset.interaction_vocab_dict = interaction_vocabs
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=0,
        collate_fn=test_dataset.collate_fn
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types

    SharedEmbeddingConfig(args)
    shared_embeddings = SharedEmbeddingModule(usernum, itemnum, feat_statistics, feat_types, args, interaction_vocabs)
    embedding_state_dict = checkpoint['embedding_state_dict']
    # Remove keys that don't belong to SharedEmbeddingModule
    filtered_state_dict = {k: v for k, v in embedding_state_dict.items() if k in shared_embeddings.state_dict()}
    shared_embeddings.load_state_dict(filtered_state_dict)
    print("✅ Shared Embeddings loaded.")

    retriever = Hydra(usernum, itemnum, feat_statistics, feat_types, args, shared_embeddings).to(args.device)
    retriever.load_state_dict(checkpoint['retriever_model_state_dict'])
    retriever.eval()
    print("✅ Retriever model loaded.")

    all_embs = []
    user_list = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):

            seq, token_type, seq_feat, user_id, timestamp, action_type, dwell_bins = batch
            seq = seq.to(args.device)
            token_type = token_type.to(args.device)
            timestamp = timestamp.to(args.device)
            action_type = action_type.to(args.device)
            dwell_bins = dwell_bins.to(args.device)
            logits = retriever.predict(seq, seq_feat, token_type, timestamp, action_type, dwell_bins)
            for i in range(logits.shape[0]):
                emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
                all_embs.append(emb)
            user_list += user_id
    # 生成候选库的embedding 以及 id文件

    retrieve_id2creative_id, candidate_features_dict = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        retriever,
        rqvae_model,  # Pass RQ-VAE model
        args,
    )
    all_embs = np.concatenate(all_embs, axis=0)
    # 保存query文件
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))
    # ANN 检索
    RERANK_CANDIDATE_COUNT = 100
    ann_cmd = (
            str(Path("/workspace", "faiss-based-ann", "faiss_demo"))
            + " --dataset_vector_file_path="
            + str(Path(os.environ.get("EVAL_RESULT_PATH"), "embedding.fbin"))
            + " --dataset_id_file_path="
            + str(Path(os.environ.get("EVAL_RESULT_PATH"), "id.u64bin"))
            + " --query_vector_file_path="
            + str(Path(os.environ.get("EVAL_RESULT_PATH"), "query.fbin"))
            + " --result_id_file_path="
            + str(Path(os.environ.get("EVAL_RESULT_PATH"), f"id{RERANK_CANDIDATE_COUNT}.u64bin"))
            + f" --query_ann_top_k=10 --faiss_M=64 --faiss_ef_construction=1280 --query_ef_search=640 --faiss_metric_type=0"
    )
    os.system(ann_cmd)

    # 取出top-k
    top10s_retrieved = read_result_ids(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))
    top10s_untrimmed = []
    for top10 in tqdm(top10s_retrieved):
        for item in top10:
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))

    top10s = [top10s_untrimmed[i: i + 10] for i in range(0, len(top10s_untrimmed), 10)]
    print(top10s)
    return top10s, user_list
import argparse
import json
import math
import os
import time
from pathlib import Path
from multiprocessing import Manager

import faiss
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from embedding import SharedEmbeddingConfig, SharedEmbeddingModule, make_embd
from hstu import HSTUModel
from hydra3 import Hydra
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import ParameterGrid



def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with linear warmup and cosine annealing.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--maxlen', default=50, type=int)

    # Baseline Model construction
    parser.add_argument('--hydra_hidden_units', default=60, type=int)
    parser.add_argument('--hydra_num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--hydra_num_heads', default=2, type=int)
    parser.add_argument('--hydra_dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=1e-5, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--loss_type', default=None, type=str)
    parser.add_argument('--num_actions', default=5, type=int, help='Number of distinct action types.')

    parser.add_argument('--num_local_blocks', default=2, type=int)
    parser.add_argument('--num_global_blocks', default=2, type=int)
    parser.add_argument('--time_span', default=512, type=int)  # For time embeddings
    parser.add_argument('--hstu_num_heads', default=2, type=int)
    parser.add_argument('--hstu_hidden_units', default=60, type=int)
    parser.add_argument('--hstu_dropout_rate', default=0.2, type=float)
    parser.add_argument('--num_candidates', default=100, type=int)
    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81', '82'], type=str, choices=[str(s) for s in range(81, 87)])

    # NEW: Add in-batch negative sampling flag
    parser.add_argument('--use_inbatch_negatives', default= True, help='Use in-batch negative sampling')
    parser.add_argument('--temperature', default=0.05, type=float, help='Temperature for InfoNCE loss')

    args = parser.parse_args()

    return args


class SparseRegularization(torch.nn.Module):
    """
    Custom regularization module for handling sparse data scenarios
    """

    def __init__(self, l2_weight=1e-5, l1_weight=1e-6, entropy_weight=1e-4):
        super().__init__()
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight
        self.entropy_weight = entropy_weight

    def compute_l2_reg(self, embeddings):
        """L2 regularization on embeddings"""
        return self.l2_weight * torch.sum(embeddings.pow(2))

    def compute_l1_reg(self, embeddings):
        """L1 regularization for sparsity"""
        return self.l1_weight * torch.sum(torch.abs(embeddings))

    def compute_entropy_reg(self, embeddings):
        """Entropy regularization to prevent collapse"""
        # Normalize embeddings
        normalized = F.normalize(embeddings, p=2, dim=-1)
        # Compute similarity matrix
        similarity = torch.mm(normalized, normalized.t())
        # Convert to probability distribution
        prob = F.softmax(similarity, dim=-1)
        # Compute entropy
        entropy = -torch.sum(prob * torch.log(prob + 1e-8))
        return self.entropy_weight * entropy

    def forward(self, embeddings, embedding_type='dense'):
        """
        Apply appropriate regularization based on embedding type
        """
        total_reg = 0.0

        if embedding_type == 'sparse':
            # For sparse features, use stronger L2 and entropy regularization
            total_reg += self.compute_l2_reg(embeddings) * 2.0
        elif embedding_type == 'dense':
            # For dense features, use standard L2
            total_reg += self.compute_l2_reg(embeddings)
        elif embedding_type == 'oov':
            # For OOV embeddings, use L1 to encourage sparsity
            total_reg += self.compute_l1_reg(embeddings)

        return total_reg


class FrequencyAwareRegularization(torch.nn.Module):
    """
    Apply different regularization strengths based on feature frequency
    """

    def __init__(self, frequency_bins=[10, 100, 1000], reg_weights=[1e-3, 1e-4, 1e-5, 1e-6]):
        super().__init__()
        self.frequency_bins = frequency_bins
        self.reg_weights = reg_weights

    def forward(self, embeddings, frequencies):
        """
        Args:
            embeddings: [num_items, embedding_dim]
            frequencies: [num_items] - frequency count for each item
        """
        total_reg = 0.0

        for i, threshold in enumerate(self.frequency_bins + [float('inf')]):
            if i == 0:
                mask = frequencies < threshold
            elif i == len(self.frequency_bins):
                mask = frequencies >= self.frequency_bins[-1]
            else:
                mask = (frequencies >= self.frequency_bins[i - 1]) & (frequencies < threshold)

            if mask.any():
                # Apply different regularization weight based on frequency
                reg_weight = self.reg_weights[i]
                masked_embeddings = embeddings[mask]
                total_reg += reg_weight * torch.sum(masked_embeddings.pow(2))

        return total_reg

def evaluate_model(model, data_loader, device, num_neg_samples=99, seed=42):
    """
    Fast evaluation using vectorized operations and minimal loops.
    """
    model.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)

    all_ranks = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating", leave=False)):
            (seq, targets, pos, neg, token_type, next_token_type,
             next_action_type, seq_feat, pos_feat, neg_feat, ts, action_type, dwell_bins) = batch

            seq = seq.to(device)
            pos = pos.to(device)
            token_type = token_type.to(device)
            ts = ts.to(device)
            action_type = action_type.to(device)
            dwell_bins = dwell_bins.to(device)

            # Get user embeddings (only for last position - most common evaluation)
            log_feats = model.log2feats(seq, seq_feat, token_type, ts, action_type, dwell_bins)
            user_embs = log_feats[:, -1, :]  # [B, D] - last position only

            # Filter valid samples
            valid_mask = (pos[:, -1] != 0)  # Check last position
            if not valid_mask.any():
                continue

            user_embs = user_embs[valid_mask]
            pos_items = pos[:, -1][valid_mask]  # [valid_batch]
            batch_size = user_embs.shape[0]

            # Vectorized negative sampling
            total_items = model.shared_embeddings.item_emb.weight.shape[0] - 1
            torch.manual_seed(seed + batch_idx)  # Deterministic sampling

            # Sample negatives for entire batch at once
            neg_items = torch.randint(1, total_items + 1,
                                      (batch_size, num_neg_samples), device=device)

            # Get all embeddings in one go
            pos_embs = model.shared_embeddings.item_emb(pos_items)
            pos_embs = model.embedding_projection(pos_embs)  # [valid_batch, D]

            neg_embs = model.shared_embeddings.item_emb(neg_items.view(-1))
            neg_embs = model.embedding_projection(neg_embs)
            neg_embs = neg_embs.view(batch_size, num_neg_samples, -1)  # [valid_batch, num_neg, D]

            # Vectorized scoring
            pos_scores = torch.sum(user_embs * pos_embs, dim=1)  # [valid_batch]
            neg_scores = torch.bmm(user_embs.unsqueeze(1),
                                   neg_embs.transpose(1, 2)).squeeze(1)  # [valid_batch, num_neg]

            # Calculate ranks efficiently
            ranks = torch.sum(neg_scores >= pos_scores.unsqueeze(1), dim=1)
            all_ranks.extend(ranks.cpu().tolist())

    # Calculate metrics
    all_ranks = np.array(all_ranks)
    hr_10 = np.mean(all_ranks < 10)
    ndcg_10 = np.mean([1.0 / np.log2(rank + 2.0) if rank < 10 else 0.0 for rank in all_ranks])

    return {
        'hr@10': hr_10,
        'ndcg@10': ndcg_10,
        'num_samples': len(all_ranks)
    }

class HyperparameterTuner:
    def __init__(self, base_args, train_loader, valid_loader, model_class, device):
        self.base_args = base_args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model_class = model_class
        self.device = device

    def grid_search_enhanced_loss(self):
        """
        Grid search for EnhancedTrainingLoss parameters
        """
        # Define parameter grid for Enhanced Loss components
        param_grid = {
            'ranking_loss_weight': [0.0, 0.1, 0.2, 0.5],
            'contrastive_loss_weight': [0.8, 1.0, 1.2],
            'diversity_loss_weight': [0.0, 0.01, 0.05, 0.1],
            'temperature': [0.05],
            'hard_ratio': [0.3],
            'T': [512],  # number of negatives
        }

        best_score = 0.0
        best_params = None
        results = []

        # Create parameter combinations
        param_combinations = list(ParameterGrid(param_grid))

        print(f"Testing {len(param_combinations)} parameter combinations...")

        for i, params in enumerate(param_combinations):
            print(f"\nTesting combination {i + 1}/{len(param_combinations)}: {params}")

            # Update args with current parameters
            current_args = self.update_args_with_params(self.base_args, params)

            # Train model with current parameters
            score = self.train_and_evaluate(current_args, max_epochs=4)  # Quick training for grid search

            results.append({
                'params': params,
                'score': score,
                'rank': i + 1
            })

            if score > best_score:
                best_score = score
                best_params = params
                print(f"New best score: {best_score:.4f}")

        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)

        # Save results
        self.save_grid_search_results(results, best_params, best_score)

        return best_params, best_score, results

    def optuna_optimization(self, n_trials=50):
        """
        Use Optuna for more sophisticated hyperparameter optimization
        """

        def objective(trial):
            # Sample parameters
            params = {
                'ranking_loss_weight': trial.suggest_float('ranking_loss_weight', 0.0, 1.0),
                'contrastive_loss_weight': trial.suggest_float('contrastive_loss_weight', 0.5, 2.0),
                'diversity_loss_weight': trial.suggest_float('diversity_loss_weight', 0.0, 0.2),
                'temperature': trial.suggest_float('temperature', 0.01, 0.5, log=True),
                'hard_ratio': trial.suggest_float('hard_ratio', 0.1, 0.8),
                'T': trial.suggest_categorical('T', [256, 512, 1024, 2048]),
                'l2_emb': trial.suggest_float('l2_emb', 1e-6, 1e-3, log=True),
                'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            }

            # Update args
            current_args = self.update_args_with_params(self.base_args, params)

            # Train and evaluate
            score = self.train_and_evaluate(current_args, max_epochs=5)

            return score

        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        print("Best trial:")
        print(f"  Score: {study.best_value:.4f}")
        print(f"  Params: {study.best_params}")

        return study.best_params, study.best_value

    def update_args_with_params(self, base_args, params):
        """Update args object with new parameters"""
        import copy
        args = copy.deepcopy(base_args)

        # Update loss component weights
        args.ranking_loss_weight = params.get('ranking_loss_weight', 0.0)
        args.contrastive_loss_weight = params.get('contrastive_loss_weight', 1.0)
        args.diversity_loss_weight = params.get('diversity_loss_weight', 0.0)

        # Update contrastive loss parameters
        args.temperature = params.get('temperature', 0.05)
        args.hard_ratio = params.get('hard_ratio', 0.3)
        args.T = params.get('T', 512)

        # Update other parameters
        if 'l2_emb' in params:
            args.l2_emb = params['l2_emb']
        if 'lr' in params:
            args.lr = params['lr']

        return args

    def train_and_evaluate(self, args, max_epochs=5):
        """
        Train model with given parameters and return validation score
        """
        # Initialize model

        usernum, itemnum = dataset.usernum, dataset.itemnum
        feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

        SharedEmbeddingConfig(args)
        shared_embeddings = SharedEmbeddingModule(
            usernum, itemnum, feat_statistics,
            feat_types, args, dataset.interaction_vocab_dict
        )

        model = Hydra(
            usernum, itemnum, feat_statistics,
            feat_types, args, shared_embeddings
        ).to(self.device)

        # Initialize model weights
        for name, param in model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass

        # Set padding embeddings to zero
        model.shared_embeddings.pos_emb.weight.data[0, :] = 0
        model.shared_embeddings.item_emb.weight.data[0, :] = 0
        model.shared_embeddings.user_emb.weight.data[0, :] = 0
        model.shared_embeddings.action_emb.weight.data[0, :] = 0

        for k in model.shared_embeddings.sparse_emb:
            model.shared_embeddings.sparse_emb[k].weight.data[0, :] = 0

        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.98),
            weight_decay=0.1
        )

        total_steps = len(self.train_loader) * max_epochs
        warmup_steps = int(0.05 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        scaler = torch.amp.GradScaler('cuda')

        best_score = 0.0

        # Training loop
        for epoch in range(max_epochs):
            model.train()
            epoch_loss = 0.0

            for step, batch in enumerate(self.train_loader):
                (seq, targets, pos, neg, token_type, next_token_type,
                 next_action_type, seq_feat, pos_feat, neg_feat, ts, action_type, dwell_bins) = batch

                # Move to device
                seq = seq.to(self.device)
                targets = targets.to(self.device)
                token_type = token_type.to(self.device)
                ts = ts.to(self.device)
                action_type = action_type.to(self.device)
                dwell_bins = dwell_bins.to(self.device)
                pos = pos.to(self.device)
                neg = neg.to(self.device)

                optimizer.zero_grad()

                with torch.amp.autocast("cuda"):
                    loss = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type,
                        seq_feat, pos_feat, neg_feat, ts, action_type, dwell_bins
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                scaler.step(optimizer)
                scaler.update()
                if optimizer.state:
                    scheduler.step()
                del seq, targets, pos, neg, token_type, next_token_type, next_action_type
                del ts, action_type, dwell_bins, loss

                epoch_loss += loss.item()

            # Validation
            model.eval()
            val_metrics = evaluate_model(model, self.valid_loader, self.device)
            current_score = val_metrics['ndcg@10']

            if current_score > best_score:
                best_score = current_score

            print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {epoch_loss / len(self.train_loader):.4f}, "
                  f"NDCG@10: {current_score:.4f}")

        return best_score

    def save_grid_search_results(self, results, best_params, best_score):
        """Save grid search results to file"""
        output = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }

        with open('grid_search_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nGrid search complete!")
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score:.4f}")
        print(f"Results saved to grid_search_results.json")


class AdaptiveLossScheduler:
    """
    Dynamically adjust loss component weights during training
    """

    def __init__(self, initial_weights, schedule_type='cosine'):
        self.initial_weights = initial_weights
        self.schedule_type = schedule_type
        self.step = 0

    def get_current_weights(self, total_steps):
        """Get current loss weights based on training progress"""
        progress = min(self.step / total_steps, 1.0)

        if self.schedule_type == 'cosine':
            # Cosine annealing for diversity loss (start high, end low)
            diversity_weight = self.initial_weights['diversity'] * 0.5 * (1 + np.cos(np.pi * progress))

            # Increase contrastive weight over time
            contrastive_weight = self.initial_weights['contrastive'] * (0.5 + 0.5 * progress)

            # Keep ranking weight constant
            ranking_weight = self.initial_weights['ranking']

        elif self.schedule_type == 'linear':
            # Linear scheduling
            diversity_weight = self.initial_weights['diversity'] * (1 - progress)
            contrastive_weight = self.initial_weights['contrastive'] * (1 + progress)
            ranking_weight = self.initial_weights['ranking']

        return {
            'diversity': diversity_weight,
            'contrastive': contrastive_weight,
            'ranking': ranking_weight
        }

    def step_scheduler(self):
        self.step += 1


def run_comprehensive_tuning(base_args, train_loader, valid_loader, model_class, device):
    """
    Run comprehensive hyperparameter tuning
    """
    tuner = HyperparameterTuner(base_args, train_loader, valid_loader, model_class, device)

    print("=" * 50)
    print("Starting Hyperparameter Tuning")
    print("=" * 50)

    # Phase 1: Coarse grid search
    print("\nPhase 1: Coarse Grid Search")
    coarse_params, coarse_score, _ = tuner.grid_search_enhanced_loss()

    # Phase 2: Fine-tuned Optuna optimization around best grid search results
    print("\nPhase 2: Fine-tuned Optuna Optimization")

    # Update base args with coarse search results
    tuned_args = tuner.update_args_with_params(base_args, coarse_params)

    # Create new tuner with refined search space around best parameters
    refined_tuner = HyperparameterTuner(tuned_args, train_loader, valid_loader, model_class, device)
    final_params, final_score = refined_tuner.optuna_optimization(n_trials=100)

    print("=" * 50)
    print("Hyperparameter Tuning Complete")
    print("=" * 50)
    print(f"Coarse Grid Search Best Score: {coarse_score:.4f}")
    print(f"Final Optuna Best Score: {final_score:.4f}")
    print(f"Final Best Parameters: {final_params}")

    return final_params, final_score

if __name__ == '__main__':
    load_dotenv("my.env")
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')
    interaction_dir = os.environ.get('TRAIN_CKPT_PATH')
    os.environ['ITEM_FEAT_FILE'] = str(Path(data_path) / 'item_feat_dict')

    args = get_args()

    # MODIFIED: Pass use_inbatch_negatives to dataset
    dataset = MyDataset(data_path, interaction_dir, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    # MODIFIED: Use the appropriate collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size= args.batch_size,
        shuffle = True,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    # model = HSTU(
    #     user_num = usernum,
    #     item_num = itemnum,
    #     feat_statistics = feat_statistics,
    #     feat_types = feat_types,
    #     args = args,
    #     attention_dim = 50,
    #     linear_hidden_dim = 50,
    # ).to(args.device)
    SharedEmbeddingConfig(args)  # Set up shared_hidden_units
    shared_embeddings = SharedEmbeddingModule(usernum, itemnum, feat_statistics, feat_types, args, dataset.interaction_vocab_dict)

    model = Hydra(usernum, itemnum, feat_statistics, feat_types, args, shared_embeddings).to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    # best_params, best_score = run_comprehensive_tuning(
    #     base_args=args,
    #     train_loader=train_loader,
    #     valid_loader=valid_loader,
    #     model_class=Hydra,
    #     device=args.device
    # )
    # print(best_params)
    # print(best_score)

    model.shared_embeddings.pos_emb.weight.data[0, :] = 0
    model.shared_embeddings.item_emb.weight.data[0, :] = 0
    model.shared_embeddings.user_emb.weight.data[0, :] = 0
    model.shared_embeddings.action_emb.weight.data[0, :] = 0

    for k in model.shared_embeddings.sparse_emb:
        model.shared_embeddings.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.1)
    item_frequencies = torch.zeros(model.item_num + 1).to(args.device)
    sparse_reg = SparseRegularization(
        l2_weight=args.l2_emb,
        l1_weight=getattr(args, 'l1_emb', 1e-6),
        entropy_weight=getattr(args, 'entropy_weight', 1e-4)
    )


    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    best_valid_loss_sum = float("inf")
    patience_counter = 0
    scaler = torch.amp.GradScaler('cuda')
    print("Start training")
    print(f"Using in-batch negatives: {args.use_inbatch_negatives}")

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # MODIFIED: Handle different batch formats for in-batch vs random negatives
            (seq, targets, pos, neg, token_type, next_token_type,
             next_action_type, seq_feat, pos_feat, neg_feat, ts, action_type, dwell_bins) = batch
            seq = seq.to(args.device)
            targets = targets.to(args.device)
            token_type = token_type.to(args.device)
            ts = ts.to(args.device)
            action_type = action_type.to(args.device)
            dwell_bins = dwell_bins.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                # Get user and item embeddings for in-batch negative sampling
                loss = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat,
                    ts, action_type, dwell_bins
                 )

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            print('Loss/train', loss.item(), global_step)

            global_step += 1

            # if args.l2_emb > 0:
            #     for param in model.shared_embeddings.item_emb.parameters():
            #         loss += args.l2_emb * torch.norm(param)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            scaler.step(optimizer)
            scaler.update()
            if optimizer.state:
                scheduler.step()

            del seq, targets, pos, neg, token_type, next_token_type, next_action_type
            del ts, action_type, dwell_bins, loss
            
        model.eval()
        valid_loss_sum = 0

        # MODIFIED: Handle validation loop with different batch formats
        with torch.no_grad():  # Add no_grad for validation
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                (seq, targets, pos, neg, token_type, next_token_type,
                 next_action_type, seq_feat, pos_feat, neg_feat, ts, action_type, dwell_bins) = batch
                seq = seq.to(args.device)
                targets = targets.to(args.device)
                token_type = token_type.to(args.device)
                ts = ts.to(args.device)
                action_type = action_type.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                dwell_bins = dwell_bins.to(args.device)

                # Get user and item embeddings for in-batch negative sampling
                loss = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat,
                    ts, action_type, dwell_bins
                )

                valid_loss_sum += loss.item()

        valid_loss_sum /= len(valid_loader)
        print('Loss/valid', valid_loss_sum, global_step)

        val_metrics = evaluate_model(model, valid_loader, args.device)

        writer.add_scalar('Val_Metrics/HR@10', val_metrics['hr@10'], global_step)
        writer.add_scalar('Val_Metrics/NDCG@10', val_metrics['ndcg@10'], global_step)

        print(f"\n📊 Epoch {epoch} Validation Results:")
        print(f"   Val HR@10:  {val_metrics['hr@10']:.4f}")
        print(f"   Val NDCG@10:{val_metrics['ndcg@10']:.4f}")

        if valid_loss_sum < best_valid_loss_sum:
            best_valid_loss_sum = valid_loss_sum
            patience_counter = 0
            # Save best model
            best_model_path = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step retriever_best_model")
            best_model_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_model_path / "retriever_best_model.pt")
            embedding_path = Path(os.environ.get('TRAIN_CKPT_PATH'), "global_step embedding")
            embedding_path.mkdir(parents=True, exist_ok=True)
            torch.save(shared_embeddings.state_dict(), embedding_path / "embedding.pt")
            print(f" Embeddings saved at {embedding_path}")
            print(f"   New best model saved with valid_loss={valid_loss_sum:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(
                    f"\n🛑 Early stopping triggered after {epoch} epochs"
                )
                break

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "hydra_info.pt")



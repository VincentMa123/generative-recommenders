import argparse
import json
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_rqvae import MmEmbDataset, RQVAE
import torch

def get_args():
    parser = argparse.ArgumentParser()
    # Train params
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=60, type=int)
    parser.add_argument('--num_blocks', default=3, type=int)
    parser.add_argument('--num_epochs', default=2, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--latent_dim', default=64, type=int)
    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


args = get_args()
load_dotenv("my.env")
data_path = os.environ.get('TRAIN_DATA_PATH')


dataset = MmEmbDataset(data_path,  args.mm_emb_id)
print(dataset.emb_list[0].shape[-1])
print(len(dataset))
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
valid_loader = DataLoader(
    valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
device = args.device

model = RQVAE(
    input_dim = dataset.emb_list[0].shape[-1],
    hidden_channels= [16, 16],
    latent_dim=16,
    num_codebooks =  3,
    codebook_size = [16, 16, 16],
    shared_codebook =  False,
    kmeans_method = "kmeans",
    kmeans_iters = 20,
    distances_method  = "cosine",
    loss_beta = 0.25,
    device = device,
    ).to(device)
print(dataset.emb_list[0].shape[-1])
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
best_valid_loss = float('inf')
print("start train")
scaler = GradScaler()
for epoch in range(args.num_epochs):
    model.train()
    for tid_batch, emb_batch in tqdm((train_loader), total=len(train_loader)):
        emb_batch = emb_batch.to(device)
        optimizer.zero_grad()
        with autocast("cuda"):
            x_hat, sid, recon_loss, rqvae_loss, total_loss = model(emb_batch)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(f"Epoch {epoch}  recon={recon_loss.item():.4f}  rq={rqvae_loss.item():.4f}")
    del x_hat, recon_loss, rqvae_loss, total_loss
    torch.cuda.empty_cache()

    model.eval()
    total_valid_loss = 0
    with torch.no_grad():
        for tid_batch, emb_batch in tqdm(valid_loader, total=len(valid_loader), desc=f"Epoch {epoch} Validation"):
            emb_batch = emb_batch.to(device)
            with autocast("cuda"):
                x_hat, sid, recon_loss, rqvae_loss, total_loss = model(emb_batch)
            total_valid_loss += total_loss.item()

    avg_valid_loss = total_valid_loss / len(valid_loader)
    print(f"Epoch {epoch} Avg Valid Loss: {avg_valid_loss:.4f}")

    # Save the model only if validation loss improves
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        best_model_path = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step rqvae_best_model")
        best_model_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), best_model_path / "rqvae_best_model.pt")


print("done train")
save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'))

# ===================================================================
# OPTIMIZATION FOR ISSUE #1: Only load embeddings that are in the training set
# ===================================================================
data_path = Path(os.environ['TRAIN_DATA_PATH'])
with open(data_path / 'indexer.pkl', 'rb') as ff:
    indexer = pickle.load(ff)

# Get the set of creative_ids that are actually in our training data
training_creative_ids = set(indexer['i'].keys())

all_tids, all_embs = [], []
# Iterate through the original full dataset to find the ones we need
items_in_training_set = 0
for tid, emb in dataset:
    # The 'tid' from MmEmbDataset is the creative_id
    if int(tid) in training_creative_ids:
        all_tids.append(int(tid))
        all_embs.append(emb)
        items_in_training_set += 1

print(f"Filtered embeddings. Kept {items_in_training_set} items that are present in the training set indexer.")

all_embs = torch.stack(all_embs, dim=0).to(device)

# ===================================================================
# Generate semantic IDs only for the relevant items
# ===================================================================
total_batch = 16384
sem_ids = []
with torch.no_grad():
    for i in range(0, len(all_embs), total_batch):
        chunk = all_embs[i : i+total_batch].to(device)
        ids  = model._get_codebook(chunk)
        sem_ids.append(ids.cpu())
sem_ids = torch.cat(sem_ids, dim=0).numpy().astype(int)

# ===================================================================
# FIX FOR ISSUE #2: Read original, write to a NEW file in a writable directory
# ===================================================================
# 1. Read the original feature file from the read-only input directory
feat_file = data_path / 'item_feat_dict.json'
item_feats = json.load(open(feat_file, 'r'))

# 2. Inject the new features
for creative_id, sid_vec in zip(all_tids, sem_ids):
    # We already filtered, so every creative_id here is guaranteed to be in the indexer
    reid = indexer['i'][creative_id]

    if str(reid) in item_feats:
        item_feats[str(reid)]["300"] = int(sid_vec[0])
        item_feats[str(reid)]["301"] = int(sid_vec[1])
        item_feats[str(reid)]["302"] = int(sid_vec[2])

# 3. Define the output path in your writable checkpoint directory
output_feat_path = save_dir / 'item_feat_dict.json'

# 4. Write the new, enriched file to the writable directory
with open(output_feat_path, 'w') as f:
    json.dump(item_feats, f)

print(f"\nSuccessfully wrote enriched feature file to: {output_feat_path}")
print("You should now update your main training script to use this new feature file.")

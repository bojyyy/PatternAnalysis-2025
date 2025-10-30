# ***Generative time-series model for Limit Order Book (LOB) events with TimeGAN***

Author: Miles Bojorge, 47438514

## Task
In this task, I will be generating synthetic sequences of Limit Order Book events using the LOBSTER dataset, more specifically AMZN level 10 data. TimeGAN was the selected model architecture for this project, since it bridges GANs to the time-series domain where preserving temporal dynamics is paramount.

# TimeGAN Architecture
TimeGAN is a framework for generating realistic time-series data across various domains. The model consists of four network components: an embedding function, recovery function, generator and discriminator. There is an additional network known as the supervisor that enforces step-wise latent transition dynamics. The autoencoding compomenents are trained jointly with the adversarial components. The result is that TimeGAN is capable of simultaneously encoding features, generating representations and iterating over time. In this project's implementation, we have added an additional network known as a patch discriminator, with the idea that this should help pressure textures on a depth heatmap to more closely resemble real LOB depth level heatmaps.

As for the motivation behind TimeGAN, vanilla GANs learn independent and identically distributed snapshots, but they struggle with dependencies over time. This is exactly what makes TimeGAN relevant to the specific domain of this project. TimeGAN blends sequence modelling via RNNs with adversarial learning to preserve autocorrelations, periodicity and regime shifts. 

## Architectural components
* Embedder: maps obseverved sequences to a latent space. Essentially, the embedder is a representation learner that compresses high-dim data while keeping temporal structure.
* Recovery: decodes from latent space back to the original feature space, which forces the latent representation to be information preserving, and not just arbitrary.
* Generator: maps noise to synthetic latent paths. This is what actually generates the sequences, but it lives in the latent space learned by the embedder.
* Supervisor: predicts the next latent step. This is what injects an explicit temporal continuity prior, so generated sequences are not jittery or just static.
* Discriminator: distinguishes real latent trajectories from generated ones.
* Specific to our LOB setup *
* Patch discriminator: enforces local "texture" in the feature space, becomes relevant 


## Parameter count
```json
{
  "config": {
    "x_dim": 31,
    "z_dim": 31,
    "h_dim": 24,
    "rnn_layers": 3,
    "dropout": 0.0,
    "lr": 0.001,
    "batch_size": 128,
    "seq_len": 24,
    "step": 150,
    "depth": 10,
    "depth_levels": 10
  },
  "environment": {
    "torch_version": "2.7.1+cu118",
    "python": "3.13.7",
    "platform": "Windows-10-10.0.19045-SP0",
    "cuda_available": true,
    "cudnn_enabled": true,
    "device_type": "cuda",
    "gpu_index": 0,
    "gpu_name": "NVIDIA GeForce RTX 4070 SUPER",
    "cuda_runtime": "11.8",
    "capability": "8.9",
    "peak_allocated_bytes": 119172608,
    "peak_reserved_bytes": 163577856
  },
  "params": {
    "embedder": {
      "trainable": 15072,
      "total": 15072
    },
    "recovery": {
      "trainable": 15175,
      "total": 15175
    },
    "generator": {
      "trainable": 15072,
      "total": 15072
    },
    "supervisor": {
      "trainable": 14400,
      "total": 14400
    },
    "discriminator": {
      "trainable": 4825,
      "total": 4825
    },
    "patch_discriminator": {
      "trainable": 22401,
      "total": 22401
    },
    "total_trainable": 86945,
    "total": 86945
  },
  "strategy": {
    "pretrain_embed_iters": 1000,
    "pretrain_supervised_iters": 1000,
    "joint_iters": 2000,
    "loss_weights": {
      "gamma": 1.0,
      "sup_w": 25.0,
      "mom_w": 30.0
    },
    "variant_notes": "Full TimeGAN: reconstruction + supervised + adversarial (with moment matching)."
  },
  "phase_times_sec": {
    "pretrain_embed": 33.77164030075073,
    "pretrain_supervised": 23.719775915145874,
    "joint": 688.9165897369385
  },
  "total_time_sec": 746.4080431461334,
  "num_batches": {
    "pretrain_embed": 9000,
    "pretrain_supervised": 9000,
    "joint_GS": 36000,
    "joint_D": 18000
  },
  "batches_per_iter": 9
}
```
# Training
## Data


## Preprocessing

# Inputs
We used paired LOBSTER csvs for a single day of trading of the AMZN stock, with level 10 data.The two csvs were:
* Messages: Time, Type, Order ID, Size, Price, Direction
* Orderbook: AskPrice1, Asksize1, BidPrice1, Bidsize1, etc. up to depth D

We load both files, verify row alignment (as specified in the data's README), and construct fixed-width windows for TimeGAN. These files were combined since orderbook data alone doesn't allow TimeGAN to learn what orders are actually driving the orderbook.

# Engineered features
* Event type one-hot over classes {1,2,3,4,5,6,7}, 7 represents a trading halt but this was kept for completeness.
* Side (0,1) was derived from direction (-1, +1)
* Midprice delta (ticks): first difference of mid (ask1 + bid1)/2 divided by the tick size, which is clipped to a default band of +- 50 ticks.
* Spread (ticks): ask1 - bid1 divided by the tick size
* Inter-arrival time: derived from the message timestamps
* Depth sizes (for heatmaps and SSIM calculation), with log1p applied.
Tick size is configured to 100 LOBSTER price units. 

# Scaling
A min-max scaler is applied to continuous features only. This includes `mid_delta_ticks`, `log_size`, `log_dt` and depth sizes. The rest of the features are not scaled, including `spread_ticks` which was intentionally kept integer-like. The scaler is fit only on train rows and then used to transform val/test.

# Windowing
From the full time series T rows, sliding windows of length `seq_len` with stride `step` were created, yielding tensors of shape `[seq_len, num_features]`. Also worth noting:
* Train/val/test row splits were done as a typical 70/15/15 split, done by row index
* Windows are produced without materialising all windows in RAM. The dataset computes the start and end slices on the fly.
* DataLoaders are constructed for each split. The scaler is fit on train only.

# Depth heatmap helper
For visualisation and SSIM, `extract_depth_matrix(window, feat_idx, K)` was exposed to build a a matrix using `[BidSize1..K, AskSize1..K] from any window. This ensures real vs synthetic heatmaps are constructed with the same column ordering.

## Configuration


## Training strategy


# Results

## Outputs


## Losses


## Error analysis



# Usage



## Dependencies


## Hyperparameters


# References


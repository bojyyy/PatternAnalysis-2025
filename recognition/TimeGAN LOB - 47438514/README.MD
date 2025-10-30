# ***Generative time-series model for Limit Order Book (LOB) events with TimeGAN***

Author: Miles Bojorge, 47438514

## Task
In this task, I will be generating synthetic sequences of Limit Order Book events using the LOBSTER dataset, more specifically AMZN level 10 data. TimeGAN was the selected model architecture for this project, since it bridges GANs to the time-series domain where preserving temporal dynamics is paramount.

# TimeGAN Architecture
TimeGAN is a framework for generating realistic time-series data across various domains. The model consists of four network components: an embedding function, recovery function, generator and discriminator. There is an additional network known as the supervisor that enforces step-wise latent transition dynamics. The autoencoding compomenents are trained jointly with the adversarial components. The result is that TimeGAN is capable of simultaneously encoding features, generating representations and iterating over time. In this project's implementation, we have added an additional network known as a patch discriminator, with the idea that this should help pressure textures on a depth heatmap to more closely resemble real LOB depth level heatmaps.

As for the motivation behind TimeGAN, vanilla GANs learn independent and identically distributed snapshots, but they struggle with dependencies over time. This is exactly what makes TimeGAN relevant to the specific domain of this project. TimeGAN blends sequence modelling via RNNs with adversarial learning to preserve autocorrelations, periodicity and regime shifts. 

## Architectural components
* Embedder: maps obseverved sequences to a latent space. Essentially, the embedder is a representation leanre that compresses high-dim data while keeping temporal structure.
* Recovery: 
* Generator
* Supervisor
* Discriminator

## Parameter count

# Training
## Data


## Preprocessing


## Configuration


# Results

## Outputs


## Losses



# Usage



## Dependencies


## Hyperparameters


# References


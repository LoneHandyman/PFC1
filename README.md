# PFC1
## Implementations:
  Look inside 'impl' directory.
  ### Compressive Transformer:
  - Compression function: 1d convolution.
  - AttentionReconstructionLoss code uses MSE to calculate errors and stop gradients from the main transformer model.
  - This implementation uses Pytorch (nn.Module) and LabML (Tranformers XL multihead attention implementation).

import os
import torch
import torch.nn as nn
from src.modeling import ImageEncoder
from src.utils import DotDict

class AttentionOnlyFinetuneEncoder(ImageEncoder):
    """
    A specialized ImageEncoder that fine-tunes only the attention module weights in the ViT.
    Corresponds to the method described in Jin et al. (2025).
    """
    def __init__(self, args, keep_lang=False):
        # 1. Call the parent constructor to build the full model as usual
        super().__init__(args, keep_lang=keep_lang)

        self.args = args

        # 2. Freeze all model parameters
        # print("Freezing all parameters of the model initially...")
        for param in self.model.parameters():
            param.requires_grad = False

        # 3. Unfreeze only the Attention module weights (Wq, Wk, Wv, Wo)
        # print("Unfreezing Attention module weights for fine-tuning...")
        self._unfreeze_attention_weights(self.model.visual)

        # 4. (Optional but recommended) Print trainable parameters for verification
        # self._verify_trainable_params()

    def _unfreeze_attention_weights(self, vit_model):
        """
        Iterate over all Transformer blocks and unfreeze the attention projection weights.
        """
        # Iterate over the model and unfreeze target parameters
        for block in vit_model.transformer.resblocks:
            # Unfreeze the combined input projection weight for Q, K, V
            block.attn.in_proj_weight.requires_grad = True

            # Unfreeze the output projection weight
            block.attn.out_proj.weight.requires_grad = True

            # Per the paper's ablation study, not fine-tuning biases yields better results; keep them frozen
            # block.attn.in_proj_bias.requires_grad = True
            # block.attn.out_proj.bias.requires_grad = True

    def _verify_trainable_params(self):
        """Print all trainable parameters for debugging and verification."""
        print("="*80)
        print("Trainable parameters in AttentionOnlyFinetuneEncoder:")
        trainable_params_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"  - {name}")
                trainable_params_count += param.numel()
        print(f"Total trainable parameters: {trainable_params_count / 1e6:.2f}M")
        print("="*80)

    def forward(self, images, calculate_ortho_loss=False, pretrained_state_dict=None):
        """
        Extended forward method to optionally compute and return the orthogonal loss.
        Consistent with the logic implemented for standard_ortho.
        """
        # Original forward pass
        features = self.model.encode_image(images)

        # Return features directly if orthogonal loss is not needed
        if not calculate_ortho_loss:
            return features

        # --- Compute orthogonal loss if requested ---
        if pretrained_state_dict is None:
            raise ValueError("pretrained_state_dict must be provided when calculate_ortho_loss is True")

        ortho_loss = 0.0
        # self.model is the open_clip model (e.g. ViT); iterate over its parameters
        for name, p_finetuned in self.model.named_parameters():
            # Only compute loss for trainable parameters with gradients
            if p_finetuned.requires_grad and p_finetuned.dim() == 2:
                if name in pretrained_state_dict:
                    p_pretrained = pretrained_state_dict[name].to(p_finetuned.device)

                    delta_W = p_finetuned - p_pretrained

                    # Compute orthogonal loss (W^T * W - I)
                    rows, cols = delta_W.shape
                    if rows < cols:
                        mat = delta_W @ delta_W.T
                        identity = torch.eye(rows, device=delta_W.device)
                    else:
                        mat = delta_W.T @ delta_W
                        identity = torch.eye(cols, device=delta_W.device)
                    
                    ortho_loss += torch.norm(mat - identity, p='fro')
        
        return features, ortho_loss

    def __call__(self, inputs, calculate_ortho_loss=False, pretrained_state_dict=None):
        # Ensure __call__ forwards all arguments
        return self.forward(inputs, calculate_ortho_loss, pretrained_state_dict)

    def save(self, filename):
        """Save model weights."""
        # print(f"Saving AttentionOnlyFinetuneEncoder state_dict to {filename}")
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Save only the state_dict; reconstruct the model on load
        torch.save(self.state_dict(), filename)

    @classmethod
    def load(cls, filename, args):
        """Load model from a state_dict."""
        # print(f"Loading AttentionOnlyFinetuneEncoder from {filename}")
        encoder = cls(args)  # Create a new instance
        state_dict = torch.load(filename, map_location='cpu')
        encoder.load_state_dict(state_dict)  # Load weights
        return encoder
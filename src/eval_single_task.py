import json
import os

from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.linearize import LinearizedImageEncoder
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.attention_only_finetune import AttentionOnlyFinetuneEncoder

args = parse_arguments()
if 'ortho' in args.finetuning_mode:
    args.save = f"checkpoints_{args.seed}/{args.finetuning_mode}_{args.lr}_lambda{args.ortho_lambda}_{args.model}"
else:
    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.finetuning_mode}_{args.lr}_{args.model}"
    else:
        args.save = f"checkpoints/{args.finetuning_mode}_{args.lr}_{args.model}"

accuracies = {}

print("*" * 100)
mode_labels = {
    "standard": "Evaluating non-linear FT models.",
    "standard_ortho": "Evaluating standard FT models with orthogonality regularization.",
    "linear": "Evaluating linear FT models.",
    "linear_ortho": "Evaluating linear FT models with orthogonality regularization.",
    "linear-2": "Evaluating Attention-Only Finetune models.",
    "linear-2_ortho": "Evaluating Attention-Only Finetune models with orthogonality regularization.",
}
print(mode_labels.get(args.finetuning_mode, f"Evaluating {args.finetuning_mode} models."))

for dataset in [
    "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN",
]:
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    mode = args.finetuning_mode

    if mode == "standard":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        try:
            task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
        except FileNotFoundError:
            print(f"Error: Could not find checkpoints for {dataset}.")
            continue

    elif mode == "standard_ortho":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/standard_ortho_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/standard_ortho_finetuned.pt"
        try:
            task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
        except FileNotFoundError:
            print(f"Error: Could not find checkpoints for {dataset}.")
            continue

    elif mode == "linear":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
        try:
            task_vector = LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
        except FileNotFoundError:
            print(f"Error: Could not find checkpoints for {dataset}.")
            continue

    elif mode == "linear_ortho":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_ortho_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_ortho_finetuned.pt"
        try:
            task_vector = LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
        except FileNotFoundError:
            print(f"Error: Could not find checkpoints for {dataset}.")
            continue

    elif mode in ("linear-2", "linear-2_ortho"):
        prefix = mode + "_"
        pretrained_checkpoint = f"{args.save}/{dataset}Val/{prefix}zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/{prefix}finetuned.pt"
        try:
            task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
        except FileNotFoundError:
            print(f"Error: Could not find checkpoints for {dataset} with mode {mode}.")
            continue

    else:
        print(f"Unknown finetuning mode: {mode}")
        continue

    for split in ["test", "val"]:
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = dataset if split == "test" else f"{dataset}Val"
        accuracies[eval_dataset] = eval_single_dataset(image_encoder, eval_dataset, args)["top1"]

# Save results
save_name_map = {
    "standard": "ft_accuracies.json",
    "standard_ortho": "standard_ortho_ft_accuracies.json",
    "linear": "linear_ft_accuracies.json",
    "linear_ortho": "linear_ortho_ft_accuracies.json",
    "linear-2": "linear-2_ft_accuracies.json",
    "linear-2_ortho": "linear-2_ortho_ft_accuracies.json",
}

save_path = os.path.join(args.save, save_name_map[args.finetuning_mode])
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w") as f:
    json.dump(accuracies, f, indent=4)
print(f"Results saved to {save_path}")

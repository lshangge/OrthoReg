import abc
import torch

from src.linearize import LinearizedImageEncoder
from src.modeling import ImageEncoder
from src.attention_only_finetune import AttentionOnlyFinetuneEncoder


class _TaskVector(abc.ABC):
    def __init__(
        self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None
    ):
        """
        Initializes the task vector from a pretrained and a finetuned checkpoints.
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passing in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert (
                pretrained_checkpoint is not None and finetuned_checkpoint is not None
            )
            with torch.no_grad():
                pretrained_obj = self._load_checkpoint(pretrained_checkpoint)
                finetuned_obj = self._load_checkpoint(finetuned_checkpoint)

                if hasattr(pretrained_obj, 'state_dict'):
                    pretrained_state_dict = pretrained_obj.state_dict()
                else:
                    pretrained_state_dict = pretrained_obj

                if hasattr(finetuned_obj, 'state_dict'):
                    finetuned_state_dict = finetuned_obj.state_dict()
                else:
                    finetuned_state_dict = finetuned_obj

                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                        continue
                    if key in finetuned_state_dict:
                        self.vector[key] = (
                            finetuned_state_dict[key] - pretrained_state_dict[key]
                        )

    @abc.abstractmethod
    def _load_checkpoint(self, checkpoint):
        raise NotImplementedError

    @abc.abstractmethod
    def _cast_to_same_type(self, other):
        raise NotImplementedError

    def __add__(self, other):
        other = self._cast_to_same_type(other)
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return self.__class__(vector=new_vector)

    def __sub__(self, other):
        return self.__add__(-other)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return self.__class__(vector=new_vector)

    def __pow__(self, power):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] ** power
        return self.__class__(vector=new_vector)

    def __mul__(self, other):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = other * self.vector[key]
        return self.__class__(vector=new_vector)

    def dot(self, other):
        other = self._cast_to_same_type(other)
        with torch.no_grad():
            dot_product = 0.0
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                dot_product += torch.sum(self.vector[key] * other.vector[key])
        return dot_product

    def norm(self):
        return torch.sqrt(self.dot(self))

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = self._load_checkpoint(pretrained_checkpoint)

            if hasattr(pretrained_model, 'state_dict'):
                new_state_dict = pretrained_model.state_dict()
            else:
                new_state_dict = pretrained_model.copy()

            pretrained_state_dict = new_state_dict.copy()

            for key in pretrained_state_dict:
                if key in self.vector:
                    new_state_dict[key] = (
                        pretrained_state_dict[key] + scaling_coef * self.vector[key]
                    )

        if hasattr(pretrained_model, 'state_dict'):
            pretrained_model.load_state_dict(new_state_dict)
            return pretrained_model
        else:
            from src.args import parse_arguments
            args = parse_arguments()
            if isinstance(self, NonLinearTaskVector):
                encoder = self._build_model_from_checkpoint(pretrained_checkpoint, args)
                encoder.load_state_dict(new_state_dict)
                return encoder
            else:
                pretrained_model.load_state_dict(new_state_dict)
                return pretrained_model


class NonLinearTaskVector(_TaskVector):
    """A task vector for nonlinear models."""

    def _load_checkpoint(self, checkpoint):
        return torch.load(checkpoint, map_location="cpu")

    def _build_model_from_checkpoint(self, checkpoint_path, args):
        mode = args.finetuning_mode
        if mode in ["linear-2", "linear-2_ortho"]:
            return AttentionOnlyFinetuneEncoder(args)
        return ImageEncoder(args)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        with torch.no_grad():
            from src.args import parse_arguments
            args = parse_arguments()
            pretrained_model = self._build_model_from_checkpoint(pretrained_checkpoint, args)
            pretrained_state_dict = torch.load(pretrained_checkpoint, map_location='cpu')

            if hasattr(pretrained_state_dict, 'state_dict'):
                pretrained_state_dict = pretrained_state_dict.state_dict()

            new_state_dict = pretrained_state_dict.copy()

            for key in pretrained_state_dict:
                if key in self.vector:
                    new_state_dict[key] += scaling_coef * self.vector[key]

            pretrained_model.load_state_dict(new_state_dict)
        return pretrained_model

    def _cast_to_same_type(self, other):
        if isinstance(other, LinearizedTaskVector):
            return linear_to_nonlinear(other, self.vector.keys())
        return other


class LinearizedTaskVector(_TaskVector):
    """A task vector for linearized models."""

    def _load_checkpoint(self, checkpoint):
        return LinearizedImageEncoder.load(checkpoint)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        with torch.no_grad():
            pretrained_model = self._load_checkpoint(pretrained_checkpoint)
            new_state_dict = pretrained_model.state_dict()
            pretrained_state_dict = new_state_dict.copy()

            for key in pretrained_state_dict:
                if key in self.vector:
                    new_state_dict[key] += scaling_coef * self.vector[key]

        pretrained_model.load_state_dict(new_state_dict)
        return pretrained_model

    def get_named_parameters(self, param_names):
        params = {k: v for k, v in self.vector.items() if "model.params0" not in k}
        return {k: v for k, v in zip(param_names, params.values())}

    def _cast_to_same_type(self, other):
        if isinstance(other, NonLinearTaskVector):
            return nonlinear_to_linear(other)
        return other


def nonlinear_to_linear(nonlinear_task_vector):
    if isinstance(nonlinear_task_vector, LinearizedTaskVector):
        return nonlinear_task_vector
    else:
        linear_params = {
            f"model.params.{i}": v
            for i, v in enumerate(nonlinear_task_vector.vector.values())
        }
        linear_params.update({
            f"model.params0.{i}": torch.zeros_like(v)
            for i, v in enumerate(nonlinear_task_vector.vector.values())
        })
        return LinearizedTaskVector(vector=linear_params)


def linear_to_nonlinear(linear_task_vector, param_names):
    if isinstance(linear_task_vector, NonLinearTaskVector):
        return linear_task_vector
    else:
        return NonLinearTaskVector(
            vector=linear_task_vector.get_named_parameters(param_names)
        )

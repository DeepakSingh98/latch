"""Experimental operation-based complete-training compiler (LATCH-OPS)."""
from .capture import (CapturedConditional, CapturedTraining, SGDConfig,
                      capture_conditional_training, capture_training, module_loss)
from .optim import AdamWConfig, initialize_optimizer_state
from .precision import PrecisionPolicy, precision_policy
from .muon import MuonConfig
from .user_optimizer import TensorUpdateRule, tensor_update
from .sampling import capture_sampled_training
from .scan import selective_scan
from .ir import CaptureError, Operation, Ref, TrainingIR, Value
from .api import capture_module, CapturedModule, ModuleUpdate, prepare_module
from .matrix_tuning import MatrixTuning
from .variants import BatchVariants, VariantContractError
from .dynamic_batch import DynamicBatch
from .inference import (CapturedInference, Inference, InferenceIR, LoopInferenceIR, capture_inference,
                        capture_loop_inference, capture_module_inference, prepare_inference)
from .loop import (BoundedLoopIR, CapturedLoop, ConditionalBody, OperationIR,
                   UniformOperationIR, capture_loop_training)
from .nested_loop import NestedLoopBody, NestedOperationIR
from .scaled_add import scaled_add
from .mixed_fma import mixed_fma64

__all__ = ["scaled_add", "mixed_fma64", "MatrixTuning", "CapturedTraining", "SGDConfig", "capture_training", "module_loss",
           "AdamWConfig", "MuonConfig", "TensorUpdateRule", "tensor_update", "PrecisionPolicy", "precision_policy", "initialize_optimizer_state", "capture_sampled_training", "selective_scan",
           "capture_module", "CapturedModule", "ModuleUpdate", "prepare_module", "DynamicBatch", "BatchVariants", "VariantContractError",
           "CapturedInference", "Inference", "InferenceIR", "LoopInferenceIR", "capture_inference",
           "capture_loop_inference", "capture_module_inference", "prepare_inference",
           "CapturedConditional", "capture_conditional_training",
           "BoundedLoopIR", "CapturedLoop", "ConditionalBody", "OperationIR",
           "UniformOperationIR", "capture_loop_training", "NestedLoopBody", "NestedOperationIR",
           "CaptureError", "Operation", "Ref", "TrainingIR", "Value"]

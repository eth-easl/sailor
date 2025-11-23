import os

import torch
import torch.nn.functional as F

from megatron import get_args, core
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from .module import MegatronModule, fp32_to_float16, float16_to_fp32
from .utils import init_method_normal, scaled_init_method_normal, attention_mask_func

from .language_model import EmbeddingPipe
from .transformer import ParallelTransformerLayerPipe, LMHeadPipe, get_num_experts_per_layer
from .enums import AttnMaskType, AttnType
from megatron.model import LayerNorm, RMSNorm

from .language_model import parallel_lm_logits
from megatron.core import mpu, tensor_parallel, sequence_parallel
from .utils import init_method_normal, scaled_init_method_normal, gather_and_init
from deepspeed.accelerator import get_accelerator

from megatron.model.fused_softmax import FusedScaleMaskSoftmax

# hard-coded here,
# TODO: pass as argument
config_layers = [
    "global",
    "local",
    "global",
    "local",
    "global",
    "local",
    "global",
    "local",
    "global",
    "local",
    "global",
    "local",
    "global",
    "local",
    "global",
    "local",
    "global",
    "local",
    "global",
    "local",
    "global",
    "local",
    "global",
    "local",
    "global",
    "local",
    "global",
    "local",
    "global",
    "local",
    "global",
    "local"
]



def CrossEntropy(output, labels):
    labels, loss_mask = labels[0], labels[1]

    args = get_args()

    # [b s] => [s b]
    labels = labels.transpose(0, 1).contiguous()
    losses = tensor_parallel.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    # [s b] => [b, s]
    losses = losses.transpose(0, 1).contiguous()
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss

class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, labels):
        return CrossEntropy(output, labels)

################# LAYERS ##################

class GPTNeoParallelAttention(MegatronModule):
    def __init__(self, init_method, config, output_layer_init_method, layer_id):

        super(GPTNeoParallelAttention, self).__init__()

        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

        self.layer_id = layer_id
        self.attention_layers = config_layers
        self.attention_type = self.attention_layers[layer_id]
        self.attn_mask_type = AttnMaskType.causal
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if args.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        max_positions = args.max_position_embeddings
        bias = torch.tril(torch.ones((max_positions, max_positions), dtype=bool)).view(
            1, 1, max_positions, max_positions
        )

        # TODO: not sure about these
        self.register_buffer("bias", bias)
        self.register_buffer("masked_bias", torch.tensor(-1e9))

        if self.attention_type == "local":
            bias = torch.bitwise_xor(bias, torch.tril(bias, -config.window_size))

        self.attn_dropout = torch.nn.Dropout(float(args.attention_dropout))
        self.resid_dropout = torch.nn.Dropout(float(args.residual_dropout))

        self.embed_dim = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        world_size = mpu.get_tensor_model_parallel_world_size()
        self.num_attention_heads_per_partition = core.utils.divide(
            args.num_attention_heads, world_size)
        self.hidden_size_per_attention_head = core.utils.divide(
            self.embed_dim, args.num_attention_heads)
        self.hidden_size_per_partition = core.utils.divide(self.embed_dim, world_size)

        self.query_key_value = tensor_parallel.ColumnParallelLinear(
            self.embed_dim,
            3 * self.embed_dim,
            config=config,
            gather_output=False,
            init_method=init_method
        )

        coeff = None
        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

         # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            config=config,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def forward(self, hidden_states,  attention_mask, layer_past=None, get_key_value=False):
        # TODO
          # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
                             (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)

        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer,key_layer,value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_x_layer, 3)


###############################################################################################################

         # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=get_accelerator().current_device_name())

        # TODO: what to use here? (also for alpha and beta)
        self.norm_factor = 1.0
        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0 / self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                                     ...,
                                     attention_scores.size(3) - 1,
                                     :attention_scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                                     ...,
                                     :attention_scores.size(3),
                                     :attention_scores.size(3)]


###############################################################################################################

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)


        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, _ = self.dense(context_layer)
        output = self.resid_dropout(output)

        # if get_key_value:
        #     output = [output, present]

        return output


class GPTNeoParallelMLP(MegatronModule):
    def __init__(self, init_method, config, output_layer_init_method, intermediate_size,  moe=False, enable_expert_tensor_parallelism=False):
        args = get_args()
        super(GPTNeoParallelMLP, self).__init__()

        embed_dim = args.hidden_size
        self.c_fc = tensor_parallel.ColumnParallelLinear(
            embed_dim,
            intermediate_size,
            config=config,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
        )

        self.c_proj = tensor_parallel.RowParallelLinear(
            intermediate_size,
            embed_dim,
            config=config,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism)

        self.activation_fn = F.gelu
        self.drop = torch.nn.Dropout(float(args.residual_dropout))

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)[0]
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.c_proj(hidden_states)[0]
        hidden_states = self.drop(hidden_states)
        return hidden_states


class GPTNeoParallelTransformerLayer(MegatronModule):
    def __init__(self, init_method, config, output_layer_init_method,
                 layer_number, moe=False, enable_expert_tensor_parallelism=False) -> None:
        args = get_args()
        super(GPTNeoParallelTransformerLayer, self).__init__()

        hidden_size = args.hidden_size
        inner_dim = 4 * hidden_size #args.intermediate_size if args.intermediate_size is not None else 4 * hidden_size

        self.ln_1 = torch.nn.LayerNorm(hidden_size, eps=args.layernorm_epsilon)
        self.attn = GPTNeoParallelAttention(init_method, config, output_layer_init_method, layer_number)
        self.ln_2 = torch.nn.LayerNorm(hidden_size, eps=args.layernorm_epsilon)
        self.mlp = GPTNeoParallelMLP(init_method, config, output_layer_init_method, inner_dim)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=None,
        use_cache=None,
        past_key_value=None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=past_key_value,
            attention_mask=attention_mask,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        # TODO: return sth else here?
        return hidden_states



class GPTNeoParallelTransformerLayerPipe(GPTNeoParallelTransformerLayer):
    def forward(self, inputs, **kwargs):

        # TODO: from LLAMA, not sure if this would work
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if torch.is_tensor(inputs) or len(inputs) == 1:
            # No attention mask forwarded, search for args.attn_mask
            if not hasattr(self, '_args'):
                self._args = get_args()
            hidden_states, attention_mask = inputs, self._args.attn_mask
            return super().forward(hidden_states, attention_mask, **kwargs)
        elif len(inputs) == 2:
            # Attention mask is an activation.
            hidden_states, attention_mask = inputs[0], inputs[1]
            return super().forward(*inputs, **kwargs), attention_mask
        else:
            raise RuntimeError('Received more inputs than understood.')



########################################################################################################################

class GPTNeoEmbedding(MegatronModule):
    """OPT Embedding, adapted from language_model.py"""

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_sequence_length,
                 embedding_dropout_prob,
                 config,
                 num_tokentypes=0,
                 embedding_weights_in_fp32=False):

        super(GPTNeoEmbedding, self).__init__()
        args = get_args()

        self.embed_dim = hidden_size
        self.init_method = config.init_method
        self.add_position_embedding = args.add_position_embedding
        # Word embeddings (parallel).
        self.embedding_weights_in_fp32 = embedding_weights_in_fp32
        self.params_dtype = args.params_dtype

        self.wte = tensor_parallel.VocabParallelEmbedding(
            vocab_size, self.embed_dim, config=config, init_method=self.init_method)
        self._word_embeddings_key = 'word_embeddings'

        # on HF, this is a 'OPTLearnedPositionalEmbedding' - how do they compare?
        self._position_embeddings_key = 'position_embeddings'
        if args.sequence_parallel:
            self.wpe = tensor_parallel.layers.SequenceParallelPositionEmbedding(
                max_sequence_length, self.embed_dim)
            # Initialize the position embeddings.
            self.init_method(self.wpe.local_embeddings.weight)
        else:
            self.wpe = torch.nn.Embedding(
                max_sequence_length, self.embed_dim)
            # Initialize the position embeddings.
            if args.perform_initialization:
                if args.zero_stage == 3:
                    gather_and_init(self.wpe.weight, self.init_method)
                else:
                    self.init_method(self.wpe.weight)

        self.drop = torch.nn.Dropout(float(args.hidden_dropout))

        self.sequence_parallel = args.sequence_parallel


    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        if self.add_position_embedding:
            self.position_embeddings.weight.data.fill_(0)
            self.position_embeddings.weight.shared = True


    @classmethod
    def from_pretrained(cls, model_path, config=None):
        module = torch.nn.utils.skip_init(cls, config).eval()  # fast init
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, 'pytorch_embs.pt',
            )))
        except:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module


    def forward(self, input_ids, position_ids, token_type_ids=None):
        # Embeddings.
        if self.embedding_weights_in_fp32:
            self.wte = self.wte.to(torch.float32)
        words_embeddings = self.wte(input_ids)

        if self.embedding_weights_in_fp32:
            words_embeddings = words_embeddings.to(self.params_dtype)

        position_embeddings = self.wpe(position_ids)
        embeddings = words_embeddings + position_embeddings

        if token_type_ids is not None:
            token_type_embedddings = self.wte(token_type_ids)
            embeddings += token_type_embedddings

        embeddings = self.drop(embeddings)

        return embeddings

class GPTNeoEmbeddingPipe(GPTNeoEmbedding):

    def forward(self, inputs, **kwargs):

        if not hasattr(self, '_args'):
            self._args = get_args()

        input_ids = inputs[0]
        position_ids = inputs[1]
        if hasattr(self._args, 'attn_mask'):
            attention_mask = None
        else:
            attention_mask = inputs[2]

        if len(inputs) == 4:
            tokentype_ids = inputs[3]
        else:
            tokentype_ids = None

        embeddings = super().forward(input_ids, position_ids, tokentype_ids)

        # If cmd args has attn_mask, we don't forward it as an activation.
        if hasattr(self._args, 'attn_mask'):
            return embeddings
        else:
            assert False
            return embeddings, attention_mask


    @property
    def word_embeddings_weight(self):
        """Easy accessory for the DeepSpeed pipeline engine to tie embeddings across stages."""
        return self.word_embeddings.weight


class GPTNeoModelPipe(PipelineModule,MegatronModule):
    """GPT-Neo Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 use_embedding=True,
                 use_transformer=True,
                 use_last=True,
                 layers_per_stage=None
        ):
        args = get_args()
        self.parallel_output = parallel_output

        if config.init_method is None:
            config.init_method = init_method_normal(config.init_method_std)

        if config.output_layer_init_method is None:
            config.output_layer_init_method = scaled_init_method_normal(config.init_method_std,
                                                                        config.num_layers)

        self.init_method = config.init_method
        self.output_layer_init_method = config.output_layer_init_method

        self.specs = []

        def _to_float16(inputs):
            if args.fp16:
                return fp32_to_float16(inputs, lambda v: v.half())
            elif args.bf16:
                return fp32_to_float16(inputs, lambda v: v.bfloat16())
            else:
                return inputs

        if use_embedding:
            # Embedding layers
            self.specs.append(_to_float16)
            self.specs.append(LayerSpec(GPTNeoEmbeddingPipe,
                args.hidden_size,
                args.padded_vocab_size,
                args.max_position_embeddings,
                args.hidden_dropout,
                config,
                num_tokentypes=num_tokentypes,
                embedding_weights_in_fp32=args.embedding_weights_in_fp32,))

            if args.fp32_residual_connection:
                self.specs.append(lambda x: x.transpose(0, 1).contiguous().float())
            else:
                self.specs.append(lambda x: x.transpose(0, 1).contiguous())

            if layers_per_stage:
                layers_per_stage[0]+=2 # for the embedding

        # TODO: add dropout (or within the embedding)

        if use_transformer:
            for layer_idx in range(args.num_layers):
                print(f"Add layer with idx {layer_idx}")
                self.specs.append(
                    LayerSpec(GPTNeoParallelTransformerLayerPipe, self.init_method, config, self.output_layer_init_method, layer_idx))

        if use_last:
            self.specs.append(LayerSpec(LayerNorm, args.hidden_size, eps=args.layernorm_epsilon))

            # Convert to fp32 if needed
            if args.fp16 or args.bf16:
                self.specs.append(float16_to_fp32)
                if layers_per_stage:
                    layers_per_stage[-1]+=1

        # Cache losses
        self.moe_loss = None
        self.last_lm_loss = None    # detached, for display only
        self.last_moe_loss = None   # detached, for display only

        if args.checkpoint_activations:
            interval = args.checkpoint_num_layers
        elif args.recompute_granularity == "full" and args.recompute_method == 'uniform':
            # deepspeed's pipeline doesn't support the block recompute method
            interval = args.recompute_num_layers
        else:
            interval = 0

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology, PipeHeterogeneousTopology
        if args.distributed_config_file:
            topo = PipeHeterogeneousTopology(
                num_pp=mpu.get_pipeline_model_parallel_world_size(),
                num_mp=mpu.get_tensor_model_parallel_world_size(),
                num_dp=mpu.get_data_parallel_world_size(),
                input_file=args.distributed_config_file
            )
        else:
            topo = PipeModelDataParallelTopology(
                num_pp=mpu.get_pipeline_model_parallel_world_size(),
                num_mp=mpu.get_tensor_model_parallel_world_size(),
                num_dp=mpu.get_data_parallel_world_size()
            )

        if layers_per_stage:
            layers_per_stage[-1] += 1 # for the loss

        print(f"======================= layers_per_stage is {layers_per_stage}")

        if layers_per_stage:
            layer_partitioning = [0] * (len(layers_per_stage) + 1)
            for i in range(len(layers_per_stage)):
                layer_partitioning[i+1] = layer_partitioning[i] + layers_per_stage[i]
        else:
            layer_partitioning = None

        print(f"============================================ layer_partitioning is {layer_partitioning}")


        super().__init__(layers=self.specs,
                         loss_fn=Loss(),
                         topology=topo,
                         activation_checkpoint_interval=interval,
                         partition_method='uniform',
                         layer_partitioning=layer_partitioning)

    def get_additional_losses(self):
        return None

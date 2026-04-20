import torch
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
from megatron.core import tensor_parallel

NO_USE_TOKEN_ID = 128004 # Never show up in data


class MultiLayerVocabParallelEmbedding(VocabParallelEmbedding):

    def forward(self, input_):
        # Handle Padding, _PAD_TOKEN_ID is -1
        pad_mask = (input_ == -1)
        masked_input_for_padding = input_.clone()
        masked_input_for_padding[pad_mask] = NO_USE_TOKEN_ID

        # task_id: 0
        if input_.dim() == 2:
            # Standard Causal LM Task，Input Shape [B, S]
            output = super().forward(masked_input_for_padding)

            output = output.clone() * (~pad_mask).unsqueeze(-1)
            return output


        # task_id = 1: q0 -> q1 # Super Resolution Task，Input Shape [B, S, 1]
        elif input_.dim() == 3 and input_.size(-1) == 1:

            masked_input_for_padding = masked_input_for_padding.squeeze(-1)
            pad_mask = pad_mask.squeeze(-1)

            output = super().forward(masked_input_for_padding)
            output = output.clone() * (~pad_mask).unsqueeze(-1)
            return output

        # task_id > 1 # Super Resolution Task，Input Shape [B, S, C]
        elif input_.dim() == 3 and input_.size(-1) > 1:

            audio_mask = (input_[..., 1:] != -1).any(dim=2)

            # -------------- Process Text ---------------------
            text_ids = input_[..., 0]  # [B, S]，include TEXT / q0 / -1 (pad)
            text_pad = (text_ids == -1)

            text_ids_for_lookup = text_ids.masked_fill(audio_mask | text_pad, NO_USE_TOKEN_ID)
            text_emb = super().forward(text_ids_for_lookup)
            text_keep = (~audio_mask) & (~text_pad)
            text_emb = text_emb * text_keep.unsqueeze(-1) # [B, S, H]

            masked_input = input_.masked_fill(input_ == -1, NO_USE_TOKEN_ID)

            #  ------------------- Step 1: Embedding lookup -------------------
            if self.tp_group.size() > 1:
                # Build the mask.
                input_mask = (masked_input < self.vocab_start_index) | (masked_input >= self.vocab_end_index)
                # Mask the input.
                masked_input_local = masked_input.clone() - self.vocab_start_index
                masked_input_local[input_mask] = 0
            else:
                input_mask = None
                masked_input_local = masked_input

            # F.embedding on [B, S, C] with weight [V_part, H] -> [B, S, C, H]
            if self.deterministic_mode:
                output_parallel_4d = self.weight[masked_input_local]
            else:
                output_parallel_4d = torch.nn.functional.embedding(masked_input_local, self.weight)

            # ------------------- Step 2: Zero out invalid embeddings -------------------
            if self.tp_group.size() > 1:
                # make embedding zero which does not belong to current GPU
                shard_mask4d = (~input_mask)[..., None]
                output_parallel_4d = output_parallel_4d * shard_mask4d

            # Make pad token zero [B, S, C]
            nonpad4d = (input_ != -1)[..., None]
            output_parallel_4d = output_parallel_4d * nonpad4d

            # Process Audio
            audio_sum = torch.sum(output_parallel_4d, dim=2) # [B, S, C, H] -> [B, S, H]
            audio_sum = audio_sum * audio_mask.unsqueeze(-1)

            # Merge Text + Audio
            output_parallel = text_emb + audio_sum

            if self.reduce_scatter_embeddings:
                # Not typically used with this kind of model, but keeping for completeness
                raise NotImplementedError("reduce_scatter_embeddings not implemented for 3D input")
            else:
                # Reduce across all the model parallel GPUs.
                output = reduce_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)

            return output
        else:
            raise ValueError(f"Unexpected input dimensions {input_.dim()}, expected 2 or 3.")


class MultiLayerLanguageModelEmbedding(LanguageModelEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom word_embeddings
        self.word_embeddings = tensor_parallel.MultiLayerVocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.config.hidden_size,
            init_method=self.config.embedding_init_method,
            reduce_scatter_embeddings=self.reduce_scatter_embeddings,
            config=self.config,
            tp_group=self.tp_group,
        )


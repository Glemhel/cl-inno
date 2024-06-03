from transformers import PreTrainedModel, PretrainedConfig
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class SimpleModelConfig(PretrainedConfig):
    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

class SimpleModelForPreTraining(PreTrainedModel):
    config_class = SimpleModelConfig

    def __init__(self, config: SimpleModelConfig):
        super().__init__(config)
        self.transformer = nn.Transformer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            num_encoder_layers=config.num_hidden_layers,
            num_decoder_layers=0,  # We're not using the decoder at this stage.
            dim_feedforward=config.intermediate_size
        )
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None:
            attention_mask = self._get_extended_attention_mask(attention_mask, input_ids.shape)
        
        embeddings = self.embedding(input_ids)
        transformer_output = self.transformer(embeddings, src_key_padding_mask=attention_mask)
        lm_logits = self.lm_head(transformer_output[0])
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=lm_logits,
            pooler_output=None,
            hidden_states=None,
            attentions=None
        )

"""Immutable public identities; no model or corpus bytes are bundled."""
MODEL_ID = "FlameF0X/TinyMoE-200m-2x16"
MODEL_REVISION = "9b209f7439aadfac282e0bf44f2fda19771aba90"
TRANSFORMERS_VERSION = "4.55.4"
TRANSFORMERS_REVISION = "d79b2d981f28b2730d402244ac3c2e9a8c054eee"
DATASET_ID = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_CONFIG = "wikitext-103-raw-v1"
EXPECTED_CONFIG = dict(model_type="mixtral", hidden_size=384, intermediate_size=768,
    num_hidden_layers=12, num_attention_heads=8, num_key_value_heads=4,
    num_local_experts=16, num_experts_per_tok=2, vocab_size=32000,
    max_position_embeddings=1024, sliding_window=1024, rope_theta=1000000.0,
    rms_norm_eps=1e-6, router_aux_loss_coef=0.02, router_jitter_noise=0.0,
    attention_dropout=0.0, hidden_act="silu", tie_word_embeddings=False,
    output_router_logits=True, use_cache=False)
# Embedding + untied head; 12*(Q/K/V/O + 16 SwiGLUs + gate + two norms); final norm.
EXPECTED_PARAMETERS = 199837056


def public_identity():
    return dict(model_id=MODEL_ID, model_revision=MODEL_REVISION,
                tokenizer_id=MODEL_ID, tokenizer_revision=MODEL_REVISION,
                transformers_version=TRANSFORMERS_VERSION,
                transformers_revision=TRANSFORMERS_REVISION,
                dataset_id=DATASET_ID, dataset_revision=DATASET_REVISION,
                dataset_config=DATASET_CONFIG)

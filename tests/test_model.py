import torch

from qwen35_nano.model import VisualARDecoder


def test_visual_decoder_outputs_expected_shape() -> None:
    decoder = VisualARDecoder(
        image_vocab_size=16384,
        image_seq_len=256,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        encoder_hidden_size=128,
    )
    decoder_input_ids = torch.randint(0, 16384, (2, 17))
    encoder_hidden_states = torch.randn(2, 8, 128)
    encoder_attention_mask = torch.ones(2, 8, dtype=torch.long)
    logits = decoder(
        decoder_input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
    )
    assert logits.shape == (2, 17, 16387)

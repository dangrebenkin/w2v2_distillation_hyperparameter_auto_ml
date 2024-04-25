import torch
import torchaudio


class ConformerSpeechRecognizer(torch.nn.Module):
    def __init__(self,
                 kernel_size,
                 ffn_dim: int,
                 feature_vector_size: int,
                 hidden_layer_size: int,
                 num_layers: int,
                 num_heads: int,
                 dropout: float,
                 depthwise_conv_kernel_size: int,
                 vocabulary_size: int):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.cnn_ = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=feature_vector_size),
            torch.nn.Conv1d(
                in_channels=feature_vector_size,
                out_channels=hidden_layer_size,
                bias=False,
                kernel_size=(kernel_size,),
                padding='same'
            ),
            torch.nn.BatchNorm1d(num_features=hidden_layer_size)
        )
        self.conformer_ = torchaudio.models.Conformer(
            input_dim=hidden_layer_size,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout
        )
        self.proba_appoximator_ = torch.nn.Linear(
            in_features=hidden_layer_size,
            out_features=vocabulary_size
        )

    def forward(self, inputs: torch.Tensor, input_lenghts: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.nn.functional.gelu(self.cnn_(inputs.permute(0, 2, 1)))
        hidden_states, _ = self.conformer_(hidden_states.permute(0, 2, 1), input_lenghts)
        output_logits = self.proba_appoximator_(hidden_states)
        return output_logits

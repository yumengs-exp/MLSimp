import torch
import torch.nn as nn
import torch.nn.functional as F
from .PositionalEncoding import PositionalEncoding
from .nn import (
    SiLU,
    linear,
    timestep_embedding,
)

class SimpTransformer(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_dim,
        encoder_n_head,
        encoder_hidden_dim,
        encoder_n_layer,
        in_channels,
        hidden_channels,
        out_channels,
        n_head,
        n_layer,
        trans_hidden_channels,
        attn_dropout,
        dropout
    ):
        super().__init__()

        # if num_heads_upsample == -1:
        #     num_heads_upsample = num_heads
        # if config is None:
        #     config = AutoConfig.from_pretrained('bert-base-uncased')

        # self.gen_simp_model_channels  = gen_simp_model_channels


        self.model_channels = model_channels = hidden_channels

        encoder_layer = nn.TransformerEncoderLayer(in_dim, encoder_n_head, encoder_hidden_dim, batch_first=True)
        self.traj_embed = nn.TransformerEncoder(encoder_layer, encoder_n_layer)
        self.traj_transform = nn.Sequential(
            nn.Linear(in_dim, encoder_hidden_dim),
            nn.Tanh(),
            nn.Linear(encoder_hidden_dim, in_channels),
        )



        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, hidden_channels),
        )
        self.time_embed_transform = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.input_up_proj = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                                           nn.Tanh(),
                                           nn.Linear(hidden_channels, hidden_channels)) #nn.Tanh()

        enc_layer = nn.TransformerEncoderLayer(hidden_channels, n_head, trans_hidden_channels,attn_dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layer)
        self.position_embeddings  = PositionalEncoding(hidden_channels)

        self.dropout = nn.Dropout(dropout)

        self.output_down_proj = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                             nn.Tanh() , nn.Linear(hidden_channels, out_channels)) #nn.Tanh()



    def save_model(self, file_path):
        # Save the model state_dict to the specified file path
        torch.save(self.state_dict(), file_path)
        print(f"Model saved at {file_path}")

    def load_model(self, file_path):
        # Load the model state_dict from the specified file path
        try:
            self.load_state_dict(torch.load(file_path))
            print(f"Model loaded from {file_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {file_path}")
        except Exception as e:
            print(f"Error: Failed to load model. {e}")

    def get_embeds(self, traj_input,  padding_mask=None):
        x = self.traj_embed(traj_input, src_key_padding_mask=padding_mask)
        x = F.normalize(x)
        x= self.traj_transform(x)
        x = F.normalize(x)
        return x


    def get_logits(self, hidden_repr, doc_embed, test=False):
        if self.logits_mode == 1:
            if test:
                return torch.bmm(hidden_repr, doc_embed.permute(0, 2, 1)) # (Batch, Max Word Len, Max Article Vocab Len)

            return self.lm_head(hidden_repr) # (Batch, Max Word Len, Vocab Size)
        else:
            raise NotImplementedError


    def forward(self, x, x_t_mask, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """


        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        emb = self.time_embed_transform(emb)
        emb = self.dropout(F.normalize(emb))

        emb_x = self.input_up_proj(x)
        emb_x = F.normalize(emb_x)
        seq_length = x.size(1)
        emb_x += emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = emb_x

        emb_inputs = F.normalize(emb_inputs)
        emb_inputs = self.position_embeddings(emb_inputs)
        input_trans_hidden_states = self.encoder(emb_inputs,src_key_padding_mask = x_t_mask)
        input_trans_hidden_states = F.normalize(input_trans_hidden_states)
        h = self.output_down_proj(input_trans_hidden_states)
        h = F.normalize(h)
        h = h.type(x.dtype)
        return h


    # def get_feature_vectors(self, x, timesteps, y=None):
    #     """
    #     Apply the model and return all of the intermediate tensors.
    #
    #     :param x: an [N x C x ...] Tensor of inputs.
    #     :param timesteps: a 1-D batch of timesteps.
    #     :param y: an [N] Tensor of labels, if class-conditional.
    #     :return: a dict with the following keys:
    #              - 'down': a list of hidden state tensors from downsampling.
    #              - 'middle': the tensor of the output of the lowest-resolution
    #                          block in the model.
    #              - 'up': a list of hidden state tensors from upsampling.
    #     """
    #     hs = []
    #     emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
    #     if self.num_classes is not None:
    #         assert y.shape == (x.shape[0],)
    #         emb = emb + self.label_emb(y)
    #     result = dict(down=[], up=[])
    #     h = x.type(self.inner_dtype)
    #     for module in self.input_blocks:
    #         h = module(h, emb)
    #         hs.append(h)
    #         result["down"].append(h.type(x.dtype))
    #     h = self.middle_block(h, emb)
    #     result["middle"] = h.type(x.dtype)
    #     for module in self.output_blocks:
    #         cat_in = th.cat([h, hs.pop()], dim=-1)
    #         h = module(cat_in, emb)
    #         result["up"].append(h.type(x.dtype))
    #     return result
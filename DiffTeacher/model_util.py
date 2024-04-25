
from .Diffusion import SpacedDiffusion
from .Transformer import SimpTransformer
from . import GaussianDiffusion as gd
from .Diffusion import space_timesteps

NUM_CLASSES = 1000



def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        model_arch='trans-unet',
        in_channel=8,
        out_channel=8,
        training_mode='emb',
        vocab_size=66,
        config_name='bert-base-uncased',
        experiment_mode='lm',
        logits_mode=1,
    )



def create_model_and_diffusion(
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
        dropout,

    learn_sigma,
    sigma_small,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    training_mode,
):


    model = create_model(
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
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl, # false
        predict_xstart=predict_xstart, # true
        rescale_timesteps=rescale_timesteps, # true
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        training_mode=training_mode, # e2e
    )
    return model, diffusion


def create_model(

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


    return SimpTransformer(
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
    )




def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    training_mode='emb',
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if training_mode == 'e2e':
        # end to end training
        if use_kl:
            loss_type = gd.LossType.E2E_KL
        else:
            loss_type = gd.LossType.E2E_MSE
    elif training_mode == 'e2e-simple':
        if use_kl:
            loss_type = gd.LossType.E2E_Simple_KL
        else:
            loss_type = gd.LossType.E2E_Simple_MSE

    else:
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    print(loss_type, learn_sigma)
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        training_mode=training_mode,
    )


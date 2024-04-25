import enum
import math

import numpy as np
import torch
import torch as th
from tqdm import tqdm
import torch.nn.functional as F

from .nn import mean_flat

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar2(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar2(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
    E2E_KL = enum.auto()
    E2E_MSE = enum.auto()
    E2E_Simple_MSE = enum.auto()
    E2E_Simple_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        training_mode='e2e',
        # model_arch='conv-unet',
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.training_mode = training_mode
        print('training mode is ', training_mode)
        self.mapping_func = None


    def training_losses(self, model, *args, **kwargs):
        if self.training_mode == 'e2e':
            return self.training_losses_e2e(model, *args, **kwargs)



    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(
        self, model, x, mask, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        # if self.model_arch == 'conv-unet' or self.model_arch == '1d-unet':
        #     B, C = x.shape[:2]
        # else:
        #     B, C = x.size(0), x.size(-1)
        # assert t.shape == (B,)

        model_output = model.model(x, mask, self._scale_timesteps(t))

        model_variance, model_log_variance = (self.posterior_variance, self.posterior_log_variance_clipped)
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        pred_xstart = model_output
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "pred_xstart_mean": model_output,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self, model, x, mask, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,
            top_p=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            mask,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if top_p is not None and top_p > 0:
            # print('top_p sampling')
            noise = th.randn_like(x)
            replace_mask = th.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = th.randn_like(noise[replace_mask])
                replace_mask = th.abs(noise) > top_p
            assert (th.abs(noise) <= top_p).all()

        else:
            noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"],
                'greedy_mean':out["mean"], 'out':out, 'pred_xstart_mean': out["pred_xstart_mean"]}



    def p_sample_loop(
        self,
        model,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        pred_xstarts = []
        final = None
        for sample in (self.p_sample_loop_progressive(
            model,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            top_p=top_p,
        )):
            final = sample
        return final

    def p_sample_loop_progressive(
        self,
        model,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        diffusion_steps = model_kwargs['diffusion_steps']
        batch = model_kwargs['batch']
        amplify_len = model_kwargs['amplify_len']



        trajs_padding, padding_mask, simp_trajs_padding, simp_padding_mask, labels, labels_mask = batch
        x_start_mean = model.model.get_embeds(trajs_padding, padding_mask)
        simp_padding_mask = torch.zeros(padding_mask.size(0),amplify_len).to(device)
        pad_mask = torch.cat([padding_mask,simp_padding_mask],dim=-1)
        shape = (x_start_mean.shape[0], x_start_mean.shape[1] + amplify_len, x_start_mean.shape[2])
        input = torch.randn(*shape).to(device)
        input[:, :trajs_padding.shape[1], :] = x_start_mean



        indices = list(range(diffusion_steps))[::-1]


        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    input,
                    pad_mask,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    top_p=top_p,
                )

                x_gen = out['sample']
                x_gen[:,:trajs_padding.shape[1],:] = x_start_mean
                out['sample'] = x_gen
                yield out
                # if i == 0:
                #     max_simp_len = sample_lens.max().item()
                #     x_gen = out['sample'][:,-max_simp_len:,:]
                #
                #     x_embeds = x_start_mean[:,:x_start_mean.size(1)-max_simp_len,:]
                #     # prob = torch.sum(x_gen.unsqueeze(2) * x_embeds.unsqueeze(1), dim=3)
                #     result_mask = simp_match_test(x_gen,x_embeds,result_mask,padding_mask[:,:result_mask.size(1)],sample_lens)
                #
                #     # simp_gen_match = x_gen @ x_start_mean[:,:-max_simp_len,:].T
                #     # simp_gen_match  = simp_gen_match * ~padding_mask[:,:-max_simp_len,:]
                #     # simp_gen_idx = F.gumbel_softmax(simp_gen_match, dim=0)
                #     yield result_mask
                # x_start_mean = out['sample']


    def get_x_start(self, x_start_mean, std):
        '''
        Using the interpolating policy OR using the convolution policy...
        :param x_start_mean:
        :return:
        '''
        noise = th.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        return (
             x_start_mean + std * noise
        )

    def token_discrete_loss(self, x_t, doc_emb, doc_mask, get_logits, input_ids, model_kwargs=None):
        if self.model_arch == 'conv-unet' or  self.model_arch == '1d-unet':
            reshaped_x_t = x_t.view(x_t.size(0), x_t.size(1), -1).permute(0, 2, 1)
        else:
            reshaped_x_t = x_t
        logits = get_logits(reshaped_x_t, doc_emb, doc_mask) # bsz, seqlen, vocab


        if model_kwargs['output_sen']:
            tokenizer = model_kwargs['tokenizer']
            t0_mask = model_kwargs['t0_mask']
            cands = th.topk(torch.softmax(logits, dim=-1), k=1, dim=-1)
            for i, seq in enumerate(cands.indices):
                if isinstance(tokenizer, dict):
                    if t0_mask[i]:
                        tokens = " ".join([tokenizer[x[0].item()] for x in seq if x[0].item() > 3])
                else:
                    tokens = tokenizer.decode(seq.squeeze(-1))
        loss_fct = th.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.shape[-1]), input_ids.view(-1)).view(logits.shape[0], -1)
        decoder_nll = decoder_nll.mean(dim=-1)
        return decoder_nll

    def x0_helper(self, model_output, x, t):
        pred_xstart = self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
        pred_prev = model_output

        return {'pred_xprev':pred_prev, 'pred_xstart':pred_xstart}


    def training_losses_e2e(self, model, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        trajs_padding = model_kwargs['trajs_padding']
        padding_mask = model_kwargs['padding_mask']
        simp_trajs_padding = model_kwargs['simp_trajs_padding']
        simp_padding_mask = model_kwargs['simp_padding_mask']
        labels = model_kwargs['labels']
        labels_mask = model_kwargs['labels_mask']

        pad_mask = torch.cat([padding_mask,simp_padding_mask],dim=-1)

        x_start_mean = torch.cat([trajs_padding, simp_trajs_padding], dim=1)
        x_start_mean = model.model.get_embeds(x_start_mean,pad_mask)
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   th.tensor([0]).to(x_start_mean.device),
                                   x_start_mean.shape)


        x_start = self.get_x_start(x_start_mean, std)

        if noise is None:
            noise = th.randn_like(x_start)  # eps
        x_t = self.q_sample(x_start, t, noise=noise)  # reparametrization trick.
        x_t[:, :trajs_padding.shape[1], :] = x_start_mean[:, :trajs_padding.shape[1], :]


        model_output = model.model(x_t, pad_mask, self._scale_timesteps(t))


        terms = {}


        target = x_start
        assert model_output.shape == target.shape == x_start.shape

        terms["mse"] = mean_flat((x_start_mean[:, trajs_padding.shape[1]:, : ] * 10 - model_output[:, trajs_padding.shape[1]:, : ] * 10) ** 2).mean()
        decoder_logits = torch.bmm(x_start_mean[:, trajs_padding.shape[1]:, : ], x_start_mean[:, :trajs_padding.shape[1], : ].permute(0, 2, 1))

        ce_loss = th.nn.CrossEntropyLoss()
        terms['decoder_nll'] = ce_loss(decoder_logits.view(-1, decoder_logits.shape[-1]),labels.view(-1))

        out_mean, _, _ = self.q_mean_variance(x_start, th.LongTensor([self.num_timesteps - 1]).to(x_start.device))
        terms['tT_loss'] = mean_flat(out_mean ** 2).mean()


        terms["loss"] = terms["mse"] + terms['tT_loss'] +0.1*terms['decoder_nll']

        return terms


        #######################################33
        # if ext_rate ==1:
        #     terms = {}
        #     simp_input = traj_input[result_mask].view(traj_input.size(0), sample_len + 2, traj_input.size(-1))
        #     terms["contra_loss"] = model.model.moco(traj_input, simp_input, padding_mask, trajs_len, sample_len)
        #     terms["mse"] = 0
        #     terms['tT_loss'] = 0
        #     terms["dissim"] =0
        #     terms["loss"] = terms["mse"] + terms['tT_loss'] + terms['contra_loss']  + terms["dissim"]
        # else:
        #     x_ext_simp = x_embeds[result_mask].view(x_embeds.size(0),torch.sum(result_mask,dim=-1)[0].item(),x_embeds.size(-1))
        #     x_start_mean = torch.cat((x_embeds, x_ext_simp, gd_padding), dim=1)
        #     std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
        #                                th.tensor([0]).to(x_start_mean.device),
        #                                x_start_mean.shape)
        #
        #     x_start = self.get_x_start(x_start_mean, std)
        #
        #     if noise is None:
        #         noise = th.randn_like(x_start) # eps
        #     x_t = self.q_sample(x_start, t, noise=noise) # reparametrization trick.
        #     x_t[:, :traj_input.shape[1], :] = x_start_mean[:, :traj_input.shape[1], :]
        #
        #
        #     model_output = model.model(x_t, gd_padding_mask, self._scale_timesteps(t))
        #
        #     simp_gen = model_output[:, -gd_padding.shape[1]:, : ]
        #     prob = torch.sum(simp_gen.unsqueeze(2) * x_embeds.unsqueeze(1), dim=3)
        #     result_mask = simp_match(prob,result_mask,padding_mask)
        #
        #
        #
        #
        #
        #     terms = {}
        #     simp_input = traj_input[result_mask].view(traj_input.size(0),sample_len+2,traj_input.size(-1))
        #     terms["contra_loss"] = model.model.moco(traj_input,simp_input, padding_mask,sample_len,result_mask)*0.1
        #
        #     target = x_start
        #     assert model_output.shape == target.shape == x_start.shape
        #
        #     terms["mse"] = mean_flat((x_start_mean[:, traj_input.shape[1]:traj_input.shape[1]+x_ext_simp.shape[1], : ] * 10 - model_output[:, traj_input.shape[1]:traj_input.shape[1]+x_ext_simp.shape[1], : ]* 10) ** 2).mean()
        #
        #     out_mean, _, _ = self.q_mean_variance(x_start, th.LongTensor([self.num_timesteps - 1]).to(x_start.device))
        #     terms['tT_loss'] =  mean_flat(out_mean ** 2).mean()
        #
        #     terms["dissim"] = torch.mean(torch.mean(F.cosine_similarity(simp_gen.unsqueeze(2),simp_gen.unsqueeze(1),dim=-1),dim=1),dim=1).mean()
        #     # terms["dissim"] = torch.Tensor([0]).cuda()
        #     terms["loss"] = terms["mse"] + terms['tT_loss']+ terms['contra_loss'] +terms["dissim"]

        return terms



def simp_match(prob, result_mask,padding_mask):
    # for i in range(x_gen.shape[0]):
    #     prob = x_gen[i] @ (x_embeds[i].T)
    #     # p = prob[result_mask[i].sum():]
    #     p = torch.sigmoid(prob)
    #     for j in range(p.shape[0]):
    #         mask = ~(result_mask[i] + padding_mask[i])
    #         mask = mask.float()
    #         mask[~mask.bool()] = float('-inf')
    #         nor_prob = F.normalize(p[j].unsqueeze(0)) * mask
    #         simp_gen_mask = F.gumbel_softmax(nor_prob, hard=True).bool()
    #         result_mask[i] = simp_gen_mask + result_mask[i]
    # a = result_mask.sum()
    prob = torch.sigmoid(prob)
    for i in range(prob.shape[1]):
        mask = ~(result_mask + padding_mask)
        # p = prob[:,i,:]
        mask= mask.float()
        mask[~mask.bool()] = float('-inf')
        nor_prob = F.normalize(prob[:,i,:] )* mask
        simp_gen_mask = F.gumbel_softmax(nor_prob, hard=True).bool()
        result_mask = simp_gen_mask + result_mask
    return result_mask


def simp_match_v2(prob, trajs_len,padding_mask):
    # for i in range(x_gen.shape[0]):
    #     prob = x_gen[i] @ (x_embeds[i].T)
    #     # p = prob[result_mask[i].sum():]
    #     p = torch.sigmoid(prob)
    #     for j in range(p.shape[0]):
    #         mask = ~(result_mask[i] + padding_mask[i])
    #         mask = mask.float()
    #         mask[~mask.bool()] = float('-inf')
    #         nor_prob = F.normalize(p[j].unsqueeze(0)) * mask
    #         simp_gen_mask = F.gumbel_softmax(nor_prob, hard=True).bool()
    #         result_mask[i] = simp_gen_mask + result_mask[i]
    # a = result_mask.sum()
    prob = torch.sigmoid(prob)

    idx = torch.zeros_like(padding_mask).bool().cuda()

    idx[:, 0] = True
    idx[torch.arange(len(trajs_len)), trajs_len - 1] = True



    for i in range(prob.shape[1]):
        mask = ~(idx+padding_mask)
        mask = mask.float()
        mask[~mask.bool()] = float('-inf')
        nor_prob = F.normalize(prob[:,i,:] )* mask
        simp_gen_mask = F.gumbel_softmax(nor_prob, hard=True).bool()
        idx = simp_gen_mask + idx
    return idx
def simp_match_test(x_gen,x_embeds, result_mask,padding_mask,sample_lens):
    # prob = torch.sigmoid(prob)
    for i in range(x_gen.shape[0]):
        prob = x_gen[i] @ (x_embeds[i].T)
        sample_len = sample_lens[i].item()
        p = prob[result_mask[i].sum():sample_len]
        p = torch.sigmoid(p)
        for j in  range(p.shape[0]):
            mask = ~(result_mask[i] + padding_mask[i])
            mask = mask.float()
            mask[~mask.bool()] = float('-inf')
            nor_prob = F.normalize(p[j].unsqueeze(0)) * mask
            simp_gen_mask = F.gumbel_softmax(nor_prob, hard=True).bool()
            result_mask[i] = simp_gen_mask + result_mask[i]

    return result_mask

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

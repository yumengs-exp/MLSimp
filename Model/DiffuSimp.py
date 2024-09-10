import torch

from Utils.dataset import read_traintrajs,DiffuSimpDataset,DiffuSimpcollate
from torch.utils.data.dataloader import DataLoader
from functools import partial

from DiffTeacher.model_util import create_model_and_diffusion
from  DiffTeacher.resample import create_named_schedule_sampler
from DiffTeacher.TrainLoop import TrainLoop


def train_diffusimp(trajs, simp_trajs_idx,params,load_model=False):


    device = params['device']
    x_max = params['maxlon']
    x_min = params['minlon']
    y_max = params['maxlat']
    y_min = params['minlat']
    diff_params = params['Diff']





    in_dim = diff_params['in_dim']
    encoder_n_head = diff_params['encoder_n_head']
    encoder_hidden_dim = diff_params['encoder_hidden_dim']
    encoder_n_layer = diff_params['encoder_n_layer']
    in_channels = diff_params['in_channels']
    hidden_channels = diff_params['hidden_channels']
    out_channels = diff_params['out_channels']
    n_head = diff_params['n_head']
    n_layer = diff_params['n_layer']
    trans_hidden_channels = diff_params['trans_hidden_channels']
    attn_dropout = diff_params['attn_dropout']
    dropout = diff_params['dropout']

    schedule_sampler = diff_params['schedule_sampler']
    lr = float(diff_params['lr'])
    batch_size = diff_params['batch_size']
    wd = diff_params['wd']
    epochs = diff_params['epochs']
    ema_rate = diff_params['ema_rate']
    log_interval = diff_params['log_interval']
    save_interval = diff_params['save_interval']
    resume_checkpoint = diff_params['resume_checkpoint']
    use_fp16 = diff_params['use_fp16']
    fp16_scale_growth = diff_params['fp16_scale_growth']
    weight_decay = diff_params['weight_decay']
    lr_anneal_steps = diff_params['lr_anneal_steps']
    checkpoint_path = diff_params['checkpoint_path']
    gradient_clipping = diff_params['gradient_clipping']
    eval_interval = diff_params['eval_interval']
    lambda1 = diff_params['lambda1']


    learn_sigma = diff_params['learn_sigma']
    sigma_small = diff_params['sigma_small']
    diffusion_steps = diff_params['diffusion_steps']
    noise_schedule = diff_params['noise_schedule']
    timestep_respacing = diff_params['timestep_respacing']
    use_kl = diff_params['use_kl']
    predict_xstart = diff_params['predict_xstart']
    rescale_timesteps = diff_params['rescale_timesteps']
    rescale_learned_sigmas = diff_params['rescale_learned_sigmas']
    training_mode = diff_params['training_mode']

    amplify_len  = diff_params['amplify_len']
    diff_step_eval = diff_params['diff_step_eval']

    train_set = DiffuSimpDataset(trajs,simp_trajs_idx)
    dataloader = DataLoader(train_set, batch_size=batch_size,shuffle=True,collate_fn=partial(DiffuSimpcollate, x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min,device=device))

    # ### init model
    model, diffusion = create_model_and_diffusion(

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
            )
    model = model.to(device)
    checkpoint_path = f'./ModelSave/{params["dataset"]}/DiffuSimp'
    if load_model:
        model.load_model(checkpoint_path)
    schedule_sampler = create_named_schedule_sampler(schedule_sampler, diffusion)

    TrainLoop(
        model=model,
        diffusion=diffusion,
        dataloader=dataloader,
        lr=lr,
        ema_rate= ema_rate,
        log_interval= log_interval,
        save_interval= save_interval,
        resume_checkpoint= resume_checkpoint,
        use_fp16= use_fp16,
        fp16_scale_growth= fp16_scale_growth,
        schedule_sampler= schedule_sampler,
        weight_decay= weight_decay,
        lr_anneal_steps= lr_anneal_steps,
        checkpoint_path=checkpoint_path,
        gradient_clipping=gradient_clipping,
        # eval_dataloader=test_dataloader,
        eval_interval=eval_interval,
        epochs=epochs,
        device =device
    ).run_loop()


    model.eval()
    dataloader_eval = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                            collate_fn=partial(DiffuSimpcollate, x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min,
                                               device=device))
    results= []
    with torch.no_grad():
        for batch in dataloader_eval:
            cond = {
                "diffusion_steps": diff_step_eval,
                "batch": batch,
                "amplify_len": amplify_len
            }

            sample_fn = (
                diffusion.p_sample_loop
            )

            out = sample_fn(
                model,
                device=device,
                clip_denoised=False,
                denoised_fn=None,
                model_kwargs=cond,
                top_p=-1,

            )
            results.append(out['sample'])
    results = torch.cat(results,dim=0)
    x_sum = results[:,-amplify_len:,:]
    x_original = results[:,:results.shape[1]-amplify_len,:]
    logits = torch.softmax(torch.bmm(x_sum,x_original.permute(0,2,1)),dim=-1)
    simp_trajs = torch.topk(logits,k=amplify_len,dim=-1)
    simp_trajs_candidate = simp_trajs.indices
    simp_trajs_idx=[]
    for i in range(simp_trajs_candidate.size(0)):
        simp_traj_idx=[]
        simp_traj_candidate = simp_trajs_candidate[i]
        for j in range(amplify_len):
            j_candidate_list = simp_traj_candidate[j].cpu().numpy()
            for j_candidate in j_candidate_list:
                if j_candidate not in simp_traj_idx:
                     simp_traj_idx.append(j_candidate)
                     break
        simp_trajs_idx.append(simp_traj_idx)

    return simp_trajs_idx







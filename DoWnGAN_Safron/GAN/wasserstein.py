from DoWnGAN.GAN.stage import StageData
import DoWnGAN.config.hyperparams as hp
from DoWnGAN.config import config
from DoWnGAN.GAN.losses import content_loss, crps_empirical
from DoWnGAN.mlflow_tools.gen_grid_plots import gen_grid_images
from DoWnGAN.mlflow_tools.mlflow_epoch import post_epoch_metric_mean, gen_batch_and_log_metrics, initialize_metric_dicts, log_network_models
import torch
from torch.autograd import grad as torch_grad

import mlflow
highres_in = True
freq_sep = True
torch.autograd.set_detect_anomaly(True)


class WassersteinGAN:
    """Implements Wasserstein GAN with gradient penalty and 
    frequency separation"""

    def __init__(self, G, C, G_optimizer, C_optimizer) -> None:
        self.G = G
        self.C = C
        self.G_optimizer = G_optimizer
        self.C_optimizer = C_optimizer
        self.num_steps = 0
        
    def _critic_train_iteration(self, coarse_list, invar_list, coarse, fine, invariant):
        """
        Performs one iteration of the critic training.
        Args:
            coarse (torch.Tensor): The coarse input.
            fine (torch.Tensor): The fine input.
        """
        fake_out = [self.G(c.to(config.device),i.to(config.device)) for (c, i) in zip(coarse_list,invar_list)]    
        fake = self._combine_tile(fake_out) ##stich together

        c_real = self.C(fine,invariant,coarse) ##make prediction for real image
        c_fake = self.C(fake,invariant,coarse) ##make prediction for generated image

        gradient_penalty = hp.gp_lambda * self._gp(fine, fake, self.C, coarse, invariant) 
            
        # Zero the gradients
        self.C_optimizer.zero_grad()
        c_real_mean = torch.mean(c_real)
        c_fake_mean = torch.mean(c_fake)

        critic_loss = c_fake_mean - c_real_mean + gradient_penalty
        critic_loss.backward(retain_graph = True)
        # Update the critic
        self.C_optimizer.step()

    def _comb_lr(self, a,b):
        a_overlap = a[:,:,:,-16:]
        b_overlap = b[:,:,:,0:16]
        avg_overlap = (a_overlap + b_overlap)/2
        comb = torch.cat([a[:,:,:,:-16],avg_overlap,b[:,:,:,16:]], dim = 3)
        return comb
    
    def _comb_tb(self, top, bottom):
        t_overlap = top[:,:,-16:,:]
        b_overlap = bottom[:,:,0:16,:]
        tb_avg = (t_overlap + b_overlap)/2
        res = torch.cat([top[:,:,:-16,:], tb_avg, bottom[:,:,16:,:]], dim = 2)
        return(res)
    
    def _combine_tile(self, g_list):
        top = self._comb_lr(g_list[0],g_list[1])
        bottom = self._comb_lr(g_list[2],g_list[3])
        res = self._comb_tb(top,bottom)
        return(res)


    def _generator_train_iteration(self, coarse_list, invar_list, coarse, fine, invariant, iteration):
        """
        Performs one iteration of the generator training.
        Args:
            coarse (torch.Tensor): The coarse input.
            fine (torch.Tensor): The fine input.
        """
        self.G_optimizer.zero_grad()
        
        fake_out = [self.G(c.to(config.device),i.to(config.device)) for (c, i) in zip(coarse_list,invar_list)]    
        fake = self._combine_tile(fake_out) ##stich together

        fake_low = hp.low(hp.rf(fake))
        real_low = hp.low(hp.rf(fine))
        c_fake = self.C(fake,invariant,coarse)
        cont_loss = content_loss(fake_low, real_low, device=config.device)

        g_loss = -torch.mean(c_fake) * hp.gamma + hp.content_lambda * cont_loss  
        g_loss.backward()

        # Update the generator
        self.G_optimizer.step()



    def _gp(self, real, fake, critic, coarse, invariant):
        current_batch_size = real.size(0)

        # Calculate interpolation
        alpha = torch.rand(current_batch_size, 1, 1, 1, requires_grad=True, device=config.device)
        alpha = alpha.expand_as(real)

        interpolated = alpha * real.data + (1 - alpha) * fake.data

        # Calculate probability of interpolated examples
        critic_interpolated = critic(interpolated, invariant, coarse)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(critic_interpolated.size(), device=config.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(hp.batch_size, -1).to(config.device)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return torch.mean((gradients_norm - 1) ** 2)


    def _train_epoch(self, dataloader, epoch):
        """
        Performs one epoch of training.
        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to use.
            epoch (int): The epoch number.
        """
        print(80*"=")
        ##print("Wasserstein GAN")
        train_metrics = initialize_metric_dicts({})
        test_metrics = initialize_metric_dicts({})

        for i,data in enumerate(dataloader):
            self._critic_train_iteration(data[0], data[1], data[2], data[3], data[4])

            if self.num_steps % hp.critic_iterations == 0:
                self._generator_train_iteration(data[0], data[1], data[2], data[3], data[4], epoch)

            # Track train set metrics
            train_metrics = gen_batch_and_log_metrics( ##fix this
                self.G,
                self.C,
                data[0],
                data[1],
                data[2],
                data[3],
                data[4],
                train_metrics,
            )
            self.num_steps += 1

        if epoch % 10 == 0:
            # Take mean of all batches and log to file
            with torch.no_grad():
                post_epoch_metric_mean(train_metrics, "train")
    
                # Generate plots from training set
                c_list, i_list, coarse, fine, invariant = next(iter(dataloader))
                gen_grid_images(self.G, c_list, i_list, coarse, fine, epoch, "train")
    
                # Log the models to mlflow pytorch models
                print(f"Artifact URI: {mlflow.get_artifact_uri()}")
                log_network_models(self.C, self.G, epoch)

    def train(self, dataloader):
        """
        Trains the model.
        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to use.
        """
        self.num_steps = 0
        for epoch in range(hp.epochs):
            self._train_epoch(dataloader, epoch)



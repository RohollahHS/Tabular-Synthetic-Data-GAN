import torch


def load_checkpoint(checkpoint_address,
                    model_name,
                    G,
                    D,
                    g_optimizer,
                    d_optimizer,
                    device):
    
    checkpoint = torch.load(f'{checkpoint_address}/last_{model_name}.pth', device)

    G.load_state_dict(checkpoint['generator_state_dict'])
    D.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['optimizer_g_state_dict'])
    d_optimizer.load_state_dict(checkpoint['optimizer_d_state_dict'])
    curr_epoch = checkpoint['epoch']
    d_losses = checkpoint['d_losses']
    g_losses = checkpoint['g_losses']
    real_scores = checkpoint['real_scores']
    fake_scores = checkpoint['fake_scores']

    print('\nChekcpoint Loaded Successfully!\n')
    
    return G, D, g_optimizer, d_optimizer, curr_epoch, d_losses, g_losses, real_scores, fake_scores


def save_model(
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    epoch,
    save_dir,
    model_name,
    d_losses,
    g_losses,
    real_scores,
    fake_scores,
):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    torch.save(
        {
            "epoch": epoch + 1,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "d_losses": d_losses,
            "g_losses": g_losses,
            "real_scores": real_scores,
            "fake_scores": fake_scores,
        },
        f"{save_dir}/last_{model_name}.pth",
    )

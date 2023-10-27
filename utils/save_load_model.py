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

    print('\nChekcpoint Loaded Successfully!\n')
    
    return G, D, g_optimizer, d_optimizer, curr_epoch


def save_model(
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    epoch,
    generator_loss,
    discriminator_loss,
    save_dir,
    model_name
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
            "generator_loss": generator_loss,
            "discriminator_loss": discriminator_loss,
        },
        f"{save_dir}/last_{model_name}.pth",
    )

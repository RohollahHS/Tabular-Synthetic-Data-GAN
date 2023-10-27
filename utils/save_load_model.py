from torch import load, save
from numpy import zeros

def load_checkpoint(args,
                    G,
                    D,
                    g_optimizer,
                    d_optimizer):
    
    checkpoint = load(f'{args.output_path}/last_{args.model_name}.pth', args.device)

    G.load_state_dict(checkpoint['generator_state_dict'])
    D.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['optimizer_g_state_dict'])
    d_optimizer.load_state_dict(checkpoint['optimizer_d_state_dict'])
    curr_epoch = checkpoint['epoch']

    d_losses = zeros(args.n_epochs)
    g_losses = zeros(args.n_epochs)
    real_scores = zeros(args.n_epochs)
    fake_scores = zeros(args.n_epochs)

    d_losses[:curr_epoch] = checkpoint['d_losses'][:curr_epoch]
    g_losses[:curr_epoch] = checkpoint['g_losses'][:curr_epoch]
    real_scores[:curr_epoch] = checkpoint['real_scores'][:curr_epoch]
    fake_scores[:curr_epoch] = checkpoint['fake_scores'][:curr_epoch]

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
    save(
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
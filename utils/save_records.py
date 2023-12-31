from datetime import datetime
import matplotlib.pyplot as plt
import pylab
import numpy as np
import os


def save_loss_records(dir_path, file_name, loss=None, epoch=None, model_name=None):
    text_file = open(f'{dir_path}/{file_name}.txt', 'a')

    if (model_name == None) and (loss != None):
        text_file.write(f'{epoch}-{loss:.4f} | ')
    elif (model_name != None) and (loss == None):
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        text_file.write(f'\n{model_name} - {now}\n')
    
    text_file.close()


def save_plots(d_losses, g_losses, fake_scores, real_scores, 
               epoch, save_dir, model_name):
    np.save(os.path.join(save_dir, f'{model_name}_d_losses.npy'), d_losses)
    np.save(os.path.join(save_dir, f'{model_name}_g_losses.npy'), g_losses)
    np.save(os.path.join(save_dir, f'{model_name}_fake_scores.npy'), fake_scores)
    np.save(os.path.join(save_dir, f'{model_name}_real_scores.npy'), real_scores)
    
    plt.figure()
    pylab.xlim(0, epoch + 1)
    plt.plot(range(1, epoch + 2), d_losses[:epoch+1], label='d loss')
    plt.plot(range(1, epoch + 2), g_losses[:epoch+1], label='g loss')    
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{model_name}_loss.pdf'))
    plt.close()

    plt.figure()
    pylab.xlim(0, epoch + 1)
    pylab.ylim(0, 1)
    plt.plot(range(1, epoch + 2), fake_scores[:epoch+1], label='fake score')
    plt.plot(range(1, epoch + 2), real_scores[:epoch+1], label='real score')    
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{model_name}_accuracy.pdf'))
    plt.close()
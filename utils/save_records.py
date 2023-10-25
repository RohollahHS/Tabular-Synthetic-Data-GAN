from datetime import datetime


def save_loss_records(dir_path, file_name, loss=None, epoch=None, model_name=None):
    text_file = open(f'{dir_path}/{file_name}.txt', 'a')

    if (model_name == None) and (loss != None):
        text_file.write(f'{epoch}-{loss:.4f} | ')
    elif (model_name != None) and (loss == None):
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        text_file.write(f'\n{model_name} - {now}\n')
    
    text_file.close()

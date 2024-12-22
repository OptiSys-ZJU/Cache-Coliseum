from model.parrot import utils

if __name__ == '__main__':
    batch_size = 32
    update_freq = 10000
    collection_multiplier = 5



    train_data = [1, 2, 3, 4, 5, 6]

    for batch_num, batch in enumerate(utils.as_batches([train_data], 2, 3)):
        print(batch_num, batch)
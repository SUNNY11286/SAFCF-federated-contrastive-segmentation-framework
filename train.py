import tensorflow as tf
import segmentation_models_3D as sm
from tqdm import tqdm
from utils import dice_coef, precision, sensitivity, specificity, sum_scaled_weights, contrastive_loss

class Trainer:
    def __init__(self, model, optimizer, loss_fn, metrics, batch_size, epochs, num_clients, num_comm):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_clients = num_clients
        self.num_comm = num_comm

    def train(self, clients, server, test_data_gen):
        dice_coee = []
        contrastv_loss = []
        cl_pe_val = []
        dc_pe_val = []

        for t in range(self.num_comm):
            print(f'\nCommunication round: {t+1}\n')
            local_weight_list = list()
            for epoch in range(self.epochs):
                print(f'\nEpoch {epoch+1}')
                client_loss = []
                for i, client in enumerate(clients):
                    print(f'Training Client {i+1}')
                    model = client['model']
                    optimizer = client['optimizer']

                    for batch_idx, (image, target) in enumerate(tqdm(client['original_data'])):
                        pro2, _ = server['model'](image)
                        with tf.GradientTape() as tape:
                            pro1, output = model(image)
                            pro3 = tf.zeros_like(pro1) if 'past_model' not in client else client['past_model'](image)[0]
                            cl = contrastive_loss(pro3, pro1, pro2, 0.5)
                            loss2 = cl
                            loss1 = self.loss_fn(target, output)
                            dc = dice_coef(target, output)
                            loss = loss1 + loss2
                            client_loss.append(loss)
                            dice_coee.append(dc)
                            contrastv_loss.append(cl)

                        grads = tape.gradient(loss, model.trainable_weights)
                        optimizer.apply_gradients(zip(grads, model.trainable_variables))

                    dc_val_pe = sum(dice_coee) / len(dice_coee)
                    cl_val_pe = sum(contrastv_loss) / len(contrastv_loss)
                    print(f'epoch dice coeff mean: {dc_val_pe}')
                    print(f'epoch contrastive loss mean: {cl_val_pe}')
                    dc_pe_val.append(dc_val_pe)
                    cl_pe_val.append(cl_val_pe)
                    dice_coee.clear()
                    contrastv_loss.clear()

                    client['model'] = model
                    local_weight_list.append(client['model'].get_weights())

                average_weight = sum_scaled_weights(local_weight_list, client["length_ratio"])
                server['model'].set_weights(average_weight)
                local_weight_list.clear()

        self.evaluate(server['model'], test_data_gen)

    def evaluate(self, model, test_data_gen):
        dice = []
        pre = []
        batch_loss = []
        se = []
        spe = []
        io = []

        for batch_idx, (images, masks) in enumerate(tqdm(test_data_gen)):
            _, logits = model(images)
            loss = self.loss_fn(masks, logits)
            batch_loss.append(loss)
            dice.append(dice_coef(masks, logits))
            pre.append(precision(masks, logits))
            se.append(sensitivity(masks, logits))
            spe.append(specificity(masks, logits))
            io.append(sm.metrics.IOUScore(threshold=0.5)(masks, logits))

        print(f'Test results: Loss: {np.mean(batch_loss)}, Dice Coeff: {np.mean(dice)}, Precision: {np.mean(pre)}, Sensitivity: {np.mean(se)}, Specificity: {np.mean(spe)}, IOU: {np.mean(io)}')
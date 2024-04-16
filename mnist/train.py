import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from dataset import Dataset
from model import UCCModel
from logging import getLogger
import os
import sys

print(os.getcwd())
sys.path.append("../")

def grad(model, inputs, labels):
    with tf.GradientTape() as tape:
        if model.alpha ==1:
            output = model(inputs)
            ucc_loss = model.loss(
                output=output,
                labels=labels,
            )
            ae_loss = 0
        else:
            output, reconstruction = model(inputs)
            ucc_loss ,ae_loss = model.loss(
                output=output, 
                labels=labels, 
                inputs=inputs, 
                reconstruction=reconstruction)
            # weighted_ucc_loss = model.alpha*ucc_loss
            # weighted_ae_loss = (1-model.alpha)*ae_loss
        loss_value =  model.alpha*ucc_loss + (1-model.alpha)*ae_loss
        accuracy = tf.keras.metrics.Accuracy()
        output = tf.argmax(output, axis=1)
        labels = tf.argmax(labels, axis=1)
        acc = accuracy(output, labels)
        return {'ucc_acc': acc, 'ucc_loss': ucc_loss, 'ae_loss':ae_loss, 'weighted_loss':loss_value}, tape.gradient(loss_value, model.trainable_variables)                                          

def eval(model, dataset):
    accuracy = tf.keras.metrics.Accuracy()
    _, [labels, inputs] = dataset.next_batch_val()
    accuracy = tf.keras.metrics.Accuracy()
    if model.alpha==1:
        output = model(inputs)
        ucc_loss =model.loss(
            output=output,
            labels=labels
        )
        ae_loss = 0
    else:
        output, reconstruction = model(inputs)
        ucc_loss ,ae_loss = model.loss(
            output=output, 
            labels=labels, 
            inputs=inputs, 
            reconstruction=reconstruction)
    loss_value =  model.alpha*ucc_loss + (1-model.alpha)*ae_loss
    output = tf.argmax(output, axis=1)
    labels = tf.argmax(labels, axis=1)
    acc = accuracy(output, labels)
    return {'ucc_acc': acc, 'ucc_loss': ucc_loss, 'ae_loss':ae_loss, 'weighted_loss':loss_value}



def train(model, optimizer, dataset, args):
    save_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    train_summary_writer = tf.summary.create_file_writer(os.path.join(save_path, "train_logs"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(save_path, "val_logs"))
    logger = getLogger()
    num_steps = args.train_num_steps
    physical_devices = tf.config.list_physical_devices('GPU')
    best_eval_acc = 0
    if physical_devices:
        with tf.device('/device:GPU:0'):
            for i in tqdm(range(num_steps)):

                _, [labels, inputs] = dataset.next_batch_train()
                loss_dict, grads = grad(model, inputs, labels)
                with train_summary_writer.as_default():
                    for key in loss_dict.keys():      
                        tf.summary.scalar(key, loss_dict[key], step=i)
                    train_summary_writer.flush()
                    
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                if i%10 == 0:
                    eval_loss_dict = eval(model, dataset)
                    with val_summary_writer.as_default():
                        for key in eval_loss_dict.keys():      
                            tf.summary.scalar(key, loss_dict[key], step=i)
                        val_summary_writer.flush()
                    logger.info(f"Step {i+1}, training: ucc_acc={loss_dict['ucc_acc']} weighted_loss={loss_dict['weighted_loss']} ucc_loss={loss_dict['ucc_loss']} ae_loss={loss_dict['ae_loss']} eval: ucc_acc={eval_loss_dict['ucc_acc']} weighted_loss={eval_loss_dict['weighted_loss']} ucc_loss={eval_loss_dict['ucc_loss']} ae_loss={eval_loss_dict['ae_loss']}")
                    acc = eval_loss_dict['ucc_acc']
                    if acc>= best_eval_acc:
                        best_eval_acc = acc
                        model.save_weights(os.path.join(save_path,"model_weights.h5"), save_format="h5")

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg):

    args = cfg.args
    dataset = Dataset(
        num_instances=args.num_instances,
        num_samples_per_class=args.num_samples_per_class,
        digit_arr= [x for x in range(10)],
        ucc_start=args.ucc_start,
        ucc_end=args.ucc_end
        )
    model = UCCModel(cfg)
    model.build([])
    print(model.summary())
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    train(model, optimizer, dataset, args)
    


if __name__ == "__main__":
    main()
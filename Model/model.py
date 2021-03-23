import json
import tensorflow as tf

def get_config(path):
    with open(path, 'r') as f:
        json_data = json.load(f)
    return json_data

def connect_TPU(BatchSizeTpu):
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    global_batch_size = BatchSizeTpu * strategy.num_replicas_in_sync

    return tpu, strategy, global_batch_size


# config: {'MaxLength': , 'BatchSizeTpu': , 'BatchSize': , 'Epochs': , 'LearningRate': ,'EvaluateEvery}

class TrainTpu(tf.keras.Model):
    def __init__(self, dataset, model, config, Tpu, Strategy, GlobalBatchSize):
        super(TrainTpu, self).__init__()
        self.dataset = dataset
        self.Strategy = Strategy
        self.MLM_Bert = model
        self.GlobalBatchSize = GlobalBatchSize
        self.Epoch = config['Epochs']
        self.evaluate_every = config['EvaluateEvery']
        self.LearningRate = config['LearningRate']
        self.Optimizer = tf.keras.optimizers.Adam(learning_rate=self.LearningRate)

    def __call__(self):  # Train Distributed MLM
        step = 0
        self.MLM_loss_func, self.MLM_train_loss = self.define_mlm_loss_and_metrics()
        for tensor in self.dataset:
            self.distributed_MLM_train_step(tensor)
            step += 1

            if step % self.evaluate_every == 0:
                train_metric = self.MLM_train_loss.result().numpy()
                print("Step %d, train loss: %.2f" % (step, train_metric))
                self.MLM_train_loss.reset_states()
            if step == self.Epoch:
                break

    @tf.function
    def distributed_MLM_train_step(self, data):
        self.Strategy.run(self.MLM_train_step, args=(data,))

    @tf.function
    def MLM_train_step(self, data):
        Feature, Label = data  # dataloader의 인풋에 맞추어 바꿔야함

        with tf.GradientTape() as tape:
            prediction = self.MLM_Bert(Feature, training=True)[0]
            loss = self.MLM_loss_func(Label, prediction)

        gradients = tape.gradient(loss, self.MLM_Bert.trainable_variables)
        self.Optimizer.apply_gradients(zip(gradients, self.MLM_Bert.trainable_variables))
        self.MLM_train_loss.update_state(loss)

    def define_mlm_loss_and_metrics(self):
        with self.Strategy.scope():
            mlm_loss_object = self.masked_sparse_categorical_crossentropy

            def MLM_loss(labels, predictions):
                per_example_loss = mlm_loss_object(labels, predictions)
                loss = tf.nn.compute_average_loss(
                    per_example_loss, global_batch_size=self.GlobalBatchSize)
                return loss

            train_mlm_loss_metric = tf.keras.metrics.Mean()

        return MLM_loss, train_mlm_loss_metric

    def masked_sparse_categorical_crossentropy(self,y_true, y_pred):
        y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
        y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))

        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked,
                                                               y_pred_masked,
                                                               from_logits=True)
        return loss

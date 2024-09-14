import os
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import pandas as pd
import pickle
import json
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
from package.constants import Feature_Type, Target, Temporal_Feature, Model_Architecture
from typing import List, Dict
import concurrent.futures

# MirroredStrategy for multi-GPU parallelism
strategy = tf.distribute.MirroredStrategy()

class MultiTaskLossLayer(layers.Layer):
    def __init__(self, is_regression, **kwargs):
        super(MultiTaskLossLayer, self).__init__(**kwargs)
        self.is_regression = tf.convert_to_tensor(is_regression, dtype=tf.float32)
        self.n_tasks = len(is_regression)
        self.log_vars = self.add_weight(name='log_vars', shape=(self.n_tasks,), initializer='zeros', trainable=True)
        self.logcosh_loss = tf.keras.losses.LogCosh(reduction=tf.keras.losses.Reduction.NONE)
        self.cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_trues, y_preds):
        task_losses = []
        for i in range(self.n_tasks):
            y_true = y_trues[i]
            y_pred = y_preds[i]
            if tf.shape(y_pred)[-1] == 1:  # Regression task
                task_loss = self.logcosh_loss(y_true, y_pred)
            else:  # Classification task
                task_loss = self.cross_entropy_loss(y_true, y_pred)
            task_losses.append(task_loss)
        # Stack the losses into one tensor
        losses = tf.stack(task_losses, axis=-1)

        # Standard deviation and coefficients
        stds = tf.sqrt(tf.exp(self.log_vars))
        coeffs = 1 / ((self.is_regression + 1) * stds**2)
        multi_task_losses = coeffs * losses + tf.math.log(stds)
        return tf.reduce_mean(multi_task_losses)  # Or sum based on preference

    def get_config(self):
        config = super(MultiTaskLossLayer, self).get_config()
        config.update({
            "is_regression": self.is_regression.numpy(),
        })
        return config


class ModelWrapper():
    def __init__(self,
                dataset_name: str,
                case_ids: pd.DataFrame,
                input_columns: List[str],
                target_columns: dict[str, Target],
                additional_columns: Dict[Feature_Type, List[str]],
                word_dicts: dict[str, Dict[str, int]],
                max_case_length: int,
                feature_type_dict: Dict[Feature_Type, List[str]],
                temporal_features: Dict[Temporal_Feature, bool],
                model_architecture: type[Model_Architecture],
                sorting: bool,
                masking: bool = True
                ):
        
        # constants
        self.embed_dim: int = 36
        self.num_heads: int = 4
        self.ff_dim: int = 64
        
        self.dataset_name = dataset_name
        self.case_ids = case_ids
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.additional_columns = additional_columns
        self.word_dicts = word_dicts
        self.max_case_length = max_case_length
        self.feature_type_dict = feature_type_dict
        self.temporal_features = temporal_features
        self.model_architecture = model_architecture
        self.sorting = sorting
        self.masking = masking
        
        self.models: List[Model] = None
        self.history = None
        
        

    class TransformerBlock(layers.Layer):
        def __init__(self, name, model_wrapper, embed_dim, num_heads, ff_dim, rate=0.1):
            super(ModelWrapper.TransformerBlock, self).__init__(name=name)
            self.model_wrapper = model_wrapper
            self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = tf.keras.Sequential(
                [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
            )
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = layers.Dropout(rate)
            self.dropout2 = layers.Dropout(rate)
            self.num_heads = num_heads
            self.supports_masking = True

        def call(self, inputs, training=None, mask=None):
            if self.model_wrapper.masking:
                    # Expand dims for num_head
                    mask = tf.expand_dims(mask, axis=1)  # Shape becomes (batch_size, 1, max_case_length)
            
            # Apply multi-head attention with masking
            attn_output = self.att(inputs, inputs, attention_mask=mask)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            
            # Apply feed-forward network
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)


    class TokenEmbedding(layers.Layer):
        """
        Token Embedding Layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality of the embedding space.
            mask_padding (bool): Whether to mask padding tokens (zero index).
        """
        def __init__(self, model_wrapper, vocab_size, embed_dim, name):
            super(ModelWrapper.TokenEmbedding, self).__init__()
            self.model_wrapper = model_wrapper
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name=name, mask_zero=model_wrapper.masking)
            self.supports_masking = True
            
        def call(self, inputs, training=None):
            return self.token_emb(inputs)
        
        
    class PositionEmbedding(layers.Layer):
        """
        Position Embedding Layer.

        Args:
            maxlen (int): Maximum length of the sequences.
            embed_dim (int): Dimensionality of the embedding space.
        """
        def __init__(self, model_wrapper, maxlen, embed_dim, name):
            super(ModelWrapper.PositionEmbedding, self).__init__()
            self.model_wrapper = model_wrapper
            self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim, name=name)
            self.supports_masking = True
            
        def call(self, inputs, training=None, mask=None):
            """
            Forward pass for position embedding.

            Args:
                x (tf.Tensor): Input tensor containing token indices or token embeddings.

            Returns:
                tf.Tensor: Output tensor with position embeddings.
            """
            
            maxlen = tf.shape(inputs)[1]  # Length of the input sequence
            positions = tf.range(start=0, limit=maxlen, delta=1)  # Generate position indices
            positions = self.pos_emb(positions)  # Get position embeddings
            positions = tf.expand_dims(positions, 0)  # Add a batch dimension (1, maxlen, embed_dim)
            return inputs + positions  # Add position embeddings to the input tensor
        
        
    # generates a mask for numerical inputs
    class NumericalMaskGeneration(layers.Layer):
        def __init__(self, model_wrapper):
            super(ModelWrapper.NumericalMaskGeneration, self).__init__()
            self.model_wrapper = model_wrapper
            self.supports_masking = True
            self.mask = None
            
        def call(self, inputs):
            self.mask = tf.not_equal(inputs, -1)
            casted_mask = tf.cast(self.mask, tf.float32)
            inputs = layers.Multiply()([inputs, casted_mask])
            
            return inputs
        
        def compute_mask(self, inputs, mask=None):
            if self.model_wrapper.masking:
                mask = self.mask
            return mask


    class MaskedGlobalAveragePooling1D(tf.keras.layers.Layer):
        def __init__(self, model_wrapper):
            super(ModelWrapper.MaskedGlobalAveragePooling1D, self).__init__()
            self.model_wrapper = model_wrapper
            
        def call(self, inputs, training=None, mask=None):
            if self.model_wrapper.masking:
                # Expand mask to match input shape
                mask = tf.expand_dims(mask, axis=-1)
                mask = tf.cast(mask, dtype=inputs.dtype)
                
                # Apply the mask (zero out masked values)
                inputs = inputs * mask
                
                # Compute the sum of the valid (non-masked) inputs
                sum_inputs = tf.reduce_sum(inputs, axis=1)
                
                # Count valid (non-masked) elements
                valid_count = tf.reduce_sum(mask, axis=1)
                
                # Avoid division by zero by ensuring valid_count is at least 1
                valid_count = tf.maximum(valid_count, 1)
                
                # Perform the average pooling
                return sum_inputs / valid_count
            else:
                # Fallback to standard pooling if no mask is provided
                return tf.reduce_mean(inputs, axis=1)
        
        def compute_mask(self, inputs, mask=None):
            return None
        
        

    # TODO: vocab_size to list of vocab_sizesnum_classes_list
    def get_model(self, input_columns: List[str], target_columns: Dict[str, Target], word_dicts: Dict[str, Dict[str, int]], max_case_length: int,
                  feature_type_dict: Dict[Feature_Type, List[str]], temporal_features: Dict[Temporal_Feature, bool],
                  model_architecture: Model_Architecture,
                  embed_dim=36, num_heads=4, ff_dim=64, num_layers=1):
        """
        Constructs the next categorical prediction model using a transformer architecture.
        
        Args:
            max_case_length (int): Maximum length of the sequences (cases).
            embed_dim (int): Dimensionality of the embeddings. Defaults to 36.
            num_heads (int): Number of attention heads. Defaults to 4.
            ff_dim (int): Dimensionality of the feed-forward layer. Defaults to 64.
            num_layers (int): Number of transformer blocks. Defaults to 1.
        
        Returns:
            tf.keras.Model: Compiled transformer model for next categorical prediction.
        """
        
        def prepare_categorical_input(feature: str):
            # generate input layer for categorical feature
            categorical_input = layers.Input(shape=(max_case_length,), name=f"input_{feature}")
            # do token embedding for categorical feature
            categorical_emb = ModelWrapper.TokenEmbedding(model_wrapper = self,
                                                            vocab_size = len(word_dicts[feature]["x_word_dict"]),
                                                            embed_dim = embed_dim,
                                                            name = f"{feature}_token-embeddings"
                                                            )(categorical_input)
            return categorical_input, [categorical_emb]
        
        
        def prepare_temporal_input(feature):
            temporal_inputs = []
            # Input Layer for temporal feature
            temporal_input = layers.Input(shape=(max_case_length,), name=f"input_{feature}")
            # append temporal feature to temporal inputs
            temporal_inputs.append(temporal_input)

            # if day_of_week is used as additional temp feature
            if temporal_features[Temporal_Feature.DAY_OF_WEEK]:
                temporal_input_day_of_week = layers.Input(shape=(max_case_length,), name=f"input_{feature}_{Temporal_Feature.DAY_OF_WEEK.value}")
                temporal_inputs.append(temporal_input_day_of_week)
            # if hour_of_day is used as additional temp feature
            if temporal_features[Temporal_Feature.HOUR_OF_DAY]:
                temporal_input_hour_of_day = layers.Input(shape=(max_case_length,), name=f"input_{feature}_{Temporal_Feature.HOUR_OF_DAY.value}")
                temporal_inputs.append(temporal_input_hour_of_day)
                
            return temporal_inputs
        
        
        def prepare_inputs():
            inputs_layers, feature_tensors = [], []
            temp_feature_exists = False
            
            for feature in input_columns:
                for feature_type, feature_lst in feature_type_dict.items():
                    if feature in feature_lst:
                        # feature is categorical
                        if feature_type is Feature_Type.CATEGORICAL:
                            categorical_input, categorical_embs = prepare_categorical_input(feature)
                            # append input layer to inputs
                            inputs_layers.append(categorical_input)
                            # append categorical token embedding to feature_tensors
                            feature_tensors.append(categorical_embs)
                            
                        # feature is temporal
                        elif feature_type is Feature_Type.TIMESTAMP:
                            temp_feature_exists = True
                            temporal_inputs = prepare_temporal_input(feature)
                            # extend inputs_layers with temporal inputs
                            inputs_layers.extend(temporal_inputs)
                            # expand dim from (None, max_case_length) to (None, max_case_length, 1) for compatability with position embedding layer
                            temporal_inputs = [tf.expand_dims(x, axis=-1) for x in temporal_inputs]
                            # Generate mask
                            if self.masking:
                                temporal_inputs = [ModelWrapper.NumericalMaskGeneration(self)(x) for x in temporal_inputs]
                            # append temporal_inputs to feature_tensors
                            feature_tensors.append(temporal_inputs)
            return inputs_layers, feature_tensors
                            
        ############################################################################################

        print("Creating model...")


        if self.masking:
            print("Masking active.")
        else:
            print("Masking not active.")

            
        # prepare inputs
        inputs_layers, feature_tensors = prepare_inputs()
        
        # common embeddings and transformers for all features
        if model_architecture is Model_Architecture.COMMON_POSEMBS_TRANSF:
            # flatten feature_tensors
            feature_tensors = [item for sublist in feature_tensors for item in sublist]
            # concat feature layers
            x = layers.Concatenate()(feature_tensors)
            # add position embedding to the concatenated layers
            x = ModelWrapper.PositionEmbedding(model_wrapper = self, maxlen=max_case_length, embed_dim=x.shape[-1],
                                               name="position-embedding_common")(x)
            
        # seperate positional embeddings and common transformer for all features
        elif model_architecture is Model_Architecture.SEPERATE_POSEMBS:
            feature_embs = []
            for idx, tensors_of_feature in enumerate(feature_tensors):
                # concat tensors of each feature
                x = layers.Concatenate()(tensors_of_feature)
                # add position embedding
                x = ModelWrapper.PositionEmbedding(model_wrapper = self, maxlen=max_case_length, embed_dim=x.shape[-1],
                                                   name=f"position-embedding_feature_{idx}")(x)
                # append to list of feature_embs
                feature_embs.append(x)
            # concat feature embs
            x = layers.Concatenate()(feature_embs)
            
        # seperate positional embeddings and transformers for each feature
        elif model_architecture is Model_Architecture.SEPERATE_TRANSF:
            feature_transf = []
            for idx, tensors_of_feature in enumerate(feature_tensors):
                # concat tensors of each feature
                x = layers.Concatenate()(tensors_of_feature)
                # add position embeddings
                x = ModelWrapper.PositionEmbedding(model_wrapper = self, maxlen=max_case_length, embed_dim=x.shape[-1],
                                                   name=f"position-embedding_feature_{idx}")(x)
                # feed into transformer block
                x = ModelWrapper.TransformerBlock(f"Seperate_Transformer_feature_{idx}", self, x.shape[-1], num_heads, ff_dim)(x)
                # append to list of feature transformer tensors
                feature_transf.append(x)
            # concat feature transformer tensors
            x = layers.Concatenate()(feature_transf)
            
            
        # seperate positional embeddings and transformers for each feature
        elif model_architecture is Model_Architecture.TIME_TARGET:
            feature_transf = []
            for idx, tensors_of_feature in enumerate(feature_tensors):
                transformers_of_feature = []
                for idx_sub, x in enumerate(tensors_of_feature):
                    # reduce mask dim
                    x = layers.Concatenate()([x])
                    # add position embeddings
                    x = ModelWrapper.PositionEmbedding(model_wrapper = self, maxlen=max_case_length, embed_dim=x.shape[-1], name=f"position-embedding_feature_{idx}")(x)
                    # feed into transformer block
                    x = ModelWrapper.TransformerBlock(f"Transformer_feature_{idx}_sub_{idx_sub}", self, x.shape[-1], num_heads, ff_dim)(x)
                    # append to list of transformers for each feature
                    transformers_of_feature.append(x)
                # if feature has multiple transformers, concat and apply another transformer
                if len(transformers_of_feature) > 1:
                    x = layers.Concatenate()(transformers_of_feature)
                    x = ModelWrapper.TransformerBlock(f"Transformer_feature_{idx}_concat", self, x.shape[-1], num_heads, ff_dim)(x)
                # append to list of feature transformer tensors
                feature_transf.append(x)
            # concat feature transformer tensors
            x = layers.Concatenate()(feature_transf)
            
        
        # Stacking multiple transformer blocks
        for idx, _ in enumerate(range(num_layers)):
            x = ModelWrapper.TransformerBlock(f"Stacked_Transformer_{idx}", self, x.shape[-1], num_heads, ff_dim)(x)

        x = ModelWrapper.MaskedGlobalAveragePooling1D(self)(x)
        
        # Fully connected layers
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        
        # Output layers for categorical features
        outputs = []
        for feature, target in target_columns.items():
            for feature_type, feature_lst in feature_type_dict.items():
                if feature in feature_lst:
                    if feature_type is Feature_Type.CATEGORICAL:
                        if target == Target.NEXT_FEATURE: dict_str = "y_next_word_dict"
                        elif target == Target.LAST_FEATURE: dict_str = "y_last_word_dict"
                        else: raise ValueError("Target type is not known.")
                        output_dim = len(word_dicts[feature][dict_str])
                        outputs.append( layers.Dense(output_dim, activation="softmax", name=f"output_{feature}")(x) )
                    if feature_type is Feature_Type.TIMESTAMP:
                        outputs.append( layers.Dense(1, activation="linear", name=f"output_{feature}")(x) )
        
        
        
        # Model definition
        transformer = Model(inputs=inputs_layers, outputs=outputs, name="next_categorical_transformer")
        
        return transformer



    def train_model(self,
                    train_token_dict_x: dict[str, NDArray[np.float32]],
                    train_token_dict_y: dict[str, NDArray[np.float32]],
                    cross_val: bool,
                    y_scaler,
                    model_epochs: int,
                    batch_size: int = 12,
                    model_learning_rate: float = 0.001,
                    n_splits: int = 5,
                    warmup_epochs: int = 0,
                    initial_lr: float = 1e-5,
                    target_lr: float = 1e-3):

        # Using MirroredStrategy for distributed training on GPUs
        with strategy.scope():
            # tf.compat.v1.disable_eager_execution()
            validation_split = 0.2
            self.models = []
            
            # Ensure no Python generators are used, and the dataset is fully optimized
            def prepare_tf_dataset(x_dict, y_dict, indices=None, use_cache=True):
                # If indices are provided, select only those rows
                if indices is not None:
                    x_dict = {key: x[indices] for key, x in x_dict.items()}
                    y_dict = {key: y[indices] for key, y in y_dict.items()}
                
                # Convert the dictionary to a TensorFlow dataset
                dataset = tf.data.Dataset.from_tensor_slices((x_dict, y_dict))
            
                # Optionally cache the dataset to avoid redundant loading
                if use_cache:
                    dataset = dataset.cache()
            
                # Shuffle and batch the dataset
                dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                return dataset
    
            
            if cross_val:
                print(f"Using {n_splits}-Fold Cross-Validation with Grouping by case_id")
                group_kfold = GroupKFold(n_splits=n_splits)
                val_histories, fold_histories = [], []
                fold = 1
    
                for train_indices, val_indices in group_kfold.split(self.case_ids, groups=self.case_ids):
                    print(f"Training fold {fold}/{n_splits}...")
                    
                    # Build and compile model for the fold
                    model = self._build_and_compile_model(train_token_dict_y, model_learning_rate)
    
                    # Prepare tf.data.Dataset for training and validation splits
                    train_dataset = prepare_tf_dataset(train_token_dict_x, train_token_dict_y, train_indices)
                    val_dataset = prepare_tf_dataset(train_token_dict_x, train_token_dict_y, val_indices)
    
                    # Train the model for the current fold
                    model, history = self._train_single_fold(
                        model,
                        train_dataset,
                        val_dataset,
                        model_epochs,
                        batch_size,
                        fold,
                        warmup_epochs,
                        initial_lr,
                        target_lr
                    )
                    self.models.append(model)
    
                    val_histories.append(history.history['val_loss'])
                    fold_histories.append(history)  # Save the entire history object for each fold
                    fold += 1
    
                # Calculate average validation performance across all folds
                avg_val_loss = np.mean([min(history) for history in val_histories])
                print(f"Average validation loss across {n_splits} folds: {avg_val_loss}")
                
                return self.models, fold_histories  # Optionally return full val_histories if needed
    
            else:
                print("Using regular train-validation split")
                train_indices, val_indices = self._split_train_val(validation_split)
                
                # Build and compile model
                model = self._build_and_compile_model(train_token_dict_y, model_learning_rate)
                
                # Prepare tf.data.Dataset for training and validation splits
                train_dataset = prepare_tf_dataset(train_token_dict_x, train_token_dict_y, train_indices)
                val_dataset = prepare_tf_dataset(train_token_dict_x, train_token_dict_y, val_indices)
    
                # Train the model without cross-validation
                model, history = self._train_single_fold(
                    model,
                    train_dataset,
                    val_dataset,
                    model_epochs,
                    batch_size,
                    None,
                    warmup_epochs,
                    initial_lr,
                    target_lr
                )
                self.models = [model]
    
                return self.models, [history]


    def _build_and_compile_model(self, train_token_dict_y, model_learning_rate: float):
        """
        Build and compile the model, checking whether it is a multi-task or single-task setup.
        """
        model = self.get_model(
            input_columns=self.input_columns,
            target_columns=self.target_columns,
            word_dicts=self.word_dicts,
            max_case_length=self.max_case_length,
            feature_type_dict=self.feature_type_dict,
            temporal_features=self.temporal_features,
            model_architecture=self.model_architecture
        )
        
        if len(self.target_columns) > 1:
            print("Using Multi-Task Learning Setup")
            # Define if output is regression task tasks
            is_regression = []
            for feature in self.target_columns.keys():
                if feature in self.feature_type_dict[Feature_Type.CATEGORICAL]:
                    is_regression.append(False)  # False for classification tasks
                elif feature in self.feature_type_dict[Feature_Type.TIMESTAMP]:
                    is_regression.append(True)  # True for regression tasks
            
            def multi_task_loss_fn(is_regression):
                multi_task_loss_layer = MultiTaskLossLayer(is_regression)
                def loss_fn(y_true, y_pred):
                    y_trues = [y_true]
                    y_preds = [y_pred]
                    return multi_task_loss_layer(y_trues, y_preds)
                return loss_fn
            
            # Define the loss functions and train metrics for each output
            losses = {}
            train_metrics = {}
            for output_key, reg in zip(train_token_dict_y.keys(), is_regression):
                # loss functions
                losses.update({output_key: multi_task_loss_fn([reg])})
                # train metrics
                if reg:
                    train_metrics.update({output_key: tf.keras.metrics.MeanAbsoluteError()})
                else:
                    train_metrics.update({output_key: tf.keras.metrics.SparseCategoricalAccuracy()})
            
            model.compile(optimizer=tf.keras.optimizers.Adam(model_learning_rate), loss=losses, metrics=train_metrics)
        else:
            print("Using Single-Task Learning Setup")
            target_feature = list(self.target_columns.keys())[0]
            feature_type = next(ftype for ftype, flist in self.additional_columns.items() if target_feature in flist)
            
            if feature_type == Feature_Type.CATEGORICAL:
                model.compile(optimizer=tf.keras.optimizers.Adam(model_learning_rate),
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
            elif feature_type == Feature_Type.TIMESTAMP:
                model.compile(optimizer=tf.keras.optimizers.Adam(model_learning_rate),
                                loss=tf.keras.losses.LogCosh(),
                                metrics=[tf.keras.metrics.MeanAbsoluteError()])
        return model

    def _split_train_val(self, validation_split: float):
        """
        Split the case_ids into train and validation indices based on validation_split.
        """
        if self.sorting:
            train_indices, val_indices = sequential_train_val_split(self.case_ids, validation_split)
        else:
            train_indices, val_indices = random_train_val_split(self.case_ids, validation_split)
        return train_indices, val_indices

    def _split_data(self, data_dict: dict[str, NDArray[np.float32]], train_indices: NDArray[np.int32], val_indices: NDArray[np.int32]):
        """
        Split the data dictionary into training and validation sets based on provided indices.
        """
        train_data_split = {key: data[train_indices] for key, data in data_dict.items()}
        val_data_split = {key: data[val_indices] for key, data in data_dict.items()}
        return train_data_split, val_data_split



    def _train_single_fold(self,
                           model,
                           train_dataset,
                           val_dataset,
                           model_epochs: int,
                           batch_size: int,
                           fold: int = None,
                           warmup_epochs: int = 5,  # Specify the number of warmup epochs
                           initial_lr: float = 1e-5,
                           target_lr: float = 1e-3):
        """
        Train the model for a single fold or without cross-validation, with a warmup phase.
        """
        
        def warmup_scheduler(epoch, lr, warmup_epochs=5, initial_lr=1e-5, target_lr=1e-3):
            """
            Warmup scheduler that gradually increases the learning rate from initial_lr to target_lr 
            over warmup_epochs and then keeps the target_lr for the remaining epochs.
            """
            if epoch < warmup_epochs:
                return initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
            else:
                return target_lr
        
        class EarlyStoppingAfterWarmup(tf.keras.callbacks.Callback):
            def __init__(self, warmup_epochs, early_stopping_callback):
                super().__init__()
                self.warmup_epochs = warmup_epochs
                self.early_stopping_callback = early_stopping_callback
                self.original_patience = early_stopping_callback.patience
                self.current_patience = early_stopping_callback.patience
    
            def on_epoch_end(self, epoch, logs=None):
                if epoch < self.warmup_epochs + self.original_patience:
                    # During warmup, disable early stopping by setting high patience
                    self.early_stopping_callback.patience = self.warmup_epochs + self.original_patience
                else:
                    # Restore original patience after warmup
                    self.early_stopping_callback.patience = self.original_patience
        
        
        class EpochSavepoint(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                # History
                self.history_path = os.path.join(train_savepoints_dir, "history.pkl")
                if os.path.isfile(self.history_path):
                    with open(self.history_path, "r") as json_file:
                        self.epoch_history = json.load(json_file)
                else:
                    self.epoch_history = []  # This will store the history for all epochs
                    
                # Current Epoch
                self.epoch_path = os.path.join(train_savepoints_dir, "current_epochs.pkl")
                if os.path.isfile(self.epoch_path):
                    with open(self.epoch_path, 'rb') as file:  # Open in binary read mode
                        self.current_epochs = pickle.load(file)
                else:
                    self.current_epochs = 0

            def on_epoch_end(self, epoch, logs=None):
                # Append the logs (metrics for the current epoch) to the full history
                self.epoch_history.append(logs.copy())
                # Save the accumulated history to a JSON file at the end of each epoch
                with open(self.history_path, 'wb') as file:  # Open in binary read mode
                    pickle.dump(self.epoch_history, file)
        
        
        class BestModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
            def __init__(self, existing_best_model_path, validation_data, save_weights_only=False, *args, **kwargs):
                """
                Args:
                    existing_best_model_path (str): Path to an existing best model (.h5 file or weights file).
                    validation_data (tuple): Validation data for evaluating the model to get the initial monitored value.
                    save_weights_only (bool): Whether to save only model weights (True) or the full model (False).
                """
                super(BestModelCheckpoint, self).__init__(save_weights_only=save_weights_only, *args, **kwargs)
                
                self.existing_best_model_path = existing_best_model_path
                self.validation_data = validation_data
                self.save_weights_only = save_weights_only

                if os.path.isfile(self.existing_best_model_path):
                    # Load the best model from the provided path
                    self._load_existing_best_model()

            def _load_existing_best_model(self):
                if self.save_weights_only:
                    # Load model weights instead of the full model
                    print(f"Loading model weights from: {self.existing_best_model_path}")
                    self.model.load_weights(self.existing_best_model_path)
                else:
                    # Load the full model
                    self.best_model = tf.keras.models.load_model(self.existing_best_model_path)
                    print(f"Loaded existing best model from: {self.existing_best_model_path}")
                
                # Evaluate the model on the validation set to get the monitored metric (e.g., val_loss)
                if self.validation_data:
                    x_val, y_val = self.validation_data
                    results = self.model.evaluate(x_val, y_val, verbose=0)
                    
                    # Get the monitored metric (assuming it's validation loss or validation accuracy)
                    if self.monitor == 'val_loss':
                        self.best = results[0]  # Assuming loss is the first metric
                    elif self.monitor == 'val_accuracy':
                        self.best = results[1]  # Assuming accuracy is the second metric
                    
                    print(f"Initialized best {self.monitor} from the existing model: {self.best}")
                else:
                    print("Validation data not provided, unable to evaluate existing best model.")
                    # Set a default "best" value if validation data is not available
                    self.best = np.inf if 'loss' in self.monitor else -np.inf

            def on_epoch_end(self, epoch, logs=None):
                # Call the original on_epoch_end method to maintain ModelCheckpoint functionality
                super(BestModelCheckpoint, self).on_epoch_end(epoch, logs)
                    
        
        # Training Savepoint dir
        train_savepoints_dir = os.path.join("datasets", self.dataset_name, "train_savepoints")
        os.makedirs(train_savepoints_dir, exist_ok=True)
        # History saver callback
        epoch_savepoint_callback = EpochSavepoint()
        
        
        
        # Early stopping callback
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True, min_delta=0.001
        )
        
        # Custom callback to activate early stopping only after the warmup phase
        early_stopping_after_warmup_callback = EarlyStoppingAfterWarmup(
            warmup_epochs=warmup_epochs, early_stopping_callback=early_stopping_callback
        )
    
        # # Model checkpoint callback
        # if fold is not None:
        #     model_specs_dir = os.path.join("datasets", self.dataset_name, "model_specs", f"fold_{fold}")
        # else:
        #     model_specs_dir = os.path.join("datasets", self.dataset_name, "model_specs")
        
        # os.makedirs(model_specs_dir, exist_ok=True)
        
        # best_model_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=os.path.join(train_savepoints_dir, "best_model.h5"),
        #     save_weights_only=True,
        #     monitor="val_loss",
        #     mode="min", save_best_only=True
        # )
        
        best_model_path = os.path.join(train_savepoints_dir, "best_model.h5")
        best_model_callback = BestModelCheckpoint(
            existing_best_model_path=best_model_path,
            validation_data=val_dataset,
            filepath=best_model_path,
            save_weights_only=True,
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        # Create a callback to save the model after each epoch
        model_savepoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(train_savepoints_dir, "model_save.h5"),
            save_weights_only=True,
            save_best_only=False,
            verbose=1
        )

        
        # Learning rate warmup callback
        lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: warmup_scheduler(epoch, lr, warmup_epochs=warmup_epochs, initial_lr=initial_lr, target_lr=target_lr)
        )
        
        # Add all callbacks to the list
        callbacks = [early_stopping_after_warmup_callback, early_stopping_callback, best_model_callback, lr_scheduler_callback,
                     model_savepoint_callback, epoch_savepoint_callback]
        
        # Train the model
        return model, model.fit(
            train_dataset,  # No need to pass x and y separately
            validation_data=val_dataset,  # Both inputs and labels are contained in the datasets
            epochs=model_epochs,
            batch_size=batch_size,
            shuffle=True,  # Can remove this if already done during dataset preparation
            callbacks=callbacks
        )


        

def random_train_val_split(case_ids, validation_split):
    # Get the number of samples (assumed to be the same for all features)
    n_samples = case_ids.shape[0]

    # Initialize GroupShuffleSplit with desired validation split
    gss = GroupShuffleSplit(n_splits=1, test_size=validation_split, random_state=42)

    # Perform the split using case_ids as the grouping factor
    train_idx, val_idx = next(gss.split(np.arange(n_samples), groups=case_ids))

    return train_idx, val_idx


def sequential_train_val_split(case_ids, validation_split):
    # Get the number of samples (assumed to be the same for all features)
    n_samples = case_ids.shape[0]
    
    # Create a DataFrame to track indices and case_ids
    data_with_case_ids = pd.DataFrame({'index': np.arange(n_samples), 'case_id': case_ids})

    # Group by case_id and get unique case_ids in the order they appear
    unique_case_ids = data_with_case_ids['case_id'].unique()

    # Determine the number of validation cases
    n_val_cases = int(len(unique_case_ids) * validation_split)

    # Select the last n_val_cases for validation
    val_case_ids = unique_case_ids[-n_val_cases:]

    # Get train and validation indices based on case_id
    train_idx = data_with_case_ids[~data_with_case_ids['case_id'].isin(val_case_ids)]['index'].values
    val_idx = data_with_case_ids[data_with_case_ids['case_id'].isin(val_case_ids)]['index'].values

    return train_idx, val_idx


def k_fold_split(case_ids, n_splits):
    # Get the number of samples (assumed to be the same for all features)
    n_samples = case_ids.shape[0]

    # Initialize GroupKFold with the desired number of splits
    gkf = GroupKFold(n_splits=n_splits)

    # Perform the splits using case_ids as the grouping factor
    splits = gkf.split(np.arange(n_samples), groups=case_ids)

    return list(splits)  # returns a list of (train_idx, val_idx) tuples for each fold
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from package.constants import Feature_Type, Target, Temporal_Feature, Model_Architecture
from typing import List, Dict



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
                input_columns: List[str],
                target_columns: dict[str, Target],
                word_dicts: dict[str, Dict[str, int]],
                max_case_length: int,
                feature_type_dict: Dict[Feature_Type, List[str]],
                temporal_features: Dict[Temporal_Feature, bool],
                model_architecture: type[Model_Architecture],
                masking: bool = True
                ):
        
        # constants
        self.embed_dim: int = 36
        self.num_heads: int = 4
        self.ff_dim: int = 64
        
        self.dataset_name = dataset_name
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.word_dicts = word_dicts
        self.max_case_length = max_case_length
        self.feature_type_dict = feature_type_dict
        self.temporal_features = temporal_features
        self.model_architecture = model_architecture
        self.masking = masking
        
        self.model: Model = None
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

        def call(self, inputs, training, mask=None):
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
            
        def call(self, inputs):
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
            
        def call(self, inputs, mask=None):
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
        
        
    # class MaskedGlobalAveragePooling1D(layers.Layer):
    #     def call(self, inputs, mask=None):
    #         assert len(inputs.shape) == 3, f"Expected inputs to have 3 dimensions, got {inputs.shape}"

    #         if mask is not None:
    #             mask = tf.cast(mask, dtype=tf.float32)
    #             # mask = tf.expand_dims(mask, axis=-1)
    #             inputs *= mask

    #             summed = tf.reduce_sum(inputs, axis=1)
    #             mask_sum = tf.reduce_sum(mask, axis=1)
    #             mask_sum = tf.maximum(mask_sum, tf.ones_like(mask_sum))
    #             return summed / mask_sum
    #         else:
    #             return tf.reduce_mean(inputs, axis=1)


    class MaskedGlobalAveragePooling1D(tf.keras.layers.Layer):
        def __init__(self, model_wrapper):
            super(ModelWrapper.MaskedGlobalAveragePooling1D, self).__init__()
            self.model_wrapper = model_wrapper
            
        def call(self, inputs, mask=None):
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
                    model_epochs: int,
                    batch_size: int = 12,
                    model_learning_rate: float = 0.001
                    ):
        
        validation_split = 0.2
        
        self.model = self.get_model(
                                input_columns=self.input_columns,
                                target_columns=self.target_columns,
                                word_dicts=self.word_dicts,
                                max_case_length=self.max_case_length,
                                feature_type_dict=self.feature_type_dict,
                                temporal_features=self.temporal_features,
                                model_architecture=self.model_architecture
                                )
        
        
        # Check the number of target columns to determine if it's a multi-task or single-task problem
        if len(self.target_columns) > 1:
            print("Using Multi-Task Learning Setup")
            # Multi-task scenario: use MultiTaskLoss
            
            # Define if output is regression task tasks
            is_regression = []
            for feature in self.target_columns.keys():
                if feature in self.feature_type_dict[Feature_Type.CATEGORICAL]:
                    is_regression.append(False)  # False for classification tasks
                elif feature in self.feature_type_dict[Feature_Type.TIMESTAMP]:
                    is_regression.append(True)  # True for regression tasks


            # Custom loss function that integrates MultiTaskLossLayer
            def multi_task_loss_fn(is_regression):
                multi_task_loss_layer = MultiTaskLossLayer(is_regression)
                def loss_fn(y_true, y_pred):
                    # Since y_true and y_pred are passed separately for each output, we wrap them in a list
                    y_trues = [y_true]
                    y_preds = [y_pred]
                    # Call the multi-task loss layer for a single task
                    return multi_task_loss_layer(y_trues, y_preds)
                return loss_fn
            
            # Define the loss functions for each output
            losses = {}
            for output_key, reg in zip(train_token_dict_y.keys(), is_regression):
                losses.update({output_key: multi_task_loss_fn([reg])})

            # Compile the model with the combined loss
            self.model.compile(
                                optimizer=tf.keras.optimizers.Adam(model_learning_rate),
                                loss=losses#,
                                #metrics=[combined_loss]
                                )
            
        else:
            print("Using Single-Task Learning Setup")
            # Single-task scenario: use standard loss
            target_feature = list(self.target_columns.keys())[0]
            # get feature_type
            for feature_type, feature_lst in self.additional_columns.items():
                if target_feature in feature_lst: break
                
            # if target is categorical
            if feature_type is Feature_Type.CATEGORICAL:
                # Classification task
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(model_learning_rate),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                )
                
            # if target is temporal
            elif feature_type is Feature_Type.TIMESTAMP:
                # Regression task
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(model_learning_rate),
                    loss=tf.keras.losses.LogCosh(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()]
                )

        # Train-validation split
        first_key = next(iter(train_token_dict_x.keys()))
        n_samples = train_token_dict_x[first_key].shape[0]
        indices = np.arange(n_samples)
        train_indices, val_indices = train_test_split(indices, test_size=validation_split, random_state=42)

        # Split the data
        train_token_dict_x_split = {key: x_data[train_indices] for key, x_data in train_token_dict_x.items()}
        val_token_dict_x_split = {key: x_data[val_indices] for key, x_data in train_token_dict_x.items()}
        train_token_dict_y_split = {key: y_data[train_indices] for key, y_data in train_token_dict_y.items()}
        val_token_dict_y_split = {key: y_data[val_indices] for key, y_data in train_token_dict_y.items()}

        # Define callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            min_delta=0.001
        )

        model_specs_dir = os.path.join("datasets", self.dataset_name, "model_specs")
        os.makedirs(model_specs_dir, exist_ok=True)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_specs_dir, "best_model.h5"),
            save_weights_only=True,
            monitor="val_loss",
            mode="min", save_best_only=True)

        # Train the model
        print("----------------------------------------------------")
        print("Training...")
        self.history = self.model.fit(
            x=train_token_dict_x_split,
            y=train_token_dict_y_split,
            validation_data=(val_token_dict_x_split, val_token_dict_y_split),
            epochs=model_epochs, batch_size=batch_size, shuffle=True,
            callbacks=[early_stopping, model_checkpoint_callback]
        )
        
        return self.model, self.history
        
        
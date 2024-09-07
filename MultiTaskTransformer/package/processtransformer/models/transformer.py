import tensorflow as tf
from tensorflow.keras import layers, Model
from ..constants import Feature_Type, Target, Temporal_Feature, Model_Architecture
from typing import List, Dict



class MultiTaskLoss(layers.Layer):
    def __init__(self, is_regression, reduction='sum', **kwargs):
        super(ModelWrapper.MultiTaskLoss, self).__init__(**kwargs)
        self.is_regression = tf.constant(is_regression, dtype=tf.float32)
        self.n_tasks = len(is_regression)
        self.log_vars = self.add_weight(name='log_vars', shape=(self.n_tasks,), initializer='zeros', trainable=True)
        self.reduction = reduction
        # self.supports_masking = True

    def call(self, losses):
        stds = tf.exp(self.log_vars)**0.5
        coeffs = 1 / ((self.is_regression + 1) * (stds**2))
        multi_task_losses = coeffs * losses + tf.math.log(stds)

        if self.reduction == 'sum':
            return tf.reduce_sum(multi_task_losses)
        elif self.reduction == 'mean':
            return tf.reduce_mean(multi_task_losses)
        else:
            return multi_task_losses

    def get_config(self):
        config = super(ModelWrapper.MultiTaskLoss, self).get_config()
        config.update({
            'is_regression': self.is_regression.numpy().tolist(),
            'reduction': self.reduction
        })
        return config


class ModelWrapper():
    def __init__(self, model_architecture: Model_Architecture, max_case_length: int, masking: bool = False,
                 embed_dim=36, num_heads=4, ff_dim=64):
        self.model_architecture = model_architecture
        self.max_case_length = max_case_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.masking = masking
        
        

    class TransformerBlock(layers.Layer):
        def __init__(self, model_wrapper, embed_dim, num_heads, ff_dim, rate=0.1, mask=None):
            super(ModelWrapper.TransformerBlock, self).__init__()
            self.model_wrapper = model_wrapper
            self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = tf.keras.Sequential(
                [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
            )
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = layers.Dropout(rate)
            self.dropout2 = layers.Dropout(rate)
            self.mask = mask
            self.num_heads = num_heads
            self.supports_masking = True

        def call(self, inputs, training, mask=None):
            
            # if input feature dim (embed_dim) is not dividable by num_heads
            # if inputs.shape[-1] % self.num_heads != 0:
            #     # Calculate the next number divisible by num_heads
            #     next_divisible = ((inputs.shape[-1] + self.num_heads - 1) // self.num_heads) * self.num_heads
            #     inputs = layers.Dense(next_divisible)(inputs)
            
            
            if self.model_wrapper.masking:
                # Expand dims for num_head
                mask = tf.expand_dims(mask, axis=1)  # Shape becomes (batch_size, 1, max_case_length)
                # Broadcast the mask across the sequence length dimension
                # mask = tf.expand_dims(mask, axis=2)  # Add another dimension (batch_size, 1, 1, max_case_length)
                # mask = tf.tile(mask, [1, self.num_heads, 14, 1])  # Broadcast to (batch_size, num_heads, max_case_length, max_case_length)
                
                print("Mask for Transformer")
                print(f"Mask shape: {mask.shape}")
                print(f"Inputs shape: {inputs.shape}")
            
            # Apply multi-head attention with masking
            attn_output = self.att(inputs, inputs, attention_mask=mask)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            
            # Apply feed-forward network
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

        def compute_mask(self, inputs, mask=None):
            if self.model_wrapper.masking:
                # TODO: print mask shape
                print("Propagated Mask Transformer")
                print(f"Mask shape: {mask.shape}")
                print(f"Inputs shape: {inputs.shape}")
            # Ensure the masking is passed through to the next layers
            return mask


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
        
        def compute_mask(self, inputs, mask=None):
            if self.model_wrapper.masking:
                print("Propagated Mask PositionEmbeddingLayer")
                print(f"Mask shape: {mask.shape}")
                print(f"Inputs shape: {inputs.shape}")
            return mask
        
        
    # generates a mask for numerical inputs
    class NumericalMaskGeneration(layers.Layer):
        def __init__(self, model_wrapper):
            super(ModelWrapper.NumericalMaskGeneration, self).__init__()
            self.model_wrapper = model_wrapper
            self.supports_masking = True
            self.mask = None
            # self.masking_layer = layers.Masking(mask_value=-1)
            
        def call(self, inputs):
            self.mask = tf.not_equal(inputs, -1)
            self.mask = tf.cast(self.mask, tf.float32)
            inputs = layers.Multiply()([inputs, self.mask])
            
            return inputs
        
        def compute_mask(self, inputs, mask=None):
            if self.model_wrapper.masking:
                mask = self.mask
                print("Propagated Mask NumericalMaskGeneration Layer")
                print(f"Mask shape: {mask.shape}")
                print(f"Inputs shape: {inputs.shape}")
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
                
                print("Mask for MaskedGlobalAveragePooling1D")
                print(f"Mask shape: {mask.shape}")
                print(f"Inputs shape: {inputs.shape}")
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
            if self.model_wrapper.masking:
                mask = tf.expand_dims(mask, axis=-1)
                print("Propagated Mask MaskedGlobalAveragePooling1D")
                print(f"Mask shape: {mask.shape}")
                print(f"Inputs shape: {inputs.shape}")
            # Return the mask unchanged
            return None#mask
        
        

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
            # temporal_input = ModelWrapper.NumericalMaskGeneration(self)(temporal_input)
            # append temporal feature to temporal inputs
            temporal_inputs.append(temporal_input)

            # if day_of_week is used as additional temp feature
            if temporal_features[Temporal_Feature.DAY_OF_WEEK]:
                temporal_input_day_of_week = layers.Input(shape=(max_case_length,), name=f"input_{feature}_{Temporal_Feature.DAY_OF_WEEK.value}")
                # Generate mask
                temporal_input_day_of_week = ModelWrapper.NumericalMaskGeneration(self)(temporal_input_day_of_week)
                temporal_inputs.append(temporal_input_day_of_week)
            # if hour_of_day is used as additional temp feature
            if temporal_features[Temporal_Feature.HOUR_OF_DAY]:
                temporal_input_hour_of_day = layers.Input(shape=(max_case_length,), name=f"input_{feature}_{Temporal_Feature.HOUR_OF_DAY.value}")
                # Generate mask
                temporal_input_hour_of_day = ModelWrapper.NumericalMaskGeneration(self)(temporal_input_hour_of_day)
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
                            if self.masking:
                                # TODO: Generate mask
                                temporal_inputs = [ModelWrapper.NumericalMaskGeneration(self)(x) for x in temporal_inputs]
                            # extend feature_tensors with temporal inputs
                            feature_tensors.extend(temporal_inputs)
                            
                            
                
            # if temp_feature_exists:
            #     # flatten temporal tensors
            #     flattened_temporal_tensors = [item for sublist in temporal_tensors for item in sublist]
                
            #     # calculate the sum of temporal_layers
            #     sum_temp_tensors = len(flattened_temporal_tensors)
                
            #     # concat temporal layers
            #     temporal_tensors_concat = layers.Concatenate()( flattened_temporal_tensors )
                
            #     # reshape temporal layers for compatability with other layers
            #     temporal_tensors_concat = layers.Reshape(( 14, sum_temp_tensors ))(temporal_tensors_concat)
                
            #     # split concatenated temporal layers again
            #     splitted_temporal_tensors = tf.split(temporal_tensors_concat, num_or_size_splits=sum_temp_tensors, axis=-1)
                
                
            #     # bring prepared temporal layers back to the shape of temporal_layers (list of lists)
            #     prepared_temporal_tensors = []
            #     index = 0
            #     for sublist in temporal_tensors:
            #         length = len(sublist)
            #         prepared_temporal_tensors.append(splitted_temporal_tensors[index:index + length])
            #         index += length
            #     # append temporal tensors to feature tensors
            #     feature_tensors.extend(prepared_temporal_tensors)
            
            
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
            # concat categorical feature layers
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
                x = ModelWrapper.TransformerBlock(self, x.shape[-1], num_heads, ff_dim)(x)
                # append to list of feature transformer tensors
                feature_transf.append(x)
            # concat feature transformer tensors
            x = layers.Concatenate()(feature_transf)
            
            
        # seperate positional embeddings and transformers for each feature
        elif model_architecture is Model_Architecture.TIME_TARGET:
            feature_transf = []
            for idx, tensors_of_feature in enumerate(feature_tensors):
                transformers_of_feature = []
                for x in tensors_of_feature:
                    # add position embeddings
                    x = ModelWrapper.PositionEmbedding(model_wrapper = self, maxlen=max_case_length, embed_dim=x.shape[-1], name=f"position-embedding_feature_{idx}")(x)
                    # feed into transformer block
                    print(f"Feature {idx}")
                    print(x.shape)
                    x = ModelWrapper.TransformerBlock(self, x.shape[-1], num_heads, ff_dim)(x)
                    # append to list of transformers for each feature
                    transformers_of_feature.append(x)
                # if feature has multiple transformers, concat and apply another transformer
                if len(transformers_of_feature) > 1:
                    x = layers.Concatenate()(transformers_of_feature)
                    print(f"Feature {idx}")
                    print(x.shape)
                    x = ModelWrapper.TransformerBlock(self, x.shape[-1], num_heads, ff_dim)(x)
                # append to list of feature transformer tensors
                feature_transf.append(x)
            # concat feature transformer tensors
            x = layers.Concatenate()(feature_transf)
            
        print("last transformer")
        print(x.shape)
        
        # Stacking multiple transformer blocks
        for _ in range(num_layers):
            x = ModelWrapper.TransformerBlock(self, x.shape[-1], num_heads, ff_dim)(x)

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



    # TODO: vocab_size to list of vocab_sizesnum_classes_list
    # def get_model(self, input_columns: List[str], target_columns: Dict[str, Target], word_dicts: Dict[str, Dict[str, int]], max_case_length: int,
    #             feature_type_dict: Dict[Feature_Type, List[str]], temporal_features: Dict[Temporal_Feature, bool],
    #             model_architecture: Model_Architecture,
    #             embed_dim=36, num_heads=4, ff_dim=64, num_layers=1):
        
    #     if self.masking:
    #         print("Masking active.")
    #     else:
    #         print("Masking not active.")
        
        
    #     inputs_layers = []
        
    #     for feature in input_columns:
    #         # Input Layer for temporal feature
    #         x = layers.Input(shape=(max_case_length,), name=f"input_{feature}")
    #         inputs_layers.append(x)
    #         # Generate mask
    #         if self.masking:
    #             x = ModelWrapper.NumericalMaskGeneration(self)(x)
                
                        
    #     x = ModelWrapper.PositionEmbedding(model_wrapper = self, maxlen=max_case_length, embed_dim=x.shape[-1], name="position-embedding")(x)
    #     x = ModelWrapper.TransformerBlock(self, x.shape[-1], num_heads, ff_dim)(x)   
        
    #     x = ModelWrapper.MaskedGlobalAveragePooling1D(self)(x)
        
    #     print(f"Shape after pooling: {x.shape}")
        
    #     # Fully connected layers
    #     x = layers.Dropout(0.1)(x)
    #     x = layers.Dense(64, activation="relu")(x)
    #     x = layers.Dropout(0.1)(x)
        
    #     # Output layers for categorical features
    #     outputs = []
    #     for feature, target in target_columns.items():
    #         for feature_type, feature_lst in feature_type_dict.items():
    #             if feature in feature_lst:
    #                 if feature_type is Feature_Type.CATEGORICAL:
    #                     if target == Target.NEXT_FEATURE: dict_str = "y_next_word_dict"
    #                     elif target == Target.LAST_FEATURE: dict_str = "y_last_word_dict"
    #                     else: raise ValueError("Target type is not known.")
    #                     output_dim = len(word_dicts[feature][dict_str])
    #                     # TODO: Debugging
    #                     output_dense = layers.Dense(output_dim, activation="softmax", name=f"output_{feature}")(x)
    #                     print(f"Output_dense Shape: {output_dense.shape}")
    #                     outputs.append(output_dense)
    #                     # outputs.append( layers.Dense(output_dim, activation="softmax", name=f"output_{feature}")(x) )
    #                 if feature_type is Feature_Type.TIMESTAMP:
    #                     output_dense = layers.Dense(1, activation="linear", name=f"output_{feature}")(x)
    #                     print(f"Output_dense Shape: {output_dense.shape}")
    #                     outputs.append(output_dense)
    #                     # outputs.append( layers.Dense(1, activation="linear", name=f"output_{feature}")(x) )
                        
    #     # Model definition
    #     transformer = Model(inputs=inputs_layers, outputs=outputs, name="next_categorical_transformer")
        
    #     return transformer
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
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name=name)
            self.supports_masking = True
            
        def call(self, inputs, mask=None):
            embeddings = self.token_emb(inputs)
            
            # Ensure the custom mask is propagated
            # if self.mask is not None:
            #     mask = self.mask
            
            return embeddings#, mask  # Return token embeddings
        
        def compute_mask(self, inputs, mask=None):
            # print("Propagated Mask Token Embedding")
            # print(f"Mask shape: {mask.shape}")
            # print(f"Inputs shape: {inputs.shape}")
            # Get the mask from the embedding layer
            
            # if mask is not None:
            # Expand mask to fit attention layer's expected shape
            # mask = tf.expand_dims(mask, axis=1)  # (batch_size, 1, seq_len)
            # mask = tf.expand_dims(mask, axis=-1)  # (batch_size, 1, seq_len, 1)
            
            # print("Propagated Mask Token Embedding Layer:")
            # print(f"Mask shape: {mask.shape}")
            # print(f"Inputs shape: {inputs.shape}")
            
            if self.model_wrapper.masking:
                mask = tf.math.not_equal(inputs, -1)
                
                print("Propagated Mask Token Embedding Mask:")
                print(f"Mask shape: {mask.shape}")
                print(f"Inputs shape: {inputs.shape}")
                
            return mask
        
        
    class PositionEmbedding(layers.Layer):
        """
        Position Embedding Layer.

        Args:
            maxlen (int): Maximum length of the sequences.
            embed_dim (int): Dimensionality of the embedding space.
        """
        def __init__(self, model_wrapper, maxlen, embed_dim, name, num_heads):
            super(ModelWrapper.PositionEmbedding, self).__init__()
            self.model_wrapper = model_wrapper
            self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim, name=name)
            self.num_heads = num_heads
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
            # Return the mask unchanged
            
            # TODO: Expand mask to fit attention layer's expected shape
            # mask = tf.expand_dims(mask, axis=1)  # (batch_size, 1, seq_len)
            # mask = tf.tile(mask, [1, 4, 1])
            # mask = tf.broadcast_to(mask, (tf.shape(mask)[0], 4, tf.shape(mask)[2], tf.shape(mask)[2]))  # (batch_size, 1, seq_len, seq_len)
            return mask
        
        
    # only ensures propagation of the mask through keras
    class MaskPropagation(layers.Layer):
        def __init__(self, model_wrapper, mask):
            super(ModelWrapper.MaskPropagation, self).__init__()
            self.model_wrapper = model_wrapper
            self.mask = mask
            self.supports_masking = True
            
        def call(self, inputs):
            # Ensure mask is recomputed and returned here
            self.add_update(self.compute_mask(inputs))
            return inputs
        
        def compute_mask(self, inputs, mask=None):
            mask = self.mask
            print("Propagated Mask MaskPropagationLayer")
            print(f"Mask shape: {mask.shape}")
            print(f"Inputs shape: {inputs.shape}")
            # Return the mask unchanged
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
    # def get_model(input_columns: List[str], target_columns: Dict[str, Target], word_dicts: Dict[str, Dict[str, int]], max_case_length: int,
    #               feature_type_dict: Dict[Feature_Type, List[str]], temporal_features: Dict[Temporal_Feature, bool],
    #               model_architecture: Model_Architecture,
    #               embed_dim=36, num_heads=4, ff_dim=64, num_layers=1, mask=None):
    #     """
    #     Constructs the next categorical prediction model using a transformer architecture.
        
    #     Args:
    #         max_case_length (int): Maximum length of the sequences (cases).
    #         embed_dim (int): Dimensionality of the embeddings. Defaults to 36.
    #         num_heads (int): Number of attention heads. Defaults to 4.
    #         ff_dim (int): Dimensionality of the feed-forward layer. Defaults to 64.
    #         num_layers (int): Number of transformer blocks. Defaults to 1.
        
    #     Returns:
    #         tf.keras.Model: Compiled transformer model for next categorical prediction.
    #     """
        
    #     def prepare_categorical_input(feature: str, mask):
    #         # generate input layer for categorical feature
    #         categorical_input = layers.Input(shape=(max_case_length,), name=f"input_{feature}")
    #         # do token embedding for categorical feature
    #         categorical_emb = TokenEmbedding(vocab_size = len(word_dicts[feature]["x_word_dict"]),
    #                                         embed_dim = embed_dim,
    #                                         name = f"{feature}_token-embeddings",
    #                                         mask = mask
    #                                         )(categorical_input)
    #         return categorical_input, [categorical_emb]
        
        
    #     def prepare_temporal_input(feature):
    #         temporal_inputs = []
    #         # Input Layer for temporal feature
    #         temporal_input = layers.Input(shape=(max_case_length,), name=f"input_{feature}")
    #         # append temporal feature to temporal inputs
    #         temporal_inputs.append(temporal_input)

    #         # if day_of_week is used as additional temp feature
    #         if temporal_features[Temporal_Feature.DAY_OF_WEEK]:
    #             temporal_input_day_of_week = layers.Input(shape=(max_case_length,), name=f"input_{feature}_{Temporal_Feature.DAY_OF_WEEK.value}")
    #             temporal_inputs.append(temporal_input_day_of_week)
    #         # if hour_of_day is used as additional temp feature
    #         if temporal_features[Temporal_Feature.HOUR_OF_DAY]:
    #             temporal_input_hour_of_day = layers.Input(shape=(max_case_length,), name=f"input_{feature}_{Temporal_Feature.HOUR_OF_DAY.value}")
    #             temporal_inputs.append(temporal_input_hour_of_day)
                
    #         return temporal_inputs
        
        
    #     def prepare_inputs(mask):
    #         inputs_layers, temporal_tensors, feature_tensors = [], [], []
    #         temp_feature_exists = False
            
    #         for feature in input_columns:
    #             for feature_type, feature_lst in feature_type_dict.items():
    #                 if feature in feature_lst:
    #                     # feature is categorical
    #                     if feature_type is Feature_Type.CATEGORICAL:
    #                         categorical_input, categorical_embs = prepare_categorical_input(feature, mask)
    #                         # append input layer to inputs
    #                         inputs_layers.append(categorical_input)
    #                         # append categorical token embedding to feature_tensors
    #                         feature_tensors.append(categorical_embs)
                            
    #                     # feature is temporal
    #                     elif feature_type is Feature_Type.TIMESTAMP:
    #                         temp_feature_exists = True
    #                         temporal_inputs = prepare_temporal_input(feature)
    #                         # append temporal inputs to inputs
    #                         inputs_layers.extend(temporal_inputs)
    #                         # propagate masks for inputs
    #                         for idx, temporal_input in enumerate(temporal_inputs):
    #                             temporal_inputs[idx] = MaskPropagation(mask)(temporal_input)
                            
    #                         # append temporal inputs to temporal layers
    #                         temporal_tensors.append(temporal_inputs)
                            
                            
                
    #         if temp_feature_exists:
    #             # flatten temporal tensors
    #             flattened_temporal_tensors = [item for sublist in temporal_tensors for item in sublist]
                
    #             # calculate the sum of temporal_layers
    #             sum_temp_tensors = len(flattened_temporal_tensors)
                
    #             # concat temporal layers
    #             temporal_tensors_concat = layers.Concatenate()( flattened_temporal_tensors )
                
    #             # reshape temporal layers for compatability with other layers
    #             temporal_tensors_concat = layers.Reshape(( 14, sum_temp_tensors ))(temporal_tensors_concat)
                
    #             # split concatenated temporal layers again
    #             splitted_temporal_tensors = tf.split(temporal_tensors_concat, num_or_size_splits=sum_temp_tensors, axis=-1)
                
                
    #             # bring prepared temporal layers back to the shape of temporal_layers (list of lists)
    #             prepared_temporal_tensors = []
    #             index = 0
    #             for sublist in temporal_tensors:
    #                 length = len(sublist)
    #                 prepared_temporal_tensors.append(splitted_temporal_tensors[index:index + length])
    #                 index += length
    #             # append temporal tensors to feature tensors
    #             feature_tensors.extend(prepared_temporal_tensors)
            
    #         return inputs_layers, feature_tensors
                            
    #     ############################################################################################

    #     print("Creating model...")
    #     original_mask = None
        
    #     if mask is None:
    #         print("no masking applied")
    #     else: print("masking applied")
        
    #     # Convert mask to a TensorFlow tensor
    #     if mask is not None:
    #         mask = tf.convert_to_tensor(mask)
    #         original_mask = mask
            
    #         # Expand dims for num_head
    #         mask = tf.expand_dims(mask, axis=1)  # Shape becomes (batch_size, 1, max_case_length)
    #         # Broadcast the mask across the sequence length dimension
    #         mask = tf.expand_dims(mask, axis=2)  # Add another dimension (batch_size, 1, 1, max_case_length)
    #         mask = tf.tile(mask, [1, num_heads, 14, 1])  # Broadcast to (batch_size, num_heads, max_case_length, max_case_length)
            
    #         # mask = tf.transpose(mask, perm=[0, 1, 3, 3])
    #         # mask = tf.tile(mask, [1, num_heads, max_case_length, 1])
            
            
    #     print(f"--------- ORIGINAL MASKING SHAPE: {original_mask.shape} ---------------")
    #     print(f"--------- TRANSFORMER MASKING SHAPE: {mask.shape} ---------------")
            
    #     # prepare inputs
    #     inputs_layers, feature_tensors = prepare_inputs(mask)
        
    #     # common embeddings and transformers for all features
    #     if model_architecture is Model_Architecture.COMMON_POSEMBS_TRANSF:
    #         # flatten feature_tensors
    #         feature_tensors = [item for sublist in feature_tensors for item in sublist]
    #         # concat categorical feature layers
    #         x = layers.Concatenate()(feature_tensors)
    #         # add position embedding to the concatenated layers
    #         pos_emb = PositionEmbedding(maxlen=max_case_length, embed_dim=x.shape[-1], name="position-embedding_common", num_heads=num_heads)
    #         x = pos_emb(x)
    #         # print(f"Pos Emb direct call: {pos_emb.mask.shape}")
            
    #         # x = PositionEmbedding(maxlen=max_case_length, embed_dim=x.shape[-1], name="position-embedding_common", num_heads=num_heads)(x)
            
    #     # seperate positional embeddings and common transformer for all features
    #     elif model_architecture is Model_Architecture.SEPERATE_POSEMBS:
    #         feature_embs = []
    #         for idx, tensors_of_feature in enumerate(feature_tensors):
    #             # concat tensors of each feature
    #             x = layers.Concatenate()(tensors_of_feature)
    #             # add position embedding
    #             x = PositionEmbedding(maxlen=max_case_length, embed_dim=x.shape[-1], name=f"position-embedding_feature_{idx}")(x)
    #             # append to list of feature_embs
    #             feature_embs.append(x)
    #         # concat feature embs
    #         x = layers.Concatenate()(feature_embs)
            
    #     # seperate positional embeddings and transformers for each feature
    #     elif model_architecture is Model_Architecture.SEPERATE_TRANSF:
    #         feature_transf = []
    #         for idx, tensors_of_feature in enumerate(feature_tensors):
    #             # concat tensors of each feature
    #             x = layers.Concatenate()(tensors_of_feature)
    #             # add position embeddings
    #             x = PositionEmbedding(maxlen=max_case_length, embed_dim=x.shape[-1], name=f"position-embedding_feature_{idx}")(x)
    #             # feed into transformer block
    #             x = TransformerBlock(x.shape[-1], num_heads, ff_dim)(x, mask=mask)
    #             # append to list of feature transformer tensors
    #             feature_transf.append(x)
    #         # concat feature transformer tensors
    #         x = layers.Concatenate()(feature_transf)
            
            
    #     # seperate positional embeddings and transformers for each feature
    #     elif model_architecture is Model_Architecture.TIME_TARGET:
    #         feature_transf = []
    #         for idx, tensors_of_feature in enumerate(feature_tensors):
    #             transformers_of_feature = []
    #             for x in tensors_of_feature:
    #                 # add position embeddings
    #                 x = PositionEmbedding(maxlen=max_case_length, embed_dim=x.shape[-1], name=f"position-embedding_feature_{idx}")(x)
    #                 # feed into transformer block
    #                 print(f"Feature {idx}")
    #                 print(x.shape)
    #                 # print("mask shape")
    #                 # print(x.compute_mask(inputs=x).shape)
    #                 x = TransformerBlock(x.shape[-1], num_heads, ff_dim)(x)
    #                 # append to list of transformers for each feature
    #                 transformers_of_feature.append(x)
    #             # if feature has multiple transformers, concat and apply another transformer
    #             if len(transformers_of_feature) > 1:
    #                 x = layers.Concatenate()(transformers_of_feature)
    #                 x = MaskPropagation(mask)(x) # TODO: test this
    #                 print(f"Feature {idx}")
    #                 print(x.shape)
    #                 # print("mask shape")
    #                 # print(x.compute_mask(inputs=x).shape)
    #                 x = TransformerBlock(x.shape[-1], num_heads, ff_dim)(x, mask=mask)
    #             # append to list of feature transformer tensors
    #             feature_transf.append(x)
    #         # concat feature transformer tensors
    #         x = layers.Concatenate()(feature_transf)
            
    #     print("last transformer")
    #     print(x.shape)
    #     # x = MaskPropagation(mask)(x)
        
    #     # print("mask shape")
    #     # print(x.compute_mask(inputs=x).shape)
    #     # Stacking multiple transformer blocks
    #     for _ in range(num_layers):
    #         x = TransformerBlock(x.shape[-1], num_heads, ff_dim)(x)
    #         # x = TransformerBlock(x.shape[-1], num_heads, ff_dim)(x, mask=None)
        
    #     # Global average pooling
    #     # if mask is not None:
    #     #     x = MaskedGlobalAveragePooling1D()(x, mask)
    #     # else:
    #     print(f"original mask: {original_mask.shape}")
    #     pooling_mask = tf.expand_dims(original_mask, axis=-1)
    #     pooling_mask = tf.tile(pooling_mask, [1, 444, 1]) # TODO: TEST
    #     print(f"pooling mask: {pooling_mask.shape}")
    #     mask_propagation_layer = MaskPropagation(pooling_mask)
    #     x = mask_propagation_layer(x)
    #     print(f"propagation Mask direct call: {mask_propagation_layer.compute_mask(x, None).shape}")
    #     # x = MaskPropagation(pooling_mask)(x)
        
    #     print("bla")
        
    #     x = MaskedGlobalAveragePooling1D()(x, pooling_mask)
    #     # x = layers.GlobalAveragePooling1D()(x)
        
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
    #                     outputs.append( layers.Dense(output_dim, activation="softmax", name=f"output_{feature}")(x) )
    #                 if feature_type is Feature_Type.TIMESTAMP:
    #                     outputs.append( layers.Dense(1, activation="linear", name=f"output_{feature}")(x) )
        
        
        
    #     # Model definition
    #     transformer = Model(inputs=inputs_layers, outputs=outputs, name="next_categorical_transformer")
        
    #     return transformer



    # TODO: vocab_size to list of vocab_sizesnum_classes_list
    def get_model(self, input_columns: List[str], target_columns: Dict[str, Target], word_dicts: Dict[str, Dict[str, int]], max_case_length: int,
                feature_type_dict: Dict[Feature_Type, List[str]], temporal_features: Dict[Temporal_Feature, bool],
                model_architecture: Model_Architecture,
                embed_dim=36, num_heads=4, ff_dim=64, num_layers=1):
        
        if self.masking:
            print("Masking active.")
        else:
            print("Masking not active.")
        
        
        inputs_layers = []
        
        for feature in input_columns:
            for feature_type, feature_lst in feature_type_dict.items():
                if feature in feature_lst:
                    # feature is categorical
                    if feature_type is Feature_Type.CATEGORICAL:
                        
                        # generate input layer for categorical feature
                        categorical_input = layers.Input(shape=(max_case_length,), name=f"input_{feature}")
                        inputs_layers.append(categorical_input)
                        # do token embedding for categorical feature
                        x = ModelWrapper.TokenEmbedding(model_wrapper = self,
                                                        vocab_size = len(word_dicts[feature]["x_word_dict"]),
                                                        embed_dim = embed_dim,
                                                        name = f"{feature}_token-embeddings"
                                                        )(categorical_input)
                        
        x = ModelWrapper.PositionEmbedding(model_wrapper = self, maxlen=max_case_length, embed_dim=x.shape[-1],
                                           name="position-embedding", num_heads=num_heads)(x)
        x = ModelWrapper.TransformerBlock(self, x.shape[-1], num_heads, ff_dim)(x)
        
                
        # print(f"original mask: {original_mask.shape}")
        # pooling_mask = tf.expand_dims(original_mask, axis=-1)
        # pooling_mask = tf.tile(pooling_mask, [1, 1, 1]) # TODO: TEST
        # print(f"pooling mask: {pooling_mask.shape}")
        # x = ModelWrapper.MaskPropagation(pooling_mask)(x)
        # x = mask_propagation_layer(x)
        # print(f"propagation Mask direct call: {mask_propagation_layer.compute_mask(x, None).shape}")       
        
        x = ModelWrapper.MaskedGlobalAveragePooling1D(self)(x)
        
        print(f"Shape after pooling: {x.shape}")
        
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
                        # TODO: Debugging
                        output_dense = layers.Dense(output_dim, activation="softmax", name=f"output_{feature}")(x)
                        print(f"Output_dense Shape: {output_dense.shape}")
                        outputs.append(output_dense)
                        # outputs.append( layers.Dense(output_dim, activation="softmax", name=f"output_{feature}")(x) )
                    if feature_type is Feature_Type.TIMESTAMP:
                        output_dense = layers.Dense(1, activation="linear", name=f"output_{feature}")(x)
                        print(f"Output_dense Shape: {output_dense.shape}")
                        outputs.append(output_dense)
                        # outputs.append( layers.Dense(1, activation="linear", name=f"output_{feature}")(x) )
                        
        # Model definition
        transformer = Model(inputs=inputs_layers, outputs=outputs, name="next_categorical_transformer")
        
        return transformer
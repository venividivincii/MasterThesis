import tensorflow as tf
from tensorflow.keras import layers, Model
from ..constants import Feature_Type, Target, Temporal_Feature, Model_Architecture
from typing import List, Dict

class TransformerBlock(layers.Layer):
    """
    Transformer Block consisting of Multi-Head Self-Attention and Feed-Forward Network with Layer Normalization and Dropout.
    
    Args:
        embed_dim (int): Dimensionality of the embedding space.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward network.
        rate (float): Dropout rate.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, mask=None, training=False):
        """
        Forward pass for the transformer block.
        
        Args:
            inputs (tf.Tensor): Input tensor.
            training (bool): Boolean indicating whether the model is in training mode.
        
        Returns:
            tf.Tensor: Output tensor after applying self-attention and feed-forward network.
        """
        
        attn_output = self.att(inputs, inputs, attention_mask=mask)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Add & Norm
        ffn_output = self.ffn(out1)  # Feed-Forward Network
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Add & Norm


class TokenEmbedding(layers.Layer):
    """
    Token Embedding Layer.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimensionality of the embedding space.
        mask_padding (bool): Whether to mask padding tokens (zero index).
    """
    def __init__(self, vocab_size, embed_dim, name, mask_padding=False):
        super(TokenEmbedding, self).__init__()
        mask_padding = False
        if mask_padding:
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        else:
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name=name)
        
    def call(self, x):
        """
        Forward pass for token embedding.

        Args:
            x (tf.Tensor): Input tensor containing token indices.

        Returns:
            tf.Tensor: Output tensor with token embeddings.
        """
        return self.token_emb(x)  # Return token embeddings
    
    
class PositionEmbedding(layers.Layer):
    """
    Position Embedding Layer.

    Args:
        maxlen (int): Maximum length of the sequences.
        embed_dim (int): Dimensionality of the embedding space.
    """
    def __init__(self, maxlen, embed_dim, name):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim, name=name)
        
    def call(self, x):
        """
        Forward pass for position embedding.

        Args:
            x (tf.Tensor): Input tensor containing token indices or token embeddings.

        Returns:
            tf.Tensor: Output tensor with position embeddings.
        """
        maxlen = tf.shape(x)[1]  # Length of the input sequence
        positions = tf.range(start=0, limit=maxlen, delta=1)  # Generate position indices
        positions = self.pos_emb(positions)  # Get position embeddings
        positions = tf.expand_dims(positions, 0)  # Add a batch dimension (1, maxlen, embed_dim)
        return x + positions  # Add position embeddings to the input tensor



class TokenAndPositionEmbedding(layers.Layer):
    """
    Token and Position Embedding Layer.
    
    Args:
        maxlen (int): Maximum length of the sequences.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimensionality of the embedding space.
    """
    def __init__(self, maxlen, vocab_size, embed_dim, mask_padding):
        super(TokenAndPositionEmbedding, self).__init__()
        # self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        # if mask_padding:
        #     self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        # else:
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=False)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        

    def call(self, x):
        """
        Forward pass for token and position embedding.
        
        Args:
            x (tf.Tensor): Input tensor containing token indices.
        
        Returns:
            tf.Tensor: Output tensor with combined token and position embeddings.
        """
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions  # Combine token and position embeddings
    
    def compute_mask(self, inputs, mask=None):
        return self.token_emb.compute_mask(inputs, mask)
    
    

class MultiTaskLoss(layers.Layer):
    def __init__(self, is_regression, reduction='sum', **kwargs):
        super(MultiTaskLoss, self).__init__(**kwargs)
        self.is_regression = tf.constant(is_regression, dtype=tf.float32)
        self.n_tasks = len(is_regression)
        self.log_vars = self.add_weight(name='log_vars', shape=(self.n_tasks,), initializer='zeros', trainable=True)
        self.reduction = reduction

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
        config = super(MultiTaskLoss, self).get_config()
        config.update({
            'is_regression': self.is_regression.numpy().tolist(),
            'reduction': self.reduction
        })
        return config    
    

# TODO: vocab_size to list of vocab_sizesnum_classes_list
def get_model(input_columns: List[str], target_columns: Dict[str, Target], word_dicts: Dict[str, Dict[str, int]], max_case_length: int,
              feature_type_dict: Dict[Feature_Type, List[str]], temporal_features: Dict[Temporal_Feature, bool],
              model_architecture: Model_Architecture,
              embed_dim=36, num_heads=4, ff_dim=64, num_layers=1, mask_padding: bool = False):
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
    # global average pooling with mask
    def masked_global_avg_pool(x, mask):
        mask = tf.cast(mask, dtype=tf.float32)  # Ensure mask is of type float32
        mask = tf.squeeze(mask, axis=[1, 2])  # Remove the extra dimensions, making mask shape [batch_size, sequence_length]
        mask = tf.expand_dims(mask, axis=-1)  # Expand the mask to shape [batch_size, sequence_length, 1]
        x *= mask  # Now x and mask have compatible shapes for element-wise multiplication
        return tf.reduce_sum(x, axis=1) / tf.reduce_sum(mask, axis=1)  # Aggregate along the sequence length
    
    
    def prepare_categorical_input(feature: str):
        # generate input layer for categorical feature
        categorical_input = layers.Input(shape=(max_case_length,), name=f"input_{feature}")
        # do token embedding for categorical feature
        categorical_emb = TokenEmbedding(vocab_size = len(word_dicts[feature]["x_word_dict"]),
                                        embed_dim = embed_dim,
                                        name = f"{feature}_token-embeddings")(categorical_input)
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
        inputs_layers, temporal_tensors, feature_tensors = [], [], []
        temp_feature_exists = False
        
        for feature in input_columns:
            for feature_type, feature_lst in feature_type_dict.items():
                if feature in feature_lst:
                    # feature is categorical
                    if feature_type is Feature_Type.CATEGORICAL:
                        categorical_input, categorical_emb_dict = prepare_categorical_input(feature)
                        # append input layer to inputs
                        inputs_layers.append(categorical_input)
                        # append categorical token embedding to feature_tensors
                        feature_tensors.append(categorical_emb_dict)
                        
                    # feature is temporal
                    elif feature_type is Feature_Type.TIMESTAMP:
                        temp_feature_exists = True
                        temporal_inputs = prepare_temporal_input(feature)
                        # append temporal inputs to inputs
                        inputs_layers.extend(temporal_inputs)
                        # append temporal inputs to temporal layers
                        temporal_tensors.append(temporal_inputs)
              
        if temp_feature_exists:
            # flatten temporal tensors
            flattened_temporal_tensors = [item for sublist in temporal_tensors for item in sublist]
            # calculate the sum of temporal_layers
            sum_temp_tensors = len(flattened_temporal_tensors)
            # concat temporal layers
            temporal_tensors_concat = layers.Concatenate()( flattened_temporal_tensors )
            # normalize temporal layers
            temporal_tensors_concat = layers.LayerNormalization()(temporal_tensors_concat)
            # reshape temporal layers for compatability with other layers
            temporal_tensors_concat = layers.Reshape(( 14, sum_temp_tensors ))(temporal_tensors_concat)
            # split concatenated temporal layers again
            splitted_temporal_tensors = tf.split(temporal_tensors_concat, num_or_size_splits=sum_temp_tensors, axis=-1)
            
            # bring prepared temporal layers back to the shape of temporal_layers (list of lists)
            prepared_temporal_tensors = []
            index = 0
            for sublist in temporal_tensors:
                length = len(sublist)
                prepared_temporal_tensors.append(splitted_temporal_tensors[index:index + length])
                index += length
            # append temporal layers to feature layers
            feature_tensors.extend(prepared_temporal_tensors)
        
        return inputs_layers, feature_tensors
                        
    ############################################################################################

    print("Creating model...")
    
    # prepare inputs
    inputs_layers, feature_tensors = prepare_inputs()
    
    # common embeddings and transformers for all features
    if model_architecture is Model_Architecture.COMMON_POSEMBS_TRANSF:
        # flatten feature_tensors
        feature_tensors = [item for sublist in feature_tensors for item in sublist]
        # concat categorical feature layers
        x = layers.Concatenate()(feature_tensors)
        # add position embedding to the concatenated layers
        x = PositionEmbedding(maxlen=max_case_length, embed_dim=x.shape[-1], name="position-embedding_common")(x)
        
    # seperate positional embeddings and common transformer for all features
    elif model_architecture is Model_Architecture.SEPERATE_POSEMBS:
        feature_embs = []
        for idx, tensors_of_feature in enumerate(feature_tensors):
            # concat tensors of each feature
            x = layers.Concatenate()(tensors_of_feature)
            # add position embedding
            x = PositionEmbedding(maxlen=max_case_length, embed_dim=x.shape[-1], name=f"position-embedding_feature_{idx}")(x)
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
            x = PositionEmbedding(maxlen=max_case_length, embed_dim=x.shape[-1], name=f"position-embedding_feature_{idx}")(x)
            # feed into transformer block
            x = TransformerBlock(x.shape[-1], num_heads, ff_dim)(x)
            # append to list of feature transformer tensors
            feature_transf.append(x)
        # concat feature transformer tensors
        x = layers.Concatenate()(feature_transf)
    
    # Stacking multiple transformer blocks
    for _ in range(num_layers):
        x = TransformerBlock(x.shape[-1], num_heads, ff_dim)(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
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
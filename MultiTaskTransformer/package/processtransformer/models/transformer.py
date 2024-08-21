import tensorflow as tf
from tensorflow.keras import layers, Model
from ..constants import Feature_Type, Target, Temporal_Feature
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
        print("x.shape")
        print(x.shape)
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

# TODO: vocab_size to list of vocab_sizesnum_classes_list
def get_model(input_columns: List[str], target_columns: Dict[str, Target], word_dicts: Dict[str, Dict[str, int]], max_case_length: int,
              feature_type_dict: Dict[Feature_Type, List[str]], temporal_features: Dict[Temporal_Feature, bool],
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

    print("Creating model...")

    # Categorical Input layers
    # categorical_inputs, categorical_feature_layers, masks = [], [], []
    inputs, feature_layers = [], []
    if Feature_Type.CATEGORICAL in feature_type_dict:
        for cat_feature in [ s for s in input_columns if s in feature_type_dict[Feature_Type.CATEGORICAL] ]:
            # Input Layer for categorical feature
            categorical_input = layers.Input(shape=(max_case_length,), name=f"input_{cat_feature}")
            inputs.append(categorical_input)
            # vocab_size, embed_dim, name
            categorical_layer = TokenEmbedding(vocab_size = len(word_dicts[cat_feature]["x_word_dict"])+1,
                                               embed_dim = embed_dim,
                                               name = f"{cat_feature}_token-embeddings")(categorical_input)
            print(f"{cat_feature} x_word_dict length:")
            print(len(word_dicts[cat_feature]["x_word_dict"]))
            feature_layers.append(categorical_layer)
            # print(f"categorical emb shape: {categorical_layer.shape}")
            # Create mask based on input
            # if mask_padding:
            #     mask = tf.cast(tf.math.not_equal(categorical_input, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
            #     masks.append(mask)
            # else: mask = None

    # Temporal Input layers
    temporal_layers = []
    if Feature_Type.TIMESTAMP in feature_type_dict:
        for temp_feature in [ s for s in input_columns if s in feature_type_dict[Feature_Type.TIMESTAMP] ]:
            # Input Layer for temporal feature
            temporal_input_feature = layers.Input(shape=(max_case_length,), name=f"input_{temp_feature}")
            # print(f"temporal_input_feature shape: {temporal_input_feature.shape}")
            temporal_layers.append(temporal_input_feature)
            # if day_of_week is used as additional temp feature
            if temporal_features[Temporal_Feature.DAY_OF_WEEK]:
                temporal_input_day_of_week = layers.Input(shape=(max_case_length,), name=f"input_{temp_feature}_{Temporal_Feature.DAY_OF_WEEK.value}")
                temporal_layers.append(temporal_input_day_of_week)
            # if hour_of_day is used as additional temp feature
            if temporal_features[Temporal_Feature.HOUR_OF_DAY]:
                temporal_input_hour_of_day = layers.Input(shape=(max_case_length,), name=f"input_{temp_feature}_{Temporal_Feature.HOUR_OF_DAY.value}")
                temporal_layers.append(temporal_input_hour_of_day)
        inputs.append(temporal_layers)
        sum_temp_layers = len(temporal_layers)
        # concat temporal layers
        temporal_layers = layers.Concatenate()(temporal_layers)
        # print(f"concat temp layers shape: {temporal_layers.shape}")
        temporal_layers = layers.LayerNormalization()(temporal_layers)
        # print(f"normalized temp layers shape: {temporal_layers.shape}")
        # reshape temporal layers for compatability with other layers
        temporal_layers = layers.Reshape((14, sum_temp_layers))(temporal_layers)
        # print(f"reshaped layers shape: {temporal_layers.shape}")
        feature_layers.append(temporal_layers)
        
        # temp = PositionEmbedding(max_case_length, embed_dim)(temp)
    mask = None
    # return None
        # Input embedding for categorical feature
        # categorical_emb = TokenAndPositionEmbedding(max_case_length, len(word_dicts[cat_feature]["x_word_dict"]), embed_dim, mask_padding)(categorical_input)
        # Transformer Block for categorical feature
        # categorical_feature_layers.append( TransformerBlock(embed_dim, num_heads, ff_dim)(categorical_emb, mask=mask) )
        
    # Temporal Input layers
    # temporal_inputs, temporal_feature_layers = [], []
    # for temp_feature in [ s for s in input_columns if s in feature_type_dict[Feature_Type.TIMESTAMP] ]:
    #     # Input Layer for temporal feature
    #     temporal_input = layers.Input(shape=(max_case_length,), name=f"input_{temp_feature}")
    #     temporal_inputs.append(temporal_input)
        
    # concat categorical feature layers
    x = layers.Concatenate()(feature_layers)
    # print(f"shape after concat all: {x.shape}")
    # print(x.shape[-1])
    # calculate the embed_dim after concatenation
    # concat_embed_dim = embed_dim*sum_layers
    # add position embedding to the concatenated layers
    # TODO: add again
    x = PositionEmbedding(maxlen=max_case_length, embed_dim=x.shape[-1], name="position-embeddings")(x)
    # feed into transformer block
    x = TransformerBlock(x.shape[-1], num_heads, ff_dim)(x)
    
    
    # Stacking multiple transformer blocks
    # for _ in range(num_layers):
    #     x = TransformerBlock(embed_dim*len(categorical_feature_layers), num_heads, ff_dim)(x, mask=mask)
    
    # Global average pooling
    if mask_padding:
        x = masked_global_avg_pool(x, mask)
    else:
        x = layers.GlobalAveragePooling1D()(x)
    
    # Fully connected layers
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    
    # Output layers for categorical features
    outputs = []
    for target_col, target in target_columns.items():
        if target == Target.NEXT_FEATURE: dict_str = "y_next_word_dict"
        elif target == Target.LAST_FEATURE: dict_str = "y_last_word_dict"
        else: raise ValueError("Target type is not known.")
        output_dim = len(word_dicts[target_col][dict_str])
        outputs.append( layers.Dense(output_dim, activation="softmax", name=f"output_{target_col}")(x) )
    
    
    
    # Model definition
    transformer = Model(inputs=inputs, outputs=outputs, name="next_categorical_transformer")
    
    return transformer
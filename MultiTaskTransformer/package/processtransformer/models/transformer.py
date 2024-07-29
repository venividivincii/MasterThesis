import tensorflow as tf
from tensorflow.keras import layers, Model
from ..constants import Feature_Type
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

    def call(self, inputs, training):
        """
        Forward pass for the transformer block.
        
        Args:
            inputs (tf.Tensor): Input tensor.
            training (bool): Boolean indicating whether the model is in training mode.
        
        Returns:
            tf.Tensor: Output tensor after applying self-attention and feed-forward network.
        """
        attn_output = self.att(inputs, inputs)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Add & Norm
        ffn_output = self.ffn(out1)  # Feed-Forward Network
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Add & Norm

class TokenAndPositionEmbedding(layers.Layer):
    """
    Token and Position Embedding Layer.
    
    Args:
        maxlen (int): Maximum length of the sequences.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimensionality of the embedding space.
    """
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
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

# TODO: vocab_size to list of vocab_sizesnum_classes_list
def get_model(train_columns: List[str], target_columns: List[str], word_dict: Dict[str, Dict[str, int]], max_case_length: int,
              vocab_size_dict: Dict[str, int], feature_type_dict: Dict[Feature_Type, List[str]],
                               embed_dim=36, num_heads=4, ff_dim=64, num_layers=1):
    """
    Constructs the next categorical prediction model using a transformer architecture.
    
    Args:
        max_case_length (int): Maximum length of the sequences (cases).
        vocab_size (int): Size of the vocabulary.
        output_dim (int): Number of output classes for the next categorical prediction.
        num_categorical_features (int): Number of additional categorical features.
        num_numerical_features (int): Number of additional numerical features.
        num_classes_list (list): List containing the number of unique classes for each categorical feature.
        embed_dim (int): Dimensionality of the embeddings. Defaults to 36.
        num_heads (int): Number of attention heads. Defaults to 4.
        ff_dim (int): Dimensionality of the feed-forward layer. Defaults to 64.
        num_layers (int): Number of transformer blocks. Defaults to 1.
    
    Returns:
        tf.keras.Model: Compiled transformer model for next categorical prediction.
    """
    print("Creating model for task next_categorical...")


    # Categorical Input layers
    categorical_inputs, categorical_feature_layers = [], []
    for cat_feature in [ s for s in train_columns if s in feature_type_dict[Feature_Type.CATEGORICAL] ]:
        # Input Layer for categorical feature
        categorical_input = layers.Input(shape=(max_case_length,), name=cat_feature)
        categorical_inputs.append(categorical_input)
        # Input embedding for categorical feature
        categorical_emb = TokenAndPositionEmbedding(max_case_length, len(word_dict[cat_feature]["x_word_dict"]), embed_dim)(categorical_input)
        # Transformer Block for categorical feature
        categorical_feature_layers.append( TransformerBlock(embed_dim, num_heads, ff_dim)(categorical_emb) )
    
    # Token and position embedding for categorical features
    # x_cat_list = []
    # for idx, input in enumerate(categorical_inputs):
    #     x_cat_feature = TokenAndPositionEmbedding(max_case_length, list(vocab_size_dict.values())[idx], embed_dim)(input)
    #     x_cat_list.append( TransformerBlock(embed_dim, num_heads, ff_dim)(x_cat_feature) )
        
    # concat categorical feature layers
    x = layers.Concatenate()(categorical_feature_layers)
    
    # Stacking multiple transformer blocks
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim*len(categorical_feature_layers), num_heads, ff_dim)(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Fully connected layers
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    
    # Output layers for categorical features
    outputs = []
    for target_col in target_columns:
        output_dim = len(word_dict[target_col]["y_word_dict"])
        outputs.append( layers.Dense(output_dim, activation="softmax", name=target_col)(x) )
    
    # Model definition
    transformer = Model(inputs=categorical_inputs, outputs=outputs, name="next_categorical_transformer")
    
    return transformer


# def get_next_time_model(max_case_length, vocab_size, output_dim = 1, 
#     embed_dim = 36, num_heads = 4, ff_dim = 64):

#     inputs = layers.Input(shape=(max_case_length,))
#     # Three time-based features
#     time_inputs = layers.Input(shape=(3,)) 
#     x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
#     x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
#     x = layers.GlobalAveragePooling1D()(x)
#     x_t = layers.Dense(32, activation="relu")(time_inputs)
#     x = layers.Concatenate()([x, x_t])
#     x = layers.Dropout(0.1)(x)
#     x = layers.Dense(128, activation="relu")(x)
#     x = layers.Dropout(0.1)(x)
#     outputs = layers.Dense(output_dim, activation="linear")(x)
#     transformer = tf.keras.Model(inputs=[inputs, time_inputs], outputs=outputs,
#         name = "next_time_transformer")
#     return transformer

# def get_remaining_time_model(max_case_length, vocab_size, output_dim = 1, 
#     embed_dim = 36, num_heads = 4, ff_dim = 64):

#     inputs = layers.Input(shape=(max_case_length,))
#     # Three time-based features
#     time_inputs = layers.Input(shape=(3,)) 
#     x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
#     x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
#     x = layers.GlobalAveragePooling1D()(x)
#     x_t = layers.Dense(32, activation="relu")(time_inputs)
#     x = layers.Concatenate()([x, x_t])
#     x = layers.Dropout(0.1)(x)
#     x = layers.Dense(128, activation="relu")(x)
#     x = layers.Dropout(0.1)(x)
#     outputs = layers.Dense(output_dim, activation="linear")(x)
#     transformer = tf.keras.Model(inputs=[inputs, time_inputs], outputs=outputs,
#         name = "remaining_time_transformer")
#     return transformer
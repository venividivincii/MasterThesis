import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

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

# TODO: vocab_size to list of vocab_sizes
def get_next_categorical_model(max_case_length, vocab_size_dict, output_dim,
                               categorical_features, numerical_features,
                               num_classes_list, embed_dim=36, num_heads=4, ff_dim=64, num_layers=1):
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
    
    # Ensure num_classes_list is a list and has the correct length
    if not isinstance(num_classes_list, list) or len(num_classes_list) != len(categorical_features):
        raise ValueError("num_classes_list must be a list with length equal to num_categorical_features")

    # Input layers
    inputs = layers.Input(shape=(max_case_length,), name="inputs")
    # additional_inputs = layers.Input(shape=(len(categorical_features) + len(numerical_features),), name="additional_inputs") if (num_categorical_features + num_numerical_features) > 0 else None
    
    # Token and position embedding for the concept:name sequence input
    x = TokenAndPositionEmbedding(max_case_length, vocab_size_dict['concept:name'], embed_dim)(inputs)
    
    if additional_inputs is not None:
        # Process additional features: separate categorical and numerical processing
        additional_embs = []
        
        # Token and position embedding for categorical features
        if len(categorical_features) > 0:
            for i in range(len(categorical_features)):
                categorical_feature = additional_inputs[:, i]
                categorical_feature = tf.cast(categorical_feature, tf.int32)  # Ensure categorical features are int32
                num_classes = num_classes_list[i]
                categorical_emb = TokenAndPositionEmbedding(1, num_classes + 1, embed_dim)(tf.expand_dims(categorical_feature, -1))
                additional_embs.append(categorical_emb)
        
        # Token and position embedding for numerical features
        if len(numerical_features) > 0:
            numerical_features = additional_inputs[:, len(categorical_features):len(categorical_features) + len(numerical_features)]
            numerical_emb = TokenAndPositionEmbedding(1, numerical_features.shape[-1], embed_dim)(tf.expand_dims(numerical_features, -1))
            additional_embs.append(numerical_emb)

        # Concatenate all additional embeddings
        combined_features = layers.Concatenate()(additional_embs) if len(additional_embs) > 1 else additional_embs[0]
        
        # Expand combined features to match the sequence length
        combined_features_expanded = layers.RepeatVector(max_case_length)(combined_features)
        
        # Combine with token embeddings
        x = layers.Concatenate()([x, combined_features_expanded])
        
        # Project combined embeddings back to the desired dimension
        x = layers.Dense(embed_dim, activation="relu")(x)
    
    # Stacking multiple transformer blocks
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Fully connected layers
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    
    # Output layer
    outputs = layers.Dense(output_dim, activation="softmax", name="outputs")(x)
    
    # Model definition
    if additional_inputs is not None:
        transformer = Model(inputs=[inputs, additional_inputs], outputs=outputs, name="next_categorical_transformer")
    else:
        transformer = Model(inputs=[inputs], outputs=outputs, name="next_categorical_transformer")
    
    return transformer


def get_next_time_model(max_case_length, vocab_size, output_dim = 1, 
    embed_dim = 36, num_heads = 4, ff_dim = 64):

    inputs = layers.Input(shape=(max_case_length,))
    # Three time-based features
    time_inputs = layers.Input(shape=(3,)) 
    x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x_t = layers.Dense(32, activation="relu")(time_inputs)
    x = layers.Concatenate()([x, x_t])
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    transformer = tf.keras.Model(inputs=[inputs, time_inputs], outputs=outputs,
        name = "next_time_transformer")
    return transformer

def get_remaining_time_model(max_case_length, vocab_size, output_dim = 1, 
    embed_dim = 36, num_heads = 4, ff_dim = 64):

    inputs = layers.Input(shape=(max_case_length,))
    # Three time-based features
    time_inputs = layers.Input(shape=(3,)) 
    x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x_t = layers.Dense(32, activation="relu")(time_inputs)
    x = layers.Concatenate()([x, x_t])
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    transformer = tf.keras.Model(inputs=[inputs, time_inputs], outputs=outputs,
        name = "remaining_time_transformer")
    return transformer
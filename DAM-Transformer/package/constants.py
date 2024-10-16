import enum
 
@enum.unique
class Feature_Type(enum.Enum):
  """Look up for feature types."""
  
  CATEGORICAL = "categorical"
  NUMERICAL = "numerical"
  TIMESTAMP = "timestamp"
  
  def get_member(value):
    for member in Feature_Type:
        if member.value == value:
            return member
    raise ValueError(f"'{value}' is not a valid value for {Feature_Type.__name__}")
  
  
@enum.unique
class Target(enum.Enum):
  """Look up for Target variable."""
  
  NEXT_FEATURE = "next_feature"
  LAST_FEATURE = "last_feature"
  
  def get_member(value):
    for member in Target:
        if member.value == value:
            return member
    raise ValueError(f"'{value}' is not a valid value for {Target.__name__}")
  
  
@enum.unique
class Temporal_Feature(enum.Enum):
  """Look up for temporal features."""
  
  DAY_OF_WEEK = "day_of_week"
  HOUR_OF_DAY = "hour_of_day"
  
  def get_member(value):
    for member in Temporal_Feature:
        if member.value == value:
            return member
    raise ValueError(f"'{value}' is not a valid value for {Temporal_Feature.__name__}")
  
  
@enum.unique
class Model_Architecture(enum.Enum):
  """Look up for model architectures."""
  
  COMMON_POSEMBS_TRANSF = "common embeddings and transformers for all features"
  SEPERATE_POSEMBS = "seperate positional embeddings and common transformer for all features"
  SEPERATE_TRANSF = "seperate positional embeddings and transformers for each feature"
  TIME_TARGET = "Architecture for prediction temporal targets"
  
  def get_member(value):
    for member in Model_Architecture:
        if member.value == value:
            return member
    raise ValueError(f"'{value}' is not a valid value for {Model_Architecture.__name__}")
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
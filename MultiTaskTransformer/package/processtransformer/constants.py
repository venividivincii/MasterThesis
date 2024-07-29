import enum

@enum.unique
class Task(enum.Enum):
  """Look up for tasks."""
  
  NEXT_CATEGORICAL = "next_categorical"
  NEXT_TIME = "next_time"
  REMAINING_TIME = "remaining_time"
  
@enum.unique
class Feature_Type(enum.Enum):
  """Look up for feature types."""
  
  CATEGORICAL = "categorical"
  NUMERICAL = "numerical"
  
  def get_member(value):
    for member in Feature_Type:
        if member.value == value:
            return member
    raise ValueError(f"'{value}' is not a valid value for {Feature_Type.__name__}")
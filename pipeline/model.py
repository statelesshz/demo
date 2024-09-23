from typing import List, Optional, Union

# 当前model仅支持transformers和diffusers？
class Model:
  tasks: Union[List[str], str] = None
  framework: str = None   # ms,pt
  backend: str = None  # transformers mindformers diffuers onediff
  requirement_dependency: Optional[List[str]] = [] # accelerate, numpy...
  
  @classmethod
  def from_pretrained():
    ...
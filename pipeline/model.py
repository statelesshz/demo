from typing import List, Optional, Union

# # 得到一个类加载器

# # 当前model仅支持transformers和diffusers？
# class Model:
#   tasks: Union[List[str], str] = None
#   framework: str = None   # ms,pt
#   backend: str = None  # transformers mindformers diffuers onediff
#   requirement_dependency: Optional[List[str]] = [] # accelerate, numpy...
  
#   @classmethod
#   def from_pretrained():
#     ...


class Model:
  @classmethod
  def from_pretrained(
    cls,
    model_name_or_path: str,
    
  )


"""
目标：
根据传入的model_name_or_path(str or os.PathLike)得到一个ModelWrapper类，
    该类包含实例化的model，framework，tasks，backend，requirement_dependency 
"""

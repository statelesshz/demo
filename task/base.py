from typing import List, Union

from .domain_type import DomainType
from .input_output_type import InputAndOutputType

class Task:
    def __init__(self,
                 domain: DomainType=None,
                 input_type: Union[List[InputAndOutputType], InputAndOutputType] = None,
                 output_type: Union[List[InputAndOutputType], InputAndOutputType] = None,
                 **kwargs):
        self._domain = domain
        if type(input_type) is InputAndOutputType:
            input_type = [input_type]
        self._input_type = input_type
        if type(output_type) is InputAndOutputType:
            output_type = [output_type]
        self._output = output_type

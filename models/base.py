class BaseModel:
    def __init__(self, task_name_list, model_id_or_path, requires):
        if type(task_name_list) == list:
            task_name_list = [task_name_list]
        self.task_name_list = task_name_list
        self.model_id_or_path = model_id_or_path
        self.requires = requires

    def init_and_load(self):
        ...

    def check_requirements(self):
        ...



class LlmModel(BaseModel):
    def __init__(self, task_name_list, model_id_or_path, requires, lora_target, template):
        super().__init__(task_name_list, model_id_or_path, requires)
        self.lora_target = lora_target
        self.template = template

    
class SDModel(BaseModel):
    def __init__(self, task_name_list, model_id_or_path, requires):
        super().__init__(task_name_list, model_id_or_path, requires)

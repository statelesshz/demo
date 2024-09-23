## [draft] Model的元数据

- 支持的tasks
    
    可用于pipeline中根据传入的model(str)推测task，由于一个模型可能支持多种task，而对于pipeline中只根据model推测task的场景，**不妨令tasks[0]是default_task**

-  framework

    同一模型结构可能有不同框架的实现，比如PT、MS等

- backend

    同一模型结构在同一框架下也有可能有多种实现，比如SD模型可能在diffuers中有实现，onediff基于diffusers优化了推理的性能，底层结构可能还是复用diffuers，这种情况我们用backend属性做区分

    有了backend属性，我们可以很容是使用各自的model_loader加载模型

- requirement_dependency

    模型实现上会依赖一些其他的三方库，因此增加该属性便于加载模型前做必要的校验

- 其他

    随时补充其他必要的元数据


遗留：
- https://github.com/modelscope/modelscope/blob/d5c9c82340f39c0c63f32503725582a0959600aa/modelscope/utils/config.py#L56
- 

**是否需要configuration.json**


- https://www.modelscope.cn/models/iic/nlp_bert_document-segmentation_chinese-base/file/view/master?fileName=configuration.json&status=1
```json
{
    "framework": "pytorch",
    "task": "document-segmentation",
    "model": {
        "type": "bert-for-document-segmentation",
        "model_config": {
            "type": "bert",
            "level": "doc"
        }
    },
    "pipeline": {
        "type": "document-segmentation"
    }
}
```

- https://www.modelscope.cn/models/ai-modelscope/stable-diffusion-xl-base-1.0/file/view/master?fileName=configuration.json&status=1

```
{
    "framework": "pytorch",
    "task": "text-to-image-synthesis",
    "pipeline": {
        "type": "diffusers-stable-diffusion"
    },
    "model": {
        "type": "stable-diffusion-xl",
        "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
        "lora_tune": true,
        "dreambooth_tune": false
    },
    "preprocessor": {
        "type": "diffusion-image-generation-preprocessor",
        "resolution": 512,
        "mean": [0.5],
        "std": [0.5]
    },
    ...
}
```

- https://www.modelscope.cn/models/llm-research/meta-llama-3-8b-instruct/file/view/master?fileName=configuration.json&status=1
```json
{"framework":"Pytorch","task":"text-generation"}
```

## [draft] 如何根据model_name_or_path(st)加载模型？

如何转换到不同backend(transformers/diffuers/mindformers)对应的模型加载器上？

### 在远端模型仓增加configuration.json文件
包含如下字段
```json
"model_type": "qwen2",
"framework": "pytorch",
"backend": "transformers",
"tasks": ["text-classification", "token-classification"],  # 第一个task作为默认task
```

有了上述信息可以把模型加载拆成两阶段：
1. 下载文件
    同时解析configuration，根据framework、backend、以及task任务去推断使用的类加载器
2. 通过具体的类加载器实例化类对象


### 通过文件格式&配置文件特征推测使用的backend&framework
1. transformers&mindformers类型推断
2. diffusers类型推断
3. mindone类型推断？

这样做的好处是无需维护任何额外的内容

---

Model需要的方法：

类方法
- `from_pretrained`

    ```python
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        cfg_dict: Config = None,
        **kwargs
    )
    ```


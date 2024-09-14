# 设计原则

## 清晰把控
明确支持的特性，从代码可以推导出完整的支持列表

## 可扩展性
容易引入新任务和依赖，不影响原有模块功能，降低新发版本回归测试工作量，插件式加入，可插拔


## 兼容性
任务减隔离，模型间隔离，框架间隔离，有效隔离mindspore和pytorch的差异


## 可运维
根据错误日志快读定位问题单责任人，输出错误码

注册task（domain （cv nlp 。。。）绑定input&output类型
注册pipeline wrapper 绑定task
注册pipeline（绑定task 指定默认模型 framework）
注册trainer wrapper（绑定task）
注册trainer（binding task framework）
注册模型（binding task framework）
注册template（binding model）
注册数据集（binding task）
注册metric（binding task）

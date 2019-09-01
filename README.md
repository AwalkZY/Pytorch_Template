# 模板说明
## 命名规则
1. 函数、函数参数、非类对象变量使用下划线法命名，类对象使用小驼峰法命名，类、常量使用大驼峰法命名。（常量可使用全大写命名）
    例如：
    ```python
    class SomeClass(object):
        def __init__(self):
            self.some_member = 0
                
        def some_function(self, some_input=2):
            some_variable = 1
            self.some_member = some_variable + some_input 
            return self.some_member
             
    someObject = SomeClass()
    ```
2. 配置、模型、训练器等变量分别以“config”，“model”，“trainer”结尾。
3. 维度、数量、长度等分别以“dim”，“num”，“length”结尾。
4. 可能被迭代访问或通过索引访问的变量，以复数形式表示，迭代获得的变量可使用对应单数形式。



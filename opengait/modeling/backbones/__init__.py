from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module

# iterate through the modules in the current package
package_dir = Path(__file__).resolve().parent  #获取当前文件的路径，并通过 resolve() 方法解析出绝对路径，然后通过 parent 属性获取其父目录。
for (_, module_name, _) in iter_modules([str(package_dir)]):  #使用 iter_modules 函数迭代当前包目录下的模块

    # import the module and iterate through its attributes
    module = import_module(f"{__name__}.{module_name}")
    for attribute_name in dir(module):  #遍历导入的模块中的所有属性名。
        attribute = getattr(module, attribute_name)  #使用 getattr 函数获取模块中属性的值。

        if isclass(attribute):
            # Add the class to this package's variables
            globals()[attribute_name] = attribute  #将类添加到当前包的全局变量中，使得其他模块可以直接访问这些类。
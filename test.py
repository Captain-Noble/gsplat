from pycolmap import SceneManager
def list_attributes_of_class(cls):
    # Get all attributes of the class, including methods and built-in attributes
    attributes = dir(cls)
    
    # Filter out methods and built-in attributes (anything starting with '__')
    filtered_attributes = [attr for attr in attributes if not callable(getattr(cls, attr)) and not attr.startswith('__')]
    
    return filtered_attributes


# 示例，列出SceneManager类的属性
scene_manager = SceneManager("data/Dustr3D_align/sparse/0/")
attributes = list_attributes_of_class(scene_manager)
print("SceneManager类的属性有：", attributes)
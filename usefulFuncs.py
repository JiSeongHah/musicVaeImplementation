import os


# 폴더 이름 생성 위한 코드
def mk_name(*args,**name_value_dict):
    total_name = ''
    additional_arg = ''

    for arg in args:
        additional_arg += (str(arg)+'_')

    for name_value in name_value_dict.items():
        name = name_value[0]
        value = name_value[1]
        total_name += (str(name)+str(value)+'_')

    total_name += additional_arg[:-1]

    return total_name


# 디렉토리 생성 코드
def createDir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'making {str(directory)} complete successfully!')
    except OSError:
        print("Error: Failed to create the directory.")
import json
def load_data_from_raw(path):
    '''
    path: path to daset dir(dev/test/train)
    ---
    return: [python iter object] with single data
    '''
    print(path)
    with open(path,'r',encoding='utf-8') as f:
        for line in iter(f):
            line_json = json.loads(line)
            yield line_json
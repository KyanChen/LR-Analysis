import json

import pandas as pd

data = pd.read_csv("../../data/泛娱乐_十二月视频搜索满意度评估_LR实验_2679条_is_ppt.csv")
data_json = {}

quality_class = {'低': 0, '中': 1, '高': 2}

value_num = {
    'note_first_img_quality_str': 1,
    'note_first_img_beauty_str': 1,
    'vid_quality': 1,
    'vid_beauty': 1,
    'is_bgm': 1,
    'is_ppt': 1,
    'note_value_scores': 6,
    'note_quality_str': 1,
    'mc_label': 1,
    'eat_label': 1,
    'quality': 1
}
total_valid_items = 0
gt_nums = {2: 0, 3: 0}
for idx, item in data.iterrows():
    key = item['note_id']
    try:
        values = {
            'note_first_img_quality_str': quality_class[item['note_first_img_quality_str']],
            'note_first_img_beauty_str': quality_class[item['note_first_img_beauty_str']],
            # 'vid_quality': 0 if pd.isna(item['vid_quality']) else int(list(eval(item['vid_quality']).keys())[0]),
            'vid_quality': int(list(eval(item['vid_quality']).keys())[0]),
            'vid_beauty': int(list(eval(item['vid_beauty']).keys())[0]),
            'is_bgm': item['is_bgm'],
            'is_ppt': item['is_ppt'],
            'note_value_scores': [float(x) for x in list(eval(item['note_value_scores']).values())],
            'note_quality_str': quality_class[item['note_quality_str']],
            'mc_label': item['mc_label'],
            'eat_label': item['eat_label'],
            'quality': item['quality']
        }

        for val_num_k, val_num_v in value_num.items():
            if isinstance(values[val_num_k], list):
                assert len(values[val_num_k]) == val_num_v
            else:
                pass
        if item['quality'] in [2, 3]:
            data_json[key] = values
            total_valid_items += 1
            gt_nums[item['quality']] += 1
    except Exception as e:
        print(key, e)

print(f'{total_valid_items}/{len(data)}')
print(gt_nums)

# check
for key, values in data_json.items():
    for v in list(values.values()):
        if isinstance(v, list):
            for x in v:
                if isinstance(x, int) or isinstance(x, float):
                    pass
                else:
                    print(type(x))
        else:
            if isinstance(v, int) or isinstance(v, float):
                pass
            else:
                print(type(v))

min_max_mapping_dict = {
    'note_first_img_quality_str': [0, 3],
    'note_first_img_beauty_str': [0, 3],
    'vid_quality': [0, 3],
    'vid_beauty': [0, 3],
    'is_bgm': [0, 1],
    'is_ppt': [0, 1],
    'note_value_scores': [0, 1],
    'note_quality_str': [0, 2],
    'mc_label': [0, 3],
    'eat_label': [0, 1],
    # 'quality': item['quality']
}

def min_max_mapping(v, min=0, max=1, key=''):
    assert min <= v <= max, f'{key}: {v}'

    return (v - min) / (max-min)


for key, values in data_json.items():
    for map_k, map_v in min_max_mapping_dict.items():
        if isinstance(values[map_k], list):
            values[map_k] = [min_max_mapping(x, min=map_v[0], max=map_v[1], key=map_k) for x in values[map_k]]
        else:
            values[map_k] = min_max_mapping(values[map_k], min=map_v[0], max=map_v[1], key=map_k)
    data_json[key] = values

json.dump(data_json, open('../../data/data.json', 'w'), indent=4)
